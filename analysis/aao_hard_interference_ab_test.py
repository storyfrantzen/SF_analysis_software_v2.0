#!/usr/bin/env python3
"""Run controlled AAO radiative-generator A/B diagnostics.

``build`` copies only the AAO compiler inputs to a separate directory and
applies one named diagnostic patch. ``run`` executes a variant with the seeds
and proposal count from an existing mode-3 calibration and compares it with
the retained baseline CSVs. Optional input overrides test the arbitrary
soft/hard separator without preparing a new campaign.

The result is diagnostic only. It must not be used for production without
generator-author approval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


Array = np.ndarray
OLD_HARD_DENOMINATOR = """c     &  /(sigu+epeps*sigl))
     &  /sigf)"""
NEW_HARD_DENOMINATOR = """     &  /(sigu+epeps*sigl))
c     &  /sigf)"""
OLD_LT_NORMALIZATION = "sqrt(epeps*(1.+epeps)/2)"
NEW_LT_NORMALIZATION = "sqrt(2.*epeps*(1.+epeps))"
INCOMING_EXTERNAL_HOOK = (
    "      incoming_loss_accept_test=xtest\n"
    "      \n"
    "      es\t= ebeam*(1.-eloss)"
)
INCOMING_EXTERNAL_OFF = (
    "      incoming_loss_accept_test=xtest\n"
    "c     DIAGNOSTIC ONLY: retain the nominal loss draws but remove their energy shift\n"
    "      eloss=0.\n"
    "      \n"
    "      es\t= ebeam*(1.-eloss)"
)
OUTGOING_EXTERNAL_HOOK = """      outgoing_loss_accept_test=xtest

      ep_out=ep_pre*(1.-eloss)"""
OUTGOING_EXTERNAL_OFF = """      outgoing_loss_accept_test=xtest
c     DIAGNOSTIC ONLY: retain the nominal loss draws but remove their energy shift
      eloss=0.

      ep_out=ep_pre*(1.-eloss)"""
NOMINAL_HARD_MODULATION = """ 123   sigf=sigf * (1.+(epeps*sigt*cos(epphi*pi/90.)
     &  +sqrt(epeps*(1.+epeps)/2)*sigi*cos(epphi*pi/180.)
     &  +ehel*sqrt(epeps*(1.-epeps)/2)*sigip*sin(epphi*pi/180.))
c     &  /(sigu+epeps*sigl))
     &  /sigf)"""
NO_HARD_MODULATION = """ 123   sigf=sigf
c     DIAGNOSTIC ONLY: hard-photon phi interference modulation disabled"""
SHIFTED_FIXED_HARD_MODULATION = """ 123   sigf=sigf * (1.+(epeps*sigt*cos(epphi*pi/90.)
     &  -sqrt(epeps*(1.+epeps)/2)*sigi*cos(epphi*pi/180.)
     &  -ehel*sqrt(epeps*(1.-epeps)/2)*sigip*sin(epphi*pi/180.))
     &  /(sigu+epeps*sigl))
c     &  /sigf)"""
CALIBRATION_SCHEMA = "# schema=aao-rad-mode3-calibration-v1"
REPORT_SCHEMA = "aao-rad-mode3-envelope-v1"


@dataclass(frozen=True)
class WeightedRows:
    trial: Array
    weight: Array
    phi_observed: Array
    phi_internal: Array
    q2_observed: Array
    xb_observed: Array
    minus_t_observed: Array


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch_hard_denominator(text: str) -> str:
    """Return source with exactly the hard-branch denominator activated."""

    count = text.count(OLD_HARD_DENOMINATOR)
    if count != 1:
        raise ValueError(
            "expected exactly one commented hard-branch denominator block; "
            f"found {count}"
        )
    patched = text.replace(OLD_HARD_DENOMINATOR, NEW_HARD_DENOMINATOR)
    if patched.count(NEW_HARD_DENOMINATOR) != 1:
        raise RuntimeError("hard-branch patch did not produce one replacement")
    return patched


def patch_lt_normalization(text: str) -> str:
    """Match both radiative LT coefficients to ``DVMPW``'s ``sigma0``.

    ``DVMPW`` returns ``sigi`` and constructs its direct cross section with
    ``sqrt(2*epsilon*(1+epsilon))*sigi*cos(phi)``.  The legacy AAO soft and
    hard reconstructions use a coefficient smaller by exactly two.  Require
    the two expected occurrences so this diagnostic cannot silently patch an
    unrelated source revision.
    """

    count = text.count(OLD_LT_NORMALIZATION)
    if count != 2:
        raise ValueError(
            "expected exactly two radiative LT-normalization factors; "
            f"found {count}"
        )
    patched = text.replace(OLD_LT_NORMALIZATION, NEW_LT_NORMALIZATION)
    if patched.count(NEW_LT_NORMALIZATION) != 2:
        raise RuntimeError(
            "LT-normalization patch did not produce two replacements"
        )
    return patched


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} block; found {count}")
    return text.replace(old, new)


def patch_variant(text: str, kind: str) -> tuple[str, list[str]]:
    """Apply one auditable diagnostic variant to AAO fixed-form source."""

    changes: list[str] = []
    patched = text
    if kind == "control":
        changes.append("none: unmodified-source rebuild control")
    if kind in {
        "denominator",
        "denominator-external-off",
        "denominator-phi-plus-180",
        "denominator-lt-normalization",
    }:
        patched = patch_hard_denominator(patched)
        changes.append("hard interference denominator: sigf -> sigu+epsilon*sigl")
    if kind in {"lt-normalization", "denominator-lt-normalization"}:
        patched = patch_lt_normalization(patched)
        changes.append(
            "soft/hard LT coefficient: sqrt(epsilon*(1+epsilon)/2) -> "
            "sqrt(2*epsilon*(1+epsilon))"
        )
    if kind in {"external-off", "denominator-external-off"}:
        patched = _replace_once(
            patched,
            INCOMING_EXTERNAL_HOOK,
            INCOMING_EXTERNAL_OFF,
            "incoming external-loss",
        )
        patched = _replace_once(
            patched,
            OUTGOING_EXTERNAL_HOOK,
            OUTGOING_EXTERNAL_OFF,
            "outgoing external-loss",
        )
        changes.append(
            "incoming/outgoing external energy shifts disabled after nominal draws"
        )
    if kind == "interference-off":
        patched = _replace_once(
            patched,
            NOMINAL_HARD_MODULATION,
            NO_HARD_MODULATION,
            "hard interference modulation",
        )
        changes.append("hard interference modulation disabled")
    if kind == "denominator-phi-plus-180":
        fixed = NOMINAL_HARD_MODULATION.replace(
            OLD_HARD_DENOMINATOR, NEW_HARD_DENOMINATOR
        )
        patched = _replace_once(
            patched,
            fixed,
            SHIFTED_FIXED_HARD_MODULATION,
            "fixed-denominator hard phi modulation",
        )
        changes.append("hard cos(phi) and sin(phi) terms shifted by 180 degrees")
    if not changes:
        raise ValueError(f"unsupported diagnostic variant: {kind}")
    return patched, changes


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolved_path(value: object, anchor: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = anchor.parent / path
    return path.resolve()


def calibration_csv_by_replica(
    report: dict[str, object], report_path: Path
) -> dict[int, Path]:
    """Resolve calibration CSVs through the report's source manifest.

    Finalized envelope reports embed completed run payloads, not the original
    manifest records that carry ``run_json``.  The source manifest is therefore
    the authoritative map from replica number to retained calibration CSV.
    """

    manifest_value = report.get("source_manifest")
    if not isinstance(manifest_value, str):
        raise ValueError(f"{report_path}: calibration lacks source_manifest")
    manifest_path = _resolved_path(manifest_value, report_path)
    manifest = _json(manifest_path)
    records = manifest.get("runs")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{manifest_path}: calibration manifest has no runs")
    result: dict[int, Path] = {}
    for record in records:
        if not isinstance(record, dict) or "replica_index" not in record:
            raise ValueError(f"{manifest_path}: malformed run record")
        run_json = record.get("run_json")
        if not isinstance(run_json, str):
            raise ValueError(f"{manifest_path}: run lacks run_json")
        replica = int(record["replica_index"])
        if replica in result:
            raise ValueError(f"{manifest_path}: duplicate replica {replica}")
        csv_path = _resolved_path(run_json, manifest_path).with_suffix(
            ".calibration.csv"
        )
        if not csv_path.is_file() or csv_path.stat().st_size <= 0:
            raise FileNotFoundError(csv_path)
        result[replica] = csv_path
    return result


def override_input_records(
    text: str,
    *,
    electron_helicity_flag: int | None = None,
    target_length_cm: float | None = None,
    minimum_photon_energy: float | None = None,
) -> tuple[str, dict[str, float | int]]:
    """Override selected legacy AAO input records while preserving comments."""

    replacements: dict[int, tuple[str, float | int]] = {}
    if electron_helicity_flag is not None:
        if electron_helicity_flag not in (0, 1):
            raise ValueError("electron helicity flag must be zero or one")
        replacements[1] = (str(electron_helicity_flag), electron_helicity_flag)
    if target_length_cm is not None:
        if target_length_cm < 0.0:
            raise ValueError("target length must be nonnegative")
        replacements[6] = (f"{target_length_cm:.17g}", target_length_cm)
    if minimum_photon_energy is not None:
        if minimum_photon_energy <= 0.0:
            raise ValueError("minimum photon energy must be positive")
        replacements[14] = (
            f"{minimum_photon_energy:.17g}",
            minimum_photon_energy,
        )
    if not replacements:
        return text, {}
    lines = text.splitlines()
    record = 0
    applied: dict[str, float | int] = {}
    names = {
        1: "electron_helicity_flag",
        6: "target_length_cm",
        14: "minimum_photon_energy_GeV",
    }
    for index, raw in enumerate(lines):
        clean = raw.split("!", 1)[0].strip()
        if not clean:
            continue
        if record in replacements:
            value, stored = replacements[record]
            comment = ""
            if "!" in raw:
                comment = "   !" + raw.split("!", 1)[1]
            lines[index] = value + comment
            applied[names[record]] = stored
        record += 1
    missing = {key for key in replacements if names[key] not in applied}
    if missing:
        raise ValueError(f"AAO input lacks records {sorted(missing)}")
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + suffix, applied


def build_variant(args: argparse.Namespace) -> int:
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (source / "aao_rad.F").is_file() or not (source / "Makefile").is_file():
        raise FileNotFoundError(f"not an AAO radiative source directory: {source}")
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"variant output is not empty: {output}")
        output.rmdir()
    build_inputs = sorted(
        {
            source / "Makefile",
            *source.glob("*.F"),
            *source.glob("*.c"),
            *source.glob("*.inc"),
        }
    )
    if len(build_inputs) < 3:
        raise RuntimeError(f"no complete AAO build inputs found in {source}")
    total_bytes = sum(path.stat().st_size for path in build_inputs)
    print(
        f"copying {len(build_inputs)} diagnostic build inputs "
        f"({total_bytes} bytes): {source} -> {output}",
        flush=True,
    )
    output.mkdir(parents=True)
    for path in build_inputs:
        shutil.copy2(path, output / path.name)
    fortran = output / "aao_rad.F"
    original_hash = sha256(fortran)
    patched_text, changes = patch_variant(
        fortran.read_text(encoding="utf-8"), args.variant_kind
    )
    fortran.write_text(
        patched_text,
        encoding="utf-8",
    )
    patched_hash = sha256(fortran)
    build_log = output / "build.make.log"
    print(f"building diagnostic executable; live log: {build_log}", flush=True)
    with build_log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            ["make", "aao_rad"],
            cwd=output,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if process.stdout is None:  # pragma: no cover - subprocess contract
            raise RuntimeError("failed to capture make output")
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    executable = output / "build" / "aao_rad"
    if returncode != 0 or not executable.is_file():
        raise RuntimeError(
            f"variant build failed with code {returncode}; see {build_log}"
        )
    provenance = {
        "schema": "aao-radiative-diagnostic-variant-v2",
        "diagnostic_only": True,
        "variant_kind": args.variant_kind,
        "source_directory": str(source),
        "copied_source": str(output),
        "source_aao_rad_F_sha256": original_hash,
        "patched_aao_rad_F_sha256": patched_hash,
        "executable": str(executable),
        "executable_sha256": sha256(executable),
        "changes": changes,
    }
    _write_json(output / "variant_provenance.json", provenance)
    print(json.dumps(provenance, indent=2))
    print(f"built {executable}")
    return 0


def read_rows(path: Path) -> WeightedRows:
    with path.open("r", encoding="utf-8", newline="") as stream:
        if stream.readline().rstrip("\n") != CALIBRATION_SCHEMA:
            raise ValueError(f"{path}: unexpected calibration schema")
        reader = csv.DictReader(stream)
        required = {
            "trial",
            "integrand_corrected",
            "phi_cm_deg",
            "q2_observed",
            "xb_observed",
            "minus_t_observed",
            "phi_observed_deg",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: calibration CSV lacks required columns")
        columns: dict[str, list[float]] = {name: [] for name in required}
        for line_number, row in enumerate(reader, start=3):
            try:
                values = {name: float(row[name]) for name in required}
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: malformed row") from error
            if (
                values["integrand_corrected"] <= 0.0
                or not all(math.isfinite(value) for value in values.values())
            ):
                raise ValueError(f"{path}:{line_number}: invalid row")
            for name, value in values.items():
                columns[name].append(value)
    trial = np.asarray(columns["trial"], dtype=np.int64)
    if trial.size != np.unique(trial).size:
        raise ValueError(f"{path}: duplicate trial identifiers")
    return WeightedRows(
        trial=trial,
        weight=np.asarray(columns["integrand_corrected"]),
        phi_observed=np.mod(columns["phi_observed_deg"], 360.0),
        phi_internal=np.mod(columns["phi_cm_deg"], 360.0),
        q2_observed=np.asarray(columns["q2_observed"]),
        xb_observed=np.asarray(columns["xb_observed"]),
        minus_t_observed=np.asarray(columns["minus_t_observed"]),
    )


def _parse_norm(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            values[key.strip()] = value.strip().split()[0]
    return values


def _histogram(rows: WeightedRows, edges: Array, angle: str) -> tuple[Array, Array]:
    phi = rows.phi_observed if angle == "observed" else rows.phi_internal
    return (
        np.histogram(phi, bins=edges, weights=rows.weight)[0],
        np.histogram(phi, bins=edges, weights=np.square(rows.weight))[0],
    )


def _cross_section(
    sumw: Array, sumw2: Array, phase_volume: float, proposals: int
) -> tuple[Array, Array]:
    value = phase_volume * sumw / proposals
    centered = np.maximum(sumw2 - np.square(sumw) / proposals, 0.0)
    error = phase_volume * np.sqrt(centered / (proposals * (proposals - 1)))
    return value, error


def _correction(
    radiative: Array, dradiative: Array, born: Array, dborn: Array
) -> tuple[Array, Array]:
    value = np.divide(
        radiative, born, out=np.full_like(radiative, np.nan), where=born > 0.0
    )
    relative_rad = np.divide(
        dradiative,
        radiative,
        out=np.zeros_like(dradiative),
        where=radiative > 0.0,
    )
    relative_born = np.divide(
        dborn, born, out=np.zeros_like(dborn), where=born > 0.0
    )
    return value, value * np.sqrt(relative_rad**2 + relative_born**2)


def _born_reference(path: Path) -> tuple[Array, Array, Array]:
    with np.load(path, allow_pickle=False) as source:
        edges = np.asarray(source["phi_edges_deg"], dtype=float)
        counts_key = "born_counts" if "born_counts" in source.files else "H_born"
        sumw2_key = (
            "born_cluster_sumw2"
            if "born_cluster_sumw2" in source.files
            else "H_born_cluster_sumw2"
        )
        counts = np.asarray(source[counts_key], dtype=float)
        sumw2 = np.asarray(source[sumw2_key], dtype=float)
        events = int(source["born_topology_events"])
        integrated = float(source["born_integrated_cross_section"])
    if counts.shape != (edges.size - 1,) or sumw2.shape != counts.shape:
        raise ValueError(f"{path}: incompatible Born phi histogram")
    return edges, integrated * counts / events, integrated * np.sqrt(sumw2) / events


def _paired_diagnostics(
    baseline: WeightedRows, variant: WeightedRows
) -> dict[str, object]:
    baseline_index = {int(trial): index for index, trial in enumerate(baseline.trial)}
    variant_index = {int(trial): index for index, trial in enumerate(variant.trial)}
    common = sorted(set(baseline_index) & set(variant_index))
    coordinate_matches = 0
    ratios: list[float] = []
    for trial in common:
        ib = baseline_index[trial]
        iv = variant_index[trial]
        delta_phi = (
            (baseline.phi_observed[ib] - variant.phi_observed[iv] + 180.0)
            % 360.0
            - 180.0
        )
        coordinates_match = (
            abs(baseline.q2_observed[ib] - variant.q2_observed[iv]) < 1.0e-8
            and abs(baseline.xb_observed[ib] - variant.xb_observed[iv]) < 1.0e-8
            and abs(
                baseline.minus_t_observed[ib] - variant.minus_t_observed[iv]
            )
            < 1.0e-8
            and abs(delta_phi) < 1.0e-7
        )
        if coordinates_match:
            coordinate_matches += 1
            ratios.append(variant.weight[iv] / baseline.weight[ib])
    ratio = np.asarray(ratios)
    return {
        "baseline_positive_rows": int(baseline.trial.size),
        "variant_positive_rows": int(variant.trial.size),
        "common_positive_trials": len(common),
        "baseline_only_trials": len(set(baseline_index) - set(variant_index)),
        "variant_only_trials": len(set(variant_index) - set(baseline_index)),
        "coordinate_matched_trials": coordinate_matches,
        "coordinate_match_fraction_of_common": (
            coordinate_matches / len(common) if common else 0.0
        ),
        "matched_weight_ratio_quantiles": (
            np.quantile(ratio, [0.0, 0.1, 0.5, 0.9, 1.0]).tolist()
            if ratio.size
            else []
        ),
        "matched_fraction_weight_unchanged_1e-6": (
            float(np.mean(np.abs(ratio - 1.0) <= 1.0e-6))
            if ratio.size
            else 0.0
        ),
    }


def run_test(args: argparse.Namespace) -> int:
    report_path = args.baseline_report.expanduser().resolve()
    report = _json(report_path)
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"{report_path}: not a finalized mode-3 calibration")
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{report_path}: no calibration runs")
    if args.replicas <= 0 or args.replicas > len(runs):
        raise ValueError(f"--replicas must lie in [1,{len(runs)}]")
    selected = runs[: args.replicas]
    baseline_csvs = calibration_csv_by_replica(report, report_path)
    executable = args.variant_executable.expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"test output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    edges, born, dborn = _born_reference(args.born_reference.expanduser().resolve())
    baseline_sumw = np.zeros(edges.size - 1)
    baseline_sumw2 = np.zeros_like(baseline_sumw)
    variant_sumw = np.zeros_like(baseline_sumw)
    variant_sumw2 = np.zeros_like(baseline_sumw)
    proposals = 0
    phase_volumes: set[float] = set()
    paired: list[dict[str, object]] = []
    run_artifacts: list[dict[str, object]] = []
    input_overrides: dict[str, float | int] | None = None
    for raw_run in selected:
        if not isinstance(raw_run, dict):
            raise ValueError("malformed calibration run record")
        input_path = Path(str(raw_run["input"])).expanduser().resolve()
        replica = int(raw_run["replica_index"])
        if replica not in baseline_csvs:
            raise ValueError(f"replica {replica}: absent from source manifest")
        baseline_csv = baseline_csvs[replica]
        baseline_rows = read_rows(baseline_csv)
        input_text, applied_overrides = override_input_records(
            input_path.read_text(encoding="utf-8"),
            electron_helicity_flag=args.electron_helicity_flag,
            target_length_cm=args.target_length_cm,
            minimum_photon_energy=args.minimum_photon_energy,
        )
        if input_overrides is None:
            input_overrides = applied_overrides
        elif input_overrides != applied_overrides:
            raise RuntimeError("input overrides differ across replicas")
        run_dir = output / "runs" / f"replica_{replica:04d}"
        run_dir.mkdir(parents=True)
        effective_input = run_dir / "variant.inp"
        effective_input.write_text(input_text, encoding="utf-8")
        with tempfile.TemporaryDirectory(
            prefix=f"variant_replica_{replica:04d}_", dir=run_dir
        ) as temporary:
            work = Path(temporary)
            completed = subprocess.run(
                [str(executable)],
                input=input_text,
                text=True,
                cwd=work,
                capture_output=True,
                check=False,
            )
            (run_dir / "variant.stdout").write_text(
                completed.stdout, encoding="utf-8"
            )
            (run_dir / "variant.stderr").write_text(
                completed.stderr, encoding="utf-8"
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"variant replica {replica} failed with code "
                    f"{completed.returncode}; see {run_dir}"
                )
            variant_csv = work / "aao_rad.mode3.csv"
            variant_norm = work / "aao_rad.norm"
            if not variant_csv.is_file() or not variant_norm.is_file():
                raise RuntimeError(f"variant replica {replica} lacks diagnostics")
            saved_csv = run_dir / "variant.calibration.csv"
            saved_norm = run_dir / "variant.norm"
            shutil.copyfile(variant_csv, saved_csv)
            shutil.copyfile(variant_norm, saved_norm)
        variant_rows = read_rows(saved_csv)
        norm = _parse_norm(saved_norm)
        variant_proposals = int(float(norm["ntries"]))
        baseline_proposals = int(raw_run["ntries"])
        if variant_proposals != baseline_proposals:
            raise ValueError(
                f"replica {replica}: proposal counts differ: "
                f"{baseline_proposals} versus {variant_proposals}"
            )
        baseline_phase_volume = float(raw_run["phase_volume"])
        variant_phase_volume = float(
            norm.get("mode3_phase_volume", baseline_phase_volume)
        )
        if not math.isclose(
            baseline_phase_volume,
            variant_phase_volume,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"replica {replica}: phase volumes differ: "
                f"{baseline_phase_volume} versus {variant_phase_volume}"
            )
        phase_volumes.add(baseline_phase_volume)
        proposals += baseline_proposals
        base_w, base_w2 = _histogram(baseline_rows, edges, args.angle)
        var_w, var_w2 = _histogram(variant_rows, edges, args.angle)
        baseline_sumw += base_w
        baseline_sumw2 += base_w2
        variant_sumw += var_w
        variant_sumw2 += var_w2
        paired_result = _paired_diagnostics(baseline_rows, variant_rows)
        paired_result["replica_index"] = replica
        paired.append(paired_result)
        run_artifacts.append(
            {
                "replica_index": replica,
                "input": str(input_path),
                "effective_input": str(effective_input),
                "baseline_csv": str(baseline_csv),
                "variant_csv": str(saved_csv),
                "variant_norm": str(saved_norm),
            }
        )
        print(
            f"replica {replica}: proposals={baseline_proposals}, "
            f"baseline rows={baseline_rows.trial.size}, "
            f"variant rows={variant_rows.trial.size}",
            flush=True,
        )
    if len(phase_volumes) != 1:
        raise ValueError("selected baseline replicas have different phase volumes")
    phase_volume = phase_volumes.pop()
    base_xs, dbase_xs = _cross_section(
        baseline_sumw, baseline_sumw2, phase_volume, proposals
    )
    var_xs, dvar_xs = _cross_section(
        variant_sumw, variant_sumw2, phase_volume, proposals
    )
    base_c, dbase_c = _correction(base_xs, dbase_xs, born, dborn)
    var_c, dvar_c = _correction(var_xs, dvar_xs, born, dborn)
    ratio = np.divide(
        var_xs, base_xs, out=np.full_like(var_xs, np.nan), where=base_xs > 0.0
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    central = (edges[:-1] >= 150.0) & (edges[1:] <= 210.0)
    prefix = output / args.artifact_stem
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        phi_edges_deg=edges,
        born_cross_section=born,
        baseline_radiative_cross_section=base_xs,
        baseline_radiative_cross_section_error=dbase_xs,
        variant_radiative_cross_section=var_xs,
        variant_radiative_cross_section_error=dvar_xs,
        baseline_C_rad=base_c,
        baseline_delta_C_rad=dbase_c,
        variant_C_rad=var_c,
        variant_delta_C_rad=dvar_c,
        variant_over_baseline=ratio,
    )
    with prefix.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "phi_low_deg",
                "phi_high_deg",
                "baseline_C_rad",
                "baseline_delta_C_rad",
                "variant_C_rad",
                "variant_delta_C_rad",
                "variant_over_baseline_radiative_xs",
            ]
        )
        for index in range(centers.size):
            writer.writerow(
                [
                    edges[index],
                    edges[index + 1],
                    base_c[index],
                    dbase_c[index],
                    var_c[index],
                    dvar_c[index],
                    ratio[index],
                ]
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    axes[0].errorbar(
        centers, base_c, yerr=dbase_c, marker="o", label=args.baseline_label
    )
    axes[0].errorbar(
        centers,
        var_c,
        yerr=dvar_c,
        marker="o",
        label=args.variant_label,
    )
    axes[0].axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel(r"$C_{rad}$")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].plot(centers, ratio, marker="o", color="tab:purple")
    axes[1].axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel(rf"{args.angle} $\phi$ [deg]")
    axes[1].set_ylabel("variant / baseline radiative cross section")
    axes[1].grid(alpha=0.2)
    figure.suptitle(
        f"{args.title}\n"
        f"{args.replicas} paired calibration replica(s), {proposals} proposals"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(prefix.with_suffix(".pdf"))
    figure.savefig(prefix.with_suffix(".png"), dpi=180)
    plt.close(figure)
    summary = {
        "schema": "aao-radiative-generator-ab-test-v2",
        "diagnostic_only": True,
        "title": args.title,
        "artifact_stem": args.artifact_stem,
        "baseline_label": args.baseline_label,
        "variant_label": args.variant_label,
        "baseline_report": str(report_path),
        "variant_executable": str(executable),
        "variant_executable_sha256": sha256(executable),
        "born_reference": str(args.born_reference.expanduser().resolve()),
        "angle": args.angle,
        "replicas": args.replicas,
        "proposals": proposals,
        "phase_volume": phase_volume,
        "input_overrides": input_overrides or {},
        "baseline_integrated_cross_section_microbarn": float(np.sum(base_xs)),
        "variant_integrated_cross_section_microbarn": float(np.sum(var_xs)),
        "integrated_variant_over_baseline": float(
            np.sum(var_xs) / np.sum(base_xs)
        ),
        "central_150_210": {
            "baseline_C_rad_mean": float(np.mean(base_c[central])),
            "variant_C_rad_mean": float(np.mean(var_c[central])),
            "difference": float(np.mean(var_c[central] - base_c[central])),
            "variant_over_baseline": float(
                np.mean(var_c[central]) / np.mean(base_c[central])
            ),
        },
        "paired_trial_diagnostics": paired,
        "runs": run_artifacts,
    }
    _write_json(prefix.with_suffix(".json"), summary)
    print(json.dumps(summary, indent=2))
    for suffix in (".npz", ".csv", ".pdf", ".png", ".json"):
        print(f"wrote {prefix.with_suffix(suffix)}")
    return 0


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="copy, patch, and build AAO variant")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--variant-kind",
        choices=(
            "control",
            "denominator",
            "external-off",
            "denominator-external-off",
            "interference-off",
            "denominator-phi-plus-180",
            "lt-normalization",
            "denominator-lt-normalization",
        ),
        default="denominator",
    )
    build.set_defaults(action=build_variant)
    run = commands.add_parser("run", help="run paired calibration A/B diagnostic")
    run.add_argument("--baseline-report", type=Path, required=True)
    run.add_argument("--variant-executable", type=Path, required=True)
    run.add_argument("--born-reference", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--replicas", type=int, default=1)
    run.add_argument("--angle", choices=("observed", "internal"), default="observed")
    run.add_argument("--electron-helicity-flag", type=int, choices=(0, 1))
    run.add_argument("--target-length-cm", type=float)
    run.add_argument("--minimum-photon-energy", type=float)
    run.add_argument("--artifact-stem", default="hard_interference_ab")
    run.add_argument("--baseline-label", default="baseline")
    run.add_argument("--variant-label", default="variant")
    run.add_argument(
        "--title", default="AAO radiative-generator diagnostic A/B"
    )
    run.set_defaults(action=run_test)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    return int(args.action(args))


if __name__ == "__main__":
    raise SystemExit(main())
