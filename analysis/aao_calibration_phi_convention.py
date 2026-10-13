#!/usr/bin/env python3
"""Test AAO radiative azimuth conventions with mode-3 calibration trials.

Mode-3 calibration runs emit no LUND events, but retain a CSV row for every
positive corrected integrand. Each row contains both the internal hadronic
azimuth, ``phi_cm_deg``, and the azimuth reconstructed from final generated
four-vectors, ``phi_observed_deg``. This utility histograms the corrected
integrand in both angles, constructs their weighted migration matrix, and
tests an explicit 180-degree remapping of the observed angle.

The radiative histograms are converted to cross sections with the calibration
phase volume and total proposal count. A previously produced AAO Born
reference NPZ supplies the Born histogram and normalization, so all three
angle hypotheses are displayed as generator-level radiative corrections.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CalibrationSpec:
    label: str
    report: Path


@dataclass
class CalibrationResult:
    label: str
    report: Path
    csv_files: list[Path]
    proposals: int
    calibration_rows: int
    integrated_cross_section: float
    integrated_cross_section_sem: float
    observed_sumw: Array
    observed_sumw2: Array
    internal_sumw: Array
    internal_sumw2: Array
    shifted_sumw: Array
    shifted_sumw2: Array
    migration_sumw: Array
    delta_edges: Array
    delta_sumw: Array
    delta_mean_deg: float
    delta_resultant: float
    delta_near_zero_fraction: float
    delta_near_180_fraction: float
    phase_volume: float


def wrap_degrees(values: Array | float) -> Array:
    """Wrap angles to the half-open interval [-180, 180)."""

    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def parse_calibration(value: str) -> CalibrationSpec:
    """Parse a ``LABEL=REPORT.json`` command-line value."""

    if "=" not in value:
        raise ValueError("calibration must be LABEL=REPORT.json")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    report = Path(raw_path).expanduser()
    if not label or not raw_path.strip():
        raise ValueError("calibration label and report path must be nonempty")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"unsupported calibration label: {label!r}")
    return CalibrationSpec(label=label, report=report)


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def calibration_csv_paths(
    report: dict[str, object], report_path: Path
) -> list[Path]:
    manifest_value = report.get("source_manifest")
    if not isinstance(manifest_value, str):
        raise ValueError(f"{report_path}: calibration lacks source_manifest")
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = report_path.parent / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{manifest_path}: calibration manifest has no runs")
    paths: list[Path] = []
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("run_json"), str):
            raise ValueError(f"{manifest_path}: malformed run record")
        path = Path(str(run["run_json"])).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve().with_suffix(".calibration.csv")
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def _calibration_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        schema = stream.readline().rstrip("\n")
        if schema != "# schema=aao-rad-mode3-calibration-v1":
            raise ValueError(f"{path}: unexpected calibration schema")
        reader = csv.DictReader(stream)
        required = {"integrand_corrected", "phi_cm_deg", "phi_observed_deg"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: calibration CSV lacks required columns")
        for line_number, row in enumerate(reader, start=3):
            try:
                weight = float(row["integrand_corrected"])
                internal = float(row["phi_cm_deg"]) % 360.0
                observed = float(row["phi_observed_deg"]) % 360.0
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: malformed row") from error
            if not (
                math.isfinite(weight)
                and weight > 0.0
                and math.isfinite(internal)
                and math.isfinite(observed)
            ):
                raise ValueError(f"{path}:{line_number}: invalid row domain")
            yield weight, internal, observed


def _histogram_uncertainty(
    sumw: Array, sumw2: Array, *, volume: float, proposals: int
) -> tuple[Array, Array]:
    if proposals <= 1 or volume <= 0.0:
        raise ValueError("phase volume and proposal count must be positive")
    cross_section = volume * sumw / proposals
    centered = np.maximum(sumw2 - np.square(sumw) / proposals, 0.0)
    uncertainty = volume * np.sqrt(centered / (proposals * (proposals - 1)))
    return cross_section, uncertainty


def correction_and_error(
    radiative: Array,
    radiative_error: Array,
    born: Array,
    born_error: Array,
) -> tuple[Array, Array]:
    correction = np.divide(
        radiative,
        born,
        out=np.full_like(radiative, np.nan, dtype=float),
        where=born > 0.0,
    )
    relative_rad = np.divide(
        radiative_error,
        radiative,
        out=np.zeros_like(radiative),
        where=radiative > 0.0,
    )
    relative_born = np.divide(
        born_error,
        born,
        out=np.zeros_like(born),
        where=born > 0.0,
    )
    return correction, correction * np.sqrt(relative_rad**2 + relative_born**2)


def load_calibration(spec: CalibrationSpec, phi_edges: Array) -> CalibrationResult:
    report_path = spec.report.resolve()
    report = _load_json(report_path)
    if report.get("schema") != "aao-rad-mode3-envelope-v1":
        raise ValueError(f"{report_path}: not a finalized mode-3 calibration")
    proposals = int(report["proposals"])
    integrated = float(report["integrated_cross_section_microbarn"])
    integrated_sem = float(report["integrated_cross_section_sem_microbarn"])
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{report_path}: calibration report has no runs")
    phase_volumes = {
        float(run["phase_volume"]) for run in runs if isinstance(run, dict)
    }
    if len(phase_volumes) != 1:
        raise ValueError(f"{report_path}: inconsistent phase volumes")
    phase_volume = phase_volumes.pop()
    files = calibration_csv_paths(report, report_path)
    nphi = phi_edges.size - 1
    observed_sumw = np.zeros(nphi)
    observed_sumw2 = np.zeros(nphi)
    internal_sumw = np.zeros(nphi)
    internal_sumw2 = np.zeros(nphi)
    shifted_sumw = np.zeros(nphi)
    shifted_sumw2 = np.zeros(nphi)
    migration_sumw = np.zeros((nphi, nphi))
    delta_edges = np.linspace(-180.0, 180.0, 73)
    delta_sumw = np.zeros(delta_edges.size - 1)
    delta_vector = 0.0j
    total_weight = 0.0
    near_zero = 0.0
    near_180 = 0.0
    rows = 0
    for path in files:
        chunk = list(_calibration_rows(path))
        if not chunk:
            continue
        values = np.asarray(chunk, dtype=float)
        weight = values[:, 0]
        internal = values[:, 1]
        observed = values[:, 2]
        shifted = (observed + 180.0) % 360.0
        delta = wrap_degrees(observed - internal)
        observed_sumw += np.histogram(observed, bins=phi_edges, weights=weight)[0]
        observed_sumw2 += np.histogram(
            observed, bins=phi_edges, weights=weight**2
        )[0]
        internal_sumw += np.histogram(internal, bins=phi_edges, weights=weight)[0]
        internal_sumw2 += np.histogram(
            internal, bins=phi_edges, weights=weight**2
        )[0]
        shifted_sumw += np.histogram(shifted, bins=phi_edges, weights=weight)[0]
        shifted_sumw2 += np.histogram(
            shifted, bins=phi_edges, weights=weight**2
        )[0]
        migration_sumw += np.histogram2d(
            internal, observed, bins=(phi_edges, phi_edges), weights=weight
        )[0]
        delta_sumw += np.histogram(delta, bins=delta_edges, weights=weight)[0]
        delta_vector += np.sum(weight * np.exp(1j * np.radians(delta)))
        total_weight += float(np.sum(weight))
        near_zero += float(np.sum(weight[np.abs(delta) < 15.0]))
        near_180 += float(np.sum(weight[np.abs(delta) > 165.0]))
        rows += weight.size
        print(
            f"[{spec.label}] read {path.name}: {weight.size} positive rows",
            flush=True,
        )
    if rows == 0 or total_weight <= 0.0:
        raise ValueError(f"{report_path}: no calibration rows")
    reconstructed = phase_volume * total_weight / proposals
    if not math.isclose(reconstructed, integrated, rel_tol=5.0e-12, abs_tol=1e-15):
        raise ValueError(
            f"{report_path}: histogrammed cross section {reconstructed} "
            f"does not reproduce finalized value {integrated}"
        )
    return CalibrationResult(
        label=spec.label,
        report=report_path,
        csv_files=files,
        proposals=proposals,
        calibration_rows=rows,
        integrated_cross_section=integrated,
        integrated_cross_section_sem=integrated_sem,
        observed_sumw=observed_sumw,
        observed_sumw2=observed_sumw2,
        internal_sumw=internal_sumw,
        internal_sumw2=internal_sumw2,
        shifted_sumw=shifted_sumw,
        shifted_sumw2=shifted_sumw2,
        migration_sumw=migration_sumw,
        delta_edges=delta_edges,
        delta_sumw=delta_sumw,
        delta_mean_deg=float(np.degrees(np.angle(delta_vector))),
        delta_resultant=float(abs(delta_vector) / total_weight),
        delta_near_zero_fraction=near_zero / total_weight,
        delta_near_180_fraction=near_180 / total_weight,
        phase_volume=phase_volume,
    )


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        action="append",
        required=True,
        metavar="LABEL=REPORT.json",
        help="finalized candidate calibration; repeat for multiple settings",
    )
    parser.add_argument(
        "--born-reference",
        required=True,
        help=(
            "soft/hard or v-cut NPZ containing the 10M Born cell histogram "
            "and normalization"
        ),
    )
    parser.add_argument("--output", required=True, help="output prefix")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    specs = [parse_calibration(value) for value in args.calibration]
    if len({spec.label for spec in specs}) != len(specs):
        raise ValueError("calibration labels must be unique")
    with np.load(args.born_reference, allow_pickle=False) as born_source:
        phi_edges = np.asarray(born_source["phi_edges_deg"], dtype=float)
        counts_key = "born_counts" if "born_counts" in born_source.files else "H_born"
        sumw2_key = (
            "born_cluster_sumw2"
            if "born_cluster_sumw2" in born_source.files
            else "H_born_cluster_sumw2"
        )
        born_counts = np.asarray(born_source[counts_key], dtype=float)
        born_sumw2 = np.asarray(born_source[sumw2_key], dtype=float)
        born_events = int(born_source["born_topology_events"])
        born_integrated = float(born_source["born_integrated_cross_section"])
    nphi = phi_edges.size - 1
    if (
        phi_edges.ndim != 1
        or nphi <= 0
        or born_counts.shape != (nphi,)
        or born_sumw2.shape != (nphi,)
        or born_events <= 0
        or born_integrated <= 0.0
    ):
        raise ValueError("Born reference has inconsistent contents")
    born_cross_section = born_integrated * born_counts / born_events
    born_error = born_integrated * np.sqrt(born_sumw2) / born_events
    results = [load_calibration(spec, phi_edges) for spec in specs]
    mappings = (
        ("observed", "observed_sumw", "observed_sumw2"),
        ("internal_cm", "internal_sumw", "internal_sumw2"),
        ("observed_plus_180", "shifted_sumw", "shifted_sumw2"),
    )
    corrections = np.empty((len(results), len(mappings), nphi))
    uncertainties = np.empty_like(corrections)
    radiative_cross_sections = np.empty_like(corrections)
    radiative_errors = np.empty_like(corrections)
    for ir, result in enumerate(results):
        for im, (_name, sumw_name, sumw2_name) in enumerate(mappings):
            rad, drad = _histogram_uncertainty(
                getattr(result, sumw_name),
                getattr(result, sumw2_name),
                volume=result.phase_volume,
                proposals=result.proposals,
            )
            correction, uncertainty = correction_and_error(
                rad, drad, born_cross_section, born_error
            )
            radiative_cross_sections[ir, im] = rad
            radiative_errors[ir, im] = drad
            corrections[ir, im] = correction
            uncertainties[ir, im] = uncertainty

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([result.label for result in results])
    mapping_names = np.asarray([mapping[0] for mapping in mappings])
    np.savez_compressed(
        output.with_suffix(".npz"),
        labels=labels,
        mappings=mapping_names,
        phi_edges_deg=phi_edges,
        born_cross_section=born_cross_section,
        born_cross_section_error=born_error,
        radiative_cross_section=radiative_cross_sections,
        radiative_cross_section_error=radiative_errors,
        C_rad=corrections,
        delta_C_rad=uncertainties,
        migration_sumw=np.asarray([result.migration_sumw for result in results]),
        delta_edges_deg=results[0].delta_edges,
        delta_sumw=np.asarray([result.delta_sumw for result in results]),
    )
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "label",
                "mapping",
                "phi_low_deg",
                "phi_high_deg",
                "radiative_cross_section_microbarn",
                "radiative_cross_section_error_microbarn",
                "born_cross_section_microbarn",
                "born_cross_section_error_microbarn",
                "C_rad",
                "delta_C_rad",
            ]
        )
        for ir, result in enumerate(results):
            for im, (mapping, _sumw, _sumw2) in enumerate(mappings):
                for iphi in range(nphi):
                    writer.writerow(
                        [
                            result.label,
                            mapping,
                            phi_edges[iphi],
                            phi_edges[iphi + 1],
                            radiative_cross_sections[ir, im, iphi],
                            radiative_errors[ir, im, iphi],
                            born_cross_section[iphi],
                            born_error[iphi],
                            corrections[ir, im, iphi],
                            uncertainties[ir, im, iphi],
                        ]
                    )

    figure, axes = plt.subplots(
        len(results),
        3,
        figsize=(16.0, max(4.4, 4.2 * len(results))),
        squeeze=False,
    )
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    for ir, result in enumerate(results):
        axis = axes[ir, 0]
        for im, label in enumerate(("observed", "internal cm", "observed + 180°")):
            axis.errorbar(
                phi_centers,
                corrections[ir, im],
                yerr=uncertainties[ir, im],
                marker="o",
                linewidth=1.1,
                markersize=3.5,
                label=label,
            )
        axis.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
        axis.set_xlabel(r"comparison angle $\phi$ [deg]")
        axis.set_ylabel(r"$C_{rad}$")
        axis.set_title(f"{result.label}: convention test")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

        migration = result.migration_sumw / np.sum(result.migration_sumw)
        mesh = axes[ir, 1].pcolormesh(
            phi_edges,
            phi_edges,
            100.0 * migration.T,
            shading="auto",
            cmap="viridis",
        )
        axes[ir, 1].plot([0.0, 360.0], [0.0, 360.0], "w--", linewidth=1.0)
        axes[ir, 1].set_xlabel(r"internal $\phi_{cm}$ [deg]")
        axes[ir, 1].set_ylabel(r"observed $\phi$ [deg]")
        axes[ir, 1].set_title("weighted migration")
        figure.colorbar(mesh, ax=axes[ir, 1], label="cross-section fraction [%]")

        delta_fraction = result.delta_sumw / np.sum(result.delta_sumw)
        axes[ir, 2].stairs(delta_fraction, result.delta_edges, color="tab:purple")
        axes[ir, 2].axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
        axes[ir, 2].set_xlabel(r"wrap($\phi_{obs}-\phi_{cm}$) [deg]")
        axes[ir, 2].set_ylabel("weighted fraction")
        axes[ir, 2].set_title(
            rf"$\langle\Delta\phi\rangle_c={result.delta_mean_deg:.1f}^\circ$, "
            rf"$R={result.delta_resultant:.3f}$"
        )
        axes[ir, 2].grid(alpha=0.2)
    figure.suptitle(
        "AAO mode-3 calibration: internal versus observed azimuth convention",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(figure)

    central = np.flatnonzero(
        (phi_edges[:-1] >= 150.0) & (phi_edges[1:] <= 210.0)
    )
    summaries: list[dict[str, object]] = []
    for ir, result in enumerate(results):
        mapping_summary: dict[str, object] = {}
        for im, (mapping, _sumw, _sumw2) in enumerate(mappings):
            mapping_summary[mapping] = {
                "C_rad": corrections[ir, im].tolist(),
                "delta_C_rad": uncertainties[ir, im].tolist(),
                "central_150_210_mean": float(
                    np.nanmean(corrections[ir, im, central])
                ),
            }
        summaries.append(
            {
                "label": result.label,
                "calibration_report": str(result.report),
                "calibration_csv_files": len(result.csv_files),
                "calibration_rows": result.calibration_rows,
                "proposals": result.proposals,
                "integrated_cross_section_microbarn": result.integrated_cross_section,
                "integrated_cross_section_sem_microbarn": (
                    result.integrated_cross_section_sem
                ),
                "histogrammed_cross_section_microbarn": float(
                    np.sum(radiative_cross_sections[ir, 0])
                ),
                "wrapped_observed_minus_internal": {
                    "circular_mean_deg": result.delta_mean_deg,
                    "resultant_length": result.delta_resultant,
                    "weighted_fraction_within_15deg_of_zero": (
                        result.delta_near_zero_fraction
                    ),
                    "weighted_fraction_within_15deg_of_180": (
                        result.delta_near_180_fraction
                    ),
                },
                "mappings": mapping_summary,
            }
        )
    summary = {
        "definition": {
            "observed": "AAO phi_observed_deg from final generated kinematics",
            "internal_cm": "AAO phi_cm_deg at the radiative vertex",
            "observed_plus_180": "(phi_observed_deg + 180 deg) modulo 360 deg",
        },
        "interpretation": (
            "observed is the nominal correction. internal_cm and "
            "observed_plus_180 are convention diagnostics; neither should replace "
            "the nominal correction unless the downstream phi convention is shown "
            "to match that mapping. All mappings preserve the same integrated "
            "radiative cross section."
        ),
        "born_reference": str(Path(args.born_reference).resolve()),
        "phi_edges_deg": phi_edges.tolist(),
        "calibrations": summaries,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    for suffix in (".npz", ".csv", ".pdf", ".png", ".json"):
        print(f"wrote {output.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
