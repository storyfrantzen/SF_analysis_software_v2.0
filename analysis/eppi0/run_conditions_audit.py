"""Reusable RCDB, QADB, and converter run-metadata audit helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


DEFAULT_RCDB_CONDITIONS = (
    "beam_current",
    "target",
    "torus_scale",
    "run_config",
)


@dataclass
class RcdbRunConditions:
    run: int
    beam_current_nA: float | None = None
    target: str | None = None
    torus_scale: float | None = None
    run_config: str | None = None
    query_error: str | None = None


@dataclass
class RunChargeRecord:
    run: int
    accumulated_charge_nC: float = 0.0
    total_events: int = 0
    passed_qadb_events: int = 0
    failed_qadb_events: int = 0
    source_files: int = 0


def parse_run_specification(specification: str) -> set[int]:
    """Parse comma/whitespace-separated run numbers and inclusive ranges."""

    runs: set[int] = set()
    for token in re.split(r"[\s,]+", specification.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*[-:]\s*(\d+))?", token)
        if match is None:
            raise ValueError(f"invalid run token: {token!r}")
        start = int(match.group(1))
        stop = int(match.group(2) or start)
        if start <= 0 or stop <= 0:
            raise ValueError("run numbers must be positive")
        if stop < start:
            raise ValueError(f"descending run range is not allowed: {token}")
        runs.update(range(start, stop + 1))
    return runs


def load_run_list(path: Path) -> set[int]:
    pieces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        pieces.append(line.split("#", 1)[0])
    return parse_run_specification(" ".join(pieces))


def load_base_manifest(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"base manifest is not an object: {path}")
    if "runs" in payload and not isinstance(payload["runs"], dict):
        raise ValueError(f"base manifest runs member is not an object: {path}")
    if "run_classes" in payload and not isinstance(payload["run_classes"], dict):
        raise ValueError(f"base manifest run_classes member is not an object: {path}")
    return payload


def parse_class_assignments(specifications: Sequence[str]) -> dict[int, str]:
    assignments: dict[int, str] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("--assign-class must have the form CLASS=RUNS")
        run_class, raw_runs = specification.split("=", 1)
        run_class = run_class.strip()
        if not run_class:
            raise ValueError("--assign-class has an empty class name")
        for run in parse_run_specification(raw_runs):
            previous = assignments.get(run)
            if previous is not None and previous != run_class:
                raise ValueError(
                    f"run {run} assigned to both {previous!r} and {run_class!r}"
                )
            assignments[run] = run_class
    return assignments


def parse_nominal_currents(specifications: Sequence[str]) -> dict[str, float | None]:
    currents: dict[str, float | None] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("--nominal-current must have the form CLASS=VALUE")
        run_class, raw_value = specification.split("=", 1)
        run_class = run_class.strip()
        raw_value = raw_value.strip()
        if not run_class:
            raise ValueError("--nominal-current has an empty class name")
        if raw_value.lower() in {"none", "null", "unknown"}:
            value = None
        else:
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"invalid nominal current for class {run_class}: {raw_value}"
                )
        currents[run_class] = value
    return currents


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "none",
        "null",
        "nan",
    }:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _import_rcdb():
    try:
        return importlib.import_module("rcdb")
    except ModuleNotFoundError as original:
        rcdb_home = os.environ.get("RCDB_HOME")
        if not rcdb_home:
            raise RuntimeError(
                "could not import rcdb and RCDB_HOME is not set; source the JLab "
                "module environment first"
            ) from original
        python_path = str(Path(rcdb_home) / "python")
        if python_path not in sys.path:
            sys.path.insert(0, python_path)
        try:
            return importlib.import_module("rcdb")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"could not import rcdb from {python_path}; check the RCDB module"
            ) from exc


def query_rcdb(
    runs: Iterable[int],
    connection: str,
    *,
    provider=None,
) -> dict[int, RcdbRunConditions]:
    """Query the conditions used by the current-efficiency audit."""

    if provider is None:
        rcdb = _import_rcdb()
        provider = rcdb.RCDBProvider(connection)
    results: dict[int, RcdbRunConditions] = {}
    successful_calls = 0
    failed_calls = 0
    for run in sorted(set(int(value) for value in runs)):
        values: dict[str, object] = {}
        errors: list[str] = []
        for name in DEFAULT_RCDB_CONDITIONS:
            try:
                condition = provider.get_condition(run, name)
                values[name] = getattr(condition, "value", None)
                successful_calls += 1
            except Exception as exc:  # RCDB backends expose several exception types.
                failed_calls += 1
                errors.append(f"{name}: {exc}")
        results[run] = RcdbRunConditions(
            run=run,
            beam_current_nA=_optional_float(values.get("beam_current")),
            target=_optional_string(values.get("target")),
            torus_scale=_optional_float(values.get("torus_scale")),
            run_config=_optional_string(values.get("run_config")),
            query_error="; ".join(errors) or None,
        )
    if results and successful_calls == 0 and failed_calls:
        first = next(iter(results.values())).query_error
        raise RuntimeError(f"every RCDB condition query failed; first error: {first}")
    return results


def write_rcdb_tsv(path: Path, records: Mapping[int, RcdbRunConditions]) -> None:
    fieldnames = [
        "run",
        "beam_current_nA",
        "target",
        "torus_scale",
        "run_config",
        "query_error",
    ]
    _write_tsv(path, fieldnames, (asdict(records[run]) for run in sorted(records)))


def load_rcdb_tsv(path: Path) -> dict[int, RcdbRunConditions]:
    records: dict[int, RcdbRunConditions] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"run", "beam_current_nA", "target", "torus_scale", "run_config"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"RCDB TSV is missing columns: {sorted(missing)}")
        for row in reader:
            run = int(row["run"])
            if run in records:
                raise ValueError(f"duplicate run {run} in RCDB TSV")
            records[run] = RcdbRunConditions(
                run=run,
                beam_current_nA=_optional_float(row["beam_current_nA"]),
                target=_optional_string(row["target"]),
                torus_scale=_optional_float(row["torus_scale"]),
                run_config=_optional_string(row["run_config"]),
                query_error=_optional_string(row.get("query_error")),
            )
    return records


def aggregate_run_charge_rows(
    rows: Iterable[RunChargeRecord],
) -> dict[int, RunChargeRecord]:
    result: dict[int, RunChargeRecord] = {}
    for row in rows:
        current = result.setdefault(row.run, RunChargeRecord(run=row.run))
        current.accumulated_charge_nC += float(row.accumulated_charge_nC)
        current.total_events += int(row.total_events)
        current.passed_qadb_events += int(row.passed_qadb_events)
        current.failed_qadb_events += int(row.failed_qadb_events)
        current.source_files += int(row.source_files)
    return result


def read_run_charge_root(paths: Sequence[Path]) -> dict[int, RunChargeRecord]:
    if not paths:
        return {}
    try:
        import ROOT  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyROOT is required for --processing-root; source the repository JLab "
            "module setup or use --run-charge-input"
        ) from exc
    ROOT.gROOT.SetBatch(True)
    columns = (
        "runNum",
        "accumulatedCharge_nC",
        "totalEvents",
        "passedQADBEvents",
        "failedQADBEvents",
    )
    rows: list[RunChargeRecord] = []
    for raw_path in paths:
        path = raw_path.resolve()
        root_file = ROOT.TFile.Open(str(path), "READ")
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"could not open processing ROOT file: {path}")
        tree = root_file.Get("RunCharge")
        if not tree:
            root_file.Close()
            raise RuntimeError(f"processing ROOT file has no RunCharge tree: {path}")
        branches = {branch.GetName() for branch in tree.GetListOfBranches()}
        missing = set(columns).difference(branches)
        root_file.Close()
        if missing:
            raise RuntimeError(
                f"RunCharge tree in {path} is missing branches: {sorted(missing)}"
            )
        arrays = ROOT.RDataFrame("RunCharge", str(path)).AsNumpy(list(columns))
        seen: set[int] = set()
        for values in zip(*(arrays[name] for name in columns), strict=True):
            run = int(values[0])
            if run in seen:
                raise RuntimeError(f"duplicate RunCharge row for run {run} in {path}")
            seen.add(run)
            rows.append(
                RunChargeRecord(
                    run=run,
                    accumulated_charge_nC=float(values[1]),
                    total_events=int(values[2]),
                    passed_qadb_events=int(values[3]),
                    failed_qadb_events=int(values[4]),
                    source_files=1,
                )
            )
    return aggregate_run_charge_rows(rows)


def write_run_charge_tsv(path: Path, records: Mapping[int, RunChargeRecord]) -> None:
    fieldnames = [
        "run",
        "accumulated_charge_nC",
        "total_events",
        "passed_qadb_events",
        "failed_qadb_events",
        "source_files",
    ]
    _write_tsv(path, fieldnames, (asdict(records[run]) for run in sorted(records)))


def load_run_charge_tsv(path: Path) -> dict[int, RunChargeRecord]:
    rows: list[RunChargeRecord] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {
            "run",
            "accumulated_charge_nC",
            "total_events",
            "passed_qadb_events",
            "failed_qadb_events",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"RunCharge TSV is missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                RunChargeRecord(
                    run=int(row["run"]),
                    accumulated_charge_nC=float(row["accumulated_charge_nC"]),
                    total_events=int(row["total_events"]),
                    passed_qadb_events=int(row["passed_qadb_events"]),
                    failed_qadb_events=int(row["failed_qadb_events"]),
                    source_files=int(row.get("source_files") or 1),
                )
            )
    return aggregate_run_charge_rows(rows)


def parse_qadb_misc_code(text: str) -> dict[int, str]:
    """Parse ``qadb-info misc --code`` output without interpreting comments."""

    comments: dict[int, list[str]] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\s*,?\s*(?:(?:#|//)\s*)?(.*?)\s*$", line)
        if match is None:
            continue
        run = int(match.group(1))
        comment = match.group(2).strip()
        if comment:
            comments.setdefault(run, []).append(comment)
        else:
            comments.setdefault(run, [])
    return {
        run: " | ".join(dict.fromkeys(values))
        for run, values in sorted(comments.items())
    }


def parse_qadb_misc(text: str) -> tuple[dict[int, str], set[int] | None]:
    """Return Misc comments and, for JSON input, the set of QADB-covered runs.

    Legacy ``--code`` output lists only runs with Misc and therefore cannot prove
    whether an omitted run was clean or absent from QADB. Its coverage result is
    consequently ``None``.
    """

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return parse_qadb_misc_code(text), None
    if not isinstance(payload, dict):
        raise ValueError("qadb-info JSON output is not an object")
    comments: dict[int, str] = {}
    covered: set[int] = set()
    for raw_run, raw_info in payload.items():
        run = int(raw_run)
        if not isinstance(raw_info, Mapping):
            raise ValueError(f"QADB JSON entry for run {run} is not an object")
        covered.add(run)
        if not bool(raw_info.get("has_misc_bit", False)):
            continue
        raw_comments = raw_info.get("comments", [])
        if isinstance(raw_comments, Mapping):
            values = [str(value) for value in raw_comments]
        elif isinstance(raw_comments, Sequence) and not isinstance(
            raw_comments, (str, bytes)
        ):
            values = [str(value) for value in raw_comments]
        elif raw_comments:
            values = [str(raw_comments)]
        else:
            values = []
        comments[run] = " | ".join(dict.fromkeys(values))
    return comments, covered


def query_qadb_misc(
    runs: Iterable[int],
    datasets: str,
    *,
    executable: str = "qadb-info",
) -> tuple[str, dict[int, str], set[int]]:
    resolved = shutil.which(executable) if "/" not in executable else executable
    if not resolved:
        raise RuntimeError(
            f"could not find {executable!r}; load the QADB module or use "
            "--qadb-misc-input"
        )
    run_text = ",".join(str(run) for run in sorted(set(runs)))
    command = [
        resolved,
        "misc",
        "--datasets",
        datasets,
        "--runs",
        run_text,
        "--json",
        "--no-list-bins",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"qadb-info misc failed with status {completed.returncode}: {detail}"
        )
    comments, covered = parse_qadb_misc(completed.stdout)
    if covered is None:  # JSON was explicitly requested, so this is malformed output.
        raise RuntimeError("qadb-info did not return JSON as requested")
    return completed.stdout, comments, covered


def current_quality(current: float | None, inherited: object = None) -> str:
    previous = _optional_string(inherited)
    if current is None:
        return "missing"
    if current < 0.0:
        return "suspect"
    if previous and previous not in {"missing"}:
        return previous
    return "unflagged"


def build_outputs(
    runs: Iterable[int],
    rcdb_records: Mapping[int, RcdbRunConditions],
    charge_records: Mapping[int, RunChargeRecord],
    *,
    run_group: str,
    period: str | None,
    beam_energy_gev: float | None,
    base_manifest: Mapping[str, object] | None = None,
    class_assignments: Mapping[int, str] | None = None,
    nominal_currents: Mapping[str, float | None] | None = None,
    qadb_comments: Mapping[int, str] | None = None,
    qadb_covered_runs: Iterable[int] | None = None,
    processing_supplied: bool = False,
) -> tuple[dict, list[dict[str, object]], dict]:
    base = dict(base_manifest or {})
    base_runs = base.get("runs", {})
    if not isinstance(base_runs, Mapping):
        raise ValueError("base manifest runs member is not an object")
    assignments = dict(class_assignments or {})
    nominal = dict(nominal_currents or {})
    qadb = None if qadb_comments is None else dict(qadb_comments)
    qadb_covered = (
        None if qadb_covered_runs is None else set(int(run) for run in qadb_covered_runs)
    )

    dataset = (
        dict(base.get("dataset", {}))
        if isinstance(base.get("dataset"), Mapping)
        else {}
    )
    dataset["run_group"] = run_group
    if period is not None:
        dataset["period"] = period
    if beam_energy_gev is not None:
        dataset["beam_energy_GeV"] = float(beam_energy_gev)
    dataset["current_unit"] = "nA"

    run_classes = (
        dict(base.get("run_classes", {}))
        if isinstance(base.get("run_classes"), Mapping)
        else {}
    )
    for run_class in set(assignments.values()).union(nominal):
        definition = dict(run_classes.get(run_class, {}))
        definition.setdefault("description", "Review and describe this run class")
        definition["nominal_current_nA"] = nominal.get(
            run_class, definition.get("nominal_current_nA")
        )
        run_classes[run_class] = definition
    if "unclassified" not in run_classes:
        run_classes["unclassified"] = {
            "description": "Requires explicit run-class review before fitting",
            "nominal_current_nA": None,
        }

    manifest_runs: dict[str, dict[str, object]] = {}
    audit_rows: list[dict[str, object]] = []
    flag_counts: dict[str, int] = {}
    for run in sorted(set(int(value) for value in runs)):
        inherited_raw = base_runs.get(str(run), {})
        if not isinstance(inherited_raw, Mapping):
            raise ValueError(f"base manifest entry for run {run} is not an object")
        inherited = dict(inherited_raw)
        rcdb = rcdb_records.get(run, RcdbRunConditions(run=run))
        charge = charge_records.get(run)
        run_class = assignments.get(run, inherited.get("run_class", "unclassified"))
        if not isinstance(run_class, str) or not run_class:
            run_class = "unclassified"
        if run_class not in run_classes:
            run_classes[run_class] = {
                "description": "Imported from the base manifest",
                "nominal_current_nA": None,
            }
        if run in assignments:
            nominal_current = run_classes[run_class].get("nominal_current_nA")
        else:
            nominal_current = inherited.get(
                "nominal_current_nA",
                run_classes[run_class].get("nominal_current_nA"),
            )
        if run_class in nominal:
            nominal_current = nominal[run_class]
        quality = current_quality(rcdb.beam_current_nA, inherited.get("rcdb_quality"))
        if qadb is None:
            qadb_is_covered = None
            qadb_misc = None
        elif qadb_covered is None:
            qadb_is_covered = None
            qadb_misc = True if run in qadb else None
        else:
            qadb_is_covered = run in qadb_covered
            qadb_misc = (run in qadb) if qadb_is_covered else None
        qadb_comment = None if qadb is None else qadb.get(run)

        entry: dict[str, object] = {
            "run_class": run_class,
            "nominal_current_nA": nominal_current,
            "rcdb_current_nA": rcdb.beam_current_nA,
            "rcdb_quality": quality,
            "target": rcdb.target,
            "torus_scale": rcdb.torus_scale,
            "run_config": rcdb.run_config,
            "qadb_covered": qadb_is_covered,
            "qadb_misc": qadb_misc,
            "qadb_misc_comment": qadb_comment,
            "notes": inherited.get("notes"),
        }
        if charge is not None:
            entry.update(
                {
                    "converter_charge_nC": charge.accumulated_charge_nC,
                    "converter_total_events": charge.total_events,
                    "converter_passed_qadb_events": charge.passed_qadb_events,
                    "converter_failed_qadb_events": charge.failed_qadb_events,
                }
            )
        manifest_runs[str(run)] = entry

        flags: list[str] = []
        if run_class == "unclassified":
            flags.append("unclassified")
        if rcdb.beam_current_nA is None:
            flags.append("missing_rcdb_current")
        elif rcdb.beam_current_nA < 0.0:
            flags.append("negative_rcdb_current")
        if rcdb.query_error:
            flags.append("rcdb_query_error")
        if processing_supplied and charge is None:
            flags.append("missing_processing_run")
        if charge is not None:
            if charge.accumulated_charge_nC <= 0.0:
                flags.append("nonpositive_converter_charge")
            if charge.total_events > 0 and charge.passed_qadb_events == 0:
                flags.append("all_converter_events_failed_qadb")
            if charge.total_events != (
                charge.passed_qadb_events + charge.failed_qadb_events
            ):
                flags.append("converter_event_count_mismatch")
        if qadb_misc:
            flags.append("qadb_misc_review")
        if qadb is not None and qadb_covered is None and run not in qadb:
            flags.append("qadb_coverage_unknown")
        elif qadb_covered is not None and not qadb_is_covered:
            flags.append("qadb_not_covered")
        for flag in flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

        audit_rows.append(
            {
                "run": run,
                "run_class": run_class,
                "nominal_current_nA": nominal_current,
                "rcdb_current_nA": rcdb.beam_current_nA,
                "rcdb_quality": quality,
                "target": rcdb.target,
                "torus_scale": rcdb.torus_scale,
                "run_config": rcdb.run_config,
                "qadb_covered": qadb_is_covered,
                "qadb_misc": qadb_misc,
                "qadb_misc_comment": qadb_comment,
                "converter_charge_nC": (
                    None if charge is None else charge.accumulated_charge_nC
                ),
                "converter_total_events": (
                    None if charge is None else charge.total_events
                ),
                "converter_passed_qadb_events": (
                    None if charge is None else charge.passed_qadb_events
                ),
                "converter_failed_qadb_events": (
                    None if charge is None else charge.failed_qadb_events
                ),
                "audit_flags": ";".join(flags),
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "provenance": dict(base.get("provenance", {}))
        if isinstance(base.get("provenance"), Mapping)
        else {},
        "run_classes": run_classes,
        "runs": manifest_runs,
    }
    summary = {
        "runs": len(manifest_runs),
        "runs_with_rcdb_current": sum(
            record.beam_current_nA is not None for record in rcdb_records.values()
        ),
        "runs_with_converter_metadata": sum(
            run in charge_records for run in map(int, manifest_runs)
        ),
        "runs_with_qadb_misc": (
            None
            if qadb is None
            else sum(run in qadb for run in map(int, manifest_runs))
        ),
        "runs_covered_by_qadb": (
            None
            if qadb_covered is None
            else sum(run in qadb_covered for run in map(int, manifest_runs))
        ),
        "converter_charge_nC": sum(
            record.accumulated_charge_nC for record in charge_records.values()
        ),
        "audit_flag_counts": dict(sorted(flag_counts.items())),
    }
    return manifest, audit_rows, summary


def write_audit_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "run",
        "run_class",
        "nominal_current_nA",
        "rcdb_current_nA",
        "rcdb_quality",
        "target",
        "torus_scale",
        "run_config",
        "qadb_covered",
        "qadb_misc",
        "qadb_misc_comment",
        "converter_charge_nC",
        "converter_total_events",
        "converter_passed_qadb_events",
        "converter_failed_qadb_events",
        "audit_flags",
    ]
    _write_tsv(path, fieldnames, rows)


def _write_tsv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: "" if value is None else value for key, value in row.items()}
            )
