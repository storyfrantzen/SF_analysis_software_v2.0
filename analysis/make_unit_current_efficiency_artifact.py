#!/usr/bin/env python3

"""Create a unit-current-weight artifact while preserving audited run selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from eppi0.current_efficiency import unit_weight_selection_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fit_summary",
        type=Path,
        help="fit_summary.json from study_data_efficiency.py",
    )
    parser.add_argument(
        "--run-yields",
        type=Path,
        help="run_yields.csv (default: sibling of fit_summary.json)",
    )
    parser.add_argument("--response-meta", type=Path, required=True)
    parser.add_argument("--reference-current-na", type=float, required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument(
        "--exclude-class",
        action="append",
        default=[],
        help="Run class to exclude downstream; repeat as needed",
    )
    parser.add_argument(
        "--exclude-run",
        action="append",
        type=int,
        default=[],
        help="Additional run to exclude downstream; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def optional_text(value: str) -> str | None:
    value = value.strip()
    return None if value in {"", "None"} else value


def optional_float(value: str) -> float | None:
    value = optional_text(value)
    return None if value is None else float(value)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean value in run-yield table: {value!r}")


def load_run_records(path: Path) -> list[SimpleNamespace]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"run-yield table is empty: {path}")
    return [
        SimpleNamespace(
            run=int(row["run"]),
            run_class=optional_text(row["run_class"]),
            current_nA=optional_float(row["current_nA"]),
            current_quality=optional_text(row["current_quality"]),
            nominal_current_nA=optional_float(row["nominal_current_nA"]),
            candidate_events=int(row["candidate_events"]),
            signal_events=float(row["signal_events"]),
            charge_c=float(row["charge_c"]),
            included=parse_bool(row["included"]),
            exclusion_reason=row["exclusion_reason"],
        )
        for row in rows
    ]


def main() -> int:
    args = parse_args()
    fit_summary = args.fit_summary.resolve()
    run_yields = (
        args.run_yields.resolve()
        if args.run_yields is not None
        else fit_summary.with_name("run_yields.csv")
    )
    response_meta = args.response_meta.resolve()
    with fit_summary.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    records = load_run_records(run_yields)

    validation = summary.get("validation", {})
    original_charge = validation.get("stored_total_charge_c")
    if original_charge is None:
        original_charge = validation.get("run_charge_sum_c")
    if original_charge is None:
        raise ValueError("fit summary has no stored or run-summed beam charge")

    filters = summary.get("filters", {})
    inherited_classes = filters.get("exclude_classes_downstream") or []
    inherited_runs = filters.get("exclude_runs_downstream") or []
    automatic_runs = (
        filters.get("automatic_low_yield_excluded_runs_downstream") or []
    )
    excluded_classes = sorted(set(inherited_classes).union(args.exclude_class))
    excluded_runs = sorted(
        {int(run) for run in inherited_runs}
        .union(int(run) for run in automatic_runs)
        .union(args.exclude_run)
    )

    payload = unit_weight_selection_artifact(
        reference_current_nA=args.reference_current_na,
        reference_label=args.reference_label,
        reference_response_meta=response_meta,
        run_records=records,
        sources={
            "fit_summary": str(fit_summary),
            "run_yields": str(run_yields),
            "response_meta": str(response_meta),
            "selection": summary.get("selection", {}),
            "reason": (
                "unit current weights used because a matched zero-current GEMC "
                "sample is unavailable"
            ),
        },
        analysis_excluded_classes=excluded_classes,
        analysis_excluded_runs=excluded_runs,
        original_beam_charge_c=float(original_charge),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selection = payload["analysis_selection"]
    print(f"Excluded classes: {', '.join(excluded_classes) or 'none'}")
    print(
        "Excluded runs: "
        + (", ".join(str(run) for run in selection["excluded_runs"]) or "none")
    )
    print(f"Original beam charge: {selection['original_beam_charge_c']:.9g} C")
    print(f"Analysis beam charge: {selection['analysis_beam_charge_c']:.9g} C")
    print(f"Reference D: {payload['reference']['D']:.9g}")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
