#!/usr/bin/env python3

"""Compile reproducible RCDB/QADB/converter run-condition audit artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from eppi0.run_conditions_audit import (
    build_outputs,
    load_base_manifest,
    load_rcdb_tsv,
    load_run_charge_tsv,
    load_run_list,
    parse_class_assignments,
    parse_nominal_currents,
    parse_qadb_misc,
    parse_run_specification,
    query_qadb_misc,
    query_rcdb,
    read_run_charge_root,
    write_audit_tsv,
    write_rcdb_tsv,
    write_run_charge_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query RCDB, audit converter RunCharge metadata and QADB Misc comments, "
            "and write a run-current manifest for study_data_efficiency.py."
        )
    )
    run_input = parser.add_argument_group("run selection")
    run_input.add_argument(
        "--runs",
        action="append",
        default=[],
        metavar="LIST",
        help="Comma/whitespace-separated runs and inclusive ranges; may be repeated",
    )
    run_input.add_argument(
        "--run-list",
        action="append",
        type=Path,
        default=[],
        help="Text file containing runs/ranges; # begins a comment; may be repeated",
    )
    run_input.add_argument(
        "--base-manifest",
        type=Path,
        help=(
            "Existing manifest whose run classes, nominal currents, manual quality "
            "labels, and notes should be preserved"
        ),
    )
    run_input.add_argument(
        "--assign-class",
        action="append",
        default=[],
        metavar="CLASS=RUNS",
        help="Assign listed runs/ranges to an analysis class; may be repeated",
    )
    run_input.add_argument(
        "--nominal-current",
        action="append",
        default=[],
        metavar="CLASS=NA",
        help="Set a class nominal current in nA (or null); may be repeated",
    )

    rcdb_input = parser.add_argument_group("RCDB")
    rcdb_input.add_argument(
        "--rcdb-connection",
        default=os.environ.get(
            "RCDB_CONNECTION", "mysql://rcdb@clasdb-farm.jlab.org/rcdb"
        ),
        help="RCDB connection string (default: RCDB_CONNECTION or farm read-only DB)",
    )
    rcdb_input.add_argument(
        "--rcdb-input",
        type=Path,
        help="Reuse a previously written rcdb_conditions.tsv instead of querying RCDB",
    )

    converter = parser.add_argument_group("converter metadata")
    converter_source = converter.add_mutually_exclusive_group()
    converter_source.add_argument(
        "--processing-root",
        action="append",
        type=Path,
        default=[],
        help="Converter ROOT file containing RunCharge; may be repeated",
    )
    converter_source.add_argument(
        "--run-charge-input",
        type=Path,
        help="Reuse a previously written converter_run_charge.tsv",
    )

    qadb = parser.add_argument_group("QADB Misc audit")
    qadb_source = qadb.add_mutually_exclusive_group()
    qadb_source.add_argument(
        "--qadb-datasets",
        help="Comma-separated qadb-info datasets; enables a live Misc query",
    )
    qadb_source.add_argument(
        "--qadb-misc-input",
        type=Path,
        help="Reuse qadb-info misc JSON or legacy --code output",
    )
    qadb.add_argument(
        "--qadb-info",
        default="qadb-info",
        help="qadb-info executable or path (default: qadb-info from PATH)",
    )

    output = parser.add_argument_group("output metadata")
    output.add_argument("--run-group", required=True, help="Dataset label, e.g. RGA or RGK")
    output.add_argument("--period", help="Dataset period, e.g. Fall 2018")
    output.add_argument("--beam-energy-gev", type=float)
    output.add_argument("--output-dir", type=Path, required=True)
    output.add_argument(
        "--manifest-name",
        default="run_currents.json",
        help="Output manifest filename (default: run_currents.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run_group.strip():
        raise ValueError("--run-group must not be empty")
    if args.beam_energy_gev is not None and args.beam_energy_gev <= 0.0:
        raise ValueError("--beam-energy-gev must be positive")
    base = load_base_manifest(args.base_manifest)
    class_assignments = parse_class_assignments(args.assign_class)
    nominal_currents = parse_nominal_currents(args.nominal_current)

    requested_runs: set[int] = set(class_assignments)
    for specification in args.runs:
        requested_runs.update(parse_run_specification(specification))
    for path in args.run_list:
        requested_runs.update(load_run_list(path))
    requested_runs.update(int(run) for run in base.get("runs", {}))

    if args.run_charge_input is not None:
        charge_records = load_run_charge_tsv(args.run_charge_input)
        processing_supplied = True
    else:
        charge_records = read_run_charge_root(args.processing_root)
        processing_supplied = bool(args.processing_root)
    requested_runs.update(charge_records)

    if args.rcdb_input is not None:
        rcdb_records = load_rcdb_tsv(args.rcdb_input)
        requested_runs.update(rcdb_records)
    else:
        if not requested_runs:
            raise ValueError(
                "no runs were supplied; use --runs, --run-list, --base-manifest, "
                "or --processing-root"
            )
        rcdb_records = query_rcdb(requested_runs, args.rcdb_connection)

    if not requested_runs:
        raise ValueError("the supplied inputs contain no runs")

    raw_qadb: str | None = None
    qadb_comments: dict[int, str] | None = None
    qadb_covered_runs: set[int] | None = None
    if args.qadb_misc_input is not None:
        raw_qadb = args.qadb_misc_input.read_text(encoding="utf-8")
        qadb_comments, qadb_covered_runs = parse_qadb_misc(raw_qadb)
    elif args.qadb_datasets:
        raw_qadb, qadb_comments, qadb_covered_runs = query_qadb_misc(
            requested_runs,
            args.qadb_datasets,
            executable=args.qadb_info,
        )

    manifest, audit_rows, summary = build_outputs(
        requested_runs,
        rcdb_records,
        charge_records,
        run_group=args.run_group,
        period=args.period,
        beam_energy_gev=args.beam_energy_gev,
        base_manifest=base,
        class_assignments=class_assignments,
        nominal_currents=nominal_currents,
        qadb_comments=qadb_comments,
        qadb_covered_runs=qadb_covered_runs,
        processing_supplied=processing_supplied,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest["provenance"].update(
        {
            "generated_by": "analysis/audit_run_conditions.py",
            "generated_at": timestamp,
            "rcdb_connection": args.rcdb_connection if args.rcdb_input is None else None,
            "rcdb_input": None if args.rcdb_input is None else str(args.rcdb_input.resolve()),
            "rcdb_conditions": ["beam_current", "target", "torus_scale", "run_config"],
            "processing_roots": [str(path.resolve()) for path in args.processing_root],
            "run_charge_input": None
            if args.run_charge_input is None
            else str(args.run_charge_input.resolve()),
            "qadb_datasets": args.qadb_datasets,
            "qadb_misc_input": None
            if args.qadb_misc_input is None
            else str(args.qadb_misc_input.resolve()),
            "base_manifest": None
            if args.base_manifest is None
            else str(args.base_manifest.resolve()),
            "notes": (
                "QADB Misc entries are audit metadata requiring analyst review; "
                "they are not automatic run rejections."
            ),
        }
    )
    summary.update(
        {
            "generated_at": timestamp,
            "run_group": args.run_group,
            "manifest": str((output_dir / args.manifest_name).resolve()),
        }
    )

    rcdb_path = output_dir / "rcdb_conditions.tsv"
    charge_path = output_dir / "converter_run_charge.tsv"
    audit_path = output_dir / "run_audit.tsv"
    manifest_path = output_dir / args.manifest_name
    summary_path = output_dir / "audit_summary.json"
    write_rcdb_tsv(rcdb_path, rcdb_records)
    write_run_charge_tsv(charge_path, charge_records)
    write_audit_tsv(audit_path, audit_rows)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if raw_qadb is not None:
        (output_dir / "qadb_misc.txt").write_text(raw_qadb, encoding="utf-8")

    print(f"Runs audited: {summary['runs']}")
    print(f"Runs with RCDB current: {summary['runs_with_rcdb_current']}")
    print(f"Runs with converter metadata: {summary['runs_with_converter_metadata']}")
    if summary["runs_with_qadb_misc"] is not None:
        if summary["runs_covered_by_qadb"] is not None:
            print(f"Runs covered by QADB: {summary['runs_covered_by_qadb']}")
        print(f"Runs requiring QADB Misc review: {summary['runs_with_qadb_misc']}")
    for flag, count in summary["audit_flag_counts"].items():
        print(f"Audit flag {flag}: {count}")
    for path in (rcdb_path, charge_path, audit_path, manifest_path, summary_path):
        print(f"Wrote {path}")
    if raw_qadb is not None:
        print(f"Wrote {output_dir / 'qadb_misc.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
