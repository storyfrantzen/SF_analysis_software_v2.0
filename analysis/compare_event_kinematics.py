#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.event_kinematics_comparison import (  # noqa: E402
    METRIC_FIELDS,
    common_variables,
    render_comparison_report,
)
from eppi0.event_kinematics_diagnostics import (  # noqa: E402
    reconstructed_topology,
    report_summary,
)
from plot_event_kinematics import (  # noqa: E402
    file_record,
    load_selection_mask,
    read_selected_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare final selected data and GEMC event kinematics, topology "
            "integrated and by reconstructed topology."
        )
    )
    parser.add_argument("data_root", type=Path)
    parser.add_argument("gemc_root", type=Path)
    parser.add_argument("--data-mask", type=Path, required=True)
    parser.add_argument("--gemc-mask", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="JSON summary path")
    parser.add_argument("--metrics", type=Path, help="CSV metrics path")
    parser.add_argument("--label", required=True, help="Comparison title")
    parser.add_argument("--data-label", default="Data")
    parser.add_argument("--gemc-label", default="GEMC")
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--minimum-ratio-count", type=int, default=5)
    parser.add_argument("--hash-inputs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_ratio_count < 1:
        raise ValueError("--minimum-ratio-count must be positive")
    config = json.loads(args.config.read_text())
    data, data_tree, data_rows = read_selected_root(
        args.data_root, args.tree, args.dictionary
    )
    gemc, gemc_tree, gemc_rows = read_selected_root(
        args.gemc_root, args.tree, args.dictionary
    )
    data_mask = load_selection_mask(args.data_mask, data_rows)
    gemc_mask = load_selection_mask(args.gemc_mask, gemc_rows)
    data_topology = reconstructed_topology(data)
    gemc_topology = reconstructed_topology(gemc)
    shared_variables = common_variables(data, gemc)
    if not shared_variables:
        raise RuntimeError("data and GEMC inputs have no common plotted variables")

    provenance = [
        f"Data ROOT: {args.data_root.resolve()}",
        f"Data mask: {args.data_mask.resolve()}",
        f"GEMC ROOT: {args.gemc_root.resolve()}",
        f"GEMC mask: {args.gemc_mask.resolve()}",
        f"Analysis config: {args.config.resolve()}",
        f"Trees: data={data_tree}, GEMC={gemc_tree}",
    ]
    pages, metric_rows = render_comparison_report(
        args.output,
        data,
        gemc,
        data_mask,
        gemc_mask,
        data_topology,
        gemc_topology,
        label=args.label,
        data_label=args.data_label,
        gemc_label=args.gemc_label,
        provenance_lines=provenance,
        minimum_ratio_count=args.minimum_ratio_count,
    )

    summary_path = args.summary or args.output.with_name(
        f"{args.output.stem}_summary.json"
    )
    metrics_path = args.metrics or args.output.with_name(
        f"{args.output.stem}_metrics.csv"
    )
    summary = {
        "schema_version": 1,
        "label": args.label,
        "comparison_definition": (
            "unit-normalized final reconstructed-candidate shapes after each "
            "sample's own exclusivity mask; GEMC is not kinematically reweighted"
        ),
        "interpretation_limitation": (
            "physics-coordinate differences may contain generator-population effects "
            "in addition to detector and reconstruction mismodeling"
        ),
        "beam_energy_GeV": float(config["beam_energy"]),
        "minimum_ratio_count": int(args.minimum_ratio_count),
        "pdf_pages": int(pages),
        "metric_rows": len(metric_rows),
        "common_plotted_variables": [item.branch for item in shared_variables],
        "data": {
            "label": args.data_label,
            "root": file_record(args.data_root, args.hash_inputs),
            "mask": file_record(args.data_mask, True),
            "selection": report_summary(data, data_mask, data_topology),
        },
        "gemc": {
            "label": args.gemc_label,
            "root": file_record(args.gemc_root, args.hash_inputs),
            "mask": file_record(args.gemc_mask, True),
            "selection": report_summary(gemc, gemc_mask, gemc_topology),
        },
        "analysis_config": file_record(args.config, True),
        "output_pdf": str(args.output.resolve()),
        "metrics_csv": str(metrics_path.resolve()),
    }
    write_json_atomic(summary_path, summary)
    write_metrics_atomic(metrics_path, metric_rows)
    print(f"Data final candidates: {summary['data']['selection']['selected_rows']}")
    print(f"GEMC final candidates: {summary['gemc']['selection']['selected_rows']}")
    print(f"Common plotted variables: {len(shared_variables)}")
    print(f"PDF pages: {pages}")
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {summary_path.resolve()}")
    print(f"Wrote {metrics_path.resolve()}")
    return 0


def write_json_atomic(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_metrics_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
