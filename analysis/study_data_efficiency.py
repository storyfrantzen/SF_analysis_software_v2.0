#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from eppi0.data_efficiency import (
    aggregate_current_groups,
    attach_relative_efficiencies,
    build_run_yields,
    current_group_rows,
    fit_linear_yield,
    run_yield_rows,
)


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "efficiency"
    / "rgk"
    / "6.535"
    / "run_currents.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the charge-normalized selected EPPI0 yield versus beam current "
            "and extrapolate it linearly to zero current."
        )
    )
    parser.add_argument("sample", type=Path, help="NPZ from export_selected_data.py")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--selection-mask",
        type=Path,
        help="Optional fixed event-level .npy/.npz signal-selection mask",
    )
    parser.add_argument(
        "--selection-mask-key",
        default="mask",
        help="Array name when --selection-mask is an NPZ (default: mask)",
    )
    parser.add_argument(
        "--include-classes",
        nargs="+",
        default=["P3", "P4"],
        help="Run classes admitted to the nominal fit (default: P3 P4)",
    )
    parser.add_argument(
        "--include-qualities",
        nargs="+",
        default=["unflagged"],
        help="RCDB current-quality labels admitted to the nominal fit",
    )
    parser.add_argument(
        "--include-run",
        action="append",
        type=int,
        default=[],
        help="Admit a run even when its class is not listed; may be repeated",
    )
    parser.add_argument(
        "--exclude-run",
        action="append",
        type=int,
        default=[],
        help="Exclude a run explicitly; may be repeated",
    )
    parser.add_argument(
        "--fit-level",
        choices=("groups", "runs"),
        default="groups",
        help="Fit charge-aggregated run classes or individual runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_efficiency/rgk_6.535"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, validation = build_run_yields(
        args.sample,
        args.manifest,
        selection_mask_path=args.selection_mask,
        selection_mask_key=args.selection_mask_key,
        include_classes=args.include_classes,
        include_qualities=args.include_qualities,
        include_runs=args.include_run,
        exclude_runs=args.exclude_run,
    )
    groups = aggregate_current_groups(records)
    fit = fit_linear_yield(records, groups, fit_level=args.fit_level)
    attach_relative_efficiencies(groups, fit)

    warnings = study_warnings(args, records, groups, fit, validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_csv = args.output_dir / "run_yields.csv"
    group_csv = args.output_dir / "current_group_yields.csv"
    summary_json = args.output_dir / "fit_summary.json"
    plots_pdf = args.output_dir / "data_efficiency_diagnostics.pdf"
    write_csv(run_csv, run_yield_rows(records))
    write_csv(group_csv, current_group_rows(groups))

    summary = {
        "schema_version": 1,
        "study": "RGK charge-normalized data yield versus beam current",
        "interpretation": (
            "Relative data efficiency only after fixed signal selection and "
            "run-condition compatibility have been validated."
        ),
        "sample": str(args.sample.resolve()),
        "manifest": str(args.manifest.resolve()),
        "selection": {
            "mode": "fixed_mask" if args.selection_mask else "all_selected_candidates",
            "mask": str(args.selection_mask.resolve()) if args.selection_mask else None,
            "mask_key": args.selection_mask_key if args.selection_mask else None,
        },
        "filters": {
            "include_classes": args.include_classes,
            "include_qualities": args.include_qualities,
            "include_runs": args.include_run,
            "exclude_runs": args.exclude_run,
        },
        "validation": validation,
        "fit": asdict(fit),
        "current_groups": [
            {
                **asdict(group),
                "run_numbers": group.run_numbers,
            }
            for group in groups
        ],
        "warnings": warnings,
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_plots(plots_pdf, records, groups, fit, args.selection_mask is not None)

    print(f"Candidate events: {validation['candidate_events']}")
    print(f"Signal events: {validation['signal_events']}")
    print(f"Included runs: {sum(record.included for record in records)}")
    print(f"Current groups: {len(groups)}")
    print(
        "Zero-current yield: "
        f"{fit.intercept_events_per_nC:.8g} +/- "
        f"{fit.intercept_uncertainty_events_per_nC:.8g} events/nC"
    )
    print(
        "Current slope: "
        f"{fit.slope_events_per_nC_per_nA:.8g} +/- "
        f"{fit.slope_uncertainty_events_per_nC_per_nA:.8g} events/(nC nA)"
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote {run_csv}")
    print(f"Wrote {group_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {plots_pdf}")
    return 0


def study_warnings(args, records, groups, fit, validation) -> list[str]:
    warnings: list[str] = []
    if args.selection_mask is None:
        warnings.append(
            "No fixed signal-selection mask was supplied; yields count all selected candidates "
            "and are not background-subtracted signal yields."
        )
    if "L5" in args.include_classes:
        warnings.append(
            "L5 is included; confirm its physics trigger and prescale are compatible with P3/P4."
        )
    if any(label != "unflagged" for label in args.include_qualities):
        warnings.append("Flagged RCDB current values are included.")
    if fit.points < 3:
        warnings.append(
            "The fit has fewer than three points; a linear zero-current extrapolation has no "
            "goodness-of-fit test and cannot test curvature."
        )
    if fit.intercept_events_per_nC <= 0.0:
        warnings.append(
            "The fitted zero-current yield is nonpositive; relative efficiencies are undefined."
        )
    if fit.slope_events_per_nC_per_nA > 0.0:
        warnings.append(
            "The fitted yield rises with current; residual background or changing run conditions "
            "may dominate over efficiency loss."
        )
    difference = validation.get("charge_difference_c")
    total = validation.get("stored_total_charge_c")
    if difference is not None and total is not None:
        tolerance = max(abs(float(total)) * 1.0e-10, 1.0e-15)
        if abs(float(difference)) > tolerance:
            warnings.append(
                "The sum of per-run charge differs from the stored file-level charge by "
                f"{float(difference):.8g} C."
            )
    if validation.get("charge_runs_missing_manifest"):
        warnings.append(
            "Some charge-bearing runs are missing from the current manifest and were excluded."
        )
    if not any(record.included for record in records):
        warnings.append("No runs passed the requested filters.")
    if len(groups) < 2:
        warnings.append("Fewer than two current groups were included.")
    return warnings


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_matplotlib() -> None:
    cache_root = Path(tempfile.gettempdir())
    mpl_dir = cache_root / "sf_analysis_matplotlib"
    xdg_dir = cache_root / "sf_analysis_xdg_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))


def write_plots(path, records, groups, fit, has_selection_mask: bool) -> None:
    _prepare_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    included = [record for record in records if record.included]
    excluded = [
        record
        for record in records
        if not record.included
        and record.current_nA is not None
        and record.yield_events_per_nC is not None
    ]
    classes = sorted({record.run_class for record in included if record.run_class})
    colors = {name: plt.get_cmap("tab10")(index % 10) for index, name in enumerate(classes)}
    selection_label = "fixed-mask signal" if has_selection_mask else "all selected candidates"

    with PdfPages(path) as pdf:
        fig, (axis, residual_axis) = plt.subplots(
            2, 1, figsize=(8.5, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        if excluded:
            axis.scatter(
                [record.current_nA for record in excluded],
                [record.yield_events_per_nC for record in excluded],
                color="0.75",
                marker="x",
                s=22,
                label="excluded runs",
                zorder=1,
            )
        for name in classes:
            members = [record for record in included if record.run_class == name]
            axis.errorbar(
                [record.current_nA for record in members],
                [record.yield_events_per_nC for record in members],
                yerr=[record.statistical_uncertainty_events_per_nC for record in members],
                fmt="o",
                markersize=4,
                linewidth=0.8,
                color=colors[name],
                alpha=0.7,
                label=f"{name} runs",
            )
        axis.errorbar(
            [group.effective_current_nA for group in groups],
            [group.yield_events_per_nC for group in groups],
            yerr=[group.statistical_uncertainty_events_per_nC for group in groups],
            fmt="s",
            markersize=7,
            color="black",
            linewidth=1.2,
            label="charge-aggregated groups",
            zorder=4,
        )
        max_current = max([group.effective_current_nA for group in groups] + [1.0])
        fit_current = np.linspace(0.0, max_current * 1.08, 200)
        axis.plot(fit_current, fit.predict(fit_current), color="black", label="linear fit")
        axis.scatter(
            [0.0],
            [fit.intercept_events_per_nC],
            marker="*",
            s=110,
            color="black",
            label="zero-current extrapolation",
            zorder=5,
        )
        axis.set_ylabel("Yield (events/nC)")
        axis.set_title(f"RGK data current study: {selection_label}")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize="small", ncol=2)

        group_current = np.asarray([group.effective_current_nA for group in groups])
        group_residual = np.asarray([group.yield_events_per_nC for group in groups]) - fit.predict(
            group_current
        )
        group_uncertainty = np.asarray(
            [group.statistical_uncertainty_events_per_nC for group in groups]
        )
        residual_axis.errorbar(
            group_current,
            group_residual / group_uncertainty,
            yerr=np.ones(group_current.size),
            fmt="s",
            color="black",
        )
        residual_axis.axhline(0.0, color="0.3", linewidth=1.0)
        residual_axis.set_xlabel("RCDB beam current (nA)")
        residual_axis.set_ylabel("Pull")
        residual_axis.grid(True, alpha=0.25)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, (yield_axis, charge_axis) = plt.subplots(2, 1, figsize=(8.5, 8.5), sharex=True)
        for name in classes:
            members = sorted(
                (record for record in included if record.run_class == name),
                key=lambda record: record.run,
            )
            yield_axis.errorbar(
                [record.run for record in members],
                [record.yield_events_per_nC for record in members],
                yerr=[record.statistical_uncertainty_events_per_nC for record in members],
                fmt="o",
                markersize=4,
                color=colors[name],
                label=name,
            )
            charge_axis.scatter(
                [record.run for record in members],
                [record.charge_nC for record in members],
                s=18,
                color=colors[name],
            )
        yield_axis.set_ylabel("Yield (events/nC)")
        yield_axis.set_title("Included-run stability")
        yield_axis.grid(True, alpha=0.25)
        yield_axis.legend(fontsize="small")
        charge_axis.set_xlabel("Run number")
        charge_axis.set_ylabel("QADB charge (nC)")
        charge_axis.grid(True, alpha=0.25)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
