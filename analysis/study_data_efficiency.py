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
from eppi0.current_efficiency import (
    CurrentEfficiencyCorrection,
    RelativeLinearEfficiency,
    correction_artifact,
)
from eppi0.gemc_efficiency import (
    attach_relative_gemc_efficiencies,
    fit_linear_efficiency,
    load_gemc_efficiency_samples,
    load_gemc_efficiencies,
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
        "--minimum-group-charge-fraction",
        type=float,
        default=0.0,
        help=(
            "Exclude an otherwise eligible run when its QADB charge is less than this "
            "fraction of the eligible charge in its run-class group (default: 0)"
        ),
    )
    gemc_input = parser.add_mutually_exclusive_group()
    gemc_input.add_argument(
        "--gemc-manifest",
        type=Path,
        help=(
            "Optional JSON manifest of current-tagged GEMC response_meta.npz files; "
            "fit and overlay their accepted/generated efficiencies"
        ),
    )
    gemc_input.add_argument(
        "--gemc-sample",
        action="append",
        nargs=3,
        metavar=("LABEL", "CURRENT_NA", "RESPONSE_META"),
        help=(
            "Add one GEMC response point directly; repeat for zero-background and "
            "merged-current samples"
        ),
    )
    parser.add_argument(
        "--reference-current-na",
        type=float,
        help=(
            "Merged-background current of the GEMC response used downstream. When omitted, "
            "infer it only if the GEMC inputs contain exactly one positive current."
        ),
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
        minimum_group_charge_fraction=args.minimum_group_charge_fraction,
    )
    groups = aggregate_current_groups(records)
    fit = fit_linear_yield(records, groups, fit_level=args.fit_level)
    attach_relative_efficiencies(groups, fit)

    gemc_points = None
    gemc_fit = None
    gemc_validation = None
    gemc_source = None
    if args.gemc_manifest is not None:
        gemc_points, gemc_validation = load_gemc_efficiencies(args.gemc_manifest)
        gemc_source = {"manifest": str(args.gemc_manifest.resolve())}
    elif args.gemc_sample is not None:
        sample_entries = [
            {
                "label": label,
                "current_nA": float(current),
                "response_meta": response_meta,
            }
            for label, current, response_meta in args.gemc_sample
        ]
        gemc_points, gemc_validation = load_gemc_efficiency_samples(
            sample_entries, base_directory=Path.cwd()
        )
        gemc_source = {"command_line_samples": sample_entries}
    if gemc_points is not None:
        gemc_fit = fit_linear_efficiency(gemc_points)
        attach_relative_gemc_efficiencies(gemc_points, gemc_fit)
    elif args.reference_current_na is not None:
        raise ValueError("--reference-current-na requires GEMC efficiency inputs")

    correction = None
    correction_payload = None
    reference_point = None
    if gemc_points is not None and gemc_fit is not None:
        reference_point = resolve_reference_point(gemc_points, args.reference_current_na)
        data_model = RelativeLinearEfficiency(
            intercept=fit.intercept_events_per_nC,
            slope_per_nA=fit.slope_events_per_nC_per_nA,
            covariance=tuple(tuple(row) for row in fit.covariance),
        )
        gemc_model = RelativeLinearEfficiency(
            intercept=gemc_fit.intercept,
            slope_per_nA=gemc_fit.slope_per_nA,
            covariance=tuple(tuple(row) for row in gemc_fit.covariance),
        )
        correction_payload = correction_artifact(
            data_model=data_model,
            gemc_model=gemc_model,
            reference_current_nA=reference_point.current_nA,
            reference_label=reference_point.label,
            reference_response_meta=Path(reference_point.response_meta),
            run_records=records,
            sources={
                "data_sample": str(args.sample.resolve()),
                "current_manifest": str(args.manifest.resolve()),
                "selection_mask": (
                    str(args.selection_mask.resolve()) if args.selection_mask else None
                ),
                "gemc": gemc_source,
            },
        )
        correction = CurrentEfficiencyCorrection(
            data_model=data_model,
            gemc_model=gemc_model,
            reference_current_nA=reference_point.current_nA,
            reference_label=reference_point.label,
            reference_response_meta=reference_point.response_meta,
            reference_response_meta_sha256=correction_payload["reference"][
                "response_meta_sha256"
            ],
            run_currents_nA={
                int(run): float(values["current_nA"])
                for run, values in correction_payload["runs"].items()
            },
            payload=correction_payload,
        )

    warnings = study_warnings(
        args, records, groups, fit, validation, gemc_points, gemc_fit, gemc_validation
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_csv = args.output_dir / "run_yields.csv"
    group_csv = args.output_dir / "current_group_yields.csv"
    summary_json = args.output_dir / "fit_summary.json"
    plots_pdf = args.output_dir / "data_efficiency_diagnostics.pdf"
    gemc_csv = args.output_dir / "gemc_efficiency_points.csv"
    correction_json = args.output_dir / "current_efficiency_correction.json"
    write_csv(run_csv, run_yield_rows(records))
    write_csv(group_csv, current_group_rows(groups))
    if gemc_points is not None:
        write_csv(gemc_csv, [asdict(point) for point in gemc_points])

    summary = {
        "schema_version": 2,
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
            "minimum_group_charge_fraction": args.minimum_group_charge_fraction,
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
    if gemc_points is not None and gemc_fit is not None:
        summary["gemc"] = {
            "source": gemc_source,
            "validation": gemc_validation,
            "fit": asdict(gemc_fit),
            "points": [asdict(point) for point in gemc_points],
        }
        summary["current_efficiency_correction"] = {
            "artifact": str(correction_json.resolve()),
            "method": correction_payload["method"],
            "reference": correction_payload["reference"],
            "fit_included_runs": correction_payload["fit_included_runs"],
            "application_run_count": len(correction_payload["runs"]),
        }
        correction_json.write_text(
            json.dumps(correction_payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_plots(
        plots_pdf,
        records,
        groups,
        fit,
        args.selection_mask is not None,
        gemc_points=gemc_points,
        gemc_fit=gemc_fit,
        correction=correction,
    )

    print(f"Candidate events: {validation['candidate_events']}")
    print(f"Signal events: {validation['signal_events']}")
    print(f"Included runs: {sum(record.included for record in records)}")
    print(f"Current groups: {len(groups)}")
    print(
        "Runs below minimum group charge fraction: "
        f"{len(validation['runs_below_minimum_group_charge_fraction'])}"
    )
    if validation["runs_below_minimum_group_charge_fraction"]:
        print(
            "Low-charge run exclusions: "
            + ", ".join(
                str(run)
                for run in validation["runs_below_minimum_group_charge_fraction"]
            )
        )
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
    if gemc_fit is not None:
        print(
            "GEMC zero-current efficiency: "
            f"{gemc_fit.intercept:.8g} +/- {gemc_fit.intercept_uncertainty:.8g}"
        )
        print(
            "GEMC current slope: "
            f"{gemc_fit.slope_per_nA:.8g} +/- "
            f"{gemc_fit.slope_uncertainty_per_nA:.8g} per nA"
        )
        print(
            "Reference correction: "
            f"I_ref={correction.reference_current_nA:.8g} nA, "
            f"D(I_ref)={correction.d_reference:.8g}"
        )
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote {run_csv}")
    print(f"Wrote {group_csv}")
    if gemc_points is not None:
        print(f"Wrote {gemc_csv}")
        print(f"Wrote {correction_json}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {plots_pdf}")
    return 0


def resolve_reference_point(points, requested_current_nA):
    if requested_current_nA is None:
        positive_currents = sorted(
            {float(point.current_nA) for point in points if point.current_nA > 0.0}
        )
        if len(positive_currents) != 1:
            raise ValueError(
                "--reference-current-na is required unless the GEMC inputs contain exactly "
                "one positive merged-background current"
            )
        requested_current_nA = positive_currents[0]
    requested_current_nA = float(requested_current_nA)
    if not np.isfinite(requested_current_nA) or requested_current_nA < 0.0:
        raise ValueError("--reference-current-na must be finite and nonnegative")
    matches = [
        point
        for point in points
        if np.isclose(point.current_nA, requested_current_nA, rtol=0.0, atol=1.0e-9)
    ]
    if len(matches) != 1:
        raise ValueError(
            "the reference current must identify exactly one supplied GEMC response; "
            f"found {len(matches)} matches at {requested_current_nA:g} nA"
        )
    return matches[0]


def study_warnings(
    args,
    records,
    groups,
    fit,
    validation,
    gemc_points=None,
    gemc_fit=None,
    gemc_validation=None,
) -> list[str]:
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
    if validation.get("runs_below_minimum_group_charge_fraction"):
        warnings.append(
            "Runs below the minimum within-group charge fraction were excluded after the "
            "ordinary run filters."
        )
    if gemc_fit is not None and gemc_fit.points < 3:
        warnings.append(
            "The GEMC current fit has only two points; it implements the assumed linear "
            "zero-background-to-merged interpolation but cannot test curvature."
        )
    if gemc_validation is not None and not gemc_validation.get(
        "truth_totals_match_reference", False
    ):
        warnings.append(
            "GEMC truth totals differ among current samples; use matched generated events "
            "or validate that generator-distribution differences do not bias the global efficiency."
        )
    if gemc_points is not None and any(
        point.uncertainty_model == "binomial_effective_weight_approximation"
        for point in gemc_points
    ):
        warnings.append(
            "At least one GEMC sample is weighted; its binomial statistical uncertainty is "
            "only an approximation unless an explicit uncertainty is supplied in the manifest."
        )
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


def _linear_prediction_uncertainty(
    current_nA: np.ndarray, covariance: list[list[float]]
) -> np.ndarray:
    current = np.asarray(current_nA, dtype=float)
    design = np.column_stack((np.ones(current.size), current))
    covariance_array = np.asarray(covariance, dtype=float)
    variance = np.einsum("ij,jk,ik->i", design, covariance_array, design)
    return np.sqrt(np.maximum(variance, 0.0))


def write_plots(
    path,
    records,
    groups,
    fit,
    has_selection_mask: bool,
    *,
    gemc_points=None,
    gemc_fit=None,
    correction=None,
) -> None:
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
    excluded_classes = sorted({record.run_class or "unclassified" for record in excluded})
    plotted_classes = sorted(set(classes).union(excluded_classes))
    colors = {
        name: plt.get_cmap("tab10")(index % 10)
        for index, name in enumerate(plotted_classes)
    }
    selection_label = "fixed-mask signal" if has_selection_mask else "all selected candidates"
    show_relative = gemc_points is not None and gemc_fit is not None
    if show_relative and fit.intercept_events_per_nC <= 0.0:
        raise ValueError("data zero-current intercept must be positive for a GEMC overlay")
    if show_relative and correction is None:
        raise ValueError("a current-efficiency correction model is required for a GEMC overlay")
    data_scale = 1.0 / fit.intercept_events_per_nC if show_relative else 1.0

    with PdfPages(path) as pdf:
        fig, (axis, residual_axis) = plt.subplots(
            2,
            1,
            figsize=(8.5, 6.5),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        for name in excluded_classes:
            members = [
                record
                for record in excluded
                if (record.run_class or "unclassified") == name
            ]
            axis.scatter(
                [record.current_nA for record in members],
                [record.yield_events_per_nC * data_scale for record in members],
                color=colors[name],
                marker="x",
                s=28,
                linewidths=1.0,
                alpha=0.65,
                label=f"{name} excluded",
                zorder=1,
            )
        for name in classes:
            members = [record for record in included if record.run_class == name]
            axis.errorbar(
                [record.current_nA for record in members],
                [record.yield_events_per_nC * data_scale for record in members],
                yerr=[
                    record.statistical_uncertainty_events_per_nC * data_scale
                    for record in members
                ],
                fmt="o",
                markersize=4,
                linewidth=0.8,
                color=colors[name],
                alpha=0.7,
                label=f"{name} runs",
            )
        axis.errorbar(
            [group.effective_current_nA for group in groups],
            [group.yield_events_per_nC * data_scale for group in groups],
            yerr=[group.statistical_uncertainty_events_per_nC * data_scale for group in groups],
            fmt="o",
            markersize=6,
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            linewidth=1.2,
            label="charge-aggregated groups",
            zorder=4,
        )
        max_current = max(
            [group.effective_current_nA for group in groups]
            + ([point.current_nA for point in gemc_points] if gemc_points else [])
            + [1.0]
        )
        fit_current = np.linspace(0.0, max_current * 1.08, 200)
        if show_relative:
            data_fit_curve = correction.data_model.relative_efficiency(fit_current)
            data_fit_uncertainty = correction.data_model.relative_uncertainty(fit_current)
        else:
            data_fit_curve = fit.predict(fit_current)
            data_fit_uncertainty = _linear_prediction_uncertainty(
                fit_current, fit.covariance
            )
        axis.plot(
            fit_current,
            data_fit_curve,
            color="black",
            label="data linear fit" if show_relative else "linear fit",
            zorder=3,
        )
        axis.fill_between(
            fit_current,
            data_fit_curve - data_fit_uncertainty,
            data_fit_curve + data_fit_uncertainty,
            color="0.35",
            alpha=0.18,
            linewidth=0.0,
            label="data fit uncertainty" if show_relative else "fit uncertainty",
            zorder=2,
        )
        axis.scatter(
            [0.0],
            [fit.intercept_events_per_nC * data_scale],
            marker="*",
            s=110,
            color="black",
            label="zero-current extrapolation",
            zorder=5,
        )
        if show_relative:
            gemc_scale = 1.0 / gemc_fit.intercept
            axis.errorbar(
                [point.current_nA for point in gemc_points],
                [point.efficiency * gemc_scale for point in gemc_points],
                yerr=[point.statistical_uncertainty * gemc_scale for point in gemc_points],
                fmt="D",
                markersize=6,
                markerfacecolor="white",
                color="#1f77b4",
                linewidth=1.2,
                label="GEMC accepted/generated",
                zorder=5,
            )
            axis.plot(
                fit_current,
                gemc_fit.predict(fit_current) * gemc_scale,
                color="#1f77b4",
                linestyle="--",
                linewidth=1.5,
                label="GEMC linear fit",
            )
            data_slope = (
                correction.data_model.slope_per_nA
                / correction.data_model.intercept
            )
            data_slope_uncertainty = float(
                correction.data_model.relative_uncertainty(1.0)
            )
            gemc_slope = (
                correction.gemc_model.slope_per_nA
                / correction.gemc_model.intercept
            )
            gemc_slope_uncertainty = float(
                correction.gemc_model.relative_uncertainty(1.0)
            )
            axis.text(
                0.985,
                0.975,
                (
                    "Normalized slopes\n"
                    rf"$\eta_{{\mathrm{{data}}}}:\ "
                    rf"({data_slope * 1.0e3:.3f}\pm"
                    rf"{data_slope_uncertainty * 1.0e3:.3f})"
                    rf"\times10^{{-3}}\ \mathrm{{nA}}^{{-1}}$"
                    "\n"
                    rf"$\eta_{{\mathrm{{GEMC}}}}:\ "
                    rf"({gemc_slope * 1.0e3:.3f}\pm"
                    rf"{gemc_slope_uncertainty * 1.0e3:.3f})"
                    rf"\times10^{{-3}}\ \mathrm{{nA}}^{{-1}}$"
                ),
                transform=axis.transAxes,
                fontsize="x-small",
                va="top",
                ha="right",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "0.7",
                    "alpha": 0.9,
                },
                zorder=8,
            )
            axis.set_ylabel(r"Relative efficiency $\eta(I)$")
            axis.set_title(f"RGK data/GEMC current study: {selection_label}")
        else:
            axis.set_ylabel("Yield (events/nC)")
            axis.set_title(f"RGK data current study: {selection_label}")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize="x-small", ncol=3)

        group_current = np.asarray([group.effective_current_nA for group in groups])
        if show_relative:
            d_curve, _ = correction.d_factor(fit_current)
            residual_axis.plot(
                fit_current,
                d_curve,
                color="black",
                label=r"$D(I)$ from fitted $\eta$",
            )
            group_data_eta = np.asarray(
                [group.yield_events_per_nC for group in groups], dtype=float
            ) / fit.intercept_events_per_nC
            group_data_sigma = np.asarray(
                [group.statistical_uncertainty_events_per_nC for group in groups],
                dtype=float,
            ) / fit.intercept_events_per_nC
            group_mc_eta = correction.gemc_model.relative_efficiency(group_current)
            group_mc_sigma = correction.gemc_model.relative_uncertainty(group_current)
            group_d = group_data_eta / group_mc_eta
            group_d_sigma = np.sqrt(
                (group_data_sigma / group_mc_eta) ** 2
                + (group_data_eta * group_mc_sigma / group_mc_eta**2) ** 2
            )
            residual_axis.errorbar(
                group_current,
                group_d,
                yerr=group_d_sigma,
                fmt="o",
                markersize=6,
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                label="data group / MC fit",
            )
            residual_axis.axhline(
                1.0,
                color="0.35",
                linewidth=1.2,
                linestyle=":",
            )
            residual_axis.set_ylabel(r"$D(I)=\eta_{data}/\eta_{MC}$")
            residual_axis.legend(fontsize="xx-small")
        else:
            group_residual = np.asarray(
                [group.yield_events_per_nC for group in groups]
            ) - fit.predict(group_current)
            group_uncertainty = np.asarray(
                [group.statistical_uncertainty_events_per_nC for group in groups]
            )
            residual_axis.errorbar(
                group_current,
                group_residual / group_uncertainty,
                yerr=np.ones(group_current.size),
                fmt="o",
                markersize=6,
                color="black",
            )
            residual_axis.axhline(0.0, color="0.3", linewidth=1.0)
            residual_axis.set_ylabel("Pull")
        residual_axis.set_xlabel("RCDB beam current (nA)")
        residual_axis.set_xlim(-0.03 * float(fit_current[-1]), float(fit_current[-1]))
        residual_axis.grid(True, alpha=0.25)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if show_relative:
            group_residual = np.asarray(
                [group.yield_events_per_nC for group in groups]
            ) - fit.predict(group_current)
            group_uncertainty = np.asarray(
                [group.statistical_uncertainty_events_per_nC for group in groups]
            )
            fig, pull_axis = plt.subplots(figsize=(8.5, 4.5))
            pull_axis.errorbar(
                group_current,
                group_residual / group_uncertainty,
                yerr=np.ones(group_current.size),
                fmt="o",
                markersize=6,
                color="black",
            )
            pull_axis.axhline(0.0, color="0.3", linewidth=1.0)
            pull_axis.set_xlabel("RCDB beam current (nA)")
            pull_axis.set_ylabel("Pull")
            pull_axis.set_title("Data current-fit pulls")
            pull_axis.grid(True, alpha=0.25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        fig, (yield_axis, charge_axis) = plt.subplots(
            2,
            1,
            figsize=(8.5, 8.5),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0]},
        )
        groups_by_name = {group.group: group for group in groups}
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
            group = groups_by_name.get(name)
            if group is not None:
                yield_axis.axhline(
                    group.yield_events_per_nC,
                    color=colors[name],
                    linestyle=":",
                    linewidth=1.6,
                    zorder=1,
                )
            charge_axis.scatter(
                [record.run for record in members],
                [record.charge_nC for record in members],
                s=18,
                color=colors[name],
            )
        yield_axis.plot(
            [],
            [],
            color="0.3",
            linestyle=":",
            linewidth=1.6,
            label="charge-weighted group mean",
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
