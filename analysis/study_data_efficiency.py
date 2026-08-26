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
    LinearYieldFit,
    SharedFractionalYieldFit,
    aggregate_current_groups,
    attach_relative_efficiencies,
    build_run_yields,
    current_group_rows,
    fit_linear_yield,
    fit_shared_fractional_yield,
    run_yield_rows,
)
from eppi0.current_efficiency import (
    CurrentEfficiencyCorrection,
    RelativeLinearEfficiency,
    correction_artifact,
    response_meta_sha256,
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
        "--background-cuts",
        type=Path,
        help=(
            "Global data exclusivity-cut NPZ used to replace selected counts with "
            "topology-specific m_gg sideband-subtracted signal yields"
        ),
    )
    parser.add_argument(
        "--background-alpha-bootstrap",
        type=int,
        default=200,
        help="Poisson refits used to audit each sideband transfer-factor uncertainty",
    )
    parser.add_argument(
        "--background-seed",
        type=int,
        default=12345,
        help="Random seed for the sideband transfer-factor bootstrap",
    )
    parser.add_argument(
        "--include-classes",
        nargs="+",
        default=["P3", "P4"],
        help=(
            "Run classes admitted to the nominal fit (default: P3 P4); this "
            "does not select the downstream physics sample"
        ),
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
        help=(
            "Admit a run to the fit even when its class is not listed; may be repeated"
        ),
    )
    parser.add_argument(
        "--exclude-run",
        action="append",
        type=int,
        default=[],
        help=(
            "Exclude a run from the current fit only; does not remove it downstream; "
            "may be repeated"
        ),
    )
    parser.add_argument(
        "--fit-level",
        choices=("groups", "runs"),
        default="groups",
        help="Fit charge-aggregated run classes or individual runs",
    )
    parser.add_argument(
        "--shared-slope-period",
        action="append",
        default=[],
        metavar="PERIOD=CLASS1,CLASS2",
        help=(
            "Fit Y_period(I)=A_period(1+beta I), with a separate zero-current "
            "normalization for each explicitly mapped period and one shared fractional "
            "slope beta; repeat once per period"
        ),
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
    parser.add_argument(
        "--low-yield-sigma-threshold",
        type=float,
        default=5.0,
        help=(
            "Exclude a run from its fit group and downstream analysis when its yield "
            "is more than this many statistical standard deviations below the "
            "leave-one-out group mean; use 0 to disable (default: 5)"
        ),
    )
    parser.add_argument(
        "--exclude-class-downstream",
        action="append",
        default=[],
        help=(
            "Assign zero downstream event weight to every run in this manifest class "
            "and remove its charge from the analysis luminosity; may be repeated"
        ),
    )
    parser.add_argument(
        "--exclude-run-downstream",
        action="append",
        type=int,
        default=[],
        help=(
            "Assign zero downstream event weight to this run and remove its charge "
            "from the analysis luminosity; may be repeated"
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


def parse_shared_slope_periods(specifications: list[str]) -> dict[str, list[str]]:
    periods: dict[str, list[str]] = {}
    assigned_classes: dict[str, str] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                "--shared-slope-period must have the form PERIOD=CLASS1,CLASS2"
            )
        raw_period, raw_classes = specification.split("=", 1)
        period = raw_period.strip()
        classes = [value.strip() for value in raw_classes.split(",") if value.strip()]
        if not period or not classes:
            raise ValueError(
                "--shared-slope-period must have a nonempty period and class list"
            )
        if period in periods:
            raise ValueError(f"duplicate --shared-slope-period name: {period}")
        for run_class in classes:
            if run_class in assigned_classes:
                raise ValueError(
                    f"run class '{run_class}' is assigned to both "
                    f"'{assigned_classes[run_class]}' and '{period}'"
                )
            assigned_classes[run_class] = period
        periods[period] = classes
    return periods


def data_relative_model(
    fit: LinearYieldFit | SharedFractionalYieldFit,
) -> RelativeLinearEfficiency:
    if isinstance(fit, SharedFractionalYieldFit):
        variance = fit.fractional_slope_uncertainty_per_nA**2
        return RelativeLinearEfficiency(
            intercept=1.0,
            slope_per_nA=fit.fractional_slope_per_nA,
            covariance=((0.0, 0.0), (0.0, variance)),
        )
    return RelativeLinearEfficiency(
        intercept=fit.intercept_events_per_nC,
        slope_per_nA=fit.slope_events_per_nC_per_nA,
        covariance=tuple(tuple(row) for row in fit.covariance),
    )


def main() -> int:
    args = parse_args()
    shared_slope_periods = parse_shared_slope_periods(args.shared_slope_period)
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
        low_yield_sigma_threshold=args.low_yield_sigma_threshold,
        background_cuts_path=args.background_cuts,
        background_alpha_bootstrap=args.background_alpha_bootstrap,
        background_seed=args.background_seed,
    )
    groups = aggregate_current_groups(records)
    if shared_slope_periods:
        fit = fit_shared_fractional_yield(
            records,
            groups,
            period_classes=shared_slope_periods,
            fit_level=args.fit_level,
        )
    else:
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
    if (
        args.exclude_class_downstream or args.exclude_run_downstream
    ) and gemc_points is None:
        raise ValueError(
            "downstream exclusions require GEMC inputs so they can be persisted in "
            "current_efficiency_correction.json"
        )

    correction = None
    correction_payload = None
    reference_point = None
    if gemc_points is not None and gemc_fit is not None:
        automatic_downstream_exclusions = validation[
            "runs_below_group_yield_threshold"
        ]
        downstream_excluded_runs = sorted(
            set(args.exclude_run_downstream).union(automatic_downstream_exclusions)
        )
        reference_point = resolve_reference_point(gemc_points, args.reference_current_na)
        data_model = data_relative_model(fit)
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
            analysis_excluded_classes=args.exclude_class_downstream,
            analysis_excluded_runs=downstream_excluded_runs,
            original_beam_charge_c=(
                validation["stored_total_charge_c"]
                if validation["stored_total_charge_c"] is not None
                else validation["run_charge_sum_c"]
            ),
            data_quantity=(
                "m_gg sideband-subtracted data signal yield in events/nC"
                if args.background_cuts is not None
                else "selected data yield in events/nC"
            ),
            sources={
                "data_sample": str(args.sample.resolve()),
                "current_manifest": str(args.manifest.resolve()),
                "selection_mask": (
                    str(args.selection_mask.resolve()) if args.selection_mask else None
                ),
                "background_cuts": (
                    str(args.background_cuts.resolve()) if args.background_cuts else None
                ),
                "background_cuts_sha256": (
                    response_meta_sha256(args.background_cuts)
                    if args.background_cuts
                    else None
                ),
                "background_subtraction": validation["background_subtraction"],
                "data_fit": asdict(fit),
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
                if values.get("current_nA") is not None
            },
            payload=correction_payload,
            run_event_weights={
                int(run): float(values["event_weight"])
                for run, values in correction_payload["runs"].items()
            },
            analysis_beam_charge_c=correction_payload["analysis_selection"][
                "analysis_beam_charge_c"
            ],
            original_beam_charge_c=correction_payload["analysis_selection"][
                "original_beam_charge_c"
            ],
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
        "schema_version": 3,
        "study": (
            f"{validation['dataset'].get('run_group')} charge-normalized data yield "
            "versus beam current"
            if validation["dataset"].get("run_group")
            else "Charge-normalized data yield versus beam current"
        ),
        "interpretation": (
            "Relative data efficiency after a common signal definition and "
            "run-condition compatibility have been validated."
        ),
        "sample": str(args.sample.resolve()),
        "manifest": str(args.manifest.resolve()),
        "selection": {
            "mode": validation["yield_mode"],
            "mask": str(args.selection_mask.resolve()) if args.selection_mask else None,
            "mask_key": args.selection_mask_key if args.selection_mask else None,
            "background_cuts": (
                str(args.background_cuts.resolve()) if args.background_cuts else None
            ),
            "background_cuts_sha256": (
                response_meta_sha256(args.background_cuts)
                if args.background_cuts
                else None
            ),
            "background_alpha_bootstrap": (
                args.background_alpha_bootstrap if args.background_cuts else None
            ),
            "background_seed": args.background_seed if args.background_cuts else None,
        },
        "filters": {
            "include_classes": args.include_classes,
            "include_qualities": args.include_qualities,
            "include_runs": args.include_run,
            "exclude_runs": args.exclude_run,
            "minimum_group_charge_fraction": args.minimum_group_charge_fraction,
            "low_yield_sigma_threshold": args.low_yield_sigma_threshold,
            "shared_slope_periods": shared_slope_periods,
            "exclude_classes_downstream": args.exclude_class_downstream,
            "exclude_runs_downstream": args.exclude_run_downstream,
            "automatic_low_yield_excluded_runs_downstream": validation[
                "runs_below_group_yield_threshold"
            ],
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
            "analysis_selection": correction_payload["analysis_selection"],
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
        validation["yield_mode"],
        run_group=validation["dataset"].get("run_group"),
        gemc_points=gemc_points,
        gemc_fit=gemc_fit,
        correction=correction,
    )

    print(f"Candidate events: {validation['candidate_events']}")
    print(f"Signal yield: {validation['signal_events']:.8g}")
    if args.background_cuts is not None:
        print(
            "Sideband subtraction: "
            f"signal-region={validation['signal_region_events']}, "
            f"sideband={validation['sideband_events']}, "
            f"background={validation['estimated_background_events']:.8g}"
        )
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
        "Runs below the group mean yield threshold: "
        f"{len(validation['runs_below_group_yield_threshold'])}"
    )
    if validation["runs_below_group_yield_threshold"]:
        print(
            "Low-yield run exclusions: "
            + ", ".join(
                str(run) for run in validation["runs_below_group_yield_threshold"]
            )
        )
    if isinstance(fit, SharedFractionalYieldFit):
        for period, intercept in fit.period_intercepts_events_per_nC.items():
            print(
                f"Zero-current yield ({period}): {intercept:.8g} +/- "
                f"{fit.period_intercept_uncertainties_events_per_nC[period]:.8g} "
                "events/nC"
            )
        print(
            "Shared fractional current slope: "
            f"{fit.fractional_slope_per_nA:.8g} +/- "
            f"{fit.fractional_slope_uncertainty_per_nA:.8g} nA^-1"
        )
    else:
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
    if args.selection_mask is None and args.background_cuts is None:
        warnings.append(
            "No fixed signal-selection mask or background-cut table was supplied; "
            "yields count all selected candidates and are not background-subtracted "
            "signal yields."
        )
    if args.background_cuts is not None:
        warnings.append(
            "The sideband transfer factors are common to all runs. Their bootstrap "
            "uncertainties are recorded as correlated systematic information and are not "
            "added independently to the current-fit point uncertainties."
        )
    if "L5" in args.include_classes:
        warnings.append(
            "L5 is included; confirm its physics trigger and prescale are compatible with P3/P4."
        )
    if any(label != "unflagged" for label in args.include_qualities):
        warnings.append("Flagged RCDB current values are included.")
    if fit.ndf <= 0:
        warnings.append(
            "The current fit has no residual degrees of freedom; it cannot test the "
            "assumed current dependence."
        )
    if isinstance(fit, SharedFractionalYieldFit):
        intercepts = list(fit.period_intercepts_events_per_nC.values())
        slope = fit.fractional_slope_per_nA
    else:
        intercepts = [fit.intercept_events_per_nC]
        slope = fit.slope_events_per_nC_per_nA
    if any(intercept <= 0.0 for intercept in intercepts):
        warnings.append(
            "At least one fitted zero-current yield is nonpositive; relative efficiencies "
            "are undefined."
        )
    if slope > 0.0:
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
    if validation.get("runs_below_group_yield_threshold"):
        warnings.append(
            "Runs more than the configured number of statistical standard deviations "
            "below their leave-one-out group mean were excluded from the current fit and "
            "will receive zero downstream weight in the correction artifact."
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


def _intercept_for_class(
    fit: LinearYieldFit | SharedFractionalYieldFit, run_class: str
) -> float:
    if isinstance(fit, SharedFractionalYieldFit):
        return fit.intercept_for_class(run_class)
    return fit.intercept_events_per_nC


def _predict_groups(
    fit: LinearYieldFit | SharedFractionalYieldFit, groups
) -> np.ndarray:
    if isinstance(fit, SharedFractionalYieldFit):
        return np.asarray(
            [
                fit.predict(
                    group.effective_current_nA,
                    period=fit.period_for_class(group.group),
                )
                for group in groups
            ],
            dtype=float,
        )
    return fit.predict(
        np.asarray([group.effective_current_nA for group in groups], dtype=float)
    )


def write_plots(
    path,
    records,
    groups,
    fit,
    yield_mode: str,
    *,
    run_group=None,
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
    selection_label = {
        "fixed_selection_mask": "fixed-mask signal",
        "mgg_sideband_subtracted": r"$m_{\gamma\gamma}$ sideband-subtracted signal",
        "all_selected_candidates": "all selected candidates",
    }.get(yield_mode, yield_mode.replace("_", " "))
    show_relative = gemc_points is not None and gemc_fit is not None
    if isinstance(fit, SharedFractionalYieldFit):
        data_intercepts = list(fit.period_intercepts_events_per_nC.values())
    else:
        data_intercepts = [fit.intercept_events_per_nC]
    if show_relative and any(value <= 0.0 for value in data_intercepts):
        raise ValueError("data zero-current intercept must be positive for a GEMC overlay")
    if show_relative and correction is None:
        raise ValueError("a current-efficiency correction model is required for a GEMC overlay")
    study_prefix = f"{str(run_group).strip()} " if run_group else ""

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
            if isinstance(fit, SharedFractionalYieldFit) and name not in fit.class_periods:
                continue
            data_scale = 1.0 / _intercept_for_class(fit, name) if show_relative else 1.0
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
            data_scale = 1.0 / _intercept_for_class(fit, name) if show_relative else 1.0
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
        group_scales = np.asarray(
            [
                1.0 / _intercept_for_class(fit, group.group)
                if show_relative
                else 1.0
                for group in groups
            ],
            dtype=float,
        )
        axis.errorbar(
            [group.effective_current_nA for group in groups],
            np.asarray([group.yield_events_per_nC for group in groups]) * group_scales,
            yerr=np.asarray(
                [group.statistical_uncertainty_events_per_nC for group in groups]
            )
            * group_scales,
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
            axis.plot(
                fit_current,
                data_fit_curve,
                color="black",
                label=(
                    "data shared fractional-slope fit"
                    if isinstance(fit, SharedFractionalYieldFit)
                    else "data linear fit"
                ),
                zorder=3,
            )
            axis.fill_between(
                fit_current,
                data_fit_curve - data_fit_uncertainty,
                data_fit_curve + data_fit_uncertainty,
                color="0.35",
                alpha=0.18,
                linewidth=0.0,
                label="data fit uncertainty",
                zorder=2,
            )
            axis.scatter(
                [0.0],
                [1.0],
                marker="*",
                s=110,
                color="black",
                label="zero-current normalization",
                zorder=5,
            )
        elif isinstance(fit, SharedFractionalYieldFit):
            period_colors = {
                period: plt.get_cmap("Dark2")(index % 8)
                for index, period in enumerate(fit.period_intercepts_events_per_nC)
            }
            for period, intercept in fit.period_intercepts_events_per_nC.items():
                curve = fit.predict(fit_current, period=period)
                uncertainty = fit.prediction_uncertainty(fit_current, period=period)
                color = period_colors[period]
                axis.plot(
                    fit_current,
                    curve,
                    color=color,
                    label=f"{period} shared-slope fit",
                    zorder=3,
                )
                axis.fill_between(
                    fit_current,
                    curve - uncertainty,
                    curve + uncertainty,
                    color=color,
                    alpha=0.14,
                    linewidth=0.0,
                    zorder=2,
                )
                axis.scatter(
                    [0.0],
                    [intercept],
                    marker="*",
                    s=90,
                    color=color,
                    zorder=5,
                )
        else:
            data_fit_curve = fit.predict(fit_current)
            data_fit_uncertainty = _linear_prediction_uncertainty(
                fit_current, fit.covariance
            )
            axis.plot(
                fit_current,
                data_fit_curve,
                color="black",
                label="linear fit",
                zorder=3,
            )
            axis.fill_between(
                fit_current,
                data_fit_curve - data_fit_uncertainty,
                data_fit_curve + data_fit_uncertainty,
                color="0.35",
                alpha=0.18,
                linewidth=0.0,
                label="fit uncertainty",
                zorder=2,
            )
            axis.scatter(
                [0.0],
                [fit.intercept_events_per_nC],
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
            axis.set_title(
                f"{study_prefix}data/GEMC current study: {selection_label}"
            )
        else:
            axis.set_ylabel("Yield (events/nC)")
            axis.set_title(f"{study_prefix}data current study: {selection_label}")
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
            ) * group_scales
            group_data_sigma = np.asarray(
                [group.statistical_uncertainty_events_per_nC for group in groups],
                dtype=float,
            ) * group_scales
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
            ) - _predict_groups(fit, groups)
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
            ) - _predict_groups(fit, groups)
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
