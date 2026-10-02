from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .elastic_momentum import (
    apply_supported_particle_correction,
    evaluate_region,
    mode_seeded_core,
    region_support_mask,
)
from .elastic_run_validation import (
    filter_arrays_by_run_classes,
    load_run_catalog,
)
from .eppi0_momentum_validation import load_aligned_selection_mask, load_eppi0_arrays
from .eppi0_proton_root_audit import _nearest_root_choice, photon_branch_metrics
from .exclusive_particle_momentum import (
    SURFACE_TERMS,
    ExclusiveFitConfig,
    _base_mask,
    _read_parameters,
    build_eppi0_particle_sample,
    fit_momentum_theta_region,
    proton_momentum_roots_eppi0,
)
from .plot_utils import save_plot


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _run_class_selection_mask(
    run_numbers: np.ndarray,
    run_catalog: Path,
    include_run_classes: list[str] | None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build an aligned mask while reusing the elastic run-catalog checks."""
    runs = np.asarray(run_numbers, dtype=int)
    _, selected_mapping, metadata = filter_arrays_by_run_classes(
        {"runNum": runs.copy()},
        load_run_catalog(run_catalog),
        include_run_classes,
        catalog_path=run_catalog,
    )
    selected_runs = np.asarray(sorted(selected_mapping), dtype=int)
    return np.isin(runs, selected_runs), metadata


def _root_quality_selection_mask(
    arrays: dict[str, np.ndarray],
    electron_parameters: dict[str, object],
    cfg: ExclusiveFitConfig,
    cohort_mask: np.ndarray | None,
    *,
    stability_parameters: dict[str, object] | None,
    photon_direction_min_gap_deg: float | None,
) -> tuple[np.ndarray, dict[str, object]]:
    entries = int(np.asarray(arrays["electronP"]).size)
    cohort = (
        np.ones(entries, dtype=bool)
        if cohort_mask is None else np.asarray(cohort_mask, dtype=bool)
    )
    electron_p, electron_support, _ = apply_supported_particle_correction(
        arrays["electronP"], arrays["electronTheta"], arrays["electronPhi"],
        arrays["electronDet"], arrays["electronSector"],
        pid=11, parameters=electron_parameters,
    )
    physics_base, _ = _base_mask(arrays, electron_p, cfg)
    cohort &= physics_base & electron_support
    roots, valid = proton_momentum_roots_eppi0(
        electron_p, arrays["electronTheta"], arrays["electronPhi"],
        arrays["protonTheta"], arrays["protonPhi"], cfg.beam_energy,
    )
    measured = np.asarray(arrays["protonP"], dtype=float)
    nominal_choice = _nearest_root_choice(roots, valid, measured)
    ambiguous = np.sum(valid, axis=1) == 2
    keep = np.ones(entries, dtype=bool)
    metadata: dict[str, object] = {
        "inputCohortEntries": int(np.count_nonzero(cohort)),
        "ambiguousInputCohortEntries": int(np.count_nonzero(cohort & ambiguous)),
    }
    if stability_parameters is not None:
        proton_sector = np.asarray(
            arrays.get("protonSector", np.zeros(entries)), dtype=int
        )
        corrected, support, _ = apply_supported_particle_correction(
            measured, arrays["protonTheta"], arrays["protonPhi"],
            arrays["protonDet"], proton_sector,
            pid=2212, parameters=stability_parameters,
        )
        corrected_choice = _nearest_root_choice(roots, valid, corrected)
        flips = ambiguous & support & (corrected_choice != nominal_choice)
        keep &= ~(cohort & flips)
        eligible = cohort & ambiguous & support
        metadata["stability"] = {
            "supportedAmbiguousEntries": int(np.count_nonzero(eligible)),
            "excludedBranchFlipEntries": int(np.count_nonzero(cohort & flips)),
            "excludedBranchFlipFraction": (
                float(np.count_nonzero(cohort & flips) / np.count_nonzero(eligible))
                if np.any(eligible) else 0.0
            ),
        }
    if photon_direction_min_gap_deg is not None:
        angles, _ = photon_branch_metrics(
            arrays, electron_p, roots, valid, cfg.beam_energy
        )
        finite = np.all(np.isfinite(angles), axis=1)
        photon_choice = np.argmin(angles, axis=1)
        gap = np.abs(angles[:, 1] - angles[:, 0])
        decisive = ambiguous & finite & (gap >= photon_direction_min_gap_deg)
        disagreements = decisive & (photon_choice != nominal_choice)
        keep &= ~(cohort & disagreements)
        eligible = cohort & decisive
        metadata["photonDirection"] = {
            "minimumGapDeg": photon_direction_min_gap_deg,
            "decisiveEntries": int(np.count_nonzero(eligible)),
            "excludedDisagreementEntries": int(
                np.count_nonzero(cohort & disagreements)
            ),
            "excludedDisagreementFraction": (
                float(
                    np.count_nonzero(cohort & disagreements) /
                    np.count_nonzero(eligible)
                ) if np.any(eligible) else 0.0
            ),
        }
    metadata["retainedCohortEntries"] = int(np.count_nonzero(cohort & keep))
    metadata["excludedCohortEntries"] = int(np.count_nonzero(cohort & ~keep))
    return keep, metadata


def _run_blocks(run_numbers: np.ndarray, target_entries: int) -> list[dict[str, object]]:
    runs, counts = np.unique(np.asarray(run_numbers, dtype=int), return_counts=True)
    if runs.size < 2:
        raise ValueError("held-out validation requires at least two runs")
    blocks: list[dict[str, object]] = []
    current_runs: list[int] = []
    current_entries = 0
    for run, count in zip(runs, counts):
        current_runs.append(int(run))
        current_entries += int(count)
        if current_entries >= target_entries:
            blocks.append({"runs": current_runs, "entries": current_entries})
            current_runs = []
            current_entries = 0
    if current_runs:
        if blocks and len(blocks) > 1 and current_entries < 0.25 * target_entries:
            blocks[-1]["runs"].extend(current_runs)
            blocks[-1]["entries"] += current_entries
        else:
            blocks.append({"runs": current_runs, "entries": current_entries})
    if len(blocks) < 2:
        midpoint = max(1, runs.size // 2)
        groups = (runs[:midpoint], runs[midpoint:])
        blocks = [{
            "runs": [int(run) for run in group],
            "entries": int(np.count_nonzero(np.isin(run_numbers, group))),
        } for group in groups if group.size]
    for index, block in enumerate(blocks):
        block["index"] = index
        block["fold"] = "A" if index % 2 == 0 else "B"
        block["runMin"] = min(block["runs"])
        block["runMax"] = max(block["runs"])
    if {block["fold"] for block in blocks} != {"A", "B"}:
        raise ValueError("could not form two nonempty run folds")
    return blocks


def _cell_validation(
    region: dict[str, object],
    sample: dict[str, np.ndarray],
    rows: np.ndarray,
    cfg: ExclusiveFitConfig,
) -> dict[str, object]:
    p = sample["momentum"][rows]
    theta = sample["thetaDeg"][rows]
    phi = sample["phiDeg"][rows]
    residual = sample["residual"][rows]
    support = region_support_mask(region, theta, phi, p)
    correction = np.zeros(p.shape, dtype=float)
    correction[support] = evaluate_region(
        region, theta[support], phi[support], p[support]
    )
    after = (1.0 + residual) / (1.0 + correction) - 1.0
    minimum = max(30, cfg.min_bin_entries // 4)
    cells: list[dict[str, object]] = []
    for cell in region.get("supportCells", []):
        pr = cell["momentumRangeGeV"]
        tr = cell["thetaRangeDeg"]
        in_cell = support & (
            (p >= float(pr[0])) & (p <= float(pr[1])) &
            (theta >= float(tr[0])) & (theta <= float(tr[1]))
        )
        if np.count_nonzero(in_cell) < minimum:
            continue
        before_core = mode_seeded_core(
            residual[in_cell],
            peak_search_max_abs_residual=cfg.peak_search_max_abs_residual,
            peak_seed_half_width=cfg.peak_seed_half_width,
            sigma_clip=cfg.core_sigma_clip,
        )
        after_core = mode_seeded_core(
            after[in_cell],
            peak_search_max_abs_residual=cfg.peak_search_max_abs_residual,
            peak_seed_half_width=cfg.peak_seed_half_width,
            sigma_clip=cfg.core_sigma_clip,
        )
        if before_core.entries < minimum or after_core.entries < minimum:
            continue
        if not np.isfinite(before_core.center) or not np.isfinite(after_core.center):
            continue
        cells.append({
            "momentumRangeGeV": pr,
            "thetaRangeDeg": tr,
            "entries": int(np.count_nonzero(in_cell)),
            "centerBefore": before_core.center,
            "centerAfter": after_core.center,
        })
    before_centers = np.asarray([cell["centerBefore"] for cell in cells])
    after_centers = np.asarray([cell["centerAfter"] for cell in cells])
    return {
        "holdoutEntries": int(rows.size),
        "supportedEntries": int(np.count_nonzero(support)),
        "validatedCells": len(cells),
        "cellCenterRmsBefore": (
            float(np.sqrt(np.mean(np.square(before_centers)))) if cells else None
        ),
        "cellCenterRmsAfter": (
            float(np.sqrt(np.mean(np.square(after_centers)))) if cells else None
        ),
        "medianAbsCellCenterAfter": (
            float(np.median(np.abs(after_centers))) if cells else None
        ),
        "cells": cells,
    }


def _region_key(detector: int, sector: int) -> str:
    return f"det{detector}" + (f"_sector{sector}" if sector else "")


def run_exclusive_validation(
    sample: dict[str, np.ndarray],
    cfg: ExclusiveFitConfig,
    *,
    particle: str,
    models: list[str],
    block_target: int,
    minimum_improvement_fraction: float,
    dataset_tag: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if not models:
        raise ValueError("at least one model is required")
    blocks = _run_blocks(sample["runNum"], block_target)
    fold_runs = {
        fold: np.asarray([
            run for block in blocks if block["fold"] == fold
            for run in block["runs"]
        ], dtype=int)
        for fold in ("A", "B")
    }
    keys = sorted({
        (int(detector), int(sector))
        for detector, sector in zip(sample["detector"], sample["sector"])
    })
    pid = 2212 if particle == "proton" else 22
    model_results: dict[str, object] = {}
    aggregate: dict[tuple[int, int], dict[str, dict[str, object]]] = {
        key: {} for key in keys
    }
    for model in models:
        model_cfg = replace(cfg, model=model)
        model_regions: list[dict[str, object]] = []
        for detector, sector in keys:
            fold_results: list[dict[str, object]] = []
            for training_fold, holdout_fold in (("A", "B"), ("B", "A")):
                region_rows = (
                    (sample["detector"] == detector) &
                    (sample["sector"] == sector)
                )
                training = region_rows & np.isin(sample["runNum"], fold_runs[training_fold])
                holdout = region_rows & np.isin(sample["runNum"], fold_runs[holdout_fold])
                try:
                    region, _ = fit_momentum_theta_region(
                        pid=pid, particle=particle, detector=detector,
                        sector=sector,
                        momentum=sample["momentum"][training],
                        theta_deg=sample["thetaDeg"][training],
                        phi_deg=sample["phiDeg"][training],
                        residual=sample["residual"][training], cfg=model_cfg,
                    )
                    validation = _cell_validation(
                        region, sample, np.flatnonzero(holdout), model_cfg
                    )
                    status = "complete" if validation["validatedCells"] else "no cells"
                except ValueError as error:
                    validation = {
                        "holdoutEntries": int(np.count_nonzero(holdout)),
                        "supportedEntries": 0, "validatedCells": 0,
                        "cellCenterRmsBefore": None, "cellCenterRmsAfter": None,
                        "medianAbsCellCenterAfter": None, "cells": [],
                    }
                    status = str(error)
                fold_results.append({
                    "trainingFold": training_fold,
                    "holdoutFold": holdout_fold,
                    "status": status,
                    **validation,
                })
            valid = [
                result for result in fold_results
                if result["cellCenterRmsAfter"] is not None
            ]
            summary = {
                "region": _region_key(detector, sector),
                "detector": detector,
                "sector": sector,
                "completeFoldCoverage": len(valid) == 2,
                "validatedCells": int(sum(item["validatedCells"] for item in valid)),
                "medianHeldOutCellRmsBefore": (
                    float(np.median([item["cellCenterRmsBefore"] for item in valid]))
                    if valid else None
                ),
                "medianHeldOutCellRmsAfter": (
                    float(np.median([item["cellCenterRmsAfter"] for item in valid]))
                    if valid else None
                ),
                "folds": fold_results,
            }
            aggregate[(detector, sector)][model] = summary
            model_regions.append(summary)
        model_results[model] = {"regions": model_regions}

    recommendations: list[dict[str, object]] = []
    pooled_regions: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for detector, sector in keys:
        candidates = [
            (model, aggregate[(detector, sector)][model])
            for model in models
            if aggregate[(detector, sector)][model]["completeFoldCoverage"]
        ]
        if not candidates:
            recommendations.append({
                "region": _region_key(detector, sector), "model": None,
                "status": "no model completed both held-out folds",
            })
            continue
        by_complexity: dict[int, tuple[str, dict[str, object]]] = {}
        for model, summary in candidates:
            complexity = len(SURFACE_TERMS[model])
            incumbent = by_complexity.get(complexity)
            if incumbent is None or summary["medianHeldOutCellRmsAfter"] < (
                incumbent[1]["medianHeldOutCellRmsAfter"]
            ):
                by_complexity[complexity] = (model, summary)
        ordered = [by_complexity[key] for key in sorted(by_complexity)]
        chosen_model, chosen_summary = ordered[0]
        for candidate_model, candidate_summary in ordered[1:]:
            required = (
                chosen_summary["medianHeldOutCellRmsAfter"] *
                (1.0 - minimum_improvement_fraction)
            )
            if candidate_summary["medianHeldOutCellRmsAfter"] <= required:
                chosen_model, chosen_summary = candidate_model, candidate_summary
        recommendation = {
            "region": _region_key(detector, sector),
            "detector": detector, "sector": sector,
            "model": chosen_model,
            "medianHeldOutCellRms": chosen_summary["medianHeldOutCellRmsAfter"],
            "status": "diagnostic recommendation pending systematic variations",
        }
        recommendations.append(recommendation)
        rows = (sample["detector"] == detector) & (sample["sector"] == sector)
        try:
            region, _ = fit_momentum_theta_region(
                pid=pid, particle=particle, detector=detector, sector=sector,
                momentum=sample["momentum"][rows],
                theta_deg=sample["thetaDeg"][rows],
                phi_deg=sample["phiDeg"][rows],
                residual=sample["residual"][rows],
                cfg=replace(cfg, model=chosen_model),
            )
            region["heldOutValidation"] = recommendation
            pooled_regions.append(region)
        except ValueError as error:
            skipped.append({"region": recommendation["region"], "reason": str(error)})

    parameters: dict[str, object] = {
        "schema": "particle_momentum_correction/v2",
        "correctionType": "fractionalMomentum",
        "beamEnergyGeV": cfg.beam_energy,
        "torus": cfg.torus,
        "calibrationChannel": "ep-pi0",
        "calibrationRole": "pooledCandidatePendingSystematicReview",
        "particle": particle,
        "datasetTag": dataset_tag,
        "regions": pooled_regions,
        "skippedRegions": skipped,
    }
    report: dict[str, object] = {
        "schema": "eppi0-particle-run-validation/v1",
        "datasetTag": dataset_tag,
        "particle": particle,
        "beamEnergyGeV": cfg.beam_energy,
        "torus": cfg.torus,
        "models": model_results,
        "runBlocks": blocks,
        "minimumModelImprovementFraction": minimum_improvement_fraction,
        "recommendation": {
            "model": "mixed" if pooled_regions else None,
            "regions": recommendations,
            "status": (
                "diagnostic recommendation pending systematic variations"
                if pooled_regions else "no region recommendation"
            ),
            "parameterFile": "recommended_parameters.json" if pooled_regions else None,
        },
    }
    return report, parameters


def _plot_comparison(report: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    models = list(report["models"])
    regions = [item["region"] for item in report["models"][models[0]]["regions"]]
    fig, axes = plt.subplots(1, len(regions), figsize=(4.0 * len(regions), 4.0),
                             squeeze=False)
    for axis, region_name in zip(axes.flat, regions):
        values = []
        for model in models:
            item = next(
                row for row in report["models"][model]["regions"]
                if row["region"] == region_name
            )
            value = item["medianHeldOutCellRmsAfter"]
            values.append(100.0 * value if value is not None else np.nan)
        axis.bar(np.arange(len(models)), values)
        axis.set_xticks(np.arange(len(models)), models, rotation=25, ha="right")
        axis.set_ylabel("held-out cell-center RMS [%]")
        axis.set_title(region_name)
        axis.grid(axis="y", alpha=0.25)
    save_plot(
        fig, output,
        f"{report['particle']} ep-pi0 held-out surface comparison",
        str(report["datasetTag"]), float(report["beamEnergyGeV"]),
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select proton or photon momentum-theta complexity using alternating "
            "held-out run blocks in ep-pi0 candidates."
        )
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--beam-energy", type=float, required=True)
    parser.add_argument("--torus", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--particle", choices=("proton", "photon"), required=True)
    parser.add_argument("--electron-parameters", type=Path, required=True)
    parser.add_argument("--proton-parameters", type=Path)
    parser.add_argument("--selection-mask", type=Path)
    parser.add_argument("--selection-mask-key", default="mask")
    parser.add_argument(
        "--run-catalog", type=Path,
        help="JSON run catalog used to restrict the calibration run classes",
    )
    parser.add_argument(
        "--include-run-classes", nargs="+",
        help="retain only these run classes; requires --run-catalog",
    )
    parser.add_argument(
        "--root-stability-parameters", type=Path,
        help=(
            "exclude events whose nearest proton root changes after applying "
            "this provisional proton correction"
        ),
    )
    parser.add_argument(
        "--exclude-photon-direction-disagreements-min-gap-deg", type=float,
        help=(
            "exclude events where the measured diphoton direction decisively "
            "prefers the alternate root by at least this angular gap"
        ),
    )
    parser.add_argument("--models", nargs="+", choices=tuple(SURFACE_TERMS),
                        default=list(SURFACE_TERMS))
    parser.add_argument("--block-target-selected", type=int, default=250_000)
    parser.add_argument("--minimum-model-improvement-fraction", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--momentum-bins", type=int, default=8)
    parser.add_argument("--theta-bins", type=int, default=8)
    parser.add_argument("--min-bin-entries", type=int, default=200)
    parser.add_argument("--min-region-entries", type=int, default=2_000)
    parser.add_argument("--max-condition-number", type=float, default=100.0)
    parser.add_argument("--max-abs-surface-correction", type=float, default=0.30)
    parser.add_argument("--min-q2", type=float, default=1.0)
    parser.add_argument("--min-w", type=float, default=2.0)
    parser.add_argument("--mgg-max-abs-gev", type=float, default=0.08)
    parser.add_argument("--photon-fit-residual-max-gev", type=float, default=0.50)
    parser.add_argument("--photon-design-condition-max", type=float, default=100.0)
    parser.add_argument("--fd-by-sector", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.particle == "photon" and args.proton_parameters is None:
        raise ValueError("photon validation requires --proton-parameters")
    if args.include_run_classes and args.run_catalog is None:
        raise ValueError("--include-run-classes requires --run-catalog")
    if (
        args.exclude_photon_direction_disagreements_min_gap_deg is not None and
        args.exclude_photon_direction_disagreements_min_gap_deg <= 0.0
    ):
        raise ValueError("photon-direction disagreement gap must be positive")
    arrays = load_eppi0_arrays(args.input_file, args.tree, args.max_rows)
    electron_parameters = _read_parameters(args.electron_parameters)
    mask = None
    if args.selection_mask:
        mask = load_aligned_selection_mask(
            args.selection_mask, int(np.asarray(arrays["electronP"]).size),
            args.selection_mask_key, allow_prefix=args.max_rows is not None,
        )
    run_selection = None
    if args.run_catalog is not None:
        if "runNum" not in arrays:
            raise ValueError("candidate tree requires runNum for run-class filtering")
        run_mask, run_selection = _run_class_selection_mask(
            arrays["runNum"], args.run_catalog, args.include_run_classes,
        )
        mask = run_mask if mask is None else (mask & run_mask)
    cfg = ExclusiveFitConfig(
        beam_energy=args.beam_energy, torus=args.torus,
        momentum_bins=args.momentum_bins, theta_bins=args.theta_bins,
        min_bin_entries=args.min_bin_entries,
        min_region_entries=args.min_region_entries,
        max_condition_number=args.max_condition_number,
        max_abs_surface_correction=args.max_abs_surface_correction,
        minimum_q2=args.min_q2, minimum_w=args.min_w,
        mgg_max_abs_gev=args.mgg_max_abs_gev,
        photon_fit_residual_max_gev=args.photon_fit_residual_max_gev,
        photon_design_condition_max=args.photon_design_condition_max,
        fd_by_sector=args.fd_by_sector,
    )
    root_selection = None
    if (
        args.root_stability_parameters is not None or
        args.exclude_photon_direction_disagreements_min_gap_deg is not None
    ):
        root_mask, root_selection = _root_quality_selection_mask(
            arrays, electron_parameters, cfg, mask,
            stability_parameters=(
                _read_parameters(args.root_stability_parameters)
                if args.root_stability_parameters is not None else None
            ),
            photon_direction_min_gap_deg=(
                args.exclude_photon_direction_disagreements_min_gap_deg
            ),
        )
        mask = root_mask if mask is None else (mask & root_mask)
    sample, selection = build_eppi0_particle_sample(
        arrays, cfg,
        electron_parameters=electron_parameters,
        proton_parameters=(
            _read_parameters(args.proton_parameters)
            if args.proton_parameters else None
        ),
        particle=args.particle,
        external_selection_mask=mask,
    )
    if "runNum" not in arrays:
        raise ValueError("candidate tree requires runNum for held-out validation")
    report, parameters = run_exclusive_validation(
        sample, cfg, particle=args.particle, models=args.models,
        block_target=args.block_target_selected,
        minimum_improvement_fraction=args.minimum_model_improvement_fraction,
        dataset_tag=args.dataset_tag,
    )
    report["selection"] = selection
    report["inputFile"] = str(args.input_file)
    report["electronParameters"] = str(args.electron_parameters)
    report["protonParameters"] = (
        str(args.proton_parameters) if args.proton_parameters else None
    )
    report["selectionMask"] = str(args.selection_mask) if args.selection_mask else None
    if run_selection is not None:
        report["runSelection"] = run_selection
        parameters["runSelection"] = run_selection
    if root_selection is not None:
        report["rootSelection"] = root_selection
        parameters["rootSelection"] = root_selection
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_validation_report.json").write_text(
        json.dumps(_json_safe(report), indent=2) + "\n"
    )
    if parameters["regions"]:
        (args.output_dir / "recommended_parameters.json").write_text(
            json.dumps(_json_safe(parameters), indent=2) + "\n"
        )
    if not args.no_plots:
        _plot_comparison(report, args.output_dir / "model_comparison.png")
    print(f"Wrote ep-pi0 {args.particle} run validation to {args.output_dir}")
    for region in report["recommendation"]["regions"]:
        print(f"  {region['region']}: {region['model'] or 'none'} ({region['status']})")
