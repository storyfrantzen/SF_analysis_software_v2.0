from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .elastic_momentum import (
    ElasticFitConfig,
    evaluate_region,
    load_elastic_arrays,
    region_support_mask,
)
from .elastic_run_validation import (
    MODEL_COMPLEXITY,
    MODEL_ORDERS,
    concatenate_arrays,
    run_validation,
    subset_arrays,
)
from .plot_utils import save_plot


@dataclass(frozen=True)
class ScanVariation:
    name: str
    label: str
    axis: str
    value: float | int | None
    overrides: dict[str, float | int | None]


def _float_slug(value: float) -> str:
    return f"{value:.2f}".replace("-", "m").replace(".", "p")


def make_one_at_a_time_variations(
    cfg: ElasticFitConfig,
    *,
    missing_energy_values: Iterable[float],
    theta_min_values: Iterable[float],
    target_cell_entry_values: Iterable[int],
) -> list[ScanVariation]:
    variations = [
        ScanVariation(
            name="nominal",
            label="nominal",
            axis="nominal",
            value=None,
            overrides={},
        )
    ]
    seen = {"nominal"}
    for value in missing_energy_values:
        value = float(value)
        if np.isclose(value, cfg.missing_energy_max_gev):
            continue
        variation = ScanVariation(
            name=f"missing_energy_{_float_slug(value)}",
            label=f"missing energy < {value:.2f} GeV",
            axis="missingEnergyMaxGeV",
            value=value,
            overrides={"missing_energy_max_gev": value},
        )
        if variation.name not in seen:
            variations.append(variation)
            seen.add(variation.name)
    for value in theta_min_values:
        value = float(value)
        if cfg.theta_min_deg is not None and np.isclose(value, cfg.theta_min_deg):
            continue
        variation = ScanVariation(
            name=f"theta_min_{_float_slug(value)}",
            label=f"theta >= {value:.2f} deg",
            axis="thetaMinDeg",
            value=value,
            overrides={"theta_min_deg": value},
        )
        if variation.name not in seen:
            variations.append(variation)
            seen.add(variation.name)
    for value in target_cell_entry_values:
        value = int(value)
        if value == cfg.target_cell_entries:
            continue
        variation = ScanVariation(
            name=f"target_cell_entries_{value}",
            label=f"target cell entries = {value}",
            axis="targetCellEntries",
            value=value,
            overrides={"target_cell_entries": value},
        )
        if variation.name not in seen:
            variations.append(variation)
            seen.add(variation.name)
    return variations


def _region_key(region: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(region["pid"]),
        int(region["detector"]),
        int(region["sector"]),
    )


def _region_label(key: tuple[int, int, int]) -> str:
    pid, detector, sector = key
    suffix = f"sector{sector}" if sector else "all_phi"
    return f"pid{pid}_det{detector}_{suffix}"


def _region_map(
    parameters: dict[str, object] | None,
) -> dict[tuple[int, int, int], dict[str, object]]:
    if parameters is None:
        return {}
    return {
        _region_key(region): region
        for region in parameters.get("regions", [])
    }


def _recommendation_map(
    report: dict[str, object],
) -> dict[tuple[int, int, int], dict[str, object]]:
    return {
        _region_key(region): region
        for region in report.get("recommendation", {}).get("regions", [])
    }


def compare_parameter_surfaces(
    nominal: dict[str, object],
    variation: dict[str, object],
) -> list[dict[str, object]]:
    nominal_regions = _region_map(nominal)
    variation_regions = _region_map(variation)
    comparisons: list[dict[str, object]] = []
    for key, nominal_region in sorted(nominal_regions.items()):
        if key not in variation_regions:
            continue
        cells = nominal_region.get("fit", {}).get("acceptedProfileCells", [])
        if not cells:
            continue
        theta = np.asarray([cell["thetaMeanDeg"] for cell in cells], dtype=float)
        phi = np.asarray([cell["phiMeanDeg"] for cell in cells], dtype=float)
        varied_region = variation_regions[key]
        common = (
            region_support_mask(nominal_region, theta, phi)
            & region_support_mask(varied_region, theta, phi)
        )
        if not np.any(common):
            continue
        difference = (
            evaluate_region(varied_region, theta[common], phi[common])
            - evaluate_region(nominal_region, theta[common], phi[common])
        )
        comparisons.append({
            "pid": key[0],
            "detector": key[1],
            "sector": key[2],
            "region": _region_label(key),
            "nominalCells": int(theta.size),
            "commonCells": int(np.count_nonzero(common)),
            "commonCellFraction": float(np.count_nonzero(common) / theta.size),
            "meanDifference": float(np.mean(difference)),
            "rmsDifference": float(np.sqrt(np.mean(np.square(difference)))),
            "medianAbsDifference": float(np.median(np.abs(difference))),
            "maxAbsDifference": float(np.max(np.abs(difference))),
        })
    return comparisons


def _compact_recommendation(report: dict[str, object]) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for region in report.get("recommendation", {}).get("regions", []):
        compact.append({
            "pid": region["pid"],
            "detector": region["detector"],
            "sector": region["sector"],
            "region": region["region"],
            "model": region.get("model"),
            "medianHeldOutCellRms": region.get("medianHeldOutCellRms"),
            "heldoutFoldCellRms": region.get("heldoutFoldCellRms"),
            "foldSurfaceAgreement": region.get("foldSurfaceAgreement"),
        })
    return compact


def build_systematic_report(
    *,
    records: list[dict[str, object]],
    variations: list[ScanVariation],
    nominal_parameters: dict[str, object],
    particle: str,
    dataset_tag: str,
    beam_energy: float,
    torus: int,
    models: list[str],
) -> dict[str, object]:
    nominal_record = next(
        record for record in records if record["variation"].name == "nominal"
    )
    nominal_report = nominal_record["report"]
    nominal_recommendations = _recommendation_map(nominal_report)
    keys = sorted(nominal_recommendations)
    variation_summaries: list[dict[str, object]] = []
    comparison_by_variation: dict[
        str, dict[tuple[int, int, int], dict[str, object]]
    ] = {}
    successful_names: set[str] = set()

    for record in records:
        variation: ScanVariation = record["variation"]
        if record["status"] != "completed":
            variation_summaries.append({
                **asdict(variation),
                "status": "failed",
                "reason": record["reason"],
                "outputDirectory": variation.name,
            })
            continue
        report = record["report"]
        parameters = record.get("parameters")
        successful_names.add(variation.name)
        comparisons = (
            compare_parameter_surfaces(nominal_parameters, parameters)
            if parameters is not None else []
        )
        comparison_by_variation[variation.name] = {
            _region_key(comparison): comparison for comparison in comparisons
        }
        recommendation = report["recommendation"]
        variation_summaries.append({
            **asdict(variation),
            "status": "completed",
            "outputDirectory": variation.name,
            "reportFile": f"{variation.name}/run_validation_report.json",
            "parameterFile": (
                f"{variation.name}/{recommendation['parameterFile']}"
                if recommendation.get("parameterFile") else None
            ),
            "selection": report["selection"],
            "runBlockDefinition": report.get("runBlockDefinition"),
            "runBlocks": report["runBlocks"],
            "recommendation": {
                "model": recommendation.get("model"),
                "status": recommendation.get("status"),
                "regions": _compact_recommendation(report),
            },
            "surfaceComparisonsToNominal": comparisons,
        })

    region_stability: list[dict[str, object]] = []
    for key in keys:
        assignments: list[dict[str, object]] = []
        for record in records:
            variation: ScanVariation = record["variation"]
            if record["status"] != "completed":
                assignments.append({
                    "variation": variation.name,
                    "status": "failed",
                    "model": None,
                })
                continue
            recommendation = _recommendation_map(record["report"]).get(key, {})
            comparison = comparison_by_variation.get(variation.name, {}).get(key)
            nominal_rms = nominal_recommendations[key].get("medianHeldOutCellRms")
            surface_ratio = None
            if (
                comparison is not None
                and nominal_rms is not None
                and float(nominal_rms) > 0.0
            ):
                surface_ratio = (
                    float(comparison["rmsDifference"]) / float(nominal_rms)
                )
            assignments.append({
                "variation": variation.name,
                "status": "completed",
                "model": recommendation.get("model"),
                "medianHeldOutCellRms": recommendation.get(
                    "medianHeldOutCellRms"
                ),
                "surfaceComparisonToNominal": comparison,
                "surfaceRmsToNominalHeldOutRmsRatio": surface_ratio,
            })
        completed = [
            entry for entry in assignments if entry["status"] == "completed"
        ]
        assigned_models = {
            entry["model"] for entry in completed if entry["model"] is not None
        }
        surface_rms_values = [
            float(entry["surfaceComparisonToNominal"]["rmsDifference"])
            for entry in completed
            if entry.get("surfaceComparisonToNominal") is not None
        ]
        common_cell_fractions = [
            float(entry["surfaceComparisonToNominal"]["commonCellFraction"])
            for entry in completed
            if entry.get("surfaceComparisonToNominal") is not None
        ]
        region_stability.append({
            "pid": key[0],
            "detector": key[1],
            "sector": key[2],
            "region": _region_label(key),
            "nominalModel": nominal_recommendations[key].get("model"),
            "stableModelAssignment": (
                len(completed) == len(variations)
                and all(entry["model"] is not None for entry in completed)
                and len(assigned_models) == 1
            ),
            "modelsObserved": sorted(
                assigned_models,
                key=lambda model: MODEL_COMPLEXITY.get(str(model), 10_000),
            ),
            "maxSurfaceRmsDifference": (
                max(surface_rms_values) if surface_rms_values else None
            ),
            "medianSurfaceRmsDifference": (
                float(np.median(surface_rms_values))
                if surface_rms_values else None
            ),
            "minimumCommonCellFraction": (
                min(common_cell_fractions) if common_cell_fractions else None
            ),
            "assignments": assignments,
        })

    failed_names = [
        variation.name for variation in variations
        if variation.name not in successful_names
    ]
    all_stable = (
        not failed_names
        and bool(region_stability)
        and all(region["stableModelAssignment"] for region in region_stability)
    )
    return {
        "schema": "elastic_momentum_systematic_scan/v1",
        "datasetTag": dataset_tag,
        "particle": particle,
        "beamEnergyGeV": beam_energy,
        "torus": torus,
        "design": "one-at-a-time around nominal",
        "models": models,
        "nominalVariation": "nominal",
        "nominalParameterFile": "nominal/recommended_mixed_parameters.json",
        "fixedRunPartitionAcrossVariations": True,
        "variationsRequested": len(variations),
        "variationsCompleted": len(successful_names),
        "failedVariations": failed_names,
        "allRegionModelAssignmentsStable": all_stable,
        "status": (
            "all detector-region model assignments are stable across the scan"
            if all_stable else
            "one or more detector-region assignments changed or a variation failed"
        ),
        "variations": variation_summaries,
        "regionStability": region_stability,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


def _write_tsv(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    variation_by_name = {
        variation["name"]: variation for variation in report["variations"]
    }
    with path.open("w", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "variation", "axis", "value", "selectedCandidates", "region",
            "model", "heldoutCellRms", "surfaceRmsFromNominal", "commonCells",
            "commonCellFraction",
        ])
        for region in report["regionStability"]:
            for assignment in region["assignments"]:
                variation = variation_by_name[assignment["variation"]]
                selection = variation.get("selection", {})
                comparison = assignment.get("surfaceComparisonToNominal") or {}
                writer.writerow([
                    assignment["variation"],
                    variation["axis"],
                    "" if variation["value"] is None else variation["value"],
                    selection.get("selectedCandidates", ""),
                    region["region"],
                    assignment.get("model") or "",
                    assignment.get("medianHeldOutCellRms") or "",
                    comparison.get("rmsDifference", ""),
                    comparison.get("commonCells", ""),
                    comparison.get("commonCellFraction", ""),
                ])


def _plot_scan_summary(
    report: dict[str, object], output_path: Path
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    successful = [
        variation for variation in report["variations"]
        if variation["status"] == "completed"
    ]
    regions = report["regionStability"]
    if not successful or not regions:
        return
    variation_names = [variation["name"] for variation in successful]
    variation_labels = []
    for variation in successful:
        if variation["axis"] == "nominal":
            variation_labels.append("nominal")
        elif variation["axis"] == "missingEnergyMaxGeV":
            variation_labels.append(f"E_miss < {float(variation['value']):.2f}")
        elif variation["axis"] == "thetaMinDeg":
            variation_labels.append(f"theta >= {float(variation['value']):.2f}")
        else:
            variation_labels.append(f"cell target {int(variation['value'])}")
    region_labels = [
        f"pid{region['pid']} d{region['detector']} s{region['sector']}"
        for region in regions
    ]
    models = sorted(
        {
            assignment["model"]
            for region in regions for assignment in region["assignments"]
            if assignment["status"] == "completed" and assignment["model"] is not None
        },
        key=lambda model: MODEL_COMPLEXITY.get(str(model), 10_000),
    )
    model_index = {model: index for index, model in enumerate(models)}
    shape = (len(successful), len(regions))
    model_values = np.full(shape, np.nan)
    heldout_values = np.full(shape, np.nan)
    surface_values = np.full(shape, np.nan)
    for column, region in enumerate(regions):
        by_variation = {
            assignment["variation"]: assignment
            for assignment in region["assignments"]
        }
        for row, name in enumerate(variation_names):
            assignment = by_variation[name]
            model = assignment.get("model")
            if model in model_index:
                model_values[row, column] = model_index[model]
            heldout = assignment.get("medianHeldOutCellRms")
            if heldout is not None:
                heldout_values[row, column] = 100.0 * float(heldout)
            comparison = assignment.get("surfaceComparisonToNominal")
            if comparison is not None:
                surface_values[row, column] = 100.0 * float(
                    comparison["rmsDifference"]
                )

    height = max(5.2, 0.58 * len(successful) + 2.6)
    fig, axes = plt.subplots(1, 3, figsize=(18.0, height))
    colors = ["#d9d9d9", "#80b1d3", "#fdb462", "#b3de69", "#bc80bd"]
    cmap = ListedColormap(colors[:max(1, len(models))])
    norm = BoundaryNorm(np.arange(len(models) + 1) - 0.5, cmap.N)
    axes[0].imshow(model_values, aspect="auto", cmap=cmap, norm=norm)
    axes[0].set_title("selected model")

    heldout_image = axes[1].imshow(heldout_values, aspect="auto", cmap="viridis")
    axes[1].set_title("held-out cell RMS [%]")
    fig.colorbar(heldout_image, ax=axes[1], fraction=0.046, pad=0.04)
    surface_image = axes[2].imshow(surface_values, aspect="auto", cmap="magma")
    axes[2].set_title("surface RMS shift from nominal [%]")
    fig.colorbar(surface_image, ax=axes[2], fraction=0.046, pad=0.04)

    for axis_index, axis in enumerate(axes):
        axis.set_xticks(
            np.arange(len(regions)), labels=region_labels,
            rotation=35, ha="right",
        )
        axis.set_yticks(np.arange(len(successful)))
        axis.set_yticklabels(variation_labels if axis_index == 0 else [])
        for row in range(shape[0]):
            for column in range(shape[1]):
                if axis_index == 0 and np.isfinite(model_values[row, column]):
                    text_value = models[int(model_values[row, column])]
                elif axis_index == 1 and np.isfinite(heldout_values[row, column]):
                    text_value = f"{heldout_values[row, column]:.3f}"
                elif axis_index == 2 and np.isfinite(surface_values[row, column]):
                    text_value = f"{surface_values[row, column]:.3f}"
                else:
                    continue
                axis.text(
                    column, row, text_value, ha="center", va="center",
                    fontsize=7, color="black",
                )
    save_plot(
        fig, output_path,
        f"{report['particle']} elastic momentum systematic scan",
        str(report["datasetTag"]), float(report["beamEnergyGeV"]),
        tight_layout=True,
    )
    plt.close(fig)


def run_systematic_scan(
    arrays: dict[str, np.ndarray],
    cfg: ElasticFitConfig,
    *,
    particle: str,
    models: list[str],
    block_target: int,
    output_dir: Path,
    dataset_tag: str,
    variations: list[ScanVariation],
    minimum_per_run_core_entries: int = 200,
    minimum_model_improvement_fraction: float = 0.10,
    make_summary_plots: bool = True,
    make_variation_plots: bool = False,
) -> dict[str, object]:
    if not variations or variations[0].name != "nominal":
        raise ValueError("systematic scan must begin with the nominal variation")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    fixed_run_groups: tuple[tuple[int, ...], ...] | None = None
    nominal_parameters: dict[str, object] | None = None
    for variation in variations:
        variation_cfg = replace(cfg, **variation.overrides)
        variation_dir = output_dir / variation.name
        variation_tag = (
            f"{dataset_tag}_{variation.name}" if dataset_tag else variation.name
        )
        print(f"[SCAN] {variation.name}: {variation.label}", flush=True)
        try:
            result = run_validation(
                arrays,
                variation_cfg,
                particle=particle,
                models=models,
                block_target=block_target,
                output_dir=variation_dir,
                dataset_tag=variation_tag,
                make_plots=make_variation_plots,
                minimum_per_run_core_entries=minimum_per_run_core_entries,
                minimum_model_improvement_fraction=(
                    minimum_model_improvement_fraction
                ),
                run_groups=fixed_run_groups,
            )
        except ValueError as error:
            if variation.name == "nominal":
                raise
            print(f"[SCAN] {variation.name} failed: {error}", flush=True)
            records.append({
                "variation": variation,
                "status": "failed",
                "reason": str(error),
            })
            continue
        if variation.name == "nominal":
            fixed_run_groups = tuple(
                tuple(int(run) for run in block["runs"])
                for block in result["runBlocks"]
            )
        parameter_file = result["recommendation"].get("parameterFile")
        parameters = None
        if parameter_file:
            parameters = json.loads(
                (variation_dir / str(parameter_file)).read_text()
            )
        if variation.name == "nominal":
            if parameters is None:
                raise ValueError(
                    "nominal variation did not produce mixed recommendation parameters"
                )
            nominal_parameters = parameters
        records.append({
            "variation": variation,
            "status": "completed",
            "report": result,
            "parameters": parameters,
        })
        print(
            f"[SCAN] {variation.name} completed: "
            f"selected={result['selection']['selectedCandidates']}; "
            f"recommendation={result['recommendation'].get('model') or 'none'}",
            flush=True,
        )

    if nominal_parameters is None:
        raise ValueError("nominal systematic-scan parameters are unavailable")
    report = build_systematic_report(
        records=records,
        variations=variations,
        nominal_parameters=nominal_parameters,
        particle=particle,
        dataset_tag=dataset_tag,
        beam_energy=cfg.beam_energy,
        torus=cfg.torus,
        models=models,
    )
    _write_json(output_dir / "systematic_scan_report.json", report)
    _write_tsv(output_dir / "systematic_scan_summary.tsv", report)
    if make_summary_plots:
        _plot_scan_summary(report, output_dir / "systematic_scan_summary.png")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-at-a-time elastic momentum selection and binning variations, "
            "using a common nominal run partition and one aggregate report."
        )
    )
    parser.add_argument("input_files", nargs="+", type=Path)
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--beam-energy", type=float, required=True)
    parser.add_argument("--torus", type=int, choices=(-1, 1), required=True)
    parser.add_argument(
        "--particle", choices=("electron", "proton"), default="electron"
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_ORDERS),
        default=["constant", "theta-linear", "theta-phi"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--block-target-selected", type=int, default=100_000)
    parser.add_argument("--min-per-run-core-entries", type=int, default=200)
    parser.add_argument(
        "--minimum-model-improvement-fraction", type=float, default=0.10
    )
    parser.add_argument(
        "--missing-energy-scan-gev", nargs="+", type=float,
        default=[0.50, 0.75, 1.00],
    )
    parser.add_argument(
        "--theta-min-scan-deg", nargs="+", type=float,
        default=[6.10, 6.50],
    )
    parser.add_argument(
        "--target-cell-entries-scan", nargs="+", type=int,
        default=[1_500, 2_000, 3_000],
    )
    parser.add_argument("--variation-plots", action="store_true")
    parser.add_argument("--no-summary-plots", action="store_true")
    parser.add_argument("--coplanarity-max-deg", type=float, default=3.0)
    parser.add_argument("--theta-balance-max-deg", type=float, default=2.0)
    parser.add_argument("--missing-energy-max-gev", type=float, default=0.75)
    parser.add_argument("--theta-min-deg", type=float, default=6.10)
    parser.add_argument("--theta-max-deg", type=float)
    parser.add_argument("--theta-trim-quantile", type=float, default=0.005)
    parser.add_argument("--residual-trim-quantile", type=float, default=0.01)
    parser.add_argument("--theta-bins", type=int, default=10)
    parser.add_argument("--phi-bins", type=int, default=7)
    parser.add_argument(
        "--profile-binning", choices=("fixed", "adaptive"), default="adaptive"
    )
    parser.add_argument("--max-theta-bin-width-deg", type=float, default=0.75)
    parser.add_argument("--target-cell-entries", type=int, default=2_000)
    parser.add_argument("--min-phi-cells-per-theta", type=int, default=1)
    parser.add_argument("--min-bin-entries", type=int, default=300)
    parser.add_argument("--min-region-entries", type=int, default=1_000)
    parser.add_argument("--peak-search-max-abs-residual", type=float, default=0.10)
    parser.add_argument("--peak-seed-half-width", type=float, default=0.03)
    parser.add_argument("--core-sigma-clip", type=float, default=3.0)
    parser.add_argument("--min-core-fraction", type=float, default=0.20)
    parser.add_argument("--min-peak-significance", type=float, default=3.0)
    parser.add_argument("--max-core-width", type=float, default=0.05)
    parser.add_argument(
        "--min-profile-cells-per-parameter", type=float, default=2.0
    )
    parser.add_argument("--max-condition-number", type=float, default=100.0)
    parser.add_argument("--max-abs-surface-correction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beam_energy <= 0.0:
        raise ValueError("beam energy must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("maximum rows must be positive")
    if args.min_per_run_core_entries < 2:
        raise ValueError("minimum per-run core entries must be at least two")
    if not 0.0 <= args.minimum_model_improvement_fraction < 1.0:
        raise ValueError("minimum model improvement fraction must be in [0, 1)")
    parts = [load_elastic_arrays(path, args.tree, None) for path in args.input_files]
    arrays = concatenate_arrays(parts)
    if args.max_rows is not None:
        arrays = subset_arrays(
            arrays,
            np.arange(next(iter(arrays.values())).size) < args.max_rows,
        )
    cfg = ElasticFitConfig(
        beam_energy=args.beam_energy,
        torus=args.torus,
        coplanarity_max_deg=args.coplanarity_max_deg,
        theta_balance_max_deg=args.theta_balance_max_deg,
        missing_energy_max_gev=args.missing_energy_max_gev,
        theta_min_deg=args.theta_min_deg,
        theta_max_deg=args.theta_max_deg,
        theta_trim_quantile=args.theta_trim_quantile,
        residual_trim_quantile=args.residual_trim_quantile,
        theta_bins=args.theta_bins,
        phi_bins=args.phi_bins,
        profile_binning=args.profile_binning,
        max_theta_bin_width_deg=args.max_theta_bin_width_deg,
        target_cell_entries=args.target_cell_entries,
        min_phi_cells_per_theta=args.min_phi_cells_per_theta,
        min_bin_entries=args.min_bin_entries,
        min_region_entries=args.min_region_entries,
        peak_search_max_abs_residual=args.peak_search_max_abs_residual,
        peak_seed_half_width=args.peak_seed_half_width,
        core_sigma_clip=args.core_sigma_clip,
        min_core_fraction=args.min_core_fraction,
        min_peak_significance=args.min_peak_significance,
        max_core_width=args.max_core_width,
        min_profile_cells_per_parameter=args.min_profile_cells_per_parameter,
        max_condition_number=args.max_condition_number,
        max_abs_surface_correction=args.max_abs_surface_correction,
    )
    variations = make_one_at_a_time_variations(
        cfg,
        missing_energy_values=args.missing_energy_scan_gev,
        theta_min_values=args.theta_min_scan_deg,
        target_cell_entry_values=args.target_cell_entries_scan,
    )
    report = run_systematic_scan(
        arrays,
        cfg,
        particle=args.particle,
        models=args.models,
        block_target=args.block_target_selected,
        output_dir=args.output_dir,
        dataset_tag=args.dataset_tag,
        variations=variations,
        minimum_per_run_core_entries=args.min_per_run_core_entries,
        minimum_model_improvement_fraction=(
            args.minimum_model_improvement_fraction
        ),
        make_summary_plots=not args.no_summary_plots,
        make_variation_plots=args.variation_plots,
    )
    print(f"Wrote systematic scan to {args.output_dir}")
    print(report["status"])
    for region in report["regionStability"]:
        observed = ", ".join(region["modelsObserved"]) or "none"
        print(
            f"  {region['region']}: nominal={region['nominalModel']}; "
            f"observed={observed}; stable={region['stableModelAssignment']}"
        )


if __name__ == "__main__":
    main()
