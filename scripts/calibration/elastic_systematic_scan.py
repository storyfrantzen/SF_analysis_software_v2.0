from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .elastic_momentum import (
    ElasticFitConfig,
    evaluate_region,
    region_support_mask,
)
from .elastic_run_validation import (
    MODEL_COMPLEXITY,
    MODEL_ORDERS,
    filter_arrays_by_run_classes,
    load_candidate_inputs,
    load_run_catalog,
    run_validation,
    subset_arrays,
)
from .plot_utils import save_plot


DOMAIN_CHANGING_AXES = {"thetaMinDeg"}


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


def _model_region_map(
    report: dict[str, object], model: str,
) -> dict[tuple[int, int, int], dict[str, object]]:
    return {
        _region_key(region): region
        for region in report.get("regionModelComparison", {}).get(model, [])
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _load_model_parameters(
    variation_dir: Path,
    report: dict[str, object],
    models: Iterable[str],
) -> dict[str, dict[str, object]]:
    parameters: dict[str, dict[str, object]] = {}
    model_reports = report.get("models", {})
    for model in models:
        model_report = model_reports.get(model, {})
        relative = model_report.get("parameterFile")
        path = (
            variation_dir / str(relative)
            if relative else variation_dir / model / "pooled_parameters.json"
        )
        if path.is_file():
            parameters[model] = _read_json(path)
    return parameters


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


def _fixed_model_stability(
    *,
    records: list[dict[str, object]],
    variations: list[ScanVariation],
    keys: list[tuple[int, int, int]],
    models: list[str],
) -> list[dict[str, object]]:
    nominal_record = next(
        record for record in records if record["variation"].name == "nominal"
    )
    nominal_parameters_by_model = nominal_record.get("modelParameters", {})
    comparisons: dict[
        str,
        dict[str, dict[tuple[int, int, int], dict[str, object]]],
    ] = {}
    for record in records:
        if record["status"] != "completed":
            continue
        variation: ScanVariation = record["variation"]
        by_model: dict[str, dict[tuple[int, int, int], dict[str, object]]] = {}
        for model in models:
            nominal = nominal_parameters_by_model.get(model)
            varied = record.get("modelParameters", {}).get(model)
            if nominal is None or varied is None:
                continue
            by_model[model] = {
                _region_key(comparison): comparison
                for comparison in compare_parameter_surfaces(nominal, varied)
            }
        comparisons[variation.name] = by_model

    same_domain_names = {
        variation.name for variation in variations
        if variation.axis not in DOMAIN_CHANGING_AXES
    }
    stability: list[dict[str, object]] = []
    for key in keys:
        model_entries: list[dict[str, object]] = []
        for model in models:
            assignments: list[dict[str, object]] = []
            nominal_summary = _model_region_map(
                nominal_record["report"], model
            ).get(key, {})
            nominal_rms = nominal_summary.get("medianCellCenterRmsAfter")
            for record in records:
                variation: ScanVariation = record["variation"]
                if record["status"] != "completed":
                    assignments.append({
                        "variation": variation.name,
                        "status": "failed",
                        "eligible": False,
                    })
                    continue
                report = record["report"]
                recommendation = _recommendation_map(report).get(key, {})
                summary = _model_region_map(report, model).get(key, {})
                comparison = (
                    comparisons.get(variation.name, {})
                    .get(model, {})
                    .get(key)
                )
                ratio = None
                if (
                    comparison is not None
                    and nominal_rms is not None
                    and float(nominal_rms) > 0.0
                ):
                    ratio = (
                        float(comparison["rmsDifference"])
                        / float(nominal_rms)
                    )
                assignments.append({
                    "variation": variation.name,
                    "status": "completed",
                    "eligible": model in recommendation.get("eligibleModels", []),
                    "automaticallySelected": recommendation.get("model") == model,
                    "medianHeldOutCellRms": summary.get(
                        "medianCellCenterRmsAfter"
                    ),
                    "heldoutFolds": summary.get("heldoutFolds"),
                    "foldSurfaceAgreement": summary.get("foldSurfaceAgreement"),
                    "surfaceComparisonToNominalSameModel": comparison,
                    "surfaceRmsToNominalHeldOutRmsRatio": ratio,
                })
            completed = [
                entry for entry in assignments
                if entry["status"] == "completed"
            ]
            same_domain = [
                entry for entry in completed
                if entry["variation"] in same_domain_names
            ]

            def _surface_values(
                entries: list[dict[str, object]],
            ) -> list[float]:
                return [
                    float(entry["surfaceComparisonToNominalSameModel"][
                        "rmsDifference"
                    ])
                    for entry in entries
                    if entry.get("surfaceComparisonToNominalSameModel") is not None
                ]

            same_domain_surface = _surface_values(same_domain)
            all_surface = _surface_values(completed)
            model_entries.append({
                "model": model,
                "nominalHeldOutCellRms": nominal_rms,
                "completeEveryVariation": len(completed) == len(variations),
                "eligibleEverySameDomainVariation": (
                    len(same_domain) == len(same_domain_names)
                    and bool(same_domain)
                    and all(entry["eligible"] for entry in same_domain)
                ),
                "eligibleEveryVariation": (
                    len(completed) == len(variations)
                    and bool(completed)
                    and all(entry["eligible"] for entry in completed)
                ),
                "maxSameDomainSurfaceRmsDifference": (
                    max(same_domain_surface) if same_domain_surface else None
                ),
                "maxSurfaceRmsDifference": (
                    max(all_surface) if all_surface else None
                ),
                "assignments": assignments,
            })
        stability.append({
            "pid": key[0],
            "detector": key[1],
            "sector": key[2],
            "region": _region_label(key),
            "models": model_entries,
        })
    return stability


def _conservative_recommendation(
    *,
    region_stability: list[dict[str, object]],
    fixed_model_stability: list[dict[str, object]],
    variations: list[ScanVariation],
) -> dict[str, object]:
    variation_by_name = {variation.name: variation for variation in variations}
    fixed_by_key = {
        _region_key(region): region for region in fixed_model_stability
    }
    regions: list[dict[str, object]] = []
    for region in region_stability:
        key = _region_key(region)
        completed = [
            assignment for assignment in region["assignments"]
            if assignment["status"] == "completed"
            and assignment.get("model") is not None
        ]
        same_domain = [
            assignment for assignment in completed
            if variation_by_name[assignment["variation"]].axis
            not in DOMAIN_CHANGING_AXES
        ]
        same_domain_complete = len(same_domain) == sum(
            variation.axis not in DOMAIN_CHANGING_AXES
            for variation in variations
        )
        all_complete = len(completed) == len(variations)
        same_models = sorted(
            {str(entry["model"]) for entry in same_domain},
            key=lambda model: MODEL_COMPLEXITY[model],
        )
        all_models = sorted(
            {str(entry["model"]) for entry in completed},
            key=lambda model: MODEL_COMPLEXITY[model],
        )
        same_domain_model = (
            min(same_models, key=lambda model: MODEL_COMPLEXITY[model])
            if same_domain_complete and same_models else None
        )
        full_scan_model = (
            min(all_models, key=lambda model: MODEL_COMPLEXITY[model])
            if all_complete and all_models else None
        )
        model_diagnostics = {
            entry["model"]: entry
            for entry in fixed_by_key.get(key, {}).get("models", [])
        }
        diagnostic = model_diagnostics.get(same_domain_model, {})
        domain_sensitive = (
            same_domain_model is not None
            and full_scan_model is not None
            and same_domain_model != full_scan_model
        )
        if same_domain_model is None:
            status = "no complete domain-preserving systematic recommendation"
        elif domain_sensitive:
            status = (
                "candidate is stable under selection/binning variations but "
                "simplifies when the fitted theta domain is restricted"
            )
        else:
            status = "candidate complexity is stable across the full scan"
        regions.append({
            "pid": key[0],
            "detector": key[1],
            "sector": key[2],
            "region": _region_label(key),
            "model": same_domain_model,
            "fullScanConservativeModel": full_scan_model,
            "domainSensitive": domain_sensitive,
            "status": status,
            "modelsObservedSameDomain": same_models,
            "modelsObservedFullScan": all_models,
            "eligibleEverySameDomainVariation": diagnostic.get(
                "eligibleEverySameDomainVariation"
            ),
            "maxSameDomainSurfaceRmsDifference": diagnostic.get(
                "maxSameDomainSurfaceRmsDifference"
            ),
            "maxSurfaceRmsDifference": diagnostic.get(
                "maxSurfaceRmsDifference"
            ),
        })
    complete = bool(regions) and all(region["model"] is not None for region in regions)
    domain_sensitive_regions = [
        region["region"] for region in regions if region["domainSensitive"]
    ]
    selected_models = {region["model"] for region in regions if region["model"]}
    overall_model = (
        None if not complete else
        next(iter(selected_models)) if len(selected_models) == 1 else
        "mixed"
    )
    return {
        "model": overall_model,
        "status": (
            "conservative same-domain candidates available; theta-domain-sensitive "
            "regions require analysis-phase-space review"
            if complete and domain_sensitive_regions else
            "conservative candidates are stable across the full scan"
            if complete else
            "one or more regions have no conservative systematic candidate"
        ),
        "strategy": (
            "for each region, take the least complex automatically selected model "
            "across nominal, missing-energy, and cell-binning variations"
        ),
        "domainChangingAxesExcludedFromModelChoice": sorted(DOMAIN_CHANGING_AXES),
        "domainSensitiveRegions": domain_sensitive_regions,
        "phaseSpaceReviewRequired": bool(domain_sensitive_regions),
        "parameterFile": None,
        "regions": regions,
    }


def _build_robust_parameters(
    nominal_parameters_by_model: dict[str, dict[str, object]],
    recommendation: dict[str, object],
    dataset_tag: str,
) -> dict[str, object] | None:
    assignments = recommendation.get("regions", [])
    if not assignments or any(entry.get("model") is None for entry in assignments):
        return None
    first_model = str(assignments[0]["model"])
    if first_model not in nominal_parameters_by_model:
        return None
    robust = copy.deepcopy(nominal_parameters_by_model[first_model])
    robust["datasetTag"] = f"{dataset_tag}_recommended_robust"
    robust["calibrationRole"] = (
        "pooledRobustCandidatePendingPhaseSpaceReview"
        if recommendation.get("phaseSpaceReviewRequired") else
        "pooledRobustCandidateAfterSystematicScan"
    )
    robust["regions"] = []
    robust["skippedRegions"] = []
    fit_configuration = copy.deepcopy(robust.get("fitConfiguration", {}))
    for field in ("thetaOrder", "fdPhiOrder", "cdFourierHarmonics"):
        fit_configuration.pop(field, None)
    fit_configuration["regionSpecificModelOrders"] = True
    robust["fitConfiguration"] = fit_configuration
    robust["modelSelection"] = {
        "strategy": recommendation["strategy"],
        "systematicScanCompleted": True,
        "phaseSpaceReviewRequired": recommendation["phaseSpaceReviewRequired"],
        "domainChangingAxesExcludedFromModelChoice": recommendation[
            "domainChangingAxesExcludedFromModelChoice"
        ],
        "assignments": [
            {
                "pid": entry["pid"],
                "detector": entry["detector"],
                "sector": entry["sector"],
                "model": entry["model"],
                "domainSensitive": entry["domainSensitive"],
            }
            for entry in assignments
        ],
    }
    region_maps = {
        model: _region_map(parameters)
        for model, parameters in nominal_parameters_by_model.items()
    }
    for entry in assignments:
        key = _region_key(entry)
        model = str(entry["model"])
        if model not in region_maps or key not in region_maps[model]:
            return None
        region = copy.deepcopy(region_maps[model][key])
        region["validationModel"] = model
        region["systematicDomainSensitive"] = bool(entry["domainSensitive"])
        robust["regions"].append(region)
    robust["regions"].sort(key=_region_key)
    return robust


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
            "fixedModelParameterFiles": {
                model: f"{variation.name}/{model}/pooled_parameters.json"
                for model in record.get("modelParameters", {})
            },
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

    fixed_model_stability = _fixed_model_stability(
        records=records,
        variations=variations,
        keys=keys,
        models=models,
    )
    conservative_recommendation = _conservative_recommendation(
        region_stability=region_stability,
        fixed_model_stability=fixed_model_stability,
        variations=variations,
    )

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
        "schema": "elastic_momentum_systematic_scan/v2",
        "datasetTag": dataset_tag,
        "particle": particle,
        "beamEnergyGeV": beam_energy,
        "torus": torus,
        "design": "one-at-a-time around nominal",
        "domainChangingAxes": sorted(DOMAIN_CHANGING_AXES),
        "models": models,
        "nominalVariation": "nominal",
        "nominalParameterFile": "nominal/recommended_mixed_parameters.json",
        "runSelection": nominal_report.get("runSelection"),
        "runBlockDefinition": nominal_report.get("runBlockDefinition"),
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
        "fixedModelStability": fixed_model_stability,
        "conservativeRecommendation": conservative_recommendation,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


def _tsv_value(value: object) -> object:
    return "" if value is None else value


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
                    _tsv_value(assignment.get("medianHeldOutCellRms")),
                    _tsv_value(comparison.get("rmsDifference")),
                    _tsv_value(comparison.get("commonCells")),
                    _tsv_value(comparison.get("commonCellFraction")),
                ])


def _write_fixed_model_tsv(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    variation_by_name = {
        variation["name"]: variation for variation in report["variations"]
    }
    with path.open("w", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "variation", "axis", "value", "region", "model", "eligible",
            "automaticallySelected", "heldoutCellRms",
            "sameModelSurfaceRmsFromNominal", "surfaceRmsToHeldoutRms",
            "commonCells", "commonCellFraction",
        ])
        for region in report["fixedModelStability"]:
            for model in region["models"]:
                for assignment in model["assignments"]:
                    variation = variation_by_name[assignment["variation"]]
                    comparison = assignment.get(
                        "surfaceComparisonToNominalSameModel"
                    ) or {}
                    writer.writerow([
                        assignment["variation"],
                        variation["axis"],
                        "" if variation["value"] is None else variation["value"],
                        region["region"],
                        model["model"],
                        assignment.get("eligible", ""),
                        assignment.get("automaticallySelected", ""),
                        _tsv_value(assignment.get("medianHeldOutCellRms")),
                        _tsv_value(comparison.get("rmsDifference")),
                        _tsv_value(
                            assignment.get(
                                "surfaceRmsToNominalHeldOutRmsRatio"
                            )
                        ),
                        _tsv_value(comparison.get("commonCells")),
                        _tsv_value(comparison.get("commonCellFraction")),
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


def _plot_fixed_model_stability(
    report: dict[str, object], output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    successful = [
        variation for variation in report["variations"]
        if variation["status"] == "completed"
    ]
    regions = report["fixedModelStability"]
    models = list(report["models"])
    if not successful or not regions or not models:
        return
    variation_names = [variation["name"] for variation in successful]
    variation_labels: list[str] = []
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
        f"s{region['sector']}" if region["sector"] else region["region"]
        for region in regions
    ]
    shape = (len(successful), len(regions))
    values_by_model: dict[str, np.ndarray] = {}
    eligible_by_model: dict[str, np.ndarray] = {}
    for model in models:
        values = np.full(shape, np.nan)
        eligible = np.zeros(shape, dtype=bool)
        for column, region in enumerate(regions):
            diagnostics = next(
                (entry for entry in region["models"] if entry["model"] == model),
                None,
            )
            if diagnostics is None:
                continue
            by_variation = {
                assignment["variation"]: assignment
                for assignment in diagnostics["assignments"]
            }
            for row, name in enumerate(variation_names):
                assignment = by_variation.get(name, {})
                comparison = assignment.get(
                    "surfaceComparisonToNominalSameModel"
                )
                if comparison is not None:
                    values[row, column] = 100.0 * float(
                        comparison["rmsDifference"]
                    )
                eligible[row, column] = bool(assignment.get("eligible", False))
        values_by_model[model] = values
        eligible_by_model[model] = eligible

    finite_parts = [
        values[np.isfinite(values)] for values in values_by_model.values()
        if np.any(np.isfinite(values))
    ]
    finite = np.concatenate(finite_parts) if finite_parts else np.asarray([])
    vmax = float(np.max(finite)) if finite.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    width = max(5.0 * len(models), 8.0)
    height = max(5.2, 0.58 * len(successful) + 2.7)
    fig, axes_value = plt.subplots(
        1, len(models), figsize=(width, height), squeeze=False,
    )
    axes = list(axes_value[0])
    image = None
    for model_index, (axis, model) in enumerate(zip(axes, models)):
        values = values_by_model[model]
        eligible = eligible_by_model[model]
        image = axis.imshow(
            values, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax,
        )
        axis.set_title(model)
        axis.set_xticks(
            np.arange(len(regions)), labels=region_labels,
            rotation=35, ha="right",
        )
        axis.set_yticks(np.arange(len(successful)))
        axis.set_yticklabels(variation_labels if model_index == 0 else [])
        for row in range(shape[0]):
            for column in range(shape[1]):
                if not np.isfinite(values[row, column]):
                    continue
                suffix = "*" if not eligible[row, column] else ""
                color = "white" if values[row, column] < 0.45 * vmax else "black"
                axis.text(
                    column, row, f"{values[row, column]:.3f}{suffix}",
                    ha="center", va="center", fontsize=7, color=color,
                )
        if image is not None:
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            if model_index == len(models) - 1:
                colorbar.set_label(
                    "same-model surface RMS shift from nominal [%]"
                )
    save_plot(
        fig, output_path,
        f"{report['particle']} fixed-model systematic stability",
        str(report["datasetTag"]), float(report["beamEnergyGeV"]),
        tight_layout=True,
    )
    plt.close(fig)


def _finalize_systematic_scan(
    *,
    records: list[dict[str, object]],
    variations: list[ScanVariation],
    nominal_parameters: dict[str, object],
    particle: str,
    dataset_tag: str,
    beam_energy: float,
    torus: int,
    models: list[str],
    output_dir: Path,
    make_summary_plots: bool,
) -> dict[str, object]:
    report = build_systematic_report(
        records=records,
        variations=variations,
        nominal_parameters=nominal_parameters,
        particle=particle,
        dataset_tag=dataset_tag,
        beam_energy=beam_energy,
        torus=torus,
        models=models,
    )
    nominal_record = next(
        record for record in records if record["variation"].name == "nominal"
    )
    robust_parameters = _build_robust_parameters(
        nominal_record.get("modelParameters", {}),
        report["conservativeRecommendation"],
        dataset_tag,
    )
    if robust_parameters is not None:
        filename = "recommended_robust_parameters.json"
        _write_json(output_dir / filename, robust_parameters)
        report["conservativeRecommendation"]["parameterFile"] = filename
    _write_json(output_dir / "systematic_scan_report.json", report)
    _write_tsv(output_dir / "systematic_scan_summary.tsv", report)
    _write_fixed_model_tsv(
        output_dir / "fixed_model_systematics.tsv", report
    )
    if make_summary_plots:
        _plot_scan_summary(report, output_dir / "systematic_scan_summary.png")
        _plot_fixed_model_stability(
            report, output_dir / "fixed_model_surface_stability.png"
        )
    return report


def rebuild_existing_systematic_scan(
    *,
    output_dir: Path,
    variations: list[ScanVariation],
    particle: str,
    dataset_tag: str,
    beam_energy: float,
    torus: int,
    models: list[str],
    make_summary_plots: bool = True,
) -> dict[str, object]:
    if not variations or variations[0].name != "nominal":
        raise ValueError("systematic scan must begin with the nominal variation")
    records: list[dict[str, object]] = []
    nominal_parameters: dict[str, object] | None = None
    for variation in variations:
        variation_dir = output_dir / variation.name
        report_path = variation_dir / "run_validation_report.json"
        if not report_path.is_file():
            if variation.name == "nominal":
                raise ValueError(
                    f"nominal validation report is unavailable: {report_path}"
                )
            records.append({
                "variation": variation,
                "status": "failed",
                "reason": f"existing validation report is unavailable: {report_path}",
            })
            continue
        report = _read_json(report_path)
        parameter_file = report.get("recommendation", {}).get("parameterFile")
        parameters = None
        if parameter_file:
            path = variation_dir / str(parameter_file)
            if path.is_file():
                parameters = _read_json(path)
        if variation.name == "nominal":
            if parameters is None:
                raise ValueError(
                    "nominal variation has no recommended mixed parameter file"
                )
            nominal_parameters = parameters
        records.append({
            "variation": variation,
            "status": "completed",
            "report": report,
            "parameters": parameters,
            "modelParameters": _load_model_parameters(
                variation_dir, report, models
            ),
        })
    if nominal_parameters is None:
        raise ValueError("nominal systematic-scan parameters are unavailable")
    return _finalize_systematic_scan(
        records=records,
        variations=variations,
        nominal_parameters=nominal_parameters,
        particle=particle,
        dataset_tag=dataset_tag,
        beam_energy=beam_energy,
        torus=torus,
        models=models,
        output_dir=output_dir,
        make_summary_plots=make_summary_plots,
    )


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
    run_class_by_run: dict[int, str] | None = None,
    run_block_mode: str = "candidate-target",
    run_class_order: Iterable[str] = (),
    run_selection: dict[str, object] | None = None,
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
                run_class_by_run=run_class_by_run,
                run_block_mode=run_block_mode,
                run_class_order=run_class_order,
                run_selection=run_selection,
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
            "modelParameters": _load_model_parameters(
                variation_dir, result, models
            ),
        })
        print(
            f"[SCAN] {variation.name} completed: "
            f"selected={result['selection']['selectedCandidates']}; "
            f"recommendation={result['recommendation'].get('model') or 'none'}",
            flush=True,
        )

    if nominal_parameters is None:
        raise ValueError("nominal systematic-scan parameters are unavailable")
    return _finalize_systematic_scan(
        records=records,
        variations=variations,
        nominal_parameters=nominal_parameters,
        particle=particle,
        dataset_tag=dataset_tag,
        beam_energy=cfg.beam_energy,
        torus=cfg.torus,
        models=models,
        output_dir=output_dir,
        make_summary_plots=make_summary_plots,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-at-a-time elastic momentum selection and binning variations, "
            "using a common nominal run partition and one aggregate report."
        )
    )
    parser.add_argument("input_files", nargs="*", type=Path)
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
    parser.add_argument(
        "--run-catalog", type=Path,
        help="JSON run catalog with a top-level runs object and run_class fields",
    )
    parser.add_argument(
        "--include-run-classes", nargs="+",
        help="retain only these run classes; requires --run-catalog",
    )
    block_group = parser.add_mutually_exclusive_group()
    block_group.add_argument(
        "--block-by-run-class", action="store_true",
        help=(
            "form candidate-target blocks within, never across, contiguous "
            "run-class periods"
        ),
    )
    block_group.add_argument(
        "--fold-by-run-class", action="store_true",
        help=(
            "with exactly two included classes, train on each class and hold out "
            "the other"
        ),
    )
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
    parser.add_argument(
        "--rebuild-existing-report", action="store_true",
        help=(
            "rebuild aggregate diagnostics from existing variation reports and "
            "pooled parameter files without reading candidate ROOT files"
        ),
    )
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
    if args.include_run_classes and args.run_catalog is None:
        raise ValueError("--include-run-classes requires --run-catalog")
    if (args.block_by_run_class or args.fold_by_run_class) and args.run_catalog is None:
        raise ValueError("run-class block modes require --run-catalog")
    if args.fold_by_run_class and len(args.include_run_classes or ()) != 2:
        raise ValueError(
            "--fold-by-run-class requires exactly two --include-run-classes"
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
    if args.rebuild_existing_report:
        report = rebuild_existing_systematic_scan(
            output_dir=args.output_dir,
            variations=variations,
            particle=args.particle,
            dataset_tag=args.dataset_tag,
            beam_energy=args.beam_energy,
            torus=args.torus,
            models=args.models,
            make_summary_plots=not args.no_summary_plots,
        )
        print(f"Rebuilt systematic report in {args.output_dir}")
    else:
        if not args.input_files:
            raise ValueError(
                "at least one candidate ROOT input is required unless "
                "--rebuild-existing-report is used"
            )
        arrays = load_candidate_inputs(args.input_files, args.tree)
        if args.max_rows is not None:
            arrays = subset_arrays(
                arrays,
                np.arange(next(iter(arrays.values())).size) < args.max_rows,
            )
        run_class_by_run: dict[int, str] | None = None
        run_selection: dict[str, object] | None = None
        if args.run_catalog is not None:
            arrays, run_class_by_run, run_selection = filter_arrays_by_run_classes(
                arrays,
                load_run_catalog(args.run_catalog),
                args.include_run_classes,
                catalog_path=args.run_catalog,
            )
        run_block_mode = (
            "class-fold" if args.fold_by_run_class else
            "class-boundary" if args.block_by_run_class else
            "candidate-target"
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
            run_class_by_run=run_class_by_run,
            run_block_mode=run_block_mode,
            run_class_order=args.include_run_classes or (),
            run_selection=run_selection,
        )
        print(f"Wrote systematic scan to {args.output_dir}")
    print(report["status"])
    for region in report["conservativeRecommendation"]["regions"]:
        observed = ", ".join(region["modelsObservedSameDomain"]) or "none"
        print(
            f"  {region['region']}: conservative={region['model']}; "
            f"same-domain observed={observed}; "
            f"domain-sensitive={region['domainSensitive']}"
        )


if __name__ == "__main__":
    main()
