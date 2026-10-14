from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from .elastic_momentum import (
    RAD_TO_DEG,
    evaluate_region,
    region_support_mask,
    sector_local_phi,
)
from .elastic_run_validation import concatenate_arrays, subset_arrays
from .plot_utils import save_plot
from .root_arrays import arrays_from_dataframe, has_column, load_dataframe


REQUIRED_COLUMNS = (
    "electronP",
    "electronTheta",
    "electronPhi",
    "electronDet",
    "electronSector",
)
OPTIONAL_COLUMNS = ("Q2", "W", "passFiducial", "passExclusivity", "runNum")


def _region_key(region: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(region["pid"]),
        int(region["detector"]),
        int(region["sector"]),
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def load_selection_mask(
    path: Path,
    expected_entries: int,
    key: str = "mask",
) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if key in loaded.files:
                values = loaded[key]
            elif len(loaded.files) == 1:
                values = loaded[loaded.files[0]]
            else:
                raise ValueError(
                    f"selection-mask NPZ has no '{key}' array; "
                    f"available arrays: {loaded.files}"
                )
        finally:
            loaded.close()
    else:
        values = loaded
    mask = np.asarray(values)
    if mask.ndim != 1 or mask.size != expected_entries:
        raise ValueError(
            f"selection mask has shape {mask.shape}; "
            f"expected ({expected_entries},)"
        )
    if mask.dtype != np.bool_:
        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError(
                "selection mask must contain only booleans or 0/1 values"
            )
        mask = mask.astype(bool)
    return mask


def summarize_exclusivity_cuts(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as saved:
        def scalar(key: str) -> object | None:
            if key not in saved.files:
                return None
            value = np.asarray(saved[key])
            return value.item() if value.size == 1 else None

        variables = (
            [str(value) for value in np.asarray(saved["variables"]).tolist()]
            if "variables" in saved.files else []
        )
        result: dict[str, object] = {
            "variables": variables,
            "retainedGroups": (
                int(np.asarray(saved["group_ids"]).size)
                if "group_ids" in saved.files else None
            ),
            "populatedGroups": (
                int(np.asarray(saved["populated_group_ids"]).size)
                if "populated_group_ids" in saved.files else None
            ),
            "droppedGroups": (
                int(np.asarray(saved["dropped_group_ids"]).size)
                if "dropped_group_ids" in saved.files else None
            ),
        }
        for source, destination in (
            ("grouping", "grouping"),
            ("estimator", "estimator"),
            ("n_sigma", "nSigma"),
            ("signal_containment", "signalContainment"),
            ("global_mode", "globalMode"),
        ):
            value = scalar(source)
            if value is not None:
                result[destination] = value
    return result


def load_eppi0_electron_arrays(
    input_file: Path,
    tree: str,
    max_rows: int | None,
) -> dict[str, np.ndarray]:
    df = load_dataframe(input_file, tree)
    missing = [column for column in REQUIRED_COLUMNS if not has_column(df, column)]
    if missing:
        raise RuntimeError(
            f"{tree} is missing selected-electron branches: {', '.join(missing)}"
        )
    optional = [column for column in OPTIONAL_COLUMNS if has_column(df, column)]
    return arrays_from_dataframe(
        df, list(REQUIRED_COLUMNS) + optional, max_rows=max_rows
    )


def _analysis_mask(
    arrays: dict[str, np.ndarray],
    *,
    minimum_electron_p: float | None,
    minimum_q2: float | None,
    minimum_w: float | None,
    require_pass_fiducial: bool,
    require_pass_exclusivity: bool,
    external_selection_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, object]]:
    entries = int(np.asarray(arrays["electronP"]).size)
    mask = np.ones(entries, dtype=bool)
    for column in REQUIRED_COLUMNS:
        mask &= np.isfinite(np.asarray(arrays[column], dtype=float))
    mask &= np.asarray(arrays["electronP"], dtype=float) > 0.0
    mask &= np.asarray(arrays["electronDet"], dtype=int) == 1
    sectors = np.asarray(arrays["electronSector"], dtype=int)
    mask &= (sectors >= 1) & (sectors <= 6)
    valid_fd = mask.copy()

    external_selected = entries
    if external_selection_mask is not None:
        external = np.asarray(external_selection_mask)
        if external.ndim != 1 or external.size != entries:
            raise ValueError(
                f"external selection mask has shape {external.shape}; "
                f"expected ({entries},)"
            )
        if external.dtype != np.bool_:
            if not np.all(np.isin(external, [0, 1])):
                raise ValueError(
                    "external selection mask must contain only booleans or 0/1 values"
                )
            external = external.astype(bool)
        external_selected = int(np.count_nonzero(external))
        mask &= external

    def apply_minimum(column: str, threshold: float | None) -> None:
        nonlocal mask
        if threshold is None:
            return
        if column not in arrays:
            raise RuntimeError(
                f"analysis threshold requires branch {column}, but it is absent"
            )
        values = np.asarray(arrays[column], dtype=float)
        mask &= np.isfinite(values) & (values >= threshold)

    apply_minimum("electronP", minimum_electron_p)
    apply_minimum("Q2", minimum_q2)
    apply_minimum("W", minimum_w)

    branch_selection: dict[str, dict[str, object]] = {}

    def apply_required_branch(column: str, requested: bool, option: str) -> None:
        nonlocal mask
        if not requested:
            branch_selection[column] = {
                "requested": False,
                "eligibleEntries": int(np.count_nonzero(mask)),
                "rejectedEntries": 0,
                "noEffect": None,
            }
            return
        if column not in arrays:
            raise RuntimeError(f"{option} needs the {column} branch")
        eligible = int(np.count_nonzero(mask))
        mask &= np.asarray(arrays[column], dtype=int) != 0
        rejected = eligible - int(np.count_nonzero(mask))
        branch_selection[column] = {
            "requested": True,
            "eligibleEntries": eligible,
            "rejectedEntries": rejected,
            "noEffect": rejected == 0,
        }

    apply_required_branch(
        "passFiducial", require_pass_fiducial, "--require-pass-fiducial"
    )
    apply_required_branch(
        "passExclusivity",
        require_pass_exclusivity,
        "--require-pass-exclusivity",
    )

    return mask, {
        "inputEntries": entries,
        "validFDElectronEntries": int(np.count_nonzero(valid_fd)),
        "selectedEntries": int(np.count_nonzero(mask)),
        "selectedFractionOfInput": float(np.mean(mask)) if entries else 0.0,
        "minimumElectronPGeV": minimum_electron_p,
        "minimumQ2GeV2": minimum_q2,
        "minimumWGeV": minimum_w,
        "requirePassFiducial": require_pass_fiducial,
        "requirePassExclusivity": require_pass_exclusivity,
        "externalSelectionMaskProvided": external_selection_mask is not None,
        "externalSelectionMaskEntries": entries,
        "externalSelectionMaskSelectedEntries": external_selected,
        "externalSelectionMaskSelectedFraction": _fraction(
            external_selected, entries
        ),
        "requestedBranchSelections": branch_selection,
    }


def _fraction(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    probabilities = (0.01, 0.05, 0.50, 0.95, 0.99)
    result = np.quantile(values, probabilities)
    return {
        f"p{int(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, result)
    }


def _support_cell_theta_range(region: dict[str, object]) -> list[float]:
    parameter_range = [float(value) for value in region["thetaRangeDeg"]]
    support_cells = list(region.get("supportCells", []))
    if not support_cells:
        return parameter_range
    lower = max(
        parameter_range[0],
        min(float(cell["thetaRangeDeg"][0]) for cell in support_cells),
    )
    upper = min(
        parameter_range[1],
        max(float(cell["thetaRangeDeg"][1]) for cell in support_cells),
    )
    if upper < lower:
        raise ValueError("support cells do not overlap the parameter theta range")
    return [lower, upper]


def analyze_phase_space_coverage(
    arrays: dict[str, np.ndarray],
    parameters: dict[str, object],
    *,
    minimum_electron_p: float | None = 2.0,
    minimum_q2: float | None = 1.0,
    minimum_w: float | None = 2.0,
    require_pass_fiducial: bool = False,
    require_pass_exclusivity: bool = False,
    external_selection_mask: np.ndarray | None = None,
    theta_split_deg: float = 6.5,
) -> tuple[dict[str, object], dict[int, dict[str, np.ndarray]]]:
    mask, selection = _analysis_mask(
        arrays,
        minimum_electron_p=minimum_electron_p,
        minimum_q2=minimum_q2,
        minimum_w=minimum_w,
        require_pass_fiducial=require_pass_fiducial,
        require_pass_exclusivity=require_pass_exclusivity,
        external_selection_mask=external_selection_mask,
    )
    selected = subset_arrays(arrays, mask)
    theta_deg = np.asarray(selected["electronTheta"], dtype=float) * RAD_TO_DEG
    sectors = np.asarray(selected["electronSector"], dtype=int)
    local_phi_deg = sector_local_phi(
        np.asarray(selected["electronPhi"], dtype=float), sectors
    )
    region_map = {
        _region_key(region): region for region in parameters.get("regions", [])
    }
    assignment_map = {
        (
            int(entry["pid"]),
            int(entry["detector"]),
            int(entry["sector"]),
        ): entry
        for entry in parameters.get("modelSelection", {}).get("assignments", [])
    }
    regions: list[dict[str, object]] = []
    plot_data: dict[int, dict[str, np.ndarray]] = {}
    total_supported = 0
    total_selected = 0
    total_categories: dict[str, int] = {
        "supported": 0,
        "belowSupportCellThetaRange": 0,
        "aboveSupportCellThetaRange": 0,
        "outsidePhiRangeWithinSupportCellThetaRange": 0,
        "insideSupportCellThetaRangeOutsideSupportCells": 0,
    }
    missing_parameter_regions: list[str] = []
    for sector in range(1, 7):
        key = (11, 1, sector)
        region = region_map.get(key)
        sector_mask = sectors == sector
        theta = theta_deg[sector_mask]
        phi = local_phi_deg[sector_mask]
        entries = int(theta.size)
        total_selected += entries
        if region is None:
            missing_parameter_regions.append(f"pid11_det1_sector{sector}")
            continue
        theta_range = [float(value) for value in region["thetaRangeDeg"]]
        support_theta_range = _support_cell_theta_range(region)
        phi_range = [float(value) for value in region["phiRangeDeg"]]
        below_support_theta = theta < support_theta_range[0]
        above_support_theta = theta > support_theta_range[1]
        support_theta_inside = ~(below_support_theta | above_support_theta)
        phi_outside = support_theta_inside & (
            (phi < phi_range[0]) | (phi > phi_range[1])
        )
        support_envelope = support_theta_inside & ~phi_outside
        supported = region_support_mask(region, theta, phi)
        between_cells = support_envelope & ~supported
        counts = {
            "supported": int(np.count_nonzero(supported)),
            "belowSupportCellThetaRange": int(
                np.count_nonzero(below_support_theta)
            ),
            "aboveSupportCellThetaRange": int(
                np.count_nonzero(above_support_theta)
            ),
            "outsidePhiRangeWithinSupportCellThetaRange": int(
                np.count_nonzero(phi_outside)
            ),
            "insideSupportCellThetaRangeOutsideSupportCells": int(
                np.count_nonzero(between_cells)
            ),
        }
        classified = sum(counts.values())
        if classified != entries:
            raise RuntimeError(
                f"coverage categories do not partition sector {sector}: "
                f"{classified} != {entries}"
            )
        parameter_below = theta < theta_range[0]
        parameter_above = theta > theta_range[1]
        parameter_theta_inside = ~(parameter_below | parameter_above)
        parameter_phi_outside = parameter_theta_inside & (
            (phi < phi_range[0]) | (phi > phi_range[1])
        )
        parameter_rectangle = parameter_theta_inside & ~parameter_phi_outside
        parameter_counts = {
            "belowThetaRange": int(np.count_nonzero(parameter_below)),
            "aboveThetaRange": int(np.count_nonzero(parameter_above)),
            "outsidePhiRangeWithinThetaRange": int(
                np.count_nonzero(parameter_phi_outside)
            ),
            "insideParameterRectangle": int(
                np.count_nonzero(parameter_rectangle)
            ),
        }
        if sum(parameter_counts.values()) != entries:
            raise RuntimeError(
                f"parameter-envelope categories do not partition sector {sector}"
            )
        correction = (
            evaluate_region(region, theta[supported], phi[supported])
            if np.any(supported) else np.asarray([], dtype=float)
        )
        assignment = assignment_map.get(key, {})
        model = assignment.get("model") or region.get("validationModel")
        total_supported += counts["supported"]
        for name, count in counts.items():
            total_categories[name] += count
        regions.append({
            "pid": 11,
            "detector": 1,
            "sector": sector,
            "region": f"pid11_det1_sector{sector}",
            "model": model,
            "domainSensitive": bool(
                assignment.get(
                    "domainSensitive", region.get("systematicDomainSensitive", False)
                )
            ),
            "entries": entries,
            "thetaRangeDeg": theta_range,
            "phiRangeDeg": phi_range,
            "parameterThetaRangeDeg": theta_range,
            "parameterPhiRangeDeg": phi_range,
            "supportCellThetaRangeDeg": support_theta_range,
            "supportCells": len(region.get("supportCells", [])),
            "supportedEntries": counts["supported"],
            "supportFraction": _fraction(counts["supported"], entries),
            "unsupportedEntries": entries - counts["supported"],
            "unsupportedFraction": _fraction(
                entries - counts["supported"], entries
            ),
            "coverageCategories": {
                name: {
                    "entries": count,
                    "fraction": _fraction(count, entries),
                }
                for name, count in counts.items()
            },
            "parameterEnvelopeCategories": {
                name: {
                    "entries": count,
                    "fraction": _fraction(count, entries),
                }
                for name, count in parameter_counts.items()
            },
            "thetaBelowSplitDeg": theta_split_deg,
            "thetaBelowSplitEntries": int(np.count_nonzero(theta < theta_split_deg)),
            "thetaBelowSplitFraction": _fraction(
                int(np.count_nonzero(theta < theta_split_deg)), entries
            ),
            "thetaQuantilesDeg": _quantiles(theta),
            "localPhiQuantilesDeg": _quantiles(phi),
            "supportedCorrectionFractionQuantiles": _quantiles(correction),
        })
        plot_data[sector] = {
            "thetaDeg": theta,
            "localPhiDeg": phi,
            "supported": supported,
        }
    if missing_parameter_regions:
        raise RuntimeError(
            "parameter file has no electron correction for: "
            + ", ".join(missing_parameter_regions)
        )
    return {
        "selection": selection,
        "thetaSplitDeg": theta_split_deg,
        "overall": {
            "entries": total_selected,
            "supportedEntries": total_supported,
            "supportFraction": _fraction(total_supported, total_selected),
            "unsupportedEntries": total_selected - total_supported,
            "unsupportedFraction": _fraction(
                total_selected - total_supported, total_selected
            ),
            "coverageCategories": {
                name: {
                    "entries": count,
                    "fraction": _fraction(count, total_selected),
                }
                for name, count in total_categories.items()
            },
        },
        "regions": regions,
    }, plot_data


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


def _write_tsv(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "sector", "model", "domainSensitive", "entries",
            "supportedEntries", "supportFraction",
            "belowSupportCellThetaRangeFraction",
            "aboveSupportCellThetaRangeFraction",
            "outsidePhiRangeWithinSupportCellThetaRangeFraction",
            "insideSupportCellThetaRangeOutsideSupportCellsFraction",
            "thetaBelowSplitFraction", "supportCellThetaMinDeg",
            "supportCellThetaMaxDeg", "parameterThetaMinDeg",
            "parameterThetaMaxDeg", "supportCells",
        ])
        for region in report["regions"]:
            categories = region["coverageCategories"]
            writer.writerow([
                region["sector"],
                region.get("model") or "",
                region["domainSensitive"],
                region["entries"],
                region["supportedEntries"],
                region["supportFraction"],
                categories["belowSupportCellThetaRange"]["fraction"],
                categories["aboveSupportCellThetaRange"]["fraction"],
                categories[
                    "outsidePhiRangeWithinSupportCellThetaRange"
                ]["fraction"],
                categories[
                    "insideSupportCellThetaRangeOutsideSupportCells"
                ]["fraction"],
                region["thetaBelowSplitFraction"],
                region["supportCellThetaRangeDeg"][0],
                region["supportCellThetaRangeDeg"][1],
                region["parameterThetaRangeDeg"][0],
                region["parameterThetaRangeDeg"][1],
                region["supportCells"],
            ])


def _plot_support_overlay(
    report: dict[str, object],
    plot_data: dict[int, dict[str, np.ndarray]],
    parameters: dict[str, object],
    output_path: Path,
    dataset_tag: str,
    beam_energy: float | None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    region_map = {
        int(region["sector"]): region
        for region in parameters.get("regions", [])
        if int(region["pid"]) == 11 and int(region["detector"]) == 1
    }
    report_map = {int(region["sector"]): region for region in report["regions"]}
    theta_parts = [
        data["thetaDeg"] for data in plot_data.values()
        if data["thetaDeg"].size
    ]
    theta_values = np.concatenate(theta_parts) if theta_parts else np.asarray([])
    region_theta_min = min(
        float(region["thetaRangeDeg"][0]) for region in region_map.values()
    )
    region_theta_max = max(
        float(region["thetaRangeDeg"][1]) for region in region_map.values()
    )
    if theta_values.size:
        theta_plot_min = min(
            region_theta_min, float(np.quantile(theta_values, 0.001))
        ) - 0.25
        theta_plot_max = max(
            region_theta_max, float(np.quantile(theta_values, 0.999))
        ) + 0.25
    else:
        theta_plot_min = region_theta_min - 0.25
        theta_plot_max = region_theta_max + 0.25
    theta_edges = np.linspace(theta_plot_min, theta_plot_max, 100)
    phi_edges = np.linspace(-30.0, 30.0, 73)
    histograms: dict[int, np.ndarray] = {}
    maximum = 1.0
    for sector, data in plot_data.items():
        histogram, _, _ = np.histogram2d(
            data["thetaDeg"], data["localPhiDeg"],
            bins=(theta_edges, phi_edges),
        )
        logged = np.log10(histogram + 1.0)
        histograms[sector] = logged
        maximum = max(maximum, float(np.max(logged)))

    fig = plt.figure(figsize=(18.5, 9.5))
    grid = fig.add_gridspec(
        2, 4, width_ratios=(1.0, 1.0, 1.0, 0.045),
        wspace=0.30, hspace=0.42,
    )
    axes = np.asarray([
        [fig.add_subplot(grid[0, column]) for column in range(3)],
        [fig.add_subplot(grid[1, column]) for column in range(3)],
    ])
    colorbar_axis = fig.add_subplot(grid[:, 3])
    image = None
    for sector, axis in enumerate(axes.flat, start=1):
        image = axis.pcolormesh(
            theta_edges,
            phi_edges,
            histograms.get(sector, np.zeros((99, 72))).T,
            cmap="viridis",
            vmin=0.0,
            vmax=maximum,
            shading="auto",
        )
        region = region_map[sector]
        for cell in region.get("supportCells", []):
            theta_range = [float(value) for value in cell["thetaRangeDeg"]]
            phi_range = [float(value) for value in cell["phiRangeDeg"]]
            axis.add_patch(Rectangle(
                (theta_range[0], phi_range[0]),
                theta_range[1] - theta_range[0],
                phi_range[1] - phi_range[0],
                fill=False,
                edgecolor="white",
                linewidth=0.75,
                alpha=0.9,
            ))
        summary = report_map[sector]
        support_theta_range = summary["supportCellThetaRangeDeg"]
        axis.axvline(
            float(support_theta_range[1]), color="#ffb000",
            linestyle=":", linewidth=1.35,
        )
        if summary["domainSensitive"]:
            axis.axvline(
                float(report["thetaSplitDeg"]), color="#ff5a5f",
                linestyle="--", linewidth=1.5,
            )
        axis.set_title(
            f"sector {sector}: {summary['model']}\n"
            f"exact support {100.0 * summary['supportFraction']:.1f}% | "
            f"cell theta <= {support_theta_range[1]:.2f} deg | "
            f"above cells "
            f"{100.0 * summary['coverageCategories']['aboveSupportCellThetaRange']['fraction']:.1f}%",
            fontsize=10,
        )
        axis.set_xlabel("electron theta [deg]")
        if sector in (1, 4):
            axis.set_ylabel("sector-local phi [deg]")
        axis.set_ylim(-30.0, 30.0)
    if image is not None:
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("log10(events + 1)")
    save_plot(
        fig,
        output_path,
        "ep-pi0 electron coverage of elastic momentum-correction support",
        dataset_tag,
        beam_energy,
        tight_layout=False,
    )
    plt.close(fig)


def run_coverage_diagnostic(
    arrays: dict[str, np.ndarray],
    parameters: dict[str, object],
    *,
    input_files: Iterable[Path],
    parameter_file: Path,
    tree: str,
    output_dir: Path,
    dataset_tag: str,
    minimum_electron_p: float | None = 2.0,
    minimum_q2: float | None = 1.0,
    minimum_w: float | None = 2.0,
    require_pass_fiducial: bool = False,
    require_pass_exclusivity: bool = False,
    external_selection_mask: np.ndarray | None = None,
    selection_mask_file: Path | None = None,
    exclusivity_cuts_file: Path | None = None,
    theta_split_deg: float = 6.5,
    make_plot: bool = True,
) -> dict[str, object]:
    coverage, plot_data = analyze_phase_space_coverage(
        arrays,
        parameters,
        minimum_electron_p=minimum_electron_p,
        minimum_q2=minimum_q2,
        minimum_w=minimum_w,
        require_pass_fiducial=require_pass_fiducial,
        require_pass_exclusivity=require_pass_exclusivity,
        external_selection_mask=external_selection_mask,
        theta_split_deg=theta_split_deg,
    )
    report = {
        "schema": "elastic_momentum_phase_space_coverage/v2",
        "datasetTag": dataset_tag,
        "inputFiles": [str(path) for path in input_files],
        "tree": tree,
        "parameterFile": str(parameter_file),
        "parameterDatasetTag": parameters.get("datasetTag"),
        "parameterCalibrationRole": parameters.get("calibrationRole"),
        "strictSelection": {
            "maskFile": (
                str(selection_mask_file) if selection_mask_file is not None else None
            ),
            "exclusivityCutsFile": (
                str(exclusivity_cuts_file)
                if exclusivity_cuts_file is not None else None
            ),
            "exclusivityCutsSummary": (
                summarize_exclusivity_cuts(exclusivity_cuts_file)
                if exclusivity_cuts_file is not None else None
            ),
        },
        **coverage,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "elastic_support_coverage.json", report)
    _write_tsv(output_dir / "elastic_support_coverage.tsv", report)
    if make_plot:
        beam_energy = parameters.get("beamEnergyGeV")
        _plot_support_overlay(
            report,
            plot_data,
            parameters,
            output_dir / "elastic_support_overlay.png",
            dataset_tag,
            float(beam_energy) if beam_energy is not None else None,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure selected ep-pi0 electron coverage of the exact adaptive "
            "support cells in an elastic momentum-correction parameter file."
        )
    )
    parser.add_argument("input_files", nargs="+", type=Path)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--min-electron-p", type=float, default=2.0)
    parser.add_argument("--min-q2", type=float, default=1.0)
    parser.add_argument("--min-w", type=float, default=2.0)
    parser.add_argument("--require-pass-fiducial", action="store_true")
    parser.add_argument("--require-pass-exclusivity", action="store_true")
    parser.add_argument(
        "--selection-mask", type=Path,
        help=(
            "External one-dimensional NPY/NPZ boolean mask aligned one-to-one "
            "with the concatenated input tree entries"
        ),
    )
    parser.add_argument(
        "--selection-mask-key", default="mask",
        help="Array name when --selection-mask is an NPZ (default: mask)",
    )
    parser.add_argument(
        "--exclusivity-cuts", type=Path,
        help=(
            "NPZ cut table recorded for provenance; the precomputed "
            "--selection-mask supplies the actual event selection"
        ),
    )
    parser.add_argument("--theta-split-deg", type=float, default=6.5)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("maximum rows must be positive")
    if args.selection_mask is not None and args.max_rows is not None:
        raise ValueError(
            "--selection-mask cannot be combined with --max-rows because the "
            "mask must align with the complete input tree"
        )
    if args.exclusivity_cuts is not None and args.selection_mask is None:
        raise ValueError("--exclusivity-cuts requires --selection-mask")
    parts = [
        load_eppi0_electron_arrays(path, args.tree, args.max_rows)
        for path in args.input_files
    ]
    arrays = concatenate_arrays(parts)
    if args.max_rows is not None and len(parts) > 1:
        arrays = subset_arrays(
            arrays,
            np.arange(next(iter(arrays.values())).size) < args.max_rows,
        )
    entries = int(np.asarray(arrays["electronP"]).size)
    external_selection_mask = (
        load_selection_mask(
            args.selection_mask, entries, key=args.selection_mask_key
        )
        if args.selection_mask is not None else None
    )
    parameters = _read_json(args.parameters)
    report = run_coverage_diagnostic(
        arrays,
        parameters,
        input_files=args.input_files,
        parameter_file=args.parameters,
        tree=args.tree,
        output_dir=args.output_dir,
        dataset_tag=args.dataset_tag,
        minimum_electron_p=args.min_electron_p,
        minimum_q2=args.min_q2,
        minimum_w=args.min_w,
        require_pass_fiducial=args.require_pass_fiducial,
        require_pass_exclusivity=args.require_pass_exclusivity,
        external_selection_mask=external_selection_mask,
        selection_mask_file=args.selection_mask,
        exclusivity_cuts_file=args.exclusivity_cuts,
        theta_split_deg=args.theta_split_deg,
        make_plot=not args.no_plot,
    )
    print(
        f"Wrote elastic support coverage diagnostics to {args.output_dir}; "
        f"overall exact support = "
        f"{100.0 * report['overall']['supportFraction']:.2f}%"
    )
    overall_categories = report["overall"]["coverageCategories"]
    above_cells = overall_categories["aboveSupportCellThetaRange"]["fraction"]
    between_cells = overall_categories[
        "insideSupportCellThetaRangeOutsideSupportCells"
    ]["fraction"]
    print(
        "  unsupported breakdown: "
        f"above accepted-cell theta={100.0 * above_cells:.2f}%; "
        f"within theta envelope but outside cells={100.0 * between_cells:.2f}%"
    )
    if external_selection_mask is not None:
        selected = int(np.count_nonzero(external_selection_mask))
        print(
            f"  external selection mask: {selected}/{entries} "
            f"({100.0 * _fraction(selected, entries):.2f}%)"
        )
        if selected == entries:
            print(
                "Warning: --selection-mask rejected zero entries; verify that "
                "it represents an additional strict selection",
                file=sys.stderr,
            )
    branch_selection = report["selection"]["requestedBranchSelections"]
    for branch, option in (
        ("passFiducial", "--require-pass-fiducial"),
        ("passExclusivity", "--require-pass-exclusivity"),
    ):
        effect = branch_selection[branch]
        if effect["requested"] and effect["noEffect"]:
            print(
                f"Warning: {option} rejected zero of "
                f"{effect['eligibleEntries']} eligible entries; the branch "
                "does not impose an additional selection on this input",
                file=sys.stderr,
            )
    for region in report["regions"]:
        print(
            f"  sector {region['sector']}: model={region['model']}; "
            f"support={100.0 * region['supportFraction']:.2f}%; "
            f"cell theta max={region['supportCellThetaRangeDeg'][1]:.3f} deg; "
            f"above cells="
            f"{100.0 * region['coverageCategories']['aboveSupportCellThetaRange']['fraction']:.2f}%"
        )


if __name__ == "__main__":
    main()
