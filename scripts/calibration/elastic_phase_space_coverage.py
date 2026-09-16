from __future__ import annotations

import argparse
import csv
import json
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
    if require_pass_fiducial:
        if "passFiducial" not in arrays:
            raise RuntimeError(
                "--require-pass-fiducial needs the passFiducial branch"
            )
        mask &= np.asarray(arrays["passFiducial"], dtype=int) != 0
    if require_pass_exclusivity:
        if "passExclusivity" not in arrays:
            raise RuntimeError(
                "--require-pass-exclusivity needs the passExclusivity branch"
            )
        mask &= np.asarray(arrays["passExclusivity"], dtype=int) != 0

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


def analyze_phase_space_coverage(
    arrays: dict[str, np.ndarray],
    parameters: dict[str, object],
    *,
    minimum_electron_p: float | None = 2.0,
    minimum_q2: float | None = 1.0,
    minimum_w: float | None = 2.0,
    require_pass_fiducial: bool = False,
    require_pass_exclusivity: bool = False,
    theta_split_deg: float = 6.5,
) -> tuple[dict[str, object], dict[int, dict[str, np.ndarray]]]:
    mask, selection = _analysis_mask(
        arrays,
        minimum_electron_p=minimum_electron_p,
        minimum_q2=minimum_q2,
        minimum_w=minimum_w,
        require_pass_fiducial=require_pass_fiducial,
        require_pass_exclusivity=require_pass_exclusivity,
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
        phi_range = [float(value) for value in region["phiRangeDeg"]]
        below_theta = theta < theta_range[0]
        above_theta = theta > theta_range[1]
        theta_inside = ~(below_theta | above_theta)
        phi_outside = theta_inside & (
            (phi < phi_range[0]) | (phi > phi_range[1])
        )
        rectangle = theta_inside & ~phi_outside
        supported = region_support_mask(region, theta, phi)
        between_cells = rectangle & ~supported
        counts = {
            "supported": int(np.count_nonzero(supported)),
            "belowThetaRange": int(np.count_nonzero(below_theta)),
            "aboveThetaRange": int(np.count_nonzero(above_theta)),
            "outsidePhiRange": int(np.count_nonzero(phi_outside)),
            "insideRangeOutsideSupportCells": int(np.count_nonzero(between_cells)),
        }
        classified = sum(counts.values())
        if classified != entries:
            raise RuntimeError(
                f"coverage categories do not partition sector {sector}: "
                f"{classified} != {entries}"
            )
        correction = (
            evaluate_region(region, theta[supported], phi[supported])
            if np.any(supported) else np.asarray([], dtype=float)
        )
        assignment = assignment_map.get(key, {})
        model = assignment.get("model") or region.get("validationModel")
        total_supported += counts["supported"]
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
            "supportedEntries", "supportFraction", "belowThetaRangeFraction",
            "aboveThetaRangeFraction", "outsidePhiRangeFraction",
            "insideRangeOutsideSupportCellsFraction", "thetaBelowSplitFraction",
            "thetaMinDeg", "thetaMaxDeg", "supportCells",
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
                categories["belowThetaRange"]["fraction"],
                categories["aboveThetaRange"]["fraction"],
                categories["outsidePhiRange"]["fraction"],
                categories["insideRangeOutsideSupportCells"]["fraction"],
                region["thetaBelowSplitFraction"],
                region["thetaRangeDeg"][0],
                region["thetaRangeDeg"][1],
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
        if summary["domainSensitive"]:
            axis.axvline(
                float(report["thetaSplitDeg"]), color="#ff5a5f",
                linestyle="--", linewidth=1.5,
            )
        axis.set_title(
            f"sector {sector}: {summary['model']}\n"
            f"exact support {100.0 * summary['supportFraction']:.1f}% | "
            f"theta < {report['thetaSplitDeg']:g}: "
            f"{100.0 * summary['thetaBelowSplitFraction']:.1f}%",
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
        theta_split_deg=theta_split_deg,
    )
    report = {
        "schema": "elastic_momentum_phase_space_coverage/v1",
        "datasetTag": dataset_tag,
        "inputFiles": [str(path) for path in input_files],
        "tree": tree,
        "parameterFile": str(parameter_file),
        "parameterDatasetTag": parameters.get("datasetTag"),
        "parameterCalibrationRole": parameters.get("calibrationRole"),
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
    parser.add_argument("--theta-split-deg", type=float, default=6.5)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("maximum rows must be positive")
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
        theta_split_deg=args.theta_split_deg,
        make_plot=not args.no_plot,
    )
    print(
        f"Wrote elastic support coverage diagnostics to {args.output_dir}; "
        f"overall exact support = "
        f"{100.0 * report['overall']['supportFraction']:.2f}%"
    )
    for region in report["regions"]:
        print(
            f"  sector {region['sector']}: model={region['model']}; "
            f"support={100.0 * region['supportFraction']:.2f}%; "
            f"theta<{report['thetaSplitDeg']:g}="
            f"{100.0 * region['thetaBelowSplitFraction']:.2f}%"
        )


if __name__ == "__main__":
    main()
