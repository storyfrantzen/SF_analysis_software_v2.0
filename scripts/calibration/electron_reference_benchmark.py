from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .elastic_momentum import (
    RAD_TO_DEG,
    ElasticFitConfig,
    evaluate_region,
    region_support_mask,
    select_calibration_events,
)
from .elastic_run_validation import (
    RunBlock,
    _cell_mask,
    _core_summary,
    _particle_view,
    _region_coordinates,
    filter_arrays_by_run_classes,
    load_candidate_inputs,
    load_run_catalog,
    make_run_blocks,
    make_run_class_fold_blocks,
    subset_arrays,
)
from .eppi0_momentum_validation import (
    EXPECTED_CENTERS,
    PLOT_LABELS,
    _json_safe,
    _write_migration_tsv,
    _write_observable_tsv,
    apply_supported_electron_correction,
    load_aligned_selection_mask,
    load_eppi0_arrays,
    run_paired_validation,
)
from .elastic_phase_space_coverage import summarize_exclusivity_cuts
from .plot_utils import save_plot


# Table 4 of Y. Guo and J. Huang, ``Beam Spin Asymmetry Measurement of
# Deeply Virtual Compton Scattering with CLAS12 at 6.535 GeV and 7.546 GeV''.
# Rows are sectors 1--6. Columns are c00,c01,c02,c10,c11,c12 in Eqs. 17--19.
YIJIE_JOSH_RGK_6535_COEFFICIENTS = np.asarray([
    [-0.0607704, 0.00947409, -2.79104e-4,
     3.40388e-4, -2.97480e-5, -3.01811e-6],
    [-0.148989, 0.0191382, -3.16141e-4,
     1.33595e-3, -1.51985e-4, -5.84338e-7],
    [-0.349771, 0.0601409, -2.21405e-3,
     2.37725e-3, -4.26788e-4, 1.58320e-5],
    [-0.142846, 0.0262234, -9.62131e-4,
     5.94672e-4, -1.12409e-4, 4.22004e-6],
    [0.158532, -0.0293364, 7.12792e-4,
     1.82795e-3, -3.16720e-4, 8.54425e-6],
    [0.0420628, -0.00371369, -1.34925e-6,
     2.05466e-3, -2.93060e-4, 7.95533e-6],
], dtype=float)

REFERENCE_METADATA = {
    "name": "yijie-josh-rgk-6535",
    "source": "YijieJoshDVCSBSA.pdf",
    "beamEnergyGeV": 6.535,
    "equations": "16-19",
    "coefficientTable": 4,
    "correctionConvention": "p_corrected = p_reconstructed + deltaP",
    "angleUnits": "degrees",
    "deltaPMomentumUnits": "GeV",
    "phiConvention": (
        "sector-local phi = unfolded global phi - 60*(sector-1); global phi is "
        "first unfolded to -25 <= phi < 335 degrees"
    ),
    "coefficientColumns": ["c00", "c01", "c02", "c10", "c11", "c12"],
    "coefficientsBySector": {
        str(sector): values.tolist()
        for sector, values in enumerate(YIJIE_JOSH_RGK_6535_COEFFICIENTS, 1)
    },
}


def unfold_reference_phi(phi_rad: np.ndarray) -> np.ndarray:
    """Map wrapped CLAS12 azimuth to the reference's continuous convention."""
    phi_deg = np.asarray(phi_rad, dtype=float) * RAD_TO_DEG
    return (phi_deg + 25.0) % 360.0 - 25.0


def reference_sector_phi(
    phi_rad: np.ndarray,
    sector: np.ndarray,
) -> np.ndarray:
    """Return the sector-local azimuth used by the reference fit."""
    unfolded, sectors = np.broadcast_arrays(
        unfold_reference_phi(phi_rad), np.asarray(sector, dtype=int)
    )
    if np.any((sectors < 1) | (sectors > 6)):
        raise ValueError("electron sectors must lie in [1, 6]")
    return unfolded - 60.0 * (sectors - 1)


def yijie_josh_rgk_6535_delta_p(
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    sector: np.ndarray,
) -> np.ndarray:
    """Evaluate the published 6.535-GeV additive electron correction in GeV."""
    theta_deg, phi_deg, sectors = np.broadcast_arrays(
        np.asarray(theta_rad, dtype=float) * RAD_TO_DEG,
        reference_sector_phi(phi_rad, sector),
        np.asarray(sector, dtype=int),
    )
    if np.any((sectors < 1) | (sectors > 6)):
        raise ValueError("electron sectors must lie in [1, 6]")
    coefficients = YIJIE_JOSH_RGK_6535_COEFFICIENTS[sectors - 1]
    a0 = (
        coefficients[..., 0]
        + coefficients[..., 1] * theta_deg
        + coefficients[..., 2] * np.square(theta_deg)
    )
    a1 = (
        coefficients[..., 3]
        + coefficients[..., 4] * theta_deg
        + coefficients[..., 5] * np.square(theta_deg)
    )
    return a0 + a1 * phi_deg


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")


def _fit_config_from_parameters(parameters: dict[str, object]) -> ElasticFitConfig:
    fit = parameters.get("fitConfiguration", {})
    selection = parameters.get("selection", {})
    return ElasticFitConfig(
        beam_energy=float(parameters["beamEnergyGeV"]),
        torus=int(parameters.get("torus", 0)),
        electron_selection=str(
            fit.get("electronSelection", selection.get("mode", "inclusive-w"))
        ),
        elastic_w_max_abs_gev=float(
            fit.get("elasticWMaxAbsGeV", selection.get("elasticWMaxAbsGeV", 0.20))
        ),
        missing_energy_max_gev=float(fit.get("missingEnergyMaxGeV", 0.75)),
        theta_min_deg=fit.get("thetaMinDeg"),
        theta_max_deg=fit.get("thetaMaxDeg"),
        theta_trim_quantile=float(fit.get("thetaTrimQuantile", 0.005)),
        residual_trim_quantile=float(fit.get("residualTrimQuantile", 0.01)),
        theta_bins=int(fit.get("thetaBins", 10)),
        phi_bins=int(fit.get("phiBins", 7)),
        profile_binning=str(fit.get("profileBinning", "adaptive")),
        max_theta_bin_width_deg=fit.get("maxThetaBinWidthDeg"),
        target_cell_entries=int(fit.get("targetCellEntries", 10_000)),
        min_phi_cells_per_theta=int(fit.get("minPhiCellsPerTheta", 2)),
        min_bin_entries=int(fit.get("minBinEntries", 300)),
        min_region_entries=int(fit.get("minRegionEntries", 1_000)),
        peak_search_max_abs_residual=float(
            fit.get("peakSearchMaxAbsResidual", 0.10)
        ),
        peak_seed_half_width=float(fit.get("peakSeedHalfWidth", 0.03)),
        core_sigma_clip=float(fit.get("coreSigmaClip", 3.0)),
        min_core_fraction=float(fit.get("minCoreFraction", 0.20)),
        min_peak_significance=float(fit.get("minPeakSignificance", 3.0)),
        max_core_width=float(fit.get("maxCoreWidth", 0.05)),
        max_condition_number=float(fit.get("maxConditionNumber", 100.0)),
        max_abs_surface_correction=float(
            fit.get("maxAbsSurfaceCorrection", 0.05)
        ),
    )


def _summarize_method_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    sectors: dict[str, object] = {}
    for sector in range(1, 7):
        selected = [row for row in rows if int(row["sector"]) == sector]
        after = [
            float(row["cellCenterRmsAfter"])
            for row in selected if row["cellCenterRmsAfter"] is not None
        ]
        before = [
            float(row["cellCenterRmsBefore"])
            for row in selected if row["cellCenterRmsBefore"] is not None
        ]
        sectors[str(sector)] = {
            "blocks": len(selected),
            "validatedCells": int(sum(int(row["validatedCells"]) for row in selected)),
            "medianBlockCellCenterRmsBefore": (
                float(np.median(before)) if before else None
            ),
            "medianBlockCellCenterRmsAfter": (
                float(np.median(after)) if after else None
            ),
        }
    all_after = [
        float(row["cellCenterRmsAfter"])
        for row in rows if row["cellCenterRmsAfter"] is not None
    ]
    return {
        "validatedCells": int(sum(int(row["validatedCells"]) for row in rows)),
        "medianBlockCellCenterRmsAfter": (
            float(np.median(all_after)) if all_after else None
        ),
        "sectors": sectors,
    }


def run_elastic_benchmark(
    arrays: dict[str, np.ndarray],
    parameters: dict[str, object],
    blocks: list[RunBlock],
    cfg: ElasticFitConfig,
) -> dict[str, object]:
    if not np.isclose(cfg.beam_energy, 6.535, rtol=0.0, atol=1.0e-9):
        raise ValueError("the published reference benchmark is defined at 6.535 GeV")
    selected, selection = select_calibration_events(arrays, cfg)
    view = _particle_view(selected, "electron", cfg.beam_energy)
    method_rows = {name: [] for name in ("none", "ours", "yijieJosh")}
    correction_values = {name: [] for name in ("ours", "yijieJosh")}
    sign_pairs: list[np.ndarray] = []

    for region in parameters.get("regions", []):
        if int(region.get("pid", 0)) != 11:
            continue
        region_mask, theta_deg, local_phi_deg = _region_coordinates(view, region)
        runs = np.asarray(view["run"])[region_mask]
        residual = np.asarray(view["residual"])[region_mask]
        measured = np.asarray(selected["electronP"], dtype=float)[region_mask]
        theta_rad = np.asarray(selected["electronTheta"], dtype=float)[region_mask]
        phi_rad = np.asarray(selected["electronPhi"], dtype=float)[region_mask]
        sectors = np.asarray(selected["electronSector"], dtype=int)[region_mask]
        support = region_support_mask(region, theta_deg, local_phi_deg)
        ours_fraction = np.zeros(theta_deg.shape, dtype=float)
        ours_fraction[support] = evaluate_region(
            region, theta_deg[support], local_phi_deg[support]
        )
        reference_delta = yijie_josh_rgk_6535_delta_p(
            theta_rad, phi_rad, sectors
        )
        reference_fraction = np.divide(
            reference_delta, measured,
            out=np.full(measured.shape, np.nan), where=measured > 0.0,
        )
        correction_values["ours"].append(ours_fraction[support])
        correction_values["yijieJosh"].append(reference_fraction[support])
        sign_pairs.append(
            np.sign(ours_fraction[support]) == np.sign(reference_fraction[support])
        )
        expected = measured * (1.0 + residual)
        corrected_momenta = {
            "none": measured,
            "ours": measured * (1.0 + ours_fraction),
            "yijieJosh": measured + np.where(support, reference_delta, 0.0),
        }
        for method, corrected_momentum in corrected_momenta.items():
            if np.any(~np.isfinite(corrected_momentum) | (corrected_momentum <= 0.0)):
                raise ValueError(f"{method} produced a non-positive electron momentum")
            corrected_residual = expected / corrected_momentum - 1.0
            for block in blocks:
                in_block = np.isin(runs, block.runs) & support
                before = _core_summary(residual[in_block], cfg)
                after = _core_summary(corrected_residual[in_block], cfg)
                cell_before: list[float] = []
                cell_after: list[float] = []
                for cell in region["fit"]["acceptedProfileCells"]:
                    in_cell = in_block & _cell_mask(theta_deg, local_phi_deg, cell)
                    before_cell = _core_summary(residual[in_cell], cfg)
                    after_cell = _core_summary(corrected_residual[in_cell], cfg)
                    if before_cell is None or after_cell is None:
                        continue
                    cell_before.append(float(before_cell["center"]))
                    cell_after.append(float(after_cell["center"]))
                method_rows[method].append({
                    "method": method,
                    "block": block.index,
                    "fold": "A" if block.index % 2 == 0 else "B",
                    "runClass": block.run_class,
                    "runMin": block.run_min,
                    "runMax": block.run_max,
                    "sector": int(region["sector"]),
                    "supportEntries": int(np.count_nonzero(in_block)),
                    "before": before,
                    "after": after,
                    "validatedCells": len(cell_after),
                    "cellCenterRmsBefore": (
                        float(np.sqrt(np.mean(np.square(cell_before))))
                        if cell_before else None
                    ),
                    "cellCenterRmsAfter": (
                        float(np.sqrt(np.mean(np.square(cell_after))))
                        if cell_after else None
                    ),
                    "maxAbsCellCenterAfter": (
                        float(np.max(np.abs(cell_after))) if cell_after else None
                    ),
                })

    correction_summary: dict[str, object] = {}
    for method, pieces in correction_values.items():
        values = np.concatenate(pieces) if pieces else np.asarray([], dtype=float)
        correction_summary[method] = {
            "entries": int(values.size),
            "meanFraction": float(np.mean(values)) if values.size else None,
            "medianFraction": float(np.median(values)) if values.size else None,
            "positiveFraction": float(np.mean(values > 0.0)) if values.size else None,
            "negativeFraction": float(np.mean(values < 0.0)) if values.size else None,
        }
    sign_values = np.concatenate(sign_pairs) if sign_pairs else np.asarray([], dtype=bool)
    return {
        "schema": "elastic-electron-reference-benchmark/v1",
        "evaluationType": (
            "fixed surfaces evaluated without refitting on identical run blocks "
            "and the accepted support cells of the local correction"
        ),
        "beamEnergyGeV": cfg.beam_energy,
        "selection": selection,
        "reference": REFERENCE_METADATA,
        "runBlocks": [block.to_json() for block in blocks],
        "methods": {
            method: {
                "summary": _summarize_method_rows(rows),
                "rows": rows,
            }
            for method, rows in method_rows.items()
        },
        "correctionSummary": correction_summary,
        "correctionSignAgreementFraction": (
            float(np.mean(sign_values)) if sign_values.size else None
        ),
    }


def _write_elastic_tsv(report: dict[str, object], path: Path) -> None:
    fields = (
        "method", "block", "fold", "runClass", "runMin", "runMax", "sector",
        "supportEntries", "validatedCells", "cellCenterRmsBefore",
        "cellCenterRmsAfter", "maxAbsCellCenterAfter", "coreCenterBefore",
        "coreCenterAfter", "coreWidthBefore", "coreWidthAfter",
    )
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for method in report["methods"].values():
            for row in method["rows"]:
                writer.writerow({
                    **{name: row.get(name) for name in fields},
                    "coreCenterBefore": (row["before"] or {}).get("center"),
                    "coreCenterAfter": (row["after"] or {}).get("center"),
                    "coreWidthBefore": (row["before"] or {}).get("coreWidth"),
                    "coreWidthAfter": (row["after"] or {}).get("coreWidth"),
                })


def _plot_elastic_benchmark(
    report: dict[str, object], output_dir: Path, dataset_tag: str
) -> None:
    import matplotlib.pyplot as plt

    methods = ("none", "ours", "yijieJosh")
    labels = {"none": "none", "ours": "ours", "yijieJosh": "Yijie/Josh"}
    sectors = np.arange(1, 7)
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for method in methods:
        values = [
            report["methods"][method]["summary"]["sectors"][str(sector)][
                "medianBlockCellCenterRmsAfter"
            ]
            for sector in sectors
        ]
        axis.plot(sectors, 100.0 * np.asarray(values, dtype=float), "o-", label=labels[method])
    axis.set_xlabel("electron sector")
    axis.set_ylabel("median block cell-center RMS [%]")
    axis.set_xticks(sectors)
    axis.grid(alpha=0.25)
    axis.legend()
    save_plot(
        fig, output_dir / "elastic_reference_cell_rms.png",
        "RGK elastic closure: native correction prescriptions on common support",
        dataset_tag, float(report["beamEnergyGeV"]),
    )
    plt.close(fig)


def _benchmark_eppi0(
    arrays: dict[str, np.ndarray],
    parameters: dict[str, object],
    cuts: object,
    binning: object,
    *,
    external_selection_mask: np.ndarray | None,
    minimum_electron_p: float,
    minimum_q2: float,
    minimum_w: float,
) -> dict[str, object]:
    configured_pids = {
        int(region.get("pid", 0)) for region in parameters.get("regions", [])
    }
    if configured_pids != {11}:
        raise ValueError(
            "the electron reference benchmark requires an electron-only parameter file"
        )
    electron_p = np.asarray(arrays["electronP"], dtype=float)
    _, common_support, _, _ = apply_supported_electron_correction(arrays, parameters)
    reference_p = electron_p + yijie_josh_rgk_6535_delta_p(
        arrays["electronTheta"], arrays["electronPhi"], arrays["electronSector"]
    )
    method_settings = {
        "none": {
            "override": electron_p,
            "label": "no electron correction",
            "convention": "p_after = p_before",
        },
        "ours": {
            "override": None,
            "label": "local elastic correction",
            "convention": "p_after = p_before * (1 + fractionalCorrection)",
        },
        "yijieJosh": {
            "override": reference_p,
            "label": "Yijie/Josh RGK 6.535-GeV correction",
            "convention": "p_after = p_before + deltaP(theta, sector-local phi)",
        },
    }
    reports: dict[str, object] = {}
    for method, setting in method_settings.items():
        report, diagnostic = run_paired_validation(
            arrays,
            parameters,
            cuts,
            binning,
            external_selection_mask=external_selection_mask,
            minimum_electron_p=minimum_electron_p,
            minimum_q2=minimum_q2,
            minimum_w=minimum_w,
            electron_momentum_override=setting["override"],
            electron_support_override=(
                common_support if setting["override"] is not None else None
            ),
            electron_correction_label=str(setting["label"]),
            electron_correction_convention=str(setting["convention"]),
        )
        if not np.array_equal(diagnostic["support"], common_support):
            raise RuntimeError(f"{method} did not retain the common support mask")
        reports[method] = report

    comparison: dict[str, object] = {}
    for quantity in EXPECTED_CENTERS:
        comparison[quantity] = {
            method: reports[method]["cohorts"]["fixedSupported"]["quantities"][
                quantity
            ]
            for method in method_settings
        }
    return {
        "schema": "eppi0-electron-reference-benchmark/v1",
        "beamEnergyGeV": float(parameters["beamEnergyGeV"]),
        "commonSupportSource": "accepted support cells of the local correction",
        "reference": REFERENCE_METADATA,
        "methods": reports,
        "observableComparison": comparison,
    }


def _write_eppi0_comparison_tsv(report: dict[str, object], path: Path) -> None:
    fields = (
        "method", "quantity", "entries", "beforeMean", "afterMean",
        "beforeRobustWidth", "afterRobustWidth", "rmsFromExpectedBefore",
        "rmsFromExpectedAfter",
    )
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for method, method_report in report["methods"].items():
            quantities = method_report["cohorts"]["fixedSupported"]["quantities"]
            for quantity, item in quantities.items():
                writer.writerow({
                    "method": method,
                    "quantity": quantity,
                    "entries": item["before"]["entries"],
                    "beforeMean": item["before"]["mean"],
                    "afterMean": item["after"]["mean"],
                    "beforeRobustWidth": item["before"]["robustWidth"],
                    "afterRobustWidth": item["after"]["robustWidth"],
                    "rmsFromExpectedBefore": item.get("rmsFromExpectedBefore"),
                    "rmsFromExpectedAfter": item.get("rmsFromExpectedAfter"),
                })


def _plot_eppi0_benchmark(
    report: dict[str, object],
    output_dir: Path,
    dataset_tag: str,
) -> None:
    import matplotlib.pyplot as plt

    quantities = ("missingEnergy", "missingPt", "m2Miss", "m2EpX", "mEggX", "deltaT")
    methods = ("none", "ours", "yijieJosh")
    labels = {"none": "none", "ours": "ours", "yijieJosh": "Yijie/Josh"}
    colors = {"none": "black", "ours": "#21618c", "yijieJosh": "#d95f02"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for axis, quantity in zip(axes.flat, quantities):
        values = []
        for method in methods:
            item = report["methods"][method]["cohorts"]["fixedSupported"][
                "quantities"
            ][quantity]
            values.append(item.get("rmsFromExpectedAfter"))
        axis.bar(np.arange(3), values, color=[colors[name] for name in methods])
        axis.set_xticks(np.arange(3), [labels[name] for name in methods], rotation=15)
        axis.set_ylabel("RMS from physical center")
        axis.set_title(PLOT_LABELS.get(quantity, quantity))
        axis.grid(axis="y", alpha=0.25)
    save_plot(
        fig, output_dir / "eppi0_reference_closure_rms.png",
        "Fixed supported ep-pi0 cohort: electron-correction benchmark",
        dataset_tag, float(report["beamEnergyGeV"]),
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    sectors = np.arange(1, 7)
    for method in ("ours", "yijieJosh"):
        means = [
            report["methods"][method]["correction"]["electronSectors"][str(sector)][
                "meanFraction"
            ]
            for sector in sectors
        ]
        axis.plot(sectors, 100.0 * np.asarray(means, dtype=float), "o-", label=labels[method])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("electron sector")
    axis.set_ylabel("mean applied momentum correction [%]")
    axis.set_xticks(sectors)
    axis.grid(alpha=0.25)
    axis.legend()
    save_plot(
        fig, output_dir / "eppi0_reference_correction_sign.png",
        "Electron correction sign and scale on the common ep-pi0 cohort",
        dataset_tag, float(report["beamEnergyGeV"]),
    )
    plt.close(fig)


def build_elastic_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare no correction, the local elastic electron correction, and "
            "the published Yijie/Josh 6.535-GeV additive surface on common cells."
        )
    )
    parser.add_argument("input_files", nargs="+", type=Path)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--run-catalog", type=Path)
    parser.add_argument("--include-run-classes", nargs="+")
    parser.add_argument("--fold-by-run-class", action="store_true")
    parser.add_argument("--block-target-selected", type=int, default=500_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def elastic_main(argv: Iterable[str] | None = None) -> None:
    args = build_elastic_parser().parse_args(argv)
    parameters = _read_json(args.parameters)
    arrays = load_candidate_inputs(args.input_files, args.tree, require_proton=False)
    if args.max_rows is not None:
        arrays = subset_arrays(
            arrays, np.arange(next(iter(arrays.values())).size) < args.max_rows
        )
    run_mapping = None
    run_selection = None
    if args.run_catalog is not None:
        arrays, run_mapping, run_selection = filter_arrays_by_run_classes(
            arrays,
            load_run_catalog(args.run_catalog),
            args.include_run_classes,
            catalog_path=args.run_catalog,
        )
    if args.fold_by_run_class:
        if run_mapping is None or len(args.include_run_classes or ()) != 2:
            raise ValueError(
                "--fold-by-run-class requires a catalog and exactly two included classes"
            )
        blocks = make_run_class_fold_blocks(
            arrays["runNum"], run_mapping, args.include_run_classes
        )
    else:
        blocks = make_run_blocks(
            arrays["runNum"], args.block_target_selected, run_mapping
        )
    report = run_elastic_benchmark(
        arrays, parameters, blocks, _fit_config_from_parameters(parameters)
    )
    report["inputFiles"] = [str(path) for path in args.input_files]
    report["parameterFile"] = str(args.parameters)
    report["runSelection"] = run_selection
    report["datasetTag"] = args.dataset_tag
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "elastic_electron_reference_benchmark.json", report)
    _write_elastic_tsv(report, args.output_dir / "elastic_electron_reference_benchmark.tsv")
    if not args.no_plots:
        _plot_elastic_benchmark(report, args.output_dir, args.dataset_tag)
    print(f"Wrote elastic electron reference benchmark to {args.output_dir}")


def build_eppi0_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare electron correction prescriptions on one fixed, common-support "
            "ep-pi0 cohort."
        )
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--exclusivity-cuts", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--selection-mask", type=Path)
    parser.add_argument("--selection-mask-key", default="mask")
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--min-electron-p", type=float, default=1.0)
    parser.add_argument("--min-q2", type=float, default=1.0)
    parser.add_argument("--min-w", type=float, default=2.0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def eppi0_main(argv: Iterable[str] | None = None) -> None:
    from eppi0.binning import from_config
    from eppi0.exclusivity import load_cuts

    args = build_eppi0_parser().parse_args(argv)
    parameters = _read_json(args.parameters)
    if not np.isclose(
        float(parameters["beamEnergyGeV"]), 6.535, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("the published reference benchmark is defined at 6.535 GeV")
    arrays = load_eppi0_arrays(args.input_file, args.tree, args.max_rows)
    entries = int(np.asarray(arrays["electronP"]).size)
    external = None
    if args.selection_mask is not None:
        external = load_aligned_selection_mask(
            args.selection_mask, entries, args.selection_mask_key,
            allow_prefix=args.max_rows is not None,
        )
    report = _benchmark_eppi0(
        arrays,
        parameters,
        load_cuts(str(args.exclusivity_cuts)),
        from_config(args.analysis_config),
        external_selection_mask=external,
        minimum_electron_p=args.min_electron_p,
        minimum_q2=args.min_q2,
        minimum_w=args.min_w,
    )
    report.update({
        "datasetTag": args.dataset_tag,
        "inputFile": str(args.input_file),
        "parameterFile": str(args.parameters),
        "selectionMask": str(args.selection_mask) if args.selection_mask else None,
        "exclusivityCutsFile": str(args.exclusivity_cuts),
        "exclusivityCuts": summarize_exclusivity_cuts(args.exclusivity_cuts),
        "analysisConfig": str(args.analysis_config),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "eppi0_electron_reference_benchmark.json", report)
    _write_eppi0_comparison_tsv(
        report, args.output_dir / "eppi0_electron_reference_benchmark.tsv"
    )
    for method, method_report in report["methods"].items():
        _write_json(args.output_dir / f"{method}_paired_validation.json", method_report)
        _write_observable_tsv(
            method_report, args.output_dir / f"{method}_observables.tsv"
        )
        _write_migration_tsv(
            method_report, args.output_dir / f"{method}_migration.tsv"
        )
    if not args.no_plots:
        _plot_eppi0_benchmark(
            report, args.output_dir, args.dataset_tag
        )
    print(f"Wrote ep-pi0 electron reference benchmark to {args.output_dir}")
