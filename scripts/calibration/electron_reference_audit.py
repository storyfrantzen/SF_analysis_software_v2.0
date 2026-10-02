from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .electron_reference_benchmark import (
    REFERENCE_METADATA,
    _fit_config_from_parameters,
    sector_continuous_reference_phi,
    unfold_reference_phi,
    yijie_josh_rgk_6535_delta_p,
)
from .elastic_momentum import (
    ELECTRON_MASS_GEV,
    PROTON_MASS_GEV,
    RAD_TO_DEG,
    ElasticFitConfig,
    select_calibration_events,
)
from .elastic_run_validation import (
    _core_summary,
    filter_arrays_by_run_classes,
    load_candidate_inputs,
    load_run_catalog,
    subset_arrays,
)
from .eppi0_momentum_validation import (
    _json_safe,
    apply_supported_electron_correction,
)
from .plot_utils import save_plot


REFERENCE_THETA_BINS_DEG = (
    (6.0, 7.0),
    (7.0, 8.0),
    (8.0, 9.0),
    (9.0, 10.0),
    (10.0, 11.0),
    (11.0, 13.0),
    (13.0, 15.0),
    (15.0, 25.0),
)

METHOD_LABELS = {
    "none": "none",
    "oursSupported": "ours (support-gated)",
    "referenceSectorContinuous": "reference: sector-continuous global phi",
    "referenceSigned": "reference control: raw signed phi",
    "referenceGlobal": "reference: global unfolded phi",
    "referenceSectorLocal": "reference: sector-local phi",
}

METHOD_COLORS = {
    "none": "black",
    "oursSupported": "#21618c",
    "referenceSectorContinuous": "#239b56",
    "referenceSigned": "#7d3c98",
    "referenceGlobal": "#c0392b",
    "referenceSectorLocal": "#d68910",
}


def electron_w(
    momentum_gev: np.ndarray,
    theta_rad: np.ndarray,
    beam_energy_gev: float,
) -> np.ndarray:
    """Reconstruct inclusive hadronic mass from the scattered electron."""
    momentum = np.asarray(momentum_gev, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    energy = np.sqrt(np.square(momentum) + ELECTRON_MASS_GEV**2)
    q2 = (
        2.0 * beam_energy_gev * (energy - momentum * np.cos(theta))
        - ELECTRON_MASS_GEV**2
    )
    w2 = (
        PROTON_MASS_GEV**2
        + 2.0 * PROTON_MASS_GEV * (beam_energy_gev - energy)
        - q2
    )
    return np.where(w2 >= 0.0, np.sqrt(np.maximum(w2, 0.0)), np.nan)


def correction_prescriptions(
    arrays: dict[str, np.ndarray],
    parameters: dict[str, object],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Build all four momentum prescriptions on one selected sample."""
    momentum = np.asarray(arrays["electronP"], dtype=float)
    theta = np.asarray(arrays["electronTheta"], dtype=float)
    phi = np.asarray(arrays["electronPhi"], dtype=float)
    sector = np.asarray(arrays["electronSector"], dtype=int)
    ours, support, ours_fraction, _ = apply_supported_electron_correction(
        arrays, parameters
    )
    sector_continuous_delta = yijie_josh_rgk_6535_delta_p(
        theta, phi, sector, phi_convention="sector-continuous-global"
    )
    signed_delta = yijie_josh_rgk_6535_delta_p(
        theta, phi, sector, phi_convention="signed-global"
    )
    global_delta = yijie_josh_rgk_6535_delta_p(
        theta, phi, sector, phi_convention="global-unfolded"
    )
    local_delta = yijie_josh_rgk_6535_delta_p(
        theta, phi, sector, phi_convention="sector-local"
    )
    momenta = {
        "none": momentum.copy(),
        "oursSupported": ours,
        "referenceSectorContinuous": momentum + sector_continuous_delta,
        "referenceSigned": momentum + signed_delta,
        "referenceGlobal": momentum + global_delta,
        "referenceSectorLocal": momentum + local_delta,
    }
    for method, corrected in momenta.items():
        if np.any(~np.isfinite(corrected) | (corrected <= 0.0)):
            raise ValueError(f"{method} produced a non-positive electron momentum")
    fractions = {
        "none": np.zeros(momentum.shape, dtype=float),
        "oursSupported": ours_fraction,
        "referenceSectorContinuous": sector_continuous_delta / momentum,
        "referenceSigned": signed_delta / momentum,
        "referenceGlobal": global_delta / momentum,
        "referenceSectorLocal": local_delta / momentum,
    }
    return momenta, support, fractions


def sector_mapping_candidates() -> list[dict[str, object]]:
    """Return every cyclic/reflected mapping of observed to table sectors."""
    observed = np.arange(6, dtype=int)
    candidates: list[dict[str, object]] = []
    for offset in range(6):
        mapping = (observed + offset) % 6 + 1
        candidates.append({
            "name": "identity" if offset == 0 else f"rotate_plus_{offset}",
            "family": "rotation",
            "offset": offset,
            "coefficientSectorByObservedSector": mapping.tolist(),
        })
    for offset in range(6):
        mapping = (offset - observed) % 6 + 1
        candidates.append({
            "name": f"reflection_{offset}",
            "family": "reflection",
            "offset": offset,
            "coefficientSectorByObservedSector": mapping.tolist(),
        })
    return candidates


def _summary(values: np.ndarray) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "entries": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "entries": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _theta_mask(theta_deg: np.ndarray, low: float, high: float) -> np.ndarray:
    upper = (
        theta_deg <= high
        if high == REFERENCE_THETA_BINS_DEG[-1][1]
        else theta_deg < high
    )
    return (theta_deg >= low) & upper


def _theta_bin_indices(theta_deg: np.ndarray) -> np.ndarray:
    indices = np.full(theta_deg.shape, -1, dtype=np.int16)
    for index, (low, high) in enumerate(REFERENCE_THETA_BINS_DEG):
        indices[_theta_mask(theta_deg, low, high)] = index
    return indices


def _grouped_indices(
    group_ids: np.ndarray,
    selection: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    """Return selected row indices grouped by integer ID with one stable sort."""
    indices = np.flatnonzero(np.asarray(selection, dtype=bool) & (group_ids >= 0))
    if not indices.size:
        return []
    order = np.argsort(group_ids[indices], kind="stable")
    indices = indices[order]
    sorted_groups = group_ids[indices]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_groups))]
    stops = np.r_[starts[1:], indices.size]
    return [
        (int(sorted_groups[start]), indices[start:stop])
        for start, stop in zip(starts, stops)
    ]


def _profile_rows(
    *,
    sample: str,
    theta_deg: np.ndarray,
    global_phi_deg: np.ndarray,
    sectors: np.ndarray,
    run_numbers: np.ndarray,
    run_class_by_run: dict[int, str] | None,
    w_by_method: dict[str, np.ndarray],
    support: np.ndarray,
    cfg: ElasticFitConfig,
    phi_cell_width_deg: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    phi_cells = int(round(360.0 / phi_cell_width_deg))
    if not np.isclose(phi_cells * phi_cell_width_deg, 360.0):
        raise ValueError("phi cell width must divide 360 degrees")
    phi_edges = -25.0 + np.arange(phi_cells + 1) * phi_cell_width_deg
    theta_indices = _theta_bin_indices(theta_deg)
    phi_indices = np.floor(
        (np.asarray(global_phi_deg) + 25.0) / phi_cell_width_deg
    ).astype(np.int16)
    valid = (
        (theta_indices >= 0)
        & (sectors >= 1)
        & (sectors <= 6)
        & (phi_indices >= 0)
        & (phi_indices < phi_cells)
    )
    group_ids = np.full(theta_deg.shape, -1, dtype=np.int32)
    group_ids[valid] = (
        (theta_indices[valid].astype(np.int32) * 6 + sectors[valid] - 1)
        * phi_cells
        + phi_indices[valid]
    )
    cohorts = {
        "allSelected": np.ones(theta_deg.shape, dtype=bool),
        "commonSupport": np.asarray(support, dtype=bool),
    }
    run_groups: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(theta_deg.shape, dtype=bool))
    ]
    if run_class_by_run:
        classes = sorted(set(run_class_by_run.values()))
        classified = np.zeros(theta_deg.shape, dtype=bool)
        for run_class in classes:
            class_runs = [
                run for run, value in run_class_by_run.items()
                if value == run_class
            ]
            class_mask = np.isin(run_numbers, class_runs)
            classified |= class_mask
            run_groups.append((str(run_class), class_mask))
        if np.any(~classified):
            run_groups.append(("unclassified", ~classified))

    for run_class, run_mask in run_groups:
        for cohort, cohort_mask in cohorts.items():
            grouped = _grouped_indices(group_ids, run_mask & cohort_mask)
            for group_id, indices in grouped:
                phi_index = group_id % phi_cells
                theta_sector = group_id // phi_cells
                sector = theta_sector % 6 + 1
                theta_index = theta_sector // 6
                theta_low, theta_high = REFERENCE_THETA_BINS_DEG[theta_index]
                phi_low = phi_edges[phi_index]
                phi_high = phi_edges[phi_index + 1]
                for method, w_values in w_by_method.items():
                    estimate = _core_summary(
                        np.asarray(w_values)[indices] - PROTON_MASS_GEV, cfg
                    )
                    rows.append({
                        "sample": sample,
                        "runClass": run_class,
                        "cohort": cohort,
                        "method": method,
                        "thetaLowDeg": theta_low,
                        "thetaHighDeg": theta_high,
                        "sector": sector,
                        "phiLowDeg": float(phi_low),
                        "phiHighDeg": float(phi_high),
                        "phiCenterDeg": float(0.5 * (phi_low + phi_high)),
                        "rawEntries": int(indices.size),
                        "coreEntries": (
                            int(estimate["coreEntries"]) if estimate else 0
                        ),
                        "wCenterGeV": (
                            float(PROTON_MASS_GEV + estimate["center"])
                            if estimate else None
                        ),
                        "wCenterErrorGeV": (
                            float(estimate["centerError"]) if estimate else None
                        ),
                        "wCoreWidthGeV": (
                            float(estimate["coreWidth"]) if estimate else None
                        ),
                        "centerOffsetMeV": (
                            float(1000.0 * estimate["center"])
                            if estimate else None
                        ),
                    })
    return rows


def _aggregate_profile_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["sample"], row["runClass"], row["cohort"], row["method"],
            row["thetaLowDeg"], row["thetaHighDeg"],
        )
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, object]] = []
    for key, cells in grouped.items():
        offsets = np.asarray([
            float(cell["centerOffsetMeV"])
            for cell in cells if cell["centerOffsetMeV"] is not None
        ])
        result.append({
            "sample": key[0],
            "runClass": key[1],
            "cohort": key[2],
            "method": key[3],
            "thetaLowDeg": key[4],
            "thetaHighDeg": key[5],
            "attemptedPhiCells": len(cells),
            "validatedPhiCells": int(offsets.size),
            "validatedPhiCellFraction": float(offsets.size / len(cells)),
            "profileEntries": int(sum(int(cell["rawEntries"]) for cell in cells)),
            "phiCellMeanOffsetMeV": (
                float(np.mean(offsets)) if offsets.size else None
            ),
            "phiCellCenterSpreadMeV": (
                float(np.std(offsets, ddof=0)) if offsets.size else None
            ),
            "phiCellCenterRmsMeV": (
                float(np.sqrt(np.mean(np.square(offsets))))
                if offsets.size else None
            ),
            "medianAbsPhiCellCenterMeV": (
                float(np.median(np.abs(offsets))) if offsets.size else None
            ),
            "maxAbsPhiCellCenterMeV": (
                float(np.max(np.abs(offsets))) if offsets.size else None
            ),
        })
    return sorted(
        result,
        key=lambda row: (
            str(row["sample"]), str(row["runClass"]), str(row["cohort"]),
            float(row["thetaLowDeg"]), str(row["method"]),
        ),
    )


def _overall_profile_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["sample"], row["runClass"], row["cohort"], row["method"],
        )
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, object]] = []
    for key, cells in grouped.items():
        offsets = np.asarray([
            float(cell["centerOffsetMeV"])
            for cell in cells if cell["centerOffsetMeV"] is not None
        ])
        result.append({
            "sample": key[0],
            "runClass": key[1],
            "cohort": key[2],
            "method": key[3],
            "attemptedPhiCells": len(cells),
            "validatedPhiCells": int(offsets.size),
            "validatedPhiCellFraction": float(offsets.size / len(cells)),
            "phiCellMeanOffsetMeV": (
                float(np.mean(offsets)) if offsets.size else None
            ),
            "phiCellCenterSpreadMeV": (
                float(np.std(offsets, ddof=0)) if offsets.size else None
            ),
            "phiCellCenterRmsMeV": (
                float(np.sqrt(np.mean(np.square(offsets))))
                if offsets.size else None
            ),
            "medianAbsPhiCellCenterMeV": (
                float(np.median(np.abs(offsets))) if offsets.size else None
            ),
            "maxAbsPhiCellCenterMeV": (
                float(np.max(np.abs(offsets))) if offsets.size else None
            ),
        })
    return sorted(
        result,
        key=lambda row: (
            str(row["sample"]), str(row["runClass"]),
            str(row["cohort"]), str(row["method"]),
        ),
    )


def _sector_mapping_audit(
    *,
    sample: str,
    arrays: dict[str, np.ndarray],
    theta_deg: np.ndarray,
    global_phi_deg: np.ndarray,
    sectors: np.ndarray,
    support: np.ndarray,
    cfg: ElasticFitConfig,
    phi_cell_width_deg: float,
) -> list[dict[str, object]]:
    """Score all detector-symmetry sector relabelings with signed global phi."""
    momentum = np.asarray(arrays["electronP"], dtype=float)
    theta_rad = np.asarray(arrays["electronTheta"], dtype=float)
    phi_rad = np.asarray(arrays["electronPhi"], dtype=float)
    run_numbers = np.asarray(
        arrays.get("runNum", np.zeros(momentum.shape)), dtype=int
    )
    reference_phi = sector_continuous_reference_phi(phi_rad, sectors)
    rows: list[dict[str, object]] = []
    for candidate in sector_mapping_candidates():
        print(
            f"[SECTOR MAP] {sample}: {candidate['name']}", flush=True
        )
        mapping = np.asarray(
            candidate["coefficientSectorByObservedSector"], dtype=int
        )
        coefficient_sector = mapping[sectors - 1]
        delta = yijie_josh_rgk_6535_delta_p(
            theta_rad,
            phi_rad,
            sectors,
            phi_convention="sector-continuous-global",
            coefficient_sector=coefficient_sector,
        )
        corrected = momentum + delta
        if np.any(~np.isfinite(corrected) | (corrected <= 0.0)):
            raise ValueError(
                f"sector mapping {candidate['name']} produced non-positive momentum"
            )
        method = str(candidate["name"])
        profile_rows = _profile_rows(
            sample=sample,
            theta_deg=theta_deg,
            global_phi_deg=global_phi_deg,
            sectors=sectors,
            run_numbers=run_numbers,
            run_class_by_run=None,
            w_by_method={
                method: electron_w(corrected, theta_rad, cfg.beam_energy)
            },
            support=support,
            cfg=cfg,
            phi_cell_width_deg=phi_cell_width_deg,
        )
        summaries = {
            str(item["cohort"]): item
            for item in _overall_profile_rows(profile_rows)
            if item["runClass"] == "all"
        }
        row = {
            "sample": sample,
            **candidate,
            "referencePhiMinimumDeg": float(np.min(reference_phi)),
            "referencePhiMaximumDeg": float(np.max(reference_phi)),
        }
        for cohort in ("allSelected", "commonSupport"):
            summary = summaries.get(cohort, {})
            prefix = "allSelected" if cohort == "allSelected" else "commonSupport"
            row[f"{prefix}ValidatedPhiCells"] = summary.get(
                "validatedPhiCells"
            )
            row[f"{prefix}PhiCellCenterRmsMeV"] = summary.get(
                "phiCellCenterRmsMeV"
            )
            row[f"{prefix}MedianAbsPhiCellCenterMeV"] = summary.get(
                "medianAbsPhiCellCenterMeV"
            )
            row[f"{prefix}MaxAbsPhiCellCenterMeV"] = summary.get(
                "maxAbsPhiCellCenterMeV"
            )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["commonSupportPhiCellCenterRmsMeV"] is None,
            row["commonSupportPhiCellCenterRmsMeV"]
            if row["commonSupportPhiCellCenterRmsMeV"] is not None
            else np.inf,
        )
    )
    for rank, row in enumerate(rows, 1):
        row["rankByCommonSupportRms"] = rank
    return rows


def _w_summary(values: np.ndarray, cfg: ElasticFitConfig) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "entries": 0,
            "meanOffsetMeV": None,
            "rmsFromProtonMassMeV": None,
            "withinPlotRangeFraction": None,
            "core": None,
        }
    offsets = finite - PROTON_MASS_GEV
    core = _core_summary(offsets, cfg)
    return {
        "entries": int(finite.size),
        "meanOffsetMeV": float(1000.0 * np.mean(offsets)),
        "rmsFromProtonMassMeV": float(
            1000.0 * np.sqrt(np.mean(np.square(offsets)))
        ),
        "withinPlotRangeFraction": float(
            np.mean((finite >= 0.80) & (finite <= 1.10))
        ),
        "core": (
            {
                "centerOffsetMeV": float(1000.0 * core["center"]),
                "widthMeV": float(1000.0 * core["coreWidth"]),
                "entries": int(core["coreEntries"]),
            }
            if core else None
        ),
    }


def _core_rows(
    *,
    sample: str,
    theta_deg: np.ndarray,
    sectors: np.ndarray,
    w_by_method: dict[str, np.ndarray],
    support: np.ndarray,
    cfg: ElasticFitConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    theta_indices = _theta_bin_indices(theta_deg)
    valid = (theta_indices >= 0) & (sectors >= 1) & (sectors <= 6)
    group_ids = np.full(theta_deg.shape, -1, dtype=np.int16)
    group_ids[valid] = theta_indices[valid] * 6 + sectors[valid] - 1
    for cohort, cohort_mask in (
        ("allSelected", np.ones(theta_deg.shape, dtype=bool)),
        ("commonSupport", np.asarray(support, dtype=bool)),
    ):
        grouped = dict(_grouped_indices(group_ids, cohort_mask))
        for theta_index, (theta_low, theta_high) in enumerate(
            REFERENCE_THETA_BINS_DEG
        ):
            for method, w_values in w_by_method.items():
                for sector in range(1, 7):
                    indices = grouped.get(
                        theta_index * 6 + sector - 1,
                        np.asarray([], dtype=int),
                    )
                    estimate = _core_summary(
                        np.asarray(w_values)[indices] - PROTON_MASS_GEV, cfg
                    )
                    rows.append({
                        "sample": sample,
                        "cohort": cohort,
                        "method": method,
                        "thetaLowDeg": theta_low,
                        "thetaHighDeg": theta_high,
                        "sector": sector,
                        "rawEntries": int(indices.size),
                        "coreEntries": (
                            int(estimate["coreEntries"]) if estimate else 0
                        ),
                        "wCenterGeV": (
                            float(PROTON_MASS_GEV + estimate["center"])
                            if estimate else None
                        ),
                        "centerOffsetMeV": (
                            float(1000.0 * estimate["center"])
                            if estimate else None
                        ),
                        "wCoreWidthMeV": (
                            float(1000.0 * estimate["coreWidth"])
                            if estimate else None
                        ),
                    })
    return rows


def run_sample_audit(
    arrays: dict[str, np.ndarray],
    parameters: dict[str, object],
    cfg: ElasticFitConfig,
    *,
    sample: str,
    run_class_by_run: dict[int, str] | None,
    phi_cell_width_deg: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    selected, selection = select_calibration_events(arrays, cfg)
    if not selected["electronP"].size:
        raise ValueError(f"{sample} selection retained no events")
    momenta, support, fractions = correction_prescriptions(selected, parameters)
    theta_rad = np.asarray(selected["electronTheta"], dtype=float)
    theta_deg = theta_rad * RAD_TO_DEG
    global_phi_deg = unfold_reference_phi(selected["electronPhi"])
    sectors = np.asarray(selected["electronSector"], dtype=int)
    runs = np.asarray(selected.get("runNum", np.zeros(theta_deg.shape)), dtype=int)
    w_by_method = {
        method: electron_w(momentum, theta_rad, cfg.beam_energy)
        for method, momentum in momenta.items()
    }
    profile_rows = _profile_rows(
        sample=sample,
        theta_deg=theta_deg,
        global_phi_deg=global_phi_deg,
        sectors=sectors,
        run_numbers=runs,
        run_class_by_run=run_class_by_run,
        w_by_method=w_by_method,
        support=support,
        cfg=cfg,
        phi_cell_width_deg=phi_cell_width_deg,
    )
    aggregate_rows = _aggregate_profile_rows(profile_rows)
    core_rows = _core_rows(
        sample=sample,
        theta_deg=theta_deg,
        sectors=sectors,
        w_by_method=w_by_method,
        support=support,
        cfg=cfg,
    )
    correction_summary: dict[str, object] = {}
    w_summary: dict[str, object] = {}
    for method, values in fractions.items():
        correction_summary[method] = {
            "allSelected": _summary(values),
            "commonSupport": _summary(np.asarray(values)[support]),
            "sectors": {
                str(sector): _summary(np.asarray(values)[sectors == sector])
                for sector in range(1, 7)
            },
        }
        w_values = np.asarray(w_by_method[method])
        w_summary[method] = {
            "allSelected": _w_summary(w_values, cfg),
            "commonSupport": _w_summary(w_values[support], cfg),
        }
    report = {
        "sample": sample,
        "selection": selection,
        "selectedEntries": int(theta_deg.size),
        "commonSupportEntries": int(np.count_nonzero(support)),
        "commonSupportFraction": float(np.mean(support)),
        "correctionSummary": correction_summary,
        "wSummary": w_summary,
        "profileSummary": aggregate_rows,
        "overallProfileSummary": _overall_profile_rows(profile_rows),
        "sectorThetaCore": core_rows,
        "sectorMappingAudit": _sector_mapping_audit(
            sample=sample,
            arrays=selected,
            theta_deg=theta_deg,
            global_phi_deg=global_phi_deg,
            sectors=sectors,
            support=support,
            cfg=cfg,
            phi_cell_width_deg=phi_cell_width_deg,
        ),
    }
    diagnostics = {
        "thetaDeg": theta_deg,
        "globalPhiDeg": global_phi_deg,
        "sector": sectors,
        "support": support,
        **{f"w_{method}": values for method, values in w_by_method.items()},
        "profileRows": np.asarray(profile_rows, dtype=object),
    }
    return report, diagnostics


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _theta_slug(low: float, high: float) -> str:
    return f"theta_{low:g}_{high:g}".replace(".", "p")


def _plot_heatmaps(
    sample: str,
    diagnostics: dict[str, np.ndarray],
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    theta = diagnostics["thetaDeg"]
    phi = diagnostics["globalPhiDeg"]
    methods = tuple(METHOD_LABELS)
    phi_edges = np.linspace(-25.0, 335.0, 181)
    w_edges = np.linspace(0.80, 1.10, 151)
    for theta_low, theta_high in REFERENCE_THETA_BINS_DEG:
        mask = _theta_mask(theta, theta_low, theta_high)
        histograms = [
            np.histogram2d(
                phi[mask], diagnostics[f"w_{method}"][mask],
                bins=(phi_edges, w_edges),
            )[0]
            for method in methods
        ]
        maximum = max(float(np.max(histogram)) for histogram in histograms)
        vmax = np.log10(maximum + 1.0) if maximum > 0.0 else 1.0
        fig = plt.figure(figsize=(4.4 * len(methods) + 1.0, 4.6))
        grid = fig.add_gridspec(
            1,
            len(methods) + 1,
            width_ratios=tuple([1.0] * len(methods) + [0.045]),
            wspace=0.30,
        )
        axes = np.asarray([
            fig.add_subplot(grid[0, index]) for index in range(len(methods))
        ])
        for axis in axes[1:]:
            axis.sharex(axes[0])
            axis.sharey(axes[0])
            axis.tick_params(labelleft=False)
        color_axis = fig.add_subplot(grid[0, len(methods)])
        mesh = None
        for axis, method, histogram in zip(axes, methods, histograms):
            mesh = axis.pcolormesh(
                phi_edges, w_edges, np.log10(histogram.T + 1.0),
                shading="auto", vmin=0.0, vmax=vmax, cmap="viridis",
            )
            axis.axhline(PROTON_MASS_GEV, color="red", linewidth=0.9)
            for boundary in (30.0, 90.0, 150.0, 210.0, 270.0, 330.0):
                axis.axvline(boundary, color="white", linewidth=0.35, alpha=0.5)
            axis.set_title(METHOD_LABELS[method])
            axis.set_xlabel(r"global unfolded $\phi_e$ [deg]")
            axis.set_xlim(-25.0, 335.0)
            axis.set_ylim(0.80, 1.10)
        axes[0].set_ylabel(r"electron-only $W$ [GeV]")
        if mesh is not None:
            fig.colorbar(
                mesh, cax=color_axis, label=r"$\log_{10}(N+1)$"
            )
        save_plot(
            fig,
            output_dir / f"{sample}_{_theta_slug(theta_low, theta_high)}_appendix_b.png",
            (
                f"{sample}: Appendix-B-style elastic comparison, "
                f"{theta_low:g} < theta_e < {theta_high:g} deg"
            ),
            dataset_tag,
            beam_energy,
            tight_layout=False,
        )
        plt.close(fig)


def _plot_profiles(
    sample: str,
    profile_rows: list[dict[str, object]],
    cohort: str,
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row for row in profile_rows
        if row["sample"] == sample
        and row["runClass"] == "all"
        and row["cohort"] == cohort
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True, sharey=True)
    for axis, (theta_low, theta_high) in zip(axes.flat, REFERENCE_THETA_BINS_DEG):
        for method in METHOD_LABELS:
            selected = [
                row for row in rows
                if row["method"] == method
                and row["thetaLowDeg"] == theta_low
                and row["thetaHighDeg"] == theta_high
                and row["centerOffsetMeV"] is not None
            ]
            selected.sort(key=lambda row: float(row["phiCenterDeg"]))
            if not selected:
                continue
            axis.plot(
                [row["phiCenterDeg"] for row in selected],
                [row["centerOffsetMeV"] for row in selected],
                ".-", markersize=3.5, linewidth=0.8,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method],
            )
        axis.axhline(0.0, color="gray", linewidth=0.7)
        for boundary in (30.0, 90.0, 150.0, 210.0, 270.0, 330.0):
            axis.axvline(boundary, color="gray", linewidth=0.35, alpha=0.45)
        axis.set_title(f"{theta_low:g} < theta_e < {theta_high:g} deg")
        axis.grid(alpha=0.18)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"global unfolded $\phi_e$ [deg]")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"fitted $W-M_p$ [MeV]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.875),
            ncol=min(5, len(METHOD_LABELS)),
            frameon=True,
        )
    save_plot(
        fig,
        output_dir / f"{sample}_{cohort}_w_peak_profiles.png",
        f"{sample}: elastic W peak versus phi ({cohort})",
        dataset_tag,
        beam_energy,
        tight_layout=False,
    )
    plt.close(fig)


def _load_and_filter(
    paths: list[Path],
    tree: str,
    *,
    require_proton: bool,
    run_catalog: dict[int, str] | None,
    run_catalog_path: Path | None,
    include_run_classes: list[str] | None,
    max_rows: int | None,
) -> tuple[dict[str, np.ndarray], dict[int, str] | None, dict[str, object] | None]:
    arrays = load_candidate_inputs(paths, tree, require_proton=require_proton)
    if max_rows is not None:
        entries = next(iter(arrays.values())).size
        arrays = subset_arrays(arrays, np.arange(entries) < max_rows)
    if run_catalog is None:
        return arrays, None, None
    return filter_arrays_by_run_classes(
        arrays,
        run_catalog,
        include_run_classes,
        catalog_path=run_catalog_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the RGK 6.535-GeV Appendix-B elastic W-versus-phi check "
            "for no correction, the local correction, the sector-continuous "
            "global-phi convention implied by the table, and three convention "
            "controls. Also scan cyclic/reflected sector labels."
        )
    )
    parser.add_argument("--inclusive-input", nargs="+", type=Path, required=True)
    parser.add_argument("--exclusive-input", nargs="+", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--run-catalog", type=Path)
    parser.add_argument("--include-run-classes", nargs="+")
    parser.add_argument("--phi-cell-width-deg", type=float, default=5.0)
    parser.add_argument("--min-cell-entries", type=int, default=100)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.phi_cell_width_deg <= 0.0 or args.phi_cell_width_deg > 30.0:
        raise ValueError("--phi-cell-width-deg must lie in (0, 30]")
    if args.min_cell_entries < 2:
        raise ValueError("--min-cell-entries must be at least 2")
    parameters = json.loads(args.parameters.read_text())
    beam_energy = float(parameters["beamEnergyGeV"])
    if not np.isclose(beam_energy, 6.535, rtol=0.0, atol=1.0e-9):
        raise ValueError("the reference audit is defined at 6.535 GeV")
    base_cfg = _fit_config_from_parameters(parameters)
    profile_cfg = replace(base_cfg, min_bin_entries=args.min_cell_entries)
    catalog = load_run_catalog(args.run_catalog) if args.run_catalog else None

    sample_specs = (
        ("inclusiveW", args.inclusive_input, False, "inclusive-w"),
        ("exclusiveEp", args.exclusive_input, True, "exclusive-ep"),
    )
    reports: dict[str, object] = {}
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    input_metadata: dict[str, object] = {}
    all_profile_rows: list[dict[str, object]] = []
    all_overall_profile_rows: list[dict[str, object]] = []
    all_core_rows: list[dict[str, object]] = []
    all_sector_mapping_rows: list[dict[str, object]] = []
    for sample, paths, require_proton, selection_mode in sample_specs:
        print(
            f"[LOAD] {sample}: {', '.join(str(path) for path in paths)}",
            flush=True,
        )
        arrays, mapping, run_selection = _load_and_filter(
            paths,
            args.tree,
            require_proton=require_proton,
            run_catalog=catalog,
            run_catalog_path=args.run_catalog,
            include_run_classes=args.include_run_classes,
            max_rows=args.max_rows,
        )
        report, diagnostic = run_sample_audit(
            arrays,
            parameters,
            replace(profile_cfg, electron_selection=selection_mode),
            sample=sample,
            run_class_by_run=mapping,
            phi_cell_width_deg=args.phi_cell_width_deg,
        )
        reports[sample] = report
        diagnostics[sample] = diagnostic
        input_metadata[sample] = {
            "files": [str(path) for path in paths],
            "runSelection": run_selection,
        }
        all_profile_rows.extend(report["profileSummary"])
        all_overall_profile_rows.extend(report["overallProfileSummary"])
        all_core_rows.extend(report["sectorThetaCore"])
        all_sector_mapping_rows.extend(report["sectorMappingAudit"])
        best_mapping = report["sectorMappingAudit"][0]
        print(
            f"[AUDIT] {sample}: selected={report['selectedEntries']}; "
            f"common support={100.0 * report['commonSupportFraction']:.2f}%; "
            f"best sector mapping={best_mapping['name']}",
            flush=True,
        )

    output = {
        "schema": "rgk-electron-reference-focused-audit/v2",
        "beamEnergyGeV": beam_energy,
        "reference": REFERENCE_METADATA,
        "referenceThetaBinsDeg": [list(values) for values in REFERENCE_THETA_BINS_DEG],
        "phiCellWidthDeg": args.phi_cell_width_deg,
        "minimumCellEntries": args.min_cell_entries,
        "methods": METHOD_LABELS,
        "inputs": input_metadata,
        "parameterFile": str(args.parameters),
        "samples": reports,
        "datasetTag": args.dataset_tag,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "electron_reference_focused_audit.json").write_text(
        json.dumps(_json_safe(output), indent=2) + "\n"
    )
    _write_rows(args.output_dir / "phi_profile_summary.tsv", all_profile_rows)
    _write_rows(
        args.output_dir / "overall_phi_profile_summary.tsv",
        all_overall_profile_rows,
    )
    _write_rows(args.output_dir / "sector_theta_core.tsv", all_core_rows)
    _write_rows(
        args.output_dir / "sector_mapping_scan.tsv", all_sector_mapping_rows
    )
    for sample, diagnostic in diagnostics.items():
        profile_rows = list(diagnostic["profileRows"])
        _write_rows(args.output_dir / f"{sample}_phi_cells.tsv", profile_rows)
        if not args.no_plots:
            print(f"[PLOT] {sample}: Appendix-B heatmaps", flush=True)
            _plot_heatmaps(
                sample, diagnostic, args.output_dir, args.dataset_tag, beam_energy
            )
            for cohort in ("allSelected", "commonSupport"):
                print(f"[PLOT] {sample}: {cohort} peak profiles", flush=True)
                _plot_profiles(
                    sample,
                    profile_rows,
                    cohort,
                    args.output_dir,
                    args.dataset_tag,
                    beam_energy,
                )
    print(f"Wrote focused electron-reference audit to {args.output_dir}")
    for sample, report in reports.items():
        print(
            f"  {sample}: selected={report['selectedEntries']}; "
            f"common support={100.0 * report['commonSupportFraction']:.2f}%"
        )
