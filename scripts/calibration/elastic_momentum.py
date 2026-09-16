from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .plot_utils import save_plot
from .root_arrays import arrays_from_dataframe, has_column, load_dataframe


PROTON_MASS_GEV = 0.9382720813
ELECTRON_MASS_GEV = 0.00051099895
RAD_TO_DEG = 180.0 / np.pi


@dataclass(frozen=True)
class ElasticFitConfig:
    beam_energy: float
    torus: int = 0
    coplanarity_max_deg: float = 3.0
    theta_balance_max_deg: float = 2.0
    missing_energy_max_gev: float = 0.75
    theta_min_deg: float | None = None
    theta_max_deg: float | None = None
    theta_trim_quantile: float = 0.005
    residual_trim_quantile: float = 0.01
    theta_bins: int = 7
    phi_bins: int = 7
    profile_binning: str = "fixed"
    max_theta_bin_width_deg: float | None = None
    target_cell_entries: int = 500
    min_phi_cells_per_theta: int = 1
    theta_order: int = 2
    phi_order: int = 2
    cd_fourier_harmonics: int = 3
    min_bin_entries: int = 40
    min_region_entries: int = 800
    peak_search_max_abs_residual: float = 0.10
    peak_seed_half_width: float = 0.03
    core_sigma_clip: float = 3.0
    min_core_fraction: float = 0.20
    min_peak_significance: float = 3.0
    max_core_width: float = 0.05
    min_profile_cells_per_parameter: float = 2.0
    max_condition_number: float = 100.0
    max_abs_surface_correction: float = 0.05


@dataclass(frozen=True)
class RegionDiagnostics:
    label: str
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    residual_before: np.ndarray
    residual_after: np.ndarray
    profile_theta_deg: np.ndarray
    profile_phi_deg: np.ndarray
    profile_residual: np.ndarray
    profile_error: np.ndarray
    profile_width: np.ndarray
    profile_entries: np.ndarray
    profile_core_fraction: np.ndarray
    profile_peak_significance: np.ndarray
    region: dict[str, object]


@dataclass(frozen=True)
class CoreEstimate:
    center: float
    error: float
    width: float
    entries: int
    total_entries: int
    mode: float
    retained_fraction: float
    peak_significance: float
    converged: bool
    mask: np.ndarray


@dataclass(frozen=True)
class ProfileGrid:
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    residual: np.ndarray
    error: np.ndarray
    width: np.ndarray
    entries: np.ndarray
    total_entries: np.ndarray
    core_fraction: np.ndarray
    peak_significance: np.ndarray
    support_cells: list[dict[str, object]]
    rejected_cells: list[dict[str, object]]


@dataclass(frozen=True)
class ProfileCellPlan:
    cells: list[dict[str, object]]
    metadata: dict[str, object]


def wrap_degrees(angle_deg: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle_deg, dtype=float)
    return (angle + 180.0) % 360.0 - 180.0


def sector_local_phi(phi_rad: np.ndarray, sector: np.ndarray) -> np.ndarray:
    return wrap_degrees(np.asarray(phi_rad) * RAD_TO_DEG - 60.0 * (np.asarray(sector) - 1.0))


def elastic_electron_momentum(theta_rad: np.ndarray, beam_energy: float) -> np.ndarray:
    """Massless-electron elastic momentum for a stationary proton target."""
    theta = np.asarray(theta_rad, dtype=float)
    return beam_energy / (
        1.0 + beam_energy * (1.0 - np.cos(theta)) / PROTON_MASS_GEV
    )


def elastic_proton_momentum(theta_rad: np.ndarray, beam_energy: float) -> np.ndarray:
    """Elastic recoil-proton momentum using only its reconstructed polar angle."""
    theta = np.asarray(theta_rad, dtype=float)
    cosine = np.cos(theta)
    numerator = 2.0 * PROTON_MASS_GEV * beam_energy * (beam_energy + PROTON_MASS_GEV) * cosine
    denominator = (
        (beam_energy + PROTON_MASS_GEV) ** 2 - beam_energy**2 * cosine**2
    )
    return numerator / denominator


def elastic_proton_theta_from_electron(
    electron_theta_rad: np.ndarray,
    beam_energy: float,
) -> np.ndarray:
    electron_p = elastic_electron_momentum(electron_theta_rad, beam_energy)
    transverse = electron_p * np.sin(electron_theta_rad)
    longitudinal = beam_energy - electron_p * np.cos(electron_theta_rad)
    return np.arctan2(transverse, longitudinal)


def load_elastic_arrays(input_file: Path, tree: str, max_rows: int | None) -> dict[str, np.ndarray]:
    df = load_dataframe(input_file, tree)
    required = [
        "electronP", "electronTheta", "electronPhi", "electronDet", "electronSector",
        "protonP", "protonTheta", "protonPhi", "protonDet", "protonSector",
    ]
    missing = [column for column in required if not has_column(df, column)]
    if missing:
        raise RuntimeError(
            f"{tree} is missing elastic-candidate branches: {', '.join(missing)}"
        )
    optional = [name for name in ("runNum", "nPid11", "nPid2212") if has_column(df, name)]
    return arrays_from_dataframe(df, required + optional, max_rows=max_rows)


def select_elastic_events(
    arrays: dict[str, np.ndarray],
    cfg: ElasticFitConfig,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    electron_theta = np.asarray(arrays["electronTheta"], dtype=float)
    proton_theta = np.asarray(arrays["protonTheta"], dtype=float)
    expected_proton_theta = elastic_proton_theta_from_electron(
        electron_theta, cfg.beam_energy
    )
    coplanarity = wrap_degrees(
        (np.asarray(arrays["protonPhi"]) - np.asarray(arrays["electronPhi"])) *
        RAD_TO_DEG - 180.0
    )
    theta_balance = (proton_theta - expected_proton_theta) * RAD_TO_DEG
    electron_p = np.asarray(arrays["electronP"], dtype=float)
    proton_p = np.asarray(arrays["protonP"], dtype=float)
    missing_energy = (
        cfg.beam_energy + PROTON_MASS_GEV
        - np.sqrt(np.square(electron_p) + ELECTRON_MASS_GEV**2)
        - np.sqrt(np.square(proton_p) + PROTON_MASS_GEV**2)
    )

    finite_columns = [
        "electronP", "electronTheta", "electronPhi", "electronDet", "electronSector",
        "protonP", "protonTheta", "protonPhi", "protonDet", "protonSector",
    ]
    mask = np.ones(electron_theta.size, dtype=bool)
    for column in finite_columns:
        mask &= np.isfinite(arrays[column])
    mask &= electron_p > 0.0
    mask &= proton_p > 0.0
    mask &= np.isfinite(missing_energy)
    mask &= np.asarray(arrays["electronDet"]) == 1
    mask &= (np.asarray(arrays["electronSector"]) >= 1) & (
        np.asarray(arrays["electronSector"]) <= 6
    )
    mask &= np.isin(np.asarray(arrays["protonDet"]), (1, 2))
    mask &= elastic_proton_momentum(proton_theta, cfg.beam_energy) > 0.0
    if "nPid11" in arrays:
        mask &= np.asarray(arrays["nPid11"]) == 1
    if "nPid2212" in arrays:
        mask &= np.asarray(arrays["nPid2212"]) == 1
    preselection = mask.copy()
    mask &= np.abs(coplanarity) <= cfg.coplanarity_max_deg
    mask &= np.abs(theta_balance) <= cfg.theta_balance_max_deg
    angular_selection = mask.copy()
    mask &= missing_energy <= cfg.missing_energy_max_gev

    selected = {name: np.asarray(values)[mask] for name, values in arrays.items()}
    selected["coplanarityDeg"] = coplanarity[mask]
    selected["thetaBalanceDeg"] = theta_balance[mask]
    selected["missingEnergyGeV"] = missing_energy[mask]
    summary: dict[str, object] = {
        "inputCandidates": int(mask.size),
        "preselectedCandidates": int(np.count_nonzero(preselection)),
        "angularSelectedCandidates": int(np.count_nonzero(angular_selection)),
        "selectedCandidates": int(np.count_nonzero(mask)),
        "selectedFraction": float(np.mean(mask)) if mask.size else 0.0,
        "coplanarityMaxDeg": cfg.coplanarity_max_deg,
        "thetaBalanceMaxDeg": cfg.theta_balance_max_deg,
        "missingEnergyMaxGeV": cfg.missing_energy_max_gev,
        "preselectionAngularClosure": {
            "coplanarityDeg": _region_summary(coplanarity[preselection]),
            "thetaBalanceDeg": _region_summary(theta_balance[preselection]),
        },
        "selectedAngularClosure": {
            "coplanarityDeg": _region_summary(coplanarity[mask]),
            "thetaBalanceDeg": _region_summary(theta_balance[mask]),
        },
        "preselectionMissingEnergyGeV": _region_summary(
            missing_energy[preselection]
        ),
        "angularSelectionMissingEnergyGeV": _region_summary(
            missing_energy[angular_selection]
        ),
        "selectedMissingEnergyGeV": _region_summary(missing_energy[mask]),
    }
    return selected, summary


def robust_core(values: np.ndarray) -> tuple[float, float, float, int]:
    core = np.asarray(values, dtype=float)
    core = core[np.isfinite(core)]
    if core.size < 2:
        return np.nan, np.nan, np.nan, int(core.size)
    for _ in range(8):
        median = float(np.median(core))
        mad = float(np.median(np.abs(core - median)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0.0:
            break
        retained = core[np.abs(core - median) <= 3.0 * sigma]
        if retained.size == core.size or retained.size < 2:
            break
        core = retained
    center = float(np.mean(core))
    width = float(np.std(core, ddof=1)) if core.size > 1 else np.nan
    error = width / np.sqrt(core.size) if width > 0.0 else np.nan
    return center, error, width, int(core.size)


def mode_seeded_core(
    values: np.ndarray,
    *,
    peak_search_max_abs_residual: float,
    peak_seed_half_width: float,
    sigma_clip: float,
) -> CoreEstimate:
    """Estimate a narrow residual peak without assuming it is the majority."""
    input_values = np.asarray(values, dtype=float)
    finite = np.isfinite(input_values)
    total_entries = int(np.count_nonzero(finite))
    empty_mask = np.zeros(input_values.shape, dtype=bool)
    if total_entries < 2:
        return CoreEstimate(
            np.nan, np.nan, np.nan, total_entries, total_entries, np.nan,
            0.0, 0.0, False, empty_mask
        )

    plausible = finite & (
        np.abs(input_values) <= peak_search_max_abs_residual
    )
    plausible_values = input_values[plausible]
    if plausible_values.size < 2:
        return CoreEstimate(
            np.nan, np.nan, np.nan, int(plausible_values.size), total_entries,
            np.nan, float(plausible_values.size / total_entries), 0.0, False,
            empty_mask,
        )

    histogram_bins = int(np.clip(np.ceil(np.sqrt(plausible_values.size)), 20, 80))
    histogram, edges = np.histogram(
        plausible_values,
        bins=histogram_bins,
        range=(-peak_search_max_abs_residual, peak_search_max_abs_residual),
    )
    smoothed = np.convolve(
        histogram.astype(float),
        np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0,
        mode="same",
    )
    peak_index = int(np.argmax(smoothed))
    mode = float(0.5 * (edges[peak_index] + edges[peak_index + 1]))

    seed_mask = plausible & (
        np.abs(input_values - mode) <= peak_seed_half_width
    )
    if np.count_nonzero(seed_mask) < 2:
        return CoreEstimate(
            np.nan, np.nan, np.nan, int(np.count_nonzero(seed_mask)),
            total_entries, mode, float(np.count_nonzero(seed_mask) / total_entries),
            0.0, False, seed_mask,
        )

    core_mask = seed_mask.copy()
    converged = False
    for _ in range(12):
        core = input_values[core_mask]
        center = float(np.mean(core))
        median = float(np.median(core))
        mad = float(np.median(np.abs(core - median)))
        width = 1.4826 * mad
        if not np.isfinite(width) or width <= 0.0:
            width = float(np.std(core, ddof=1)) if core.size > 1 else np.nan
        if not np.isfinite(width) or width <= 0.0:
            break
        candidate_mask = plausible & (
            np.abs(input_values - center) <= sigma_clip * width
        )
        if np.count_nonzero(candidate_mask) < 2:
            break
        if np.array_equal(candidate_mask, core_mask):
            converged = True
            break
        core_mask = candidate_mask

    core = input_values[core_mask]
    if core.size < 2:
        return CoreEstimate(
            np.nan, np.nan, np.nan, int(core.size), total_entries, mode,
            float(core.size / total_entries), 0.0, converged, core_mask,
        )
    center = float(np.mean(core))
    width = float(np.std(core, ddof=1))
    error = width / np.sqrt(core.size) if width > 0.0 else np.nan
    sideband_mask = finite & (
        np.abs(input_values - center) > sigma_clip * width
    ) & (
        np.abs(input_values - center) <= 2.0 * sigma_clip * width
    )
    sideband_entries = int(np.count_nonzero(sideband_mask))
    peak_significance = max(
        0.0,
        float(core.size - sideband_entries) /
        np.sqrt(max(float(core.size + sideband_entries), 1.0)),
    )
    return CoreEstimate(
        center=center,
        error=error,
        width=width,
        entries=int(core.size),
        total_entries=total_entries,
        mode=mode,
        retained_fraction=float(core.size / total_entries),
        peak_significance=peak_significance,
        converged=converged,
        mask=core_mask,
    )


def _split_wide_bins(edges: np.ndarray, maximum_width: float | None) -> np.ndarray:
    """Split quantile intervals that are too wide in physical angle."""
    if maximum_width is None:
        return edges
    split_edges = [float(edges[0])]
    for lower, upper in zip(edges[:-1], edges[1:]):
        pieces = max(1, int(np.ceil((upper - lower) / maximum_width)))
        split_edges.extend(np.linspace(lower, upper, pieces + 1)[1:].tolist())
    return np.asarray(split_edges, dtype=float)


def _profile_cell_plan(
    theta: np.ndarray,
    phi: np.ndarray,
    theta_edges: np.ndarray,
    phi_range: tuple[float, float],
    cfg: ElasticFitConfig,
    minimum_phi_cells: int = 1,
) -> ProfileCellPlan:
    """Plan fixed cells or occupancy-adaptive phi cells inside theta slices."""
    cells: list[dict[str, object]] = []
    slices: list[dict[str, object]] = []
    fixed_phi_edges = np.linspace(phi_range[0], phi_range[1], cfg.phi_bins + 1)
    theta_pairs = list(zip(theta_edges[:-1], theta_edges[1:]))
    for theta_index, (theta_lo, theta_hi) in enumerate(theta_pairs):
        theta_high_inclusive = theta_index == len(theta_pairs) - 1
        theta_mask = (theta >= theta_lo) & (
            (theta <= theta_hi) if theta_high_inclusive else (theta < theta_hi)
        )
        slice_entries = int(np.count_nonzero(theta_mask))
        if cfg.profile_binning == "adaptive" and slice_entries:
            occupancy_target = max(
                1, int(np.ceil(slice_entries / cfg.target_cell_entries))
            )
            maximum_supported = max(1, slice_entries // cfg.min_bin_entries)
            requested_phi_bins = min(cfg.phi_bins, maximum_supported)
            if maximum_supported >= minimum_phi_cells:
                requested_phi_bins = max(
                    minimum_phi_cells,
                    min(requested_phi_bins, occupancy_target),
                )
            else:
                requested_phi_bins = min(requested_phi_bins, occupancy_target)
            phi_edges = np.unique(np.quantile(
                phi[theta_mask], np.linspace(0.0, 1.0, requested_phi_bins + 1)
            ))
        elif cfg.profile_binning == "adaptive":
            phi_edges = np.asarray([], dtype=float)
        else:
            phi_edges = fixed_phi_edges
        phi_pairs = list(zip(phi_edges[:-1], phi_edges[1:]))
        for phi_index, (phi_lo, phi_hi) in enumerate(phi_pairs):
            cells.append({
                "thetaRangeDeg": [float(theta_lo), float(theta_hi)],
                "phiRangeDeg": [float(phi_lo), float(phi_hi)],
                "thetaBin": theta_index,
                "phiBin": phi_index,
                "thetaHighInclusive": theta_high_inclusive,
                "phiHighInclusive": phi_index == len(phi_pairs) - 1,
            })
        slices.append({
            "thetaBin": theta_index,
            "thetaRangeDeg": [float(theta_lo), float(theta_hi)],
            "rawEntries": slice_entries,
            "phiEdgesDeg": [float(value) for value in phi_edges],
            "plannedPhiCells": len(phi_pairs),
        })
    return ProfileCellPlan(
        cells=cells,
        metadata={
            "mode": cfg.profile_binning,
            "requestedThetaBins": cfg.theta_bins,
            "actualThetaBins": len(theta_pairs),
            "thetaEdgesDeg": [float(value) for value in theta_edges],
            "maxThetaBinWidthDeg": cfg.max_theta_bin_width_deg,
            "maximumPhiBinsPerTheta": cfg.phi_bins,
            "minimumPhiCellsForBasis": minimum_phi_cells,
            "targetCellEntries": (
                cfg.target_cell_entries if cfg.profile_binning == "adaptive" else None
            ),
            "plannedProfileCells": len(cells),
            "thetaSlices": slices,
        },
    )


def _cell_mask(
    theta: np.ndarray,
    phi: np.ndarray,
    cell: dict[str, object],
) -> np.ndarray:
    theta_lo, theta_hi = (float(value) for value in cell["thetaRangeDeg"])
    phi_lo, phi_hi = (float(value) for value in cell["phiRangeDeg"])
    theta_upper = (
        theta <= theta_hi if cell.get("thetaHighInclusive", False)
        else theta < theta_hi
    )
    phi_upper = (
        phi <= phi_hi if cell.get("phiHighInclusive", False)
        else phi < phi_hi
    )
    return (
        (theta >= theta_lo) & theta_upper &
        (phi >= phi_lo) & phi_upper
    )


def _profile_grid(
    theta: np.ndarray,
    phi: np.ndarray,
    residual: np.ndarray,
    planned_cells: list[dict[str, object]],
    cfg: ElasticFitConfig,
) -> ProfileGrid:
    theta_points: list[float] = []
    phi_points: list[float] = []
    residual_points: list[float] = []
    errors: list[float] = []
    widths: list[float] = []
    entries: list[int] = []
    total_entries: list[int] = []
    core_fractions: list[float] = []
    peak_significances: list[float] = []
    support_cells: list[dict[str, list[float]]] = []
    rejected_cells: list[dict[str, object]] = []
    for planned_cell in planned_cells:
        cell = _cell_mask(theta, phi, planned_cell)
        cell_entries = int(np.count_nonzero(cell))
        cell_id = dict(planned_cell)
        if cell_entries < cfg.min_bin_entries:
            rejected_cells.append({
                **cell_id,
                "reason": "rawEntries",
                "value": cell_entries,
                "threshold": cfg.min_bin_entries,
            })
            continue
        estimate = mode_seeded_core(
            residual[cell],
            peak_search_max_abs_residual=cfg.peak_search_max_abs_residual,
            peak_seed_half_width=cfg.peak_seed_half_width,
            sigma_clip=cfg.core_sigma_clip,
        )
        quality_failure: tuple[str, float, float] | None = None
        if estimate.entries < cfg.min_bin_entries:
            quality_failure = (
                "coreEntries", float(estimate.entries), float(cfg.min_bin_entries)
            )
        elif not np.isfinite(estimate.center) or not np.isfinite(estimate.error) or estimate.error <= 0:
            quality_failure = ("finiteCenterError", 0.0, 1.0)
        elif abs(estimate.center) > cfg.peak_search_max_abs_residual:
            quality_failure = (
                "absoluteCenter", abs(estimate.center),
                cfg.peak_search_max_abs_residual,
            )
        elif not np.isfinite(estimate.width) or estimate.width > cfg.max_core_width:
            quality_failure = (
                "coreWidth", estimate.width, cfg.max_core_width
            )
        elif estimate.retained_fraction < cfg.min_core_fraction:
            quality_failure = (
                "coreFraction", estimate.retained_fraction, cfg.min_core_fraction
            )
        elif estimate.peak_significance < cfg.min_peak_significance:
            quality_failure = (
                "peakSignificance", estimate.peak_significance,
                cfg.min_peak_significance,
            )
        if quality_failure is not None:
            reason, value, threshold = quality_failure
            rejected_cells.append({
                **cell_id,
                "reason": reason,
                "value": float(value),
                "threshold": float(threshold),
                "rawEntries": cell_entries,
                "coreEntries": estimate.entries,
                "mode": estimate.mode if np.isfinite(estimate.mode) else None,
            })
            continue
        cell_theta = theta[cell]
        cell_phi = phi[cell]
        theta_points.append(float(np.mean(cell_theta[estimate.mask])))
        phi_points.append(float(np.mean(cell_phi[estimate.mask])))
        residual_points.append(estimate.center)
        errors.append(estimate.error)
        widths.append(estimate.width)
        entries.append(estimate.entries)
        total_entries.append(estimate.total_entries)
        core_fractions.append(estimate.retained_fraction)
        peak_significances.append(estimate.peak_significance)
        support_cells.append(cell_id)
    return ProfileGrid(
        theta_deg=np.asarray(theta_points, dtype=float),
        phi_deg=np.asarray(phi_points, dtype=float),
        residual=np.asarray(residual_points, dtype=float),
        error=np.asarray(errors, dtype=float),
        width=np.asarray(widths, dtype=float),
        entries=np.asarray(entries, dtype=int),
        total_entries=np.asarray(total_entries, dtype=int),
        core_fraction=np.asarray(core_fractions, dtype=float),
        peak_significance=np.asarray(peak_significances, dtype=float),
        support_cells=support_cells,
        rejected_cells=rejected_cells,
    )


def _adaptive_slice_cells(
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    theta_bin: int,
    theta_range: list[float],
    theta_high_inclusive: bool,
    requested_phi_cells: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Build quantile-phi cells for one adaptive theta slice."""
    theta_lo, theta_hi = (float(value) for value in theta_range)
    theta_upper = (
        theta <= theta_hi if theta_high_inclusive else theta < theta_hi
    )
    theta_mask = (theta >= theta_lo) & theta_upper
    if not np.any(theta_mask) or requested_phi_cells < 1:
        return [], np.asarray([], dtype=float)
    phi_edges = np.unique(np.quantile(
        phi[theta_mask], np.linspace(0.0, 1.0, requested_phi_cells + 1)
    ))
    phi_pairs = list(zip(phi_edges[:-1], phi_edges[1:]))
    cells = [
        {
            "thetaRangeDeg": [theta_lo, theta_hi],
            "phiRangeDeg": [float(phi_lo), float(phi_hi)],
            "thetaBin": theta_bin,
            "phiBin": phi_index,
            "thetaHighInclusive": theta_high_inclusive,
            "phiHighInclusive": phi_index == len(phi_pairs) - 1,
        }
        for phi_index, (phi_lo, phi_hi) in enumerate(phi_pairs)
    ]
    return cells, phi_edges


def _adaptive_population_fallback(
    theta: np.ndarray,
    phi: np.ndarray,
    residual: np.ndarray,
    cell_plan: ProfileCellPlan,
    cfg: ElasticFitConfig,
) -> ProfileCellPlan:
    """Coarsen phi cells when quantile children fail population thresholds.

    Planning uses raw occupancy, while acceptance uses the retained peak core.
    A subdivision can therefore create a child just below the core-entry
    threshold even though its parent is usable. Retry the whole theta slice
    with fewer phi cells in that case. Failures of the peak-quality checks
    remain explicit holes and never trigger merging.
    """
    if cfg.profile_binning != "adaptive":
        return cell_plan

    initial_cells_by_theta: dict[int, list[dict[str, object]]] = {}
    for cell in cell_plan.cells:
        initial_cells_by_theta.setdefault(int(cell["thetaBin"]), []).append(cell)

    slices = list(cell_plan.metadata["thetaSlices"])
    final_cells: list[dict[str, object]] = []
    final_slices: list[dict[str, object]] = []
    fallback_slices = 0
    fallback_steps = 0
    population_reasons = {"rawEntries", "coreEntries"}

    for slice_index, slice_metadata in enumerate(slices):
        theta_bin = int(slice_metadata["thetaBin"])
        initial_cells = initial_cells_by_theta.get(theta_bin, [])
        requested_phi_cells = max(len(initial_cells), 1)
        theta_high_inclusive = slice_index == len(slices) - 1
        candidate_cells = initial_cells
        candidate_edges = np.asarray(slice_metadata["phiEdgesDeg"], dtype=float)
        attempts: list[dict[str, object]] = []

        while candidate_cells:
            candidate_grid = _profile_grid(
                theta, phi, residual, candidate_cells, cfg
            )
            rejection_counts: dict[str, int] = {}
            for rejected in candidate_grid.rejected_cells:
                reason = str(rejected["reason"])
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            population_failures = sum(
                count for reason, count in rejection_counts.items()
                if reason in population_reasons
            )
            quality_failures = sum(
                count for reason, count in rejection_counts.items()
                if reason not in population_reasons
            )
            attempts.append({
                "requestedPhiCells": requested_phi_cells,
                "plannedPhiCells": len(candidate_cells),
                "phiEdgesDeg": [float(value) for value in candidate_edges],
                "acceptedPhiCells": len(candidate_grid.support_cells),
                "rejectedPhiCells": len(candidate_grid.rejected_cells),
                "rejectionCounts": rejection_counts,
            })
            if (
                population_failures == 0
                or quality_failures > 0
                or requested_phi_cells <= 1
            ):
                break
            requested_phi_cells -= 1
            candidate_cells, candidate_edges = _adaptive_slice_cells(
                theta,
                phi,
                theta_bin=theta_bin,
                theta_range=list(slice_metadata["thetaRangeDeg"]),
                theta_high_inclusive=theta_high_inclusive,
                requested_phi_cells=requested_phi_cells,
            )

        applied = len(attempts) > 1
        if applied:
            fallback_slices += 1
            fallback_steps += len(attempts) - 1
        final_cells.extend(candidate_cells)
        final_slices.append({
            **slice_metadata,
            "initialPlannedPhiCells": len(initial_cells),
            "phiEdgesDeg": [float(value) for value in candidate_edges],
            "plannedPhiCells": len(candidate_cells),
            "populationFallbackApplied": applied,
            "populationFallbackAttempts": attempts,
        })

    return ProfileCellPlan(
        cells=final_cells,
        metadata={
            **cell_plan.metadata,
            "initialPlannedProfileCells": len(cell_plan.cells),
            "plannedProfileCells": len(final_cells),
            "populationFallbackThetaSlices": fallback_slices,
            "populationFallbackSteps": fallback_steps,
            "thetaSlices": final_slices,
        },
    )


def _polynomial_matrix(
    theta_normalized: np.ndarray,
    phi_normalized: np.ndarray,
    theta_order: int,
    phi_order: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    definitions = [
        {"thetaPower": theta_power, "phiPower": phi_power}
        for theta_power in range(theta_order + 1)
        for phi_power in range(phi_order + 1)
    ]
    matrix = np.column_stack([
        np.power(theta_normalized, term["thetaPower"]) *
        np.power(phi_normalized, term["phiPower"])
        for term in definitions
    ])
    return matrix, definitions


def _fourier_matrix(
    theta_normalized: np.ndarray,
    phi_deg: np.ndarray,
    theta_order: int,
    harmonics: int,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    columns: list[np.ndarray] = []
    definitions: list[dict[str, int | str]] = []
    phi_rad = np.deg2rad(phi_deg)
    for theta_power in range(theta_order + 1):
        theta_term = np.power(theta_normalized, theta_power)
        columns.append(theta_term)
        definitions.append({
            "thetaPower": theta_power, "component": "constant", "harmonic": 0
        })
        for harmonic in range(1, harmonics + 1):
            columns.append(theta_term * np.cos(harmonic * phi_rad))
            definitions.append({
                "thetaPower": theta_power, "component": "cos", "harmonic": harmonic
            })
            columns.append(theta_term * np.sin(harmonic * phi_rad))
            definitions.append({
                "thetaPower": theta_power, "component": "sin", "harmonic": harmonic
            })
    return np.column_stack(columns), definitions


def evaluate_region(
    region: dict[str, object],
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    theta_normalized = (
        np.asarray(theta_deg) - float(region["thetaCenterDeg"])
    ) / float(region["thetaScaleDeg"])
    terms = list(region["terms"])
    if region["basis"] == "polynomial":
        phi_normalized = (
            np.asarray(phi_deg) - float(region["phiCenterDeg"])
        ) / float(region["phiScaleDeg"])
        return np.sum(np.column_stack([
            float(term["coefficient"]) *
            np.power(theta_normalized, int(term["thetaPower"])) *
            np.power(phi_normalized, int(term["phiPower"]))
            for term in terms
        ]), axis=1)

    phi_rad = np.deg2rad(phi_deg)
    columns: list[np.ndarray] = []
    for term in terms:
        component = str(term["component"])
        harmonic = int(term["harmonic"])
        phi_term = np.ones_like(phi_rad)
        if component == "cos":
            phi_term = np.cos(harmonic * phi_rad)
        elif component == "sin":
            phi_term = np.sin(harmonic * phi_rad)
        columns.append(
            float(term["coefficient"]) *
            np.power(theta_normalized, int(term["thetaPower"])) * phi_term
        )
    return np.sum(np.column_stack(columns), axis=1)


def region_support_mask(
    region: dict[str, object],
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=float)
    phi = np.asarray(phi_deg, dtype=float)
    theta_range = list(region["thetaRangeDeg"])
    phi_range = list(region["phiRangeDeg"])
    mask = (
        (theta >= float(theta_range[0])) &
        (theta <= float(theta_range[1])) &
        (phi >= float(phi_range[0])) &
        (phi <= float(phi_range[1]))
    )
    support_cells = list(region.get("supportCells", []))
    if not support_cells:
        return mask
    cell_mask = np.zeros(theta.shape, dtype=bool)
    for cell in support_cells:
        cell_theta = list(cell["thetaRangeDeg"])
        cell_phi = list(cell["phiRangeDeg"])
        cell_mask |= (
            (theta >= float(cell_theta[0])) &
            (theta <= float(cell_theta[1])) &
            (phi >= float(cell_phi[0])) &
            (phi <= float(cell_phi[1]))
        )
    return mask & cell_mask


def _accepted_cell_mask(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    cell: dict[str, object],
) -> np.ndarray:
    """Use the same half-open intervals as the profile-cell extraction."""
    return _cell_mask(theta_deg, phi_deg, cell)


def fit_region(
    *,
    pid: int,
    detector: int,
    sector: int,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    residual: np.ndarray,
    basis: str,
    cfg: ElasticFitConfig,
) -> tuple[dict[str, object], RegionDiagnostics]:
    if cfg.theta_bins < 1 or cfg.phi_bins < 1:
        raise ValueError("theta_bins and phi_bins must be positive")
    if cfg.profile_binning not in ("fixed", "adaptive"):
        raise ValueError("profile_binning must be fixed or adaptive")
    if (
        cfg.max_theta_bin_width_deg is not None
        and cfg.max_theta_bin_width_deg <= 0.0
    ):
        raise ValueError("maximum theta-bin width must be positive")
    if cfg.target_cell_entries < 2:
        raise ValueError("target cell entries must be at least two")
    if cfg.min_phi_cells_per_theta < 1:
        raise ValueError("minimum phi cells per theta slice must be positive")
    if cfg.min_phi_cells_per_theta > cfg.phi_bins:
        raise ValueError("minimum phi cells per theta slice exceeds phi bins")
    if cfg.theta_order < 0 or cfg.phi_order < 0 or cfg.cd_fourier_harmonics < 0:
        raise ValueError("fit orders and Fourier harmonics must be nonnegative")
    if cfg.min_bin_entries < 2 or cfg.min_region_entries < 2:
        raise ValueError("minimum entry counts must be at least two")
    if (
        cfg.profile_binning == "adaptive"
        and cfg.target_cell_entries < cfg.min_bin_entries
    ):
        raise ValueError(
            "adaptive target cell entries must not be below min_bin_entries"
        )
    if not 0.0 < cfg.missing_energy_max_gev:
        raise ValueError("missing-energy maximum must be positive")
    if not 0.0 < cfg.peak_search_max_abs_residual < 1.0:
        raise ValueError("peak-search residual range must be in (0, 1)")
    if not 0.0 < cfg.peak_seed_half_width <= cfg.peak_search_max_abs_residual:
        raise ValueError("peak seed half-width must not exceed the search range")
    if cfg.core_sigma_clip <= 0.0 or cfg.max_core_width <= 0.0:
        raise ValueError("core sigma clip and maximum width must be positive")
    if not 0.0 < cfg.min_core_fraction <= 1.0:
        raise ValueError("minimum core fraction must be in (0, 1]")
    if cfg.min_peak_significance < 0.0:
        raise ValueError("minimum peak significance must be nonnegative")
    if cfg.min_profile_cells_per_parameter < 1.0:
        raise ValueError("profile-cells-per-parameter must be at least one")
    if cfg.max_condition_number <= 1.0:
        raise ValueError("maximum condition number must exceed one")
    if not 0.0 < cfg.max_abs_surface_correction < 1.0:
        raise ValueError("maximum absolute surface correction must be in (0, 1)")
    if (
        cfg.theta_min_deg is not None and cfg.theta_max_deg is not None
        and cfg.theta_min_deg >= cfg.theta_max_deg
    ):
        raise ValueError("minimum theta must be below maximum theta")
    finite = np.isfinite(theta_deg) & np.isfinite(phi_deg) & np.isfinite(residual)
    theta = np.asarray(theta_deg, dtype=float)[finite]
    phi = np.asarray(phi_deg, dtype=float)[finite]
    residual_values = np.asarray(residual, dtype=float)[finite]
    phi_range = (-30.0, 30.0) if basis == "polynomial" else (-180.0, 180.0)
    phi_support = (phi >= phi_range[0]) & (phi <= phi_range[1])
    theta = theta[phi_support]
    phi = phi[phi_support]
    residual_values = residual_values[phi_support]
    requested_theta_support = np.ones(theta.shape, dtype=bool)
    if cfg.theta_min_deg is not None:
        requested_theta_support &= theta >= cfg.theta_min_deg
    if cfg.theta_max_deg is not None:
        requested_theta_support &= theta <= cfg.theta_max_deg
    theta = theta[requested_theta_support]
    phi = phi[requested_theta_support]
    residual_values = residual_values[requested_theta_support]
    if theta.size < cfg.min_region_entries:
        raise ValueError(f"region has only {theta.size} entries")

    q = cfg.theta_trim_quantile
    if not 0.0 <= q < 0.5:
        raise ValueError("theta_trim_quantile must satisfy 0 <= q < 0.5")
    theta_min, theta_max = np.quantile(theta, (q, 1.0 - q))
    if not theta_max > theta_min:
        raise ValueError("region has no finite theta span")
    theta_support = (theta >= theta_min) & (theta <= theta_max)
    theta = theta[theta_support]
    phi = phi[theta_support]
    residual_values = residual_values[theta_support]

    rq = cfg.residual_trim_quantile
    if not 0.0 <= rq < 0.5:
        raise ValueError("residual_trim_quantile must satisfy 0 <= q < 0.5")
    residual_min, residual_max = np.quantile(residual_values, (rq, 1.0 - rq))
    residual_support = (residual_values >= residual_min) & (residual_values <= residual_max)
    theta = theta[residual_support]
    phi = phi[residual_support]
    residual_values = residual_values[residual_support]
    if theta.size < cfg.min_region_entries:
        raise ValueError(f"region has only {theta.size} entries after trimming")

    theta_edges = np.unique(np.quantile(
        theta, np.linspace(0.0, 1.0, cfg.theta_bins + 1)
    ))
    if cfg.profile_binning == "adaptive":
        theta_edges = _split_wide_bins(
            theta_edges, cfg.max_theta_bin_width_deg
        )
    minimum_phi_cells = (
        cfg.phi_order + 1 if basis == "polynomial"
        else 2 * cfg.cd_fourier_harmonics + 1
    )
    minimum_phi_cells = max(minimum_phi_cells, cfg.min_phi_cells_per_theta)
    cell_plan = _profile_cell_plan(
        theta, phi, theta_edges, phi_range, cfg, minimum_phi_cells
    )
    cell_plan = _adaptive_population_fallback(
        theta, phi, residual_values, cell_plan, cfg
    )
    profile = _profile_grid(
        theta, phi, residual_values, cell_plan.cells, cfg
    )

    theta_center = 0.5 * (theta_min + theta_max)
    theta_scale = 0.5 * (theta_max - theta_min)
    theta_normalized = (profile.theta_deg - theta_center) / theta_scale
    if basis == "polynomial":
        phi_center = 0.0
        phi_scale = 30.0
        matrix, definitions = _polynomial_matrix(
            theta_normalized,
            (profile.phi_deg - phi_center) / phi_scale,
            cfg.theta_order,
            cfg.phi_order,
        )
        phi_variable = "sectorLocal"
    elif basis == "fourier":
        phi_center = 0.0
        phi_scale = 180.0
        matrix, definitions = _fourier_matrix(
            theta_normalized,
            profile.phi_deg,
            cfg.theta_order,
            cfg.cd_fourier_harmonics,
        )
        phi_variable = "global"
    else:
        raise ValueError(f"unsupported basis {basis}")

    if basis == "polynomial" and cfg.phi_order > 0:
        accepted_by_theta: dict[int, int] = {}
        for cell in profile.support_cells:
            theta_bin = int(cell["thetaBin"])
            accepted_by_theta[theta_bin] = accepted_by_theta.get(theta_bin, 0) + 1
        required_theta_slices = cfg.theta_order + 1
        independent_theta_slices = sum(
            count >= cfg.phi_order + 1 for count in accepted_by_theta.values()
        )
        if independent_theta_slices < required_theta_slices:
            raise ValueError(
                "phi-dependent surface has only "
                f"{independent_theta_slices} theta slices with at least "
                f"{cfg.phi_order + 1} accepted phi cells; requires "
                f"{required_theta_slices}"
            )

    minimum_profile_cells = int(np.ceil(
        cfg.min_profile_cells_per_parameter * matrix.shape[1]
    ))
    if matrix.shape[0] < minimum_profile_cells:
        raise ValueError(
            f"region has {matrix.shape[0]} usable profile cells; requires "
            f"{minimum_profile_cells} for {matrix.shape[1]} terms"
        )
    weights = 1.0 / profile.error
    weighted_matrix = matrix * weights[:, None]
    weighted_residual = profile.residual * weights
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        weighted_matrix, weighted_residual, rcond=None
    )
    if rank != matrix.shape[1] or not np.all(np.isfinite(coefficients)):
        raise ValueError("region fit is rank deficient")
    condition_number = float(singular_values[0] / singular_values[-1])
    if not np.isfinite(condition_number) or condition_number > cfg.max_condition_number:
        raise ValueError(
            f"region weighted design condition number {condition_number:.6g} exceeds "
            f"{cfg.max_condition_number:.6g}"
        )

    terms = [
        {**definition, "coefficient": float(coefficient)}
        for definition, coefficient in zip(definitions, coefficients)
    ]
    predicted = matrix @ coefficients
    chi2 = float(np.sum(np.square((profile.residual - predicted) / profile.error)))
    ndof = int(matrix.shape[0] - matrix.shape[1])
    region: dict[str, object] = {
        "pid": pid,
        "detector": detector,
        "sector": sector,
        "thetaRangeDeg": [float(theta_min), float(theta_max)],
        "phiRangeDeg": [phi_range[0], phi_range[1]],
        "thetaCenterDeg": float(theta_center),
        "thetaScaleDeg": float(theta_scale),
        "phiCenterDeg": phi_center,
        "phiScaleDeg": phi_scale,
        "phiVariable": phi_variable,
        "basis": basis,
        "terms": terms,
        "supportCells": profile.support_cells,
        "fit": {
            "entries": int(theta.size),
            "profileCells": int(matrix.shape[0]),
            "parameters": int(matrix.shape[1]),
            "chi2": chi2,
            "ndof": ndof,
            "chi2PerNdf": chi2 / ndof if ndof > 0 else None,
            "weightedDesignConditionNumber": condition_number,
            "weightedDesignSingularValues": [
                float(value) for value in singular_values
            ],
            "residualCoreRange": [float(residual_min), float(residual_max)],
            "peakSearchMaxAbsResidual": cfg.peak_search_max_abs_residual,
            "profileBinning": {
                **cell_plan.metadata,
                "acceptedProfileCells": len(profile.support_cells),
                "rejectedProfileCells": len(profile.rejected_cells),
            },
            "acceptedProfileCells": [
                {
                    **support,
                    "thetaMeanDeg": float(profile.theta_deg[index]),
                    "phiMeanDeg": float(profile.phi_deg[index]),
                    "center": float(profile.residual[index]),
                    "centerError": float(profile.error[index]),
                    "coreWidth": float(profile.width[index]),
                    "coreEntries": int(profile.entries[index]),
                    "rawEntries": int(profile.total_entries[index]),
                    "coreFraction": float(profile.core_fraction[index]),
                    "peakSignificance": float(profile.peak_significance[index]),
                }
                for index, support in enumerate(profile.support_cells)
            ],
            "rejectedProfileCells": profile.rejected_cells,
        },
    }
    support_sample_theta: list[float] = []
    support_sample_phi: list[float] = []
    for cell in profile.support_cells:
        theta_lo, theta_hi = (float(value) for value in cell["thetaRangeDeg"])
        phi_lo, phi_hi = (float(value) for value in cell["phiRangeDeg"])
        theta_samples = np.linspace(theta_lo, theta_hi, 9)
        phi_samples = np.linspace(phi_lo, phi_hi, 9)
        for theta_sample in theta_samples:
            support_sample_theta.extend([float(theta_sample)] * phi_samples.size)
            support_sample_phi.extend(float(value) for value in phi_samples)
    support_values = evaluate_region(
        region,
        np.asarray(support_sample_theta, dtype=float),
        np.asarray(support_sample_phi, dtype=float),
    )
    maximum_support_correction = float(np.max(np.abs(support_values)))
    region["fit"]["maxAbsSurfaceCorrectionOnSupportGrid"] = (
        maximum_support_correction
    )
    if maximum_support_correction > cfg.max_abs_surface_correction:
        raise ValueError(
            "surface reaches an absolute correction of "
            f"{maximum_support_correction:.6g} on its support grid; exceeds "
            f"{cfg.max_abs_surface_correction:.6g}"
        )
    supported = region_support_mask(region, theta, phi)
    fitted_correction = np.zeros(theta.shape, dtype=float)
    fitted_correction[supported] = evaluate_region(
        region, theta[supported], phi[supported]
    )
    region["fit"]["applicationSupportEntries"] = int(np.count_nonzero(supported))
    region["fit"]["applicationSupportFraction"] = float(np.mean(supported))
    residual_after = (1.0 + residual_values) / (1.0 + fitted_correction) - 1.0
    for cell in region["fit"]["acceptedProfileCells"]:
        cell_mask = _accepted_cell_mask(
            theta, phi, cell
        )
        after_estimate = mode_seeded_core(
            residual_after[cell_mask],
            peak_search_max_abs_residual=cfg.peak_search_max_abs_residual,
            peak_seed_half_width=cfg.peak_seed_half_width,
            sigma_clip=cfg.core_sigma_clip,
        )
        fitted_at_center = float(evaluate_region(
            region,
            np.asarray([cell["thetaMeanDeg"]]),
            np.asarray([cell["phiMeanDeg"]]),
        )[0])
        cell["surfaceAtCellMean"] = fitted_at_center
        cell["centerMinusSurface"] = float(cell["center"] - fitted_at_center)
        cell["afterCenter"] = (
            after_estimate.center if np.isfinite(after_estimate.center) else None
        )
        cell["afterCenterError"] = (
            after_estimate.error if np.isfinite(after_estimate.error) else None
        )
        cell["afterCoreWidth"] = (
            after_estimate.width if np.isfinite(after_estimate.width) else None
        )
        cell["afterCoreEntries"] = after_estimate.entries
    label = f"pid{pid}_det{detector}" + (f"_sector{sector}" if sector else "")
    diagnostics = RegionDiagnostics(
        label=label,
        theta_deg=theta,
        phi_deg=phi,
        residual_before=residual_values,
        residual_after=residual_after,
        profile_theta_deg=profile.theta_deg,
        profile_phi_deg=profile.phi_deg,
        profile_residual=profile.residual,
        profile_error=profile.error,
        profile_width=profile.width,
        profile_entries=profile.entries,
        profile_core_fraction=profile.core_fraction,
        profile_peak_significance=profile.peak_significance,
        region=region,
    )
    return region, diagnostics


def _region_summary(values: np.ndarray) -> dict[str, float]:
    center, _, width, retained = robust_core(values)
    return {
        "coreMean": center,
        "coreWidth": width,
        "coreEntries": retained,
        "median": float(np.median(values)),
    }


def derive_corrections(
    arrays: dict[str, np.ndarray],
    cfg: ElasticFitConfig,
    particles: Iterable[str] = ("electron", "proton"),
) -> tuple[dict[str, object], list[RegionDiagnostics]]:
    if cfg.torus not in (-1, 1):
        raise ValueError("torus must be -1 or 1 when exporting corrections")
    selected, selection_summary = select_elastic_events(arrays, cfg)
    if selection_summary["selectedCandidates"] == 0:
        raise ValueError("no candidates survive the elastic angular selection")

    regions: list[dict[str, object]] = []
    diagnostics: list[RegionDiagnostics] = []
    skipped: list[dict[str, object]] = []
    requested = set(particles)
    particle_inputs = []
    if "electron" in requested:
        particle_inputs.append((
            "electron", 11,
            np.asarray(selected["electronP"]),
            np.asarray(selected["electronTheta"]),
            np.asarray(selected["electronPhi"]),
            np.asarray(selected["electronDet"]),
            np.asarray(selected["electronSector"]),
            elastic_electron_momentum(selected["electronTheta"], cfg.beam_energy),
        ))
    if "proton" in requested:
        particle_inputs.append((
            "proton", 2212,
            np.asarray(selected["protonP"]),
            np.asarray(selected["protonTheta"]),
            np.asarray(selected["protonPhi"]),
            np.asarray(selected["protonDet"]),
            np.asarray(selected["protonSector"]),
            elastic_proton_momentum(selected["protonTheta"], cfg.beam_energy),
        ))

    for particle_name, pid, momentum, theta_rad, phi_rad, detector, sector, expected in particle_inputs:
        residual = expected / momentum - 1.0
        theta_deg = theta_rad * RAD_TO_DEG
        for detector_id in (1, 2):
            if particle_name == "electron" and detector_id != 1:
                continue
            if detector_id == 1:
                for sector_id in range(1, 7):
                    region_mask = (detector == detector_id) & (sector == sector_id)
                    label = f"{particle_name}_FD_sector{sector_id}"
                    try:
                        region, diagnostic = fit_region(
                            pid=pid,
                            detector=detector_id,
                            sector=sector_id,
                            theta_deg=theta_deg[region_mask],
                            phi_deg=sector_local_phi(phi_rad[region_mask], sector[region_mask]),
                            residual=residual[region_mask],
                            basis="polynomial",
                            cfg=cfg,
                        )
                    except ValueError as error:
                        skipped.append({"region": label, "reason": str(error)})
                        continue
                    region["particle"] = particle_name
                    region["fit"]["before"] = _region_summary(diagnostic.residual_before)
                    region["fit"]["after"] = _region_summary(diagnostic.residual_after)
                    supported = region_support_mask(
                        region, diagnostic.theta_deg, diagnostic.phi_deg
                    )
                    region["fit"]["beforeSupported"] = _region_summary(
                        diagnostic.residual_before[supported]
                    )
                    region["fit"]["afterSupported"] = _region_summary(
                        diagnostic.residual_after[supported]
                    )
                    regions.append(region)
                    diagnostics.append(diagnostic)
            else:
                region_mask = detector == detector_id
                label = f"{particle_name}_CD"
                try:
                    region, diagnostic = fit_region(
                        pid=pid,
                        detector=detector_id,
                        sector=0,
                        theta_deg=theta_deg[region_mask],
                        phi_deg=wrap_degrees(phi_rad[region_mask] * RAD_TO_DEG),
                        residual=residual[region_mask],
                        basis="fourier",
                        cfg=cfg,
                    )
                except ValueError as error:
                    skipped.append({"region": label, "reason": str(error)})
                    continue
                region["particle"] = particle_name
                region["fit"]["before"] = _region_summary(diagnostic.residual_before)
                region["fit"]["after"] = _region_summary(diagnostic.residual_after)
                supported = region_support_mask(
                    region, diagnostic.theta_deg, diagnostic.phi_deg
                )
                region["fit"]["beforeSupported"] = _region_summary(
                    diagnostic.residual_before[supported]
                )
                region["fit"]["afterSupported"] = _region_summary(
                    diagnostic.residual_after[supported]
                )
                regions.append(region)
                diagnostics.append(diagnostic)

    if not regions:
        reasons = "; ".join(f"{entry['region']}: {entry['reason']}" for entry in skipped)
        raise ValueError(f"no detector region could be fitted ({reasons})")
    output: dict[str, object] = {
        "schema": "elastic_momentum_correction/v1",
        "correctionType": "fractionalMomentum",
        "beamEnergyGeV": cfg.beam_energy,
        "torus": cfg.torus,
        "selection": selection_summary,
        "fitConfiguration": {
            "missingEnergyMaxGeV": cfg.missing_energy_max_gev,
            "thetaMinDeg": cfg.theta_min_deg,
            "thetaMaxDeg": cfg.theta_max_deg,
            "thetaTrimQuantile": cfg.theta_trim_quantile,
            "residualTrimQuantile": cfg.residual_trim_quantile,
            "thetaBins": cfg.theta_bins,
            "phiBins": cfg.phi_bins,
            "profileBinning": cfg.profile_binning,
            "maxThetaBinWidthDeg": cfg.max_theta_bin_width_deg,
            "targetCellEntries": cfg.target_cell_entries,
            "minPhiCellsPerTheta": cfg.min_phi_cells_per_theta,
            "thetaOrder": cfg.theta_order,
            "fdPhiOrder": cfg.phi_order,
            "cdFourierHarmonics": cfg.cd_fourier_harmonics,
            "minBinEntries": cfg.min_bin_entries,
            "minRegionEntries": cfg.min_region_entries,
            "peakSearchMaxAbsResidual": cfg.peak_search_max_abs_residual,
            "peakSeedHalfWidth": cfg.peak_seed_half_width,
            "coreSigmaClip": cfg.core_sigma_clip,
            "minCoreFraction": cfg.min_core_fraction,
            "minPeakSignificance": cfg.min_peak_significance,
            "maxCoreWidth": cfg.max_core_width,
            "minProfileCellsPerParameter": cfg.min_profile_cells_per_parameter,
            "maxConditionNumber": cfg.max_condition_number,
            "maxAbsSurfaceCorrection": cfg.max_abs_surface_correction,
        },
        "regions": regions,
        "skippedRegions": skipped,
    }
    return output, diagnostics


def _accepted_theta_slices(
    cells: list[dict[str, object]],
) -> list[tuple[tuple[float, float], list[dict[str, object]]]]:
    slices: dict[tuple[float, float], list[dict[str, object]]] = {}
    for cell in cells:
        theta_range = tuple(float(value) for value in cell["thetaRangeDeg"])
        slices.setdefault(theta_range, []).append(cell)
    return [
        (theta_range, sorted(slice_cells, key=lambda cell: cell["phiMeanDeg"]))
        for theta_range, slice_cells in sorted(slices.items())
    ]


def _plot_profile_cell_map(
    diagnostic: RegionDiagnostics,
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Patch, Rectangle

    fit = diagnostic.region["fit"]
    accepted = fit["acceptedProfileCells"]
    rejected = fit["rejectedProfileCells"]
    center_percent = 100.0 * np.asarray(
        [cell["center"] for cell in accepted], dtype=float
    )
    core_entries = np.asarray(
        [cell["coreEntries"] for cell in accepted], dtype=float
    )

    def rectangles(cells: list[dict[str, object]]) -> list[Rectangle]:
        patches = []
        for cell in cells:
            theta_lo, theta_hi = (float(value) for value in cell["thetaRangeDeg"])
            phi_lo, phi_hi = (float(value) for value in cell["phiRangeDeg"])
            patches.append(Rectangle(
                (theta_lo, phi_lo), theta_hi - theta_lo, phi_hi - phi_lo
            ))
        return patches

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    center_limit = max(0.25, 1.15 * float(np.max(np.abs(center_percent))))
    collections = (
        PatchCollection(
            rectangles(accepted), cmap="coolwarm", edgecolor="0.2",
            linewidth=0.5,
        ),
        PatchCollection(
            rectangles(accepted), cmap="viridis", edgecolor="0.2",
            linewidth=0.5,
        ),
    )
    collections[0].set_array(center_percent)
    collections[0].set_clim(-center_limit, center_limit)
    collections[1].set_array(core_entries)
    for axis, collection in zip(axes, collections):
        axis.add_collection(collection)
        for patch in rectangles(rejected):
            patch.set_facecolor("0.88")
            patch.set_edgecolor("0.45")
            patch.set_hatch("//")
            patch.set_linewidth(0.6)
            axis.add_patch(patch)
        axis.set_xlim(*diagnostic.region["thetaRangeDeg"])
        axis.set_ylim(*diagnostic.region["phiRangeDeg"])
        axis.set_xlabel("$\\theta$ [deg]")
        axis.set_ylabel(
            "sector-local $\\phi$ [deg]"
            if diagnostic.region["phiVariable"] == "sectorLocal"
            else "global $\\phi$ [deg]"
        )
    fig.colorbar(
        collections[0], ax=axes[0], label="elastic residual center [%]"
    )
    fig.colorbar(collections[1], ax=axes[1], label="retained core events")
    axes[0].set_title("accepted-cell residual center")
    axes[1].set_title("accepted-cell population")
    binning = fit["profileBinning"]
    axes[1].legend(
        handles=[Patch(facecolor="0.88", edgecolor="0.45", hatch="//",
                       label="rejected cell")],
        loc="best", fontsize="small",
    )
    initial_planned = int(binning.get(
        "initialPlannedProfileCells", binning["plannedProfileCells"]
    ))
    final_planned = int(binning["plannedProfileCells"])
    fallback_slices = int(binning.get("populationFallbackThetaSlices", 0))
    plan_summary = f"{final_planned} cells"
    if fallback_slices:
        plan_summary = (
            f"{initial_planned} -> {final_planned} cells after population "
            f"fallback in {fallback_slices} theta slices"
        )
    fig.text(
        0.5, 0.01,
        f"{binning['mode']} plan: {plan_summary}; "
        f"{binning['acceptedProfileCells']} accepted; "
        f"{binning['rejectedProfileCells']} rejected",
        ha="center", fontsize="small",
    )
    save_plot(
        fig,
        output_dir / f"{diagnostic.label}_profile_cell_map.png",
        f"Elastic profile-cell geometry: {diagnostic.label}",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)


def _plot_phi_profiles(
    diagnostic: RegionDiagnostics,
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    region = diagnostic.region
    slices = _accepted_theta_slices(region["fit"]["acceptedProfileCells"])
    columns = 2 if len(slices) > 1 else 1
    rows = int(np.ceil(len(slices) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(6.2 * columns, 2.8 * rows),
        sharex=True, sharey=True, squeeze=False,
    )
    plotted_values: list[float] = []
    for axis, (theta_range, cells) in zip(axes.flat, slices):
        theta_reference = float(np.mean([cell["thetaMeanDeg"] for cell in cells]))
        before_phi = np.asarray([cell["phiMeanDeg"] for cell in cells], dtype=float)
        before_center = 100.0 * np.asarray([cell["center"] for cell in cells], dtype=float)
        before_error = 100.0 * np.asarray([cell["centerError"] for cell in cells], dtype=float)
        axis.errorbar(
            before_phi, before_center, yerr=before_error, fmt="o", markersize=4,
            capsize=2, color="tab:blue", label="before: cell core",
        )
        plotted_values.extend(np.abs(before_center) + before_error)
        after_cells = [
            cell for cell in cells if cell["afterCenter"] is not None
            and cell["afterCenterError"] is not None
        ]
        if after_cells:
            after_phi = np.asarray([cell["phiMeanDeg"] for cell in after_cells], dtype=float)
            after_center = 100.0 * np.asarray([cell["afterCenter"] for cell in after_cells], dtype=float)
            after_error = 100.0 * np.asarray(
                [cell["afterCenterError"] for cell in after_cells], dtype=float
            )
            axis.errorbar(
                after_phi, after_center, yerr=after_error, fmt="s", markersize=4,
                capsize=2, color="tab:green", label="after: re-extracted core",
            )
            plotted_values.extend(np.abs(after_center) + after_error)
        for index, cell in enumerate(cells):
            phi_lo, phi_hi = (float(value) for value in cell["phiRangeDeg"])
            phi_curve = np.linspace(phi_lo, phi_hi, 35)
            theta_curve = np.full(phi_curve.shape, theta_reference)
            fitted_curve = 100.0 * evaluate_region(region, theta_curve, phi_curve)
            axis.plot(
                phi_curve, fitted_curve, color="tab:orange", linewidth=1.4,
                label="direct surface at slice mean $\\theta$" if index == 0 else None,
            )
            plotted_values.extend(np.abs(fitted_curve))
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.set_title(f"$\\theta$ = {theta_range[0]:.1f}–{theta_range[1]:.1f}°  |  {len(cells)} cells")
        axis.set_xlim(float(region["phiRangeDeg"][0]), float(region["phiRangeDeg"][1]))
        axis.set_xlabel("sector-local $\\phi$ [deg]" if region["phiVariable"] == "sectorLocal"
                        else "global $\\phi$ [deg]")
        axis.set_ylabel("elastic residual / correction [%]")
    for axis in axes.flat[len(slices):]:
        axis.set_visible(False)
    limit = max(0.25, 1.15 * max(plotted_values, default=0.0))
    for axis in axes.flat[:len(slices)]:
        axis.set_ylim(-limit, limit)
    axes.flat[0].legend(loc="best", fontsize="small")
    save_plot(
        fig,
        output_dir / f"{diagnostic.label}_profile_vs_phi_by_theta.png",
        f"Elastic cell profiles by polar-angle slice: {diagnostic.label}",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)


def _plot_cell_fit_discrepancies(
    diagnostic: RegionDiagnostics,
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    cells = diagnostic.region["fit"]["acceptedProfileCells"]
    theta = np.asarray([cell["thetaMeanDeg"] for cell in cells], dtype=float)
    phi = np.asarray([cell["phiMeanDeg"] for cell in cells], dtype=float)
    discrepancy = 100.0 * np.asarray(
        [cell["centerMinusSurface"] for cell in cells], dtype=float
    )
    error = 100.0 * np.asarray([cell["centerError"] for cell in cells], dtype=float)
    color_limit = max(0.025, 1.15 * float(np.max(np.abs(discrepancy))))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    scatter = axes[0].scatter(
        theta, phi, c=discrepancy, cmap="coolwarm", vmin=-color_limit,
        vmax=color_limit, s=45,
    )
    fig.colorbar(scatter, ax=axes[0], label="cell center − surface [%]")
    axes[0].set_xlabel("$\\theta$ [deg]")
    axes[0].set_ylabel("sector-local $\\phi$ [deg]" if diagnostic.region["phiVariable"] == "sectorLocal"
                       else "global $\\phi$ [deg]")
    axes[0].set_title("where the surface misses a cell")
    axes[1].errorbar(
        theta, discrepancy, yerr=error, fmt="o", color="tab:blue",
        markersize=4, capsize=2,
    )
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].set_xlabel("$\\theta$ [deg]")
    axes[1].set_ylabel("cell center − surface [%]")
    axes[1].set_title("cell-level fit discrepancy")
    save_plot(
        fig,
        output_dir / f"{diagnostic.label}_cell_fit_discrepancy.png",
        f"Training-cell fit discrepancies: {diagnostic.label}",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)


def phi_slice_line_parameters(
    cells: list[dict[str, object]],
    phi_scale_deg: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Secondary weighted line fit used only to visualize FD slice trends."""
    if len(cells) < 2:
        return None
    phi = np.asarray([cell["phiMeanDeg"] for cell in cells], dtype=float) / phi_scale_deg
    residual = np.asarray([cell["center"] for cell in cells], dtype=float)
    error = np.asarray([cell["centerError"] for cell in cells], dtype=float)
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(residual)) or not np.all(
        np.isfinite(error) & (error > 0.0)
    ):
        return None
    matrix = np.column_stack((np.ones(phi.size), phi))
    weighted_matrix = matrix / error[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_matrix, residual / error, rcond=None
    )
    if rank != 2:
        return None
    covariance = np.linalg.pinv(weighted_matrix.T @ weighted_matrix)
    coefficient_errors = np.sqrt(np.diag(covariance))
    return coefficients, coefficient_errors


def _plot_slice_phi_coefficients(
    diagnostic: RegionDiagnostics,
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    region = diagnostic.region
    if region["basis"] != "polynomial" or any(
        int(term["phiPower"]) > 1 for term in region["terms"]
    ):
        return
    rows: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for theta_range, cells in _accepted_theta_slices(region["fit"]["acceptedProfileCells"]):
        result = phi_slice_line_parameters(cells, float(region["phiScaleDeg"]))
        if result is None:
            continue
        coefficients, coefficient_errors = result
        rows.append((
            float(np.mean([cell["thetaMeanDeg"] for cell in cells])),
            0.5 * (theta_range[1] - theta_range[0]),
            coefficients,
            coefficient_errors,
        ))
    if not rows:
        return
    theta = np.asarray([row[0] for row in rows], dtype=float)
    theta_half_width = np.asarray([row[1] for row in rows], dtype=float)
    coefficients = np.asarray([row[2] for row in rows], dtype=float)
    coefficient_errors = np.asarray([row[3] for row in rows], dtype=float)
    theta_curve = np.linspace(*region["thetaRangeDeg"], 150)
    at_phi_zero = evaluate_region(region, theta_curve, np.zeros(theta_curve.shape))
    at_phi_plus_scale = evaluate_region(
        region, theta_curve,
        np.full(theta_curve.shape, float(region["phiScaleDeg"])),
    )
    surface_terms = (at_phi_zero, at_phi_plus_scale - at_phi_zero)
    labels = (
        "intercept at local $\\phi=0$ [%]",
        "change per +30° local $\\phi$ [%]",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for index, axis in enumerate(axes):
        axis.errorbar(
            theta, 100.0 * coefficients[:, index],
            xerr=theta_half_width, yerr=100.0 * coefficient_errors[:, index],
            fmt="o", markersize=4, capsize=2, color="tab:blue",
            label="independent slice summaries",
        )
        axis.plot(
            theta_curve, 100.0 * surface_terms[index], color="tab:orange",
            linewidth=1.5, label="simultaneous fitted surface",
        )
        axis.set_xlabel("$\\theta$ [deg]")
        axis.set_ylabel(labels[index])
        axis.set_title("slice intercept" if index == 0 else "slice azimuthal slope")
        axis.legend(loc="best", fontsize="small")
    save_plot(
        fig,
        output_dir / f"{diagnostic.label}_phi_coefficients_vs_theta.png",
        f"Visual FD slice summaries (not a second correction fit): {diagnostic.label}",
        dataset_tag,
        beam_energy,
    )
    plt.close(fig)


def plot_diagnostics(
    diagnostics: list[RegionDiagnostics],
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for diagnostic in diagnostics:
        fit = diagnostic.region["fit"]
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        center_percent = 100.0 * diagnostic.profile_residual
        color_limit = max(0.25, 1.15 * float(np.max(np.abs(center_percent))))
        scatter = axes[0].scatter(
            diagnostic.profile_theta_deg,
            diagnostic.profile_phi_deg,
            c=center_percent,
            cmap="coolwarm",
            s=35,
            vmin=-color_limit,
            vmax=color_limit,
        )
        rejected = fit.get("rejectedProfileCells", [])
        rejected_with_entries = [
            cell for cell in rejected if cell.get("reason") != "rawEntries"
        ]
        if rejected_with_entries:
            axes[0].scatter(
                [0.5 * sum(cell["thetaRangeDeg"]) for cell in rejected_with_entries],
                [0.5 * sum(cell["phiRangeDeg"]) for cell in rejected_with_entries],
                marker="x",
                color="black",
                s=25,
                label="rejected peak",
            )
            axes[0].legend(loc="best", fontsize="small")
        axes[0].set_xlabel(r"$\theta$ [deg]")
        axes[0].set_ylabel(r"$\phi$ variable [deg]")
        axes[0].set_title("accepted profile cells")
        fig.colorbar(scatter, ax=axes[0], label=r"$100(p_{el}/p-1)$ [%]")

        common_range = np.quantile(diagnostic.residual_before, (0.005, 0.995))
        axes[1].hist(
            100.0 * diagnostic.residual_before,
            bins=100,
            range=tuple(100.0 * common_range),
            histtype="step",
            label="before",
        )
        axes[1].hist(
            100.0 * diagnostic.residual_after,
            bins=100,
            range=tuple(100.0 * common_range),
            histtype="step",
            label="after",
        )
        axes[1].set_xlabel(r"$100(p_{el}/p-1)$ [%]")
        axes[1].set_ylabel("candidates")
        axes[1].set_title("closure: broad view")
        axes[1].legend()

        core_limit = 100.0 * float(
            fit.get("peakSearchMaxAbsResidual", 0.10)
        )
        axes[2].hist(
            100.0 * diagnostic.residual_before,
            bins=100,
            range=(-core_limit, core_limit),
            histtype="step",
            label="before",
        )
        axes[2].hist(
            100.0 * diagnostic.residual_after,
            bins=100,
            range=(-core_limit, core_limit),
            histtype="step",
            label="after",
        )
        axes[2].axvline(0.0, color="black", linewidth=1)
        axes[2].set_xlabel(r"$100(p_{el}/p-1)$ [%]")
        axes[2].set_ylabel("candidates")
        axes[2].set_title("closure: peak view")
        axes[2].legend()

        axes[3].scatter(
            diagnostic.theta_deg,
            100.0 * diagnostic.residual_after,
            s=1,
            alpha=0.08,
        )
        axes[3].axhline(0.0, color="black", linewidth=1)
        axes[3].set_xlabel(r"$\theta$ [deg]")
        axes[3].set_ylabel("post-correction residual [%]")
        axes[3].set_title("residual closure vs theta")

        save_plot(
            fig,
            output_dir / f"{diagnostic.label}.png",
            f"Elastic momentum correction: {diagnostic.label}",
            dataset_tag,
            beam_energy,
        )
        plt.close(fig)
        _plot_profile_cell_map(diagnostic, output_dir, dataset_tag, beam_energy)
        _plot_phi_profiles(diagnostic, output_dir, dataset_tag, beam_energy)
        _plot_cell_fit_discrepancies(diagnostic, output_dir, dataset_tag, beam_energy)
        _plot_slice_phi_coefficients(diagnostic, output_dir, dataset_tag, beam_energy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive electron and proton fractional momentum corrections from "
            "selected elastic ep candidates."
        )
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--beam-energy", type=float, required=True)
    parser.add_argument("--torus", type=int, choices=(-1, 1), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("parameters/momentum/elastic_momentum_corrections.json"),
    )
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--particle", choices=("electron", "proton", "both"), default="both")
    parser.add_argument("--coplanarity-max-deg", type=float, default=3.0)
    parser.add_argument("--theta-balance-max-deg", type=float, default=2.0)
    parser.add_argument("--missing-energy-max-gev", type=float, default=0.75)
    parser.add_argument("--theta-min-deg", type=float)
    parser.add_argument("--theta-max-deg", type=float)
    parser.add_argument("--theta-trim-quantile", type=float, default=0.005)
    parser.add_argument("--residual-trim-quantile", type=float, default=0.01)
    parser.add_argument("--theta-bins", type=int, default=7)
    parser.add_argument("--phi-bins", type=int, default=7)
    parser.add_argument(
        "--profile-binning", choices=("fixed", "adaptive"), default="fixed",
        help=(
            "fixed uses one global phi grid; adaptive uses occupancy-quantile phi "
            "cells inside each theta slice"
        ),
    )
    parser.add_argument(
        "--max-theta-bin-width-deg", type=float,
        help=(
            "in adaptive mode, split quantile theta intervals wider than this "
            "physical angle"
        ),
    )
    parser.add_argument(
        "--target-cell-entries", type=int, default=500,
        help=(
            "adaptive-mode target raw population; sparse theta slices use fewer "
            "phi cells"
        ),
    )
    parser.add_argument("--min-phi-cells-per-theta", type=int, default=1)
    parser.add_argument("--theta-order", type=int, default=2)
    parser.add_argument("--fd-phi-order", type=int, default=2)
    parser.add_argument("--cd-fourier-harmonics", type=int, default=3)
    parser.add_argument("--min-bin-entries", type=int, default=40)
    parser.add_argument("--min-region-entries", type=int, default=800)
    parser.add_argument("--peak-search-max-abs-residual", type=float, default=0.10)
    parser.add_argument("--peak-seed-half-width", type=float, default=0.03)
    parser.add_argument("--core-sigma-clip", type=float, default=3.0)
    parser.add_argument("--min-core-fraction", type=float, default=0.20)
    parser.add_argument("--min-peak-significance", type=float, default=3.0)
    parser.add_argument("--max-core-width", type=float, default=0.05)
    parser.add_argument("--min-profile-cells-per-parameter", type=float, default=2.0)
    parser.add_argument("--max-condition-number", type=float, default=100.0)
    parser.add_argument("--max-abs-surface-correction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beam_energy <= 0.0:
        raise ValueError("beam energy must be positive")
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
        theta_order=args.theta_order,
        phi_order=args.fd_phi_order,
        cd_fourier_harmonics=args.cd_fourier_harmonics,
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
    particles = ("electron", "proton") if args.particle == "both" else (args.particle,)
    arrays = load_elastic_arrays(args.input_file, args.tree, args.max_rows)
    correction, diagnostics = derive_corrections(arrays, cfg, particles)
    correction["datasetTag"] = args.dataset_tag
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        json.dump(correction, output, indent=2, allow_nan=False)
        output.write("\n")
    print(
        f"Wrote {len(correction['regions'])} elastic momentum regions to {args.output}"
    )
    for skipped in correction["skippedRegions"]:
        print(f"Warning: skipped {skipped['region']}: {skipped['reason']}")
    if args.plot_dir:
        plot_diagnostics(diagnostics, args.plot_dir, args.dataset_tag, args.beam_energy)


if __name__ == "__main__":
    main()
