from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .elastic_momentum import (
    RAD_TO_DEG,
    apply_supported_particle_correction,
    evaluate_region,
    mode_seeded_core,
    region_support_mask,
    sector_local_phi,
    wrap_degrees,
)
from .eppi0_momentum_validation import (
    ELECTRON_MASS_GEV,
    PI0_MASS_GEV,
    PROTON_MASS_GEV,
    _four_vector,
    _m2,
    _unit_vectors,
    compute_eppi0_observables,
    load_aligned_selection_mask,
    load_eppi0_arrays,
)
from .plot_utils import save_plot


SURFACE_TERMS = {
    "constant": ((0, 0),),
    "momentum-linear": ((0, 0), (1, 0)),
    "theta-linear": ((0, 0), (0, 1)),
    "momentum-theta": ((0, 0), (1, 0), (0, 1), (1, 1)),
}


@dataclass(frozen=True)
class ExclusiveFitConfig:
    beam_energy: float
    torus: int
    model: str = "momentum-theta"
    momentum_bins: int = 8
    theta_bins: int = 8
    min_bin_entries: int = 200
    min_region_entries: int = 2_000
    min_cells_per_parameter: float = 2.0
    max_condition_number: float = 100.0
    peak_search_max_abs_residual: float = 0.30
    peak_seed_half_width: float = 0.06
    core_sigma_clip: float = 3.0
    min_core_fraction: float = 0.15
    min_peak_significance: float = 3.0
    max_core_width: float = 0.15
    max_abs_surface_correction: float = 0.30
    trim_quantile: float = 0.005
    minimum_q2: float = 1.0
    minimum_w: float = 2.0
    mgg_max_abs_gev: float = 0.08
    photon_fit_residual_max_gev: float = 0.50
    photon_design_condition_max: float = 100.0
    fd_by_sector: bool = False


def _read_parameters(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("schema") not in {
        "elastic_momentum_correction/v1", "particle_momentum_correction/v2",
    }:
        raise ValueError(f"unsupported momentum-correction schema in {path}")
    return payload


def _validate_parameter_provenance(
    parameters: dict[str, object],
    cfg: ExclusiveFitConfig,
    label: str,
) -> None:
    if not np.isclose(
        float(parameters["beamEnergyGeV"]), cfg.beam_energy,
        rtol=0.0, atol=1.0e-9,
    ):
        raise ValueError(f"{label} beam energy does not match requested data")
    if int(parameters["torus"]) != cfg.torus:
        raise ValueError(f"{label} torus polarity does not match requested data")


def expected_proton_momentum_eppi0(
    electron_momentum: np.ndarray,
    electron_theta: np.ndarray,
    electron_phi: np.ndarray,
    proton_theta: np.ndarray,
    proton_phi: np.ndarray,
    beam_energy: float,
    measured_proton_momentum: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve (target + q - p_proton)^2 = m_pi0^2 for |p_proton|.

    Only the corrected electron four-vector and measured proton direction enter
    the equation.  If both quadratic roots are physical, measured proton
    momentum is used solely to choose the branch, not to set the solution.
    """
    roots, valid = proton_momentum_roots_eppi0(
        electron_momentum, electron_theta, electron_phi,
        proton_theta, proton_phi, beam_energy,
    )
    ambiguity = np.sum(valid, axis=1) > 1
    if measured_proton_momentum is None:
        score = np.where(valid, -roots, np.inf)
    else:
        measured = np.asarray(measured_proton_momentum, dtype=float)
        score = np.where(valid, np.abs(roots - measured[:, None]), np.inf)
    choice = np.argmin(score, axis=1)
    expected = roots[np.arange(roots.shape[0]), choice]
    expected[~np.any(valid, axis=1)] = np.nan
    return expected, ambiguity


def proton_momentum_roots_eppi0(
    electron_momentum: np.ndarray,
    electron_theta: np.ndarray,
    electron_phi: np.ndarray,
    proton_theta: np.ndarray,
    proton_phi: np.ndarray,
    beam_energy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return both roots and their physical-validity masks for the pi0 constraint."""
    electron = _four_vector(
        electron_momentum, electron_theta, electron_phi, ELECTRON_MASS_GEV
    )
    entries = electron.shape[0]
    hadronic = np.zeros((entries, 4), dtype=float)
    hadronic[:, 0] = beam_energy + PROTON_MASS_GEV
    hadronic[:, 3] = beam_energy
    hadronic -= electron
    direction = _unit_vectors(proton_theta, proton_phi)
    eh = hadronic[:, 0]
    b = np.sum(hadronic[:, 1:] * direction, axis=1)
    hadronic_m2 = _m2(hadronic)
    c = 0.5 * (hadronic_m2 + PROTON_MASS_GEV**2 - PI0_MASS_GEV**2)
    denominator = np.square(eh) - np.square(b)
    discriminant = np.square(c) - PROTON_MASS_GEV**2 * denominator
    sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        roots = np.column_stack((
            (c * b - eh * sqrt_discriminant) / denominator,
            (c * b + eh * sqrt_discriminant) / denominator,
        ))
    proton_energy = np.sqrt(np.square(roots) + PROTON_MASS_GEV**2)
    valid = (
        np.isfinite(roots) & (roots > 0.0) &
        (eh[:, None] - proton_energy > PI0_MASS_GEV) &
        (c[:, None] + b[:, None] * roots >= 0.0) &
        (discriminant[:, None] >= 0.0) &
        (denominator[:, None] > 0.0)
    )
    return roots, valid


def expected_photon_energies_eppi0(
    pi0_four_vector: np.ndarray,
    gamma1_theta: np.ndarray,
    gamma1_phi: np.ndarray,
    gamma2_theta: np.ndarray,
    gamma2_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit two photon energies to a calculated pi0 four-vector.

    The four-vector components are given equal weight.  The returned residual is
    the Euclidean norm of the four-component closure vector in GeV; solutions
    with non-positive energy or singular photon geometry are marked invalid.
    """
    target = np.asarray(pi0_four_vector, dtype=float)
    n1 = _unit_vectors(gamma1_theta, gamma1_phi)
    n2 = _unit_vectors(gamma2_theta, gamma2_phi)
    dot = np.sum(n1 * n2, axis=1)
    cross = 1.0 + dot
    diagonal = 2.0
    determinant = diagonal**2 - np.square(cross)
    b1 = target[:, 0] + np.sum(n1 * target[:, 1:], axis=1)
    b2 = target[:, 0] + np.sum(n2 * target[:, 1:], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        energy1 = (diagonal * b1 - cross * b2) / determinant
        energy2 = (diagonal * b2 - cross * b1) / determinant
        condition = np.sqrt((diagonal + np.abs(cross)) /
                            (diagonal - np.abs(cross)))
    fitted = np.column_stack((
        energy1 + energy2,
        energy1[:, None] * n1 + energy2[:, None] * n2,
    ))
    residual = np.linalg.norm(fitted - target, axis=1)
    invalid = (
        ~np.isfinite(energy1) | ~np.isfinite(energy2) |
        ~np.isfinite(condition) | ~np.isfinite(residual) |
        (energy1 <= 0.0) | (energy2 <= 0.0) | (determinant <= 1.0e-10)
    )
    energy1[invalid] = np.nan
    energy2[invalid] = np.nan
    residual[invalid] = np.nan
    condition[invalid] = np.nan
    return energy1, energy2, residual, condition


def _profile_cells(
    momentum: np.ndarray,
    theta: np.ndarray,
    residual: np.ndarray,
    cfg: ExclusiveFitConfig,
) -> list[dict[str, object]]:
    momentum_edges = np.unique(np.quantile(
        momentum, np.linspace(0.0, 1.0, cfg.momentum_bins + 1)
    ))
    theta_edges = np.unique(np.quantile(
        theta, np.linspace(0.0, 1.0, cfg.theta_bins + 1)
    ))
    cells: list[dict[str, object]] = []
    for ip, (p_lo, p_hi) in enumerate(zip(momentum_edges[:-1], momentum_edges[1:])):
        p_mask = (momentum >= p_lo) & (
            (momentum <= p_hi) if ip == len(momentum_edges) - 2
            else (momentum < p_hi)
        )
        for itheta, (theta_lo, theta_hi) in enumerate(
            zip(theta_edges[:-1], theta_edges[1:])
        ):
            mask = p_mask & (theta >= theta_lo) & (
                (theta <= theta_hi) if itheta == len(theta_edges) - 2
                else (theta < theta_hi)
            )
            raw_entries = int(np.count_nonzero(mask))
            if raw_entries < cfg.min_bin_entries:
                continue
            estimate = mode_seeded_core(
                residual[mask],
                peak_search_max_abs_residual=cfg.peak_search_max_abs_residual,
                peak_seed_half_width=cfg.peak_seed_half_width,
                sigma_clip=cfg.core_sigma_clip,
            )
            if (
                estimate.entries < cfg.min_bin_entries or
                not np.isfinite(estimate.center) or
                not np.isfinite(estimate.error) or estimate.error <= 0.0 or
                not np.isfinite(estimate.width) or
                estimate.width > cfg.max_core_width or
                estimate.retained_fraction < cfg.min_core_fraction or
                estimate.peak_significance < cfg.min_peak_significance
            ):
                continue
            selected_indices = np.flatnonzero(mask)[estimate.mask]
            cells.append({
                "momentumRangeGeV": [float(p_lo), float(p_hi)],
                "thetaRangeDeg": [float(theta_lo), float(theta_hi)],
                "phiRangeDeg": [-180.0, 180.0],
                "momentumMeanGeV": float(np.mean(momentum[selected_indices])),
                "thetaMeanDeg": float(np.mean(theta[selected_indices])),
                "center": estimate.center,
                "centerError": estimate.error,
                "coreWidth": estimate.width,
                "coreEntries": estimate.entries,
                "rawEntries": raw_entries,
                "coreFraction": estimate.retained_fraction,
                "peakSignificance": estimate.peak_significance,
            })
    return cells


def fit_momentum_theta_region(
    *,
    pid: int,
    particle: str,
    detector: int,
    sector: int,
    momentum: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    residual: np.ndarray,
    cfg: ExclusiveFitConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if cfg.model not in SURFACE_TERMS:
        raise ValueError(f"unsupported surface model {cfg.model}")
    finite = (
        np.isfinite(momentum) & np.isfinite(theta_deg) & np.isfinite(phi_deg) &
        np.isfinite(residual) & (momentum > 0.0)
    )
    p = np.asarray(momentum, dtype=float)[finite]
    theta = np.asarray(theta_deg, dtype=float)[finite]
    phi = np.asarray(phi_deg, dtype=float)[finite]
    values = np.asarray(residual, dtype=float)[finite]
    if p.size < cfg.min_region_entries:
        raise ValueError(f"region has only {p.size} finite entries")
    q = cfg.trim_quantile
    p_min, p_max = np.quantile(p, (q, 1.0 - q))
    theta_min, theta_max = np.quantile(theta, (q, 1.0 - q))
    keep = (
        (p >= p_min) & (p <= p_max) &
        (theta >= theta_min) & (theta <= theta_max)
    )
    p, theta, phi, values = p[keep], theta[keep], phi[keep], values[keep]
    p_center, p_scale = 0.5 * (p_min + p_max), 0.5 * (p_max - p_min)
    theta_center = 0.5 * (theta_min + theta_max)
    theta_scale = 0.5 * (theta_max - theta_min)
    if p_scale <= 0.0 or theta_scale <= 0.0:
        raise ValueError("region has no finite momentum/theta span")
    cells = _profile_cells(p, theta, values, cfg)
    definitions = SURFACE_TERMS[cfg.model]
    minimum_cells = int(np.ceil(cfg.min_cells_per_parameter * len(definitions)))
    if len(cells) < minimum_cells:
        raise ValueError(
            f"region has {len(cells)} accepted cells; requires {minimum_cells}"
        )
    cell_p = np.asarray([cell["momentumMeanGeV"] for cell in cells])
    cell_theta = np.asarray([cell["thetaMeanDeg"] for cell in cells])
    cell_y = np.asarray([cell["center"] for cell in cells])
    cell_error = np.asarray([cell["centerError"] for cell in cells])
    pn = (cell_p - p_center) / p_scale
    tn = (cell_theta - theta_center) / theta_scale
    matrix = np.column_stack([
        np.power(pn, p_power) * np.power(tn, theta_power)
        for p_power, theta_power in definitions
    ])
    weighted = matrix / cell_error[:, None]
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        weighted, cell_y / cell_error, rcond=None
    )
    if rank != matrix.shape[1] or not np.all(np.isfinite(coefficients)):
        raise ValueError("momentum-theta fit is rank deficient")
    condition = float(singular_values[0] / singular_values[-1])
    if condition > cfg.max_condition_number:
        raise ValueError(
            f"weighted design condition number {condition:.6g} exceeds "
            f"{cfg.max_condition_number:.6g}"
        )
    terms = [
        {
            "momentumPower": int(p_power),
            "thetaPower": int(theta_power),
            "phiPower": 0,
            "coefficient": float(coefficient),
        }
        for (p_power, theta_power), coefficient in zip(definitions, coefficients)
    ]
    predicted_cells = matrix @ coefficients
    chi2 = float(np.sum(np.square((cell_y - predicted_cells) / cell_error)))
    ndof = len(cells) - len(definitions)
    region: dict[str, object] = {
        "pid": pid,
        "particle": particle,
        "detector": detector,
        "sector": sector,
        "momentumRangeGeV": [float(p_min), float(p_max)],
        "momentumCenterGeV": float(p_center),
        "momentumScaleGeV": float(p_scale),
        "thetaRangeDeg": [float(theta_min), float(theta_max)],
        "thetaCenterDeg": float(theta_center),
        "thetaScaleDeg": float(theta_scale),
        "phiRangeDeg": [-180.0, 180.0],
        "phiCenterDeg": 0.0,
        "phiScaleDeg": 180.0,
        "phiVariable": "global",
        "basis": "polynomial",
        "terms": terms,
        "supportCells": [{
            "momentumRangeGeV": cell["momentumRangeGeV"],
            "thetaRangeDeg": cell["thetaRangeDeg"],
            "phiRangeDeg": cell["phiRangeDeg"],
        } for cell in cells],
        "fit": {
            "model": cfg.model,
            "entries": int(p.size),
            "profileCells": len(cells),
            "parameters": len(definitions),
            "chi2": chi2,
            "ndof": ndof,
            "chi2PerNdf": chi2 / ndof if ndof > 0 else None,
            "weightedDesignConditionNumber": condition,
            "weightedDesignSingularValues": [float(x) for x in singular_values],
            "acceptedProfileCells": cells,
        },
    }
    support = region_support_mask(region, theta, phi, p)
    correction = np.zeros(p.shape, dtype=float)
    correction[support] = evaluate_region(
        region, theta[support], phi[support], p[support]
    )
    max_abs = float(np.max(np.abs(correction[support]))) if np.any(support) else 0.0
    if max_abs > cfg.max_abs_surface_correction:
        raise ValueError(
            f"surface reaches {max_abs:.6g}; exceeds "
            f"{cfg.max_abs_surface_correction:.6g}"
        )
    after = (1.0 + values) / (1.0 + correction) - 1.0
    region["fit"]["applicationSupportFraction"] = float(np.mean(support))
    region["fit"]["maxAbsSurfaceCorrectionOnCandidates"] = max_abs
    region["fit"]["beforeSupported"] = _summary(values[support])
    region["fit"]["afterSupported"] = _summary(after[support])
    return region, {
        "momentum": p,
        "theta": theta,
        "residualBefore": values,
        "residualAfter": after,
        "support": support,
    }


def _summary(values: np.ndarray) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"entries": 0, "mean": None, "rms": None, "median": None,
                "robustWidth": None}
    q16, median, q84 = np.quantile(finite, (0.16, 0.5, 0.84))
    return {
        "entries": int(finite.size),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "median": float(median),
        "robustWidth": float(0.5 * (q84 - q16)),
    }


def _base_mask(
    arrays: dict[str, np.ndarray],
    electron_momentum: np.ndarray,
    cfg: ExclusiveFitConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    observables = compute_eppi0_observables(
        arrays, electron_momentum, cfg.beam_energy
    )
    entries = electron_momentum.size
    mask = np.ones(entries, dtype=bool)
    for values in arrays.values():
        if np.asarray(values).shape == (entries,) and np.issubdtype(
            np.asarray(values).dtype, np.number
        ):
            mask &= np.isfinite(np.asarray(values, dtype=float))
    mask &= observables["Q2"] >= cfg.minimum_q2
    mask &= observables["W"] >= cfg.minimum_w
    mask &= np.abs(observables["mGG"] - PI0_MASS_GEV) <= cfg.mgg_max_abs_gev
    return mask, observables


def _region_groups(
    detector: np.ndarray,
    sector: np.ndarray,
    *,
    fd_by_sector: bool,
) -> list[tuple[int, int, np.ndarray]]:
    groups: list[tuple[int, int, np.ndarray]] = []
    for detector_id in sorted(int(x) for x in np.unique(detector)):
        detector_rows = detector == detector_id
        if detector_id == 1 and fd_by_sector:
            for sector_id in range(1, 7):
                rows = detector_rows & (sector == sector_id)
                if np.any(rows):
                    groups.append((detector_id, sector_id, rows))
        elif np.any(detector_rows):
            groups.append((detector_id, 0, detector_rows))
    return groups


def build_eppi0_particle_sample(
    arrays: dict[str, np.ndarray],
    cfg: ExclusiveFitConfig,
    *,
    electron_parameters: dict[str, object],
    particle: str,
    proton_parameters: dict[str, object] | None = None,
    external_selection_mask: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build the flat proton or pooled-photon residual sample for validation."""
    if particle not in {"proton", "photon"}:
        raise ValueError("particle sample must be proton or photon")
    _validate_parameter_provenance(electron_parameters, cfg, "electron parameters")
    if proton_parameters is not None:
        _validate_parameter_provenance(proton_parameters, cfg, "proton parameters")
    entries = int(np.asarray(arrays["electronP"]).size)
    electron_p, electron_support, _ = apply_supported_particle_correction(
        arrays["electronP"], arrays["electronTheta"], arrays["electronPhi"],
        arrays["electronDet"], arrays["electronSector"],
        pid=11, parameters=electron_parameters,
    )
    base, _ = _base_mask(arrays, electron_p, cfg)
    base &= electron_support
    if external_selection_mask is not None:
        external = np.asarray(external_selection_mask, dtype=bool)
        if external.shape != base.shape:
            raise ValueError("external selection mask does not match candidate entries")
        base &= external
    expected_proton, ambiguity = expected_proton_momentum_eppi0(
        electron_p,
        arrays["electronTheta"], arrays["electronPhi"],
        arrays["protonTheta"], arrays["protonPhi"], cfg.beam_energy,
        arrays["protonP"],
    )
    base &= np.isfinite(expected_proton)
    run_numbers = np.asarray(
        arrays.get("runNum", np.zeros(entries, dtype=int)), dtype=int
    )
    if particle == "proton":
        measured = np.asarray(arrays["protonP"], dtype=float)
        detector = np.asarray(arrays["protonDet"], dtype=int)
        sector = np.asarray(
            arrays.get("protonSector", np.zeros(entries, dtype=int)), dtype=int
        )
        if not cfg.fd_by_sector:
            sector = np.zeros(entries, dtype=int)
        residual = expected_proton / measured - 1.0
        selected = base & np.isfinite(residual) & (measured > 0.0)
        return {
            "momentum": measured[selected],
            "thetaDeg": np.asarray(arrays["protonTheta"])[selected] * RAD_TO_DEG,
            "phiDeg": wrap_degrees(
                np.asarray(arrays["protonPhi"])[selected] * RAD_TO_DEG
            ),
            "detector": detector[selected],
            "sector": sector[selected],
            "residual": residual[selected],
            "runNum": run_numbers[selected],
        }, {
            "inputCandidates": entries,
            "selectedCandidates": int(np.count_nonzero(selected)),
            "electronSupportFraction": float(np.mean(electron_support)),
            "ambiguousRootFraction": float(np.mean(ambiguity[base])) if np.any(base) else 0.0,
        }

    if proton_parameters is None:
        raise ValueError("photon sample requires proton_parameters")
    proton_sector = np.asarray(
        arrays.get("protonSector", np.zeros(entries, dtype=int)), dtype=int
    )
    proton_p, proton_support, _ = apply_supported_particle_correction(
        arrays["protonP"], arrays["protonTheta"], arrays["protonPhi"],
        arrays["protonDet"], proton_sector,
        pid=2212, parameters=proton_parameters,
    )
    electron = _four_vector(
        electron_p, arrays["electronTheta"], arrays["electronPhi"],
        ELECTRON_MASS_GEV,
    )
    proton = _four_vector(
        proton_p, arrays["protonTheta"], arrays["protonPhi"], PROTON_MASS_GEV
    )
    pi0_expected = np.zeros_like(electron)
    pi0_expected[:, 0] = cfg.beam_energy + PROTON_MASS_GEV
    pi0_expected[:, 3] = cfg.beam_energy
    pi0_expected -= electron + proton
    e1, e2, closure, condition = expected_photon_energies_eppi0(
        pi0_expected,
        arrays["gamma1Theta"], arrays["gamma1Phi"],
        arrays["gamma2Theta"], arrays["gamma2Phi"],
    )
    selected_event = (
        base & proton_support & np.isfinite(closure) &
        (closure <= cfg.photon_fit_residual_max_gev) &
        (condition <= cfg.photon_design_condition_max)
    )
    measured = np.concatenate((
        np.asarray(arrays["gamma1P"], dtype=float),
        np.asarray(arrays["gamma2P"], dtype=float),
    ))
    expected = np.concatenate((e1, e2))
    theta = np.concatenate((
        np.asarray(arrays["gamma1Theta"], dtype=float),
        np.asarray(arrays["gamma2Theta"], dtype=float),
    ))
    phi = np.concatenate((
        np.asarray(arrays["gamma1Phi"], dtype=float),
        np.asarray(arrays["gamma2Phi"], dtype=float),
    ))
    detector = np.concatenate((
        np.asarray(arrays["gamma1Det"], dtype=int),
        np.asarray(arrays["gamma2Det"], dtype=int),
    ))
    selected = np.concatenate((selected_event, selected_event))
    run = np.concatenate((run_numbers, run_numbers))
    residual = expected / measured - 1.0
    selected &= np.isfinite(residual) & (measured > 0.0)
    return {
        "momentum": measured[selected],
        "thetaDeg": theta[selected] * RAD_TO_DEG,
        "phiDeg": wrap_degrees(phi[selected] * RAD_TO_DEG),
        "detector": detector[selected],
        "sector": np.zeros(np.count_nonzero(selected), dtype=int),
        "residual": residual[selected],
        "runNum": run[selected],
    }, {
        "inputCandidates": entries,
        "selectedEvents": int(np.count_nonzero(selected_event)),
        "selectedPhotons": int(np.count_nonzero(selected)),
        "electronSupportFraction": float(np.mean(electron_support)),
        "protonSupportFraction": float(np.mean(proton_support)),
        "photonFitResidualGeV": _summary(closure[selected_event]),
        "photonDesignCondition": _summary(condition[selected_event]),
    }


def derive_eppi0_particle_corrections(
    arrays: dict[str, np.ndarray],
    cfg: ExclusiveFitConfig,
    *,
    electron_parameters: dict[str, object],
    particle: str,
    proton_parameters: dict[str, object] | None = None,
    external_selection_mask: np.ndarray | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, np.ndarray]]]:
    if particle not in {"proton", "photon", "both"}:
        raise ValueError("particle must be proton, photon, or both")
    _validate_parameter_provenance(electron_parameters, cfg, "electron parameters")
    if proton_parameters is not None:
        _validate_parameter_provenance(proton_parameters, cfg, "proton parameters")
    electron_p, electron_support, _ = apply_supported_particle_correction(
        arrays["electronP"], arrays["electronTheta"], arrays["electronPhi"],
        arrays["electronDet"], arrays["electronSector"],
        pid=11, parameters=electron_parameters,
    )
    base, observables = _base_mask(arrays, electron_p, cfg)
    base &= electron_support
    if external_selection_mask is not None:
        external_selection_mask = np.asarray(external_selection_mask, dtype=bool)
        if external_selection_mask.shape != base.shape:
            raise ValueError("external selection mask does not match candidate entries")
        base &= external_selection_mask
    expected_proton, ambiguous = expected_proton_momentum_eppi0(
        electron_p,
        arrays["electronTheta"], arrays["electronPhi"],
        arrays["protonTheta"], arrays["protonPhi"], cfg.beam_energy,
        arrays["protonP"],
    )
    base &= np.isfinite(expected_proton)

    regions: list[dict[str, object]] = []
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    skipped: list[dict[str, object]] = []
    proton_output: dict[str, object] | None = None
    if particle in {"proton", "both"}:
        residual = expected_proton / np.asarray(arrays["protonP"], dtype=float) - 1.0
        detector = np.asarray(arrays["protonDet"], dtype=int)
        sector = np.zeros(detector.shape, dtype=int)
        if "protonSector" in arrays:
            sector = np.asarray(arrays["protonSector"], dtype=int)
        for detector_id, sector_id, group in _region_groups(
            detector, sector, fd_by_sector=cfg.fd_by_sector
        ):
            rows = base & group
            label = f"proton_det{detector_id}" + (
                f"_sector{sector_id}" if sector_id else ""
            )
            try:
                region, diagnostic = fit_momentum_theta_region(
                    pid=2212, particle="proton", detector=detector_id,
                    sector=sector_id,
                    momentum=np.asarray(arrays["protonP"])[rows],
                    theta_deg=np.asarray(arrays["protonTheta"])[rows] * RAD_TO_DEG,
                    phi_deg=wrap_degrees(
                        np.asarray(arrays["protonPhi"])[rows] * RAD_TO_DEG
                    ),
                    residual=residual[rows], cfg=cfg,
                )
            except ValueError as error:
                skipped.append({"region": label, "reason": str(error)})
                continue
            regions.append(region)
            diagnostics[label] = diagnostic
        proton_output = {
            "schema": "particle_momentum_correction/v2",
            "correctionType": "fractionalMomentum",
            "beamEnergyGeV": cfg.beam_energy,
            "torus": cfg.torus,
            "regions": [region for region in regions if region["pid"] == 2212],
        }

    if particle in {"photon", "both"}:
        active_proton_parameters = proton_parameters or proton_output
        if not active_proton_parameters or not active_proton_parameters.get("regions"):
            raise ValueError(
                "photon derivation requires validated proton parameters or a "
                "successful proton fit in --particle both mode"
            )
        proton_sector = np.asarray(
            arrays.get("protonSector", np.zeros(base.shape, dtype=int)), dtype=int
        )
        proton_p, proton_support, _ = apply_supported_particle_correction(
            arrays["protonP"], arrays["protonTheta"], arrays["protonPhi"],
            arrays["protonDet"], proton_sector,
            pid=2212, parameters=active_proton_parameters,
        )
        photon_base = base & proton_support
        electron = _four_vector(
            electron_p, arrays["electronTheta"], arrays["electronPhi"],
            ELECTRON_MASS_GEV,
        )
        proton = _four_vector(
            proton_p, arrays["protonTheta"], arrays["protonPhi"],
            PROTON_MASS_GEV,
        )
        pi0_expected = np.zeros_like(electron)
        pi0_expected[:, 0] = cfg.beam_energy + PROTON_MASS_GEV
        pi0_expected[:, 3] = cfg.beam_energy
        pi0_expected -= electron + proton
        e1, e2, photon_fit_residual, photon_condition = expected_photon_energies_eppi0(
            pi0_expected,
            arrays["gamma1Theta"], arrays["gamma1Phi"],
            arrays["gamma2Theta"], arrays["gamma2Phi"],
        )
        photon_base &= np.isfinite(photon_fit_residual)
        photon_base &= photon_fit_residual <= cfg.photon_fit_residual_max_gev
        photon_base &= photon_condition <= cfg.photon_design_condition_max
        photon_measured = np.concatenate((
            np.asarray(arrays["gamma1P"], dtype=float),
            np.asarray(arrays["gamma2P"], dtype=float),
        ))
        photon_expected = np.concatenate((e1, e2))
        photon_theta = np.concatenate((
            np.asarray(arrays["gamma1Theta"], dtype=float),
            np.asarray(arrays["gamma2Theta"], dtype=float),
        ))
        photon_phi = np.concatenate((
            np.asarray(arrays["gamma1Phi"], dtype=float),
            np.asarray(arrays["gamma2Phi"], dtype=float),
        ))
        photon_detector = np.concatenate((
            np.asarray(arrays["gamma1Det"], dtype=int),
            np.asarray(arrays["gamma2Det"], dtype=int),
        ))
        photon_rows = np.concatenate((photon_base, photon_base))
        photon_residual = photon_expected / photon_measured - 1.0
        for detector_id in sorted(int(x) for x in np.unique(photon_detector)):
            rows = photon_rows & (photon_detector == detector_id)
            label = f"photon_det{detector_id}"
            try:
                region, diagnostic = fit_momentum_theta_region(
                    pid=22, particle="photon", detector=detector_id,
                    sector=0, momentum=photon_measured[rows],
                    theta_deg=photon_theta[rows] * RAD_TO_DEG,
                    phi_deg=wrap_degrees(photon_phi[rows] * RAD_TO_DEG),
                    residual=photon_residual[rows], cfg=cfg,
                )
            except ValueError as error:
                skipped.append({"region": label, "reason": str(error)})
                continue
            diagnostics[label] = diagnostic
            regions.append(region)

    selection = {
        "inputCandidates": int(base.size),
        "electronSupportedCandidates": int(np.count_nonzero(electron_support)),
        "baseSelectedCandidates": int(np.count_nonzero(base)),
        "ambiguousProtonRootCandidates": int(np.count_nonzero(base & ambiguous)),
        "minimumQ2GeV2": cfg.minimum_q2,
        "minimumWGeV": cfg.minimum_w,
        "mggMaxAbsGeV": cfg.mgg_max_abs_gev,
        "externalSelectionMask": external_selection_mask is not None,
        "externalSelectionEntries": (
            int(np.count_nonzero(external_selection_mask))
            if external_selection_mask is not None else None
        ),
        "uncorrectedObservables": {
            "Q2": _summary(observables["Q2"][base]),
            "W": _summary(observables["W"][base]),
            "mGG": _summary(observables["mGG"][base]),
        },
    }
    output: dict[str, object] = {
        "schema": "particle_momentum_correction/v2",
        "correctionType": "fractionalMomentum",
        "beamEnergyGeV": cfg.beam_energy,
        "torus": cfg.torus,
        "calibrationChannel": "ep-pi0",
        "stageOrder": ["protonEnergyLoss", "electronMomentum", "protonMomentum",
                       "photonEnergy"],
        "selection": selection,
        "fitConfiguration": {
            "model": cfg.model,
            "momentumBins": cfg.momentum_bins,
            "thetaBins": cfg.theta_bins,
            "minBinEntries": cfg.min_bin_entries,
            "minRegionEntries": cfg.min_region_entries,
            "maxConditionNumber": cfg.max_condition_number,
            "maxAbsSurfaceCorrection": cfg.max_abs_surface_correction,
            "fdBySector": cfg.fd_by_sector,
            "photonFitResidualMaxGeV": cfg.photon_fit_residual_max_gev,
            "photonDesignConditionMax": cfg.photon_design_condition_max,
        },
        "regions": regions,
        "skippedRegions": skipped,
    }
    return output, selection, diagnostics


def plot_exclusive_diagnostics(
    diagnostics: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for label, values in diagnostics.items():
        support = values["support"]
        before = 100.0 * values["residualBefore"][support]
        after = 100.0 * values["residualAfter"][support]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        scatter = axes[0].scatter(
            values["momentum"][support], values["theta"][support],
            c=before, s=2, alpha=0.25, cmap="coolwarm",
        )
        axes[0].set_xlabel("reconstructed momentum / energy [GeV]")
        axes[0].set_ylabel(r"$\theta$ [deg]")
        axes[0].set_title("supported candidates")
        fig.colorbar(scatter, ax=axes[0], label="pre-correction residual [%]")
        limit = float(np.quantile(np.abs(np.concatenate((before, after))), 0.995))
        limit = max(limit, 1.0)
        axes[1].hist(before, bins=100, range=(-limit, limit), histtype="step",
                     label="before")
        axes[1].hist(after, bins=100, range=(-limit, limit), histtype="step",
                     label="after")
        axes[1].axvline(0.0, color="black", linewidth=1)
        axes[1].set_xlabel("kinematic residual [%]")
        axes[1].set_ylabel("candidates")
        axes[1].legend()
        axes[1].set_title("same supported candidates")
        axes[2].scatter(values["momentum"][support], after, s=1, alpha=0.08)
        axes[2].axhline(0.0, color="black", linewidth=1)
        axes[2].set_xlabel("reconstructed momentum / energy [GeV]")
        axes[2].set_ylabel("post-correction residual [%]")
        axes[2].set_title("closure versus momentum")
        save_plot(
            fig, output_dir / f"{label}.png",
            f"ep-pi0 kinematic momentum correction: {label}",
            dataset_tag, beam_energy,
        )
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive proton and photon fractional momentum/energy corrections "
            "from ep-pi0 exclusive kinematic constraints."
        )
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--beam-energy", type=float, required=True)
    parser.add_argument("--torus", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--electron-parameters", type=Path, required=True)
    parser.add_argument("--proton-parameters", type=Path)
    parser.add_argument("--selection-mask", type=Path)
    parser.add_argument("--selection-mask-key", default="mask")
    parser.add_argument("--particle", choices=("proton", "photon", "both"),
                        default="proton")
    parser.add_argument("--model", choices=tuple(SURFACE_TERMS),
                        default="momentum-theta")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--momentum-bins", type=int, default=8)
    parser.add_argument("--theta-bins", type=int, default=8)
    parser.add_argument("--min-bin-entries", type=int, default=200)
    parser.add_argument("--min-region-entries", type=int, default=2_000)
    parser.add_argument("--min-cells-per-parameter", type=float, default=2.0)
    parser.add_argument("--max-condition-number", type=float, default=100.0)
    parser.add_argument("--peak-search-max-abs-residual", type=float, default=0.30)
    parser.add_argument("--peak-seed-half-width", type=float, default=0.06)
    parser.add_argument("--max-core-width", type=float, default=0.15)
    parser.add_argument("--max-abs-surface-correction", type=float, default=0.30)
    parser.add_argument("--min-q2", type=float, default=1.0)
    parser.add_argument("--min-w", type=float, default=2.0)
    parser.add_argument("--mgg-max-abs-gev", type=float, default=0.08)
    parser.add_argument("--photon-fit-residual-max-gev", type=float, default=0.50)
    parser.add_argument("--photon-design-condition-max", type=float, default=100.0)
    parser.add_argument("--fd-by-sector", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExclusiveFitConfig(
        beam_energy=args.beam_energy,
        torus=args.torus,
        model=args.model,
        momentum_bins=args.momentum_bins,
        theta_bins=args.theta_bins,
        min_bin_entries=args.min_bin_entries,
        min_region_entries=args.min_region_entries,
        min_cells_per_parameter=args.min_cells_per_parameter,
        max_condition_number=args.max_condition_number,
        peak_search_max_abs_residual=args.peak_search_max_abs_residual,
        peak_seed_half_width=args.peak_seed_half_width,
        max_core_width=args.max_core_width,
        max_abs_surface_correction=args.max_abs_surface_correction,
        minimum_q2=args.min_q2,
        minimum_w=args.min_w,
        mgg_max_abs_gev=args.mgg_max_abs_gev,
        photon_fit_residual_max_gev=args.photon_fit_residual_max_gev,
        photon_design_condition_max=args.photon_design_condition_max,
        fd_by_sector=args.fd_by_sector,
    )
    arrays = load_eppi0_arrays(args.input_file, args.tree, args.max_rows)
    selection_mask = None
    if args.selection_mask is not None:
        selection_mask = load_aligned_selection_mask(
            args.selection_mask,
            int(np.asarray(arrays["electronP"]).size),
            args.selection_mask_key,
            allow_prefix=args.max_rows is not None,
        )
    output, _, diagnostics = derive_eppi0_particle_corrections(
        arrays, cfg,
        electron_parameters=_read_parameters(args.electron_parameters),
        proton_parameters=(
            _read_parameters(args.proton_parameters)
            if args.proton_parameters else None
        ),
        particle=args.particle,
        external_selection_mask=selection_mask,
    )
    output["datasetTag"] = args.dataset_tag
    output["upstreamParameters"] = {
        "electron": str(args.electron_parameters),
        "proton": str(args.proton_parameters) if args.proton_parameters else None,
        "selectionMask": str(args.selection_mask) if args.selection_mask else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(output['regions'])} ep-pi0 correction regions to {args.output}")
    for skipped in output["skippedRegions"]:
        print(f"Warning: skipped {skipped['region']}: {skipped['reason']}")
    if args.plot_dir:
        plot_exclusive_diagnostics(
            diagnostics, args.plot_dir, args.dataset_tag, args.beam_energy
        )
