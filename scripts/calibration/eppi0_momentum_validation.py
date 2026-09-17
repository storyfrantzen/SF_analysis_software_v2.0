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
from .elastic_phase_space_coverage import summarize_exclusivity_cuts
from .plot_utils import save_plot
from .root_arrays import arrays_from_dataframe, has_column, load_dataframe


ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from eppi0.binning import AnalysisBinning, from_config  # noqa: E402
from eppi0.exclusivity import ExclusivityCuts, apply_cuts, load_cuts  # noqa: E402
from eppi0.topology import ft_photon_count  # noqa: E402


ELECTRON_MASS_GEV = 0.00051099895
PROTON_MASS_GEV = 0.9382720813
PI0_MASS_GEV = 0.1349768

ROLE_COLUMNS = {
    "electronP": ("electronP",),
    "electronTheta": ("electronTheta",),
    "electronPhi": ("electronPhi",),
    "electronDet": ("electronDet", "eDet"),
    "electronSector": ("electronSector", "eSector"),
    "protonP": ("protonP",),
    "protonTheta": ("protonTheta",),
    "protonPhi": ("protonPhi",),
    "protonDet": ("protonDet", "pDet"),
    "gamma1P": ("gamma1P",),
    "gamma1Theta": ("gamma1Theta",),
    "gamma1Phi": ("gamma1Phi",),
    "gamma1Det": ("gamma1Det", "g1Det"),
    "gamma2P": ("gamma2P",),
    "gamma2Theta": ("gamma2Theta",),
    "gamma2Phi": ("gamma2Phi",),
    "gamma2Det": ("gamma2Det", "g2Det"),
}

REFERENCE_COLUMNS = (
    "Q2", "nu", "xB", "y", "W", "t", "t_pi0", "trentoPhi",
    "pi0_p", "pi0_theta", "pi0_phi", "pi0_deltaPhi", "pi0_thetaX",
    "m_gg", "m2_miss", "m2_epX", "m2_epi0X", "m_eggX", "E_miss",
    "pT_miss", "theta_e_g1", "theta_e_g2", "theta_g1_g2",
)

REFERENCE_TO_OBSERVABLE = {
    "Q2": "Q2",
    "nu": "nu",
    "xB": "xB",
    "y": "y",
    "W": "W",
    "t": "t",
    "t_pi0": "tPi0",
    "trentoPhi": "trentoPhi",
    "pi0_p": "pi0P",
    "pi0_theta": "pi0Theta",
    "pi0_phi": "pi0Phi",
    "pi0_deltaPhi": "pi0DeltaPhi",
    "pi0_thetaX": "pi0ThetaX",
    "m_gg": "mGG",
    "m2_miss": "m2Miss",
    "m2_epX": "m2EpX",
    "m2_epi0X": "m2EPi0X",
    "m_eggX": "mEggX",
    "E_miss": "missingEnergy",
    "pT_miss": "missingPt",
    "theta_e_g1": "thetaEGamma1",
    "theta_e_g2": "thetaEGamma2",
    "theta_g1_g2": "thetaGamma1Gamma2",
}

ANGLE_OBSERVABLES = {
    "trentoPhi", "pi0Theta", "pi0Phi", "pi0DeltaPhi", "pi0ThetaX",
    "missingTheta", "missingPhi", "thetaEGamma1", "thetaEGamma2",
    "thetaGamma1Gamma2",
}

INVARIANT_OBSERVABLES = (
    "electronTheta", "electronPhi", "t", "pi0P", "pi0Theta", "pi0Phi",
    "mGG", "thetaEGamma1", "thetaEGamma2", "thetaGamma1Gamma2",
)

EXCLUSIVITY_VALUE_NAMES = {
    "rec_m_gg": "mGG",
    "rec_pT_miss": "missingPt",
    "rec_m2_epX": "m2EpX",
    "rec_m_eggX": "mEggX",
    "rec_E_miss": "missingEnergy",
    "rec_m2_miss": "m2Miss",
}

EXPECTED_CENTERS = {
    "mGG": PI0_MASS_GEV,
    "missingPt": 0.0,
    "m2EpX": PI0_MASS_GEV**2,
    "mEggX": PROTON_MASS_GEV,
    "missingEnergy": 0.0,
    "m2Miss": 0.0,
    "pi0DeltaPhi": 0.0,
    "pi0ThetaX": 0.0,
    "deltaT": 0.0,
}

SUMMARY_QUANTITIES = (
    "electronP", "electronEnergy", "Q2", "nu", "xB", "y", "W", "t",
    "tPi0", "deltaT", "trentoPhi", "pi0P", "pi0Theta", "mGG", "missingP",
    "missingPt", "missingEnergy", "m2Miss", "m2EpX", "m2EPi0X", "mEggX",
    "pi0DeltaPhi", "pi0ThetaX", "thetaEGamma1", "thetaEGamma2",
    "thetaGamma1Gamma2",
)

PLOT_QUANTITIES = (
    "Q2", "W", "xB", "deltaT", "mGG", "missingPt", "missingEnergy",
    "m2Miss", "m2EpX", "mEggX", "pi0DeltaPhi", "pi0ThetaX",
)

PLOT_LABELS = {
    "Q2": r"$Q^2$ [GeV$^2$]",
    "W": r"$W$ [GeV]",
    "xB": r"$x_B$",
    "tPi0": r"$-t_{e\pi^0}$ [GeV$^2$]",
    "deltaT": r"$(-t)_{e\pi^0}-(-t)_p$ [GeV$^2$]",
    "mGG": r"$m_{\gamma\gamma}$ [GeV]",
    "missingPt": r"$p_T^{miss}$ [GeV]",
    "missingEnergy": r"$E_{miss}$ [GeV]",
    "m2Miss": r"$M_X^2(ep\pi^0)$ [GeV$^2$]",
    "m2EpX": r"$M_X^2(ep)$ [GeV$^2$]",
    "mEggX": r"$M_X(e\gamma\gamma)$ [GeV]",
    "pi0DeltaPhi": r"$\Delta\phi(\pi^0,X_{ep})$ [deg]",
    "pi0ThetaX": r"$\theta(\pi^0,X_{ep})$ [deg]",
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


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


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _wrap_radians(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _four_vector(
    momentum: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    mass: float,
) -> np.ndarray:
    p = np.asarray(momentum, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    transverse = p * np.sin(theta)
    return np.column_stack((
        np.sqrt(np.square(p) + mass**2),
        transverse * np.cos(phi),
        transverse * np.sin(phi),
        p * np.cos(theta),
    ))


def _m2(vector: np.ndarray) -> np.ndarray:
    return np.square(vector[:, 0]) - np.sum(np.square(vector[:, 1:]), axis=1)


def _spatial_magnitude(vector: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(vector[:, 1:]), axis=1))


def _vector_phi(vector: np.ndarray) -> np.ndarray:
    return np.arctan2(vector[:, 2], vector[:, 1])


def _vector_theta(vector: np.ndarray) -> np.ndarray:
    transverse = np.hypot(vector[:, 1], vector[:, 2])
    return np.arctan2(transverse, vector[:, 3])


def _unit_vectors(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return np.column_stack((
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ))


def _angle_between_units(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cosine = np.sum(left * right, axis=1)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _trento_phi(
    electron: np.ndarray,
    proton: np.ndarray,
    beam_energy: float,
) -> np.ndarray:
    beam_vector = np.zeros_like(electron[:, 1:])
    beam_vector[:, 2] = 1.0
    q = np.column_stack((-electron[:, 1], -electron[:, 2], -electron[:, 3]))
    q[:, 2] += beam_energy
    q_norm = np.linalg.norm(q, axis=1)
    q_unit = np.divide(
        q, q_norm[:, None], out=np.full_like(q, np.nan), where=q_norm[:, None] > 0
    )
    n_lepton = np.cross(beam_vector, electron[:, 1:])
    n_hadron = np.cross(proton[:, 1:], q)
    nl_norm = np.linalg.norm(n_lepton, axis=1)
    nh_norm = np.linalg.norm(n_hadron, axis=1)
    n_lepton = np.divide(
        n_lepton, nl_norm[:, None], out=np.full_like(n_lepton, np.nan),
        where=nl_norm[:, None] > 0,
    )
    n_hadron = np.divide(
        n_hadron, nh_norm[:, None], out=np.full_like(n_hadron, np.nan),
        where=nh_norm[:, None] > 0,
    )
    cosine = np.sum(n_lepton * n_hadron, axis=1)
    sine = np.sum(q_unit * np.cross(n_lepton, n_hadron), axis=1)
    return np.arctan2(sine, cosine)


def compute_eppi0_observables(
    arrays: dict[str, np.ndarray],
    electron_momentum: np.ndarray,
    beam_energy: float,
) -> dict[str, np.ndarray]:
    """Recompute the post-process ep-pi0 kinematics with one electron p array."""
    entries = int(np.asarray(electron_momentum).size)
    electron = _four_vector(
        electron_momentum, arrays["electronTheta"], arrays["electronPhi"],
        ELECTRON_MASS_GEV,
    )
    proton = _four_vector(
        arrays["protonP"], arrays["protonTheta"], arrays["protonPhi"],
        PROTON_MASS_GEV,
    )
    gamma1 = _four_vector(
        arrays["gamma1P"], arrays["gamma1Theta"], arrays["gamma1Phi"], 0.0,
    )
    gamma2 = _four_vector(
        arrays["gamma2P"], arrays["gamma2Theta"], arrays["gamma2Phi"], 0.0,
    )
    pi0 = gamma1 + gamma2
    beam = np.zeros((entries, 4), dtype=float)
    beam[:, 0] = beam_energy
    beam[:, 3] = beam_energy
    target = np.zeros((entries, 4), dtype=float)
    target[:, 0] = PROTON_MASS_GEV
    q = beam - electron
    missing = beam + target - electron - proton - pi0
    ep_x = beam + target - electron - proton
    epi0_x = beam + target - electron - pi0

    q2 = -_m2(q)
    nu = beam_energy - electron[:, 0]
    xb = np.divide(
        q2, 2.0 * PROTON_MASS_GEV * nu,
        out=np.full(entries, np.nan), where=nu != 0.0,
    )
    w = np.sqrt(np.maximum(0.0, _m2(target + q)))
    pi0_m2 = _m2(pi0)
    epi0_m2 = _m2(epi0_x)
    proton_t = -_m2(target - proton)
    pi0_t = -_m2(beam - electron - pi0)

    electron_unit = _unit_vectors(arrays["electronTheta"], arrays["electronPhi"])
    gamma1_unit = _unit_vectors(arrays["gamma1Theta"], arrays["gamma1Phi"])
    gamma2_unit = _unit_vectors(arrays["gamma2Theta"], arrays["gamma2Phi"])
    pi0_p = _spatial_magnitude(pi0)
    pi0_unit = np.divide(
        pi0[:, 1:], pi0_p[:, None], out=np.full_like(pi0[:, 1:], np.nan),
        where=pi0_p[:, None] > 0,
    )
    ep_x_p = _spatial_magnitude(ep_x)
    ep_x_unit = np.divide(
        ep_x[:, 1:], ep_x_p[:, None], out=np.full_like(ep_x[:, 1:], np.nan),
        where=ep_x_p[:, None] > 0,
    )

    return {
        "electronP": np.asarray(electron_momentum, dtype=float),
        "electronEnergy": electron[:, 0],
        "electronTheta": np.asarray(arrays["electronTheta"], dtype=float),
        "electronPhi": np.asarray(arrays["electronPhi"], dtype=float),
        "Q2": q2,
        "nu": nu,
        "xB": xb,
        "y": nu / beam_energy,
        "W": w,
        "t": proton_t,
        "tPi0": pi0_t,
        "deltaT": pi0_t - proton_t,
        "trentoPhi": _trento_phi(electron, proton, beam_energy),
        "pi0P": pi0_p,
        "pi0Theta": _vector_theta(pi0),
        "pi0Phi": _vector_phi(pi0),
        "pi0DeltaPhi": _wrap_radians(_vector_phi(pi0) - _vector_phi(ep_x)),
        "pi0ThetaX": _angle_between_units(pi0_unit, ep_x_unit),
        "mGG": np.sqrt(np.maximum(0.0, pi0_m2)),
        "m2Miss": _m2(missing),
        "m2EpX": _m2(ep_x),
        "m2EPi0X": epi0_m2,
        "mEggX": np.where(epi0_m2 >= 0.0, np.sqrt(np.maximum(0.0, epi0_m2)), np.nan),
        "missingEnergy": missing[:, 0],
        "missingPt": np.hypot(missing[:, 1], missing[:, 2]),
        "missingP": _spatial_magnitude(missing),
        "missingTheta": _vector_theta(missing),
        "missingPhi": _vector_phi(missing),
        "thetaEGamma1": _angle_between_units(electron_unit, gamma1_unit),
        "thetaEGamma2": _angle_between_units(electron_unit, gamma2_unit),
        "thetaGamma1Gamma2": _angle_between_units(gamma1_unit, gamma2_unit),
    }


def load_eppi0_arrays(
    input_file: Path,
    tree: str,
    max_rows: int | None,
) -> dict[str, np.ndarray]:
    df = load_dataframe(input_file, tree)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for destination, candidates in ROLE_COLUMNS.items():
        source = next((name for name in candidates if has_column(df, name)), None)
        if source is None:
            missing.append(f"{destination} ({'/'.join(candidates)})")
        else:
            resolved[destination] = source
    if missing:
        raise RuntimeError(
            f"{tree} is missing selected-particle branches: {', '.join(missing)}"
        )

    optional = [
        name for name in ("runNum", "eventNum", *REFERENCE_COLUMNS)
        if has_column(df, name) and name not in resolved.values()
    ]
    source_columns = list(dict.fromkeys([*resolved.values(), *optional]))
    loaded = arrays_from_dataframe(df, source_columns, max_rows=max_rows)
    arrays = {destination: loaded[source] for destination, source in resolved.items()}
    arrays.update({name: loaded[name] for name in optional})
    return arrays


def load_aligned_selection_mask(
    path: Path,
    expected_entries: int,
    key: str = "mask",
    *,
    allow_prefix: bool = False,
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
                    f"selection-mask NPZ has no '{key}' array; available: {loaded.files}"
                )
        finally:
            loaded.close()
    else:
        values = loaded
    mask = np.asarray(values)
    valid_size = mask.size == expected_entries or (allow_prefix and mask.size >= expected_entries)
    if mask.ndim != 1 or not valid_size:
        relation = "at least" if allow_prefix else "exactly"
        raise ValueError(
            f"selection mask has shape {mask.shape}; expected {relation} "
            f"{expected_entries} entries"
        )
    mask = mask[:expected_entries]
    if mask.dtype != np.bool_:
        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError("selection mask must contain only booleans or 0/1 values")
        mask = mask.astype(bool)
    return mask


def apply_supported_electron_correction(
    arrays: dict[str, np.ndarray],
    parameters: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    momentum = np.asarray(arrays["electronP"], dtype=float)
    theta_deg = np.asarray(arrays["electronTheta"], dtype=float) * RAD_TO_DEG
    sectors = np.asarray(arrays["electronSector"], dtype=int)
    local_phi = sector_local_phi(np.asarray(arrays["electronPhi"], dtype=float), sectors)
    detectors = np.asarray(arrays["electronDet"], dtype=int)
    support = np.zeros(momentum.size, dtype=bool)
    correction = np.zeros(momentum.size, dtype=float)
    regions: dict[tuple[int, int], dict[str, object]] = {}
    for region in parameters.get("regions", []):
        if int(region.get("pid", 0)) != 11:
            continue
        key = (int(region["detector"]), int(region["sector"]))
        if key in regions:
            raise ValueError(f"duplicate electron correction region det/sector={key}")
        regions[key] = region
    for sector in range(1, 7):
        region = regions.get((1, sector))
        if region is None:
            continue
        sector_rows = (detectors == 1) & (sectors == sector)
        rows = np.flatnonzero(sector_rows)
        if rows.size == 0:
            continue
        in_support = region_support_mask(
            region, theta_deg[rows], local_phi[rows]
        )
        supported_rows = rows[in_support]
        support[supported_rows] = True
        if supported_rows.size:
            correction[supported_rows] = evaluate_region(
                region, theta_deg[supported_rows], local_phi[supported_rows]
            )
    corrected = momentum * (1.0 + correction)
    invalid = support & (~np.isfinite(corrected) | (corrected <= 0.0))
    if np.any(invalid):
        raise ValueError(
            f"electron correction produced {np.count_nonzero(invalid)} invalid momenta"
        )
    corrected[~support] = momentum[~support]
    return corrected, support, correction, local_phi


def _base_analysis_mask(
    arrays: dict[str, np.ndarray],
    observables: dict[str, np.ndarray],
    *,
    minimum_electron_p: float,
    minimum_q2: float,
    minimum_w: float,
) -> np.ndarray:
    required = tuple(ROLE_COLUMNS)
    entries = int(np.asarray(arrays["electronP"]).size)
    mask = np.ones(entries, dtype=bool)
    for name in required:
        mask &= np.isfinite(np.asarray(arrays[name], dtype=float))
    mask &= np.asarray(observables["electronP"]) >= minimum_electron_p
    mask &= np.asarray(observables["Q2"]) >= minimum_q2
    mask &= np.asarray(observables["W"]) >= minimum_w
    mask &= np.asarray(arrays["electronDet"], dtype=int) == 1
    sectors = np.asarray(arrays["electronSector"], dtype=int)
    mask &= (sectors >= 1) & (sectors <= 6)
    mask &= np.isin(np.asarray(arrays["protonDet"], dtype=int), (1, 2))
    return mask


def _individual_base_threshold_masks(
    arrays: dict[str, np.ndarray],
    observables: dict[str, np.ndarray],
    *,
    minimum_electron_p: float,
    minimum_q2: float,
    minimum_w: float,
) -> dict[str, np.ndarray]:
    valid = _base_analysis_mask(
        arrays,
        observables,
        minimum_electron_p=-np.inf,
        minimum_q2=-np.inf,
        minimum_w=-np.inf,
    )
    return {
        "electronP": valid & (observables["electronP"] >= minimum_electron_p),
        "Q2": valid & (observables["Q2"] >= minimum_q2),
        "W": valid & (observables["W"] >= minimum_w),
    }


def _exclusivity_values(observables: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        persisted: np.asarray(observables[internal], dtype=float)
        for persisted, internal in EXCLUSIVITY_VALUE_NAMES.items()
    }


def _apply_persisted_exclusivity(
    cuts: ExclusivityCuts,
    binning: AnalysisBinning,
    arrays: dict[str, np.ndarray],
    observables: dict[str, np.ndarray],
    *,
    only_variable: str | None = None,
) -> np.ndarray:
    iq2, ixb, it, _ = binning.indices(
        observables["Q2"], observables["xB"], observables["t"],
        observables["trentoPhi"],
    )
    ft_photons = ft_photon_count(arrays["gamma1Det"], arrays["gamma2Det"])
    values = _exclusivity_values(observables)
    absent = set(cuts.variables).difference(values)
    if absent:
        raise ValueError(
            "validator cannot recompute exclusivity variables: "
            + ", ".join(sorted(absent))
        )
    excluded = ()
    if only_variable is not None:
        if only_variable not in cuts.variables:
            raise ValueError(f"unknown exclusivity variable: {only_variable}")
        excluded = tuple(name for name in cuts.variables if name != only_variable)
    return apply_cuts(
        cuts, values, arrays["protonDet"], ft_photons, iq2, ixb, it,
        exclude_variables=excluded,
    )


def _distribution_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    selected = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {
            "entries": 0, "mean": None, "std": None, "median": None,
            "q16": None, "q84": None, "robustWidth": None,
        }
    q16, median, q84 = np.quantile(selected, [0.16, 0.5, 0.84])
    return {
        "entries": int(selected.size),
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
        "median": float(median),
        "q16": float(q16),
        "q84": float(q84),
        "robustWidth": float(0.5 * (q84 - q16)),
    }


def _paired_quantity_summary(
    before: np.ndarray,
    after: np.ndarray,
    mask: np.ndarray,
    expected_center: float | None,
) -> dict[str, object]:
    result = {
        "before": _distribution_summary(before, mask),
        "after": _distribution_summary(after, mask),
        "delta": _distribution_summary(np.asarray(after) - np.asarray(before), mask),
    }
    if expected_center is not None:
        valid = np.asarray(mask, dtype=bool)
        valid &= np.isfinite(before) & np.isfinite(after)
        result["expectedCenter"] = expected_center
        if np.any(valid):
            result["rmsFromExpectedBefore"] = float(
                np.sqrt(np.mean(np.square(np.asarray(before)[valid] - expected_center)))
            )
            result["rmsFromExpectedAfter"] = float(
                np.sqrt(np.mean(np.square(np.asarray(after)[valid] - expected_center)))
            )
        else:
            result["rmsFromExpectedBefore"] = None
            result["rmsFromExpectedAfter"] = None
    return result


def _cohort_summary(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, object]:
    return {
        "entries": int(np.count_nonzero(mask)),
        "quantities": {
            name: _paired_quantity_summary(
                before[name], after[name], mask, EXPECTED_CENTERS.get(name)
            )
            for name in SUMMARY_QUANTITIES
        },
    }


def _migration_summary(before: np.ndarray, after: np.ndarray) -> dict[str, object]:
    before = np.asarray(before, dtype=bool)
    after = np.asarray(after, dtype=bool)
    both = int(np.count_nonzero(before & after))
    lost = int(np.count_nonzero(before & ~after))
    gained = int(np.count_nonzero(~before & after))
    neither = int(np.count_nonzero(~before & ~after))
    union = both + lost + gained
    return {
        "entries": int(before.size),
        "beforeSelected": both + lost,
        "afterSelected": both + gained,
        "bothSelected": both,
        "lostAfterCorrection": lost,
        "gainedAfterCorrection": gained,
        "neitherSelected": neither,
        "netSelectedChange": gained - lost,
        "retainedFractionOfBefore": _fraction(both, both + lost),
        "jaccard": _fraction(both, union),
    }


def _maximum_difference(
    before: np.ndarray,
    after: np.ndarray,
    mask: np.ndarray,
    *,
    circular: bool = False,
) -> dict[str, object]:
    mask = np.asarray(mask, dtype=bool)
    difference = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    if circular:
        difference = _wrap_radians(difference)
    valid = mask & np.isfinite(difference)
    if not np.any(valid):
        return {"entries": 0, "maximumAbsoluteDifference": None, "rmsDifference": None}
    values = difference[valid]
    return {
        "entries": int(values.size),
        "maximumAbsoluteDifference": float(np.max(np.abs(values))),
        "rmsDifference": float(np.sqrt(np.mean(np.square(values)))),
    }


def _audit_invariants(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    support: np.ndarray,
    tolerance: float,
) -> dict[str, object]:
    entries = support.size
    all_rows = np.ones(entries, dtype=bool)
    unchanged = {
        name: _maximum_difference(
            before[name], after[name], all_rows, circular=name in ANGLE_OBSERVABLES
        )
        for name in INVARIANT_OBSERVABLES
    }
    unsupported = {
        name: _maximum_difference(
            before[name], after[name], ~support, circular=name in ANGLE_OBSERVABLES
        )
        for name in before
    }
    checked = [
        item["maximumAbsoluteDifference"]
        for item in (*unchanged.values(), *unsupported.values())
        if item["maximumAbsoluteDifference"] is not None
    ]
    maximum = max(checked, default=0.0)
    return {
        "tolerance": tolerance,
        "passed": bool(maximum <= tolerance),
        "maximumObservedAbsoluteDifference": float(maximum),
        "electronMagnitudeInvariantQuantities": unchanged,
        "unsupportedEventQuantities": unsupported,
    }


def _audit_reference_branches(
    arrays: dict[str, np.ndarray],
    before: dict[str, np.ndarray],
    tolerance: float,
) -> dict[str, object]:
    quantities: dict[str, object] = {}
    for branch, observable in REFERENCE_TO_OBSERVABLE.items():
        if branch not in arrays:
            continue
        values = _maximum_difference(
            np.asarray(arrays[branch]), before[observable],
            np.ones(np.asarray(arrays[branch]).size, dtype=bool),
            circular=observable in ANGLE_OBSERVABLES,
        )
        values["branch"] = branch
        quantities[observable] = values
    tested = [
        item["maximumAbsoluteDifference"] for item in quantities.values()
        if item["maximumAbsoluteDifference"] is not None
    ]
    maximum = max(tested, default=0.0)
    return {
        "tolerance": tolerance,
        "testedQuantities": len(tested),
        "passed": bool(tested and maximum <= tolerance),
        "maximumObservedAbsoluteDifference": float(maximum),
        "quantities": quantities,
        "note": (
            "This compares the independent NumPy recomputation with the existing "
            "post-process branches; it is diagnostic rather than a correction invariant."
        ),
    }


def _mask_agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    disagreement = reference ^ candidate
    return {
        "referenceSelected": int(np.count_nonzero(reference)),
        "recomputedSelected": int(np.count_nonzero(candidate)),
        "disagreements": int(np.count_nonzero(disagreement)),
        "agreementFraction": float(np.mean(~disagreement)) if reference.size else 1.0,
        "referenceOnly": int(np.count_nonzero(reference & ~candidate)),
        "recomputedOnly": int(np.count_nonzero(~reference & candidate)),
    }


def run_paired_validation(
    arrays: dict[str, np.ndarray],
    parameters: dict[str, object],
    cuts: ExclusivityCuts,
    binning: AnalysisBinning,
    *,
    external_selection_mask: np.ndarray | None = None,
    minimum_electron_p: float = 2.0,
    minimum_q2: float = 1.0,
    minimum_w: float = 2.0,
    invariant_tolerance: float = 1.0e-10,
    reference_tolerance: float = 1.0e-7,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    entries = int(np.asarray(arrays["electronP"]).size)
    if any(np.asarray(arrays[name]).size != entries for name in ROLE_COLUMNS):
        raise ValueError("selected-particle arrays do not all have the same length")
    beam_energy = float(parameters["beamEnergyGeV"])
    corrected_p, support, correction, local_phi = apply_supported_electron_correction(
        arrays, parameters
    )
    before = compute_eppi0_observables(arrays, arrays["electronP"], beam_energy)
    after = compute_eppi0_observables(arrays, corrected_p, beam_energy)
    base_before = _base_analysis_mask(
        arrays, before, minimum_electron_p=minimum_electron_p,
        minimum_q2=minimum_q2, minimum_w=minimum_w,
    )
    base_after = _base_analysis_mask(
        arrays, after, minimum_electron_p=minimum_electron_p,
        minimum_q2=minimum_q2, minimum_w=minimum_w,
    )
    base_thresholds_before = _individual_base_threshold_masks(
        arrays,
        before,
        minimum_electron_p=minimum_electron_p,
        minimum_q2=minimum_q2,
        minimum_w=minimum_w,
    )
    base_thresholds_after = _individual_base_threshold_masks(
        arrays,
        after,
        minimum_electron_p=minimum_electron_p,
        minimum_q2=minimum_q2,
        minimum_w=minimum_w,
    )
    cuts_before = _apply_persisted_exclusivity(cuts, binning, arrays, before)
    cuts_after = _apply_persisted_exclusivity(cuts, binning, arrays, after)
    selected_before = base_before & cuts_before
    selected_after = base_after & cuts_after

    if external_selection_mask is None:
        fixed = selected_before.copy()
        fixed_source = "recomputed before-correction selection"
        mask_agreement = None
    else:
        fixed = np.asarray(external_selection_mask, dtype=bool)
        if fixed.size != entries:
            raise ValueError("external selection mask length does not match input arrays")
        fixed_source = "external strict selection mask"
        mask_agreement = _mask_agreement(fixed, selected_before)

    sectors = np.asarray(arrays["electronSector"], dtype=int)
    migration = {
        "overall": _migration_summary(selected_before, selected_after),
        "baseThresholds": _migration_summary(base_before, base_after),
        "allExclusivityWindows": _migration_summary(cuts_before, cuts_after),
        "supported": _migration_summary(
            selected_before[support], selected_after[support]
        ),
        "unsupported": _migration_summary(
            selected_before[~support], selected_after[~support]
        ),
        "sectors": {
            str(sector): _migration_summary(
                selected_before[sectors == sector], selected_after[sectors == sector]
            )
            for sector in range(1, 7)
        },
        "individualCuts": {},
        "individualBaseThresholds": {
            name: _migration_summary(
                base_thresholds_before[name], base_thresholds_after[name]
            )
            for name in base_thresholds_before
        },
    }
    for variable in cuts.variables:
        pass_before = _apply_persisted_exclusivity(
            cuts, binning, arrays, before, only_variable=variable
        )
        pass_after = _apply_persisted_exclusivity(
            cuts, binning, arrays, after, only_variable=variable
        )
        migration["individualCuts"][variable] = _migration_summary(
            pass_before, pass_after
        )
    invariants = _audit_invariants(before, after, support, invariant_tolerance)
    if not invariants["passed"]:
        raise RuntimeError(
            "paired-validation invariant failed: maximum difference "
            f"{invariants['maximumObservedAbsoluteDifference']:.6g} exceeds "
            f"{invariant_tolerance:.6g}"
        )
    reference_audit = _audit_reference_branches(arrays, before, reference_tolerance)

    fixed_supported = fixed & support
    fixed_unsupported = fixed & ~support
    cohorts = {
        "fixedAll": _cohort_summary(before, after, fixed),
        "fixedSupported": _cohort_summary(before, after, fixed_supported),
        "fixedUnsupported": _cohort_summary(before, after, fixed_unsupported),
        "reselectedCommon": _cohort_summary(
            before, after, selected_before & selected_after
        ),
    }
    sector_cohorts = {
        str(sector): _cohort_summary(
            before, after, fixed_supported & (sectors == sector)
        )
        for sector in range(1, 7)
    }
    support_count = int(np.count_nonzero(support))
    fixed_count = int(np.count_nonzero(fixed))
    correction_by_sector: dict[str, object] = {}
    for sector in range(1, 7):
        rows = (sectors == sector) & support
        values = correction[rows]
        correction_by_sector[str(sector)] = {
            "entries": int(values.size),
            "meanFraction": float(np.mean(values)) if values.size else None,
            "medianFraction": float(np.median(values)) if values.size else None,
            "minimumFraction": float(np.min(values)) if values.size else None,
            "maximumFraction": float(np.max(values)) if values.size else None,
        }
    report: dict[str, object] = {
        "schema": "eppi0-paired-electron-momentum-validation-v1",
        "beamEnergyGeV": beam_energy,
        "parameterDatasetTag": parameters.get("datasetTag", ""),
        "entries": entries,
        "selection": {
            "minimumElectronPGeV": minimum_electron_p,
            "minimumQ2GeV2": minimum_q2,
            "minimumWGeV": minimum_w,
            "fixedCohortSource": fixed_source,
            "fixedEntries": fixed_count,
            "fixedSupportedEntries": int(np.count_nonzero(fixed_supported)),
            "fixedSupportFraction": _fraction(
                int(np.count_nonzero(fixed_supported)), fixed_count
            ),
            "allSupportedEntries": support_count,
            "allSupportFraction": _fraction(support_count, entries),
            "baseBeforeEntries": int(np.count_nonzero(base_before)),
            "baseAfterEntries": int(np.count_nonzero(base_after)),
            "reselectedBeforeEntries": int(np.count_nonzero(selected_before)),
            "reselectedAfterEntries": int(np.count_nonzero(selected_after)),
            "externalMaskAgreement": mask_agreement,
        },
        "migration": migration,
        "correction": {
            "convention": "p_after = p_before * (1 + fractionalCorrection)",
            "supportedOnly": True,
            "sectors": correction_by_sector,
        },
        "cohorts": cohorts,
        "fixedSupportedBySector": sector_cohorts,
        "invariants": invariants,
        "referenceRecomputationAudit": reference_audit,
    }
    diagnostics = {
        "support": support,
        "correctionFraction": correction,
        "deltaElectronP": corrected_p - np.asarray(arrays["electronP"], dtype=float),
        "localPhiDeg": local_phi,
        "baseBefore": base_before,
        "baseAfter": base_after,
        "selectedBefore": selected_before,
        "selectedAfter": selected_after,
        "fixed": fixed,
        **{f"before_{name}": values for name, values in before.items()},
        **{f"after_{name}": values for name, values in after.items()},
    }
    return report, diagnostics


def _write_observable_tsv(report: dict[str, object], path: Path) -> None:
    fields = (
        "cohort", "quantity", "entries", "beforeMean", "afterMean", "deltaMean",
        "beforeStd", "afterStd", "beforeMedian", "afterMedian",
        "beforeRobustWidth", "afterRobustWidth", "expectedCenter",
        "rmsFromExpectedBefore", "rmsFromExpectedAfter",
    )
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for cohort_name, cohort in report["cohorts"].items():
            for name, item in cohort["quantities"].items():
                writer.writerow({
                    "cohort": cohort_name,
                    "quantity": name,
                    "entries": item["before"]["entries"],
                    "beforeMean": item["before"]["mean"],
                    "afterMean": item["after"]["mean"],
                    "deltaMean": item["delta"]["mean"],
                    "beforeStd": item["before"]["std"],
                    "afterStd": item["after"]["std"],
                    "beforeMedian": item["before"]["median"],
                    "afterMedian": item["after"]["median"],
                    "beforeRobustWidth": item["before"]["robustWidth"],
                    "afterRobustWidth": item["after"]["robustWidth"],
                    "expectedCenter": item.get("expectedCenter"),
                    "rmsFromExpectedBefore": item.get("rmsFromExpectedBefore"),
                    "rmsFromExpectedAfter": item.get("rmsFromExpectedAfter"),
                })
        for sector, cohort in report["fixedSupportedBySector"].items():
            for name, item in cohort["quantities"].items():
                writer.writerow({
                    "cohort": f"fixedSupportedSector{sector}",
                    "quantity": name,
                    "entries": item["before"]["entries"],
                    "beforeMean": item["before"]["mean"],
                    "afterMean": item["after"]["mean"],
                    "deltaMean": item["delta"]["mean"],
                    "beforeStd": item["before"]["std"],
                    "afterStd": item["after"]["std"],
                    "beforeMedian": item["before"]["median"],
                    "afterMedian": item["after"]["median"],
                    "beforeRobustWidth": item["before"]["robustWidth"],
                    "afterRobustWidth": item["after"]["robustWidth"],
                    "expectedCenter": item.get("expectedCenter"),
                    "rmsFromExpectedBefore": item.get("rmsFromExpectedBefore"),
                    "rmsFromExpectedAfter": item.get("rmsFromExpectedAfter"),
                })


def _write_migration_tsv(report: dict[str, object], path: Path) -> None:
    rows = [("overall", report["migration"]["overall"])]
    rows.extend(
        (name, report["migration"][name])
        for name in (
            "baseThresholds", "allExclusivityWindows", "supported", "unsupported"
        )
    )
    rows.extend(
        (f"sector{sector}", report["migration"]["sectors"][str(sector)])
        for sector in range(1, 7)
    )
    rows.extend(
        (f"base:{name}", values)
        for name, values in report["migration"]["individualBaseThresholds"].items()
    )
    rows.extend(
        (f"cut:{name}", values)
        for name, values in report["migration"]["individualCuts"].items()
    )
    fields = ("group", *next(iter(rows))[1].keys())
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for group, values in rows:
            writer.writerow({"group": group, **values})


def _plot_values(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values * RAD_TO_DEG if name in ANGLE_OBSERVABLES else values


def _robust_range(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    values = np.concatenate((before[mask], after[mask]))
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    low, high = np.quantile(values, [0.005, 0.995])
    if not np.isfinite(low + high) or high <= low:
        return None
    padding = 0.04 * (high - low)
    return float(low - padding), float(high + padding)


def _make_plots(
    report: dict[str, object],
    diagnostics: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    dataset_tag: str,
    beam_energy: float,
) -> None:
    import matplotlib.pyplot as plt

    mask = diagnostics["fixed"] & diagnostics["support"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for axis, name in zip(axes.flat, PLOT_QUANTITIES):
        before = _plot_values(name, diagnostics[f"before_{name}"])
        after = _plot_values(name, diagnostics[f"after_{name}"])
        value_range = _robust_range(before, after, mask)
        if value_range is not None and np.count_nonzero(mask) > 1:
            axis.hist(before[mask], bins=70, range=value_range, density=True,
                      histtype="step", linewidth=1.2, label="before")
            axis.hist(after[mask], bins=70, range=value_range, density=True,
                      histtype="step", linewidth=1.2, label="after")
        axis.set_xlabel(PLOT_LABELS[name])
        axis.set_ylabel("normalized candidates")
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    save_plot(
        fig, output_dir / "fixed_supported_observables.png",
        "Fixed supported ep-pi0 cohort: before/after electron correction",
        dataset_tag, beam_energy,
    )
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for axis, name in zip(axes.flat, PLOT_QUANTITIES):
        before = _plot_values(name, diagnostics[f"before_{name}"])
        after = _plot_values(name, diagnostics[f"after_{name}"])
        delta = after - before
        selected = delta[mask & np.isfinite(delta)]
        if selected.size > 1:
            low, high = np.quantile(selected, [0.005, 0.995])
            if high > low:
                axis.hist(selected, bins=70, range=(low, high), histtype="step")
        axis.axvline(0.0, color="black", linewidth=0.7)
        axis.set_xlabel("after - before: " + PLOT_LABELS[name])
        axis.set_ylabel("candidates")
        axis.grid(alpha=0.2)
    save_plot(
        fig, output_dir / "fixed_supported_deltas.png",
        "Fixed supported ep-pi0 cohort: paired observable changes",
        dataset_tag, beam_energy,
    )
    plt.close(fig)

    sectors = np.arange(1, 7)
    lost = np.asarray([
        report["migration"]["sectors"][str(sector)]["lostAfterCorrection"]
        for sector in sectors
    ])
    gained = np.asarray([
        report["migration"]["sectors"][str(sector)]["gainedAfterCorrection"]
        for sector in sectors
    ])
    before_count = np.asarray([
        report["migration"]["sectors"][str(sector)]["beforeSelected"]
        for sector in sectors
    ])
    lost_fraction = np.divide(lost, before_count, out=np.zeros(6), where=before_count > 0)
    gained_fraction = np.divide(gained, before_count, out=np.zeros(6), where=before_count > 0)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(sectors - 0.18, 100.0 * lost_fraction, width=0.36, label="lost")
    axis.bar(sectors + 0.18, 100.0 * gained_fraction, width=0.36, label="gained")
    axis.set_xlabel("electron sector")
    axis.set_ylabel("fraction of before-selected cohort [%]")
    axis.set_xticks(sectors)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save_plot(
        fig, output_dir / "selection_migration.png",
        "Strict-exclusivity migrations under electron correction",
        dataset_tag, beam_energy,
    )
    plt.close(fig)

    sector_closure_quantities = (
        "missingEnergy", "missingPt", "m2Miss", "m2EpX", "mEggX", "deltaT"
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, name in zip(axes.flat, sector_closure_quantities):
        before_rms = np.asarray([
            report["fixedSupportedBySector"][str(sector)]["quantities"][name][
                "rmsFromExpectedBefore"
            ]
            for sector in sectors
        ], dtype=float)
        after_rms = np.asarray([
            report["fixedSupportedBySector"][str(sector)]["quantities"][name][
                "rmsFromExpectedAfter"
            ]
            for sector in sectors
        ], dtype=float)
        axis.plot(sectors, before_rms, marker="o", label="before")
        axis.plot(sectors, after_rms, marker="o", label="after")
        axis.set_title(PLOT_LABELS.get(name, name))
        axis.set_xlabel("electron sector")
        axis.set_ylabel("RMS from physical center")
        axis.set_xticks(sectors)
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    save_plot(
        fig, output_dir / "sector_closure_rms.png",
        "Fixed supported ep-pi0 cohort: closure RMS by electron sector",
        dataset_tag, beam_energy,
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    theta = np.asarray(arrays["electronTheta"]) * RAD_TO_DEG
    sectors_array = np.asarray(arrays["electronSector"], dtype=int)
    for sector, axis in enumerate(axes.flat, 1):
        sector_mask = mask & (sectors_array == sector)
        if np.any(sector_mask):
            points = axis.hexbin(
                theta[sector_mask], diagnostics["localPhiDeg"][sector_mask],
                C=100.0 * diagnostics["correctionFraction"][sector_mask],
                reduce_C_function=np.mean, gridsize=35, mincnt=1, cmap="coolwarm",
            )
            fig.colorbar(points, ax=axis, label="mean correction [%]")
        axis.set_title(f"sector {sector}")
        axis.set_xlabel(r"electron $\theta$ [deg]")
        axis.set_ylabel(r"sector-local $\phi$ [deg]")
    save_plot(
        fig, output_dir / "correction_response.png",
        "Applied correction over the fixed supported ep-pi0 cohort",
        dataset_tag, beam_energy,
    )
    plt.close(fig)


def _write_event_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
) -> None:
    payload = {
        name: np.asarray(arrays[name])
        for name in ("runNum", "eventNum") if name in arrays
    }
    payload.update({
        "support": diagnostics["support"],
        "correctionFraction": diagnostics["correctionFraction"],
        "deltaElectronP": diagnostics["deltaElectronP"],
        "fixedCohort": diagnostics["fixed"],
        "selectedBefore": diagnostics["selectedBefore"],
        "selectedAfter": diagnostics["selectedAfter"],
    })
    for name in SUMMARY_QUANTITIES:
        payload[f"before_{name}"] = diagnostics[f"before_{name}"]
        payload[f"after_{name}"] = diagnostics[f"after_{name}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply an elastic electron correction in memory and perform paired "
            "ep-pi0 validation on fixed and reselected cohorts."
        )
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--exclusivity-cuts", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--selection-mask", type=Path)
    parser.add_argument("--selection-mask-key", default="mask")
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--beam-energy", type=float)
    parser.add_argument("--min-electron-p", type=float, default=2.0)
    parser.add_argument("--min-q2", type=float, default=1.0)
    parser.add_argument("--min-w", type=float, default=2.0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--event-output", type=Path)
    parser.add_argument("--invariant-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--reference-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    parameters = _read_json(args.parameters)
    parameter_beam = float(parameters["beamEnergyGeV"])
    if args.beam_energy is not None:
        if not np.isclose(args.beam_energy, parameter_beam, rtol=0.0, atol=1.0e-9):
            raise ValueError(
                f"--beam-energy {args.beam_energy:g} does not match parameter file "
                f"{parameter_beam:g}"
            )
    arrays = load_eppi0_arrays(args.input_file, args.tree, args.max_rows)
    entries = int(np.asarray(arrays["electronP"]).size)
    external = None
    if args.selection_mask is not None:
        external = load_aligned_selection_mask(
            args.selection_mask, entries, args.selection_mask_key,
            allow_prefix=args.max_rows is not None,
        )
    cuts = load_cuts(str(args.exclusivity_cuts))
    binning = from_config(args.analysis_config)
    report, diagnostics = run_paired_validation(
        arrays, parameters, cuts, binning,
        external_selection_mask=external,
        minimum_electron_p=args.min_electron_p,
        minimum_q2=args.min_q2,
        minimum_w=args.min_w,
        invariant_tolerance=args.invariant_tolerance,
        reference_tolerance=args.reference_tolerance,
    )
    report["datasetTag"] = args.dataset_tag
    report["inputFile"] = str(args.input_file)
    report["parameterFile"] = str(args.parameters)
    report["exclusivityCutsFile"] = str(args.exclusivity_cuts)
    report["exclusivityCuts"] = summarize_exclusivity_cuts(args.exclusivity_cuts)
    report["analysisConfig"] = str(args.analysis_config)
    report["selectionMask"] = (
        str(args.selection_mask) if args.selection_mask is not None else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "paired_eppi0_momentum_validation.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2) + "\n")
    _write_observable_tsv(report, args.output_dir / "paired_observable_summary.tsv")
    _write_migration_tsv(report, args.output_dir / "selection_migration.tsv")
    if args.event_output is not None:
        _write_event_npz(args.event_output, arrays, diagnostics)
    if not args.no_plots:
        _make_plots(
            report, diagnostics, arrays, args.output_dir, args.dataset_tag,
            parameter_beam,
        )

    selection = report["selection"]
    migration = report["migration"]["overall"]
    print(f"Wrote paired ep-pi0 momentum validation to {args.output_dir}")
    print(
        f" fixed cohort: {selection['fixedEntries']} events; "
        f"exact support={100.0 * selection['fixedSupportFraction']:.2f}%"
    )
    print(
        " reselected migration: "
        f"lost={migration['lostAfterCorrection']}; "
        f"gained={migration['gainedAfterCorrection']}; "
        f"retained={100.0 * migration['retainedFractionOfBefore']:.3f}%"
    )
    print(
        " invariants: PASS; reference recomputation: "
        + ("PASS" if report["referenceRecomputationAudit"]["passed"] else "CHECK")
    )


if __name__ == "__main__":
    main()
