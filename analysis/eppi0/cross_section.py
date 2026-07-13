from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import AnalysisBinning


ELECTRON_CHARGE_C = 1.602176634e-19
AVOGADRO = 6.02214076e23
ALPHA_EM = 1.0 / 137.035999084


@dataclass(frozen=True)
class Target:
    length_cm: float = 5.0
    density_g_cm3: float = 0.071
    molar_mass_g: float = 1.008


def integrated_luminosity_fb(charge_coulombs: float, target: Target = Target()) -> float:
    nuclei_per_area = AVOGADRO * target.length_cm * target.density_g_cm3 / target.molar_mass_g
    beam_electrons = charge_coulombs / ELECTRON_CHARGE_C
    return nuclei_per_area * beam_electrons * 1.0e-39


def virtual_photon_epsilon(q2: np.ndarray, y: np.ndarray, beam_energy: float) -> np.ndarray:
    q2, y = np.broadcast_arrays(np.asarray(q2, float), np.asarray(y, float))
    correction = q2 / (4.0 * beam_energy**2)
    numerator = 1.0 - y - correction
    denominator = 1.0 - y + 0.5 * y**2 + correction
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def virtual_photon_flux(
    q2: np.ndarray, xb: np.ndarray, beam_energy: float, proton_mass: float = 0.9382720813
) -> np.ndarray:
    q2, xb = np.broadcast_arrays(np.asarray(q2, float), np.asarray(xb, float))
    valid = (q2 > 0) & (xb > 0)
    y = np.divide(q2, 2.0 * proton_mass * xb * beam_energy, out=np.zeros_like(q2), where=valid)
    epsilon = virtual_photon_epsilon(q2, y, beam_energy)
    denominator = (proton_mass * beam_energy) ** 2 * xb**3 * (1.0 - epsilon)
    numerator = ALPHA_EM / (8.0 * np.pi) * q2 * (1.0 - xb)
    flux = np.divide(numerator, denominator, out=np.zeros_like(q2), where=valid & (denominator > 0))
    return np.where(np.isfinite(flux) & (flux > 0), flux, 0.0)


def physical_bin_volumes(
    binning: AnalysisBinning,
    beam_energy: float,
    proton_mass: float = 0.9382720813,
    q2_minimum: float = 1.0,
    w_minimum: float = 2.0,
    y_maximum: float = 0.8,
    integration_points: int = 100,
) -> np.ndarray:
    """Return `(Q2, xB, t, phi)` volumes in GeV^4 rad."""
    nq2, nxb, nt, nphi = binning.shape
    qx_area = np.zeros((nq2, nxb), dtype=float)
    for iq2 in range(nq2):
        q_low, q_high = binning.q2_edges[iq2 : iq2 + 2]
        for ixb in range(nxb):
            x_low, x_high = binning.xb_edges[ixb : ixb + 2]
            dx = (x_high - x_low) / integration_points
            x = x_low + (np.arange(integration_points) + 0.5) * dx
            physical_low = np.maximum(
                q2_minimum, (w_minimum**2 - proton_mass**2) / (1.0 / x - 1.0)
            )
            physical_high = y_maximum * 2.0 * proton_mass * beam_energy * x
            overlap = np.maximum(0.0, np.minimum(q_high, physical_high) - np.maximum(q_low, physical_low))
            qx_area[iq2, ixb] = overlap.sum() * dx
    dt = np.diff(binning.t_edges)
    dphi = np.deg2rad(np.diff(binning.phi_edges))
    return qx_area[:, :, None, None] * dt[None, None, :, None] * dphi[None, None, None, :]


def reduced_cross_section(
    yields: np.ndarray,
    uncertainties: np.ndarray,
    q2_means: np.ndarray,
    xb_means: np.ndarray,
    volumes: np.ndarray,
    luminosity_fb: float,
    beam_energy: float,
    branching_ratio: float = 0.988,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.broadcast_arrays(yields, uncertainties, q2_means, xb_means, volumes)
    yields, uncertainties, q2_means, xb_means, volumes = [np.asarray(a, float) for a in arrays]
    flux = virtual_photon_flux(q2_means, xb_means, beam_energy)
    denominator = luminosity_fb * branching_ratio * volumes * flux
    mask = (denominator > 0) & np.isfinite(yields) & (yields >= 0)
    if valid is not None:
        mask &= np.broadcast_to(np.asarray(valid, bool), mask.shape)
    # yield / fb^-1 is fb; 1 fb = 1e-6 nb.
    scale = np.divide(1.0e-6, denominator, out=np.zeros_like(denominator), where=mask)
    return yields * scale, uncertainties * scale
