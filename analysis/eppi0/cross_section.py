from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import AnalysisBinning
from .exclusive_kinematics import t_limits_pi0
from .phase_space import ELECTRON_MASS_GEV, scattered_electron_momentum


ELECTRON_CHARGE_C = 1.602176634e-19
AVOGADRO = 6.02214076e23
ALPHA_EM = 1.0 / 137.035999084
DEFAULT_VOLUME_INTEGRATION_POINTS = 400


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
    electron_p_minimum: float = 0.0,
    integration_points: int = DEFAULT_VOLUME_INTEGRATION_POINTS,
) -> np.ndarray:
    """Return selected exclusive `(Q2, xB, t, phi)` volumes in GeV^4 rad."""
    if integration_points <= 0:
        raise ValueError("integration_points must be positive")
    nq2, nxb, nt, nphi = binning.shape
    qxt_volume = np.zeros((nq2, nxb, nt), dtype=float)
    for iq2 in range(nq2):
        q_low, q_high = binning.q2_edges[iq2 : iq2 + 2]
        dq = (q_high - q_low) / integration_points
        q2 = q_low + (np.arange(integration_points) + 0.5) * dq
        for ixb in range(nxb):
            x_low, x_high = binning.xb_edges[ixb : ixb + 2]
            dx = (x_high - x_low) / integration_points
            x = x_low + (np.arange(integration_points) + 0.5) * dx
            q2_mesh, xb_mesh = np.meshgrid(q2, x, indexing="ij")
            w2 = proton_mass**2 + q2_mesh * (1.0 / xb_mesh - 1.0)
            electron_p = scattered_electron_momentum(
                q2_mesh,
                xb_mesh,
                beam_energy,
                proton_mass=proton_mass,
            )
            electron_energy = np.sqrt(electron_p**2 + ELECTRON_MASS_GEV**2)
            sin2_half = np.divide(
                q2_mesh,
                4.0 * beam_energy * electron_energy,
                out=np.full_like(q2_mesh, np.nan),
                where=electron_energy > 0.0,
            )
            selected = (
                (q2_mesh >= q2_minimum)
                & (w2 >= w_minimum**2)
                & (electron_p >= electron_p_minimum)
                & (electron_energy > 0.0)
                & (sin2_half > 0.0)
                & (sin2_half < 1.0)
            )
            t_low, t_high = t_limits_pi0(xb_mesh, q2_mesh)
            selected &= np.isfinite(t_low) & np.isfinite(t_high)
            for it, (minus_t_low, minus_t_high) in enumerate(
                zip(binning.t_edges[:-1], binning.t_edges[1:])
            ):
                signed_bin_low = -minus_t_high
                signed_bin_high = -minus_t_low
                overlap = np.maximum(
                    0.0,
                    np.minimum(t_high, signed_bin_high)
                    - np.maximum(t_low, signed_bin_low),
                )
                overlap = np.where(selected, overlap, 0.0)
                qxt_volume[iq2, ixb, it] = overlap.sum() * dq * dx
    dphi = np.deg2rad(np.diff(binning.phi_edges))
    return qxt_volume[:, :, :, None] * dphi[None, None, None, :]


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
