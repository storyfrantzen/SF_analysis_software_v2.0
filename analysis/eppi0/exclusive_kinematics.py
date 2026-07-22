from __future__ import annotations

import numpy as np


Array = np.ndarray
PROTON_MASS_GEV = 0.9382720813
PI0_MASS_GEV = 0.1349768


def kallen(a: Array, b: float, c: float) -> Array:
    return a * a + b * b + c * c - 2.0 * a * b - 2.0 * a * c - 2.0 * b * c


def t_limits_pi0(xb: Array, q2: Array) -> tuple[Array, Array]:
    """Return signed physical t limits for gamma* p -> p pi0."""
    xb, q2 = np.broadcast_arrays(np.asarray(xb, float), np.asarray(q2, float))
    t_low = np.full(xb.shape, np.nan, dtype=float)
    t_high = np.full(xb.shape, np.nan, dtype=float)

    valid = (xb > 0.0) & (xb < 1.0) & (q2 > 0.0)
    w2 = np.full(xb.shape, np.nan, dtype=float)
    w2[valid] = PROTON_MASS_GEV**2 + q2[valid] * (1.0 / xb[valid] - 1.0)
    valid &= w2 > (PROTON_MASS_GEV + PI0_MASS_GEV) ** 2
    if not np.any(valid):
        return t_low, t_high

    w = np.sqrt(w2[valid])
    q0_cm = (w2[valid] - PROTON_MASS_GEV**2 - q2[valid]) / (2.0 * w)
    q_cm2 = q0_cm * q0_cm + q2[valid]
    epi_cm = (w2[valid] + PI0_MASS_GEV**2 - PROTON_MASS_GEV**2) / (2.0 * w)
    ppi_cm2 = kallen(w2[valid], PI0_MASS_GEV**2, PROTON_MASS_GEV**2) / (4.0 * w2[valid])
    subvalid = (q_cm2 > 0.0) & (ppi_cm2 > 0.0)
    if not np.any(subvalid):
        return t_low, t_high

    valid_indices = np.flatnonzero(valid)
    out_indices = valid_indices[subvalid]
    q_cm = np.sqrt(q_cm2[subvalid])
    ppi_cm = np.sqrt(ppi_cm2[subvalid])
    t_forward = (
        PI0_MASS_GEV**2
        - q2.ravel()[out_indices]
        - 2.0 * q0_cm[subvalid] * epi_cm[subvalid]
        + 2.0 * q_cm * ppi_cm
    )
    t_backward = (
        PI0_MASS_GEV**2
        - q2.ravel()[out_indices]
        - 2.0 * q0_cm[subvalid] * epi_cm[subvalid]
        - 2.0 * q_cm * ppi_cm
    )
    flat_low = t_low.ravel()
    flat_high = t_high.ravel()
    flat_low[out_indices] = np.minimum(t_backward, t_forward)
    flat_high[out_indices] = np.maximum(t_backward, t_forward)
    return t_low, t_high


def physical_mask(xb: Array, q2: Array, signed_t: Array, beam_energy: float) -> Array:
    """Return the physical ep -> e p pi0 mask for signed t values."""
    xb, q2, signed_t = np.broadcast_arrays(
        np.asarray(xb, float),
        np.asarray(q2, float),
        np.asarray(signed_t, float),
    )
    valid = (xb > 0.0) & (xb < 1.0) & (q2 > 0.0)
    w2 = PROTON_MASS_GEV**2 + q2 * (1.0 / xb - 1.0)
    valid &= w2 > (PROTON_MASS_GEV + PI0_MASS_GEV) ** 2
    nu = np.divide(
        q2,
        2.0 * PROTON_MASS_GEV * xb,
        out=np.full_like(q2, np.nan),
        where=xb > 0.0,
    )
    eprime = float(beam_energy) - nu
    valid &= eprime > 0.0
    sin2_half = np.divide(
        q2,
        4.0 * float(beam_energy) * eprime,
        out=np.full_like(q2, np.nan),
        where=eprime > 0.0,
    )
    valid &= (sin2_half > 0.0) & (sin2_half < 1.0)
    t_low, t_high = t_limits_pi0(xb, q2)
    return valid & np.isfinite(t_low) & np.isfinite(t_high) & (signed_t >= t_low) & (signed_t <= t_high)
