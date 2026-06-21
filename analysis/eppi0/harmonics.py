from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class HarmonicFit:
    parameters: Array
    covariance: Array
    chi2_ndf: float
    points: int


def fit_phi(phi_deg: Array, values: Array, uncertainties: Array) -> HarmonicFit | None:
    """Fit `A + B cos(phi) + C cos(2 phi)` by weighted least squares."""
    phi_deg, values, uncertainties = np.broadcast_arrays(
        np.asarray(phi_deg, float), np.asarray(values, float), np.asarray(uncertainties, float)
    )
    valid = np.isfinite(phi_deg) & np.isfinite(values) & np.isfinite(uncertainties) & (uncertainties > 0)
    if valid.sum() < 4:
        return None
    phi = np.deg2rad(phi_deg[valid])
    y, sigma = values[valid], uncertainties[valid]
    design = np.column_stack((np.ones(phi.size), np.cos(phi), np.cos(2.0 * phi)))
    weighted_design = design / sigma[:, None]
    weighted_y = y / sigma
    parameters, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
    if rank < 3:
        return None
    normal = weighted_design.T @ weighted_design
    covariance = np.linalg.inv(normal)
    residual = (y - design @ parameters) / sigma
    ndf = y.size - 3
    return HarmonicFit(parameters, covariance, float(residual @ residual / ndf), int(y.size))


def fit_grid(values: Array, uncertainties: Array, phi_edges: Array) -> dict[str, Array]:
    """Fit every `(Q2, xB, t)` cell of a four-dimensional cross section."""
    values = np.asarray(values, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    if values.shape != uncertainties.shape or values.ndim != 4:
        raise ValueError("cross section and uncertainty must be equal 4D arrays")
    phi_edges = np.asarray(phi_edges, dtype=float)
    if phi_edges.size != values.shape[-1] + 1:
        raise ValueError("phi edges do not match cross-section bins")
    centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    grid_shape = values.shape[:-1]
    parameters = np.full(grid_shape + (3,), np.nan)
    covariance = np.full(grid_shape + (3, 3), np.nan)
    chi2_ndf = np.full(grid_shape, np.nan)
    points = np.zeros(grid_shape, dtype=np.int32)
    for index in np.ndindex(grid_shape):
        fit = fit_phi(centers, values[index], uncertainties[index])
        if fit is None:
            continue
        parameters[index] = fit.parameters
        covariance[index] = fit.covariance
        chi2_ndf[index] = fit.chi2_ndf
        points[index] = fit.points
    return {
        "parameters": parameters,
        "covariance": covariance,
        "chi2_ndf": chi2_ndf,
        "points": points,
    }
