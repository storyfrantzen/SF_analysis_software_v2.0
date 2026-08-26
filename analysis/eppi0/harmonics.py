from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


DEFAULT_MINIMUM_POINTS = 12
DEFAULT_MAXIMUM_CHI2_NDF = 3.0
DEFAULT_MAXIMUM_COVARIANCE_CONDITION = 1.0e4
DEFAULT_MAXIMUM_RELATIVE_A_UNCERTAINTY = 0.5

QUALITY_SPARSE = np.uint16(1 << 0)
QUALITY_COVARIANCE_INVALID = np.uint16(1 << 1)
QUALITY_COVARIANCE_ILL_CONDITIONED = np.uint16(1 << 2)
QUALITY_A_UNCERTAIN = np.uint16(1 << 3)
QUALITY_CHI2_LARGE = np.uint16(1 << 4)
QUALITY_NEGATIVE_CROSS_SECTION = np.uint16(1 << 5)
QUALITY_FIT_FAILED = np.uint16(1 << 6)

QUALITY_REASON_NAMES = (
    "sparse_phi_coverage",
    "invalid_covariance",
    "ill_conditioned_covariance",
    "uncertain_A",
    "large_chi2_ndf",
    "negative_fitted_cross_section",
    "fit_failed",
)
QUALITY_REASON_BITS = np.asarray(
    [
        QUALITY_SPARSE,
        QUALITY_COVARIANCE_INVALID,
        QUALITY_COVARIANCE_ILL_CONDITIONED,
        QUALITY_A_UNCERTAIN,
        QUALITY_CHI2_LARGE,
        QUALITY_NEGATIVE_CROSS_SECTION,
        QUALITY_FIT_FAILED,
    ],
    dtype=np.uint16,
)


@dataclass(frozen=True)
class HarmonicFit:
    parameters: Array
    covariance: Array
    chi2_ndf: float
    points: int
    covariance_condition: float
    minimum_cross_section: float


def minimum_harmonic_cross_section(parameters: Array) -> float:
    """Return the exact minimum of `A + B cos(phi) + C cos(2 phi)` over phi."""
    a, b, c = np.asarray(parameters, dtype=float)
    if not np.all(np.isfinite((a, b, c))):
        return np.nan
    # With x = cos(phi), the fit is 2 C x^2 + B x + (A - C), x in [-1, 1].
    candidates = [a + b + c, a - b + c]
    if c > 0.0:
        vertex = -b / (4.0 * c)
        if -1.0 < vertex < 1.0:
            candidates.append(2.0 * c * vertex * vertex + b * vertex + a - c)
    return float(min(candidates))


def _valid_points(
    phi_deg: Array,
    values: Array,
    uncertainties: Array,
    validity_mask: Array | None,
) -> tuple[Array, Array, Array, Array]:
    phi_deg, values, uncertainties = np.broadcast_arrays(
        np.asarray(phi_deg, float), np.asarray(values, float), np.asarray(uncertainties, float)
    )
    valid = (
        np.isfinite(phi_deg)
        & np.isfinite(values)
        & np.isfinite(uncertainties)
        & (uncertainties > 0.0)
    )
    if validity_mask is not None:
        valid &= np.broadcast_to(np.asarray(validity_mask, dtype=bool), valid.shape)
    return phi_deg, values, uncertainties, valid


def fit_phi(
    phi_deg: Array,
    values: Array,
    uncertainties: Array,
    validity_mask: Array | None = None,
) -> HarmonicFit | None:
    """Fit `A + B cos(phi) + C cos(2 phi)` by weighted least squares."""
    phi_deg, values, uncertainties, valid = _valid_points(
        phi_deg, values, uncertainties, validity_mask
    )
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
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None
    residual = (y - design @ parameters) / sigma
    ndf = y.size - 3
    return HarmonicFit(
        parameters,
        covariance,
        float(residual @ residual / ndf),
        int(y.size),
        float(np.linalg.cond(covariance)),
        minimum_harmonic_cross_section(parameters),
    )


def _quality_status(
    fit: HarmonicFit | None,
    points: int,
    *,
    minimum_points: int,
    maximum_chi2_ndf: float,
    maximum_covariance_condition: float,
    maximum_relative_a_uncertainty: float,
    require_nonnegative: bool,
) -> np.uint16:
    status = np.uint16(0)
    if points < minimum_points:
        status |= QUALITY_SPARSE
    if fit is None:
        return status | QUALITY_FIT_FAILED

    covariance = np.asarray(fit.covariance, dtype=float)
    covariance_valid = np.all(np.isfinite(covariance))
    if covariance_valid:
        try:
            covariance_valid = bool(np.all(np.linalg.eigvalsh(covariance) > 0.0))
        except np.linalg.LinAlgError:
            covariance_valid = False
    if not covariance_valid:
        status |= QUALITY_COVARIANCE_INVALID
    if (
        not np.isfinite(fit.covariance_condition)
        or fit.covariance_condition > maximum_covariance_condition
    ):
        status |= QUALITY_COVARIANCE_ILL_CONDITIONED

    a = float(fit.parameters[0])
    a_variance = float(covariance[0, 0])
    relative_a_uncertainty = (
        np.sqrt(a_variance) / abs(a)
        if np.isfinite(a) and a != 0.0 and np.isfinite(a_variance) and a_variance >= 0.0
        else np.inf
    )
    if relative_a_uncertainty > maximum_relative_a_uncertainty:
        status |= QUALITY_A_UNCERTAIN
    if not np.isfinite(fit.chi2_ndf) or fit.chi2_ndf > maximum_chi2_ndf:
        status |= QUALITY_CHI2_LARGE
    if require_nonnegative:
        scale = max(1.0, float(np.max(np.abs(fit.parameters))))
        if (
            not np.isfinite(fit.minimum_cross_section)
            or fit.minimum_cross_section < -1.0e-10 * scale
        ):
            status |= QUALITY_NEGATIVE_CROSS_SECTION
    return status


def fit_grid(
    values: Array,
    uncertainties: Array,
    phi_edges: Array,
    *,
    validity_mask: Array | None = None,
    minimum_points: int = DEFAULT_MINIMUM_POINTS,
    maximum_chi2_ndf: float = DEFAULT_MAXIMUM_CHI2_NDF,
    maximum_covariance_condition: float = DEFAULT_MAXIMUM_COVARIANCE_CONDITION,
    maximum_relative_a_uncertainty: float = DEFAULT_MAXIMUM_RELATIVE_A_UNCERTAINTY,
    require_nonnegative: bool = True,
) -> dict[str, Array]:
    """Fit every `(Q2, xB, t)` cell of a four-dimensional cross section."""
    values = np.asarray(values, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    if values.shape != uncertainties.shape or values.ndim != 4:
        raise ValueError("cross section and uncertainty must be equal 4D arrays")
    phi_edges = np.asarray(phi_edges, dtype=float)
    if phi_edges.size != values.shape[-1] + 1:
        raise ValueError("phi edges do not match cross-section bins")
    if validity_mask is None:
        validity_mask = np.ones(values.shape, dtype=bool)
    else:
        validity_mask = np.asarray(validity_mask, dtype=bool)
        if validity_mask.shape != values.shape:
            raise ValueError("validity mask does not match cross-section bins")
    if minimum_points < 4:
        raise ValueError("minimum_points must be at least 4")
    for name, threshold in (
        ("maximum_chi2_ndf", maximum_chi2_ndf),
        ("maximum_covariance_condition", maximum_covariance_condition),
        ("maximum_relative_a_uncertainty", maximum_relative_a_uncertainty),
    ):
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    grid_shape = values.shape[:-1]
    parameters = np.full(grid_shape + (3,), np.nan)
    covariance = np.full(grid_shape + (3, 3), np.nan)
    chi2_ndf = np.full(grid_shape, np.nan)
    points = np.zeros(grid_shape, dtype=np.int32)
    fit_success = np.zeros(grid_shape, dtype=bool)
    quality_mask = np.zeros(grid_shape, dtype=bool)
    quality_status = np.zeros(grid_shape, dtype=np.uint16)
    covariance_condition = np.full(grid_shape, np.nan)
    minimum_cross_section = np.full(grid_shape, np.nan)
    parameter_uncertainties = np.full(grid_shape + (3,), np.nan)
    relative_a_uncertainty = np.full(grid_shape, np.nan)
    for index in np.ndindex(grid_shape):
        _, _, _, valid = _valid_points(
            centers, values[index], uncertainties[index], validity_mask[index]
        )
        points[index] = int(np.count_nonzero(valid))
        fit = fit_phi(
            centers,
            values[index],
            uncertainties[index],
            validity_mask=validity_mask[index],
        )
        status = _quality_status(
            fit,
            int(points[index]),
            minimum_points=minimum_points,
            maximum_chi2_ndf=maximum_chi2_ndf,
            maximum_covariance_condition=maximum_covariance_condition,
            maximum_relative_a_uncertainty=maximum_relative_a_uncertainty,
            require_nonnegative=require_nonnegative,
        )
        quality_status[index] = status
        if fit is None:
            continue
        fit_success[index] = True
        parameters[index] = fit.parameters
        covariance[index] = fit.covariance
        chi2_ndf[index] = fit.chi2_ndf
        covariance_condition[index] = fit.covariance_condition
        minimum_cross_section[index] = fit.minimum_cross_section
        diagonal = np.diag(fit.covariance)
        parameter_uncertainties[index] = np.sqrt(
            np.where(diagonal >= 0.0, diagonal, np.nan)
        )
        a = float(fit.parameters[0])
        relative_a_uncertainty[index] = (
            parameter_uncertainties[index][0] / abs(a)
            if np.isfinite(a) and a != 0.0
            else np.inf
        )
        quality_mask[index] = status == 0
    return {
        "parameters": parameters,
        "covariance": covariance,
        "parameter_uncertainties": parameter_uncertainties,
        "chi2_ndf": chi2_ndf,
        "points": points,
        "fit_success": fit_success,
        "quality_mask": quality_mask,
        "quality_status": quality_status,
        "covariance_condition": covariance_condition,
        "relative_A_uncertainty": relative_a_uncertainty,
        "minimum_fitted_cross_section": minimum_cross_section,
    }
