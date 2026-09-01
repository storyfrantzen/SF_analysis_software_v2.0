from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import AnalysisBinning
from .cross_section import virtual_photon_epsilon


Array = np.ndarray
PROTON_MASS_GEV = 0.9382720813
STRUCTURE_FUNCTION_NAMES = np.asarray(["sigma_U", "sigma_LT", "sigma_TT"])


@dataclass(frozen=True)
class StructureFunctionResult:
    values: Array
    covariance: Array
    uncertainties: Array
    epsilon: Array
    valid: Array


def epsilon_from_xb_q2(
    q2: Array,
    xb: Array,
    beam_energy: float,
    proton_mass: float = PROTON_MASS_GEV,
) -> Array:
    """Return virtual-photon polarization at fixed ``(Q2, xB, E)``."""
    q2, xb = np.broadcast_arrays(np.asarray(q2, float), np.asarray(xb, float))
    valid = (
        np.isfinite(q2)
        & np.isfinite(xb)
        & (q2 > 0.0)
        & (xb > 0.0)
        & np.isfinite(beam_energy)
        & (beam_energy > 0.0)
    )
    y = np.divide(
        q2,
        2.0 * proton_mass * xb * beam_energy,
        out=np.full_like(q2, np.nan),
        where=valid,
    )
    epsilon = virtual_photon_epsilon(q2, y, beam_energy)
    return np.where(valid & (epsilon > 0.0) & (epsilon < 1.0), epsilon, np.nan)


def harmonic_to_structure_functions(
    parameters: Array,
    covariance: Array,
    epsilon: Array,
) -> StructureFunctionResult:
    """Transform ``(A, B, C)`` and its full covariance to ``(U, LT, TT)``.

    The input convention is

    ``d2sigma/(dt dphi) = A + B cos(phi) + C cos(2 phi)``.
    """
    parameters = np.asarray(parameters, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    if parameters.ndim < 1 or parameters.shape[-1] != 3:
        raise ValueError("harmonic parameters must end in an axis of length 3")
    if covariance.shape != parameters.shape[:-1] + (3, 3):
        raise ValueError("harmonic covariance shape does not match parameters")
    try:
        epsilon = np.broadcast_to(epsilon, parameters.shape[:-1])
    except ValueError as exc:
        raise ValueError("epsilon shape does not match harmonic grid") from exc

    epsilon_valid = np.isfinite(epsilon) & (epsilon > 0.0) & (epsilon < 1.0)
    scales = np.full(parameters.shape, np.nan, dtype=float)
    scales[..., 0] = 2.0 * np.pi
    scales[..., 1] = np.divide(
        2.0 * np.pi,
        np.sqrt(2.0 * epsilon * (1.0 + epsilon)),
        out=np.full_like(epsilon, np.nan),
        where=epsilon_valid,
    )
    scales[..., 2] = np.divide(
        2.0 * np.pi,
        epsilon,
        out=np.full_like(epsilon, np.nan),
        where=epsilon_valid,
    )

    values = parameters * scales
    transformed_covariance = (
        covariance * scales[..., :, None] * scales[..., None, :]
    )
    diagonal = np.diagonal(transformed_covariance, axis1=-2, axis2=-1)
    uncertainties = np.sqrt(np.where(diagonal >= 0.0, diagonal, np.nan))
    valid = (
        epsilon_valid
        & np.all(np.isfinite(parameters), axis=-1)
        & np.all(np.isfinite(covariance), axis=(-2, -1))
        & np.all(np.isfinite(values), axis=-1)
        & np.all(np.isfinite(transformed_covariance), axis=(-2, -1))
        & np.all(np.isfinite(uncertainties), axis=-1)
    )
    values = np.where(valid[..., None], values, np.nan)
    transformed_covariance = np.where(
        valid[..., None, None], transformed_covariance, np.nan
    )
    uncertainties = np.where(valid[..., None], uncertainties, np.nan)
    return StructureFunctionResult(
        values=values,
        covariance=transformed_covariance,
        uncertainties=uncertainties,
        epsilon=epsilon,
        valid=valid,
    )


def harmonic_reference_coordinates(
    binning: AnalysisBinning,
    cross_section,
) -> tuple[Array, Array]:
    """Reduce stored 4D flux coordinates to one ``(Q2, xB)`` per harmonic fit.

    Coordinates are inverse-variance averaged over the valid phi bins. This is
    exact when bin-centering supplies a common reference coordinate and remains
    auditable for legacy artifacts whose event means vary with phi.
    """

    def four_dimensional(name: str, fallback: Array) -> Array:
        if name not in cross_section.files:
            return np.asarray(fallback, dtype=float)
        values = np.asarray(cross_section[name], dtype=float)
        if values.shape == binning.shape:
            return values
        if values.size == binning.size:
            return binning.unflatten(values.reshape(-1))
        raise ValueError(
            f"cross-section {name} has shape {values.shape}; expected "
            f"{binning.shape} or {binning.size} flat values"
        )

    q2_centers = 0.5 * (binning.q2_edges[:-1] + binning.q2_edges[1:])
    xb_centers = 0.5 * (binning.xb_edges[:-1] + binning.xb_edges[1:])
    q2_fallback = np.broadcast_to(
        q2_centers[:, None, None, None], binning.shape
    )
    xb_fallback = np.broadcast_to(
        xb_centers[None, :, None, None], binning.shape
    )
    q2 = four_dimensional("flux_q2_coordinate", q2_fallback)
    xb = four_dimensional("flux_xb_coordinate", xb_fallback)

    if "final_validity_mask" in cross_section.files:
        valid = np.asarray(cross_section["final_validity_mask"], dtype=bool)
        if valid.shape != binning.shape:
            raise ValueError("cross-section final_validity_mask has incompatible shape")
    else:
        valid = np.ones(binning.shape, dtype=bool)
    if "uncertainty" in cross_section.files:
        uncertainty = np.asarray(cross_section["uncertainty"], dtype=float)
        if uncertainty.shape != binning.shape:
            raise ValueError("cross-section uncertainty has incompatible shape")
        weights = np.divide(
            1.0,
            uncertainty**2,
            out=np.zeros_like(uncertainty),
            where=valid & np.isfinite(uncertainty) & (uncertainty > 0.0),
        )
    else:
        weights = valid.astype(float)

    def weighted_phi_mean(values: Array, fallback3: Array) -> Array:
        local_valid = valid & np.isfinite(values) & (weights > 0.0)
        numerator = np.sum(np.where(local_valid, values * weights, 0.0), axis=-1)
        denominator = np.sum(np.where(local_valid, weights, 0.0), axis=-1)
        return np.divide(
            numerator,
            denominator,
            out=np.asarray(fallback3, dtype=float).copy(),
            where=denominator > 0.0,
        )

    q2_fallback3 = np.broadcast_to(q2_centers[:, None, None], binning.shape[:3])
    xb_fallback3 = np.broadcast_to(xb_centers[None, :, None], binning.shape[:3])
    return (
        weighted_phi_mean(q2, q2_fallback3),
        weighted_phi_mean(xb, xb_fallback3),
    )
