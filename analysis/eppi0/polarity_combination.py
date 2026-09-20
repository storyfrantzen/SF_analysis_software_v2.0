"""Covariance-aware combination of matching torus-polarity cross sections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PolarityCombination:
    values: Array
    uncertainties: Array
    valid: Array
    contributor_count: Array
    left_weight: Array
    right_weight: Array
    overlap_pull: Array
    overlap_chi2: Array
    shared_relative_uncertainty: Array


def combine_polarity_measurements(
    left_values: Array,
    left_uncertainties: Array,
    left_valid: Array,
    right_values: Array,
    right_uncertainties: Array,
    right_valid: Array,
    *,
    shared_relative_uncertainty: Array | float = 0.0,
) -> PolarityCombination:
    """Combine matching measurements with a shared multiplicative uncertainty.

    The two-by-two covariance in every common bin is

    ``diag(total_sigma**2)`` with off-diagonal
    ``r_shared**2 * left_value * right_value``.

    This is the BLUE solution for the supplied covariance. Bins available from
    only one polarity are copied without reducing their uncertainty.
    """

    left_values = np.asarray(left_values, dtype=float)
    right_values = np.asarray(right_values, dtype=float)
    left_uncertainties = np.asarray(left_uncertainties, dtype=float)
    right_uncertainties = np.asarray(right_uncertainties, dtype=float)
    left_valid = np.asarray(left_valid, dtype=bool)
    right_valid = np.asarray(right_valid, dtype=bool)
    arrays = (
        right_values,
        left_uncertainties,
        right_uncertainties,
        left_valid,
        right_valid,
    )
    if any(values.shape != left_values.shape for values in arrays):
        raise ValueError("polarity cross-section arrays must have identical shapes")
    shared = np.broadcast_to(
        np.asarray(shared_relative_uncertainty, dtype=float), left_values.shape
    ).copy()
    if np.any(~np.isfinite(shared)) or np.any(shared < 0.0):
        raise ValueError("shared relative uncertainty must be finite and nonnegative")

    left_valid = (
        left_valid
        & np.isfinite(left_values)
        & np.isfinite(left_uncertainties)
        & (left_uncertainties > 0.0)
    )
    right_valid = (
        right_valid
        & np.isfinite(right_values)
        & np.isfinite(right_uncertainties)
        & (right_uncertainties > 0.0)
    )
    valid = left_valid | right_valid
    both = left_valid & right_valid
    only_left = left_valid & ~right_valid
    only_right = right_valid & ~left_valid

    values = np.full(left_values.shape, np.nan)
    uncertainties = np.full(left_values.shape, np.nan)
    left_weight = np.zeros(left_values.shape, dtype=float)
    right_weight = np.zeros(left_values.shape, dtype=float)
    overlap_pull = np.full(left_values.shape, np.nan)
    overlap_chi2 = np.full(left_values.shape, np.nan)
    contributor_count = left_valid.astype(np.uint8) + right_valid.astype(np.uint8)

    values[only_left] = left_values[only_left]
    uncertainties[only_left] = left_uncertainties[only_left]
    left_weight[only_left] = 1.0
    values[only_right] = right_values[only_right]
    uncertainties[only_right] = right_uncertainties[only_right]
    right_weight[only_right] = 1.0

    for index in zip(*np.nonzero(both)):
        first = float(left_values[index])
        second = float(right_values[index])
        sigma_first = float(left_uncertainties[index])
        sigma_second = float(right_uncertainties[index])
        shared_fraction = float(shared[index])
        covariance = shared_fraction**2 * first * second
        matrix = np.asarray(
            [
                [sigma_first**2, covariance],
                [covariance, sigma_second**2],
            ],
            dtype=float,
        )
        eigenvalues = np.linalg.eigvalsh(matrix)
        if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0:
            raise ValueError(
                "non-positive polarity covariance in common bin "
                f"{index}: eigenvalues={eigenvalues.tolist()}"
            )
        inverse = np.linalg.inv(matrix)
        ones = np.ones(2, dtype=float)
        normalization = float(ones @ inverse @ ones)
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise ValueError(f"invalid BLUE normalization in common bin {index}")
        weights = inverse @ ones / normalization
        pair = np.asarray([first, second], dtype=float)
        values[index] = float(weights @ pair)
        uncertainties[index] = float(np.sqrt(1.0 / normalization))
        left_weight[index] = float(weights[0])
        right_weight[index] = float(weights[1])

        difference_variance = float(
            matrix[0, 0] + matrix[1, 1] - 2.0 * matrix[0, 1]
        )
        if difference_variance <= 0.0:
            raise ValueError(f"non-positive difference variance in common bin {index}")
        pull = (second - first) / np.sqrt(difference_variance)
        overlap_pull[index] = float(pull)
        overlap_chi2[index] = float(pull**2)

    return PolarityCombination(
        values=values,
        uncertainties=uncertainties,
        valid=valid,
        contributor_count=contributor_count,
        left_weight=left_weight,
        right_weight=right_weight,
        overlap_pull=overlap_pull,
        overlap_chi2=overlap_chi2,
        shared_relative_uncertainty=shared,
    )
