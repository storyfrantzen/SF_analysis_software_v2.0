from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags, hstack


Array = np.ndarray


@dataclass(frozen=True)
class ResponseResult:
    matrix: csr_matrix
    core: csr_matrix
    truth_total: Array
    reconstructed_total: Array
    efficiency: Array
    feed_in_fraction: float
    feed_in_shape: Array
    response_variance_sum: Array


def build_response(
    truth_flat: Array,
    reconstructed_flat: Array,
    reconstructed_selected: Array,
    number_of_bins: int,
    weights: Array | None = None,
) -> ResponseResult:
    """Build the legacy response convention without per-event Python loops.

    The core matrix has reconstructed bins as rows and truth bins as columns.
    Its column sums are reconstruction efficiencies.  An additional normalized
    column records events reconstructed inside the analysis range but generated
    outside it.
    """
    truth_flat = np.asarray(truth_flat, dtype=np.int64)
    reconstructed_flat = np.asarray(reconstructed_flat, dtype=np.int64)
    reconstructed_selected = np.asarray(reconstructed_selected, dtype=bool)
    if not (truth_flat.shape == reconstructed_flat.shape == reconstructed_selected.shape):
        raise ValueError("truth, reconstruction, and selection arrays must have equal shapes")
    if weights is None:
        weights = np.ones(truth_flat.size, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != truth_flat.shape:
            raise ValueError("weights must match event arrays")

    truth_inside = (truth_flat >= 0) & (truth_flat < number_of_bins)
    rec_inside = (
        reconstructed_selected
        & (reconstructed_flat >= 0)
        & (reconstructed_flat < number_of_bins)
    )

    truth_total = np.bincount(
        truth_flat[truth_inside], weights=weights[truth_inside], minlength=number_of_bins
    ).astype(float)
    reconstructed_total = np.bincount(
        reconstructed_flat[rec_inside], weights=weights[rec_inside], minlength=number_of_bins
    ).astype(float)

    migrated = truth_inside & rec_inside
    counts = csr_matrix(
        (
            weights[migrated],
            (reconstructed_flat[migrated], truth_flat[migrated]),
        ),
        shape=(number_of_bins, number_of_bins),
        dtype=float,
    )
    inverse_truth = np.divide(
        1.0, truth_total, out=np.zeros_like(truth_total), where=truth_total > 0
    )
    core = counts.dot(diags(inverse_truth, format="csr")).tocsr()
    efficiency = np.asarray(core.sum(axis=0)).ravel()

    feed_in = (~truth_inside) & rec_inside
    feed_counts = np.bincount(
        reconstructed_flat[feed_in], weights=weights[feed_in], minlength=number_of_bins
    ).astype(float)
    feed_sum = float(feed_counts.sum())
    feed_shape = feed_counts / feed_sum if feed_sum > 0 else np.zeros(number_of_bins)
    rec_sum = float(reconstructed_total.sum())
    feed_fraction = feed_sum / rec_sum if rec_sum > 0 else 0.0
    matrix = hstack([core, csr_matrix(feed_shape[:, None])], format="csr")

    variance_sum = _multinomial_variance_sum(core.tocsc(), truth_total)
    return ResponseResult(
        matrix=matrix,
        core=core,
        truth_total=truth_total,
        reconstructed_total=reconstructed_total,
        efficiency=efficiency,
        feed_in_fraction=feed_fraction,
        feed_in_shape=feed_shape,
        response_variance_sum=variance_sum,
    )


def _multinomial_variance_sum(core: csc_matrix, truth_total: Array) -> Array:
    variance = np.zeros_like(truth_total, dtype=float)
    for column in np.flatnonzero(truth_total > 0):
        start, stop = core.indptr[column : column + 2]
        probabilities = core.data[start:stop]
        variance[column] = np.sum(probabilities * (1.0 - probabilities)) / truth_total[column]
    return variance
