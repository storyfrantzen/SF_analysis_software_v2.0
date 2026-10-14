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


def build_response_from_counts(
    truth_total: Array,
    reconstructed_total: Array,
    migration_rows: Array,
    migration_cols: Array,
    migration_weights: Array,
    feed_counts: Array,
    *,
    compute_variance: bool = True,
) -> ResponseResult:
    """Build the response from pre-accumulated histogram and migration counts."""
    truth_total = np.asarray(truth_total, dtype=float)
    reconstructed_total = np.asarray(reconstructed_total, dtype=float)
    feed_counts = np.asarray(feed_counts, dtype=float)
    if truth_total.ndim != 1:
        raise ValueError("truth_total must be one-dimensional")
    number_of_bins = truth_total.size
    if reconstructed_total.shape != truth_total.shape or feed_counts.shape != truth_total.shape:
        raise ValueError("histogram shapes must match")

    migration_rows = np.asarray(migration_rows, dtype=np.int64)
    migration_cols = np.asarray(migration_cols, dtype=np.int64)
    migration_weights = np.asarray(migration_weights, dtype=float)
    if not (migration_rows.shape == migration_cols.shape == migration_weights.shape):
        raise ValueError("migration arrays must have equal shapes")

    counts = csr_matrix(
        (migration_weights, (migration_rows, migration_cols)),
        shape=(number_of_bins, number_of_bins),
        dtype=float,
    )
    inverse_truth = np.divide(
        1.0, truth_total, out=np.zeros_like(truth_total), where=truth_total > 0
    )
    core = counts.dot(diags(inverse_truth, format="csr")).tocsr()
    efficiency = np.asarray(core.sum(axis=0)).ravel()

    feed_sum = float(feed_counts.sum())
    feed_shape = feed_counts / feed_sum if feed_sum > 0 else np.zeros(number_of_bins)
    rec_sum = float(reconstructed_total.sum())
    feed_fraction = feed_sum / rec_sum if rec_sum > 0 else 0.0
    matrix = hstack([core, csr_matrix(feed_shape[:, None])], format="csr")
    variance_sum = (
        _multinomial_variance_sum(core.tocsc(), truth_total)
        if compute_variance
        else np.zeros_like(truth_total, dtype=float)
    )
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


def require_integer_counts(values: Array, name: str) -> None:
    """Reject weighted or otherwise invalid response sufficient statistics."""
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not np.allclose(values, np.rint(values), rtol=0.0, atol=1.0e-7):
        raise ValueError(
            f"{name} are weighted rather than integer event counts; "
            "count-bootstrap response replicas require unweighted counts"
        )


def response_counts_from_artifacts(
    core: csr_matrix,
    truth_total: Array,
    reconstructed_total: Array,
    feed_in_fraction: float,
    feed_in_shape: Array,
) -> tuple[Array, Array, csr_matrix]:
    """Recover integer truth, feed-in, and migration counts from response files."""
    core = core.tocsr()
    truth_total = np.asarray(truth_total, dtype=float)
    reconstructed_total = np.asarray(reconstructed_total, dtype=float)
    feed_in_shape = np.asarray(feed_in_shape, dtype=float)
    if core.shape != (truth_total.size, truth_total.size):
        raise ValueError("response core dimensions do not match truth counts")
    if reconstructed_total.shape != truth_total.shape:
        raise ValueError("reconstructed counts do not match truth counts")
    if feed_in_shape.shape != truth_total.shape:
        raise ValueError("feed-in shape does not match truth counts")
    fraction = float(feed_in_fraction)
    if not np.isfinite(fraction) or fraction < 0.0 or fraction >= 1.0:
        raise ValueError("feed-in fraction must be finite and below one")

    migration = core.multiply(truth_total[np.newaxis, :]).tocsr()
    feed_total = fraction * float(reconstructed_total.sum())
    feed_counts = feed_total * feed_in_shape
    require_integer_counts(truth_total, "truth counts")
    require_integer_counts(reconstructed_total, "reconstructed counts")
    require_integer_counts(migration.data, "migration counts")
    require_integer_counts(feed_counts, "feed-in counts")
    rebuilt_reconstructed = (
        np.asarray(migration.sum(axis=1)).ravel() + feed_counts
    )
    if not np.allclose(
        rebuilt_reconstructed,
        reconstructed_total,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "response matrix and metadata do not reconstruct the stored REC counts"
        )
    return truth_total, feed_counts, migration


def count_bootstrap_response(
    truth_total: Array,
    feed_counts: Array,
    migration_counts: csr_matrix,
    rng: np.random.Generator,
) -> ResponseResult:
    """Poisson-resample migration, missed, and feed-in event counts.

    Independent Poisson cells are the unconditional counterpart of a
    multinomial response experiment. The generated truth denominator is
    rebuilt from fluctuated migration and missed-event counts, preserving
    their normalization covariance in every response replica.
    """
    truth_total = np.asarray(truth_total, dtype=float)
    feed_counts = np.asarray(feed_counts, dtype=float)
    migration = migration_counts.tocoo(copy=True)
    require_integer_counts(truth_total, "truth counts")
    require_integer_counts(feed_counts, "feed-in counts")
    require_integer_counts(migration.data, "migration counts")
    migrated_by_truth = np.asarray(migration_counts.sum(axis=0)).ravel()
    missed = np.clip(truth_total - migrated_by_truth, 0.0, None)
    require_integer_counts(missed, "missed counts")

    sampled_migration = rng.poisson(
        np.rint(migration.data).astype(np.int64)
    ).astype(float)
    sampled_missed = rng.poisson(np.rint(missed).astype(np.int64)).astype(float)
    sampled_feed = rng.poisson(np.rint(feed_counts).astype(np.int64)).astype(float)
    sampled_counts = csr_matrix(
        (sampled_migration, (migration.row, migration.col)),
        shape=migration.shape,
    )
    sampled_truth = np.asarray(sampled_counts.sum(axis=0)).ravel() + sampled_missed
    sampled_reconstructed = (
        np.asarray(sampled_counts.sum(axis=1)).ravel() + sampled_feed
    )
    sampled = sampled_counts.tocoo()
    return build_response_from_counts(
        sampled_truth,
        sampled_reconstructed,
        sampled.row,
        sampled.col,
        sampled.data,
        sampled_feed,
        compute_variance=False,
    )


def _multinomial_variance_sum(core: csc_matrix, truth_total: Array) -> Array:
    variance = np.zeros_like(truth_total, dtype=float)
    for column in np.flatnonzero(truth_total > 0):
        start, stop = core.indptr[column : column + 2]
        probabilities = core.data[start:stop]
        variance[column] = np.sum(probabilities * (1.0 - probabilities)) / truth_total[column]
    return variance
