from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .binning import AnalysisBinning
from .response import count_bootstrap_response, response_counts_from_artifacts


Array = np.ndarray


@dataclass(frozen=True)
class UnfoldingResult:
    unfolded: Array
    history: Array
    kl_divergence: Array


def subtract_feed_in(measured: Array, fraction: float, shape: Array) -> Array:
    measured = np.asarray(measured, dtype=float)
    shape = np.asarray(shape, dtype=float)
    if measured.shape != shape.shape:
        raise ValueError("measured spectrum and feed-in shape must match")
    return np.clip(measured - fraction * measured.sum() * shape, 0.0, None)


def iterative_bayes(
    response_core: csr_matrix,
    measured: Array,
    efficiency: Array,
    iterations: int,
    prior: Array | None = None,
    minimum_acceptance: float = 0.005,
) -> UnfoldingResult:
    measured = np.asarray(measured, dtype=float)
    efficiency = np.asarray(efficiency, dtype=float)
    number_of_bins = efficiency.size
    if response_core.shape != (number_of_bins, number_of_bins):
        raise ValueError("response core dimensions do not match efficiency")
    if measured.shape != (number_of_bins,):
        raise ValueError("measured spectrum dimensions do not match efficiency")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")

    if prior is None:
        estimate = np.ones(number_of_bins, dtype=float)
    else:
        estimate = np.asarray(prior, dtype=float).copy()
        if estimate.shape != (number_of_bins,):
            raise ValueError("prior dimensions do not match efficiency")

    valid = efficiency > minimum_acceptance
    estimate[~valid] = 0.0
    transpose = response_core.T.tocsr()
    history: list[Array] = []
    divergences: list[float] = []

    for _ in range(iterations):
        denominator = response_core.dot(estimate)
        ratio = np.divide(
            measured, denominator, out=np.zeros_like(measured), where=denominator > 0
        )
        update = transpose.dot(ratio)
        next_estimate = np.zeros_like(estimate)
        next_estimate[valid] = estimate[valid] * update[valid] / efficiency[valid]
        np.maximum(next_estimate, 0.0, out=next_estimate)
        divergences.append(_kl(next_estimate, estimate))
        history.append(next_estimate.copy())
        estimate = next_estimate

    return UnfoldingResult(
        unfolded=estimate,
        history=np.asarray(history),
        kl_divergence=np.asarray(divergences),
    )


def bootstrap_ensemble(
    response_core: csr_matrix,
    measured: Array,
    efficiency: Array,
    iterations: int,
    prior: Array,
    minimum_acceptance: float = 0.005,
    experiments: int = 200,
    seed: int | None = None,
    feed_in_fraction: float = 0.0,
    feed_in_shape: Array | None = None,
    measured_variance: Array | None = None,
    recompute_data_prior: bool = False,
) -> Array:
    """Bootstrap the unfolded estimator.

    Set ``recompute_data_prior`` when the central IBU prior was constructed as
    ``measured / efficiency``.  Each replica then rebuilds that prior from its
    fluctuated measured spectrum.  Holding a data-derived prior fixed omits a
    first-order source of statistical variation, especially for early IBU
    iterations.  Leave it false only for a prior independent of the measured
    sample.
    """
    if experiments <= 0:
        raise ValueError("experiments must be positive")
    measured = np.asarray(measured, dtype=float)
    variance = (
        measured
        if measured_variance is None
        else np.asarray(measured_variance, dtype=float)
    )
    if variance.shape != measured.shape:
        raise ValueError("measured variance dimensions do not match measured spectrum")
    if np.any(~np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("measured variance must be finite and nonnegative")
    shape = (
        np.zeros_like(measured)
        if feed_in_shape is None
        else np.asarray(feed_in_shape, dtype=float)
    )
    rng = np.random.default_rng(seed)
    samples = np.empty((experiments, measured.size), dtype=float)
    acceptance_valid = np.asarray(efficiency, dtype=float) > minimum_acceptance
    for index in range(experiments):
        fluctuated = fluctuate_weighted_poisson(rng, measured, variance)
        corrected = subtract_feed_in(fluctuated, feed_in_fraction, shape)
        replica_prior = prior
        if recompute_data_prior:
            replica_prior = np.divide(
                fluctuated,
                efficiency,
                out=np.zeros_like(fluctuated),
                where=acceptance_valid,
            )
        samples[index] = iterative_bayes(
            response_core,
            corrected,
            efficiency,
            iterations,
            prior=replica_prior,
            minimum_acceptance=minimum_acceptance,
        ).unfolded
    return samples


def joint_count_bootstrap_ensemble(
    response_core: csr_matrix,
    truth_total: Array,
    reconstructed_total: Array,
    feed_in_fraction: float,
    feed_in_shape: Array,
    measured: Array,
    measured_variance: Array,
    iterations: int,
    *,
    minimum_acceptance: float = 0.005,
    experiments: int = 200,
    seed: int | None = None,
    progress_label: str | None = None,
) -> Array:
    """Bootstrap the complete data-and-finite-response unfolding estimator."""
    if experiments < 2:
        raise ValueError("joint count bootstrap requires at least two replicas")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    measured = np.asarray(measured, dtype=float)
    measured_variance = np.asarray(measured_variance, dtype=float)
    if measured.shape != measured_variance.shape:
        raise ValueError("measured variance dimensions do not match measured spectrum")
    if np.any(~np.isfinite(measured_variance)) or np.any(measured_variance < 0.0):
        raise ValueError("measured variance must be finite and nonnegative")
    truth, feed, migration = response_counts_from_artifacts(
        response_core,
        truth_total,
        reconstructed_total,
        feed_in_fraction,
        feed_in_shape,
    )
    response_rng = np.random.default_rng(seed)
    data_seed = None if seed is None else int(seed) ^ 0xD1B54A32D192ED03
    data_rng = np.random.default_rng(data_seed)
    samples = np.empty((experiments, measured.size), dtype=float)
    progress_step = max(1, experiments // 5)
    for replica_index in range(experiments):
        response = count_bootstrap_response(truth, feed, migration, response_rng)
        fluctuated = fluctuate_weighted_poisson(
            data_rng, measured, measured_variance
        )
        acceptance_valid = response.efficiency > minimum_acceptance
        prior = np.divide(
            fluctuated,
            response.efficiency,
            out=np.zeros_like(fluctuated),
            where=acceptance_valid,
        )
        if iterations == 0:
            samples[replica_index] = prior
        else:
            corrected = subtract_feed_in(
                fluctuated,
                response.feed_in_fraction,
                response.feed_in_shape,
            )
            samples[replica_index] = iterative_bayes(
                response.core,
                corrected,
                response.efficiency,
                iterations,
                prior=prior,
                minimum_acceptance=minimum_acceptance,
            ).unfolded
        completed = replica_index + 1
        if progress_label is not None and (
            completed % progress_step == 0 or completed == experiments
        ):
            print(
                f"[COUNT-BOOTSTRAP] {progress_label} replicas "
                f"{completed}/{experiments}",
                flush=True,
            )
    return samples


def bootstrap_uncertainty(
    response_core: csr_matrix,
    measured: Array,
    efficiency: Array,
    iterations: int,
    prior: Array,
    minimum_acceptance: float = 0.005,
    experiments: int = 200,
    seed: int | None = None,
    feed_in_fraction: float = 0.0,
    feed_in_shape: Array | None = None,
    measured_variance: Array | None = None,
    recompute_data_prior: bool = False,
) -> tuple[Array, Array]:
    samples = bootstrap_ensemble(
        response_core,
        measured,
        efficiency,
        iterations,
        prior,
        minimum_acceptance=minimum_acceptance,
        experiments=experiments,
        seed=seed,
        feed_in_fraction=feed_in_fraction,
        feed_in_shape=feed_in_shape,
        measured_variance=measured_variance,
        recompute_data_prior=recompute_data_prior,
    )
    return samples.mean(axis=0), samples.std(axis=0, ddof=1)


def phi_block_covariance(samples: Array, binning: AnalysisBinning) -> Array:
    """Return phi covariance blocks in analysis ``(Q2, xB, -t)`` cell order.

    Unfolded vectors use the legacy flat order ``(xB, Q2, phi, -t)`` with
    ``-t`` fastest.  Consequently, adjacent flat entries are not adjacent phi
    bins.  Convert through :class:`AnalysisBinning` before forming blocks.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("bootstrap samples must be a 2D array with at least two replicas")
    if samples.shape[1] != binning.size:
        raise ValueError("bootstrap samples do not match the analysis binning")
    phi_bins = binning.shape[-1]
    blocks = binning.unflatten(samples).reshape(samples.shape[0], -1, phi_bins)
    centered = blocks - blocks.mean(axis=0, keepdims=True)
    return np.einsum("eci,ecj->cij", centered, centered) / (samples.shape[0] - 1)


def jackknife_phi_covariance(samples: Array, binning: AnalysisBinning) -> Array:
    """Return delete-one-subsample covariance blocks among phi bins.

    ``samples`` contains estimates recomputed after deleting each independent
    response subsample in turn.  The jackknife normalization is therefore
    ``(m - 1) / m`` times the centered outer-product sum, rather than the
    ordinary sample-covariance normalization.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("jackknife samples must be a 2D array with at least two replicas")
    if samples.shape[1] != binning.size:
        raise ValueError("jackknife samples do not match the analysis binning")
    phi_bins = binning.shape[-1]
    blocks = binning.unflatten(samples).reshape(samples.shape[0], -1, phi_bins)
    centered = blocks - blocks.mean(axis=0, keepdims=True)
    replicas = samples.shape[0]
    return (
        np.einsum("eci,ecj->cij", centered, centered)
        * (replicas - 1)
        / replicas
    )


def diagonal_phi_covariance(variance: Array, binning: AnalysisBinning) -> Array:
    """Pack legacy-flat variances into analysis-ordered phi covariance blocks."""
    variance = np.asarray(variance, dtype=float)
    if variance.ndim != 1 or variance.size != binning.size:
        raise ValueError("flattened variance does not match the analysis binning")
    phi_bins = binning.shape[-1]
    blocks = binning.unflatten(variance).reshape(-1, phi_bins)
    covariance = np.zeros((blocks.shape[0], phi_bins, phi_bins), dtype=float)
    diagonal = np.arange(phi_bins)
    covariance[:, diagonal, diagonal] = blocks
    return covariance


def fluctuate_weighted_poisson(
    rng: np.random.Generator, measured: Array, variance: Array
) -> Array:
    """Moment-match a weighted Poisson sum using an effective count and scale."""
    measured = np.asarray(measured, dtype=float)
    variance = np.asarray(variance, dtype=float)
    fluctuated = np.zeros_like(measured)
    positive = (measured > 0.0) & (variance > 0.0)
    scale = np.divide(
        variance,
        measured,
        out=np.ones_like(measured),
        where=positive,
    )
    effective_count = np.divide(
        measured,
        scale,
        out=np.zeros_like(measured),
        where=positive,
    )
    fluctuated[positive] = (
        rng.poisson(effective_count[positive]).astype(float) * scale[positive]
    )
    return fluctuated


# Compatibility for code that imported the earlier private helper.
_fluctuate_weighted_poisson = fluctuate_weighted_poisson


def _kl(new: Array, old: Array) -> float:
    new_sum, old_sum = new.sum(), old.sum()
    if new_sum <= 0 or old_sum <= 0:
        return 0.0
    p, q = new / new_sum, old / old_sum
    valid = (p > 0) & (q > 0)
    return float(np.sum(p[valid] * np.log(p[valid] / q[valid])))
