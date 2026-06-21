from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


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
) -> tuple[Array, Array]:
    if experiments <= 0:
        raise ValueError("experiments must be positive")
    measured = np.asarray(measured, dtype=float)
    shape = np.zeros_like(measured) if feed_in_shape is None else np.asarray(feed_in_shape, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty((experiments, measured.size), dtype=float)
    for index in range(experiments):
        fluctuated = rng.poisson(measured).astype(float)
        corrected = subtract_feed_in(fluctuated, feed_in_fraction, shape)
        samples[index] = iterative_bayes(
            response_core,
            corrected,
            efficiency,
            iterations,
            prior=prior,
            minimum_acceptance=minimum_acceptance,
        ).unfolded
    return samples.mean(axis=0), samples.std(axis=0, ddof=1)


def _kl(new: Array, old: Array) -> float:
    new_sum, old_sum = new.sum(), old.sum()
    if new_sum <= 0 or old_sum <= 0:
        return 0.0
    p, q = new / new_sum, old / old_sum
    valid = (p > 0) & (q > 0)
    return float(np.sum(p[valid] * np.log(p[valid] / q[valid])))
