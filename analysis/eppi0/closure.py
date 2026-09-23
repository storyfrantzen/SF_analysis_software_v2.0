from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix

from .binning import AnalysisBinning
from .harmonics import fit_grid
from .response import (
    ResponseResult,
    build_response_from_counts,
    count_bootstrap_response,
    require_integer_counts,
)
from .unfolding import (
    bootstrap_ensemble,
    diagonal_phi_covariance,
    fluctuate_weighted_poisson,
    iterative_bayes,
    jackknife_phi_covariance,
    phi_block_covariance,
    subtract_feed_in,
)


Array = np.ndarray


@dataclass(frozen=True)
class SplitClosureInputs:
    """Fold-local response counts and independently weighted validation spectra."""

    fold_truth_total: Array
    fold_reconstructed_total: Array
    fold_feed_counts: Array
    fold_migration_counts: tuple[csr_matrix, ...]
    validation_truth: Array
    validation_measured: Array
    validation_variance: Array
    stress_names: tuple[str, ...]

    def validate(self) -> None:
        truth = np.asarray(self.fold_truth_total, dtype=float)
        reconstructed = np.asarray(self.fold_reconstructed_total, dtype=float)
        feed = np.asarray(self.fold_feed_counts, dtype=float)
        if truth.ndim != 2:
            raise ValueError("fold_truth_total must have shape (fold, bin)")
        if reconstructed.shape != truth.shape or feed.shape != truth.shape:
            raise ValueError("all fold response histograms must have the same shape")
        folds, bins = truth.shape
        if folds < 2:
            raise ValueError("at least two folds are required for split-sample closure")
        if len(self.fold_migration_counts) != folds:
            raise ValueError("one migration-count matrix is required per fold")
        if any(matrix.shape != (bins, bins) for matrix in self.fold_migration_counts):
            raise ValueError("fold migration-count dimensions do not match the histograms")
        expected = (len(self.stress_names), folds, bins)
        for name, values in (
            ("validation_truth", self.validation_truth),
            ("validation_measured", self.validation_measured),
            ("validation_variance", self.validation_variance),
        ):
            values = np.asarray(values, dtype=float)
            if values.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class ClosureScanResult:
    unfolded: Array
    uncertainty: Array
    validity: Array
    refolded: Array
    training_efficiency: Array
    metrics: tuple[dict[str, float | int | str], ...]
    harmonic_cell_metrics: tuple[dict[str, float | int | str], ...]
    fixed_truth_metrics: tuple[dict[str, float | int | str], ...]
    fixed_truth_harmonic_cell_metrics: tuple[dict[str, float | int | str], ...]
    recommended_iterations: int


def deterministic_folds(
    source_file_id: Array,
    source_event_index: Array,
    number_of_folds: int,
    seed: int = 731_921,
) -> Array:
    """Assign source-aware event identities to stable folds with SplitMix64."""
    if number_of_folds < 2:
        raise ValueError("number_of_folds must be at least two")
    file_id, event_index = np.broadcast_arrays(
        np.asarray(source_file_id, dtype=np.uint64),
        np.asarray(source_event_index, dtype=np.uint64),
    )
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    seed_term = np.uint64((int(seed) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    x = (
        file_id * np.uint64(0xD6E8FEB86659FD93)
        + event_index * np.uint64(0xA5A3564E27F8862B)
        + seed_term
    ) & mask
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & mask
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & mask
    x ^= x >> np.uint64(31)
    return (x % np.uint64(number_of_folds)).astype(np.int16, copy=False)


def stress_weights(
    q2: Array,
    xb: Array,
    minus_t: Array,
    phi_rad: Array,
    names: Iterable[str],
    *,
    strength: float = 0.6,
    q2_range: tuple[float, float] | None = None,
    xb_range: tuple[float, float] | None = None,
    t_range: tuple[float, float] | None = None,
) -> Array:
    """Return positive truth-shape stress factors for closure pseudo-data."""
    q2, xb, minus_t, phi_rad = np.broadcast_arrays(
        np.asarray(q2, dtype=float),
        np.asarray(xb, dtype=float),
        np.asarray(minus_t, dtype=float),
        np.asarray(phi_rad, dtype=float),
    )
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("stress strength must be finite and nonnegative")
    zq = _scaled_coordinate(q2, q2_range)
    zx = _scaled_coordinate(xb, xb_range)
    zt = _scaled_coordinate(minus_t, t_range)
    phi = np.mod(np.nan_to_num(phi_rad, nan=0.0), 2.0 * np.pi)
    output = []
    for name in names:
        if name == "nominal":
            factor = np.ones(q2.shape, dtype=float)
        elif name == "q2_tilt":
            factor = np.exp(strength * zq)
        elif name == "xb_t_tilt":
            factor = np.exp(0.5 * strength * (zx + zt))
        elif name == "phi_harmonic":
            factor = 1.0 + strength * (
                0.45 * np.cos(phi) + 0.25 * np.cos(2.0 * phi)
            )
        elif name == "combined":
            factor = np.exp(strength * (0.45 * zq + 0.30 * zx + 0.25 * zt))
            factor *= 1.0 + strength * (
                0.35 * np.cos(phi) - 0.20 * np.cos(2.0 * phi)
            )
        else:
            raise ValueError(f"unknown closure stress: {name}")
        output.append(np.clip(factor, 0.05, 20.0))
    return np.asarray(output, dtype=float)


def _response_from_folds(
    inputs: SplitClosureInputs, included_folds: Iterable[int]
) -> ResponseResult:
    kept = tuple(int(index) for index in included_folds)
    if not kept:
        raise ValueError("at least one fold is required to build a response")
    folds = inputs.fold_truth_total.shape[0]
    if any(index < 0 or index >= folds for index in kept):
        raise IndexError("response fold is outside the available fold range")
    if len(set(kept)) != len(kept):
        raise ValueError("response folds must be unique")
    counts = sum(
        (inputs.fold_migration_counts[index] for index in kept),
        start=csr_matrix(inputs.fold_migration_counts[0].shape, dtype=float),
    ).tocoo()
    return build_response_from_counts(
        np.sum(inputs.fold_truth_total[list(kept)], axis=0),
        np.sum(inputs.fold_reconstructed_total[list(kept)], axis=0),
        counts.row,
        counts.col,
        counts.data,
        np.sum(inputs.fold_feed_counts[list(kept)], axis=0),
    )


def training_response(inputs: SplitClosureInputs, held_out_fold: int) -> ResponseResult:
    inputs.validate()
    folds = inputs.fold_truth_total.shape[0]
    if held_out_fold < 0 or held_out_fold >= folds:
        raise IndexError("held-out fold is outside the available fold range")
    kept = [index for index in range(folds) if index != held_out_fold]
    return _response_from_folds(inputs, kept)


def training_response_jackknife(
    inputs: SplitClosureInputs, held_out_fold: int
) -> tuple[ResponseResult, ...]:
    """Build responses deleting each training fold in addition to the held-out fold."""
    inputs.validate()
    folds = inputs.fold_truth_total.shape[0]
    training_folds = [index for index in range(folds) if index != held_out_fold]
    if len(training_folds) < 2:
        raise ValueError("response jackknife requires at least three total folds")
    return tuple(
        _response_from_folds(
            inputs, [index for index in training_folds if index != deleted]
        )
        for deleted in training_folds
    )


def _training_response_counts(
    inputs: SplitClosureInputs, held_out_fold: int
) -> tuple[Array, Array, csr_matrix]:
    """Return integer-valued truth, feed, and migration training counts."""
    inputs.validate()
    folds = inputs.fold_truth_total.shape[0]
    if held_out_fold < 0 or held_out_fold >= folds:
        raise IndexError("held-out fold is outside the available fold range")
    kept = [index for index in range(folds) if index != held_out_fold]
    truth = np.sum(inputs.fold_truth_total[kept], axis=0)
    feed = np.sum(inputs.fold_feed_counts[kept], axis=0)
    migration = sum(
        (inputs.fold_migration_counts[index] for index in kept),
        start=csr_matrix(inputs.fold_migration_counts[0].shape, dtype=float),
    ).tocsr()
    require_integer_counts(truth, "training truth counts")
    require_integer_counts(feed, "training feed-in counts")
    require_integer_counts(migration.data, "training migration counts")
    migrated_by_truth = np.asarray(migration.sum(axis=0)).ravel()
    missed = truth - migrated_by_truth
    tolerance = 1.0e-7 * np.maximum(1.0, truth)
    if np.any(missed < -tolerance):
        raise ValueError("migration counts exceed generated truth counts")
    require_integer_counts(np.clip(missed, 0.0, None), "training missed counts")
    return truth, feed, migration


def _forward_measured_expectation(response: ResponseResult, truth: Array) -> Array:
    """Return the reconstructed expectation represented by a response model.

    ``response.core`` predicts selected events generated inside the truth range.
    The feed-in component is normalized so that the same fractional subtraction
    used by the unfolding estimator recovers that in-range expectation exactly.
    This makes the model suitable for fixed-truth pseudoexperiments without
    centering them on an already fluctuated held-out reconstructed histogram.
    """
    truth = np.asarray(truth, dtype=float)
    if truth.shape != response.truth_total.shape:
        raise ValueError("fixed truth target does not match the response dimensions")
    in_range = np.asarray(response.core.dot(truth)).ravel()
    fraction = float(response.feed_in_fraction)
    if not np.isfinite(fraction) or fraction < 0.0 or fraction >= 1.0:
        raise ValueError("response feed-in fraction must be finite and below one")
    if fraction == 0.0:
        return in_range
    feed_total = fraction * float(in_range.sum()) / (1.0 - fraction)
    return in_range + feed_total * np.asarray(response.feed_in_shape, dtype=float)


def _unfold_count_bootstrap_replica(
    response: ResponseResult,
    measured: Array,
    iterations: tuple[int, ...],
    *,
    minimum_acceptance: float,
) -> Array:
    """Run every requested estimator on one measured/response replica."""
    measured = np.asarray(measured, dtype=float)
    acceptance_valid = response.efficiency > minimum_acceptance
    prior = np.divide(
        measured,
        response.efficiency,
        out=np.zeros_like(measured),
        where=acceptance_valid,
    )
    corrected = subtract_feed_in(
        measured, response.feed_in_fraction, response.feed_in_shape
    )
    estimates = np.empty((len(iterations), measured.size), dtype=float)
    for iteration_index, iteration in enumerate(iterations):
        if iteration == 0:
            estimates[iteration_index] = prior
        else:
            estimates[iteration_index] = iterative_bayes(
                response.core,
                corrected,
                response.efficiency,
                iteration,
                prior=prior,
                minimum_acceptance=minimum_acceptance,
            ).unfolded
    return estimates


def _count_bootstrap_ensembles(
    inputs: SplitClosureInputs,
    held_out_fold: int,
    measured: Array,
    variance: Array,
    fixed_truth: Array,
    nominal_response: ResponseResult,
    iterations: tuple[int, ...],
    *,
    minimum_acceptance: float,
    experiments: int,
    seed: int,
    progress_label: str | None = None,
) -> tuple[Array, Array]:
    """Return observed-centered and fixed-truth estimator ensembles.

    Both ensembles share each count-bootstrap response replica.  The first
    fluctuates the observed held-out reconstructed spectrum and estimates the
    covariance of the central closure result.  The second draws independent
    Poisson pseudo-data from the nominal response applied to a fixed truth
    target.  Keeping those centers distinct prevents the held-out sample's
    original counting fluctuation from being counted a second time in the
    fixed-truth coverage numerator.
    """
    truth, feed, migration = _training_response_counts(inputs, held_out_fold)
    measured = np.asarray(measured, dtype=float)
    variance = np.asarray(variance, dtype=float)
    fixed_expectation = _forward_measured_expectation(nominal_response, fixed_truth)
    response_rng = np.random.default_rng(seed)
    observed_rng = np.random.default_rng(seed ^ 0x5DEECE66D)
    fixed_rng = np.random.default_rng(seed ^ 0xD1B54A32D192ED03)
    observed_samples = np.empty(
        (len(iterations), experiments, measured.size), dtype=float
    )
    fixed_samples = np.empty_like(observed_samples)
    progress_step = max(1, experiments // 5)
    for replica_index in range(experiments):
        response = count_bootstrap_response(truth, feed, migration, response_rng)
        observed = fluctuate_weighted_poisson(observed_rng, measured, variance)
        fixed = fixed_rng.poisson(fixed_expectation).astype(float)
        observed_samples[:, replica_index] = _unfold_count_bootstrap_replica(
            response,
            observed,
            iterations,
            minimum_acceptance=minimum_acceptance,
        )
        fixed_samples[:, replica_index] = _unfold_count_bootstrap_replica(
            response,
            fixed,
            iterations,
            minimum_acceptance=minimum_acceptance,
        )
        completed = replica_index + 1
        if progress_label is not None and (
            completed % progress_step == 0 or completed == experiments
        ):
            print(
                f"[COUNT-BOOTSTRAP] {progress_label} replicas "
                f"{completed}/{experiments}",
                flush=True,
            )
    return observed_samples, fixed_samples


def run_closure_scan(
    inputs: SplitClosureInputs,
    binning: AnalysisBinning,
    iterations: Iterable[int],
    *,
    minimum_acceptance: float = 0.005,
    minimum_truth: float = 20.0,
    bootstrap: int = 50,
    seed: int = 731_921,
    minimum_harmonic_points: int = 8,
    response_uncertainty: str = "analytic-diagonal",
) -> ClosureScanResult:
    """Evaluate held-out truth recovery, refolding, pulls, and harmonics."""
    inputs.validate()
    iteration_values = tuple(sorted(set(int(value) for value in iterations)))
    if not iteration_values or iteration_values[0] < 0:
        raise ValueError("iterations must contain nonnegative integers")
    if bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap must be zero or at least two")
    if response_uncertainty not in {
        "analytic-diagonal",
        "fold-jackknife",
        "count-bootstrap",
    }:
        raise ValueError(
            "response_uncertainty must be analytic-diagonal, fold-jackknife, "
            "or count-bootstrap"
        )
    stresses = len(inputs.stress_names)
    folds, bins = inputs.fold_truth_total.shape
    if bins != binning.size:
        raise ValueError("closure inputs do not match the analysis binning")
    if response_uncertainty == "count-bootstrap":
        minimum_replicas = 2 * (binning.shape[-1] + 3)
        if bootstrap < minimum_replicas:
            raise ValueError(
                "count-bootstrap requires at least "
                f"{minimum_replicas} replicas so its independent calibration half "
                "can invert a complete phi covariance block"
            )
    shape = (stresses, len(iteration_values), folds, bins)
    unfolded_all = np.full(shape, np.nan, dtype=float)
    uncertainty_all = np.full(shape, np.nan, dtype=float)
    validity_all = np.zeros(shape, dtype=bool)
    refolded_all = np.full(shape, np.nan, dtype=float)
    efficiencies = np.zeros((folds, bins), dtype=float)
    metrics: list[dict[str, float | int | str]] = []
    harmonic_cell_metrics: list[dict[str, float | int | str]] = []
    fixed_truth_metrics: list[dict[str, float | int | str]] = []
    fixed_truth_harmonic_cell_metrics: list[dict[str, float | int | str]] = []

    for fold in range(folds):
        response = training_response(inputs, fold)
        jackknife_responses = (
            training_response_jackknife(inputs, fold)
            if response_uncertainty == "fold-jackknife"
            else ()
        )
        efficiencies[fold] = response.efficiency
        acceptance_valid = response.efficiency > minimum_acceptance
        for stress_index, stress_name in enumerate(inputs.stress_names):
            target = np.asarray(inputs.validation_truth[stress_index, fold], dtype=float)
            measured = np.asarray(inputs.validation_measured[stress_index, fold], dtype=float)
            variance = np.asarray(inputs.validation_variance[stress_index, fold], dtype=float)
            feed_corrected = subtract_feed_in(
                measured, response.feed_in_fraction, response.feed_in_shape
            )
            prior = np.divide(
                measured,
                response.efficiency,
                out=np.zeros_like(measured),
                where=acceptance_valid,
            )
            count_ensembles = (
                _count_bootstrap_ensembles(
                    inputs,
                    fold,
                    measured,
                    variance,
                    target,
                    response,
                    iteration_values,
                    minimum_acceptance=minimum_acceptance,
                    experiments=bootstrap,
                    seed=seed + 1009 * fold + 97 * stress_index,
                    progress_label=f"fold={fold} stress={stress_name}",
                )
                if response_uncertainty == "count-bootstrap"
                else None
            )
            count_samples = count_ensembles[0] if count_ensembles is not None else None
            fixed_truth_samples = (
                count_ensembles[1] if count_ensembles is not None else None
            )
            for iteration_index, iteration in enumerate(iteration_values):
                if iteration == 0:
                    unfolded = prior.copy()
                else:
                    unfolded = iterative_bayes(
                        response.core,
                        feed_corrected,
                        response.efficiency,
                        iteration,
                        prior=prior,
                        minimum_acceptance=minimum_acceptance,
                    ).unfolded
                if count_samples is not None:
                    estimator_samples = count_samples[iteration_index]
                    sigma = estimator_samples.std(axis=0, ddof=1)
                    total_covariance_phi = phi_block_covariance(
                        estimator_samples, binning.shape[-1]
                    ).reshape(
                        binning.shape[:-1]
                        + (binning.shape[-1], binning.shape[-1])
                    )
                    covariance_samples = bootstrap
                else:
                    if iteration == 0:
                        sigma_stat = np.divide(
                            np.sqrt(variance),
                            response.efficiency,
                            out=np.zeros_like(variance),
                            where=acceptance_valid,
                        )
                        statistical_covariance_phi = diagonal_phi_covariance(
                            sigma_stat * sigma_stat, binning.shape[-1]
                        )
                    elif bootstrap >= 2:
                        bootstrap_samples = bootstrap_ensemble(
                            response.core,
                            measured,
                            response.efficiency,
                            iteration,
                            prior,
                            minimum_acceptance=minimum_acceptance,
                            experiments=bootstrap,
                            seed=seed + 1009 * fold + 97 * stress_index + iteration,
                            feed_in_fraction=response.feed_in_fraction,
                            feed_in_shape=response.feed_in_shape,
                            measured_variance=variance,
                            recompute_data_prior=True,
                        )
                        sigma_stat = bootstrap_samples.std(axis=0, ddof=1)
                        statistical_covariance_phi = phi_block_covariance(
                            bootstrap_samples, binning.shape[-1]
                        )
                    else:
                        sigma_stat = np.divide(
                            np.sqrt(variance),
                            response.efficiency,
                            out=np.zeros_like(variance),
                            where=acceptance_valid,
                        )
                        statistical_covariance_phi = diagonal_phi_covariance(
                            sigma_stat * sigma_stat, binning.shape[-1]
                        )
                    if jackknife_responses:
                        response_samples = np.empty(
                            (len(jackknife_responses), bins), dtype=float
                        )
                        for replica_index, replica in enumerate(jackknife_responses):
                            replica_valid = replica.efficiency > minimum_acceptance
                            replica_prior = np.divide(
                                measured,
                                replica.efficiency,
                                out=np.zeros_like(measured),
                                where=replica_valid,
                            )
                            if iteration == 0:
                                response_samples[replica_index] = replica_prior
                            else:
                                replica_corrected = subtract_feed_in(
                                    measured,
                                    replica.feed_in_fraction,
                                    replica.feed_in_shape,
                                )
                                response_samples[replica_index] = iterative_bayes(
                                    replica.core,
                                    replica_corrected,
                                    replica.efficiency,
                                    iteration,
                                    prior=replica_prior,
                                    minimum_acceptance=minimum_acceptance,
                                ).unfolded
                        response_covariance_phi = jackknife_phi_covariance(
                            response_samples, binning.shape[-1]
                        )
                        response_variance = np.diagonal(
                            response_covariance_phi, axis1=-2, axis2=-1
                        ).reshape(-1)
                        sigma_response = np.sqrt(
                            np.clip(response_variance, 0.0, None)
                        )
                    else:
                        sensitivity = np.divide(
                            unfolded,
                            response.efficiency,
                            out=np.zeros_like(unfolded),
                            where=acceptance_valid,
                        )
                        sigma_response = sensitivity * np.sqrt(
                            response.response_variance_sum
                        )
                        response_covariance_phi = diagonal_phi_covariance(
                            sigma_response * sigma_response, binning.shape[-1]
                        )
                    sigma = np.hypot(sigma_stat, sigma_response)
                    total_covariance_phi = (
                        statistical_covariance_phi + response_covariance_phi
                    ).reshape(
                        binning.shape[:-1]
                        + (binning.shape[-1], binning.shape[-1])
                    )
                    covariance_samples = bootstrap if bootstrap >= 2 else None
                valid = (
                    acceptance_valid
                    & (target >= minimum_truth)
                    & np.isfinite(unfolded)
                    & np.isfinite(sigma)
                    & (sigma > 0.0)
                )
                refolded = np.asarray(response.core.dot(unfolded)).ravel()
                unfolded_all[stress_index, iteration_index, fold] = unfolded
                uncertainty_all[stress_index, iteration_index, fold] = sigma
                validity_all[stress_index, iteration_index, fold] = valid
                refolded_all[stress_index, iteration_index, fold] = refolded
                metrics.append(
                    closure_metrics(
                        target,
                        unfolded,
                        sigma,
                        valid,
                        feed_corrected,
                        variance,
                        refolded,
                        binning,
                        fold=fold,
                        stress=stress_name,
                        iterations=iteration,
                        minimum_harmonic_points=minimum_harmonic_points,
                        covariance_phi=total_covariance_phi,
                        covariance_samples=covariance_samples,
                        harmonic_cell_metrics=harmonic_cell_metrics,
                    )
                )
                if fixed_truth_samples is not None:
                    calibration_count = bootstrap // 2
                    fixed_summary, fixed_cells = fixed_truth_pseudoexperiment_metrics(
                        target,
                        fixed_truth_samples[iteration_index, calibration_count:],
                        fixed_truth_samples[iteration_index, :calibration_count],
                        valid,
                        binning,
                        fold=fold,
                        stress=stress_name,
                        iterations=iteration,
                        minimum_harmonic_points=minimum_harmonic_points,
                    )
                    fixed_truth_metrics.append(fixed_summary)
                    fixed_truth_harmonic_cell_metrics.extend(fixed_cells)

    recommendation = recommend_iterations(metrics, iteration_values)
    return ClosureScanResult(
        unfolded=unfolded_all,
        uncertainty=uncertainty_all,
        validity=validity_all,
        refolded=refolded_all,
        training_efficiency=efficiencies,
        metrics=tuple(metrics),
        harmonic_cell_metrics=tuple(harmonic_cell_metrics),
        fixed_truth_metrics=tuple(fixed_truth_metrics),
        fixed_truth_harmonic_cell_metrics=tuple(
            fixed_truth_harmonic_cell_metrics
        ),
        recommended_iterations=recommendation,
    )


def closure_metrics(
    target: Array,
    unfolded: Array,
    uncertainty: Array,
    valid: Array,
    measured_feed_corrected: Array,
    measured_variance: Array,
    refolded: Array,
    binning: AnalysisBinning,
    *,
    fold: int,
    stress: str,
    iterations: int,
    minimum_harmonic_points: int,
    covariance_phi: Array | None = None,
    covariance_samples: int | None = None,
    harmonic_cell_metrics: list[dict[str, float | int | str]] | None = None,
) -> dict[str, float | int | str]:
    target = np.asarray(target, dtype=float)
    unfolded = np.asarray(unfolded, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    delta = unfolded - target
    fractional = np.divide(delta, target, out=np.full_like(delta, np.nan), where=target > 0)
    pull = np.divide(
        delta,
        uncertainty,
        out=np.full_like(delta, np.nan),
        where=uncertainty > 0,
    )
    values = pull[valid]
    fraction_values = fractional[valid]
    target_norm = float(np.sum(target[valid] ** 2))
    normalized_mse = (
        float(np.sum(delta[valid] ** 2) / target_norm) if target_norm > 0.0 else np.nan
    )
    target_total = float(np.sum(target[valid]))
    global_bias = (
        float(np.sum(delta[valid]) / target_total) if target_total > 0.0 else np.nan
    )

    rec_valid = (
        np.isfinite(measured_feed_corrected)
        & np.isfinite(measured_variance)
        & (measured_variance > 0.0)
        & np.isfinite(refolded)
    )
    rec_pull = np.divide(
        refolded - measured_feed_corrected,
        np.sqrt(measured_variance),
        out=np.full_like(refolded, np.nan),
        where=rec_valid,
    )
    harmonic, harmonic_cells = _harmonic_metrics(
        target,
        unfolded,
        uncertainty,
        valid,
        binning,
        minimum_points=minimum_harmonic_points,
        covariance_phi=covariance_phi,
        covariance_samples=covariance_samples,
    )
    if harmonic_cell_metrics is not None:
        for row in harmonic_cells:
            row.update(
                {
                    "fold": int(fold),
                    "stress": stress,
                    "iterations": int(iterations),
                }
            )
            harmonic_cell_metrics.append(row)
    result: dict[str, float | int | str] = {
        "fold": int(fold),
        "stress": stress,
        "iterations": int(iterations),
        "valid_bins": int(np.count_nonzero(valid)),
        "target_total": target_total,
        "unfolded_total": float(np.sum(unfolded[valid])),
        "global_relative_bias": global_bias,
        "normalized_mse": normalized_mse,
        "median_absolute_fractional_bias": _median_or_nan(np.abs(fraction_values)),
        "fractional_bias_q10": _quantile_or_nan(fraction_values, 0.10),
        "fractional_bias_median": _quantile_or_nan(fraction_values, 0.50),
        "fractional_bias_q90": _quantile_or_nan(fraction_values, 0.90),
        "pull_mean": _mean_or_nan(values),
        "pull_std": _std_or_nan(values),
        "pull_q10": _quantile_or_nan(values, 0.10),
        "pull_median": _quantile_or_nan(values, 0.50),
        "pull_q90": _quantile_or_nan(values, 0.90),
        "pull_absolute_gt2": int(np.count_nonzero(np.abs(values) > 2.0)),
        "pull_absolute_gt3": int(np.count_nonzero(np.abs(values) > 3.0)),
        "coverage_1sigma": _mean_or_nan(np.abs(values) <= 1.0),
        "coverage_2sigma": _mean_or_nan(np.abs(values) <= 2.0),
        "refold_valid_bins": int(np.count_nonzero(rec_valid)),
        "refold_chi2_ndf": _mean_or_nan(rec_pull[rec_valid] ** 2),
        "refold_pull_mean": _mean_or_nan(rec_pull[rec_valid]),
        "refold_pull_std": _std_or_nan(rec_pull[rec_valid]),
    }
    result.update(harmonic)
    return result


def fixed_truth_pseudoexperiment_metrics(
    target: Array,
    evaluation_samples: Array,
    calibration_samples: Array,
    valid: Array,
    binning: AnalysisBinning,
    *,
    fold: int,
    stress: str,
    iterations: int,
    minimum_harmonic_points: int,
) -> tuple[
    dict[str, float | int | str],
    list[dict[str, float | int | str]],
]:
    """Measure coverage using disjoint covariance and evaluation replicas.

    The held-out truth histogram is fixed.  The first half of the joint
    data-and-response replicas estimates covariance; the second half evaluates
    pulls and coverage.  This avoids both validation-target shot noise and the
    circular use of one ensemble to define and assess its own covariance.
    """
    target = np.asarray(target, dtype=float)
    evaluation = np.asarray(evaluation_samples, dtype=float)
    calibration = np.asarray(calibration_samples, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if evaluation.ndim != 2 or calibration.ndim != 2:
        raise ValueError("fixed-truth pseudoexperiment samples must be two-dimensional")
    if evaluation.shape[1] != target.size or calibration.shape[1] != target.size:
        raise ValueError("fixed-truth pseudoexperiment bins do not match the target")
    if evaluation.shape[0] < 2 or calibration.shape[0] < 2:
        raise ValueError("both pseudoexperiment halves require at least two replicas")

    phi_bins = binning.shape[-1]
    covariance_phi = phi_block_covariance(calibration, phi_bins).reshape(
        binning.shape[:-1] + (phi_bins, phi_bins)
    )
    variance = np.diagonal(covariance_phi, axis1=-2, axis2=-1).reshape(-1)
    sigma = np.sqrt(np.clip(variance, 0.0, None))
    pseudo_valid = valid & np.isfinite(sigma) & (sigma > 0.0)
    pulls = np.divide(
        evaluation - target[None, :],
        sigma[None, :],
        out=np.full_like(evaluation, np.nan),
        where=pseudo_valid[None, :],
    )[:, pseudo_valid]
    deltas = evaluation[:, pseudo_valid] - target[pseudo_valid][None, :]
    target_norm = float(np.sum(target[pseudo_valid] ** 2))
    target_total = float(np.sum(target[pseudo_valid]))
    mean_estimate = evaluation[:, pseudo_valid].mean(axis=0)
    summary: dict[str, float | int | str] = {
        "fold": int(fold),
        "stress": stress,
        "iterations": int(iterations),
        "calibration_pseudoexperiments": int(calibration.shape[0]),
        "evaluation_pseudoexperiments": int(evaluation.shape[0]),
        "valid_bins": int(np.count_nonzero(pseudo_valid)),
        "target_total": target_total,
        "unfolded_total": float(np.sum(mean_estimate)),
        "global_relative_bias": (
            float(np.sum(mean_estimate - target[pseudo_valid]) / target_total)
            if target_total > 0.0
            else np.nan
        ),
        "normalized_mse": (
            float(np.mean(np.sum(deltas * deltas, axis=1)) / target_norm)
            if target_norm > 0.0
            else np.nan
        ),
        "pull_mean": _mean_or_nan(pulls[np.isfinite(pulls)]),
        "pull_std": _std_or_nan(pulls[np.isfinite(pulls)]),
        "coverage_1sigma": _mean_or_nan(np.abs(pulls[np.isfinite(pulls)]) <= 1.0),
        "coverage_2sigma": _mean_or_nan(np.abs(pulls[np.isfinite(pulls)]) <= 2.0),
    }
    harmonic_summary, cell_rows = _fixed_truth_harmonic_metrics(
        target,
        evaluation,
        sigma,
        pseudo_valid,
        covariance_phi,
        binning,
        covariance_samples=calibration.shape[0],
        minimum_points=minimum_harmonic_points,
    )
    summary.update(harmonic_summary)
    for row in cell_rows:
        row.update(
            {
                "fold": int(fold),
                "stress": stress,
                "iterations": int(iterations),
                "calibration_pseudoexperiments": int(calibration.shape[0]),
                "evaluation_pseudoexperiments": int(evaluation.shape[0]),
            }
        )
    return summary, cell_rows


def _fixed_truth_harmonic_metrics(
    target: Array,
    evaluation_samples: Array,
    uncertainty: Array,
    valid: Array,
    covariance_phi: Array,
    binning: AnalysisBinning,
    *,
    covariance_samples: int,
    minimum_points: int,
) -> tuple[
    dict[str, float | int],
    list[dict[str, float | int | str]],
]:
    """Apply fixed GLS maps to independent evaluation pseudoexperiments."""
    phi_bins = binning.shape[-1]
    centers = 0.5 * (binning.phi_edges[:-1] + binning.phi_edges[1:])
    design_all = np.column_stack(
        (
            np.ones(phi_bins),
            np.cos(np.deg2rad(centers)),
            np.cos(2.0 * np.deg2rad(centers)),
        )
    )
    target_blocks = np.asarray(target, dtype=float).reshape(-1, phi_bins)
    evaluation_blocks = np.asarray(evaluation_samples, dtype=float).reshape(
        evaluation_samples.shape[0], -1, phi_bins
    )
    uncertainty_blocks = np.asarray(uncertainty, dtype=float).reshape(-1, phi_bins)
    valid_blocks = np.asarray(valid, dtype=bool).reshape(-1, phi_bins)
    covariance_blocks = np.asarray(covariance_phi, dtype=float).reshape(
        -1, phi_bins, phi_bins
    )
    minimum = max(4, min(int(minimum_points), phi_bins))
    coefficient_pulls: list[list[Array]] = [[], [], []]
    cell_rows: list[dict[str, float | int | str]] = []
    for cell in range(target_blocks.shape[0]):
        mask = (
            valid_blocks[cell]
            & np.isfinite(target_blocks[cell])
            & np.isfinite(uncertainty_blocks[cell])
            & (uncertainty_blocks[cell] > 0.0)
        )
        points = int(np.count_nonzero(mask))
        if points < minimum:
            continue
        covariance = covariance_blocks[cell][np.ix_(mask, mask)]
        if not np.all(np.isfinite(covariance)):
            continue
        try:
            lower = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            continue
        design = design_all[mask]
        weighted_design = np.linalg.solve(lower, design)
        if np.linalg.matrix_rank(weighted_design) < 3:
            continue
        normal = weighted_design.T @ weighted_design
        try:
            parameter_covariance = np.linalg.inv(normal)
        except np.linalg.LinAlgError:
            continue
        if covariance_samples <= points + 2:
            continue
        precision_correction = (
            covariance_samples - points - 2.0
        ) / (covariance_samples - 1.0)
        parameter_uncertainty = np.sqrt(
            np.clip(np.diag(parameter_covariance) / precision_correction, 0.0, None)
        )
        covariance_inverse_design = np.linalg.solve(
            lower.T, np.linalg.solve(lower, design)
        )
        linear_map = parameter_covariance @ covariance_inverse_design.T
        target_parameters = linear_map @ target_blocks[cell, mask]
        estimates = evaluation_blocks[:, cell, mask] @ linear_map.T
        pulls = np.divide(
            estimates - target_parameters[None, :],
            parameter_uncertainty[None, :],
            out=np.full_like(estimates, np.nan),
            where=parameter_uncertainty[None, :] > 0.0,
        )
        index = tuple(
            int(value)
            for value in np.unravel_index(cell, binning.shape[:-1])
        )
        row: dict[str, float | int | str] = _cell_coordinates(binning, index)
        row["phi_points"] = points
        for coefficient, label in enumerate(("A", "B", "C")):
            finite = pulls[:, coefficient][np.isfinite(pulls[:, coefficient])]
            coefficient_pulls[coefficient].append(finite)
            row[f"target_{label}"] = float(target_parameters[coefficient])
            row[f"mean_estimate_{label}"] = float(
                np.mean(estimates[:, coefficient])
            )
            row[f"uncertainty_{label}"] = float(
                parameter_uncertainty[coefficient]
            )
            row[f"pull_mean_{label}"] = _mean_or_nan(finite)
            row[f"pull_std_{label}"] = _std_or_nan(finite)
            row[f"coverage_1sigma_{label}"] = _mean_or_nan(
                np.abs(finite) <= 1.0
            )
            row[f"coverage_2sigma_{label}"] = _mean_or_nan(
                np.abs(finite) <= 2.0
            )
        cell_rows.append(row)

    summary: dict[str, float | int] = {
        "harmonic_common_cells": len(cell_rows)
    }
    for coefficient, label in enumerate(("A", "B", "C")):
        combined = (
            np.concatenate(coefficient_pulls[coefficient])
            if coefficient_pulls[coefficient]
            else np.empty(0, dtype=float)
        )
        summary[f"harmonic_{label}_pull_mean"] = _mean_or_nan(combined)
        summary[f"harmonic_{label}_pull_std"] = _std_or_nan(combined)
        summary[f"harmonic_{label}_pull_absolute_gt2"] = int(
            np.count_nonzero(np.abs(combined) > 2.0)
        )
    return summary, cell_rows


def recommend_iterations(
    metrics: Iterable[dict[str, float | int | str]],
    iteration_values: Iterable[int],
) -> int:
    """Choose the iteration count with minimum median held-out normalized MSE."""
    rows = tuple(metrics)
    candidates = []
    for iteration in iteration_values:
        values = np.asarray(
            [
                float(row["normalized_mse"])
                for row in rows
                if int(row["iterations"]) == int(iteration)
                and np.isfinite(float(row["normalized_mse"]))
            ],
            dtype=float,
        )
        score = float(np.median(values)) if values.size else np.inf
        candidates.append((score, int(iteration)))
    return min(candidates)[1]


def assess_iteration_coverage(
    metrics: Iterable[dict[str, float | int | str]],
    iteration_values: Iterable[int],
) -> list[dict[str, object]]:
    """Apply an explicit, conservative coverage gate to each iteration.

    The gate is deliberately separate from the minimum-MSE recommendation:
    regularization can minimize mean-squared error while still producing
    confidence intervals that are too narrow. Medians are taken across the
    independent fold/stress rows before applying the documented tolerances.
    """
    rows = tuple(metrics)
    limits = {
        "absolute_pull_mean": (0.0, 0.25),
        "pull_std": (0.80, 1.20),
        "coverage_1sigma": (0.63, 0.73),
        "coverage_2sigma": (0.93, 0.97),
        "absolute_harmonic_pull_mean": (0.0, 0.30),
        "harmonic_pull_std": (0.75, 1.25),
    }
    assessments: list[dict[str, object]] = []
    for iteration in iteration_values:
        selected = [
            row for row in rows if int(row["iterations"]) == int(iteration)
        ]
        medians: dict[str, float] = {}
        for field in ("pull_mean", "pull_std", "coverage_1sigma", "coverage_2sigma"):
            values = np.asarray([float(row[field]) for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            medians[field] = float(np.median(finite)) if finite.size else np.nan
        for label in ("A", "B", "C"):
            for suffix in ("pull_mean", "pull_std"):
                field = f"harmonic_{label}_{suffix}"
                values = np.asarray([float(row[field]) for row in selected], dtype=float)
                finite = values[np.isfinite(values)]
                medians[field] = float(np.median(finite)) if finite.size else np.nan

        checks = {
            "pull_mean": _inside(abs(medians["pull_mean"]), limits["absolute_pull_mean"]),
            "pull_std": _inside(medians["pull_std"], limits["pull_std"]),
            "coverage_1sigma": _inside(
                medians["coverage_1sigma"], limits["coverage_1sigma"]
            ),
            "coverage_2sigma": _inside(
                medians["coverage_2sigma"], limits["coverage_2sigma"]
            ),
        }
        for label in ("A", "B", "C"):
            checks[f"harmonic_{label}_pull_mean"] = _inside(
                abs(medians[f"harmonic_{label}_pull_mean"]),
                limits["absolute_harmonic_pull_mean"],
            )
            checks[f"harmonic_{label}_pull_std"] = _inside(
                medians[f"harmonic_{label}_pull_std"],
                limits["harmonic_pull_std"],
            )
        assessments.append(
            {
                "iterations": int(iteration),
                "certified": bool(checks) and all(checks.values()),
                "failed_checks": [name for name, passed in checks.items() if not passed],
                "medians": medians,
                "limits": limits,
            }
        )
    return assessments


def _inside(value: float, limits: tuple[float, float]) -> bool:
    return bool(np.isfinite(value) and limits[0] <= value <= limits[1])


def aggregate_metrics(
    metrics: Iterable[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    rows = tuple(metrics)
    numeric_fields = [
        key
        for key, value in rows[0].items()
        if key not in {"fold", "stress", "iterations"} and isinstance(value, (int, float))
    ] if rows else []
    output = []
    keys = sorted({(str(row["stress"]), int(row["iterations"])) for row in rows})
    for stress, iteration in keys:
        selected = [
            row for row in rows
            if str(row["stress"]) == stress and int(row["iterations"]) == iteration
        ]
        aggregate: dict[str, float | int | str] = {
            "stress": stress,
            "iterations": iteration,
            "folds": len(selected),
        }
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in selected], dtype=float)
            aggregate[field] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan
        output.append(aggregate)
    return output


def _harmonic_metrics(
    target: Array,
    unfolded: Array,
    uncertainty: Array,
    valid: Array,
    binning: AnalysisBinning,
    *,
    minimum_points: int,
    covariance_phi: Array | None = None,
    covariance_samples: int | None = None,
) -> tuple[
    dict[str, float | int],
    list[dict[str, float | int | str]],
]:
    shaped_target = binning.unflatten(target)
    shaped_unfolded = binning.unflatten(unfolded)
    shaped_uncertainty = binning.unflatten(uncertainty)
    shaped_valid = binning.unflatten(valid)
    kwargs = dict(
        validity_mask=shaped_valid,
        measurement_covariance_phi=covariance_phi,
        measurement_covariance_samples=covariance_samples,
        minimum_points=max(4, min(minimum_points, binning.shape[-1])),
        require_nonnegative=False,
    )
    unfolded_fit = fit_grid(
        shaped_unfolded,
        shaped_uncertainty,
        binning.phi_edges,
        **kwargs,
    )
    truth_fit = fit_grid(
        shaped_target,
        shaped_uncertainty,
        binning.phi_edges,
        **kwargs,
    )
    common = unfolded_fit["fit_success"] & truth_fit["fit_success"]
    output: dict[str, float | int] = {
        "harmonic_common_cells": int(np.count_nonzero(common))
    }
    cell_rows: list[dict[str, float | int | str]] = []
    for index in zip(*np.nonzero(common), strict=True):
        row: dict[str, float | int | str] = _cell_coordinates(binning, index)
        row["phi_points"] = int(unfolded_fit["points"][index])
        for coefficient, label in enumerate(("A", "B", "C")):
            target_value = float(truth_fit["parameters"][index + (coefficient,)])
            estimate = float(unfolded_fit["parameters"][index + (coefficient,)])
            sigma_value = float(
                unfolded_fit["parameter_uncertainties"][index + (coefficient,)]
            )
            row[f"target_{label}"] = target_value
            row[f"estimate_{label}"] = estimate
            row[f"uncertainty_{label}"] = sigma_value
            row[f"pull_{label}"] = (
                (estimate - target_value) / sigma_value
                if np.isfinite(sigma_value) and sigma_value > 0.0
                else np.nan
            )
        cell_rows.append(row)
    for coefficient, label in enumerate(("A", "B", "C")):
        delta = (
            unfolded_fit["parameters"][..., coefficient]
            - truth_fit["parameters"][..., coefficient]
        )
        sigma = unfolded_fit["parameter_uncertainties"][..., coefficient]
        coefficient_valid = common & np.isfinite(delta) & np.isfinite(sigma) & (sigma > 0.0)
        pulls = np.divide(
            delta,
            sigma,
            out=np.full_like(delta, np.nan),
            where=coefficient_valid,
        )[coefficient_valid]
        output[f"harmonic_{label}_pull_mean"] = _mean_or_nan(pulls)
        output[f"harmonic_{label}_pull_std"] = _std_or_nan(pulls)
        output[f"harmonic_{label}_pull_absolute_gt2"] = int(
            np.count_nonzero(np.abs(pulls) > 2.0)
        )
    return output, cell_rows


def _cell_coordinates(
    binning: AnalysisBinning, index: tuple[int, int, int]
) -> dict[str, float | int | str]:
    q2, xb, minus_t = (int(value) for value in index)
    return {
        "q2_bin": q2,
        "xb_bin": xb,
        "t_bin": minus_t,
        "q2_low": float(binning.q2_edges[q2]),
        "q2_high": float(binning.q2_edges[q2 + 1]),
        "xb_low": float(binning.xb_edges[xb]),
        "xb_high": float(binning.xb_edges[xb + 1]),
        "t_low": float(binning.t_edges[minus_t]),
        "t_high": float(binning.t_edges[minus_t + 1]),
    }


def _scaled_coordinate(values: Array, limits: tuple[float, float] | None) -> Array:
    values = np.asarray(values, dtype=float)
    if limits is None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros_like(values)
        lower, upper = float(np.min(finite)), float(np.max(finite))
    else:
        lower, upper = map(float, limits)
    if not upper > lower:
        return np.zeros_like(values)
    scaled = 2.0 * (values - lower) / (upper - lower) - 1.0
    return np.clip(np.nan_to_num(scaled, nan=0.0, posinf=1.5, neginf=-1.5), -1.5, 1.5)


def _mean_or_nan(values: Array) -> float:
    values = np.asarray(values)
    return float(np.mean(values)) if values.size else np.nan


def _std_or_nan(values: Array) -> float:
    values = np.asarray(values)
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def _median_or_nan(values: Array) -> float:
    values = np.asarray(values)
    return float(np.median(values)) if values.size else np.nan


def _quantile_or_nan(values: Array, quantile: float) -> float:
    values = np.asarray(values)
    return float(np.quantile(values, quantile)) if values.size else np.nan
