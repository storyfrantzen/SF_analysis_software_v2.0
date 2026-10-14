from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from .exclusivity import ExclusivityCuts, apply_cuts, event_group_ids
from .exclusivity_models import FitEstimate, estimate_model


Array = np.ndarray

METHOD = "topology-mgg-nminus1-sideband-v1"
_MASS_VARIABLE = "rec_m_gg"
_PI0_MASS_GEV = 0.1349768


@dataclass(frozen=True)
class BackgroundSubtraction:
    signal_region_mask: Array
    sideband_mask: Array
    net_event_weights: Array
    signal_region: Array
    signal_region_unweighted: Array
    signal_region_variance: Array
    sideband: Array
    sideband_unweighted: Array
    sideband_variance: Array
    estimated_background: Array
    estimated_background_variance: Array
    background_subtracted: Array
    background_subtracted_variance: Array
    group_ids: Array
    signal_lower: Array
    signal_upper: Array
    fit_lower: Array
    fit_upper: Array
    alpha: Array
    alpha_uncertainty: Array
    fit_model: Array
    fit_entries: Array
    fit_bic: Array
    fit_deviance: Array
    fit_ndof: Array
    signal_region_by_group: Array
    sideband_by_group: Array
    sideband_variance_by_group: Array
    estimated_background_by_group: Array
    estimated_background_variance_by_group: Array


def estimate_mgg_background(
    *,
    cuts: ExclusivityCuts,
    values: dict[str, Array],
    proton_detector: Array,
    ft_photons: Array,
    iq2: Array,
    ixb: Array,
    it: Array,
    rec_flat: Array,
    base_mask: Array,
    event_weights: Array,
    number_of_bins: int,
    alpha_bootstrap: int = 200,
    seed: int | None = None,
) -> BackgroundSubtraction:
    """Estimate and subtract the nonpeaking m_gg background before unfolding.

    Each retained detector/photon topology is refitted in the m_gg N-1 sample.
    The nominal m_gg window from ``cuts`` defines the signal region; the rest of
    the fitted mass domain defines the sidebands.  The fitted linear-background
    shape supplies the signal-to-sideband transfer factor ``alpha``.  Observed
    sideband events are then transferred independently in every reconstructed
    analysis bin.
    """
    if not cuts.global_mode:
        raise ValueError(
            "background subtraction currently requires global-by-topology "
            "exclusivity cuts"
        )
    if _MASS_VARIABLE not in cuts.variables:
        raise ValueError("exclusivity table does not contain rec_m_gg")
    if alpha_bootstrap < 0:
        raise ValueError("alpha_bootstrap must be nonnegative")

    rec_flat = np.asarray(rec_flat, dtype=np.int64)
    base = np.asarray(base_mask, dtype=bool)
    weights = np.asarray(event_weights, dtype=float)
    event_count = rec_flat.size
    if base.shape != (event_count,) or weights.shape != (event_count,):
        raise ValueError("background-subtraction arrays must match event count")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("background-subtraction weights must be finite and nonnegative")
    if any(np.asarray(values[name]).shape != (event_count,) for name in cuts.variables):
        raise ValueError("exclusivity arrays must match event count")

    nminus1 = base & apply_cuts(
        cuts,
        values,
        proton_detector,
        ft_photons,
        iq2,
        ixb,
        it,
        exclude_variables=(_MASS_VARIABLE,),
    )
    groups = event_group_ids(
        proton_detector, ft_photons, iq2, ixb, it, cuts.global_mode
    )
    masses = np.asarray(values[_MASS_VARIABLE], dtype=float)
    variable_index = cuts.variables.index(_MASS_VARIABLE)
    signal_mask = np.zeros(event_count, dtype=bool)
    sideband_mask = np.zeros(event_count, dtype=bool)

    group_count = cuts.group_ids.size
    signal_lower = np.empty(group_count, dtype=float)
    signal_upper = np.empty(group_count, dtype=float)
    fit_lower = np.empty(group_count, dtype=float)
    fit_upper = np.empty(group_count, dtype=float)
    alpha = np.empty(group_count, dtype=float)
    alpha_uncertainty = np.empty(group_count, dtype=float)
    fit_model = np.empty(group_count, dtype="<U64")
    fit_entries = np.empty(group_count, dtype=np.int64)
    fit_bic = np.empty(group_count, dtype=float)
    fit_deviance = np.empty(group_count, dtype=float)
    fit_ndof = np.empty(group_count, dtype=np.int64)

    rng = np.random.default_rng(seed)
    for row, group_id in enumerate(cuts.group_ids):
        group_mask = nminus1 & (groups == group_id) & np.isfinite(masses)
        estimate, reason = _fit_mass(cuts, masses[group_mask], variable_index)
        if estimate is None:
            raise ValueError(
                f"rec_m_gg N-1 background fit failed for group {int(group_id)}: "
                f"{reason}"
            )
        lo = float(cuts.lower[row, variable_index])
        hi = float(cuts.upper[row, variable_index])
        if lo < estimate.fit_lower or hi > estimate.fit_upper:
            raise ValueError(
                f"nominal rec_m_gg window [{lo:.6g}, {hi:.6g}] lies outside "
                f"the N-1 fit domain [{estimate.fit_lower:.6g}, "
                f"{estimate.fit_upper:.6g}] for group {int(group_id)}"
            )
        transfer = _background_transfer(estimate, lo, hi)
        uncertainty = _bootstrap_transfer_uncertainty(
            estimate, lo, hi, alpha_bootstrap, rng
        )
        in_group_fit = group_mask & (masses >= estimate.fit_lower) & (
            masses <= estimate.fit_upper
        )
        in_signal = in_group_fit & (masses >= lo) & (masses <= hi)
        in_sideband = in_group_fit & ~in_signal
        signal_mask |= in_signal
        sideband_mask |= in_sideband

        signal_lower[row] = lo
        signal_upper[row] = hi
        fit_lower[row] = estimate.fit_lower
        fit_upper[row] = estimate.fit_upper
        alpha[row] = transfer
        alpha_uncertainty[row] = uncertainty
        fit_model[row] = estimate.fit_model
        fit_entries[row] = estimate.fit_entries
        fit_bic[row] = estimate.bic
        fit_deviance[row] = estimate.deviance
        fit_ndof[row] = estimate.fit_ndof

    in_range = (rec_flat >= 0) & (rec_flat < number_of_bins)
    signal_mask &= in_range
    sideband_mask &= in_range
    signal_region = _histogram(rec_flat, signal_mask, weights, number_of_bins)
    signal_region_unweighted = _histogram(
        rec_flat, signal_mask, np.ones(event_count), number_of_bins
    )
    signal_region_variance = _histogram(
        rec_flat, signal_mask, weights * weights, number_of_bins
    )
    sideband = _histogram(rec_flat, sideband_mask, weights, number_of_bins)
    sideband_unweighted = _histogram(
        rec_flat, sideband_mask, np.ones(event_count), number_of_bins
    )
    sideband_variance = _histogram(
        rec_flat, sideband_mask, weights * weights, number_of_bins
    )

    estimated_background = np.zeros(number_of_bins, dtype=float)
    estimated_background_variance = np.zeros(number_of_bins, dtype=float)
    signal_region_by_group = np.zeros((group_count, number_of_bins), dtype=float)
    sideband_by_group = np.zeros((group_count, number_of_bins), dtype=float)
    sideband_variance_by_group = np.zeros(
        (group_count, number_of_bins), dtype=float
    )
    estimated_background_by_group = np.zeros(
        (group_count, number_of_bins), dtype=float
    )
    estimated_background_variance_by_group = np.zeros(
        (group_count, number_of_bins), dtype=float
    )
    net_event_weights = np.zeros(event_count, dtype=float)
    net_event_weights[signal_mask] = weights[signal_mask]
    for row, group_id in enumerate(cuts.group_ids):
        group_signal = signal_mask & (groups == group_id)
        group_sideband = sideband_mask & (groups == group_id)
        signal_region_by_group[row] = _histogram(
            rec_flat, group_signal, weights, number_of_bins
        )
        group_sideband_yield = _histogram(
            rec_flat, group_sideband, weights, number_of_bins
        )
        group_sideband_variance = _histogram(
            rec_flat, group_sideband, weights * weights, number_of_bins
        )
        group_background = alpha[row] * group_sideband_yield
        group_background_variance = (
            alpha[row] ** 2 * group_sideband_variance
            + alpha_uncertainty[row] ** 2 * group_sideband_yield**2
        )
        sideband_by_group[row] = group_sideband_yield
        sideband_variance_by_group[row] = group_sideband_variance
        estimated_background_by_group[row] = group_background
        estimated_background_variance_by_group[row] = group_background_variance
        estimated_background += group_background
        estimated_background_variance += group_background_variance
        net_event_weights[group_sideband] = -alpha[row] * weights[group_sideband]

    background_subtracted = signal_region - estimated_background
    background_subtracted_variance = (
        signal_region_variance + estimated_background_variance
    )
    return BackgroundSubtraction(
        signal_region_mask=signal_mask,
        sideband_mask=sideband_mask,
        net_event_weights=net_event_weights,
        signal_region=signal_region,
        signal_region_unweighted=signal_region_unweighted,
        signal_region_variance=signal_region_variance,
        sideband=sideband,
        sideband_unweighted=sideband_unweighted,
        sideband_variance=sideband_variance,
        estimated_background=estimated_background,
        estimated_background_variance=estimated_background_variance,
        background_subtracted=background_subtracted,
        background_subtracted_variance=background_subtracted_variance,
        group_ids=cuts.group_ids.copy(),
        signal_lower=signal_lower,
        signal_upper=signal_upper,
        fit_lower=fit_lower,
        fit_upper=fit_upper,
        alpha=alpha,
        alpha_uncertainty=alpha_uncertainty,
        fit_model=fit_model,
        fit_entries=fit_entries,
        fit_bic=fit_bic,
        fit_deviance=fit_deviance,
        fit_ndof=fit_ndof,
        signal_region_by_group=signal_region_by_group,
        sideband_by_group=sideband_by_group,
        sideband_variance_by_group=sideband_variance_by_group,
        estimated_background_by_group=estimated_background_by_group,
        estimated_background_variance_by_group=(
            estimated_background_variance_by_group
        ),
    )


def _fit_mass(
    cuts: ExclusivityCuts, values: Array, variable_index: int
) -> tuple[FitEstimate | None, str]:
    return estimate_model(
        values,
        _MASS_VARIABLE,
        cuts.n_sigma,
        cuts.minimum_events,
        cuts.fit_window_sigma,
        cuts.fit_max_iterations,
        cuts.fit_convergence,
        cuts.fit_histogram_bins,
        cuts.minimum_signal_fraction,
        cuts.minimum_peak_significance,
        _PI0_MASS_GEV,
        0.0,
        0.08,
        0.08,
        cut_containment=float(cuts.cut_containments[variable_index]),
        cut_component=str(cuts.cut_components[variable_index]),
        continuous_refinement=cuts.continuous_refinement,
    )


def _background_transfer(estimate: FitEstimate, lo: float, hi: float) -> float:
    edges = estimate.histogram_edges
    signal_overlap = _overlap_fraction(edges, lo, hi)
    background = np.asarray(estimate.background_counts, dtype=float)
    signal = float(np.sum(background * signal_overlap))
    sideband = float(np.sum(background * (1.0 - signal_overlap)))
    if not np.isfinite(signal) or not np.isfinite(sideband) or sideband <= 0.0:
        raise ValueError("fitted rec_m_gg background has no sideband support")
    return signal / sideband


def _bootstrap_transfer_uncertainty(
    estimate: FitEstimate,
    lo: float,
    hi: float,
    experiments: int,
    rng: np.random.Generator,
) -> float:
    if experiments <= 1:
        return 0.0
    edges = estimate.histogram_edges
    x = 0.5 * (edges[:-1] + edges[1:])
    span = float(edges[-1] - edges[0])
    u = np.clip((x - edges[0]) / span, 0.0, 1.0)
    left = 2.0 * (1.0 - u)
    right = 2.0 * u
    left /= np.sum(left)
    right /= np.sum(right)
    design = np.column_stack((left, right))
    fit_sideband = np.abs(x - estimate.center) >= 2.5 * estimate.sigma
    signal_overlap = _overlap_fraction(edges, lo, hi)
    means = np.maximum(np.asarray(estimate.observed_counts, dtype=float), 0.0)
    transfers = []
    for _ in range(experiments):
        fluctuated = rng.poisson(means).astype(float)
        coefficients, _ = nnls(design[fit_sideband], fluctuated[fit_sideband])
        background = design @ coefficients
        signal = float(np.sum(background * signal_overlap))
        sideband = float(np.sum(background * (1.0 - signal_overlap)))
        if np.isfinite(signal) and np.isfinite(sideband) and sideband > 0.0:
            transfers.append(signal / sideband)
    if len(transfers) < max(2, (experiments + 1) // 2):
        raise ValueError("too few finite rec_m_gg transfer factors in bootstrap")
    return float(np.std(transfers, ddof=1))


def _overlap_fraction(edges: Array, lo: float, hi: float) -> Array:
    left = np.asarray(edges[:-1], dtype=float)
    right = np.asarray(edges[1:], dtype=float)
    overlap = np.maximum(0.0, np.minimum(right, hi) - np.maximum(left, lo))
    return np.divide(overlap, right - left, out=np.zeros_like(overlap), where=right > left)


def _histogram(flat: Array, mask: Array, weights: Array, size: int) -> Array:
    return np.bincount(flat[mask], weights=weights[mask], minlength=size).astype(float)
