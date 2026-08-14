from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray

GROUPING = "proton-detector+ft-photon-count-v1"
ESTIMATOR = "binned-gaussian-core-tail-background-mixture-v3"
_TOPOLOGY_RADIX = 4
_BIN_RADIX = 128

DEFAULT_VARIABLES = (
    "rec_m_gg",
    "rec_pT_miss",
    "rec_m2_epX",
    "rec_m_eggX",
    "rec_E_miss",
    "rec_m2_miss",
)

_PI0_MASS_GEV = 0.1349768
_PROTON_MASS_GEV = 0.9382721
_EXPECTED_CENTERS = {
    "rec_m_gg": _PI0_MASS_GEV,
    "rec_m2_epX": _PI0_MASS_GEV**2,
    "rec_m_eggX": _PROTON_MASS_GEV,
    "rec_E_miss": 0.0,
    "rec_m2_miss": 0.0,
}
_PHYSICAL_LOWER_BOUNDS = {
    "rec_m_gg": 0.0,
    "rec_pT_miss": 0.0,
    "rec_m_eggX": 0.0,
}
_MAXIMUM_CENTER_DEVIATIONS = {
    "rec_m_gg": 0.08,
    "rec_m2_epX": 0.20,
    "rec_m_eggX": 0.40,
    "rec_E_miss": 1.0,
    "rec_m2_miss": 1.0,
}
_MAXIMUM_SIGMAS = {
    "rec_m_gg": 0.08,
    "rec_pT_miss": 0.50,
    "rec_m2_epX": 0.20,
    "rec_m_eggX": 0.30,
    "rec_E_miss": 1.0,
    "rec_m2_miss": 1.0,
}
_SIGMA_SCALES = np.asarray((0.40, 0.55, 0.70, 0.85, 1.0, 1.2, 1.45))
_CENTER_OFFSETS = np.asarray((-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0))


@dataclass(frozen=True)
class CoreEstimate:
    center: float
    sigma: float
    entries: int
    fit_entries: int
    signal_entries: float
    signal_fraction: float
    peak_significance: float
    iterations: int
    fit_model: str


@dataclass(frozen=True)
class ExclusivityCuts:
    variables: tuple[str, ...]
    group_ids: Array
    lower: Array
    upper: Array
    centers: Array
    sigmas: Array
    fit_entries: Array
    signal_entries: Array
    signal_fractions: Array
    peak_significance: Array
    iterations: Array
    fit_model: Array
    window_source: Array
    global_mode: bool
    n_sigma: float
    minimum_events: int
    fit_window_sigma: float
    fit_max_iterations: int
    fit_convergence: float
    fit_histogram_bins: int
    minimum_signal_fraction: float
    minimum_peak_significance: float
    grouping: str
    estimator: str


def derive_cuts(
    values: dict[str, Array],
    proton_detector: Array,
    ft_photons: Array,
    iq2: Array,
    ixb: Array,
    it: Array,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    topologies: tuple[int, ...] = (1, 2),
    photon_topologies: tuple[int, ...] = (0, 1, 2),
    n_sigma: float = 3.0,
    minimum_events: int = 50,
    global_mode: bool = False,
    fit_window_sigma: float = 5.0,
    fit_max_iterations: int = 100,
    fit_convergence: float = 1.0e-5,
    fit_histogram_bins: int = 160,
    minimum_signal_fraction: float = 0.1,
    minimum_peak_significance: float = 3.0,
) -> ExclusivityCuts:
    """Derive sequential signal-plus-background windows with topology fallbacks.

    A bounded Gaussian signal and nonnegative linear background are fitted in
    each local kinematic group. If that fit is sparse or insignificant, the
    group inherits the fit pooled over the same proton detector and FT-photon
    multiplicity. Every retained group has a finite window for every variable.
    """
    _validate_settings(
        n_sigma,
        minimum_events,
        fit_window_sigma,
        fit_max_iterations,
        fit_convergence,
        fit_histogram_bins,
        minimum_signal_fraction,
        minimum_peak_significance,
    )
    detector = np.asarray(proton_detector, dtype=np.int64)
    ft_photons = np.asarray(ft_photons, dtype=np.int64)
    iq2, ixb, it = [np.asarray(item, dtype=np.int64) for item in (iq2, ixb, it)]
    event_count = detector.size
    if not (ft_photons.size == iq2.size == ixb.size == it.size == event_count):
        raise ValueError("topology and bin-index arrays must have equal lengths")
    arrays = {name: np.asarray(values[name], dtype=float) for name in variables}
    if any(array.size != event_count for array in arrays.values()):
        raise ValueError("exclusivity arrays must match event count")

    in_range = (iq2 >= 0) & (ixb >= 0) & (it >= 0)
    in_range &= np.isin(detector, topologies)
    in_range &= np.isin(ft_photons, photon_topologies)
    groups = _group_ids(detector, ft_photons, iq2, ixb, it, global_mode)
    event_topologies = detector * _TOPOLOGY_RADIX + ft_photons
    populated = np.unique(groups[in_range])
    shape = (populated.size, len(variables))
    lower = np.full(shape, np.nan)
    upper = np.full(shape, np.nan)
    centers = np.full(shape, np.nan)
    sigmas = np.full(shape, np.nan)
    fit_entries = np.zeros(shape, dtype=np.int64)
    signal_entries = np.zeros(shape, dtype=float)
    signal_fractions = np.zeros(shape, dtype=float)
    peak_significance = np.zeros(shape, dtype=float)
    iterations = np.zeros(shape, dtype=np.int64)
    fit_model = np.full(shape, "", dtype="<U24")
    window_source = np.full(shape, "", dtype="<U24")

    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    left = np.searchsorted(sorted_groups, populated, side="left")
    right = np.searchsorted(sorted_groups, populated, side="right")
    group_rows = [
        rows[in_range[rows]]
        for start, stop in zip(left, right, strict=True)
        for rows in (order[start:stop],)
    ]
    group_topologies = np.asarray(
        [event_topologies[rows[0]] for rows in group_rows], dtype=np.int64
    )
    running = [np.ones(rows.size, dtype=bool) for rows in group_rows]
    active = np.ones(populated.size, dtype=bool)

    for variable_index, name in enumerate(variables):
        expected_center = _EXPECTED_CENTERS.get(name)
        physical_lower = _PHYSICAL_LOWER_BOUNDS.get(name)
        maximum_center_deviation = _MAXIMUM_CENTER_DEVIATIONS.get(name)
        maximum_sigma = _MAXIMUM_SIGMAS.get(name)
        pooled: dict[int, CoreEstimate | None] = {}
        for topology in np.unique(group_topologies[active]):
            pieces = [
                arrays[name][group_rows[index][running[index]]]
                for index in np.flatnonzero(active & (group_topologies == topology))
            ]
            pooled_values = np.concatenate(pieces) if pieces else np.empty(0)
            pooled[int(topology)] = _estimate_core(
                pooled_values,
                minimum_events,
                fit_window_sigma,
                fit_max_iterations,
                fit_convergence,
                fit_histogram_bins,
                minimum_signal_fraction,
                minimum_peak_significance,
                expected_center,
                physical_lower,
                maximum_center_deviation,
                maximum_sigma,
            )

        for group_index in np.flatnonzero(active):
            rows = group_rows[group_index]
            local = _estimate_core(
                arrays[name][rows[running[group_index]]],
                minimum_events,
                fit_window_sigma,
                fit_max_iterations,
                fit_convergence,
                fit_histogram_bins,
                minimum_signal_fraction,
                minimum_peak_significance,
                expected_center,
                physical_lower,
                maximum_center_deviation,
                maximum_sigma,
            )
            estimate = local
            source = "local"
            if estimate is None:
                estimate = pooled.get(int(group_topologies[group_index]))
                source = "topology_fallback"
            if estimate is None:
                active[group_index] = False
                continue

            lo = estimate.center - n_sigma * estimate.sigma
            hi = estimate.center + n_sigma * estimate.sigma
            if physical_lower is not None:
                lo = max(lo, physical_lower)
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                active[group_index] = False
                continue

            lower[group_index, variable_index] = lo
            upper[group_index, variable_index] = hi
            centers[group_index, variable_index] = estimate.center
            sigmas[group_index, variable_index] = estimate.sigma
            fit_entries[group_index, variable_index] = estimate.fit_entries
            signal_entries[group_index, variable_index] = estimate.signal_entries
            signal_fractions[group_index, variable_index] = estimate.signal_fraction
            peak_significance[group_index, variable_index] = estimate.peak_significance
            iterations[group_index, variable_index] = estimate.iterations
            fit_model[group_index, variable_index] = estimate.fit_model
            window_source[group_index, variable_index] = source
            raw = arrays[name][rows]
            running[group_index] &= np.isfinite(raw) & (raw >= lo) & (raw <= hi)

    retained = active & np.all(np.isfinite(lower) & np.isfinite(upper), axis=1)
    return ExclusivityCuts(
        variables=variables,
        group_ids=populated[retained].astype(np.int64, copy=False),
        lower=lower[retained],
        upper=upper[retained],
        centers=centers[retained],
        sigmas=sigmas[retained],
        fit_entries=fit_entries[retained],
        signal_entries=signal_entries[retained],
        signal_fractions=signal_fractions[retained],
        peak_significance=peak_significance[retained],
        iterations=iterations[retained],
        fit_model=fit_model[retained],
        window_source=window_source[retained],
        global_mode=global_mode,
        n_sigma=n_sigma,
        minimum_events=minimum_events,
        fit_window_sigma=fit_window_sigma,
        fit_max_iterations=fit_max_iterations,
        fit_convergence=fit_convergence,
        fit_histogram_bins=fit_histogram_bins,
        minimum_signal_fraction=minimum_signal_fraction,
        minimum_peak_significance=minimum_peak_significance,
        grouping=GROUPING,
        estimator=ESTIMATOR,
    )


def apply_cuts(
    cuts: ExclusivityCuts,
    values: dict[str, Array],
    proton_detector: Array,
    ft_photons: Array,
    iq2: Array,
    ixb: Array,
    it: Array,
) -> Array:
    if cuts.grouping != GROUPING:
        raise ValueError(f"unsupported exclusivity grouping: {cuts.grouping}")
    if cuts.estimator != ESTIMATOR:
        raise ValueError(f"unsupported exclusivity estimator: {cuts.estimator}")
    detector = np.asarray(proton_detector, dtype=np.int64)
    ft_photons = np.asarray(ft_photons, dtype=np.int64)
    groups = _group_ids(detector, ft_photons, iq2, ixb, it, cuts.global_mode)
    mask = np.zeros(detector.size, dtype=bool)
    if cuts.group_ids.size == 0:
        return mask
    positions = np.searchsorted(cuts.group_ids, groups)
    bounded = positions < cuts.group_ids.size
    known = np.zeros(detector.size, dtype=bool)
    known[bounded] = cuts.group_ids[positions[bounded]] == groups[bounded]
    rows = np.flatnonzero(known)
    mask[rows] = True
    for variable_index, name in enumerate(cuts.variables):
        raw = np.asarray(values[name], dtype=float)
        lo = cuts.lower[positions[rows], variable_index]
        hi = cuts.upper[positions[rows], variable_index]
        if not np.all(np.isfinite(lo) & np.isfinite(hi)):
            raise ValueError("exclusivity cut table contains inactive windows")
        mask[rows] &= np.isfinite(raw[rows]) & (raw[rows] >= lo) & (raw[rows] <= hi)
    return mask


def estimate_window(
    values: Array,
    n_sigma: float,
    minimum_events: int,
    *,
    expected_center: float | None = None,
    physical_lower: float | None = None,
    maximum_center_deviation: float | None = None,
    maximum_sigma: float | None = None,
    fit_window_sigma: float = 5.0,
    fit_max_iterations: int = 100,
    fit_convergence: float = 1.0e-5,
    fit_histogram_bins: int = 160,
    minimum_signal_fraction: float = 0.1,
    minimum_peak_significance: float = 3.0,
) -> tuple[float, float] | None:
    """Return an n-sigma window from a Gaussian-plus-linear-background fit."""
    _validate_settings(
        n_sigma,
        minimum_events,
        fit_window_sigma,
        fit_max_iterations,
        fit_convergence,
        fit_histogram_bins,
        minimum_signal_fraction,
        minimum_peak_significance,
    )
    estimate = _estimate_core(
        values,
        minimum_events,
        fit_window_sigma,
        fit_max_iterations,
        fit_convergence,
        fit_histogram_bins,
        minimum_signal_fraction,
        minimum_peak_significance,
        expected_center,
        physical_lower,
        maximum_center_deviation,
        maximum_sigma,
    )
    if estimate is None:
        return None
    lo = estimate.center - n_sigma * estimate.sigma
    if physical_lower is not None:
        lo = max(lo, physical_lower)
    return float(lo), float(estimate.center + n_sigma * estimate.sigma)


def save_cuts(path: str, cuts: ExclusivityCuts) -> None:
    np.savez_compressed(
        path,
        variables=np.asarray(cuts.variables),
        group_ids=cuts.group_ids,
        lower=cuts.lower,
        upper=cuts.upper,
        centers=cuts.centers,
        sigmas=cuts.sigmas,
        fit_entries=cuts.fit_entries,
        signal_entries=cuts.signal_entries,
        signal_fractions=cuts.signal_fractions,
        peak_significance=cuts.peak_significance,
        iterations=cuts.iterations,
        fit_model=cuts.fit_model,
        window_source=cuts.window_source,
        global_mode=cuts.global_mode,
        n_sigma=cuts.n_sigma,
        minimum_events=cuts.minimum_events,
        fit_window_sigma=cuts.fit_window_sigma,
        fit_max_iterations=cuts.fit_max_iterations,
        fit_convergence=cuts.fit_convergence,
        fit_histogram_bins=cuts.fit_histogram_bins,
        minimum_signal_fraction=cuts.minimum_signal_fraction,
        minimum_peak_significance=cuts.minimum_peak_significance,
        grouping=cuts.grouping,
        estimator=cuts.estimator,
    )


def load_cuts(path: str) -> ExclusivityCuts:
    saved = np.load(path, allow_pickle=False)
    if "grouping" not in saved.files:
        raise ValueError(
            "exclusivity cut table predates FT-photon topology grouping; re-derive it"
        )
    grouping = str(np.asarray(saved["grouping"]).item())
    if grouping != GROUPING:
        raise ValueError(f"unsupported exclusivity grouping: {grouping}")
    if "estimator" not in saved.files:
        raise ValueError(
            "exclusivity cut table predates signal-background fitting; re-derive it"
        )
    estimator = str(np.asarray(saved["estimator"]).item())
    if estimator != ESTIMATOR:
        raise ValueError(
            f"unsupported exclusivity estimator: {estimator}; re-derive the cut table"
        )
    cuts = ExclusivityCuts(
        variables=tuple(str(item) for item in saved["variables"]),
        group_ids=saved["group_ids"],
        lower=saved["lower"],
        upper=saved["upper"],
        centers=saved["centers"],
        sigmas=saved["sigmas"],
        fit_entries=saved["fit_entries"],
        signal_entries=saved["signal_entries"],
        signal_fractions=saved["signal_fractions"],
        peak_significance=saved["peak_significance"],
        iterations=saved["iterations"],
        fit_model=saved["fit_model"],
        window_source=saved["window_source"],
        global_mode=bool(saved["global_mode"]),
        n_sigma=float(saved["n_sigma"]),
        minimum_events=int(saved["minimum_events"]),
        fit_window_sigma=float(saved["fit_window_sigma"]),
        fit_max_iterations=int(saved["fit_max_iterations"]),
        fit_convergence=float(saved["fit_convergence"]),
        fit_histogram_bins=int(saved["fit_histogram_bins"]),
        minimum_signal_fraction=float(saved["minimum_signal_fraction"]),
        minimum_peak_significance=float(saved["minimum_peak_significance"]),
        grouping=grouping,
        estimator=estimator,
    )
    expected_shape = (cuts.group_ids.size, len(cuts.variables))
    diagnostic_arrays = (
        cuts.lower,
        cuts.upper,
        cuts.centers,
        cuts.sigmas,
        cuts.fit_entries,
        cuts.signal_entries,
        cuts.signal_fractions,
        cuts.peak_significance,
        cuts.iterations,
        cuts.fit_model,
        cuts.window_source,
    )
    if any(array.shape != expected_shape for array in diagnostic_arrays):
        raise ValueError("exclusivity cut table has inconsistent array shapes")
    if not np.all(np.isfinite(cuts.lower) & np.isfinite(cuts.upper)):
        raise ValueError("exclusivity cut table contains inactive windows")
    return cuts


def _estimate_core(
    values: Array,
    minimum_events: int,
    fit_window_sigma: float,
    max_iterations: int,
    convergence: float,
    histogram_bins: int,
    minimum_signal_fraction: float,
    minimum_peak_significance: float,
    expected_center: float | None,
    physical_lower: float | None,
    maximum_center_deviation: float | None,
    maximum_sigma: float | None,
) -> CoreEstimate | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    entries = int(finite.size)
    if entries < minimum_events:
        return None

    seed = _mode_seed(finite, histogram_bins, expected_center)
    if seed is None:
        return None
    seed_center, seed_sigma = seed
    qlo, qhi = np.quantile(finite, [0.005, 0.995])
    fit_lo = max(float(qlo), seed_center - fit_window_sigma * seed_sigma)
    fit_hi = min(float(qhi), seed_center + fit_window_sigma * seed_sigma)
    if physical_lower is not None:
        fit_lo = max(fit_lo, physical_lower)
    if not (np.isfinite(fit_lo) and np.isfinite(fit_hi) and fit_hi > fit_lo):
        return None
    selected = finite[(finite >= fit_lo) & (finite <= fit_hi)]
    minimum_fit_entries = max(20, int(np.ceil(0.5 * minimum_events)))
    if selected.size < minimum_fit_entries:
        return None

    bins = min(histogram_bins, max(20, 2 * int(np.sqrt(selected.size))))
    counts, edges = np.histogram(selected, bins=bins, range=(fit_lo, fit_hi))
    x = 0.5 * (edges[:-1] + edges[1:])
    span = fit_hi - fit_lo
    u = np.clip((x - fit_lo) / span, 0.0, 1.0)
    background_left = 2.0 * (1.0 - u)
    background_right = 2.0 * u
    background_left /= np.sum(background_left)
    background_right /= np.sum(background_right)

    best: tuple[float, float, float, Array, int, bool] | None = None
    for sigma_scale in _SIGMA_SCALES:
        sigma = float(seed_sigma * sigma_scale)
        if sigma <= 0 or sigma >= 0.5 * span:
            continue
        for center_offset in _CENTER_OFFSETS:
            center = float(seed_center + center_offset * seed_sigma)
            if not fit_lo < center < fit_hi:
                continue
            signal = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            signal_sum = float(np.sum(signal))
            if not np.isfinite(signal_sum) or signal_sum <= 0:
                continue
            signal /= signal_sum
            tail_sigma = max(2.5 * sigma, 1.75 * seed_sigma)
            tail = np.exp(-0.5 * ((x - center) / tail_sigma) ** 2)
            tail /= np.sum(tail)
            for use_tail, components in (
                (False, np.vstack((signal, background_left, background_right))),
                (
                    True,
                    np.vstack((signal, tail, background_left, background_right)),
                ),
            ):
                weights, iterations = _mixture_weights(
                    counts, components, max_iterations, convergence
                )
                density = np.maximum(weights @ components, np.finfo(float).tiny)
                nll = float(-np.sum(counts * np.log(density)))
                free_weights = components.shape[0] - 1
                bic = 2.0 * nll + free_weights * np.log(selected.size)
                if best is None or bic < best[0]:
                    best = (bic, center, sigma, weights, iterations, use_tail)

    if best is None:
        return None
    _, center, sigma, weights, iterations, use_tail = best
    signal_fraction = float(weights[0])
    signal_entries = signal_fraction * selected.size
    core = np.abs(x - center) <= 2.0 * sigma
    signal_shape = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    signal_shape /= np.sum(signal_shape)
    tail_sigma = max(2.5 * sigma, 1.75 * seed_sigma)
    tail_shape = np.exp(-0.5 * ((x - center) / tail_sigma) ** 2)
    tail_shape /= np.sum(tail_shape)
    signal_core = signal_entries * float(np.sum(signal_shape[core]))
    if use_tail:
        background_core = selected.size * float(
            weights[1] * np.sum(tail_shape[core])
            + weights[2] * np.sum(background_left[core])
            + weights[3] * np.sum(background_right[core])
        )
    else:
        background_core = selected.size * float(
            weights[1] * np.sum(background_left[core])
            + weights[2] * np.sum(background_right[core])
        )
    significance = signal_core / np.sqrt(max(signal_core + background_core, 1.0))
    bin_width = float(edges[1] - edges[0])
    if signal_fraction < minimum_signal_fraction:
        return None
    if significance < minimum_peak_significance:
        return None
    if maximum_sigma is not None and sigma > maximum_sigma:
        return None
    if expected_center is not None and maximum_center_deviation is not None:
        tolerance = max(maximum_center_deviation, 2.0 * bin_width)
        if abs(center - expected_center) > tolerance:
            return None
    return CoreEstimate(
        center=center,
        sigma=sigma,
        entries=entries,
        fit_entries=int(selected.size),
        signal_entries=float(signal_entries),
        signal_fraction=signal_fraction,
        peak_significance=float(significance),
        iterations=iterations,
        fit_model="core+tail+linear" if use_tail else "core+linear",
    )


def _mixture_weights(
    counts: Array,
    components: Array,
    max_iterations: int,
    convergence: float,
) -> tuple[Array, int]:
    weights = np.full(components.shape[0], 1.0 / components.shape[0])
    total = float(np.sum(counts))
    for iteration in range(1, max_iterations + 1):
        density = np.maximum(weights @ components, np.finfo(float).tiny)
        responsibilities = components * weights[:, None] / density[None, :]
        next_weights = (responsibilities @ counts) / total
        if np.max(np.abs(next_weights - weights)) <= convergence:
            return next_weights, iteration
        weights = next_weights
    return weights, max_iterations


def _mode_seed(
    values: Array,
    histogram_bins: int,
    expected_center: float | None,
) -> tuple[float, float] | None:
    lo, hi = np.quantile(values, [0.005, 0.995])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return None
    bins = min(histogram_bins, max(16, int(np.sqrt(values.size))))
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    if not np.any(counts):
        return None
    smoothed = np.convolve(counts.astype(float), [1.0, 2.0, 1.0], mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    baseline = float(np.median(smoothed))
    global_peak = int(np.argmax(smoothed))
    local_maximum = np.ones(smoothed.size, dtype=bool)
    local_maximum[1:] &= smoothed[1:] >= smoothed[:-1]
    local_maximum[:-1] &= smoothed[:-1] >= smoothed[1:]
    threshold = baseline + 0.35 * max(0.0, smoothed[global_peak] - baseline)
    candidates = np.flatnonzero(local_maximum & (smoothed >= threshold))
    peak = global_peak
    if expected_center is not None and candidates.size:
        peak = int(candidates[np.argmin(np.abs(centers[candidates] - expected_center))])

    half_height = baseline + 0.5 * max(0.0, smoothed[peak] - baseline)
    left = peak
    while left > 0 and smoothed[left] >= half_height:
        left -= 1
    right = peak
    while right < smoothed.size - 1 and smoothed[right] >= half_height:
        right += 1
    bin_width = float(edges[1] - edges[0])
    sigma = max(float(centers[right] - centers[left]) / 2.355, bin_width)
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    center = float(centers[peak])
    near_peak = values[np.abs(values - center) <= 1.5 * sigma]
    if near_peak.size:
        center = float(np.median(near_peak))
    return center, sigma


def _validate_settings(
    n_sigma: float,
    minimum_events: int,
    fit_window_sigma: float,
    fit_max_iterations: int,
    fit_convergence: float,
    fit_histogram_bins: int,
    minimum_signal_fraction: float,
    minimum_peak_significance: float,
) -> None:
    if not np.isfinite(n_sigma) or n_sigma <= 0:
        raise ValueError("n_sigma must be positive")
    if minimum_events < 12:
        raise ValueError("minimum_events must be at least 12")
    if not np.isfinite(fit_window_sigma) or fit_window_sigma <= 2.0:
        raise ValueError("fit_window_sigma must be greater than 2")
    if fit_max_iterations < 1:
        raise ValueError("fit_max_iterations must be positive")
    if not np.isfinite(fit_convergence) or fit_convergence <= 0:
        raise ValueError("fit_convergence must be positive")
    if fit_histogram_bins < 20:
        raise ValueError("fit_histogram_bins must be at least 20")
    if not np.isfinite(minimum_signal_fraction) or not 0 < minimum_signal_fraction <= 1:
        raise ValueError("minimum_signal_fraction must be in (0, 1]")
    if not np.isfinite(minimum_peak_significance) or minimum_peak_significance <= 0:
        raise ValueError("minimum_peak_significance must be positive")


def _group_ids(
    detector: Array,
    ft_photons: Array,
    iq2: Array,
    ixb: Array,
    it: Array,
    global_mode: bool,
) -> Array:
    detector = np.asarray(detector, dtype=np.int64)
    ft_photons = np.asarray(ft_photons, dtype=np.int64)
    topology = detector * _TOPOLOGY_RADIX + ft_photons
    if global_mode:
        return topology
    iq2, ixb, it = [np.asarray(item, dtype=np.int64) for item in (iq2, ixb, it)]
    return (((topology * _BIN_RADIX) + iq2) * _BIN_RADIX + ixb) * _BIN_RADIX + it
