from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray

GROUPING = "proton-detector+ft-photon-count-v1"
ESTIMATOR = "mode-seeded-iterative-gaussian-core-v1"
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
    "rec_pT_miss": 0.0,
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


@dataclass(frozen=True)
class CoreEstimate:
    center: float
    sigma: float
    entries: int
    core_entries: int
    iterations: int


@dataclass(frozen=True)
class ExclusivityCuts:
    variables: tuple[str, ...]
    group_ids: Array
    lower: Array
    upper: Array
    centers: Array
    sigmas: Array
    fit_entries: Array
    core_entries: Array
    iterations: Array
    window_source: Array
    global_mode: bool
    n_sigma: float
    minimum_events: int
    core_clip_sigma: float
    core_max_iterations: int
    core_convergence: float
    core_histogram_bins: int
    minimum_core_fraction: float
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
    core_clip_sigma: float = 2.0,
    core_max_iterations: int = 12,
    core_convergence: float = 1.0e-4,
    core_histogram_bins: int = 100,
    minimum_core_fraction: float = 0.2,
) -> ExclusivityCuts:
    """Derive sequential signal-core windows with topology-pooled fallbacks.

    A local window is estimated independently in each kinematic group. When a
    local group is sparse or its core fit is unstable, it inherits a window
    pooled only over the same proton detector and FT-photon multiplicity. A
    group is retained only when every sequential variable has a finite window.
    """
    _validate_settings(
        n_sigma,
        minimum_events,
        core_clip_sigma,
        core_max_iterations,
        core_convergence,
        core_histogram_bins,
        minimum_core_fraction,
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
    core_entries = np.zeros(shape, dtype=np.int64)
    iterations = np.zeros(shape, dtype=np.int64)
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
                core_clip_sigma,
                core_max_iterations,
                core_convergence,
                core_histogram_bins,
                minimum_core_fraction,
                expected_center=_EXPECTED_CENTERS.get(name),
            )

        for group_index in np.flatnonzero(active):
            rows = group_rows[group_index]
            local = _estimate_core(
                arrays[name][rows[running[group_index]]],
                minimum_events,
                core_clip_sigma,
                core_max_iterations,
                core_convergence,
                core_histogram_bins,
                minimum_core_fraction,
                expected_center=_EXPECTED_CENTERS.get(name),
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
            if name in _PHYSICAL_LOWER_BOUNDS:
                lo = max(lo, _PHYSICAL_LOWER_BOUNDS[name])
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                active[group_index] = False
                continue

            lower[group_index, variable_index] = lo
            upper[group_index, variable_index] = hi
            centers[group_index, variable_index] = estimate.center
            sigmas[group_index, variable_index] = estimate.sigma
            fit_entries[group_index, variable_index] = estimate.entries
            core_entries[group_index, variable_index] = estimate.core_entries
            iterations[group_index, variable_index] = estimate.iterations
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
        core_entries=core_entries[retained],
        iterations=iterations[retained],
        window_source=window_source[retained],
        global_mode=global_mode,
        n_sigma=n_sigma,
        minimum_events=minimum_events,
        core_clip_sigma=core_clip_sigma,
        core_max_iterations=core_max_iterations,
        core_convergence=core_convergence,
        core_histogram_bins=core_histogram_bins,
        minimum_core_fraction=minimum_core_fraction,
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
        # Robust cut tables are complete by construction. Refuse malformed
        # tables rather than silently disabling a sequential variable.
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
    core_clip_sigma: float = 2.0,
    core_max_iterations: int = 12,
    core_convergence: float = 1.0e-4,
    core_histogram_bins: int = 100,
    minimum_core_fraction: float = 0.2,
) -> tuple[float, float] | None:
    """Return a symmetric n-sigma window around a robust Gaussian core."""
    _validate_settings(
        n_sigma,
        minimum_events,
        core_clip_sigma,
        core_max_iterations,
        core_convergence,
        core_histogram_bins,
        minimum_core_fraction,
    )
    estimate = _estimate_core(
        values,
        minimum_events,
        core_clip_sigma,
        core_max_iterations,
        core_convergence,
        core_histogram_bins,
        minimum_core_fraction,
        expected_center,
    )
    if estimate is None:
        return None
    return (
        float(estimate.center - n_sigma * estimate.sigma),
        float(estimate.center + n_sigma * estimate.sigma),
    )


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
        core_entries=cuts.core_entries,
        iterations=cuts.iterations,
        window_source=cuts.window_source,
        global_mode=cuts.global_mode,
        n_sigma=cuts.n_sigma,
        minimum_events=cuts.minimum_events,
        core_clip_sigma=cuts.core_clip_sigma,
        core_max_iterations=cuts.core_max_iterations,
        core_convergence=cuts.core_convergence,
        core_histogram_bins=cuts.core_histogram_bins,
        minimum_core_fraction=cuts.minimum_core_fraction,
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
            "exclusivity cut table predates the robust Gaussian-core estimator; re-derive it"
        )
    estimator = str(np.asarray(saved["estimator"]).item())
    if estimator != ESTIMATOR:
        raise ValueError(f"unsupported exclusivity estimator: {estimator}")
    cuts = ExclusivityCuts(
        variables=tuple(str(item) for item in saved["variables"]),
        group_ids=saved["group_ids"],
        lower=saved["lower"],
        upper=saved["upper"],
        centers=saved["centers"],
        sigmas=saved["sigmas"],
        fit_entries=saved["fit_entries"],
        core_entries=saved["core_entries"],
        iterations=saved["iterations"],
        window_source=saved["window_source"],
        global_mode=bool(saved["global_mode"]),
        n_sigma=float(saved["n_sigma"]),
        minimum_events=int(saved["minimum_events"]),
        core_clip_sigma=float(saved["core_clip_sigma"]),
        core_max_iterations=int(saved["core_max_iterations"]),
        core_convergence=float(saved["core_convergence"]),
        core_histogram_bins=int(saved["core_histogram_bins"]),
        minimum_core_fraction=float(saved["minimum_core_fraction"]),
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
        cuts.core_entries,
        cuts.iterations,
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
    clip_sigma: float,
    max_iterations: int,
    convergence: float,
    histogram_bins: int,
    minimum_core_fraction: float,
    expected_center: float | None,
) -> CoreEstimate | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    entries = int(finite.size)
    if entries < minimum_events:
        return None

    seed = _mode_seed(finite, histogram_bins, expected_center)
    if seed is None:
        return None
    center, sigma = seed
    sigma_correction = _truncated_normal_sigma_correction(clip_sigma)
    minimum_core_events = max(12, int(math.ceil(0.5 * minimum_events)))
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        core = np.abs(finite - center) <= clip_sigma * sigma
        if np.count_nonzero(core) < minimum_core_events:
            return None
        next_center = float(np.mean(finite[core]))
        clipped_sigma = float(np.std(finite[core], ddof=1))
        next_sigma = clipped_sigma * sigma_correction
        if not np.isfinite(next_sigma) or next_sigma <= np.finfo(float).eps:
            return None
        center_change = abs(next_center - center) / next_sigma
        sigma_change = abs(next_sigma - sigma) / next_sigma
        center, sigma = next_center, next_sigma
        if max(center_change, sigma_change) <= convergence:
            break

    core = np.abs(finite - center) <= clip_sigma * sigma
    core_count = int(np.count_nonzero(core))
    if core_count < minimum_core_events or core_count / entries < minimum_core_fraction:
        return None
    if expected_center is not None and abs(center - expected_center) > 3.0 * sigma:
        return None
    return CoreEstimate(center, sigma, entries, core_count, iterations)


def _mode_seed(
    values: Array,
    histogram_bins: int,
    expected_center: float | None,
) -> tuple[float, float] | None:
    lo, hi = np.quantile(values, [0.005, 0.995])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return None
    bins = min(histogram_bins, max(24, 2 * int(np.sqrt(values.size))))
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    if not np.any(counts):
        return None
    smoothed = np.convolve(
        counts.astype(float), [1.0, 2.0, 3.0, 2.0, 1.0], mode="same"
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = int(np.argmax(smoothed))
    if expected_center is not None:
        candidates = np.flatnonzero(smoothed >= 0.5 * smoothed[peak])
        if candidates.size:
            peak = int(candidates[np.argmin(np.abs(centers[candidates] - expected_center))])

    baseline = float(np.median(smoothed))
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


def _truncated_normal_sigma_correction(clip_sigma: float) -> float:
    normalization = math.erf(clip_sigma / math.sqrt(2.0))
    phi = math.exp(-0.5 * clip_sigma * clip_sigma) / math.sqrt(2.0 * math.pi)
    variance = 1.0 - 2.0 * clip_sigma * phi / normalization
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("core clip sigma gives an invalid truncated-normal variance")
    return 1.0 / math.sqrt(variance)


def _validate_settings(
    n_sigma: float,
    minimum_events: int,
    core_clip_sigma: float,
    core_max_iterations: int,
    core_convergence: float,
    core_histogram_bins: int,
    minimum_core_fraction: float,
) -> None:
    if not np.isfinite(n_sigma) or n_sigma <= 0:
        raise ValueError("n_sigma must be positive")
    if minimum_events < 12:
        raise ValueError("minimum_events must be at least 12")
    if not np.isfinite(core_clip_sigma) or core_clip_sigma <= 1.0:
        raise ValueError("core_clip_sigma must be greater than 1")
    if core_max_iterations < 1:
        raise ValueError("core_max_iterations must be positive")
    if not np.isfinite(core_convergence) or core_convergence <= 0:
        raise ValueError("core_convergence must be positive")
    if core_histogram_bins < 10:
        raise ValueError("core_histogram_bins must be at least 10")
    if not np.isfinite(minimum_core_fraction) or not 0 < minimum_core_fraction <= 1:
        raise ValueError("minimum_core_fraction must be in (0, 1]")


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
    # Pairing is collision-free for the supported nonnegative bin indices.
    return (((topology * _BIN_RADIX) + iq2) * _BIN_RADIX + ixb) * _BIN_RADIX + it
