from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

from .exclusivity_models import FitEstimate, estimate_model, maximum_fit_parameters


Array = np.ndarray

GROUPING = "proton-detector+ft-photon-count-v1"
ESTIMATOR = "topology-variable-signal-models-v5"
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
    fit_parameter_names: Array
    fit_parameter_values: Array
    window_source: Array
    populated_group_ids: Array
    dropped_group_ids: Array
    dropped_variables: Array
    dropped_reasons: Array
    global_mode: bool
    n_sigma: float
    signal_containment: float
    minimum_events: int
    fit_window_sigma: float
    fit_max_iterations: int
    fit_convergence: float
    fit_histogram_bins: int
    minimum_signal_fraction: float
    minimum_peak_significance: float
    maximum_local_sigma_ratio: float
    maximum_local_center_shift_sigma: float
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
    global_mode: bool = True,
    fit_window_sigma: float = 5.0,
    fit_max_iterations: int = 100,
    fit_convergence: float = 1.0e-5,
    fit_histogram_bins: int = 160,
    minimum_signal_fraction: float = 0.1,
    minimum_peak_significance: float = 3.0,
    maximum_local_sigma_ratio: float = 2.0,
    maximum_local_center_shift_sigma: float = 2.5,
) -> ExclusivityCuts:
    """Derive sequential signal-plus-background windows with topology fallbacks.

    A variable-specific signal and smooth background are fitted in each
    selected group. Global mode pools kinematic bins within detector topology.
    In per-bin mode, a sparse or insignificant local fit inherits the fit
    pooled over the same proton detector and FT-photon multiplicity. Every
    retained group has a finite signal-containment window for every variable.
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
        maximum_local_sigma_ratio,
        maximum_local_center_shift_sigma,
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
    fit_model = np.full(shape, "", dtype="<U64")
    parameter_shape = (*shape, maximum_fit_parameters())
    fit_parameter_names = np.full(parameter_shape, "", dtype="<U32")
    fit_parameter_values = np.full(parameter_shape, np.nan)
    window_source = np.full(shape, "", dtype="<U32")
    dropped_variables = np.full(populated.size, "", dtype="<U64")
    dropped_reasons = np.full(populated.size, "", dtype="<U256")

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
        pooled: dict[int, tuple[FitEstimate | None, str]] = {}
        for topology in np.unique(group_topologies[active]):
            pieces = [
                arrays[name][group_rows[index][running[index]]]
                for index in np.flatnonzero(active & (group_topologies == topology))
            ]
            pooled_values = np.concatenate(pieces) if pieces else np.empty(0)
            pooled[int(topology)] = estimate_model(
                pooled_values,
                name,
                n_sigma,
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
            local, local_reason = estimate_model(
                arrays[name][rows[running[group_index]]],
                name,
                n_sigma,
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
            reference, reference_reason = pooled.get(
                int(group_topologies[group_index]),
                (None, "topology reference was not constructed"),
            )
            estimate = local
            source = "local"
            if local is not None and reference is not None and not _locally_consistent(
                local,
                reference,
                maximum_local_sigma_ratio,
                maximum_local_center_shift_sigma,
            ):
                estimate = reference
                source = "topology_consistency_fallback"
            elif estimate is None:
                estimate = reference
                source = "topology_fallback"
            if estimate is None:
                active[group_index] = False
                dropped_variables[group_index] = name
                dropped_reasons[group_index] = (
                    f"local fit: {local_reason}; topology fit: {reference_reason}"
                )
                continue

            lo = estimate.lower
            hi = estimate.upper
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                active[group_index] = False
                dropped_variables[group_index] = name
                dropped_reasons[group_index] = "derived window was non-finite or empty"
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
            parameter_count = len(estimate.parameter_names)
            fit_parameter_names[
                group_index, variable_index, :parameter_count
            ] = estimate.parameter_names
            fit_parameter_values[
                group_index, variable_index, :parameter_count
            ] = estimate.parameter_values
            window_source[group_index, variable_index] = source
            raw = arrays[name][rows]
            running[group_index] &= np.isfinite(raw) & (raw >= lo) & (raw <= hi)

    complete = np.all(np.isfinite(lower) & np.isfinite(upper), axis=1)
    retained = active & complete
    unexplained = ~retained & (dropped_reasons == "")
    for group_index in np.flatnonzero(unexplained):
        missing = np.flatnonzero(
            ~(np.isfinite(lower[group_index]) & np.isfinite(upper[group_index]))
        )
        if missing.size:
            dropped_variables[group_index] = variables[int(missing[0])]
        dropped_reasons[group_index] = "cut table was incomplete"
    dropped = ~retained
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
        fit_parameter_names=fit_parameter_names[retained],
        fit_parameter_values=fit_parameter_values[retained],
        window_source=window_source[retained],
        populated_group_ids=populated.astype(np.int64, copy=False),
        dropped_group_ids=populated[dropped].astype(np.int64, copy=False),
        dropped_variables=dropped_variables[dropped],
        dropped_reasons=dropped_reasons[dropped],
        global_mode=global_mode,
        n_sigma=n_sigma,
        signal_containment=float(erf(n_sigma / sqrt(2.0))),
        minimum_events=minimum_events,
        fit_window_sigma=fit_window_sigma,
        fit_max_iterations=fit_max_iterations,
        fit_convergence=fit_convergence,
        fit_histogram_bins=fit_histogram_bins,
        minimum_signal_fraction=minimum_signal_fraction,
        minimum_peak_significance=minimum_peak_significance,
        maximum_local_sigma_ratio=maximum_local_sigma_ratio,
        maximum_local_center_shift_sigma=maximum_local_center_shift_sigma,
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
    maximum_local_sigma_ratio: float = 2.0,
    maximum_local_center_shift_sigma: float = 2.5,
) -> tuple[float, float] | None:
    """Return a Gaussian-equivalent containment window for a peaked variable."""
    _validate_settings(
        n_sigma,
        minimum_events,
        fit_window_sigma,
        fit_max_iterations,
        fit_convergence,
        fit_histogram_bins,
        minimum_signal_fraction,
        minimum_peak_significance,
        maximum_local_sigma_ratio,
        maximum_local_center_shift_sigma,
    )
    estimate, _ = estimate_model(
        values,
        "rec_m2_epX",
        n_sigma,
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
    return estimate.lower, estimate.upper


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
        fit_parameter_names=cuts.fit_parameter_names,
        fit_parameter_values=cuts.fit_parameter_values,
        window_source=cuts.window_source,
        populated_group_ids=cuts.populated_group_ids,
        dropped_group_ids=cuts.dropped_group_ids,
        dropped_variables=cuts.dropped_variables,
        dropped_reasons=cuts.dropped_reasons,
        global_mode=cuts.global_mode,
        n_sigma=cuts.n_sigma,
        signal_containment=cuts.signal_containment,
        minimum_events=cuts.minimum_events,
        fit_window_sigma=cuts.fit_window_sigma,
        fit_max_iterations=cuts.fit_max_iterations,
        fit_convergence=cuts.fit_convergence,
        fit_histogram_bins=cuts.fit_histogram_bins,
        minimum_signal_fraction=cuts.minimum_signal_fraction,
        minimum_peak_significance=cuts.minimum_peak_significance,
        maximum_local_sigma_ratio=cuts.maximum_local_sigma_ratio,
        maximum_local_center_shift_sigma=cuts.maximum_local_center_shift_sigma,
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
        fit_parameter_names=saved["fit_parameter_names"],
        fit_parameter_values=saved["fit_parameter_values"],
        window_source=saved["window_source"],
        populated_group_ids=(
            saved["populated_group_ids"]
            if "populated_group_ids" in saved.files
            else saved["group_ids"]
        ),
        dropped_group_ids=(
            saved["dropped_group_ids"]
            if "dropped_group_ids" in saved.files
            else np.empty(0, dtype=np.int64)
        ),
        dropped_variables=(
            saved["dropped_variables"]
            if "dropped_variables" in saved.files
            else np.empty(0, dtype="<U1")
        ),
        dropped_reasons=(
            saved["dropped_reasons"]
            if "dropped_reasons" in saved.files
            else np.empty(0, dtype="<U1")
        ),
        global_mode=bool(saved["global_mode"]),
        n_sigma=float(saved["n_sigma"]),
        signal_containment=float(saved["signal_containment"]),
        minimum_events=int(saved["minimum_events"]),
        fit_window_sigma=float(saved["fit_window_sigma"]),
        fit_max_iterations=int(saved["fit_max_iterations"]),
        fit_convergence=float(saved["fit_convergence"]),
        fit_histogram_bins=int(saved["fit_histogram_bins"]),
        minimum_signal_fraction=float(saved["minimum_signal_fraction"]),
        minimum_peak_significance=float(saved["minimum_peak_significance"]),
        maximum_local_sigma_ratio=float(saved["maximum_local_sigma_ratio"]),
        maximum_local_center_shift_sigma=float(
            saved["maximum_local_center_shift_sigma"]
        ),
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
    parameter_shape = (*expected_shape, maximum_fit_parameters())
    if cuts.fit_parameter_names.shape != parameter_shape:
        raise ValueError("exclusivity cut table has inconsistent fit parameters")
    if cuts.fit_parameter_values.shape != parameter_shape:
        raise ValueError("exclusivity cut table has inconsistent fit parameters")
    if cuts.dropped_group_ids.shape != cuts.dropped_variables.shape:
        raise ValueError("exclusivity cut table has inconsistent dropped-group diagnostics")
    if cuts.dropped_group_ids.shape != cuts.dropped_reasons.shape:
        raise ValueError("exclusivity cut table has inconsistent dropped-group diagnostics")
    return cuts


def _locally_consistent(
    local: FitEstimate,
    reference: FitEstimate,
    maximum_sigma_ratio: float,
    maximum_center_shift_sigma: float,
) -> bool:
    if local.sigma > maximum_sigma_ratio * reference.sigma:
        return False
    center_shift = abs(local.center - reference.center)
    return center_shift <= maximum_center_shift_sigma * reference.sigma


def _validate_settings(
    n_sigma: float,
    minimum_events: int,
    fit_window_sigma: float,
    fit_max_iterations: int,
    fit_convergence: float,
    fit_histogram_bins: int,
    minimum_signal_fraction: float,
    minimum_peak_significance: float,
    maximum_local_sigma_ratio: float,
    maximum_local_center_shift_sigma: float,
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
    if not np.isfinite(maximum_local_sigma_ratio) or maximum_local_sigma_ratio < 1:
        raise ValueError("maximum_local_sigma_ratio must be at least 1")
    if (
        not np.isfinite(maximum_local_center_shift_sigma)
        or maximum_local_center_shift_sigma <= 0
    ):
        raise ValueError("maximum_local_center_shift_sigma must be positive")


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


def topology_ids_from_groups(group_ids: Array, global_mode: bool) -> Array:
    """Recover detector-topology IDs from global or kinematically binned groups."""
    groups = np.asarray(group_ids, dtype=np.int64)
    if global_mode:
        return groups
    return groups // (_BIN_RADIX**3)
