from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

from .exclusivity_models import FitEstimate, estimate_model, maximum_fit_parameters


Array = np.ndarray

GROUPING = "proton-detector+ft-photon-count-v1"
ESTIMATOR = "topology-variable-signal-models-bootstrap-audit-v8"
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
_DEFAULT_CUT_COMPONENTS = {
    "rec_m_gg": "core",
    "rec_pT_miss": "signal",
    "rec_m2_epX": "core",
    "rec_m_eggX": "signal",
    "rec_E_miss": "core",
    "rec_m2_miss": "core",
}


@dataclass(frozen=True)
class ExclusivityCuts:
    variables: tuple[str, ...]
    group_ids: Array
    lower: Array
    upper: Array
    fit_lower: Array
    fit_upper: Array
    centers: Array
    sigmas: Array
    fit_entries: Array
    cut_entries: Array
    extrapolated_cut_entries: Array
    signal_entries: Array
    signal_fractions: Array
    cut_component_fractions: Array
    nuisance_fractions: Array
    background_fractions: Array
    peak_significance: Array
    iterations: Array
    fit_model: Array
    fit_parameter_names: Array
    fit_parameter_values: Array
    bic: Array
    delta_bic: Array
    pearson_chi2: Array
    deviance: Array
    fit_ndof: Array
    continuously_refined: Array
    histogram_bin_count: Array
    histogram_edges: Array
    observed_counts: Array
    expected_counts: Array
    cut_signal_counts: Array
    noncut_component_counts: Array
    background_counts: Array
    window_source: Array
    cumulative_before: Array
    cumulative_after: Array
    nminus1_entries: Array
    nminus1_passing: Array
    populated_group_ids: Array
    dropped_group_ids: Array
    dropped_variables: Array
    dropped_reasons: Array
    global_mode: bool
    n_sigma: float
    signal_containment: float
    cut_containments: Array
    cut_components: Array
    nminus1_audit_lower: Array
    nminus1_audit_upper: Array
    nminus1_audit_centers: Array
    nminus1_audit_sigmas: Array
    nminus1_audit_fit_entries: Array
    nminus1_audit_source: Array
    nminus1_audit_reasons: Array
    nminus1_audit_success: Array
    nminus1_audit_complete: bool
    nminus1_audit_within_tolerance: bool
    nminus1_audit_maximum_boundary_change: float
    nminus1_audit_boundary_tolerance: float
    continuous_refinement: bool
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
    cut_containments: dict[str, float] | None = None,
    cut_components: dict[str, str] | None = None,
    nminus1_audit_boundary_tolerance: float = 0.02,
    continuous_refinement: bool = True,
) -> ExclusivityCuts:
    """Derive deterministic topology cuts and audit them with fixed N-1 fits."""
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
    if not np.isfinite(nminus1_audit_boundary_tolerance) or not (
        0.0 < nminus1_audit_boundary_tolerance < 1.0
    ):
        raise ValueError(
            "nminus1_audit_boundary_tolerance must lie between zero and one"
        )

    base_containment = float(erf(n_sigma / sqrt(2.0)))
    containment_overrides = cut_containments or {}
    component_overrides = cut_components or {}
    variable_containments = np.asarray(
        [containment_overrides.get(name, base_containment) for name in variables],
        dtype=float,
    )
    variable_components = np.asarray(
        [
            component_overrides.get(
                name, _DEFAULT_CUT_COMPONENTS.get(name, "core")
            )
            for name in variables
        ]
    )
    if not np.all((variable_containments > 0.0) & (variable_containments < 1.0)):
        raise ValueError("every cut containment must lie between zero and one")
    if not np.all(np.isin(variable_components, ("core", "signal"))):
        raise ValueError("cut components must be either 'core' or 'signal'")

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
    fit_lower = np.full(shape, np.nan)
    fit_upper = np.full(shape, np.nan)
    centers = np.full(shape, np.nan)
    sigmas = np.full(shape, np.nan)
    fit_entries = np.zeros(shape, dtype=np.int64)
    cut_entries = np.zeros(shape, dtype=np.int64)
    extrapolated_cut_entries = np.zeros(shape, dtype=np.int64)
    signal_entries = np.zeros(shape, dtype=float)
    signal_fractions = np.zeros(shape, dtype=float)
    cut_component_fractions = np.zeros(shape, dtype=float)
    nuisance_fractions = np.zeros(shape, dtype=float)
    background_fractions = np.zeros(shape, dtype=float)
    peak_significance = np.zeros(shape, dtype=float)
    iterations = np.zeros(shape, dtype=np.int64)
    fit_model = np.full(shape, "", dtype="<U64")
    parameter_shape = (*shape, maximum_fit_parameters())
    fit_parameter_names = np.full(parameter_shape, "", dtype="<U32")
    fit_parameter_values = np.full(parameter_shape, np.nan)
    bic = np.full(shape, np.nan)
    delta_bic = np.full(shape, np.nan)
    pearson_chi2 = np.full(shape, np.nan)
    deviance = np.full(shape, np.nan)
    fit_ndof = np.zeros(shape, dtype=np.int64)
    continuously_refined = np.zeros(shape, dtype=bool)
    histogram_bin_count = np.zeros(shape, dtype=np.int64)
    histogram_edges = np.full((*shape, fit_histogram_bins + 1), np.nan)
    observed_counts = np.full((*shape, fit_histogram_bins), np.nan)
    expected_counts = np.full((*shape, fit_histogram_bins), np.nan)
    cut_signal_counts = np.full((*shape, fit_histogram_bins), np.nan)
    noncut_component_counts = np.full((*shape, fit_histogram_bins), np.nan)
    background_counts = np.full((*shape, fit_histogram_bins), np.nan)
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
    active = np.ones(populated.size, dtype=bool)

    def fit_stage(
        variable_index: int,
        selections: list[Array],
        stage_active: Array,
    ) -> list[tuple[FitEstimate | None, str, str]]:
        name = variables[variable_index]
        expected_center = _EXPECTED_CENTERS.get(name)
        physical_lower = _PHYSICAL_LOWER_BOUNDS.get(name)
        maximum_center_deviation = _MAXIMUM_CENTER_DEVIATIONS.get(name)
        maximum_sigma = _MAXIMUM_SIGMAS.get(name)
        pooled: dict[int, tuple[FitEstimate | None, str]] = {}
        for topology in np.unique(group_topologies[stage_active]):
            pieces = [
                arrays[name][group_rows[index][selections[index]]]
                for index in np.flatnonzero(
                    stage_active & (group_topologies == topology)
                )
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
                cut_containment=float(variable_containments[variable_index]),
                cut_component=str(variable_components[variable_index]),
                continuous_refinement=continuous_refinement,
            )

        results: list[tuple[FitEstimate | None, str, str]] = [
            (None, "inactive group", "") for _ in range(populated.size)
        ]
        for group_index in np.flatnonzero(stage_active):
            rows = group_rows[group_index]
            reference, reference_reason = pooled.get(
                int(group_topologies[group_index]),
                (None, "topology reference was not constructed"),
            )
            if global_mode:
                local, local_reason = reference, reference_reason
            else:
                local, local_reason = estimate_model(
                    arrays[name][rows[selections[group_index]]],
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
                    cut_containment=float(variable_containments[variable_index]),
                    cut_component=str(variable_components[variable_index]),
                    continuous_refinement=continuous_refinement,
                )
            estimate = local
            source = "global" if global_mode else "local"
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
                results[group_index] = (
                    None,
                    f"local fit: {local_reason}; topology fit: {reference_reason}",
                    source,
                )
                continue
            results[group_index] = (estimate, "", source)
        return results

    def store_estimate(
        group_index: int,
        variable_index: int,
        estimate: FitEstimate,
        source: str,
    ) -> None:
        lower[group_index, variable_index] = estimate.lower
        upper[group_index, variable_index] = estimate.upper
        fit_lower[group_index, variable_index] = estimate.fit_lower
        fit_upper[group_index, variable_index] = estimate.fit_upper
        centers[group_index, variable_index] = estimate.center
        sigmas[group_index, variable_index] = estimate.sigma
        fit_entries[group_index, variable_index] = estimate.fit_entries
        cut_entries[group_index, variable_index] = estimate.cut_entries
        extrapolated_cut_entries[
            group_index, variable_index
        ] = estimate.extrapolated_cut_entries
        signal_entries[group_index, variable_index] = estimate.signal_entries
        signal_fractions[group_index, variable_index] = estimate.signal_fraction
        cut_component_fractions[
            group_index, variable_index
        ] = estimate.cut_component_fraction
        nuisance_fractions[group_index, variable_index] = estimate.nuisance_fraction
        background_fractions[
            group_index, variable_index
        ] = estimate.background_fraction
        peak_significance[group_index, variable_index] = estimate.peak_significance
        iterations[group_index, variable_index] = estimate.iterations
        fit_model[group_index, variable_index] = estimate.fit_model
        fit_parameter_names[group_index, variable_index] = ""
        fit_parameter_values[group_index, variable_index] = np.nan
        parameter_count = len(estimate.parameter_names)
        fit_parameter_names[
            group_index, variable_index, :parameter_count
        ] = estimate.parameter_names
        fit_parameter_values[
            group_index, variable_index, :parameter_count
        ] = estimate.parameter_values
        bic[group_index, variable_index] = estimate.bic
        delta_bic[group_index, variable_index] = estimate.delta_bic
        pearson_chi2[group_index, variable_index] = estimate.pearson_chi2
        deviance[group_index, variable_index] = estimate.deviance
        fit_ndof[group_index, variable_index] = estimate.fit_ndof
        continuously_refined[
            group_index, variable_index
        ] = estimate.continuously_refined
        count = estimate.observed_counts.size
        histogram_bin_count[group_index, variable_index] = count
        histogram_edges[group_index, variable_index] = np.nan
        observed_counts[group_index, variable_index] = np.nan
        expected_counts[group_index, variable_index] = np.nan
        cut_signal_counts[group_index, variable_index] = np.nan
        noncut_component_counts[group_index, variable_index] = np.nan
        background_counts[group_index, variable_index] = np.nan
        histogram_edges[
            group_index, variable_index, : count + 1
        ] = estimate.histogram_edges
        observed_counts[
            group_index, variable_index, :count
        ] = estimate.observed_counts
        expected_counts[
            group_index, variable_index, :count
        ] = estimate.expected_counts
        cut_signal_counts[
            group_index, variable_index, :count
        ] = estimate.cut_signal_counts
        noncut_component_counts[
            group_index, variable_index, :count
        ] = estimate.noncut_component_counts
        background_counts[
            group_index, variable_index, :count
        ] = estimate.background_counts
        window_source[group_index, variable_index] = source

    # Bootstrap in one fixed physics order so that combinatorial backgrounds in
    # the later missing quantities do not dominate their initial fits. The
    # ordering is based on variable identity, not the caller's tuple order.
    initialization_rank = {
        name: index for index, name in enumerate(DEFAULT_VARIABLES)
    }
    initialization_order = sorted(
        range(len(variables)),
        key=lambda index: (initialization_rank.get(variables[index], 1000), index),
    )
    running = [np.ones(rows.size, dtype=bool) for rows in group_rows]
    for variable_index in initialization_order:
        name = variables[variable_index]
        stage_active = active.copy()
        results = fit_stage(variable_index, running, stage_active)
        for group_index in np.flatnonzero(stage_active):
            estimate, reason, source = results[group_index]
            if estimate is None:
                active[group_index] = False
                dropped_variables[group_index] = name
                dropped_reasons[group_index] = f"bootstrap fit: {reason}"
                continue
            store_estimate(group_index, variable_index, estimate, source)
            rows = group_rows[group_index]
            raw = arrays[name][rows]
            running[group_index] &= (
                np.isfinite(raw)
                & (raw >= estimate.lower)
                & (raw <= estimate.upper)
            )

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

    # Audit every immutable bootstrap window once on the sample passing the
    # other five immutable windows. These fits intentionally do not update the
    # cut table and cannot drop a group: they measure sensitivity to the N-1
    # conditioning without creating a feedback loop between correlated tails.
    nminus1_audit_lower = np.full(shape, np.nan)
    nminus1_audit_upper = np.full(shape, np.nan)
    nminus1_audit_centers = np.full(shape, np.nan)
    nminus1_audit_sigmas = np.full(shape, np.nan)
    nminus1_audit_fit_entries = np.zeros(shape, dtype=np.int64)
    nminus1_audit_source = np.full(shape, "", dtype="<U32")
    nminus1_audit_reasons = np.full(shape, "", dtype="<U256")
    nminus1_audit_success = np.zeros(shape, dtype=bool)
    audit_pending: list[list[tuple[FitEstimate | None, str, str]]] = []
    for variable_index, _ in enumerate(variables):
        selections = []
        for group_index, rows in enumerate(group_rows):
            selected_rows = np.ones(rows.size, dtype=bool)
            if retained[group_index]:
                for other_index, other_name in enumerate(variables):
                    if other_index == variable_index:
                        continue
                    raw = arrays[other_name][rows]
                    selected_rows &= (
                        np.isfinite(raw)
                        & (raw >= lower[group_index, other_index])
                        & (raw <= upper[group_index, other_index])
                    )
            selections.append(selected_rows)
        audit_pending.append(fit_stage(variable_index, selections, retained))

    audit_changes = []
    for group_index in np.flatnonzero(retained):
        for variable_index, _ in enumerate(variables):
            estimate, reason, source = audit_pending[variable_index][group_index]
            nminus1_audit_source[group_index, variable_index] = source
            if estimate is None:
                nminus1_audit_reasons[group_index, variable_index] = reason
                continue
            nminus1_audit_success[group_index, variable_index] = True
            nminus1_audit_lower[group_index, variable_index] = estimate.lower
            nminus1_audit_upper[group_index, variable_index] = estimate.upper
            nminus1_audit_centers[group_index, variable_index] = estimate.center
            nminus1_audit_sigmas[group_index, variable_index] = estimate.sigma
            nminus1_audit_fit_entries[
                group_index, variable_index
            ] = estimate.fit_entries
            nominal_width = (
                upper[group_index, variable_index]
                - lower[group_index, variable_index]
            )
            audit_width = estimate.upper - estimate.lower
            denominator = max(
                abs(nominal_width), abs(audit_width), np.finfo(float).eps
            )
            audit_changes.extend(
                (
                    abs(estimate.lower - lower[group_index, variable_index])
                    / denominator,
                    abs(estimate.upper - upper[group_index, variable_index])
                    / denominator,
                )
            )
    retained_audit_success = nminus1_audit_success[retained]
    nminus1_audit_complete = bool(
        retained_audit_success.size and np.all(retained_audit_success)
    )
    nminus1_audit_maximum_boundary_change = max(audit_changes, default=0.0)
    nminus1_audit_within_tolerance = bool(
        nminus1_audit_complete
        and nminus1_audit_maximum_boundary_change
        <= nminus1_audit_boundary_tolerance
    )

    cumulative_before = np.zeros(shape, dtype=np.int64)
    cumulative_after = np.zeros(shape, dtype=np.int64)
    nminus1_entries = np.zeros(shape, dtype=np.int64)
    nminus1_passing = np.zeros(shape, dtype=np.int64)
    for group_index in np.flatnonzero(retained):
        rows = group_rows[group_index]
        passes = []
        for variable_index, name in enumerate(variables):
            raw = arrays[name][rows]
            passes.append(
                np.isfinite(raw)
                & (raw >= lower[group_index, variable_index])
                & (raw <= upper[group_index, variable_index])
            )
        cumulative = np.ones(rows.size, dtype=bool)
        for variable_index, passed in enumerate(passes):
            cumulative_before[group_index, variable_index] = np.count_nonzero(
                cumulative
            )
            cumulative &= passed
            cumulative_after[group_index, variable_index] = np.count_nonzero(
                cumulative
            )
        all_passing = np.logical_and.reduce(passes)
        for variable_index, _ in enumerate(variables):
            other_passes = [
                passed
                for index, passed in enumerate(passes)
                if index != variable_index
            ]
            other_passing = (
                np.logical_and.reduce(other_passes)
                if other_passes
                else np.ones(rows.size, dtype=bool)
            )
            nminus1_entries[group_index, variable_index] = np.count_nonzero(
                other_passing
            )
            nminus1_passing[group_index, variable_index] = np.count_nonzero(
                all_passing
            )

    return ExclusivityCuts(
        variables=variables,
        group_ids=populated[retained].astype(np.int64, copy=False),
        lower=lower[retained],
        upper=upper[retained],
        fit_lower=fit_lower[retained],
        fit_upper=fit_upper[retained],
        centers=centers[retained],
        sigmas=sigmas[retained],
        fit_entries=fit_entries[retained],
        cut_entries=cut_entries[retained],
        extrapolated_cut_entries=extrapolated_cut_entries[retained],
        signal_entries=signal_entries[retained],
        signal_fractions=signal_fractions[retained],
        cut_component_fractions=cut_component_fractions[retained],
        nuisance_fractions=nuisance_fractions[retained],
        background_fractions=background_fractions[retained],
        peak_significance=peak_significance[retained],
        iterations=iterations[retained],
        fit_model=fit_model[retained],
        fit_parameter_names=fit_parameter_names[retained],
        fit_parameter_values=fit_parameter_values[retained],
        bic=bic[retained],
        delta_bic=delta_bic[retained],
        pearson_chi2=pearson_chi2[retained],
        deviance=deviance[retained],
        fit_ndof=fit_ndof[retained],
        continuously_refined=continuously_refined[retained],
        histogram_bin_count=histogram_bin_count[retained],
        histogram_edges=histogram_edges[retained],
        observed_counts=observed_counts[retained],
        expected_counts=expected_counts[retained],
        cut_signal_counts=cut_signal_counts[retained],
        noncut_component_counts=noncut_component_counts[retained],
        background_counts=background_counts[retained],
        window_source=window_source[retained],
        cumulative_before=cumulative_before[retained],
        cumulative_after=cumulative_after[retained],
        nminus1_entries=nminus1_entries[retained],
        nminus1_passing=nminus1_passing[retained],
        populated_group_ids=populated.astype(np.int64, copy=False),
        dropped_group_ids=populated[dropped].astype(np.int64, copy=False),
        dropped_variables=dropped_variables[dropped],
        dropped_reasons=dropped_reasons[dropped],
        global_mode=global_mode,
        n_sigma=n_sigma,
        signal_containment=base_containment,
        cut_containments=variable_containments,
        cut_components=variable_components,
        nminus1_audit_lower=nminus1_audit_lower[retained],
        nminus1_audit_upper=nminus1_audit_upper[retained],
        nminus1_audit_centers=nminus1_audit_centers[retained],
        nminus1_audit_sigmas=nminus1_audit_sigmas[retained],
        nminus1_audit_fit_entries=nminus1_audit_fit_entries[retained],
        nminus1_audit_source=nminus1_audit_source[retained],
        nminus1_audit_reasons=nminus1_audit_reasons[retained],
        nminus1_audit_success=nminus1_audit_success[retained],
        nminus1_audit_complete=nminus1_audit_complete,
        nminus1_audit_within_tolerance=nminus1_audit_within_tolerance,
        nminus1_audit_maximum_boundary_change=(
            nminus1_audit_maximum_boundary_change
        ),
        nminus1_audit_boundary_tolerance=nminus1_audit_boundary_tolerance,
        continuous_refinement=continuous_refinement,
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
    *,
    exclude_variables: tuple[str, ...] = (),
) -> Array:
    if cuts.grouping != GROUPING:
        raise ValueError(f"unsupported exclusivity grouping: {cuts.grouping}")
    if cuts.estimator != ESTIMATOR:
        raise ValueError(f"unsupported exclusivity estimator: {cuts.estimator}")
    detector = np.asarray(proton_detector, dtype=np.int64)
    ft_photons = np.asarray(ft_photons, dtype=np.int64)
    excluded = set(exclude_variables)
    unknown = excluded.difference(cuts.variables)
    if unknown:
        raise ValueError(
            "cannot exclude variables absent from the exclusivity table: "
            + ", ".join(sorted(unknown))
        )
    groups = event_group_ids(
        detector, ft_photons, iq2, ixb, it, cuts.global_mode
    )
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
        if name in excluded:
            continue
        raw = np.asarray(values[name], dtype=float)
        lo = cuts.lower[positions[rows], variable_index]
        hi = cuts.upper[positions[rows], variable_index]
        if not np.all(np.isfinite(lo) & np.isfinite(hi)):
            raise ValueError("exclusivity cut table contains inactive windows")
        mask[rows] &= np.isfinite(raw[rows]) & (raw[rows] >= lo) & (raw[rows] <= hi)
    return mask


def event_group_ids(
    detector: Array,
    ft_photons: Array,
    iq2: Array,
    ixb: Array,
    it: Array,
    global_mode: bool,
) -> Array:
    """Return the persisted exclusivity-group ID for every event."""
    return _group_ids(detector, ft_photons, iq2, ixb, it, global_mode)


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
        fit_lower=cuts.fit_lower,
        fit_upper=cuts.fit_upper,
        centers=cuts.centers,
        sigmas=cuts.sigmas,
        fit_entries=cuts.fit_entries,
        cut_entries=cuts.cut_entries,
        extrapolated_cut_entries=cuts.extrapolated_cut_entries,
        signal_entries=cuts.signal_entries,
        signal_fractions=cuts.signal_fractions,
        cut_component_fractions=cuts.cut_component_fractions,
        nuisance_fractions=cuts.nuisance_fractions,
        background_fractions=cuts.background_fractions,
        peak_significance=cuts.peak_significance,
        iterations=cuts.iterations,
        fit_model=cuts.fit_model,
        fit_parameter_names=cuts.fit_parameter_names,
        fit_parameter_values=cuts.fit_parameter_values,
        bic=cuts.bic,
        delta_bic=cuts.delta_bic,
        pearson_chi2=cuts.pearson_chi2,
        deviance=cuts.deviance,
        fit_ndof=cuts.fit_ndof,
        continuously_refined=cuts.continuously_refined,
        histogram_bin_count=cuts.histogram_bin_count,
        histogram_edges=cuts.histogram_edges,
        observed_counts=cuts.observed_counts,
        expected_counts=cuts.expected_counts,
        cut_signal_counts=cuts.cut_signal_counts,
        noncut_component_counts=cuts.noncut_component_counts,
        background_counts=cuts.background_counts,
        window_source=cuts.window_source,
        cumulative_before=cuts.cumulative_before,
        cumulative_after=cuts.cumulative_after,
        nminus1_entries=cuts.nminus1_entries,
        nminus1_passing=cuts.nminus1_passing,
        populated_group_ids=cuts.populated_group_ids,
        dropped_group_ids=cuts.dropped_group_ids,
        dropped_variables=cuts.dropped_variables,
        dropped_reasons=cuts.dropped_reasons,
        global_mode=cuts.global_mode,
        n_sigma=cuts.n_sigma,
        signal_containment=cuts.signal_containment,
        cut_containments=cuts.cut_containments,
        cut_components=cuts.cut_components,
        nminus1_audit_lower=cuts.nminus1_audit_lower,
        nminus1_audit_upper=cuts.nminus1_audit_upper,
        nminus1_audit_centers=cuts.nminus1_audit_centers,
        nminus1_audit_sigmas=cuts.nminus1_audit_sigmas,
        nminus1_audit_fit_entries=cuts.nminus1_audit_fit_entries,
        nminus1_audit_source=cuts.nminus1_audit_source,
        nminus1_audit_reasons=cuts.nminus1_audit_reasons,
        nminus1_audit_success=cuts.nminus1_audit_success,
        nminus1_audit_complete=cuts.nminus1_audit_complete,
        nminus1_audit_within_tolerance=cuts.nminus1_audit_within_tolerance,
        nminus1_audit_maximum_boundary_change=(
            cuts.nminus1_audit_maximum_boundary_change
        ),
        nminus1_audit_boundary_tolerance=cuts.nminus1_audit_boundary_tolerance,
        continuous_refinement=cuts.continuous_refinement,
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
        fit_lower=saved["fit_lower"],
        fit_upper=saved["fit_upper"],
        centers=saved["centers"],
        sigmas=saved["sigmas"],
        fit_entries=saved["fit_entries"],
        cut_entries=saved["cut_entries"],
        extrapolated_cut_entries=saved["extrapolated_cut_entries"],
        signal_entries=saved["signal_entries"],
        signal_fractions=saved["signal_fractions"],
        cut_component_fractions=saved["cut_component_fractions"],
        nuisance_fractions=saved["nuisance_fractions"],
        background_fractions=saved["background_fractions"],
        peak_significance=saved["peak_significance"],
        iterations=saved["iterations"],
        fit_model=saved["fit_model"],
        fit_parameter_names=saved["fit_parameter_names"],
        fit_parameter_values=saved["fit_parameter_values"],
        bic=saved["bic"],
        delta_bic=saved["delta_bic"],
        pearson_chi2=saved["pearson_chi2"],
        deviance=saved["deviance"],
        fit_ndof=saved["fit_ndof"],
        continuously_refined=saved["continuously_refined"],
        histogram_bin_count=saved["histogram_bin_count"],
        histogram_edges=saved["histogram_edges"],
        observed_counts=saved["observed_counts"],
        expected_counts=saved["expected_counts"],
        cut_signal_counts=saved["cut_signal_counts"],
        noncut_component_counts=saved["noncut_component_counts"],
        background_counts=saved["background_counts"],
        window_source=saved["window_source"],
        cumulative_before=saved["cumulative_before"],
        cumulative_after=saved["cumulative_after"],
        nminus1_entries=saved["nminus1_entries"],
        nminus1_passing=saved["nminus1_passing"],
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
        cut_containments=saved["cut_containments"],
        cut_components=saved["cut_components"],
        nminus1_audit_lower=saved["nminus1_audit_lower"],
        nminus1_audit_upper=saved["nminus1_audit_upper"],
        nminus1_audit_centers=saved["nminus1_audit_centers"],
        nminus1_audit_sigmas=saved["nminus1_audit_sigmas"],
        nminus1_audit_fit_entries=saved["nminus1_audit_fit_entries"],
        nminus1_audit_source=saved["nminus1_audit_source"],
        nminus1_audit_reasons=saved["nminus1_audit_reasons"],
        nminus1_audit_success=saved["nminus1_audit_success"],
        nminus1_audit_complete=bool(saved["nminus1_audit_complete"]),
        nminus1_audit_within_tolerance=bool(
            saved["nminus1_audit_within_tolerance"]
        ),
        nminus1_audit_maximum_boundary_change=float(
            saved["nminus1_audit_maximum_boundary_change"]
        ),
        nminus1_audit_boundary_tolerance=float(
            saved["nminus1_audit_boundary_tolerance"]
        ),
        continuous_refinement=bool(saved["continuous_refinement"]),
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
        cuts.fit_lower,
        cuts.fit_upper,
        cuts.centers,
        cuts.sigmas,
        cuts.fit_entries,
        cuts.cut_entries,
        cuts.extrapolated_cut_entries,
        cuts.signal_entries,
        cuts.signal_fractions,
        cuts.cut_component_fractions,
        cuts.nuisance_fractions,
        cuts.background_fractions,
        cuts.peak_significance,
        cuts.iterations,
        cuts.fit_model,
        cuts.bic,
        cuts.delta_bic,
        cuts.pearson_chi2,
        cuts.deviance,
        cuts.fit_ndof,
        cuts.continuously_refined,
        cuts.histogram_bin_count,
        cuts.window_source,
        cuts.cumulative_before,
        cuts.cumulative_after,
        cuts.nminus1_entries,
        cuts.nminus1_passing,
        cuts.nminus1_audit_lower,
        cuts.nminus1_audit_upper,
        cuts.nminus1_audit_centers,
        cuts.nminus1_audit_sigmas,
        cuts.nminus1_audit_fit_entries,
        cuts.nminus1_audit_source,
        cuts.nminus1_audit_reasons,
        cuts.nminus1_audit_success,
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
    histogram_shape = (*expected_shape, cuts.fit_histogram_bins)
    edge_shape = (*expected_shape, cuts.fit_histogram_bins + 1)
    if cuts.histogram_edges.shape != edge_shape:
        raise ValueError("exclusivity cut table has inconsistent histogram edges")
    for histogram in (
        cuts.observed_counts,
        cuts.expected_counts,
        cuts.cut_signal_counts,
        cuts.noncut_component_counts,
        cuts.background_counts,
    ):
        if histogram.shape != histogram_shape:
            raise ValueError("exclusivity cut table has inconsistent fit histograms")
    variable_shape = (len(cuts.variables),)
    if cuts.cut_containments.shape != variable_shape:
        raise ValueError("exclusivity cut table has inconsistent cut containments")
    if cuts.cut_components.shape != variable_shape:
        raise ValueError("exclusivity cut table has inconsistent cut components")
    successful_audits = cuts.nminus1_audit_success
    successful_values = (
        cuts.nminus1_audit_lower[successful_audits],
        cuts.nminus1_audit_upper[successful_audits],
        cuts.nminus1_audit_centers[successful_audits],
        cuts.nminus1_audit_sigmas[successful_audits],
    )
    if any(not np.all(np.isfinite(array)) for array in successful_values):
        raise ValueError("exclusivity cut table has invalid N-1 audit estimates")
    if cuts.nminus1_audit_complete != bool(
        successful_audits.size and np.all(successful_audits)
    ):
        raise ValueError("exclusivity cut table has inconsistent N-1 audit status")
    if not np.isfinite(cuts.nminus1_audit_maximum_boundary_change) or (
        cuts.nminus1_audit_maximum_boundary_change < 0.0
    ):
        raise ValueError("exclusivity cut table has invalid N-1 audit change")
    if not 0.0 < cuts.nminus1_audit_boundary_tolerance < 1.0:
        raise ValueError("exclusivity cut table has invalid N-1 audit tolerance")
    expected_audit_stability = bool(
        cuts.nminus1_audit_complete
        and cuts.nminus1_audit_maximum_boundary_change
        <= cuts.nminus1_audit_boundary_tolerance
    )
    if cuts.nminus1_audit_within_tolerance != expected_audit_stability:
        raise ValueError("exclusivity cut table has inconsistent N-1 audit stability")
    if np.any((~successful_audits) & (cuts.nminus1_audit_reasons == "")):
        raise ValueError("exclusivity cut table omits an N-1 audit failure reason")
    if cuts.dropped_group_ids.shape != cuts.dropped_variables.shape:
        raise ValueError("exclusivity cut table has inconsistent dropped-group diagnostics")
    if cuts.dropped_group_ids.shape != cuts.dropped_reasons.shape:
        raise ValueError("exclusivity cut table has inconsistent dropped-group diagnostics")
    if np.any(cuts.extrapolated_cut_entries != 0):
        raise ValueError("exclusivity cut table accepts events beyond a fitted domain")
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
