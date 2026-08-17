from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
from scipy.optimize import brentq, minimize, nnls
from scipy.special import i0e, ndtr
from scipy.stats import exponnorm, rice


Array = np.ndarray

_SQRT_TWO_PI = sqrt(2.0 * np.pi)
_SIGMA_SCALES = np.asarray((0.40, 0.55, 0.70, 0.85, 1.0, 1.2, 1.45))
_CENTER_OFFSETS = np.asarray((-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0))
_ASYMMETRY_RATIOS = np.asarray((0.50, 0.70, 1.0, 1.4, 2.0))
_MAX_FIT_PARAMETERS = 16


@dataclass(frozen=True)
class Component:
    kind: str
    parameters: tuple[float, ...]
    name: str


@dataclass(frozen=True)
class Candidate:
    name: str
    components: tuple[Component, ...]
    center: float
    scale: float
    parameter_names: tuple[str, ...]
    parameter_values: tuple[float, ...]
    free_shape_parameters: int
    one_sided: bool = False
    sideband_linear: bool = False
    signal_components: int = 1
    nuisance_components: int = 0


@dataclass(frozen=True)
class Evaluation:
    candidate: Candidate
    weights: Array
    components: Array
    iterations: int
    nll: float
    bic: float
    parameter_count: int
    refined: bool = False


@dataclass(frozen=True)
class FitEstimate:
    center: float
    sigma: float
    lower: float
    upper: float
    fit_lower: float
    fit_upper: float
    entries: int
    fit_entries: int
    cut_entries: int
    extrapolated_cut_entries: int
    signal_entries: float
    signal_fraction: float
    cut_component_fraction: float
    nuisance_fraction: float
    background_fraction: float
    peak_significance: float
    iterations: int
    fit_model: str
    parameter_names: tuple[str, ...]
    parameter_values: tuple[float, ...]
    bic: float
    delta_bic: float
    pearson_chi2: float
    deviance: float
    fit_ndof: int
    continuously_refined: bool
    histogram_edges: Array
    observed_counts: Array
    expected_counts: Array
    cut_signal_counts: Array
    noncut_component_counts: Array
    background_counts: Array


def estimate_model(
    values: Array,
    variable: str,
    n_sigma: float,
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
    cut_containment: float | None = None,
    cut_component: str = "core",
    require_cut_within_fit: bool = True,
    continuous_refinement: bool = True,
) -> tuple[FitEstimate | None, str]:
    containment = (
        float(cut_containment)
        if cut_containment is not None
        else erf(n_sigma / sqrt(2.0))
    )
    if not 0.0 < containment < 1.0:
        return None, f"cut containment must be between zero and one ({containment})"
    if cut_component not in {"core", "signal"}:
        return None, f"unsupported cut component policy: {cut_component}"

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    entries = int(finite.size)
    if entries < minimum_events:
        return None, f"too few finite entries ({entries} < {minimum_events})"

    seed = _mode_seed(finite, histogram_bins, expected_center)
    if seed is None:
        return None, "could not determine a finite mode and seed width"
    seed_center, seed_sigma = seed
    fit_extent = max(fit_window_sigma, 10.0)
    if variable == "rec_m_eggX":
        fit_extent = max(fit_extent, 12.0)
    elif variable == "rec_m2_miss":
        fit_extent = max(fit_extent, 20.0)
    fit_lo = seed_center - fit_extent * seed_sigma
    fit_hi = seed_center + fit_extent * seed_sigma
    if variable == "rec_pT_miss":
        fit_lo = 0.0
    elif physical_lower is not None:
        fit_lo = max(fit_lo, physical_lower)
    if not (np.isfinite(fit_lo) and np.isfinite(fit_hi) and fit_hi > fit_lo):
        return None, "fit window was non-finite or empty"

    selected = finite[(finite >= fit_lo) & (finite <= fit_hi)]
    minimum_fit_entries = max(20, int(np.ceil(0.5 * minimum_events)))
    if selected.size < minimum_fit_entries:
        return None, (
            f"too few entries in fit window ({selected.size} < {minimum_fit_entries})"
        )

    bins = min(histogram_bins, max(20, 2 * int(np.sqrt(selected.size))))
    counts, edges = np.histogram(selected, bins=bins, range=(fit_lo, fit_hi))
    x = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    span = fit_hi - fit_lo
    u = np.clip((x - fit_lo) / span, 0.0, 1.0)
    background_left = 2.0 * (1.0 - u)
    background_right = 2.0 * u
    background_left /= np.sum(background_left)
    background_right /= np.sum(background_right)

    candidates = _model_candidates(
        variable, seed_center, seed_sigma, x, bin_width, fit_lo, fit_hi
    )
    family_best: dict[str, Evaluation] = {}
    for candidate in candidates:
        evaluation = _evaluate_candidate(
            candidate,
            counts,
            x,
            background_left,
            background_right,
            max_iterations,
            convergence,
            selected.size,
        )
        if evaluation is None:
            continue
        previous = family_best.get(candidate.name)
        if previous is None or evaluation.bic < previous.bic:
            family_best[candidate.name] = evaluation

    if not family_best:
        return None, f"no finite {variable} model candidate"

    family_results = []
    for evaluation in family_best.values():
        if continuous_refinement:
            evaluation = _refine_evaluation(
                evaluation,
                counts,
                x,
                background_left,
                background_right,
                fit_lo,
                fit_hi,
                bin_width,
                max_iterations,
                convergence,
                selected.size,
            )
        family_results.append(evaluation)
    family_results.sort(key=lambda item: item.bic)
    best = family_results[0]
    delta_bic = (
        float(family_results[1].bic - best.bic)
        if len(family_results) > 1
        else float("inf")
    )

    candidate = best.candidate
    weights = best.weights
    components = best.components
    iterations = best.iterations
    fit_component_count = len(candidate.components)
    signal_count = candidate.signal_components
    nuisance_count = candidate.nuisance_components
    cut_count = 1 if cut_component == "core" else signal_count
    fitted_signal_weights = weights[:signal_count]
    signal_fraction = float(np.sum(fitted_signal_weights))
    signal_entries = signal_fraction * selected.size
    if signal_fraction <= 0.0:
        return None, "fitted signal fraction was zero"
    cut_component_fraction = float(np.sum(weights[:cut_count]))
    nuisance_fraction = float(
        np.sum(weights[signal_count : signal_count + nuisance_count])
    )
    background_fraction = float(
        np.sum(weights[signal_count + nuisance_count :])
    )

    global_signal_weights = _untruncate_signal_weights(
        candidate.components[:cut_count],
        weights[:cut_count],
        fit_lo,
        fit_hi,
    )
    lower, upper = _signal_window(
        candidate.components[:cut_count],
        global_signal_weights,
        containment,
        candidate.one_sided,
    )
    if physical_lower is not None:
        lower = max(lower, physical_lower)
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        return None, "derived signal-containment window was non-finite or empty"
    cut_tolerance = 0.5 * bin_width
    if require_cut_within_fit and (
        lower < fit_lo - cut_tolerance or upper > fit_hi + cut_tolerance
    ):
        return None, (
            f"derived cut [{lower:.6g}, {upper:.6g}] extrapolates beyond fitted "
            f"domain [{fit_lo:.6g}, {fit_hi:.6g}]"
        )

    in_window = (x >= lower) & (x <= upper)
    signal_in_window = selected.size * float(
        np.sum(
            [
                weights[index] * np.sum(components[index, in_window])
                for index in range(cut_count)
            ]
        )
    )
    background_in_window = selected.size * float(
        np.sum(
            [
                weights[index] * np.sum(components[index, in_window])
                for index in range(cut_count, components.shape[0])
            ]
        )
    )
    significance = signal_in_window / np.sqrt(
        max(signal_in_window + background_in_window, 1.0)
    )

    if signal_fraction < minimum_signal_fraction:
        return None, (
            f"signal fraction {signal_fraction:.6g} below minimum "
            f"{minimum_signal_fraction:.6g}"
        )
    if significance < minimum_peak_significance:
        return None, (
            f"peak significance {significance:.6g} below minimum "
            f"{minimum_peak_significance:.6g}"
        )
    if maximum_sigma is not None and candidate.scale > maximum_sigma:
        return None, (
            f"characteristic scale {candidate.scale:.6g} exceeds maximum "
            f"{maximum_sigma:.6g}"
        )
    if expected_center is not None and maximum_center_deviation is not None:
        tolerance = max(maximum_center_deviation, 2.0 * bin_width)
        if abs(candidate.center - expected_center) > tolerance:
            return None, (
                f"center shift {abs(candidate.center - expected_center):.6g} "
                f"exceeds maximum {tolerance:.6g}"
            )

    parameter_names = list(candidate.parameter_names)
    parameter_values = list(candidate.parameter_values)
    if fit_component_count > 1:
        for component, fraction in zip(
            candidate.components, weights[:fit_component_count], strict=True
        ):
            parameter_names.append(f"{component.name}_fit_fraction")
            parameter_values.append(float(fraction))
    if len(parameter_names) > _MAX_FIT_PARAMETERS:
        raise RuntimeError("exclusivity model exceeds fit-parameter storage")

    expected_counts = selected.size * (weights @ components)
    cut_signal_counts = selected.size * np.sum(
        components[:cut_count] * weights[:cut_count, None], axis=0
    )
    noncut_component_counts = selected.size * np.sum(
        components[cut_count:fit_component_count]
        * weights[cut_count:fit_component_count, None],
        axis=0,
    ) if fit_component_count > cut_count else np.zeros_like(expected_counts)
    background_counts = selected.size * np.sum(
        components[fit_component_count:]
        * weights[fit_component_count:, None],
        axis=0,
    )
    pearson_chi2 = float(
        np.sum((counts - expected_counts) ** 2 / np.maximum(expected_counts, 1.0))
    )
    positive = counts > 0
    deviance = float(
        2.0
        * (
            np.sum(counts[positive] * np.log(counts[positive] / expected_counts[positive]))
            - np.sum(counts - expected_counts)
        )
    )
    fit_ndof = max(1, counts.size - 1 - best.parameter_count)
    in_cut = (finite >= lower) & (finite <= upper)
    in_fit = (finite >= fit_lo) & (finite <= fit_hi)
    cut_entries = int(np.count_nonzero(in_cut))
    extrapolated_cut_entries = int(np.count_nonzero(in_cut & ~in_fit))

    return FitEstimate(
        center=candidate.center,
        sigma=candidate.scale,
        lower=float(lower),
        upper=float(upper),
        fit_lower=float(fit_lo),
        fit_upper=float(fit_hi),
        entries=entries,
        fit_entries=int(selected.size),
        cut_entries=cut_entries,
        extrapolated_cut_entries=extrapolated_cut_entries,
        signal_entries=float(signal_entries),
        signal_fraction=signal_fraction,
        cut_component_fraction=cut_component_fraction,
        nuisance_fraction=nuisance_fraction,
        background_fraction=background_fraction,
        peak_significance=float(significance),
        iterations=iterations,
        fit_model=candidate.name,
        parameter_names=tuple(parameter_names),
        parameter_values=tuple(parameter_values),
        bic=float(best.bic),
        delta_bic=delta_bic,
        pearson_chi2=pearson_chi2,
        deviance=deviance,
        fit_ndof=fit_ndof,
        continuously_refined=best.refined,
        histogram_edges=edges,
        observed_counts=counts.astype(float),
        expected_counts=expected_counts,
        cut_signal_counts=cut_signal_counts,
        noncut_component_counts=noncut_component_counts,
        background_counts=background_counts,
    ), ""


def maximum_fit_parameters() -> int:
    return _MAX_FIT_PARAMETERS


def _model_candidates(
    variable: str,
    seed_center: float,
    seed_sigma: float,
    x: Array,
    bin_width: float,
    fit_lo: float,
    fit_hi: float,
) -> list[Candidate]:
    if variable == "rec_m_gg":
        return _gaussian_candidates(
            seed_center, seed_sigma, fit_lo, fit_hi, sideband_linear=True
        )
    if variable == "rec_pT_miss":
        return _rice_candidates(seed_center, seed_sigma, x, bin_width)
    if variable == "rec_m_eggX":
        return _split_gaussian_candidates(seed_center, seed_sigma, fit_lo, fit_hi)
    if variable == "rec_E_miss":
        return _positive_tail_candidates(seed_center, seed_sigma, fit_lo, fit_hi)
    if variable == "rec_m2_miss":
        return _asymmetric_laplace_candidates(
            seed_center, seed_sigma, fit_lo, fit_hi
        )
    return _gaussian_tail_candidates(seed_center, seed_sigma, fit_lo, fit_hi)


def _gaussian_candidates(
    center: float,
    sigma: float,
    fit_lo: float,
    fit_hi: float,
    *,
    sideband_linear: bool = False,
) -> list[Candidate]:
    candidates = []
    name = "gaussian+sideband-linear" if sideband_linear else "gaussian+linear"
    for sigma_scale in _SIGMA_SCALES:
        width = float(sigma * sigma_scale)
        if width <= 0.0 or width >= 0.5 * (fit_hi - fit_lo):
            continue
        for offset in _CENTER_OFFSETS:
            mu = float(center + offset * sigma)
            if not fit_lo < mu < fit_hi:
                continue
            candidates.append(
                Candidate(
                    name=name,
                    components=(Component("gaussian", (mu, width), "core"),),
                    center=mu,
                    scale=width,
                    parameter_names=("mu", "sigma"),
                    parameter_values=(mu, width),
                    free_shape_parameters=2,
                    sideband_linear=sideband_linear,
                )
            )
    return candidates


def _gaussian_tail_candidates(
    center: float, sigma: float, fit_lo: float, fit_hi: float
) -> list[Candidate]:
    candidates = _gaussian_candidates(center, sigma, fit_lo, fit_hi)
    for sigma_scale in _SIGMA_SCALES:
        width = float(sigma * sigma_scale)
        if width <= 0.0 or width >= 0.5 * (fit_hi - fit_lo):
            continue
        for offset in _CENTER_OFFSETS:
            mu = float(center + offset * sigma)
            if not fit_lo < mu < fit_hi:
                continue
            tail_width = max(2.5 * width, 1.75 * sigma)
            candidates.append(
                Candidate(
                    name="gaussian+broad-gaussian+linear",
                    components=(
                        Component("gaussian", (mu, width), "core"),
                        Component("gaussian", (mu, tail_width), "broad_tail"),
                    ),
                    center=mu,
                    scale=width,
                    parameter_names=("mu", "sigma", "tail_sigma"),
                    parameter_values=(mu, width, tail_width),
                    free_shape_parameters=3,
                    signal_components=1,
                    nuisance_components=1,
                )
            )
    return candidates


def _rice_candidates(
    center: float, seed_sigma: float, x: Array, bin_width: float
) -> list[Candidate]:
    base = max(seed_sigma, 0.5 * center, bin_width)
    sigma_scales = np.asarray((0.35, 0.50, 0.70, 0.90, 1.15, 1.45, 1.80))
    nu_scales = np.asarray((0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50))
    candidates = []
    for sigma_scale in sigma_scales:
        width = float(base * sigma_scale)
        if width <= 0.0:
            continue
        for nu_scale in nu_scales:
            nu = float(max(center, base) * nu_scale)
            component = Component("rice", (nu, width), "rice")
            shape = _component_pdf(component, x)
            mode = float(x[int(np.argmax(shape))])
            rayleigh = nu == 0.0
            candidates.append(
                Candidate(
                    name=("rayleigh+linear" if rayleigh else "rice+linear"),
                    components=(component,),
                    center=mode,
                    scale=width,
                    parameter_names=(("sigma",) if rayleigh else ("nu", "sigma")),
                    parameter_values=((width,) if rayleigh else (nu, width)),
                    free_shape_parameters=(1 if rayleigh else 2),
                    one_sided=True,
                )
            )
    return candidates


def _split_gaussian_candidates(
    center: float, sigma: float, fit_lo: float, fit_hi: float
) -> list[Candidate]:
    candidates = _gaussian_candidates(center, sigma, fit_lo, fit_hi)
    for sigma_scale in _SIGMA_SCALES:
        base = float(sigma * sigma_scale)
        if base <= 0.0:
            continue
        for ratio in _ASYMMETRY_RATIOS:
            if ratio == 1.0:
                continue
            left_width = base / sqrt(float(ratio))
            right_width = base * sqrt(float(ratio))
            if max(left_width, right_width) >= 0.5 * (fit_hi - fit_lo):
                continue
            for offset in _CENTER_OFFSETS:
                mu = float(center + offset * sigma)
                if not fit_lo < mu < fit_hi:
                    continue
                candidates.append(
                    Candidate(
                        name="split-gaussian+linear",
                        components=(
                            Component(
                                "split_gaussian",
                                (mu, left_width, right_width),
                                "split_core",
                            ),
                        ),
                        center=mu,
                        scale=max(left_width, right_width),
                        parameter_names=("mu", "sigma_left", "sigma_right"),
                        parameter_values=(mu, left_width, right_width),
                        free_shape_parameters=3,
                    )
                )
    return candidates


def _positive_tail_candidates(
    center: float, sigma: float, fit_lo: float, fit_hi: float
) -> list[Candidate]:
    candidates = _gaussian_candidates(center, sigma, fit_lo, fit_hi)
    for sigma_scale in _SIGMA_SCALES:
        width = float(sigma * sigma_scale)
        if width <= 0.0 or width >= 0.5 * (fit_hi - fit_lo):
            continue
        for offset in _CENTER_OFFSETS:
            mu = float(center + offset * sigma)
            if not fit_lo < mu < fit_hi:
                continue
            for tail_multiple in (1.0, 2.0, 4.0):
                tau = tail_multiple * width
                candidates.append(
                    Candidate(
                        name="gaussian+positive-exgaussian+linear",
                        components=(
                            Component("gaussian", (mu, width), "core"),
                            Component("exgaussian", (mu, width, tau), "positive_tail"),
                        ),
                        center=mu,
                        scale=width,
                        parameter_names=("mu", "sigma", "tail_tau"),
                        parameter_values=(mu, width, tau),
                        free_shape_parameters=3,
                        signal_components=2,
                    )
                )
    return candidates


def _asymmetric_laplace_candidates(
    center: float, seed_sigma: float, fit_lo: float, fit_hi: float
) -> list[Candidate]:
    candidates = []
    base_scale = seed_sigma / max(np.log(2.0) / 1.1775, 1.0e-6)
    for scale_factor in _SIGMA_SCALES:
        base = float(base_scale * scale_factor)
        if base <= 0.0:
            continue
        for ratio in _ASYMMETRY_RATIOS:
            left_scale = base / sqrt(float(ratio))
            right_scale = base * sqrt(float(ratio))
            if max(left_scale, right_scale) >= 0.5 * (fit_hi - fit_lo):
                continue
            for offset in _CENTER_OFFSETS:
                mu = float(center + offset * seed_sigma)
                if not fit_lo < mu < fit_hi:
                    continue
                symmetric = ratio == 1.0
                base_name = (
                    "laplace+linear"
                    if symmetric
                    else "asymmetric-laplace+linear"
                )
                parameter_names = (
                    ("mu", "b")
                    if symmetric
                    else ("mu", "b_left", "b_right")
                )
                parameter_values = (
                    (mu, left_scale)
                    if symmetric
                    else (mu, left_scale, right_scale)
                )
                core = Component(
                    "asymmetric_laplace",
                    (mu, left_scale, right_scale),
                    "cusp_core",
                )
                candidates.append(
                    Candidate(
                        name=base_name,
                        components=(core,),
                        center=mu,
                        scale=max(left_scale, right_scale),
                        parameter_names=parameter_names,
                        parameter_values=parameter_values,
                        free_shape_parameters=(2 if symmetric else 3),
                    )
                )
                for broad_ratio in (2.5, 4.0):
                    broad_scale = broad_ratio * max(left_scale, right_scale)
                    candidates.append(
                        Candidate(
                            name=(
                                "laplace+broad-laplace+linear"
                                if symmetric
                                else "asymmetric-laplace+broad-laplace+linear"
                            ),
                            components=(
                                core,
                                Component(
                                    "asymmetric_laplace",
                                    (mu, broad_scale, broad_scale),
                                    "broad_cusp_nuisance",
                                ),
                            ),
                            center=mu,
                            scale=max(left_scale, right_scale),
                            parameter_names=(*parameter_names, "nuisance_b"),
                            parameter_values=(*parameter_values, broad_scale),
                            free_shape_parameters=(3 if symmetric else 4),
                            signal_components=1,
                            nuisance_components=1,
                        )
                    )
    return candidates


def _evaluate_candidate(
    candidate: Candidate,
    counts: Array,
    x: Array,
    background_left: Array,
    background_right: Array,
    max_iterations: int,
    convergence: float,
    fit_entries: int,
) -> Evaluation | None:
    component_shapes = []
    for component in candidate.components:
        shape = _normalised_shape(_component_pdf(component, x))
        if shape is None:
            return None
        component_shapes.append(shape)

    if candidate.sideband_linear:
        background = _sideband_linear_shape(
            counts,
            x,
            candidate.center,
            candidate.scale,
            background_left,
            background_right,
        )
        if background is None:
            return None
        background_shapes = [background]
        background_shape_parameters = 2
    else:
        background_shapes = [background_left, background_right]
        background_shape_parameters = 0

    components = np.vstack((*component_shapes, *background_shapes))
    weights, iterations = _mixture_weights(
        counts, components, max_iterations, convergence
    )
    density = np.maximum(weights @ components, np.finfo(float).tiny)
    nll = float(-np.sum(counts * np.log(density)))
    free_weights = components.shape[0] - 1
    parameter_count = (
        candidate.free_shape_parameters
        + free_weights
        + background_shape_parameters
    )
    bic = 2.0 * nll + parameter_count * np.log(fit_entries)
    return Evaluation(
        candidate=candidate,
        weights=weights,
        components=components,
        iterations=iterations,
        nll=nll,
        bic=float(bic),
        parameter_count=parameter_count,
    )


def _refine_evaluation(
    initial: Evaluation,
    counts: Array,
    x: Array,
    background_left: Array,
    background_right: Array,
    fit_lo: float,
    fit_hi: float,
    bin_width: float,
    max_iterations: int,
    convergence: float,
    fit_entries: int,
) -> Evaluation:
    candidate = initial.candidate
    parameters = np.asarray(candidate.parameter_values, dtype=float)
    bounds = _parameter_bounds(
        candidate.parameter_names, fit_lo, fit_hi, bin_width
    )
    if len(bounds) != parameters.size:
        return initial
    parameters = np.asarray(
        [
            np.clip(value, lower_bound, upper_bound)
            for value, (lower_bound, upper_bound) in zip(
                parameters, bounds, strict=True
            )
        ],
        dtype=float,
    )

    def objective(raw: Array) -> float:
        refined_candidate = _candidate_with_parameters(candidate, raw)
        if refined_candidate is None:
            return 1.0e100
        evaluation = _evaluate_candidate(
            refined_candidate,
            counts,
            x,
            background_left,
            background_right,
            min(max_iterations, 60),
            convergence,
            fit_entries,
        )
        return evaluation.nll if evaluation is not None else 1.0e100

    result = minimize(
        objective,
        parameters,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": min(max_iterations, 60),
            "xtol": max(1.0e-8, 1.0e-5 * (fit_hi - fit_lo)),
            "ftol": 1.0e-7,
        },
    )
    if not np.all(np.isfinite(result.x)):
        return initial
    refined_candidate = _candidate_with_parameters(candidate, result.x)
    if refined_candidate is None:
        return initial
    refined = _evaluate_candidate(
        refined_candidate,
        counts,
        x,
        background_left,
        background_right,
        max_iterations,
        convergence,
        fit_entries,
    )
    if refined is None or refined.bic >= initial.bic:
        return initial
    return Evaluation(
        candidate=refined.candidate,
        weights=refined.weights,
        components=refined.components,
        iterations=int(getattr(result, "nit", refined.iterations)),
        nll=refined.nll,
        bic=refined.bic,
        parameter_count=refined.parameter_count,
        refined=True,
    )


def _parameter_bounds(
    names: tuple[str, ...], fit_lo: float, fit_hi: float, bin_width: float
) -> list[tuple[float, float]]:
    span = fit_hi - fit_lo
    minimum_width = max(0.25 * bin_width, span / 10000.0)
    bounds = []
    for name in names:
        if name == "mu":
            bounds.append((fit_lo + 0.25 * bin_width, fit_hi - 0.25 * bin_width))
        elif name == "nu":
            bounds.append((0.0, max(fit_hi, span)))
        elif name in {"tail_sigma", "tail_tau", "nuisance_b"}:
            bounds.append((minimum_width, span))
        else:
            bounds.append((minimum_width, 0.49 * span))
    return bounds


def _candidate_with_parameters(
    template: Candidate, raw: Array
) -> Candidate | None:
    values = tuple(float(item) for item in raw)
    name = template.name
    if name in {"gaussian+linear", "gaussian+sideband-linear"}:
        mu, sigma = values
        components = (Component("gaussian", (mu, sigma), "core"),)
        center, scale = mu, sigma
    elif name == "gaussian+broad-gaussian+linear":
        mu, sigma, tail_sigma = values
        if tail_sigma < 2.0 * sigma:
            return None
        components = (
            Component("gaussian", (mu, sigma), "core"),
            Component("gaussian", (mu, tail_sigma), "broad_tail"),
        )
        center, scale = mu, sigma
    elif name == "rayleigh+linear":
        (sigma,) = values
        components = (Component("rice", (0.0, sigma), "rice"),)
        center, scale = sigma, sigma
    elif name == "rice+linear":
        nu, sigma = values
        components = (Component("rice", (nu, sigma), "rice"),)
        center = 0.5 * (nu + np.sqrt(nu * nu + 4.0 * sigma * sigma))
        scale = sigma
    elif name == "split-gaussian+linear":
        mu, left, right = values
        components = (Component("split_gaussian", values, "split_core"),)
        center, scale = mu, max(left, right)
    elif name == "gaussian+positive-exgaussian+linear":
        mu, sigma, tau = values
        components = (
            Component("gaussian", (mu, sigma), "core"),
            Component("exgaussian", (mu, sigma, tau), "positive_tail"),
        )
        center, scale = mu, sigma
    elif name == "laplace+linear":
        mu, width = values
        components = (
            Component("asymmetric_laplace", (mu, width, width), "cusp_core"),
        )
        center, scale = mu, width
    elif name == "asymmetric-laplace+linear":
        mu, left, right = values
        components = (
            Component("asymmetric_laplace", values, "cusp_core"),
        )
        center, scale = mu, max(left, right)
    elif name == "laplace+broad-laplace+linear":
        mu, width, nuisance = values
        if nuisance < 2.0 * width:
            return None
        components = (
            Component("asymmetric_laplace", (mu, width, width), "cusp_core"),
            Component(
                "asymmetric_laplace",
                (mu, nuisance, nuisance),
                "broad_cusp_nuisance",
            ),
        )
        center, scale = mu, width
    elif name == "asymmetric-laplace+broad-laplace+linear":
        mu, left, right, nuisance = values
        if nuisance < 2.0 * max(left, right):
            return None
        components = (
            Component("asymmetric_laplace", (mu, left, right), "cusp_core"),
            Component(
                "asymmetric_laplace",
                (mu, nuisance, nuisance),
                "broad_cusp_nuisance",
            ),
        )
        center, scale = mu, max(left, right)
    else:
        return None
    return Candidate(
        name=template.name,
        components=components,
        center=float(center),
        scale=float(scale),
        parameter_names=template.parameter_names,
        parameter_values=values,
        free_shape_parameters=template.free_shape_parameters,
        one_sided=template.one_sided,
        sideband_linear=template.sideband_linear,
        signal_components=template.signal_components,
        nuisance_components=template.nuisance_components,
    )


def _component_pdf(component: Component, x: Array) -> Array:
    if component.kind == "gaussian":
        mu, sigma = component.parameters
        z = (x - mu) / sigma
        return np.exp(-0.5 * z * z) / (sigma * _SQRT_TWO_PI)
    if component.kind == "split_gaussian":
        mu, left, right = component.parameters
        sigma = np.where(x < mu, left, right)
        normalisation = sqrt(2.0 / np.pi) / (left + right)
        return normalisation * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    if component.kind == "rice":
        nu, sigma = component.parameters
        result = np.zeros_like(x, dtype=float)
        positive = x > 0.0
        z = x[positive] * nu / (sigma * sigma)
        log_pdf = (
            np.log(x[positive])
            - 2.0 * np.log(sigma)
            - (x[positive] ** 2 + nu * nu) / (2.0 * sigma * sigma)
            + np.log(i0e(z))
            + np.abs(z)
        )
        result[positive] = np.exp(log_pdf)
        return result
    if component.kind == "exgaussian":
        mu, sigma, tau = component.parameters
        return exponnorm.pdf(x, tau / sigma, loc=mu, scale=sigma)
    if component.kind == "asymmetric_laplace":
        mu, left, right = component.parameters
        result = np.empty_like(x, dtype=float)
        left_mask = x < mu
        result[left_mask] = np.exp((x[left_mask] - mu) / left)
        result[~left_mask] = np.exp(-(x[~left_mask] - mu) / right)
        return result / (left + right)
    raise ValueError(f"unsupported exclusivity component: {component.kind}")


def _component_cdf(component: Component, value: float | Array) -> Array:
    x = np.asarray(value, dtype=float)
    if component.kind == "gaussian":
        mu, sigma = component.parameters
        return ndtr((x - mu) / sigma)
    if component.kind == "split_gaussian":
        mu, left, right = component.parameters
        left_cdf = (
            2.0 * left / (left + right) * ndtr((x - mu) / left)
        )
        right_cdf = left / (left + right) + (
            2.0
            * right
            / (left + right)
            * (ndtr((x - mu) / right) - 0.5)
        )
        return np.where(x < mu, left_cdf, right_cdf)
    if component.kind == "rice":
        nu, sigma = component.parameters
        return rice.cdf(x, nu / sigma, scale=sigma)
    if component.kind == "exgaussian":
        mu, sigma, tau = component.parameters
        return exponnorm.cdf(x, tau / sigma, loc=mu, scale=sigma)
    if component.kind == "asymmetric_laplace":
        mu, left, right = component.parameters
        left_cdf = (
            left / (left + right) * np.exp((x - mu) / left)
        )
        right_cdf = 1.0 - (
            right / (left + right) * np.exp(-(x - mu) / right)
        )
        return np.where(x < mu, left_cdf, right_cdf)
    raise ValueError(f"unsupported exclusivity component: {component.kind}")


def _component_bounds(component: Component) -> tuple[float, float]:
    if component.kind == "gaussian":
        mu, sigma = component.parameters
        return mu - 30.0 * sigma, mu + 30.0 * sigma
    if component.kind == "split_gaussian":
        mu, left, right = component.parameters
        return mu - 30.0 * left, mu + 30.0 * right
    if component.kind == "rice":
        nu, sigma = component.parameters
        return 0.0, nu + 30.0 * sigma
    if component.kind == "exgaussian":
        mu, sigma, tau = component.parameters
        return mu - 30.0 * sigma, mu + 30.0 * (sigma + tau)
    if component.kind == "asymmetric_laplace":
        mu, left, right = component.parameters
        return mu - 50.0 * left, mu + 50.0 * right
    raise ValueError(f"unsupported exclusivity component: {component.kind}")


def _normalised_shape(values: Array) -> Array | None:
    shape = np.asarray(values, dtype=float)
    total = float(np.sum(shape))
    if not np.all(np.isfinite(shape)) or total <= 0.0:
        return None
    return shape / total


def _sideband_linear_shape(
    counts: Array,
    x: Array,
    center: float,
    scale: float,
    background_left: Array,
    background_right: Array,
) -> Array | None:
    sideband = np.abs(x - center) >= 2.5 * scale
    if np.count_nonzero(sideband) < 6 or np.sum(counts[sideband]) <= 0:
        return None
    design = np.column_stack((background_left[sideband], background_right[sideband]))
    coefficients, _ = nnls(design, counts[sideband].astype(float))
    background = coefficients[0] * background_left + coefficients[1] * background_right
    return _normalised_shape(background)


def _untruncate_signal_weights(
    components: tuple[Component, ...],
    fitted_weights: Array,
    fit_lo: float,
    fit_hi: float,
) -> Array:
    masses = np.asarray(
        [
            float(_component_cdf(component, fit_hi) - _component_cdf(component, fit_lo))
            for component in components
        ]
    )
    masses = np.maximum(masses, np.finfo(float).tiny)
    weights = fitted_weights / masses
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(components), 1.0 / len(components))
    return weights / total


def _signal_window(
    components: tuple[Component, ...],
    weights: Array,
    containment: float,
    one_sided: bool,
) -> tuple[float, float]:
    lower_bound = min(_component_bounds(component)[0] for component in components)
    upper_bound = max(_component_bounds(component)[1] for component in components)

    def mixture_cdf(value: float) -> float:
        return float(
            np.sum(
                [
                    weight * float(_component_cdf(component, value))
                    for component, weight in zip(components, weights, strict=True)
                ]
            )
        )

    def quantile(probability: float) -> float:
        return float(
            brentq(
                lambda value: mixture_cdf(value) - probability,
                lower_bound,
                upper_bound,
                maxiter=200,
            )
        )

    if one_sided:
        return max(0.0, lower_bound), quantile(containment)
    tail_probability = 0.5 * (1.0 - containment)
    return quantile(tail_probability), quantile(1.0 - tail_probability)


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
