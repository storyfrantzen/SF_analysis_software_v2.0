from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
from scipy.optimize import brentq, nnls
from scipy.special import i0e, ndtr
from scipy.stats import exponnorm, rice


Array = np.ndarray

_SQRT_TWO_PI = sqrt(2.0 * np.pi)
_SIGMA_SCALES = np.asarray((0.40, 0.55, 0.70, 0.85, 1.0, 1.2, 1.45))
_CENTER_OFFSETS = np.asarray((-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0))
_ASYMMETRY_RATIOS = np.asarray((0.50, 0.70, 1.0, 1.4, 2.0))
_MAX_FIT_PARAMETERS = 8


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
    primary_components: int = 1


@dataclass(frozen=True)
class FitEstimate:
    center: float
    sigma: float
    lower: float
    upper: float
    entries: int
    fit_entries: int
    signal_entries: float
    signal_fraction: float
    peak_significance: float
    iterations: int
    fit_model: str
    parameter_names: tuple[str, ...]
    parameter_values: tuple[float, ...]


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
) -> tuple[FitEstimate | None, str]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    entries = int(finite.size)
    if entries < minimum_events:
        return None, f"too few finite entries ({entries} < {minimum_events})"

    seed = _mode_seed(finite, histogram_bins, expected_center)
    if seed is None:
        return None, "could not determine a finite mode and seed width"
    seed_center, seed_sigma = seed
    qlo, qhi = np.quantile(finite, [0.005, 0.995])
    fit_lo = max(float(qlo), seed_center - fit_window_sigma * seed_sigma)
    fit_hi = min(float(qhi), seed_center + fit_window_sigma * seed_sigma)
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
    best: tuple[float, Candidate, Array, Array, int] | None = None
    for candidate in candidates:
        signal_shapes = []
        valid = True
        for component in candidate.components:
            shape = _normalised_shape(_component_pdf(component, x))
            if shape is None:
                valid = False
                break
            signal_shapes.append(shape)
        if not valid:
            continue

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
                continue
            background_shapes = [background]
            background_shape_parameters = 2
        else:
            background_shapes = [background_left, background_right]
            background_shape_parameters = 0

        components = np.vstack((*signal_shapes, *background_shapes))
        weights, iterations = _mixture_weights(
            counts, components, max_iterations, convergence
        )
        density = np.maximum(weights @ components, np.finfo(float).tiny)
        nll = float(-np.sum(counts * np.log(density)))
        free_weights = components.shape[0] - 1
        parameters = (
            candidate.free_shape_parameters
            + free_weights
            + background_shape_parameters
        )
        bic = 2.0 * nll + parameters * np.log(selected.size)
        if best is None or bic < best[0]:
            best = (bic, candidate, weights, components, iterations)

    if best is None:
        return None, f"no finite {variable} model candidate"

    _, candidate, weights, components, iterations = best
    fit_component_count = len(candidate.components)
    primary_count = candidate.primary_components
    fitted_signal_weights = weights[:primary_count]
    signal_fraction = float(np.sum(fitted_signal_weights))
    signal_entries = signal_fraction * selected.size
    if signal_fraction <= 0.0:
        return None, "fitted signal fraction was zero"

    global_signal_weights = _untruncate_signal_weights(
        candidate.components[:primary_count],
        fitted_signal_weights,
        fit_lo,
        fit_hi,
    )
    containment = erf(n_sigma / sqrt(2.0))
    lower, upper = _signal_window(
        candidate.components[:primary_count],
        global_signal_weights,
        containment,
        candidate.one_sided,
    )
    if physical_lower is not None:
        lower = max(lower, physical_lower)
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        return None, "derived signal-containment window was non-finite or empty"

    in_window = (x >= lower) & (x <= upper)
    signal_in_window = selected.size * float(
        np.sum(
            [
                weights[index] * np.sum(components[index, in_window])
                for index in range(primary_count)
            ]
        )
    )
    background_in_window = selected.size * float(
        np.sum(
            [
                weights[index] * np.sum(components[index, in_window])
                for index in range(primary_count, components.shape[0])
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
        component_weights = _untruncate_signal_weights(
            candidate.components,
            weights[:fit_component_count],
            fit_lo,
            fit_hi,
        )
        for component, fraction in zip(
            candidate.components, component_weights, strict=True
        ):
            parameter_names.append(f"{component.name}_fraction")
            parameter_values.append(float(fraction))
    if len(parameter_names) > _MAX_FIT_PARAMETERS:
        raise RuntimeError("exclusivity model exceeds fit-parameter storage")

    return FitEstimate(
        center=candidate.center,
        sigma=candidate.scale,
        lower=float(lower),
        upper=float(upper),
        entries=entries,
        fit_entries=int(selected.size),
        signal_entries=float(signal_entries),
        signal_fraction=signal_fraction,
        peak_significance=float(significance),
        iterations=iterations,
        fit_model=candidate.name,
        parameter_names=tuple(parameter_names),
        parameter_values=tuple(parameter_values),
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
                        primary_components=2,
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
                candidates.append(
                    Candidate(
                        name=("laplace+linear" if symmetric else "asymmetric-laplace+linear"),
                        components=(
                            Component(
                                "asymmetric_laplace",
                                (mu, left_scale, right_scale),
                                "cusp",
                            ),
                        ),
                        center=mu,
                        scale=max(left_scale, right_scale),
                        parameter_names=(
                            ("mu", "b")
                            if symmetric
                            else ("mu", "b_left", "b_right")
                        ),
                        parameter_values=(
                            (mu, left_scale)
                            if symmetric
                            else (mu, left_scale, right_scale)
                        ),
                        free_shape_parameters=(2 if symmetric else 3),
                    )
                )
    return candidates


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
