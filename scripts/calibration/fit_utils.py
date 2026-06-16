from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BinnedProfile:
    centers: np.ndarray
    means: np.ndarray
    errors: np.ndarray
    counts: np.ndarray


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def adaptive_edges(values: np.ndarray, n_bins: int, value_range: tuple[float, float]) -> np.ndarray:
    lo, hi = value_range
    selected = values[np.isfinite(values) & (values >= lo) & (values <= hi)]
    if selected.size < n_bins + 1:
        return np.linspace(lo, hi, n_bins + 1)
    edges = np.quantile(selected, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = lo
    edges[-1] = hi
    return np.unique(edges)


def fixed_edges(n_bins: int, value_range: tuple[float, float]) -> np.ndarray:
    return np.linspace(value_range[0], value_range[1], n_bins + 1)


def binned_profile(x: np.ndarray,
                   y: np.ndarray,
                   edges: np.ndarray,
                   min_entries: int = 20) -> BinnedProfile:
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full_like(centers, np.nan, dtype=float)
    errors = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == len(edges) - 2:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        vals = y[mask & np.isfinite(y)]
        counts[i] = vals.size
        if vals.size < min_entries:
            continue
        means[i] = float(np.mean(vals))
        errors[i] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan

    return BinnedProfile(centers=centers, means=means, errors=errors, counts=counts)


def weighted_polyfit_ascending(x: np.ndarray,
                               y: np.ndarray,
                               degree: int,
                               yerr: np.ndarray | None = None) -> list[float]:
    mask = finite_mask(x, y)
    if yerr is not None:
        mask &= np.isfinite(yerr) & (yerr > 0)
    if np.count_nonzero(mask) < degree + 1:
        return [float("nan")] * (degree + 1)

    weights = None if yerr is None else 1.0 / yerr[mask]
    descending = np.polyfit(x[mask], y[mask], degree, w=weights)
    return [float(v) for v in descending[::-1]]


def weighted_polyfit_descending(x: np.ndarray,
                                y: np.ndarray,
                                degree: int,
                                yerr: np.ndarray | None = None) -> list[float]:
    mask = finite_mask(x, y)
    if yerr is not None:
        mask &= np.isfinite(yerr) & (yerr > 0)
    if np.count_nonzero(mask) < degree + 1:
        return [float("nan")] * (degree + 1)

    weights = None if yerr is None else 1.0 / yerr[mask]
    return [float(v) for v in np.polyfit(x[mask], y[mask], degree, w=weights)]


def design_matrix(p: np.ndarray, form: str) -> np.ndarray:
    if form == "[0] + [1]/p + [2]/(p^2)":
        return np.column_stack([np.ones_like(p), 1.0 / p, 1.0 / (p * p)])
    if form == "[0] + [1]/p":
        return np.column_stack([np.ones_like(p), 1.0 / p])
    if form == "[0] + [1]*p + [2]*p^2":
        return np.column_stack([np.ones_like(p), p, p * p])
    if form == "[0] + [1]*p":
        return np.column_stack([np.ones_like(p), p])
    raise ValueError(f"Unsupported correction form: {form}")


def fit_linear_form(p: np.ndarray,
                    y: np.ndarray,
                    form: str,
                    yerr: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    mask = finite_mask(p, y) & (p != 0)
    if yerr is not None:
        mask &= np.isfinite(yerr) & (yerr > 0)
    x = design_matrix(p[mask], form)
    if x.shape[0] < x.shape[1]:
        return np.full(x.shape[1], np.nan), np.full(x.shape[1], np.nan)

    if yerr is None:
        beta, *_ = np.linalg.lstsq(x, y[mask], rcond=None)
        return beta, np.full_like(beta, np.nan, dtype=float)

    weights = 1.0 / yerr[mask]
    xw = x * weights[:, None]
    yw = y[mask] * weights
    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    residual = yw - xw @ beta
    ndof = max(1, x.shape[0] - x.shape[1])
    sigma2 = float(np.sum(residual * residual) / ndof)
    cov = sigma2 * np.linalg.pinv(xw.T @ xw)
    return beta, np.sqrt(np.diag(cov))


def json_ready(values: Iterable[float]) -> list[float]:
    ready: list[float] = []
    for value in values:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Cannot export nonfinite calibration coefficient")
        ready.append(value)
    return ready
