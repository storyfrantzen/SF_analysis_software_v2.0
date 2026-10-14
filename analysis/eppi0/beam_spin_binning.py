from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .beam_spin import sine_bin_averages
from .binning import AnalysisBinning


Array = np.ndarray


@dataclass(frozen=True)
class PreparedBinningSample:
    label: str
    q2: Array
    xb: Array
    minus_t: Array
    phi_rad: Array
    denominator_weight: Array
    denominator_variance_weight: Array
    numerator_variance_weight: Array
    signal_region_weight: Array
    background_weight: Array
    plus_signal: Array
    minus_signal: Array
    response_q2: Array | None = None
    response_xb: Array | None = None
    response_minus_t: Array | None = None
    response_phi_rad: Array | None = None
    response_truth: Array | None = None
    response_accepted: Array | None = None

    def __post_init__(self) -> None:
        size = np.asarray(self.q2).size
        names = (
            "q2", "xb", "minus_t", "phi_rad", "denominator_weight",
            "denominator_variance_weight", "numerator_variance_weight",
            "signal_region_weight", "background_weight", "plus_signal",
            "minus_signal",
        )
        if any(np.asarray(getattr(self, name)).shape != (size,) for name in names):
            raise ValueError(f"prepared sample {self.label!r} has inconsistent arrays")
        response = (
            self.response_q2, self.response_xb, self.response_minus_t,
            self.response_phi_rad, self.response_truth, self.response_accepted,
        )
        present = [value is not None for value in response]
        if any(present) and not all(present):
            raise ValueError("response support requires all response arrays")
        if all(present):
            response_size = np.asarray(self.response_q2).size
            if any(np.asarray(value).shape != (response_size,) for value in response):
                raise ValueError("response arrays have inconsistent shapes")


@dataclass(frozen=True)
class BinningThresholds:
    minimum_effective_signal_events: float = 400.0
    maximum_projected_uncertainty: float = 0.10
    minimum_phi_coverage_fraction: float = 0.80
    maximum_phi_gap_deg: float = 60.0
    maximum_background_fraction: float = 0.20
    minimum_response_coverage_fraction: float = 0.80


@dataclass(frozen=True)
class SampleCellMetrics:
    effective_signal_events: Array
    projected_uncertainty: Array
    populated_phi_bins: Array
    maximum_phi_gap_deg: Array
    background_fraction: Array
    response_coverage_fraction: Array
    good: Array


def coarsen_edges(edges: Array, requested_bins: int) -> Array:
    source = np.asarray(edges, dtype=float)
    available = source.size - 1
    if requested_bins < 1 or requested_bins > available:
        raise ValueError(
            f"requested {requested_bins} bins from an axis with {available} bins"
        )
    indices = np.rint(np.linspace(0, available, requested_bins + 1)).astype(int)
    if np.unique(indices).size != indices.size:
        raise RuntimeError("coarsening did not produce unique source edges")
    return source[indices]


def candidate_binnings(
    source: AnalysisBinning,
    q2_bin_counts: list[int],
    xb_bin_counts: list[int],
    t_bin_counts: list[int],
    phi_bin_counts: list[int],
) -> list[tuple[str, AnalysisBinning]]:
    output: list[tuple[str, AnalysisBinning]] = []
    for nq2, nxb, nt, nphi in product(
        sorted(set(q2_bin_counts)),
        sorted(set(xb_bin_counts)),
        sorted(set(t_bin_counts)),
        sorted(set(phi_bin_counts)),
    ):
        if nphi < 4:
            raise ValueError("candidate phi bin counts must be at least four")
        label = f"q{nq2}_x{nxb}_t{nt}_p{nphi}"
        output.append(
            (
                label,
                AnalysisBinning(
                    coarsen_edges(source.q2_edges, nq2),
                    coarsen_edges(source.xb_edges, nxb),
                    coarsen_edges(source.t_edges, nt),
                    np.linspace(0.0, 360.0, nphi + 1),
                ),
            )
        )
    return output


def automatic_bin_counts(number_of_bins: int) -> list[int]:
    """Return compact half, three-quarter, and native-resolution candidates."""
    return sorted(
        {
            max(1, int(np.ceil(number_of_bins / 2.0))),
            max(1, int(np.ceil(3.0 * number_of_bins / 4.0))),
            number_of_bins,
        }
    )


def evaluate_sample(
    sample: PreparedBinningSample,
    binning: AnalysisBinning,
    thresholds: BinningThresholds,
) -> SampleCellMetrics:
    flat = binning.coordinates_to_flat(
        sample.q2, sample.xb, sample.minus_t, sample.phi_rad
    )
    in_range = (flat >= 0) & (flat < binning.size)

    def histogram(values: Array) -> Array:
        raw = np.asarray(values, dtype=float)
        return binning.unflatten(
            np.bincount(
                flat[in_range], weights=raw[in_range], minlength=binning.size
            ).astype(float)
        )

    denominator = histogram(sample.denominator_weight)
    denominator_variance = histogram(sample.denominator_variance_weight)
    numerator_variance = histogram(sample.numerator_variance_weight)
    signal_region = histogram(sample.signal_region_weight)
    background = histogram(sample.background_weight)
    plus = histogram(sample.plus_signal)
    minus = histogram(sample.minus_signal)

    phi_valid = (
        (denominator > 0.0)
        & (numerator_variance > 0.0)
        & (plus > 0.0)
        & (minus > 0.0)
    )
    projected_phi_error = np.divide(
        np.sqrt(numerator_variance),
        denominator,
        out=np.full(denominator.shape, np.inf),
        where=denominator > 0.0,
    )
    basis = sine_bin_averages(binning.phi_edges)
    information = np.sum(
        np.divide(
            basis[None, None, None, :] ** 2,
            projected_phi_error**2,
            out=np.zeros_like(projected_phi_error),
            where=phi_valid & np.isfinite(projected_phi_error),
        ),
        axis=-1,
    )
    projected = np.divide(
        1.0,
        np.sqrt(information),
        out=np.full(information.shape, np.inf),
        where=information > 0.0,
    )
    denominator_3d = np.sum(denominator, axis=-1)
    denominator_variance_3d = np.sum(denominator_variance, axis=-1)
    effective = np.divide(
        denominator_3d**2,
        denominator_variance_3d,
        out=np.zeros_like(denominator_3d),
        where=denominator_variance_3d > 0.0,
    )
    populated = np.count_nonzero(phi_valid, axis=-1)
    maximum_gap = circular_maximum_gap(phi_valid, binning.phi_edges)
    signal_3d = np.sum(signal_region, axis=-1)
    background_3d = np.sum(background, axis=-1)
    background_fraction = np.divide(
        background_3d,
        signal_3d,
        out=np.full(signal_3d.shape, np.inf),
        where=signal_3d > 0.0,
    )

    if sample.response_truth is None:
        response_coverage = np.ones(populated.shape, dtype=float)
    else:
        response_flat = binning.coordinates_to_flat(
            sample.response_q2,
            sample.response_xb,
            sample.response_minus_t,
            sample.response_phi_rad,
        )
        response_in_range = (response_flat >= 0) & (response_flat < binning.size)

        def response_histogram(values: Array) -> Array:
            return binning.unflatten(
                np.bincount(
                    response_flat[response_in_range],
                    weights=np.asarray(values, dtype=float)[response_in_range],
                    minlength=binning.size,
                ).astype(float)
            )

        truth = response_histogram(sample.response_truth)
        accepted = response_histogram(sample.response_accepted)
        response_coverage = np.mean((truth > 0.0) & (accepted > 0.0), axis=-1)

    required_phi = int(
        np.ceil(thresholds.minimum_phi_coverage_fraction * binning.shape[-1])
    )
    good = (
        (effective >= thresholds.minimum_effective_signal_events)
        & (projected <= thresholds.maximum_projected_uncertainty)
        & (populated >= required_phi)
        & (maximum_gap <= thresholds.maximum_phi_gap_deg)
        & (background_fraction <= thresholds.maximum_background_fraction)
        & (
            response_coverage
            >= thresholds.minimum_response_coverage_fraction
        )
    )
    return SampleCellMetrics(
        effective_signal_events=effective,
        projected_uncertainty=projected,
        populated_phi_bins=populated,
        maximum_phi_gap_deg=maximum_gap,
        background_fraction=background_fraction,
        response_coverage_fraction=response_coverage,
        good=good,
    )


def circular_maximum_gap(valid: Array, phi_edges_deg: Array) -> Array:
    mask = np.asarray(valid, dtype=bool)
    edges = np.asarray(phi_edges_deg, dtype=float)
    if mask.ndim < 1 or mask.shape[-1] != edges.size - 1:
        raise ValueError("valid mask and phi edges are incompatible")
    widths = np.diff(edges)
    rows = mask.reshape(-1, mask.shape[-1])
    output = np.empty(rows.shape[0], dtype=float)
    for index, row in enumerate(rows):
        if np.all(row):
            output[index] = 0.0
            continue
        if not np.any(row):
            output[index] = float(np.sum(widths))
            continue
        present = np.flatnonzero(row)
        gaps = []
        for left, right in zip(present, np.r_[present[1:], present[:1] + row.size]):
            missing = np.arange(left + 1, right)
            gaps.append(float(np.sum(widths[missing % row.size])))
        output[index] = max(gaps, default=0.0)
    return output.reshape(mask.shape[:-1])


def pareto_frontier(
    good_cells: Array, median_uncertainty: Array, total_3d_cells: Array
) -> Array:
    good = np.asarray(good_cells, dtype=float)
    error = np.asarray(median_uncertainty, dtype=float)
    total = np.asarray(total_3d_cells, dtype=float)
    frontier = np.ones(good.shape, dtype=bool)
    for index in range(good.size):
        dominates = (
            (good >= good[index])
            & (error <= error[index])
            & (total <= total[index])
            & (
                (good > good[index])
                | (error < error[index])
                | (total < total[index])
            )
        )
        dominates[index] = False
        frontier[index] = not np.any(dominates)
    return frontier
