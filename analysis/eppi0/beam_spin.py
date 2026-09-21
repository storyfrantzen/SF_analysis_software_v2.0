from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class HelicityAudit:
    runs: Array
    plus_charge_nc: Array
    minus_charge_nc: Array
    interval_runs: Array
    interval_min: Array
    interval_max: Array
    interval_pass: Array
    interval_sign: Array


@dataclass(frozen=True)
class PolarizationTable:
    runs: Array
    values: Array
    uncertainties: Array
    labels: Array
    payload: dict


@dataclass(frozen=True)
class BeamSpinResult:
    asymmetry: Array
    statistical_uncertainty: Array
    polarization_uncertainty: Array
    polarization_period_labels: Array
    polarization_asymmetry_shifts: Array
    valid: Array
    numerator: Array
    denominator: Array
    numerator_variance: Array
    denominator_variance: Array
    numerator_denominator_covariance: Array
    plus_yield: Array
    minus_yield: Array
    plus_charge_balanced_yield: Array
    minus_charge_balanced_yield: Array
    plus_event_count: Array
    minus_event_count: Array
    event_count: Array
    used_event_count: int
    rejected_helicity_event_count: int
    rejected_run_event_count: int


@dataclass(frozen=True)
class SineFitResult:
    amplitude: Array
    uncertainty: Array
    polarization_uncertainty: Array
    chi2: Array
    ndof: Array
    point_count: Array
    valid: Array
    quality: Array


def load_helicity_audit(directory: str | Path) -> HelicityAudit:
    directory = Path(directory)
    with (directory / "run_helicity_charge.tsv").open(
        newline="", encoding="utf-8"
    ) as source:
        run_rows = list(csv.DictReader(source, delimiter="\t"))
    with (directory / "helicity_sign_intervals.tsv").open(
        newline="", encoding="utf-8"
    ) as source:
        interval_rows = list(csv.DictReader(source, delimiter="\t"))
    if not run_rows or not interval_rows:
        raise ValueError(f"helicity audit is empty: {directory}")
    usable = [
        row
        for row in run_rows
        if int(row["run_sign"]) != 0
        and float(row["physical_plus_charge_nC"]) > 0.0
        and float(row["physical_minus_charge_nC"]) > 0.0
    ]
    if not usable:
        raise ValueError(f"helicity audit has no runs with usable +/- charge: {directory}")
    return HelicityAudit(
        runs=np.asarray([int(row["run"]) for row in usable], dtype=np.int64),
        plus_charge_nc=np.asarray(
            [float(row["physical_plus_charge_nC"]) for row in usable]
        ),
        minus_charge_nc=np.asarray(
            [float(row["physical_minus_charge_nC"]) for row in usable]
        ),
        interval_runs=np.asarray([int(row["run"]) for row in interval_rows], dtype=np.int64),
        interval_min=np.asarray(
            [int(row["event_min"]) for row in interval_rows], dtype=np.int64
        ),
        interval_max=np.asarray(
            [int(row["event_max"]) for row in interval_rows], dtype=np.int64
        ),
        interval_pass=np.asarray(
            [bool(int(row["qadb_pass"])) for row in interval_rows], dtype=bool
        ),
        interval_sign=np.asarray(
            [int(row["helicity_sign"]) for row in interval_rows], dtype=np.int8
        ),
    )


def parse_run_specification(value: object) -> set[int]:
    if isinstance(value, list):
        return {int(run) for run in value}
    if not isinstance(value, str):
        raise ValueError("polarization period runs must be a list or range string")
    output: set[int] = set()
    for token in value.replace(",", " ").split():
        if "-" not in token:
            output.add(int(token))
            continue
        first_text, last_text = token.split("-", 1)
        first, last = int(first_text), int(last_text)
        if last < first:
            raise ValueError(f"descending run range: {token}")
        output.update(range(first, last + 1))
    return output


def load_polarization_manifest(path: str | Path) -> PolarizationTable:
    path = Path(path)
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported polarization manifest schema: {path}")
    if not isinstance(payload.get("source"), str) or not payload["source"].strip():
        raise ValueError(f"polarization manifest requires a nonempty source: {path}")
    periods = payload.get("periods")
    if not isinstance(periods, list) or not periods:
        raise ValueError(f"polarization manifest has no periods: {path}")
    by_run: dict[int, tuple[float, float, str]] = {}
    for period in periods:
        value = float(period["polarization"])
        uncertainty = float(period["uncertainty"])
        label = str(period["label"])
        if not 0.0 < value <= 1.0:
            raise ValueError(f"polarization must be a positive magnitude: {label}")
        if not 0.0 <= uncertainty < value:
            raise ValueError(f"invalid polarization uncertainty: {label}")
        for run in parse_run_specification(period["runs"]):
            if run in by_run:
                raise ValueError(f"run {run} occurs in multiple polarization periods")
            by_run[run] = (value, uncertainty, label)
    runs = np.asarray(sorted(by_run), dtype=np.int64)
    return PolarizationTable(
        runs=runs,
        values=np.asarray([by_run[int(run)][0] for run in runs]),
        uncertainties=np.asarray([by_run[int(run)][1] for run in runs]),
        labels=np.asarray([by_run[int(run)][2] for run in runs]),
        payload=payload,
    )


def values_by_run(query_runs: Array, runs: Array, values: Array, *, name: str) -> Array:
    query = np.asarray(query_runs, dtype=np.int64)
    keys = np.asarray(runs, dtype=np.int64)
    order = np.argsort(keys)
    keys = keys[order]
    source = np.asarray(values)[order]
    indices = np.searchsorted(keys, query)
    found = (indices < keys.size) & (keys[np.minimum(indices, keys.size - 1)] == query)
    output = np.full(query.shape, np.nan, dtype=float)
    output[found] = source[indices[found]]
    if np.any(~found):
        missing = np.unique(query[~found])
        raise ValueError(f"{name} is missing runs: {missing.tolist()}")
    return output


def corrected_helicity(
    event_runs: Array, event_numbers: Array, raw_helicity: Array, audit: HelicityAudit
) -> tuple[Array, Array]:
    event_runs = np.asarray(event_runs, dtype=np.int64)
    event_numbers = np.asarray(event_numbers, dtype=np.int64)
    raw = np.asarray(raw_helicity, dtype=np.int64)
    if not (event_runs.shape == event_numbers.shape == raw.shape):
        raise ValueError("run, event, and helicity arrays must have equal shapes")
    corrected = np.zeros(raw.shape, dtype=np.int8)
    accepted = np.zeros(raw.shape, dtype=bool)
    for run in np.unique(event_runs):
        event_rows = np.flatnonzero(event_runs == run)
        interval_rows = np.flatnonzero(audit.interval_runs == run)
        if interval_rows.size == 0:
            continue
        order = np.argsort(audit.interval_min[interval_rows])
        rows = interval_rows[order]
        starts = audit.interval_min[rows]
        position = np.searchsorted(starts, event_numbers[event_rows], side="right") - 1
        present = position >= 0
        candidate = np.clip(position, 0, rows.size - 1)
        matched_rows = rows[candidate]
        present &= event_numbers[event_rows] <= audit.interval_max[matched_rows]
        present &= audit.interval_pass[matched_rows]
        signs = audit.interval_sign[matched_rows]
        present &= signs != 0
        local = np.zeros(event_rows.size, dtype=np.int8)
        local[present] = (raw[event_rows[present]] * signs[present]).astype(np.int8)
        corrected[event_rows] = local
        accepted[event_rows[present]] = True
    accepted &= np.isin(corrected, (-1, 1))
    corrected[~accepted] = 0
    return corrected, accepted


def extract_beam_spin(
    *,
    flat_bins: Array,
    event_runs: Array,
    event_numbers: Array,
    raw_helicity: Array,
    net_event_weights: Array,
    active_events: Array,
    audit: HelicityAudit,
    polarization: PolarizationTable,
    number_of_bins: int,
) -> BeamSpinResult:
    flat = np.asarray(flat_bins, dtype=np.int64)
    runs = np.asarray(event_runs, dtype=np.int64)
    net = np.asarray(net_event_weights, dtype=float)
    active = np.asarray(active_events, dtype=bool)
    if not (flat.shape == runs.shape == net.shape == active.shape):
        raise ValueError("beam-spin event arrays must have equal shapes")
    helicity, helicity_ok = corrected_helicity(
        runs, event_numbers, raw_helicity, audit
    )
    included_runs = np.intersect1d(audit.runs, polarization.runs)
    in_range = (flat >= 0) & (flat < number_of_bins)
    required_runs = np.unique(runs[active & in_range])
    missing_runs = np.setdiff1d(required_runs, included_runs)
    if missing_runs.size:
        raise ValueError(
            "active events have no usable helicity charge or polarization for runs: "
            + ", ".join(str(int(run)) for run in missing_runs)
        )
    run_ok = np.isin(runs, included_runs)
    used = active & in_range & helicity_ok & run_ok
    if not np.any(used):
        raise ValueError("no events survive helicity, run, and analysis-bin requirements")

    used_runs = runs[used]
    qplus = values_by_run(
        used_runs, audit.runs, audit.plus_charge_nc, name="positive-helicity charge"
    )
    qminus = values_by_run(
        used_runs, audit.runs, audit.minus_charge_nc, name="negative-helicity charge"
    )
    p = values_by_run(
        used_runs, polarization.runs, polarization.values, name="polarization"
    )
    if np.any((qplus <= 0.0) | (qminus <= 0.0) | (p <= 0.0)):
        raise ValueError("usable events require positive helicity charges and polarization")
    h = helicity[used].astype(float)
    charge_for_state = np.where(h > 0.0, qplus, qminus)
    balanced_charge = 0.5 * (qplus + qminus)
    normalization = balanced_charge / charge_for_state
    weights = net[used]
    denominator_weight = weights * normalization
    numerator_weight = denominator_weight * h / p
    bins = flat[used]

    def histogram(weight: Array) -> Array:
        return np.bincount(bins, weights=weight, minlength=number_of_bins).astype(float)

    numerator = histogram(numerator_weight)
    denominator = histogram(denominator_weight)
    numerator_variance = histogram(numerator_weight**2)
    denominator_variance = histogram(denominator_weight**2)
    covariance = histogram(numerator_weight * denominator_weight)
    plus_yield = histogram(weights * (h > 0.0))
    minus_yield = histogram(weights * (h < 0.0))
    plus_balanced = histogram(denominator_weight * (h > 0.0))
    minus_balanced = histogram(denominator_weight * (h < 0.0))
    plus_event_count = histogram((h > 0.0).astype(float)).astype(np.int64)
    minus_event_count = histogram((h < 0.0).astype(float)).astype(np.int64)
    event_count = plus_event_count + minus_event_count

    asymmetry = np.divide(
        numerator,
        denominator,
        out=np.full(number_of_bins, np.nan),
        where=denominator > 0.0,
    )
    variance = np.full(number_of_bins, np.nan)
    positive = denominator > 0.0
    variance[positive] = (
        numerator_variance[positive] / denominator[positive] ** 2
        + numerator[positive] ** 2
        * denominator_variance[positive]
        / denominator[positive] ** 4
        - 2.0
        * numerator[positive]
        * covariance[positive]
        / denominator[positive] ** 3
    )
    variance = np.maximum(variance, 0.0)
    uncertainty = np.sqrt(variance)
    valid = positive & (plus_event_count > 0) & (minus_event_count > 0)
    valid &= np.isfinite(asymmetry) & np.isfinite(uncertainty) & (uncertainty > 0.0)

    # Treat each manifest period's uncertainty as correlated within that period
    # and independent between periods. Re-evaluate the numerator after shifting
    # one period at a time; the denominator is polarization independent.
    polarization_variance = np.zeros(number_of_bins, dtype=float)
    used_labels = values_by_run(
        used_runs,
        polarization.runs,
        np.arange(polarization.runs.size, dtype=float),
        name="polarization manifest",
    ).astype(np.int64)
    period_labels = np.unique(polarization.labels[used_labels])
    period_shifts = np.zeros((period_labels.size, number_of_bins), dtype=float)
    for period_index, label in enumerate(period_labels):
        manifest_rows = polarization.labels == label
        value = float(np.unique(polarization.values[manifest_rows]).item())
        error = float(np.unique(polarization.uncertainties[manifest_rows]).item())
        if error == 0.0:
            continue
        event_period = polarization.labels[used_labels] == label
        period_component = histogram(numerator_weight * event_period)
        derivative = -period_component / value
        contribution = np.divide(
            derivative * error,
            denominator,
            out=np.zeros(number_of_bins, dtype=float),
            where=denominator > 0.0,
        )
        period_shifts[period_index] = contribution
        polarization_variance += contribution**2
    polarization_uncertainty = np.sqrt(polarization_variance)
    polarization_uncertainty[~positive] = np.nan

    return BeamSpinResult(
        asymmetry=asymmetry,
        statistical_uncertainty=uncertainty,
        polarization_uncertainty=polarization_uncertainty,
        polarization_period_labels=period_labels,
        polarization_asymmetry_shifts=period_shifts,
        valid=valid,
        numerator=numerator,
        denominator=denominator,
        numerator_variance=numerator_variance,
        denominator_variance=denominator_variance,
        numerator_denominator_covariance=covariance,
        plus_yield=plus_yield,
        minus_yield=minus_yield,
        plus_charge_balanced_yield=plus_balanced,
        minus_charge_balanced_yield=minus_balanced,
        plus_event_count=plus_event_count,
        minus_event_count=minus_event_count,
        event_count=event_count,
        used_event_count=int(np.count_nonzero(used)),
        rejected_helicity_event_count=int(np.count_nonzero(active & ~helicity_ok)),
        rejected_run_event_count=int(np.count_nonzero(active & helicity_ok & ~run_ok)),
    )


def fit_sine_amplitudes(
    asymmetry: Array,
    uncertainty: Array,
    valid: Array,
    phi_centers_deg: Array,
    *,
    phi_edges_deg: Array | None = None,
    polarization_shifts: Array | None = None,
    minimum_points: int = 8,
    maximum_chi2_ndf: float = 5.0,
    maximum_uncertainty: float = 0.25,
) -> SineFitResult:
    values = np.asarray(asymmetry, dtype=float)
    errors = np.asarray(uncertainty, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 4 or values.shape != errors.shape or values.shape != mask.shape:
        raise ValueError("asymmetry arrays must have equal (Q2, xB, t, phi) shapes")
    phi = np.asarray(phi_centers_deg, dtype=float)
    if phi.shape != (values.shape[-1],):
        raise ValueError("phi centers do not match asymmetry phi dimension")
    output_shape = values.shape[:-1]
    amplitude = np.full(output_shape, np.nan)
    amplitude_error = np.full(output_shape, np.nan)
    amplitude_polarization_error = np.full(output_shape, np.nan)
    chi2 = np.full(output_shape, np.nan)
    ndof = np.zeros(output_shape, dtype=np.int64)
    points = np.zeros(output_shape, dtype=np.int64)
    fit_valid = np.zeros(output_shape, dtype=bool)
    quality = np.zeros(output_shape, dtype=bool)
    sine = (
        sine_bin_averages(phi_edges_deg)
        if phi_edges_deg is not None
        else np.sin(np.deg2rad(phi))
    )
    if sine.shape != (values.shape[-1],):
        raise ValueError("phi edges do not match asymmetry phi dimension")
    if polarization_shifts is not None:
        polarization_shifts = np.asarray(polarization_shifts, dtype=float)
        if polarization_shifts.ndim != 5 or polarization_shifts.shape[1:] != values.shape:
            raise ValueError(
                "polarization shifts must have shape (period, Q2, xB, t, phi)"
            )
    for index in np.ndindex(output_shape):
        keep = mask[index] & np.isfinite(values[index]) & np.isfinite(errors[index])
        keep &= errors[index] > 0.0
        points[index] = int(np.count_nonzero(keep))
        if points[index] < minimum_points:
            continue
        weight = 1.0 / errors[index][keep] ** 2
        denominator = float(np.sum(weight * sine[keep] ** 2))
        if denominator <= 0.0:
            continue
        fitted = float(np.sum(weight * sine[keep] * values[index][keep]) / denominator)
        error = float(np.sqrt(1.0 / denominator))
        residual = values[index][keep] - fitted * sine[keep]
        value_chi2 = float(np.sum(weight * residual**2))
        value_ndof = points[index] - 1
        amplitude[index] = fitted
        amplitude_error[index] = error
        if polarization_shifts is not None:
            coefficients = weight * sine[keep] / denominator
            period_amplitude_shifts = np.sum(
                polarization_shifts[(slice(None),) + index][:, keep]
                * coefficients[None, :],
                axis=1,
            )
            amplitude_polarization_error[index] = float(
                np.sqrt(np.sum(period_amplitude_shifts**2))
            )
        chi2[index] = value_chi2
        ndof[index] = value_ndof
        fit_valid[index] = True
        quality[index] = (
            value_ndof > 0
            and value_chi2 / value_ndof <= maximum_chi2_ndf
            and error <= maximum_uncertainty
            and abs(fitted) <= 1.0
        )
    return SineFitResult(
        amplitude=amplitude,
        uncertainty=amplitude_error,
        polarization_uncertainty=amplitude_polarization_error,
        chi2=chi2,
        ndof=ndof,
        point_count=points,
        valid=fit_valid,
        quality=quality,
    )


def sine_bin_averages(phi_edges_deg: Array) -> Array:
    """Return the exact uniform-in-phi average of ``sin(phi)`` in each bin."""
    edges = np.asarray(phi_edges_deg, dtype=float)
    if (
        edges.ndim != 1
        or edges.size < 2
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("phi edges must be a finite, strictly increasing 1D array")
    radians = np.deg2rad(edges)
    return np.divide(
        np.cos(radians[:-1]) - np.cos(radians[1:]),
        np.diff(radians),
    )
