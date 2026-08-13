"""Charge-normalized RGK data-yield study as a function of beam current."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class RunYield:
    run: int
    run_class: str | None
    current_nA: float | None
    current_quality: str | None
    nominal_current_nA: float | None
    candidate_events: int
    signal_events: int
    charge_c: float
    charge_nC: float
    total_events: int | None
    passed_qadb_events: int | None
    failed_qadb_events: int | None
    yield_events_per_nC: float | None
    statistical_uncertainty_events_per_nC: float | None
    group_charge_fraction: float | None
    included: bool
    exclusion_reason: str


@dataclass
class CurrentGroupYield:
    group: str
    runs: int
    run_numbers: list[int]
    effective_current_nA: float
    signal_events: int
    charge_c: float
    charge_nC: float
    yield_events_per_nC: float
    statistical_uncertainty_events_per_nC: float
    relative_efficiency: float | None = None
    relative_efficiency_uncertainty: float | None = None


@dataclass
class LinearYieldFit:
    intercept_events_per_nC: float
    intercept_uncertainty_events_per_nC: float
    slope_events_per_nC_per_nA: float
    slope_uncertainty_events_per_nC_per_nA: float
    covariance: list[list[float]]
    chi2: float
    ndf: int
    points: int
    fit_level: str

    def predict(self, current_nA: np.ndarray | float) -> np.ndarray:
        current = np.asarray(current_nA, dtype=float)
        return self.intercept_events_per_nC + self.slope_events_per_nC_per_nA * current

    def relative_efficiency(self, current_nA: float) -> tuple[float | None, float | None]:
        intercept = self.intercept_events_per_nC
        slope = self.slope_events_per_nC_per_nA
        if not np.isfinite(intercept) or intercept <= 0.0:
            return None, None
        current = float(current_nA)
        efficiency = 1.0 + current * slope / intercept
        covariance = np.asarray(self.covariance, dtype=float)
        gradient = np.array(
            [-current * slope / intercept**2, current / intercept], dtype=float
        )
        variance = float(gradient @ covariance @ gradient)
        uncertainty = float(np.sqrt(max(variance, 0.0)))
        return float(efficiency), uncertainty


def load_current_manifest(path: Path) -> tuple[dict, dict[int, dict]]:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if "runs" not in manifest or not isinstance(manifest["runs"], dict):
        raise ValueError(f"current manifest has no runs object: {path}")
    runs: dict[int, dict] = {}
    for key, value in manifest["runs"].items():
        try:
            run = int(key)
        except ValueError as exc:
            raise ValueError(f"invalid run number in current manifest: {key}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"manifest entry for run {run} is not an object")
        runs[run] = value
    return manifest, runs


def load_selection_mask(path: Path, expected_size: int, key: str = "mask") -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if key in loaded.files:
                values = loaded[key]
            elif len(loaded.files) == 1:
                values = loaded[loaded.files[0]]
            else:
                raise ValueError(
                    f"selection-mask NPZ has no '{key}' array; available arrays: {loaded.files}"
                )
        finally:
            loaded.close()
    else:
        values = loaded
    mask = np.asarray(values)
    if mask.ndim != 1 or mask.size != expected_size:
        raise ValueError(
            f"selection mask has shape {mask.shape}; expected ({expected_size},)"
        )
    if mask.dtype != np.bool_:
        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError("selection mask must contain only booleans or 0/1 values")
        mask = mask.astype(bool)
    return mask


def _unique_map(keys: np.ndarray, values: np.ndarray, label: str) -> dict[int, float | int]:
    if keys.ndim != 1 or values.ndim != 1 or keys.size != values.size:
        raise ValueError(f"{label} arrays must be one-dimensional and have equal lengths")
    result: dict[int, float | int] = {}
    for raw_key, raw_value in zip(keys, values, strict=True):
        key = int(raw_key)
        if key in result:
            raise ValueError(f"duplicate run {key} in {label} metadata")
        result[key] = raw_value.item() if hasattr(raw_value, "item") else raw_value
    return result


def _optional_run_map(sample, key: str, charge_runs: np.ndarray) -> dict[int, int]:
    if key not in sample.files:
        return {}
    values = np.asarray(sample[key])
    mapped = _unique_map(charge_runs, values, key)
    return {run: int(value) for run, value in mapped.items()}


def build_run_yields(
    sample_path: Path,
    manifest_path: Path,
    *,
    selection_mask_path: Path | None = None,
    selection_mask_key: str = "mask",
    include_classes: Iterable[str] = ("P3", "P4"),
    include_qualities: Iterable[str] = ("unflagged",),
    include_runs: Iterable[int] = (),
    exclude_runs: Iterable[int] = (),
    minimum_group_charge_fraction: float = 0.0,
) -> tuple[list[RunYield], dict]:
    include_class_set = set(include_classes)
    include_quality_set = set(include_qualities)
    include_run_set = {int(run) for run in include_runs}
    exclude_run_set = {int(run) for run in exclude_runs}
    minimum_group_charge_fraction = float(minimum_group_charge_fraction)
    if not 0.0 <= minimum_group_charge_fraction < 1.0:
        raise ValueError("minimum_group_charge_fraction must be in [0,1)")
    if not include_class_set and not include_run_set:
        raise ValueError("at least one included run class or explicit run is required")

    _, manifest_runs = load_current_manifest(manifest_path)
    with np.load(sample_path, allow_pickle=False) as sample:
        required = {"run", "beam_charge_run", "beam_charge_by_run_c"}
        missing = sorted(required.difference(sample.files))
        if missing:
            raise ValueError(f"data sample is missing required arrays: {missing}")
        event_runs = np.asarray(sample["run"], dtype=np.int64)
        if event_runs.ndim != 1:
            raise ValueError("sample run array must be one-dimensional")
        mask = (
            load_selection_mask(selection_mask_path, event_runs.size, selection_mask_key)
            if selection_mask_path is not None
            else np.ones(event_runs.size, dtype=bool)
        )
        charge_runs = np.asarray(sample["beam_charge_run"], dtype=np.int64)
        charges_c = np.asarray(sample["beam_charge_by_run_c"], dtype=float)
        charge_map = {
            run: float(value)
            for run, value in _unique_map(charge_runs, charges_c, "beam charge").items()
        }
        total_events = _optional_run_map(sample, "run_total_events", charge_runs)
        passed_events = _optional_run_map(sample, "run_passed_qadb_events", charge_runs)
        failed_events = _optional_run_map(sample, "run_failed_qadb_events", charge_runs)
        stored_total_charge_c = (
            float(np.asarray(sample["beam_charge_c"]).reshape(-1)[0])
            if "beam_charge_c" in sample.files
            else None
        )

    candidate_counts = Counter(int(run) for run in event_runs)
    signal_counts = Counter(int(run) for run in event_runs[mask])
    missing_charge_runs = sorted(set(candidate_counts).difference(charge_map))
    if missing_charge_runs:
        raise ValueError(
            "selected events have no run-charge metadata for runs: "
            + ", ".join(str(run) for run in missing_charge_runs)
        )

    records: list[RunYield] = []
    all_runs = sorted(set(charge_map).union(candidate_counts))
    for run in all_runs:
        info = manifest_runs.get(run)
        reasons: list[str] = []
        if info is None:
            run_class = None
            current = None
            quality = None
            nominal_current = None
            reasons.append("missing_current_manifest")
        else:
            run_class = info.get("run_class")
            current = _optional_float(info.get("rcdb_current_nA"))
            quality = info.get("rcdb_quality")
            nominal_current = _optional_float(info.get("nominal_current_nA"))
            explicitly_included = run in include_run_set
            if not explicitly_included and run_class not in include_class_set:
                reasons.append("run_class_not_included")
            if quality not in include_quality_set:
                reasons.append("current_quality_not_included")
            if current is None:
                reasons.append("missing_current")
        if run in exclude_run_set:
            reasons.append("explicitly_excluded")

        charge_c = float(charge_map.get(run, 0.0))
        if not np.isfinite(charge_c) or charge_c <= 0.0:
            reasons.append("nonpositive_charge")
        charge_nC = charge_c * 1.0e9
        candidates = int(candidate_counts.get(run, 0))
        signals = int(signal_counts.get(run, 0))
        if charge_nC > 0.0:
            yield_value = signals / charge_nC
            uncertainty = np.sqrt(max(signals, 1)) / charge_nC
        else:
            yield_value = None
            uncertainty = None
        records.append(
            RunYield(
                run=run,
                run_class=run_class,
                current_nA=current,
                current_quality=quality,
                nominal_current_nA=nominal_current,
                candidate_events=candidates,
                signal_events=signals,
                charge_c=charge_c,
                charge_nC=charge_nC,
                total_events=total_events.get(run),
                passed_qadb_events=passed_events.get(run),
                failed_qadb_events=failed_events.get(run),
                yield_events_per_nC=yield_value,
                statistical_uncertainty_events_per_nC=uncertainty,
                group_charge_fraction=None,
                included=not reasons,
                exclusion_reason=";".join(reasons),
            )
        )

    low_contribution_runs = apply_minimum_group_charge_fraction(
        records, minimum_group_charge_fraction
    )

    run_charge_sum_c = float(sum(charge_map.values()))
    charge_difference_c = (
        run_charge_sum_c - stored_total_charge_c
        if stored_total_charge_c is not None
        else None
    )
    validation = {
        "candidate_events": int(event_runs.size),
        "signal_events": int(mask.sum()),
        "charge_rows": len(charge_map),
        "run_charge_sum_c": run_charge_sum_c,
        "stored_total_charge_c": stored_total_charge_c,
        "charge_difference_c": charge_difference_c,
        "selected_runs_missing_charge": missing_charge_runs,
        "charge_runs_missing_manifest": sorted(set(charge_map).difference(manifest_runs)),
        "minimum_group_charge_fraction": minimum_group_charge_fraction,
        "runs_below_minimum_group_charge_fraction": low_contribution_runs,
    }
    return records, validation


def apply_minimum_group_charge_fraction(
    records: Iterable[RunYield], minimum_fraction: float
) -> list[int]:
    """Reject otherwise eligible runs carrying too little of their group's charge.

    Fractions are calculated once from all runs that pass the ordinary class,
    current-quality, and explicit-run filters.  This preserves an auditable
    denominator and avoids order-dependent iterative removal.
    """
    minimum_fraction = float(minimum_fraction)
    if not 0.0 <= minimum_fraction < 1.0:
        raise ValueError("minimum_fraction must be in [0,1)")
    grouped: dict[str, list[RunYield]] = {}
    for record in records:
        if record.included:
            if record.run_class is None:
                raise ValueError(f"included run {record.run} has no run class")
            grouped.setdefault(record.run_class, []).append(record)

    rejected: list[int] = []
    for name, members in grouped.items():
        total_charge = float(sum(member.charge_c for member in members))
        if total_charge <= 0.0:
            raise ValueError(f"current group {name} has nonpositive eligible charge")
        for member in members:
            fraction = member.charge_c / total_charge
            member.group_charge_fraction = float(fraction)
            if fraction < minimum_fraction:
                member.included = False
                reason = "below_minimum_group_charge_fraction"
                member.exclusion_reason = (
                    f"{member.exclusion_reason};{reason}"
                    if member.exclusion_reason
                    else reason
                )
                rejected.append(member.run)
    return sorted(rejected)


def aggregate_current_groups(records: Iterable[RunYield]) -> list[CurrentGroupYield]:
    grouped: dict[str, list[RunYield]] = {}
    for record in records:
        if record.included:
            if record.run_class is None or record.current_nA is None:
                raise ValueError(f"included run {record.run} has incomplete current metadata")
            grouped.setdefault(record.run_class, []).append(record)

    groups: list[CurrentGroupYield] = []
    for name, members in grouped.items():
        charge_c = float(sum(member.charge_c for member in members))
        charge_nC = charge_c * 1.0e9
        if charge_nC <= 0.0:
            raise ValueError(f"current group {name} has nonpositive charge")
        signals = int(sum(member.signal_events for member in members))
        current = float(
            sum(member.charge_c * float(member.current_nA) for member in members)
            / charge_c
        )
        groups.append(
            CurrentGroupYield(
                group=name,
                runs=len(members),
                run_numbers=[member.run for member in members],
                effective_current_nA=current,
                signal_events=signals,
                charge_c=charge_c,
                charge_nC=charge_nC,
                yield_events_per_nC=signals / charge_nC,
                statistical_uncertainty_events_per_nC=np.sqrt(max(signals, 1))
                / charge_nC,
            )
        )
    groups.sort(key=lambda group: group.effective_current_nA)
    return groups


def fit_linear_yield(
    records: Iterable[RunYield],
    groups: Iterable[CurrentGroupYield],
    *,
    fit_level: str = "groups",
) -> LinearYieldFit:
    if fit_level == "groups":
        points = list(groups)
        current = np.asarray([point.effective_current_nA for point in points], dtype=float)
        values = np.asarray([point.yield_events_per_nC for point in points], dtype=float)
        uncertainties = np.asarray(
            [point.statistical_uncertainty_events_per_nC for point in points], dtype=float
        )
    elif fit_level == "runs":
        run_points = [record for record in records if record.included]
        points = run_points
        current = np.asarray([record.current_nA for record in run_points], dtype=float)
        values = np.asarray([record.yield_events_per_nC for record in run_points], dtype=float)
        uncertainties = np.asarray(
            [record.statistical_uncertainty_events_per_nC for record in run_points],
            dtype=float,
        )
    else:
        raise ValueError("fit_level must be 'groups' or 'runs'")

    if current.size < 2:
        raise ValueError("at least two included fit points are required")
    if np.unique(current).size < 2:
        raise ValueError("fit points must contain at least two distinct currents")
    if not (
        np.all(np.isfinite(current))
        and np.all(np.isfinite(values))
        and np.all(np.isfinite(uncertainties))
        and np.all(uncertainties > 0.0)
    ):
        raise ValueError("fit inputs contain invalid values or uncertainties")

    design = np.column_stack((np.ones(current.size), current))
    inverse_variance = 1.0 / uncertainties**2
    normal = design.T @ (inverse_variance[:, None] * design)
    covariance = np.linalg.inv(normal)
    parameters = covariance @ (design.T @ (inverse_variance * values))
    residual = values - design @ parameters
    chi2 = float(np.sum((residual / uncertainties) ** 2))
    return LinearYieldFit(
        intercept_events_per_nC=float(parameters[0]),
        intercept_uncertainty_events_per_nC=float(np.sqrt(covariance[0, 0])),
        slope_events_per_nC_per_nA=float(parameters[1]),
        slope_uncertainty_events_per_nC_per_nA=float(np.sqrt(covariance[1, 1])),
        covariance=covariance.tolist(),
        chi2=chi2,
        ndf=int(current.size - 2),
        points=int(current.size),
        fit_level=fit_level,
    )


def attach_relative_efficiencies(
    groups: Iterable[CurrentGroupYield], fit: LinearYieldFit
) -> None:
    for group in groups:
        efficiency, uncertainty = fit.relative_efficiency(group.effective_current_nA)
        group.relative_efficiency = efficiency
        group.relative_efficiency_uncertainty = uncertainty


def run_yield_rows(records: Iterable[RunYield]) -> list[dict]:
    return [asdict(record) for record in records]


def current_group_rows(groups: Iterable[CurrentGroupYield]) -> list[dict]:
    rows = []
    for group in groups:
        row = asdict(group)
        row["run_numbers"] = ";".join(str(run) for run in group.run_numbers)
        rows.append(row)
    return rows


def _optional_float(value) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None
