"""Charge-normalized data-yield study as a function of beam current."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .background_subtraction import METHOD as BACKGROUND_METHOD
from .background_subtraction import estimate_mgg_background
from .exclusivity import load_cuts


@dataclass
class RunYield:
    run: int
    run_class: str | None
    current_nA: float | None
    current_quality: str | None
    nominal_current_nA: float | None
    candidate_events: int
    signal_events: float
    signal_statistical_variance: float
    signal_region_events: int
    sideband_events: int
    estimated_background_events: float
    charge_c: float
    charge_nC: float
    total_events: int | None
    passed_qadb_events: int | None
    failed_qadb_events: int | None
    yield_events_per_nC: float | None
    statistical_uncertainty_events_per_nC: float | None
    group_charge_fraction: float | None
    group_yield_pull: float | None
    included: bool
    exclusion_reason: str


@dataclass
class CurrentGroupYield:
    group: str
    runs: int
    run_numbers: list[int]
    effective_current_nA: float
    signal_events: float
    signal_statistical_variance: float
    signal_region_events: int
    sideband_events: int
    estimated_background_events: float
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
    fit_model: str = "single_intercept_absolute_slope"

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


@dataclass
class SharedFractionalYieldFit:
    """One zero-current normalization per period and one fractional current slope."""

    period_intercepts_events_per_nC: dict[str, float]
    period_intercept_uncertainties_events_per_nC: dict[str, float]
    fractional_slope_per_nA: float
    fractional_slope_uncertainty_per_nA: float
    parameter_names: list[str]
    covariance: list[list[float]]
    period_classes: dict[str, list[str]]
    class_periods: dict[str, str]
    chi2: float
    ndf: int
    points: int
    fit_level: str
    fit_model: str = "shared_fractional_slope_separate_intercepts"

    def period_for_class(self, run_class: str) -> str:
        try:
            return self.class_periods[run_class]
        except KeyError as exc:
            raise ValueError(
                f"run class '{run_class}' is not assigned to a shared-slope period"
            ) from exc

    def intercept_for_class(self, run_class: str) -> float:
        return self.period_intercepts_events_per_nC[
            self.period_for_class(run_class)
        ]

    def predict(self, current_nA: np.ndarray | float, *, period: str) -> np.ndarray:
        current = np.asarray(current_nA, dtype=float)
        try:
            intercept = self.period_intercepts_events_per_nC[period]
        except KeyError as exc:
            raise ValueError(f"unknown shared-slope period '{period}'") from exc
        return intercept * (1.0 + self.fractional_slope_per_nA * current)

    def prediction_uncertainty(
        self, current_nA: np.ndarray | float, *, period: str
    ) -> np.ndarray:
        current = np.atleast_1d(np.asarray(current_nA, dtype=float))
        try:
            period_index = self.parameter_names.index(f"intercept:{period}")
            intercept = self.period_intercepts_events_per_nC[period]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown shared-slope period '{period}'") from exc
        beta_index = self.parameter_names.index("fractional_slope_per_nA")
        gradient = np.zeros((current.size, len(self.parameter_names)), dtype=float)
        gradient[:, period_index] = 1.0 + self.fractional_slope_per_nA * current
        gradient[:, beta_index] = intercept * current
        covariance = np.asarray(self.covariance, dtype=float)
        variance = np.einsum("ij,jk,ik->i", gradient, covariance, gradient)
        result = np.sqrt(np.maximum(variance, 0.0))
        return result if np.ndim(current_nA) else result[0]

    def relative_efficiency(
        self, current_nA: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        current = np.asarray(current_nA, dtype=float)
        efficiency = 1.0 + self.fractional_slope_per_nA * current
        uncertainty = np.abs(current) * self.fractional_slope_uncertainty_per_nA
        if np.ndim(current_nA):
            return efficiency, uncertainty
        return float(efficiency), float(uncertainty)


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


def _sum_by_run(runs: np.ndarray, values: np.ndarray) -> dict[int, float]:
    runs = np.asarray(runs, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if runs.ndim != 1 or values.shape != runs.shape:
        raise ValueError("per-run accumulation arrays must be one-dimensional and aligned")
    unique_runs, inverse = np.unique(runs, return_inverse=True)
    totals = np.bincount(inverse, weights=values, minlength=unique_runs.size)
    return {
        int(run): float(total)
        for run, total in zip(unique_runs, totals, strict=True)
    }


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
    low_yield_sigma_threshold: float = 5.0,
    background_cuts_path: Path | None = None,
    background_alpha_bootstrap: int = 200,
    background_seed: int | None = 12345,
) -> tuple[list[RunYield], dict]:
    include_class_set = set(include_classes)
    include_quality_set = set(include_qualities)
    include_run_set = {int(run) for run in include_runs}
    exclude_run_set = {int(run) for run in exclude_runs}
    minimum_group_charge_fraction = float(minimum_group_charge_fraction)
    if not 0.0 <= minimum_group_charge_fraction < 1.0:
        raise ValueError("minimum_group_charge_fraction must be in [0,1)")
    low_yield_sigma_threshold = float(low_yield_sigma_threshold)
    if not np.isfinite(low_yield_sigma_threshold) or low_yield_sigma_threshold < 0.0:
        raise ValueError("low_yield_sigma_threshold must be finite and nonnegative")
    if not include_class_set and not include_run_set:
        raise ValueError("at least one included run class or explicit run is required")

    manifest, manifest_runs = load_current_manifest(manifest_path)
    with np.load(sample_path, allow_pickle=False) as sample:
        required = {"run", "beam_charge_run", "beam_charge_by_run_c"}
        missing = sorted(required.difference(sample.files))
        if missing:
            raise ValueError(f"data sample is missing required arrays: {missing}")
        event_runs = np.asarray(sample["run"], dtype=np.int64)
        if event_runs.ndim != 1:
            raise ValueError("sample run array must be one-dimensional")
        fixed_mask = (
            load_selection_mask(selection_mask_path, event_runs.size, selection_mask_key)
            if selection_mask_path is not None
            else np.ones(event_runs.size, dtype=bool)
        )
        background_metadata = None
        if background_cuts_path is not None:
            cuts = load_cuts(str(background_cuts_path))
            background_required = {
                *cuts.variables,
                "rec_proton_detector",
                "rec_ft_photon_count",
            }
            background_missing = sorted(background_required.difference(sample.files))
            if background_missing:
                raise ValueError(
                    "background-subtracted current study requires compact-data arrays: "
                    + ", ".join(background_missing)
                )
            base_mask = (
                np.asarray(sample["rec_selected"], dtype=bool)
                if "rec_selected" in sample.files
                else np.ones(event_runs.size, dtype=bool)
            )
            if base_mask.shape != event_runs.shape:
                raise ValueError("sample rec_selected array must match the run array")
            dummy_index = np.zeros(event_runs.size, dtype=np.int64)
            background = estimate_mgg_background(
                cuts=cuts,
                values={name: np.asarray(sample[name]) for name in cuts.variables},
                proton_detector=np.asarray(sample["rec_proton_detector"]),
                ft_photons=np.asarray(sample["rec_ft_photon_count"]),
                iq2=dummy_index,
                ixb=dummy_index,
                it=dummy_index,
                rec_flat=dummy_index,
                base_mask=base_mask,
                event_weights=np.ones(event_runs.size, dtype=float),
                number_of_bins=1,
                alpha_bootstrap=background_alpha_bootstrap,
                seed=background_seed,
            )
            if selection_mask_path is not None and not np.array_equal(
                base_mask & fixed_mask, background.signal_region_mask
            ):
                disagreement = int(
                    np.count_nonzero(
                        (base_mask & fixed_mask) != background.signal_region_mask
                    )
                )
                raise ValueError(
                    "--selection-mask disagrees with the signal region derived from "
                    f"--background-cuts for {disagreement} events"
                )
            signal_region_mask = background.signal_region_mask
            sideband_mask = background.sideband_mask
            signal_event_weights = background.net_event_weights
            signal_variance_weights = signal_event_weights**2
            yield_mode = "mgg_sideband_subtracted"
            background_metadata = {
                "method": BACKGROUND_METHOD,
                "cuts": str(Path(background_cuts_path).resolve()),
                "alpha_bootstrap": int(background_alpha_bootstrap),
                "seed": background_seed,
                "group_ids": background.group_ids.tolist(),
                "signal_lower": background.signal_lower.tolist(),
                "signal_upper": background.signal_upper.tolist(),
                "fit_lower": background.fit_lower.tolist(),
                "fit_upper": background.fit_upper.tolist(),
                "alpha": background.alpha.tolist(),
                "alpha_uncertainty": background.alpha_uncertainty.tolist(),
                "fit_model": background.fit_model.tolist(),
                "fit_entries": background.fit_entries.tolist(),
                "statistical_variance": (
                    "Poisson signal-plus-alpha-squared-sideband counting variance; "
                    "the common transfer-factor fit uncertainty is retained separately "
                    "as a correlated systematic"
                ),
            }
        else:
            signal_region_mask = fixed_mask
            sideband_mask = np.zeros(event_runs.size, dtype=bool)
            signal_event_weights = signal_region_mask.astype(float)
            signal_variance_weights = signal_region_mask.astype(float)
            yield_mode = (
                "fixed_selection_mask"
                if selection_mask_path is not None
                else "all_selected_candidates"
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
    signal_counts = _sum_by_run(event_runs, signal_event_weights)
    signal_variances = _sum_by_run(event_runs, signal_variance_weights)
    signal_region_counts = Counter(int(run) for run in event_runs[signal_region_mask])
    sideband_counts = Counter(int(run) for run in event_runs[sideband_mask])
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
        signals = float(signal_counts.get(run, 0.0))
        signal_variance = float(signal_variances.get(run, 0.0))
        signal_region = int(signal_region_counts.get(run, 0))
        sideband = int(sideband_counts.get(run, 0))
        estimated_background = float(signal_region - signals)
        if charge_nC > 0.0:
            yield_value = signals / charge_nC
            uncertainty = np.sqrt(max(signal_variance, 1.0)) / charge_nC
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
                signal_statistical_variance=signal_variance,
                signal_region_events=signal_region,
                sideband_events=sideband,
                estimated_background_events=estimated_background,
                charge_c=charge_c,
                charge_nC=charge_nC,
                total_events=total_events.get(run),
                passed_qadb_events=passed_events.get(run),
                failed_qadb_events=failed_events.get(run),
                yield_events_per_nC=yield_value,
                statistical_uncertainty_events_per_nC=uncertainty,
                group_charge_fraction=None,
                group_yield_pull=None,
                included=not reasons,
                exclusion_reason=";".join(reasons),
            )
        )

    low_contribution_runs = apply_minimum_group_charge_fraction(
        records, minimum_group_charge_fraction
    )
    low_yield_runs = apply_low_yield_outlier_rejection(
        records, low_yield_sigma_threshold
    )

    run_charge_sum_c = float(sum(charge_map.values()))
    charge_difference_c = (
        run_charge_sum_c - stored_total_charge_c
        if stored_total_charge_c is not None
        else None
    )
    validation = {
        "candidate_events": int(event_runs.size),
        "yield_mode": yield_mode,
        "signal_events": float(signal_event_weights.sum()),
        "signal_statistical_variance": float(signal_variance_weights.sum()),
        "signal_region_events": int(signal_region_mask.sum()),
        "sideband_events": int(sideband_mask.sum()),
        "estimated_background_events": float(
            signal_region_mask.sum() - signal_event_weights.sum()
        ),
        "background_subtraction": background_metadata,
        "charge_rows": len(charge_map),
        "run_charge_sum_c": run_charge_sum_c,
        "stored_total_charge_c": stored_total_charge_c,
        "charge_difference_c": charge_difference_c,
        "selected_runs_missing_charge": missing_charge_runs,
        "charge_runs_missing_manifest": sorted(set(charge_map).difference(manifest_runs)),
        "dataset": manifest.get("dataset", {}),
        "minimum_group_charge_fraction": minimum_group_charge_fraction,
        "runs_below_minimum_group_charge_fraction": low_contribution_runs,
        "low_yield_sigma_threshold": low_yield_sigma_threshold,
        "runs_below_group_yield_threshold": low_yield_runs,
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


def apply_low_yield_outlier_rejection(
    records: Iterable[RunYield], sigma_threshold: float = 5.0
) -> list[int]:
    """Reject runs significantly below their current group's yield.

    Each run is compared with the charge-aggregated yield of the other eligible
    runs in its class.  The pull denominator combines the run's statistical
    uncertainty with the statistical uncertainty of that leave-one-out group
    mean.  Removing the lowest failing run and repeating makes the result
    deterministic while preventing a low run from diluting its own discrepancy.

    A threshold of zero disables this filter.  Groups containing only one run
    cannot define an independent reference and are left unchanged.
    """
    sigma_threshold = float(sigma_threshold)
    if not np.isfinite(sigma_threshold) or sigma_threshold < 0.0:
        raise ValueError("sigma_threshold must be finite and nonnegative")
    if sigma_threshold == 0.0:
        return []

    grouped: dict[str, list[RunYield]] = {}
    for record in records:
        if record.included:
            if record.run_class is None:
                raise ValueError(f"included run {record.run} has no run class")
            grouped.setdefault(record.run_class, []).append(record)

    rejected: list[int] = []
    for name, initial_members in grouped.items():
        members = list(initial_members)
        while len(members) >= 2:
            pulls: dict[int, float] = {}
            for member in members:
                others = [other for other in members if other is not member]
                reference_charge_nC = float(sum(other.charge_nC for other in others))
                if reference_charge_nC <= 0.0:
                    raise ValueError(
                        f"current group {name} has nonpositive leave-one-out charge"
                    )
                reference_signal = float(sum(other.signal_events for other in others))
                reference_variance = float(
                    sum(other.signal_statistical_variance for other in others)
                )
                reference_yield = reference_signal / reference_charge_nC
                reference_uncertainty = (
                    np.sqrt(max(reference_variance, 1.0)) / reference_charge_nC
                )
                if (
                    member.yield_events_per_nC is None
                    or member.statistical_uncertainty_events_per_nC is None
                ):
                    raise ValueError(
                        f"included run {member.run} has no finite yield information"
                    )
                combined_uncertainty = float(
                    np.hypot(
                        member.statistical_uncertainty_events_per_nC,
                        reference_uncertainty,
                    )
                )
                if not np.isfinite(combined_uncertainty) or combined_uncertainty <= 0.0:
                    raise ValueError(
                        f"included run {member.run} has invalid yield uncertainty"
                    )
                pull = (
                    member.yield_events_per_nC - reference_yield
                ) / combined_uncertainty
                member.group_yield_pull = float(pull)
                pulls[member.run] = float(pull)

            worst = min(members, key=lambda member: pulls[member.run])
            if pulls[worst.run] >= -sigma_threshold:
                break
            worst.included = False
            reason = "below_group_mean_yield_threshold"
            worst.exclusion_reason = (
                f"{worst.exclusion_reason};{reason}"
                if worst.exclusion_reason
                else reason
            )
            rejected.append(worst.run)
            members.remove(worst)

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
        signals = float(sum(member.signal_events for member in members))
        signal_variance = float(
            sum(member.signal_statistical_variance for member in members)
        )
        signal_region = int(sum(member.signal_region_events for member in members))
        sideband = int(sum(member.sideband_events for member in members))
        estimated_background = float(
            sum(member.estimated_background_events for member in members)
        )
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
                signal_statistical_variance=signal_variance,
                signal_region_events=signal_region,
                sideband_events=sideband,
                estimated_background_events=estimated_background,
                charge_c=charge_c,
                charge_nC=charge_nC,
                yield_events_per_nC=signals / charge_nC,
                statistical_uncertainty_events_per_nC=np.sqrt(
                    max(signal_variance, 1.0)
                )
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


def fit_shared_fractional_yield(
    records: Iterable[RunYield],
    groups: Iterable[CurrentGroupYield],
    *,
    period_classes: Mapping[str, Iterable[str]],
    fit_level: str = "groups",
) -> SharedFractionalYieldFit:
    """Fit ``Y_period(I) = A_period * (1 + beta * I)`` jointly.

    The supplied period mapping is deliberately explicit: run-class names alone do
    not reliably encode changes in target, trigger, detector state, or normalization.
    """

    normalized_period_classes: dict[str, list[str]] = {}
    class_periods: dict[str, str] = {}
    for raw_period, raw_classes in period_classes.items():
        period = str(raw_period).strip()
        classes = [str(value).strip() for value in raw_classes if str(value).strip()]
        if not period:
            raise ValueError("shared-slope period names must not be empty")
        if not classes:
            raise ValueError(f"shared-slope period '{period}' contains no run classes")
        if period in normalized_period_classes:
            raise ValueError(f"duplicate shared-slope period '{period}'")
        for run_class in classes:
            if run_class in class_periods:
                raise ValueError(
                    f"run class '{run_class}' is assigned to both "
                    f"'{class_periods[run_class]}' and '{period}'"
                )
            class_periods[run_class] = period
        normalized_period_classes[period] = classes

    periods = list(normalized_period_classes)
    if len(periods) < 2:
        raise ValueError("the shared fractional-slope model requires at least two periods")

    if fit_level == "groups":
        points = list(groups)
        point_classes = [point.group for point in points]
        current = np.asarray([point.effective_current_nA for point in points], dtype=float)
        values = np.asarray([point.yield_events_per_nC for point in points], dtype=float)
        uncertainties = np.asarray(
            [point.statistical_uncertainty_events_per_nC for point in points], dtype=float
        )
    elif fit_level == "runs":
        points = [record for record in records if record.included]
        point_classes = [record.run_class for record in points]
        current = np.asarray([record.current_nA for record in points], dtype=float)
        values = np.asarray([record.yield_events_per_nC for record in points], dtype=float)
        uncertainties = np.asarray(
            [record.statistical_uncertainty_events_per_nC for record in points], dtype=float
        )
    else:
        raise ValueError("fit_level must be 'groups' or 'runs'")

    missing_classes = sorted(
        {str(value) for value in point_classes if value is None or value not in class_periods}
    )
    if missing_classes:
        raise ValueError(
            "included fit classes are absent from --shared-slope-period mappings: "
            + ", ".join(missing_classes)
        )
    point_periods = [class_periods[str(value)] for value in point_classes]
    unused_periods = [period for period in periods if period not in point_periods]
    if unused_periods:
        raise ValueError(
            "shared-slope periods contain no included fit points: "
            + ", ".join(unused_periods)
        )

    parameter_count = len(periods) + 1
    if current.size < parameter_count:
        raise ValueError(
            f"the shared-slope fit needs at least {parameter_count} points for "
            f"{len(periods)} period intercepts and one slope"
        )
    if np.unique(current).size < 2:
        raise ValueError("fit points must contain at least two distinct currents")
    if not (
        np.all(np.isfinite(current))
        and np.all(np.isfinite(values))
        and np.all(np.isfinite(uncertainties))
        and np.all(uncertainties > 0.0)
        and np.all(values > 0.0)
    ):
        raise ValueError("fit inputs contain invalid values or uncertainties")

    period_indices = np.asarray([periods.index(period) for period in point_periods])
    inverse_variance = 1.0 / uncertainties**2
    initial_intercepts = np.asarray(
        [
            np.average(
                values[period_indices == index],
                weights=inverse_variance[period_indices == index],
            )
            for index in range(len(periods))
        ],
        dtype=float,
    )
    fractional_slope_seeds: list[float] = []
    for index in range(len(initial_intercepts)):
        selected = period_indices == index
        if np.count_nonzero(selected) >= 2 and np.unique(current[selected]).size >= 2:
            absolute_slope, absolute_intercept = np.polyfit(
                current[selected], values[selected], 1
            )
            if np.isfinite(absolute_intercept) and absolute_intercept > 0.0:
                fractional_slope_seeds.append(float(absolute_slope / absolute_intercept))
    initial_beta = float(np.median(fractional_slope_seeds)) if fractional_slope_seeds else 0.0
    maximum_current = float(np.max(current))
    beta_lower = -0.999 / maximum_current if maximum_current > 0.0 else -np.inf
    initial_beta = max(initial_beta, beta_lower + 1.0e-8)
    initial = np.concatenate((initial_intercepts, [initial_beta]))

    def residual(parameters: np.ndarray) -> np.ndarray:
        intercepts = parameters[:-1]
        beta = parameters[-1]
        model = intercepts[period_indices] * (1.0 + beta * current)
        return (model - values) / uncertainties

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        intercepts = parameters[:-1]
        beta = parameters[-1]
        result = np.zeros((current.size, parameter_count), dtype=float)
        result[np.arange(current.size), period_indices] = (
            1.0 + beta * current
        ) / uncertainties
        result[:, -1] = intercepts[period_indices] * current / uncertainties
        return result

    from scipy.optimize import least_squares

    lower = np.concatenate((np.full(len(periods), np.finfo(float).tiny), [beta_lower]))
    upper = np.full(parameter_count, np.inf)
    optimized = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=(lower, upper),
        x_scale="jac",
    )
    if not optimized.success:
        raise ValueError(f"shared fractional-slope fit failed: {optimized.message}")
    normal = optimized.jac.T @ optimized.jac
    if np.linalg.matrix_rank(normal) < parameter_count:
        raise ValueError("shared fractional-slope covariance is singular")
    covariance = np.linalg.inv(normal)
    parameters = optimized.x
    intercepts = parameters[:-1]
    beta = float(parameters[-1])
    return SharedFractionalYieldFit(
        period_intercepts_events_per_nC={
            period: float(intercepts[index]) for index, period in enumerate(periods)
        },
        period_intercept_uncertainties_events_per_nC={
            period: float(np.sqrt(max(covariance[index, index], 0.0)))
            for index, period in enumerate(periods)
        },
        fractional_slope_per_nA=beta,
        fractional_slope_uncertainty_per_nA=float(
            np.sqrt(max(covariance[-1, -1], 0.0))
        ),
        parameter_names=[f"intercept:{period}" for period in periods]
        + ["fractional_slope_per_nA"],
        covariance=covariance.tolist(),
        period_classes=normalized_period_classes,
        class_periods=class_periods,
        chi2=float(np.sum(residual(parameters) ** 2)),
        ndf=int(current.size - parameter_count),
        points=int(current.size),
        fit_level=fit_level,
    )


def attach_relative_efficiencies(
    groups: Iterable[CurrentGroupYield],
    fit: LinearYieldFit | SharedFractionalYieldFit,
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
