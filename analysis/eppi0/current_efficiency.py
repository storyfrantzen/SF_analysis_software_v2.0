"""Scalar data/GEMC current-efficiency correction models and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class RelativeLinearEfficiency:
    """A linear efficiency normalized to its fitted zero-current intercept."""

    intercept: float
    slope_per_nA: float
    covariance: tuple[tuple[float, float], tuple[float, float]]

    def __post_init__(self) -> None:
        if not np.isfinite(self.intercept) or self.intercept <= 0.0:
            raise ValueError("efficiency-model intercept must be positive and finite")
        if not np.isfinite(self.slope_per_nA):
            raise ValueError("efficiency-model slope must be finite")
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            raise ValueError("efficiency-model covariance must be a finite 2x2 matrix")
        if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-14):
            raise ValueError("efficiency-model covariance must be symmetric")

    def relative_efficiency(
        self, current_nA: np.ndarray | float
    ) -> np.ndarray:
        current = np.asarray(current_nA, dtype=float)
        return 1.0 + current * self.slope_per_nA / self.intercept

    def relative_uncertainty(
        self, current_nA: np.ndarray | float
    ) -> np.ndarray:
        current = np.asarray(current_nA, dtype=float)
        gradient_intercept = -current * self.slope_per_nA / self.intercept**2
        gradient_slope = current / self.intercept
        covariance = np.asarray(self.covariance, dtype=float)
        variance = (
            covariance[0, 0] * gradient_intercept**2
            + 2.0 * covariance[0, 1] * gradient_intercept * gradient_slope
            + covariance[1, 1] * gradient_slope**2
        )
        return np.sqrt(np.maximum(variance, 0.0))

    def as_dict(self, *, quantity: str) -> dict:
        return {
            "form": "eta(I) = (intercept + slope_per_nA * I) / intercept",
            "quantity": quantity,
            "intercept": self.intercept,
            "slope_per_nA": self.slope_per_nA,
            "covariance": [list(row) for row in self.covariance],
        }

    @classmethod
    def from_dict(cls, values: Mapping) -> "RelativeLinearEfficiency":
        covariance = np.asarray(values["covariance"], dtype=float)
        return cls(
            intercept=float(values["intercept"]),
            slope_per_nA=float(values["slope_per_nA"]),
            covariance=tuple(tuple(float(item) for item in row) for row in covariance),
        )


@dataclass(frozen=True)
class CurrentEfficiencyCorrection:
    """Post-doc-style scalar correction for a response at one reference current."""

    data_model: RelativeLinearEfficiency
    gemc_model: RelativeLinearEfficiency
    reference_current_nA: float
    reference_label: str
    reference_response_meta: str
    reference_response_meta_sha256: str
    run_currents_nA: Mapping[int, float]
    payload: Mapping

    def __post_init__(self) -> None:
        if not np.isfinite(self.reference_current_nA) or self.reference_current_nA < 0.0:
            raise ValueError("reference current must be finite and nonnegative")
        eta_mc = float(self.gemc_model.relative_efficiency(self.reference_current_nA))
        if not np.isfinite(eta_mc) or eta_mc <= 0.0:
            raise ValueError("GEMC relative efficiency is nonpositive at the reference current")
        for run, current in self.run_currents_nA.items():
            if not np.isfinite(current) or current < 0.0:
                raise ValueError(f"invalid current for run {run}: {current}")
            eta_data = float(self.data_model.relative_efficiency(current))
            if not np.isfinite(eta_data) or eta_data <= 0.0:
                raise ValueError(
                    f"data relative efficiency is nonpositive for run {run} at {current} nA"
                )

    @property
    def eta_mc_reference(self) -> float:
        return float(self.gemc_model.relative_efficiency(self.reference_current_nA))

    @property
    def d_reference(self) -> float:
        eta_data = float(self.data_model.relative_efficiency(self.reference_current_nA))
        return eta_data / self.eta_mc_reference

    def d_factor(
        self, current_nA: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        current = np.asarray(current_nA, dtype=float)
        eta_data = self.data_model.relative_efficiency(current)
        eta_mc = self.gemc_model.relative_efficiency(current)
        if np.any(eta_data <= 0.0) or np.any(eta_mc <= 0.0):
            raise ValueError("D(I) is undefined where a fitted relative efficiency is nonpositive")
        sigma_data = self.data_model.relative_uncertainty(current)
        sigma_mc = self.gemc_model.relative_uncertainty(current)
        value = eta_data / eta_mc
        variance = (sigma_data / eta_mc) ** 2 + (
            eta_data * sigma_mc / eta_mc**2
        ) ** 2
        return value, np.sqrt(np.maximum(variance, 0.0))

    def weights_for_currents(
        self, current_nA: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        current = np.asarray(current_nA, dtype=float)
        eta_data = self.data_model.relative_efficiency(current)
        eta_mc_reference = self.eta_mc_reference
        if np.any(eta_data <= 0.0):
            raise ValueError("current-efficiency weight has a nonpositive data denominator")
        sigma_data = self.data_model.relative_uncertainty(current)
        sigma_mc_reference = float(
            self.gemc_model.relative_uncertainty(self.reference_current_nA)
        )
        value = eta_mc_reference / eta_data
        variance = (sigma_mc_reference / eta_data) ** 2 + (
            eta_mc_reference * sigma_data / eta_data**2
        ) ** 2
        return value, np.sqrt(np.maximum(variance, 0.0))

    def event_weights(self, runs: np.ndarray) -> np.ndarray:
        runs = np.asarray(runs, dtype=np.int64)
        if runs.ndim != 1:
            raise ValueError("event run numbers must be one-dimensional")
        unique_runs = np.unique(runs)
        missing = [int(run) for run in unique_runs if int(run) not in self.run_currents_nA]
        if missing:
            raise ValueError(
                "selected data events have no usable current in the correction artifact for runs: "
                + ", ".join(str(run) for run in missing)
            )
        weights = np.empty(runs.size, dtype=float)
        for run in unique_runs:
            mask = runs == run
            value, _ = self.weights_for_currents(self.run_currents_nA[int(run)])
            weights[mask] = float(value)
        return weights


def response_meta_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correction_artifact(
    *,
    data_model: RelativeLinearEfficiency,
    gemc_model: RelativeLinearEfficiency,
    reference_current_nA: float,
    reference_label: str,
    reference_response_meta: Path,
    run_records: Iterable,
    sources: Mapping,
) -> dict:
    reference_response_meta = reference_response_meta.resolve()
    correction = CurrentEfficiencyCorrection(
        data_model=data_model,
        gemc_model=gemc_model,
        reference_current_nA=float(reference_current_nA),
        reference_label=reference_label,
        reference_response_meta=str(reference_response_meta),
        reference_response_meta_sha256=response_meta_sha256(reference_response_meta),
        run_currents_nA={
            int(record.run): float(record.current_nA)
            for record in run_records
            if record.current_nA is not None and record.charge_c > 0.0
        },
        payload={},
    )
    runs = {}
    for record in run_records:
        if record.current_nA is None or record.charge_c <= 0.0:
            continue
        weight, uncertainty = correction.weights_for_currents(float(record.current_nA))
        runs[str(record.run)] = {
            "current_nA": float(record.current_nA),
            "run_class": record.run_class,
            "current_quality": record.current_quality,
            "fit_included": bool(record.included),
            "fit_exclusion_reason": record.exclusion_reason,
            "event_weight": float(weight),
            "event_weight_fit_uncertainty": float(uncertainty),
        }
    eta_data_reference = float(data_model.relative_efficiency(reference_current_nA))
    eta_mc_reference = float(gemc_model.relative_efficiency(reference_current_nA))
    return {
        "schema_version": 1,
        "method": "postdoc_scalar_data_over_gemc_current_efficiency",
        "definitions": {
            "D(I)": "eta_data(I) / eta_MC(I)",
            "event_weight": "eta_MC(I_reference) / eta_data(I_run)",
            "equivalent_decomposition": (
                "[eta_data(I_reference) / eta_data(I_run)] / D(I_reference)"
            ),
        },
        "normalization": (
            "Apply event weights to the reconstructed data histogram before unfolding; "
            "divide the final corrected yield by the original physical beam charge."
        ),
        "data_model": data_model.as_dict(quantity="selected data yield in events/nC"),
        "gemc_model": gemc_model.as_dict(quantity="accepted/generated GEMC efficiency"),
        "reference": {
            "current_nA": float(reference_current_nA),
            "label": reference_label,
            "response_meta": str(reference_response_meta),
            "response_meta_sha256": correction.reference_response_meta_sha256,
            "eta_data": eta_data_reference,
            "eta_MC": eta_mc_reference,
            "D": eta_data_reference / eta_mc_reference,
        },
        "runs": runs,
        "fit_included_runs": sorted(int(record.run) for record in run_records if record.included),
        "sources": dict(sources),
    }


def load_current_efficiency_correction(
    path: str | Path,
) -> CurrentEfficiencyCorrection:
    artifact_path = Path(path).resolve()
    with artifact_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"unsupported current-efficiency schema in {artifact_path}: "
            f"{payload.get('schema_version')}"
        )
    reference = payload.get("reference", {})
    runs = payload.get("runs")
    if not isinstance(runs, dict) or not runs:
        raise ValueError(f"current-efficiency artifact has no usable runs: {artifact_path}")
    return CurrentEfficiencyCorrection(
        data_model=RelativeLinearEfficiency.from_dict(payload["data_model"]),
        gemc_model=RelativeLinearEfficiency.from_dict(payload["gemc_model"]),
        reference_current_nA=float(reference["current_nA"]),
        reference_label=str(reference["label"]),
        reference_response_meta=str(reference["response_meta"]),
        reference_response_meta_sha256=str(reference["response_meta_sha256"]),
        run_currents_nA={
            int(run): float(values["current_nA"]) for run, values in runs.items()
        },
        payload=payload,
    )
