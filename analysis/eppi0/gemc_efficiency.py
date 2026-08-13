"""GEMC reconstruction efficiency as a function of merged-background current."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class GEMCEfficiencyPoint:
    label: str
    current_nA: float
    response_meta: str
    generated_weight: float
    accepted_weight: float
    efficiency: float
    statistical_uncertainty: float
    uncertainty_model: str
    relative_efficiency: float | None = None
    relative_efficiency_uncertainty: float | None = None


@dataclass(frozen=True)
class LinearEfficiencyFit:
    intercept: float
    intercept_uncertainty: float
    slope_per_nA: float
    slope_uncertainty_per_nA: float
    covariance: list[list[float]]
    chi2: float
    ndf: int
    points: int

    def predict(self, current_nA: np.ndarray | float) -> np.ndarray:
        current = np.asarray(current_nA, dtype=float)
        return self.intercept + self.slope_per_nA * current

    def relative_efficiency(self, current_nA: float) -> tuple[float | None, float | None]:
        if not np.isfinite(self.intercept) or self.intercept <= 0.0:
            return None, None
        current = float(current_nA)
        value = 1.0 + current * self.slope_per_nA / self.intercept
        covariance = np.asarray(self.covariance, dtype=float)
        gradient = np.array(
            [
                -current * self.slope_per_nA / self.intercept**2,
                current / self.intercept,
            ],
            dtype=float,
        )
        variance = float(gradient @ covariance @ gradient)
        return float(value), float(np.sqrt(max(variance, 0.0)))


def load_gemc_efficiencies(
    manifest_path: Path,
) -> tuple[list[GEMCEfficiencyPoint], dict]:
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("GEMC efficiency manifest must contain a nonempty samples array")
    return load_gemc_efficiency_samples(samples, base_directory=manifest_path.parent)


def load_gemc_efficiency_samples(
    samples: list[dict], *, base_directory: Path
) -> tuple[list[GEMCEfficiencyPoint], dict]:
    if not samples:
        raise ValueError("at least one GEMC efficiency sample is required")

    points: list[GEMCEfficiencyPoint] = []
    labels: set[str] = set()
    reference_edges: dict[str, np.ndarray] | None = None
    reference_truth: np.ndarray | None = None
    edge_keys = ("q2_edges", "xb_edges", "t_edges", "phi_edges")
    truth_matches_reference: list[bool] = []
    maximum_truth_difference = 0.0

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"GEMC sample {index} is not an object")
        label = str(sample.get("label", f"sample_{index}"))
        if label in labels:
            raise ValueError(f"duplicate GEMC sample label: {label}")
        labels.add(label)
        current = float(sample["current_nA"])
        if not np.isfinite(current) or current < 0.0:
            raise ValueError(f"invalid GEMC current for {label}: {current}")

        raw_meta = Path(sample["response_meta"]).expanduser()
        meta_path = raw_meta if raw_meta.is_absolute() else base_directory / raw_meta
        meta_path = meta_path.resolve()
        with np.load(meta_path, allow_pickle=False) as metadata:
            required = {"truth_total", "efficiency", *edge_keys}
            missing = sorted(required.difference(metadata.files))
            if missing:
                raise ValueError(f"{meta_path} is missing response arrays: {missing}")
            truth = np.asarray(metadata["truth_total"], dtype=float)
            efficiency_by_bin = np.asarray(metadata["efficiency"], dtype=float)
            edges = {key: np.asarray(metadata[key], dtype=float) for key in edge_keys}

        if truth.ndim != 1 or efficiency_by_bin.shape != truth.shape:
            raise ValueError(f"response arrays have incompatible shapes in {meta_path}")
        if not np.all(np.isfinite(truth)) or np.any(truth < 0.0):
            raise ValueError(f"invalid truth totals in {meta_path}")
        if not np.all(np.isfinite(efficiency_by_bin)) or np.any(efficiency_by_bin < 0.0):
            raise ValueError(f"invalid efficiencies in {meta_path}")
        generated = float(truth.sum())
        accepted = float(np.sum(truth * efficiency_by_bin))
        if generated <= 0.0:
            raise ValueError(f"GEMC sample {label} has no generated weight in range")
        efficiency = accepted / generated
        if efficiency < 0.0 or efficiency > 1.0 + 1.0e-9:
            raise ValueError(f"GEMC efficiency is outside [0,1] for {label}: {efficiency}")

        explicit_uncertainty = sample.get("statistical_uncertainty")
        if explicit_uncertainty is not None:
            uncertainty = float(explicit_uncertainty)
            uncertainty_model = "manifest"
        else:
            uncertainty = float(np.sqrt(max(efficiency * (1.0 - efficiency), 0.0) / generated))
            integer_like = np.allclose(truth, np.rint(truth), rtol=0.0, atol=1.0e-8)
            uncertainty_model = (
                "binomial_unweighted"
                if integer_like
                else "binomial_effective_weight_approximation"
            )
        if not np.isfinite(uncertainty) or uncertainty <= 0.0:
            raise ValueError(f"invalid GEMC uncertainty for {label}: {uncertainty}")

        if reference_edges is None:
            reference_edges = edges
            reference_truth = truth
            truth_matches_reference.append(True)
        else:
            for key in edge_keys:
                if not np.array_equal(edges[key], reference_edges[key]):
                    raise ValueError(f"GEMC response binning differs for {label}: {key}")
            assert reference_truth is not None
            if truth.shape != reference_truth.shape:
                raise ValueError(f"GEMC truth shape differs for {label}")
            matches = bool(np.allclose(truth, reference_truth, rtol=1.0e-10, atol=1.0e-8))
            truth_matches_reference.append(matches)
            denominator = np.maximum(np.abs(reference_truth), 1.0)
            maximum_truth_difference = max(
                maximum_truth_difference,
                float(np.max(np.abs(truth - reference_truth) / denominator)),
            )

        points.append(
            GEMCEfficiencyPoint(
                label=label,
                current_nA=current,
                response_meta=str(meta_path),
                generated_weight=generated,
                accepted_weight=accepted,
                efficiency=float(efficiency),
                statistical_uncertainty=uncertainty,
                uncertainty_model=uncertainty_model,
            )
        )

    points.sort(key=lambda point: point.current_nA)
    validation = {
        "samples": len(points),
        "binning_identical": True,
        "truth_totals_match_reference": all(truth_matches_reference),
        "maximum_relative_truth_total_difference": maximum_truth_difference,
    }
    return points, validation


def fit_linear_efficiency(points: list[GEMCEfficiencyPoint]) -> LinearEfficiencyFit:
    if len(points) < 2:
        raise ValueError("at least two GEMC efficiency points are required")
    current = np.asarray([point.current_nA for point in points], dtype=float)
    efficiency = np.asarray([point.efficiency for point in points], dtype=float)
    uncertainty = np.asarray(
        [point.statistical_uncertainty for point in points], dtype=float
    )
    if np.unique(current).size < 2:
        raise ValueError("GEMC efficiency points need at least two distinct currents")

    design = np.column_stack((np.ones(current.size), current))
    inverse_variance = 1.0 / uncertainty**2
    normal = design.T @ (inverse_variance[:, None] * design)
    covariance = np.linalg.inv(normal)
    parameters = covariance @ (design.T @ (inverse_variance * efficiency))
    residual = efficiency - design @ parameters
    chi2 = float(np.sum((residual / uncertainty) ** 2))
    return LinearEfficiencyFit(
        intercept=float(parameters[0]),
        intercept_uncertainty=float(np.sqrt(covariance[0, 0])),
        slope_per_nA=float(parameters[1]),
        slope_uncertainty_per_nA=float(np.sqrt(covariance[1, 1])),
        covariance=covariance.tolist(),
        chi2=chi2,
        ndf=len(points) - 2,
        points=len(points),
    )


def attach_relative_gemc_efficiencies(
    points: list[GEMCEfficiencyPoint], fit: LinearEfficiencyFit
) -> None:
    if fit.intercept <= 0.0:
        return
    for point in points:
        point.relative_efficiency = point.efficiency / fit.intercept
        point.relative_efficiency_uncertainty = (
            point.statistical_uncertainty / fit.intercept
        )
