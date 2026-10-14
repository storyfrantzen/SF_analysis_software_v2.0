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
    native_generated_weight: float
    native_accepted_weight: float
    native_efficiency: float
    common_truth_support_fraction: float
    truth_standardization: str
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
    *,
    topology_group_ids: tuple[int, ...] | list[int] = (),
) -> tuple[list[GEMCEfficiencyPoint], dict]:
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("GEMC efficiency manifest must contain a nonempty samples array")
    return load_gemc_efficiency_samples(
        samples,
        base_directory=manifest_path.parent,
        topology_group_ids=topology_group_ids,
    )


def load_gemc_efficiency_samples(
    samples: list[dict],
    *,
    base_directory: Path,
    topology_group_ids: tuple[int, ...] | list[int] = (),
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
    selected_topology_groups = tuple(
        sorted({int(group_id) for group_id in topology_group_ids})
    )
    truth_arrays: list[np.ndarray] = []
    accepted_arrays: list[np.ndarray] = []
    explicit_uncertainties: list[float | None] = []
    integer_like_truth: list[bool] = []

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
            if selected_topology_groups:
                required.update(
                    {
                        "reconstructed_topology_ids",
                        "accepted_topology_counts",
                    }
                )
            missing = sorted(required.difference(metadata.files))
            if missing:
                raise ValueError(f"{meta_path} is missing response arrays: {missing}")
            truth = np.asarray(metadata["truth_total"], dtype=float)
            efficiency_by_bin = np.asarray(metadata["efficiency"], dtype=float)
            edges = {key: np.asarray(metadata[key], dtype=float) for key in edge_keys}
            if selected_topology_groups:
                topology_ids = np.asarray(
                    metadata["reconstructed_topology_ids"], dtype=np.int64
                )
                topology_counts = np.asarray(
                    metadata["accepted_topology_counts"], dtype=float
                )

        if truth.ndim != 1 or efficiency_by_bin.shape != truth.shape:
            raise ValueError(f"response arrays have incompatible shapes in {meta_path}")
        if not np.all(np.isfinite(truth)) or np.any(truth < 0.0):
            raise ValueError(f"invalid truth totals in {meta_path}")
        if not np.all(np.isfinite(efficiency_by_bin)) or np.any(efficiency_by_bin < 0.0):
            raise ValueError(f"invalid efficiencies in {meta_path}")
        if selected_topology_groups:
            if topology_ids.ndim != 1 or topology_counts.shape != (
                topology_ids.size,
                truth.size,
            ):
                raise ValueError(
                    f"topology response arrays have incompatible shapes in {meta_path}"
                )
            if not np.all(np.isfinite(topology_counts)) or np.any(topology_counts < 0.0):
                raise ValueError(f"invalid topology response counts in {meta_path}")
            integrated_accepted_by_bin = truth * efficiency_by_bin
            if not np.allclose(
                topology_counts.sum(axis=0),
                integrated_accepted_by_bin,
                rtol=1.0e-10,
                atol=1.0e-8,
            ):
                raise ValueError(
                    "accepted topology counts do not reconstruct the integrated "
                    f"efficiency numerator in {meta_path}"
                )
            missing_topologies = sorted(
                set(selected_topology_groups).difference(
                    int(value) for value in topology_ids
                )
            )
            if missing_topologies:
                raise ValueError(
                    f"{meta_path} is missing reconstructed topology IDs: "
                    + ", ".join(str(value) for value in missing_topologies)
                )
            topology_rows = np.isin(
                topology_ids, np.asarray(selected_topology_groups, dtype=np.int64)
            )
            accepted_by_bin = topology_counts[topology_rows].sum(axis=0)
        else:
            accepted_by_bin = truth * efficiency_by_bin
        generated = float(truth.sum())
        accepted = float(accepted_by_bin.sum())
        if generated <= 0.0:
            raise ValueError(f"GEMC sample {label} has no generated weight in range")
        efficiency = accepted / generated
        if efficiency < 0.0 or efficiency > 1.0 + 1.0e-9:
            raise ValueError(f"GEMC efficiency is outside [0,1] for {label}: {efficiency}")

        integer_like = np.allclose(truth, np.rint(truth), rtol=0.0, atol=1.0e-8)
        explicit_uncertainty = sample.get("statistical_uncertainty")
        if explicit_uncertainty is not None:
            uncertainty = float(explicit_uncertainty)
            uncertainty_model = "manifest"
        else:
            uncertainty = float(np.sqrt(max(efficiency * (1.0 - efficiency), 0.0) / generated))
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
                native_generated_weight=generated,
                native_accepted_weight=accepted,
                native_efficiency=float(efficiency),
                common_truth_support_fraction=1.0,
                truth_standardization="native_truth_distribution",
            )
        )
        truth_arrays.append(truth)
        accepted_arrays.append(accepted_by_bin)
        explicit_uncertainties.append(
            float(explicit_uncertainty) if explicit_uncertainty is not None else None
        )
        integer_like_truth.append(integer_like)

    reference_index = int(np.argmin([point.current_nA for point in points]))
    common_support = np.logical_and.reduce([truth > 0.0 for truth in truth_arrays])
    if not np.any(common_support):
        raise ValueError("GEMC samples have no common generated-truth support")
    reference_common_truth = np.where(
        common_support, truth_arrays[reference_index], 0.0
    )
    reference_common_total = float(reference_common_truth.sum())
    if reference_common_total <= 0.0:
        raise ValueError("lowest-current GEMC sample has no truth on common support")
    common_truth_weights = reference_common_truth / reference_common_total
    standardization = "lowest_current_truth_distribution_on_common_support"

    common_support_fractions: list[float] = []
    for index, point in enumerate(points):
        truth = truth_arrays[index]
        accepted_by_bin = accepted_arrays[index]
        efficiency_by_bin = np.divide(
            accepted_by_bin,
            truth,
            out=np.zeros_like(accepted_by_bin, dtype=float),
            where=truth > 0.0,
        )
        standardized_efficiency = float(
            np.sum(
                common_truth_weights[common_support]
                * efficiency_by_bin[common_support]
            )
        )
        if not 0.0 <= standardized_efficiency <= 1.0 + 1.0e-9:
            raise ValueError(
                f"standardized GEMC efficiency is outside [0,1] for {point.label}: "
                f"{standardized_efficiency}"
            )
        support_fraction = float(truth[common_support].sum() / truth.sum())
        common_support_fractions.append(support_fraction)

        explicit_uncertainty = explicit_uncertainties[index]
        if explicit_uncertainty is not None:
            if point.native_efficiency > 0.0:
                uncertainty = float(
                    explicit_uncertainty
                    * standardized_efficiency
                    / point.native_efficiency
                )
            else:
                uncertainty = float(explicit_uncertainty)
            uncertainty_model = "manifest_fractional_rescaled_for_common_truth"
        else:
            variance = float(
                np.sum(
                    common_truth_weights[common_support] ** 2
                    * efficiency_by_bin[common_support]
                    * (1.0 - efficiency_by_bin[common_support])
                    / truth[common_support]
                )
            )
            uncertainty = float(np.sqrt(max(variance, 0.0)))
            uncertainty_model = (
                "common_truth_stratified_binomial_unweighted"
                if integer_like_truth[index]
                else "common_truth_stratified_binomial_effective_weight_approximation"
            )
        if not np.isfinite(uncertainty) or uncertainty <= 0.0:
            raise ValueError(
                f"invalid standardized GEMC uncertainty for {point.label}: {uncertainty}"
            )

        point.accepted_weight = standardized_efficiency * point.generated_weight
        point.efficiency = standardized_efficiency
        point.statistical_uncertainty = uncertainty
        point.uncertainty_model = uncertainty_model
        point.common_truth_support_fraction = support_fraction
        point.truth_standardization = standardization

    order = np.argsort([point.current_nA for point in points])
    points = [points[index] for index in order]
    validation = {
        "samples": len(points),
        "binning_identical": True,
        "truth_totals_match_reference": all(truth_matches_reference),
        "maximum_relative_truth_total_difference": maximum_truth_difference,
        "truth_standardization": standardization,
        "truth_standardization_reference_label": points[0].label,
        "common_truth_supported_bins": int(np.count_nonzero(common_support)),
        "total_truth_bins": int(common_support.size),
        "common_truth_support_fractions": {
            points[position].label: common_support_fractions[index]
            for position, index in enumerate(order)
        },
        "topology_group_ids": list(selected_topology_groups),
        "accepted_weight_definition": (
            "common-truth-standardized accepted equivalent for the selected reconstructed "
            "topology groups, restricted to generated events in the analysis phase space "
            "and excluding feed-in"
            if selected_topology_groups
            else "common-truth-standardized integrated accepted equivalent"
        ),
        "native_accepted_weight_definition": (
            "sum of generated-event-weighted accepted_topology_counts"
            if selected_topology_groups
            else "sum of truth_total times integrated response efficiency"
        ),
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
