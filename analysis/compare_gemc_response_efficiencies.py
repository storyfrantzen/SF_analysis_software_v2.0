#!/usr/bin/env python3

"""Compare global GEMC efficiencies before and after truth-mixture standardization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EDGE_KEYS = ("q2_edges", "xb_edges", "t_edges", "phi_edges")


@dataclass(frozen=True)
class ResponseSample:
    label: str
    current_nA: float
    path: Path
    truth: np.ndarray
    integrated_accepted: np.ndarray
    topology_accepted: dict[int, np.ndarray]
    topology_labels: dict[int, str]
    edges: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native global response efficiencies with efficiencies "
            "standardized to one common generated-truth mixture."
        )
    )
    parser.add_argument(
        "--sample",
        action="append",
        nargs=3,
        required=True,
        metavar=("LABEL", "CURRENT_NA", "RESPONSE_META"),
        help="response sample; repeat for each current",
    )
    return parser.parse_args()


def scalar_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_sample(label: str, current: str, path_text: str) -> ResponseSample:
    path = Path(path_text)
    with np.load(path, allow_pickle=False) as metadata:
        required = {"truth_total", "efficiency", *EDGE_KEYS}
        missing = sorted(required.difference(metadata.files))
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        truth = np.asarray(metadata["truth_total"], dtype=float).reshape(-1)
        efficiency = np.asarray(metadata["efficiency"], dtype=float).reshape(-1)
        if truth.shape != efficiency.shape:
            raise ValueError(f"{path} has incompatible truth and efficiency shapes")
        integrated_accepted = truth * efficiency
        topology_accepted: dict[int, np.ndarray] = {}
        topology_labels: dict[int, str] = {}
        topology_fields = {
            "accepted_topology_counts",
            "reconstructed_topology_ids",
        }
        if topology_fields.issubset(metadata.files):
            accepted = np.asarray(metadata["accepted_topology_counts"], dtype=float)
            ids = np.asarray(metadata["reconstructed_topology_ids"], dtype=int)
            if accepted.shape != (ids.size, truth.size):
                raise ValueError(
                    f"{path} has incompatible accepted_topology_counts shape "
                    f"{accepted.shape}"
                )
            labels = (
                np.asarray(metadata["reconstructed_topology_labels"])
                if "reconstructed_topology_labels" in metadata.files
                else np.asarray([str(value) for value in ids])
            )
            if labels.size != ids.size:
                raise ValueError(f"{path} has incompatible topology labels")
            for index, topology_id in enumerate(ids):
                topology_accepted[int(topology_id)] = accepted[index].reshape(-1)
                topology_labels[int(topology_id)] = scalar_text(labels[index])
        edges = {key: np.asarray(metadata[key]) for key in EDGE_KEYS}
    sample = ResponseSample(
        label=label,
        current_nA=float(current),
        path=path,
        truth=truth,
        integrated_accepted=integrated_accepted,
        topology_accepted=topology_accepted,
        topology_labels=topology_labels,
        edges=edges,
    )
    return sample


def validate_samples(samples: list[ResponseSample]) -> None:
    if len(samples) < 2:
        raise ValueError("at least two --sample entries are required")
    currents = [sample.current_nA for sample in samples]
    if len(set(currents)) != len(currents):
        raise ValueError("sample currents must be distinct")
    reference = samples[0]
    reference_edges = reference.edges
    for sample in samples[1:]:
        if sample.truth.shape != reference.truth.shape:
            raise ValueError("response samples have different truth shapes")
        edges = sample.edges
        if any(not np.array_equal(edges[key], reference_edges[key]) for key in EDGE_KEYS):
            raise ValueError("response samples have different analysis binning")


def common_truth_weights(samples: list[ResponseSample]) -> tuple[np.ndarray, np.ndarray]:
    support = np.logical_and.reduce([sample.truth > 0.0 for sample in samples])
    if not np.any(support):
        raise ValueError("response samples have no common truth support")
    reference_truth = np.where(support, samples[0].truth, 0.0)
    reference_total = float(reference_truth.sum())
    if reference_total <= 0.0:
        raise ValueError(f"{samples[0].label} has no truth on common support")
    weights = reference_truth / reference_total
    return support, weights


def native_efficiency(sample: ResponseSample, accepted: np.ndarray) -> float:
    truth_total = float(sample.truth.sum())
    return float(accepted.sum() / truth_total) if truth_total > 0.0 else float("nan")


def standardized_efficiency(
    sample: ResponseSample,
    accepted: np.ndarray,
    support: np.ndarray,
    weights: np.ndarray,
) -> float:
    per_bin = np.divide(
        accepted,
        sample.truth,
        out=np.zeros_like(accepted, dtype=float),
        where=sample.truth > 0.0,
    )
    return float(np.sum(weights[support] * per_bin[support]))


def fractional_slope(currents: np.ndarray, efficiencies: np.ndarray) -> float:
    design = np.column_stack([np.ones(currents.size), currents])
    intercept, slope = np.linalg.lstsq(design, efficiencies, rcond=None)[0]
    if not np.isfinite(intercept) or intercept <= 0.0:
        return float("nan")
    return float(slope / intercept)


def print_comparison(
    name: str,
    samples: list[ResponseSample],
    accepted_arrays: list[np.ndarray],
    support: np.ndarray,
    weights: np.ndarray,
) -> None:
    native = np.asarray(
        [native_efficiency(sample, accepted) for sample, accepted in zip(samples, accepted_arrays)]
    )
    standardized = np.asarray(
        [
            standardized_efficiency(sample, accepted, support, weights)
            for sample, accepted in zip(samples, accepted_arrays)
        ]
    )
    currents = np.asarray([sample.current_nA for sample in samples])
    print(f"\n{name}")
    for sample, native_value, standardized_value in zip(samples, native, standardized):
        print(
            f"  {sample.label} at {sample.current_nA:g} nA: "
            f"native={native_value:.10g} common_truth={standardized_value:.10g}"
        )
    native_beta = fractional_slope(currents, native)
    standardized_beta = fractional_slope(currents, standardized)
    print("  native fractional slope =", native_beta, "nA^-1")
    print("  common-truth fractional slope =", standardized_beta, "nA^-1")
    print(
        "  truth-mixture contribution to slope =",
        native_beta - standardized_beta,
        "nA^-1",
    )


def main() -> int:
    args = parse_args()
    samples = [load_sample(*specification) for specification in args.sample]
    samples.sort(key=lambda sample: sample.current_nA)
    validate_samples(samples)
    support, weights = common_truth_weights(samples)

    print("Samples:")
    for sample in samples:
        support_fraction = float(sample.truth[support].sum() / sample.truth.sum())
        print(
            f"  {sample.label}: current={sample.current_nA:g} nA "
            f"truth={sample.truth.sum():.0f} common-support fraction={support_fraction:.8f} "
            f"path={sample.path}"
        )
    print("Common supported bins =", int(np.count_nonzero(support)), "/", support.size)
    print("Truth standardization reference =", samples[0].label)

    print_comparison(
        "topology integrated",
        samples,
        [sample.integrated_accepted for sample in samples],
        support,
        weights,
    )

    common_topologies = set(samples[0].topology_accepted)
    for sample in samples[1:]:
        common_topologies &= set(sample.topology_accepted)
    for topology_id in sorted(common_topologies):
        label = samples[0].topology_labels.get(topology_id, str(topology_id))
        print_comparison(
            f"topology {topology_id} ({label})",
            samples,
            [sample.topology_accepted[topology_id] for sample in samples],
            support,
            weights,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
