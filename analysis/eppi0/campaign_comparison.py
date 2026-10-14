"""Utilities for matched-validity comparisons between analysis campaigns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CommonValidityMasks:
    """Paired 4D masks over exactly matching configured kinematic bins."""

    left: Array
    right: Array
    left_q2_indices: Array
    right_q2_indices: Array


def _matching_q2_bins(left_edges: Array, right_edges: Array) -> tuple[Array, Array]:
    left_edges = np.asarray(left_edges, dtype=float)
    right_edges = np.asarray(right_edges, dtype=float)
    if left_edges.ndim != 1 or right_edges.ndim != 1:
        raise ValueError("Q2 edges must be one-dimensional")
    if left_edges.size < 2 or right_edges.size < 2:
        raise ValueError("Q2 edges must describe at least one bin")

    left_indices: list[int] = []
    right_indices: list[int] = []
    right_intervals = [
        (float(right_edges[index]), float(right_edges[index + 1]))
        for index in range(right_edges.size - 1)
    ]
    for left_index in range(left_edges.size - 1):
        interval = (
            float(left_edges[left_index]),
            float(left_edges[left_index + 1]),
        )
        matches = [
            right_index
            for right_index, candidate in enumerate(right_intervals)
            if np.allclose(interval, candidate, rtol=0.0, atol=1.0e-12)
        ]
        if len(matches) > 1:
            raise ValueError(f"right campaign repeats Q2 interval {interval}")
        if matches:
            left_indices.append(left_index)
            right_indices.append(matches[0])
    if not left_indices:
        raise ValueError("campaigns have no exactly matching Q2 bins")
    return np.asarray(left_indices, dtype=np.int64), np.asarray(
        right_indices, dtype=np.int64
    )


def common_validity_masks(
    left_mask: Array,
    right_mask: Array,
    *,
    left_q2_edges: Array,
    right_q2_edges: Array,
    left_xb_edges: Array,
    right_xb_edges: Array,
    left_t_edges: Array,
    right_t_edges: Array,
    left_phi_edges: Array,
    right_phi_edges: Array,
) -> CommonValidityMasks:
    """Intersect validity in bins with identical four-dimensional boundaries."""

    left_mask = np.asarray(left_mask, dtype=bool)
    right_mask = np.asarray(right_mask, dtype=bool)
    if left_mask.ndim != 4 or right_mask.ndim != 4:
        raise ValueError("campaign validity masks must be four-dimensional")
    if left_mask.shape[1:] != right_mask.shape[1:]:
        raise ValueError("campaign validity masks have incompatible xB, -t, or phi shapes")
    for name, left, right in (
        ("xB", left_xb_edges, right_xb_edges),
        ("-t", left_t_edges, right_t_edges),
        ("phi", left_phi_edges, right_phi_edges),
    ):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        if left.shape != right.shape or not np.allclose(
            left, right, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"campaigns have incompatible {name} bin edges")

    left_q2_indices, right_q2_indices = _matching_q2_bins(
        left_q2_edges, right_q2_edges
    )
    left_common = np.zeros_like(left_mask)
    right_common = np.zeros_like(right_mask)
    for left_index, right_index in zip(left_q2_indices, right_q2_indices):
        common = left_mask[left_index] & right_mask[right_index]
        left_common[left_index] = common
        right_common[right_index] = common
    return CommonValidityMasks(
        left=left_common,
        right=right_common,
        left_q2_indices=left_q2_indices,
        right_q2_indices=right_q2_indices,
    )
