from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


Array = np.ndarray


def bin_indices(values: Array, edges: Array) -> Array:
    """Return zero-based bins, using -1 outside the half-open edge range."""
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    indices = np.searchsorted(edges, values, side="right") - 1
    valid = np.isfinite(values) & (indices >= 0) & (indices < edges.size - 1)
    return np.where(valid, indices, -1).astype(np.int64, copy=False)


@dataclass(frozen=True)
class AnalysisBinning:
    q2_edges: Array
    xb_edges: Array
    t_edges: Array
    phi_edges: Array

    def __post_init__(self) -> None:
        for name in ("q2_edges", "xb_edges", "t_edges", "phi_edges"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0):
                raise ValueError(f"{name} must be a strictly increasing 1D array")
            object.__setattr__(self, name, values)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (
            self.q2_edges.size - 1,
            self.xb_edges.size - 1,
            self.t_edges.size - 1,
            self.phi_edges.size - 1,
        )

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def indices(self, q2: Array, xb: Array, minus_t: Array, phi_rad: Array) -> tuple[Array, ...]:
        phi_deg = np.mod(np.asarray(phi_rad, dtype=float), 2.0 * np.pi) * 180.0 / np.pi
        return (
            bin_indices(q2, self.q2_edges),
            bin_indices(xb, self.xb_edges),
            bin_indices(minus_t, self.t_edges),
            bin_indices(phi_deg, self.phi_edges),
        )

    def flatten(self, iq2: Array, ixb: Array, it: Array, iphi: Array) -> Array:
        """Use the exact legacy order: xB, Q2, phi, then t fastest."""
        nq2, _, nt, nphi = self.shape
        iq2, ixb, it, iphi = np.broadcast_arrays(iq2, ixb, it, iphi)
        valid = (iq2 >= 0) & (ixb >= 0) & (it >= 0) & (iphi >= 0)
        flat = ixb * nq2 * nphi * nt + iq2 * nphi * nt + iphi * nt + it
        return np.where(valid, flat, -1).astype(np.int64, copy=False)

    def coordinates_to_flat(
        self, q2: Array, xb: Array, minus_t: Array, phi_rad: Array
    ) -> Array:
        return self.flatten(*self.indices(q2, xb, minus_t, phi_rad))

    def unflatten(self, values: Array) -> Array:
        """Convert a legacy flat vector to `(Q2, xB, t, phi)`."""
        values = np.asarray(values)
        if values.shape[-1] != self.size:
            raise ValueError(f"last dimension must contain {self.size} bins")
        nq2, nxb, nt, nphi = self.shape
        leading = values.shape[:-1]
        reshaped = values.reshape(*leading, nxb, nq2, nphi, nt)
        axes = tuple(range(len(leading))) + (
            len(leading) + 1,
            len(leading),
            len(leading) + 3,
            len(leading) + 2,
        )
        return reshaped.transpose(axes)

    def flatten_values(self, values: Array) -> Array:
        """Convert `(Q2, xB, t, phi)` arrays to the legacy flat order."""
        values = np.asarray(values)
        if values.shape[-4:] != self.shape:
            raise ValueError(f"last four dimensions must be {self.shape}")
        leading = values.shape[:-4]
        offset = len(leading)
        axes = tuple(range(offset)) + (offset + 1, offset, offset + 3, offset + 2)
        return values.transpose(axes).reshape(*leading, self.size)

    def bin_means(
        self,
        flat: Array,
        values: Mapping[str, Array],
        *,
        weights: Array | None = None,
    ) -> dict[str, Array]:
        """Compute all per-bin means with one bincount per variable."""
        flat = np.asarray(flat, dtype=np.int64)
        valid_bin = (flat >= 0) & (flat < self.size)
        if weights is None:
            event_weights = np.ones(flat.shape, dtype=float)
        else:
            event_weights = np.asarray(weights, dtype=float)
            if event_weights.shape != flat.shape:
                raise ValueError("weights shape does not match flat indices")
            if not np.all(np.isfinite(event_weights)) or np.any(event_weights < 0.0):
                raise ValueError("weights must be finite and nonnegative")
        output: dict[str, Array] = {}
        for name, raw in values.items():
            raw = np.asarray(raw, dtype=float)
            if raw.shape != flat.shape:
                raise ValueError(f"{name} shape does not match flat indices")
            valid = valid_bin & np.isfinite(raw) & (event_weights > 0.0)
            weight_sums = np.bincount(
                flat[valid], weights=event_weights[valid], minlength=self.size
            )
            sums = np.bincount(
                flat[valid], weights=raw[valid] * event_weights[valid], minlength=self.size
            )
            output[name] = np.divide(
                sums,
                weight_sums,
                out=np.full(self.size, np.nan),
                where=weight_sums > 0,
            )
        return output


def from_config(path: str | Path) -> AnalysisBinning:
    with Path(path).open(encoding="utf-8") as source:
        config = json.load(source)["binning"]
    return AnalysisBinning(
        config["Q2"], config["xB"], config["minus_t"], config["phi_deg"]
    )


def legacy_binning() -> AnalysisBinning:
    return AnalysisBinning(
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.6, 5.5, 7.0, 10.5],
        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.38, 0.48, 0.58, 0.7],
        [0.09, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0, 1.5, 2.0],
        np.linspace(0.0, 360.0, 21),
    )
