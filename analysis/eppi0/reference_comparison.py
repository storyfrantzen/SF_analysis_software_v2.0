"""Match published structure functions to configured analysis bins."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np


Array = np.ndarray
STRUCTURE_FUNCTIONS = ("sigma_U", "sigma_LT", "sigma_TT")


@dataclass(frozen=True)
class ReferenceTable:
    q2: Array
    xb: Array
    minus_t: Array
    values: Array
    statistical: Array
    systematic: Array


@dataclass(frozen=True)
class ReferenceBinMatch:
    q2_index: Array
    xb_index: Array
    t_index: Array


def load_reference_table(path: str | Path) -> ReferenceTable:
    """Load the documented CLAS6-style structure-function CSV schema."""

    required = ["Q2", "xB", "minus_t"]
    for name in STRUCTURE_FUNCTIONS:
        required.extend((name, f"{name}_stat", f"{name}_sys"))
    columns = {name: [] for name in required}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(set(required) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"reference table is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            try:
                for name in required:
                    columns[name].append(float(row[name]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"reference table row {row_number} contains a nonnumeric value"
                ) from exc
    if not columns["Q2"]:
        raise ValueError("reference table contains no rows")

    values = np.column_stack(
        [np.asarray(columns[name], dtype=float) for name in STRUCTURE_FUNCTIONS]
    )
    statistical = np.column_stack(
        [
            np.asarray(columns[f"{name}_stat"], dtype=float)
            for name in STRUCTURE_FUNCTIONS
        ]
    )
    systematic = np.column_stack(
        [
            np.asarray(columns[f"{name}_sys"], dtype=float)
            for name in STRUCTURE_FUNCTIONS
        ]
    )
    if not (
        np.all(np.isfinite(values))
        and np.all(np.isfinite(statistical))
        and np.all(np.isfinite(systematic))
        and np.all(statistical >= 0.0)
        and np.all(systematic >= 0.0)
    ):
        raise ValueError("reference values and uncertainties must be finite")
    return ReferenceTable(
        q2=np.asarray(columns["Q2"], dtype=float),
        xb=np.asarray(columns["xB"], dtype=float),
        minus_t=np.asarray(columns["minus_t"], dtype=float),
        values=values,
        statistical=statistical,
        systematic=systematic,
    )


def _indices(edges: Array, coordinates: Array, label: str) -> Array:
    edges = np.asarray(edges, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"{label} edges must be a strictly increasing vector")
    result = np.searchsorted(edges, coordinates, side="right") - 1
    inside = (
        np.isfinite(coordinates)
        & (coordinates >= edges[0])
        & (coordinates < edges[-1])
    )
    if not np.all(inside):
        bad = np.flatnonzero(~inside).tolist()
        raise ValueError(f"reference {label} coordinates outside analysis bins: {bad}")
    return np.asarray(result, dtype=np.int64)


def match_reference_bins(
    table: ReferenceTable,
    *,
    q2_edges: Array,
    xb_edges: Array,
    t_edges: Array,
) -> ReferenceBinMatch:
    """Assign every reference point to one configured three-dimensional bin."""

    match = ReferenceBinMatch(
        q2_index=_indices(q2_edges, table.q2, "Q2"),
        xb_index=_indices(xb_edges, table.xb, "xB"),
        t_index=_indices(t_edges, table.minus_t, "-t"),
    )
    keys = list(zip(match.q2_index, match.xb_index, match.t_index))
    if len(keys) != len(set(keys)):
        raise ValueError("reference table has multiple rows in one analysis bin")
    return match
