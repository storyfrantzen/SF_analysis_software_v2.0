from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray

SELECTED_FIELDS = (
    "source_file_id",
    "source_event_index",
    "run",
    "event",
    "role",
    "occurrence",
    "particle_index",
    "pid",
    "detector",
    "sector",
    "rec_p",
    "rec_theta",
    "rec_phi",
)

RECONSTRUCTED_FIELDS = (
    "source_file_id",
    "source_event_index",
    "particle_index",
    "pid",
    "detector",
    "sector",
    "rec_p",
    "rec_theta",
    "rec_phi",
    "matched_gen_index",
    "match_angle_deg",
    "gen_pid",
    "gen_p",
    "gen_theta",
    "gen_phi",
)


@dataclass(frozen=True)
class ParticleMatchJoinStats:
    selected_rows: int
    converter_rows_scanned: int
    converter_rows_found: int
    generated_matches: int
    generated_unmatched: int


def join_selected_particle_matches(
    selected: dict[str, Array],
    reconstructed_chunks: Iterable[dict[str, Array]],
    *,
    kinematic_rtol: float = 1.0e-10,
    kinematic_atol: float = 1.0e-12,
) -> tuple[dict[str, Array], ParticleMatchJoinStats]:
    """Join selected role rows to their converter REC/GEN particle rows.

    The source-aware event key is required because run/event identifiers can repeat
    across concatenated GEMC files.  The selected particle index identifies the
    exact reconstructed row chosen by post-processing.  All selected rows are
    retained, including rows without a valid generated-particle match.
    """
    _validate_table(selected, SELECTED_FIELDS, "selected particle table")
    selected_rows = _table_size(selected)
    selected_keys = particle_keys(
        selected["source_file_id"],
        selected["source_event_index"],
        selected["particle_index"],
    )
    selected_order = np.argsort(selected_keys, order=selected_keys.dtype.names)
    sorted_selected_keys = selected_keys[selected_order]
    if selected_rows > 1 and np.any(
        sorted_selected_keys[1:] == sorted_selected_keys[:-1]
    ):
        raise ValueError(
            "selected particle table contains duplicate "
            "(sourceFileId, sourceEventIndex, particleIdx) keys"
        )

    output = _initial_output(selected)
    found = output["converter_row_found"]
    converter_rows_scanned = 0

    for chunk_index, chunk in enumerate(reconstructed_chunks, start=1):
        _validate_table(
            chunk,
            RECONSTRUCTED_FIELDS,
            f"reconstructed particle chunk {chunk_index}",
        )
        chunk_rows = _table_size(chunk)
        converter_rows_scanned += chunk_rows
        if selected_rows == 0 or chunk_rows == 0:
            continue

        chunk_keys = particle_keys(
            chunk["source_file_id"],
            chunk["source_event_index"],
            chunk["particle_index"],
        )
        positions = np.searchsorted(sorted_selected_keys, chunk_keys)
        bounded = positions < selected_rows
        matches = np.zeros(chunk_rows, dtype=bool)
        matches[bounded] = (
            sorted_selected_keys[positions[bounded]] == chunk_keys[bounded]
        )
        if not np.any(matches):
            continue

        source_rows = np.flatnonzero(matches)
        target_rows = selected_order[positions[matches]]
        if np.unique(target_rows).size != target_rows.size or np.any(found[target_rows]):
            raise ValueError(
                "converter particle table contains duplicate selected-particle keys"
            )
        _validate_selected_identity(selected, chunk, target_rows, source_rows)
        _validate_selected_kinematics(
            selected,
            chunk,
            target_rows,
            source_rows,
            rtol=kinematic_rtol,
            atol=kinematic_atol,
        )

        found[target_rows] = True
        for name in (
            "matched_gen_index",
            "match_angle_deg",
            "gen_pid",
            "gen_p",
            "gen_theta",
            "gen_phi",
        ):
            output[name][target_rows] = np.asarray(chunk[name])[source_rows]

    missing = np.flatnonzero(~found)
    if missing.size:
        first = int(missing[0])
        raise ValueError(
            f"converter particle table is missing {missing.size} selected rows; "
            f"first missing key={_format_key(selected_keys[first])}"
        )

    claimed_match = output["matched_gen_index"] >= 0
    finite_gen = (
        np.isfinite(output["gen_p"])
        & np.isfinite(output["gen_theta"])
        & np.isfinite(output["gen_phi"])
    )
    bad_pid = claimed_match & (output["gen_pid"] != output["pid"])
    bad_kinematics = claimed_match & ~finite_gen
    if np.any(bad_pid):
        raise ValueError(
            f"{np.count_nonzero(bad_pid)} selected rows have mismatched REC/GEN PIDs"
        )
    if np.any(bad_kinematics):
        raise ValueError(
            f"{np.count_nonzero(bad_kinematics)} selected rows claim a GEN match "
            "but have non-finite generated kinematics"
        )
    output["gen_matched"] = claimed_match & finite_gen

    generated_matches = int(np.count_nonzero(output["gen_matched"]))
    stats = ParticleMatchJoinStats(
        selected_rows=selected_rows,
        converter_rows_scanned=converter_rows_scanned,
        converter_rows_found=int(np.count_nonzero(found)),
        generated_matches=generated_matches,
        generated_unmatched=selected_rows - generated_matches,
    )
    return output, stats


def particle_match_summary(
    matches: dict[str, Array],
    stats: ParticleMatchJoinStats,
    *,
    warning_angle_deg: float = 2.5,
) -> dict[str, object]:
    """Build JSON-ready overall and role/detector matching diagnostics."""
    if not np.isfinite(warning_angle_deg) or warning_angle_deg < 0.0:
        raise ValueError("warning_angle_deg must be finite and non-negative")
    rows = _table_size(matches)
    if rows != stats.selected_rows:
        raise ValueError("particle-match rows do not agree with join statistics")

    role = np.asarray(matches["role"]).astype(str)
    detector = np.asarray(matches["detector"], dtype=np.int64)
    matched = np.asarray(matches["gen_matched"], dtype=bool)
    angle = np.asarray(matches["match_angle_deg"], dtype=float)
    groups = []
    for role_name in sorted(np.unique(role).tolist()):
        role_mask = role == role_name
        for detector_id in sorted(np.unique(detector[role_mask]).tolist()):
            mask = role_mask & (detector == detector_id)
            groups.append(
                _summary_group(
                    role_name,
                    int(detector_id),
                    mask,
                    matched,
                    angle,
                    warning_angle_deg,
                )
            )

    matched_angles = angle[matched & np.isfinite(angle)]
    return {
        "selected_rows": stats.selected_rows,
        "converter_rows_scanned": stats.converter_rows_scanned,
        "converter_rows_found": stats.converter_rows_found,
        "generated_matches": stats.generated_matches,
        "generated_unmatched": stats.generated_unmatched,
        "generated_match_fraction": (
            stats.generated_matches / stats.selected_rows
            if stats.selected_rows
            else None
        ),
        "warning_angle_deg": float(warning_angle_deg),
        "match_angle_median_deg": _percentile_or_none(matched_angles, 50.0),
        "match_angle_p95_deg": _percentile_or_none(matched_angles, 95.0),
        "match_angle_warning_fraction": (
            float(np.mean(matched_angles >= warning_angle_deg))
            if matched_angles.size
            else None
        ),
        "groups": groups,
    }


def particle_keys(
    source_file_id: Array, source_event_index: Array, particle_index: Array
) -> Array:
    source_file_id = np.asarray(source_file_id, dtype=np.uint64)
    source_event_index = np.asarray(source_event_index, dtype=np.uint64)
    particle_index = np.asarray(particle_index, dtype=np.int64)
    if not (
        source_file_id.shape == source_event_index.shape == particle_index.shape
    ):
        raise ValueError("particle-key arrays must have equal shapes")
    keys = np.empty(
        source_file_id.size,
        dtype=[
            ("source_file_id", "<u8"),
            ("source_event_index", "<u8"),
            ("particle_index", "<i8"),
        ],
    )
    keys["source_file_id"] = source_file_id
    keys["source_event_index"] = source_event_index
    keys["particle_index"] = particle_index
    return keys


def _initial_output(selected: dict[str, Array]) -> dict[str, Array]:
    rows = _table_size(selected)
    output = {name: np.asarray(selected[name]).copy() for name in SELECTED_FIELDS}
    output.update(
        converter_row_found=np.zeros(rows, dtype=bool),
        matched_gen_index=np.full(rows, -999, dtype=np.int64),
        match_angle_deg=np.full(rows, np.nan),
        gen_pid=np.full(rows, -999, dtype=np.int64),
        gen_p=np.full(rows, np.nan),
        gen_theta=np.full(rows, np.nan),
        gen_phi=np.full(rows, np.nan),
        gen_matched=np.zeros(rows, dtype=bool),
    )
    return output


def _validate_table(
    table: dict[str, Array], required: tuple[str, ...], description: str
) -> None:
    missing = sorted(set(required) - set(table))
    if missing:
        raise ValueError(f"{description} is missing fields: {missing}")
    rows = _table_size(table)
    inconsistent = [
        name for name in required if np.asarray(table[name]).size != rows
    ]
    if inconsistent:
        raise ValueError(
            f"{description} has inconsistent field lengths: {inconsistent}"
        )


def _table_size(table: dict[str, Array]) -> int:
    if not table:
        return 0
    return int(np.asarray(next(iter(table.values()))).size)


def _validate_selected_identity(
    selected: dict[str, Array],
    chunk: dict[str, Array],
    target_rows: Array,
    source_rows: Array,
) -> None:
    for name in ("pid", "detector", "sector"):
        left = np.asarray(selected[name])[target_rows]
        right = np.asarray(chunk[name])[source_rows]
        mismatch = left != right
        if np.any(mismatch):
            raise ValueError(
                f"selected and converter particle {name} differ for "
                f"{np.count_nonzero(mismatch)} joined rows"
            )


def _validate_selected_kinematics(
    selected: dict[str, Array],
    chunk: dict[str, Array],
    target_rows: Array,
    source_rows: Array,
    *,
    rtol: float,
    atol: float,
) -> None:
    for name in ("rec_p", "rec_theta", "rec_phi"):
        left = np.asarray(selected[name], dtype=float)[target_rows]
        right = np.asarray(chunk[name], dtype=float)[source_rows]
        close = np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
        if not np.all(close):
            difference = np.abs(left[~close] - right[~close])
            maximum = float(np.nanmax(difference)) if difference.size else np.nan
            raise ValueError(
                f"selected and converter {name} differ for "
                f"{np.count_nonzero(~close)} joined rows; maximum absolute "
                f"difference={maximum:.6g}"
            )


def _summary_group(
    role: str,
    detector: int,
    mask: Array,
    matched: Array,
    angle: Array,
    warning_angle_deg: float,
) -> dict[str, object]:
    selected_rows = int(np.count_nonzero(mask))
    group_matched = mask & matched
    matched_rows = int(np.count_nonzero(group_matched))
    angles = angle[group_matched & np.isfinite(angle)]
    return {
        "role": role,
        "detector": detector,
        "selected_rows": selected_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": selected_rows - matched_rows,
        "match_fraction": matched_rows / selected_rows if selected_rows else None,
        "match_angle_median_deg": _percentile_or_none(angles, 50.0),
        "match_angle_p95_deg": _percentile_or_none(angles, 95.0),
        "match_angle_warning_fraction": (
            float(np.mean(angles >= warning_angle_deg)) if angles.size else None
        ),
    }


def _percentile_or_none(values: Array, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def _format_key(key: np.void) -> str:
    return (
        f"({int(key['source_file_id'])}, "
        f"{int(key['source_event_index'])}, "
        f"{int(key['particle_index'])})"
    )
