#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.event_sample import (
    build_generated_sample,
    generated_particle_columns,
    generated_sample_from_tree,
    join_reconstructed,
)


GEN_COLUMNS = [
    "event.runNum",
    "event.eventNum",
    "gen.pid",
    "gen.p",
    "gen.theta",
    "gen.phi",
]

GEN_SOURCE_COLUMNS = ["event.sourceFileId", "event.sourceEventIndex"]

GENERATED_SOURCE_COLUMNS = ["sourceFileId", "sourceEventIndex"]

GENERATED_EVENT_COLUMNS = [
    "runNum",
    "eventNum",
    "topologyValid",
    "Q2",
    "xB",
    "minusT",
    "trentoPhi",
    "radiative",
    "weight",
]

GENERATED_EVENT_PARTICLE_COLUMNS = [
    "electronP",
    "electronTheta",
    "electronPhi",
    "protonP",
    "protonTheta",
    "protonPhi",
    "gamma1P",
    "gamma1Theta",
    "gamma1Phi",
    "gamma2P",
    "gamma2Theta",
    "gamma2Phi",
    "pi0P",
    "pi0Theta",
    "pi0Phi",
]

REC_SOURCE_COLUMNS = ["sourceFileId", "sourceEventIndex"]

REC_KEY_COLUMNS = {"runNum", "eventNum", *REC_SOURCE_COLUMNS}

SELECTED_PARTICLE_COLUMNS = (
    "selectedRoles",
    "selectedIdx",
    "selectedPid",
    "selectedDet",
    "selectedSector",
    "selectedP",
    "selectedTheta",
    "selectedPhi",
)

SELECTED_PARTICLE_FIELDS = (
    ("selectedIdx", "Idx", np.int64, -999),
    ("selectedPid", "Pid", np.int64, -999),
    ("selectedDet", "Det", np.int64, -999),
    ("selectedSector", "Sector", np.int64, -999),
    ("selectedP", "P", float, np.nan),
    ("selectedTheta", "Theta", float, np.nan),
    ("selectedPhi", "Phi", float, np.nan),
)

REC_COLUMN_ALIASES = {
    "Q2": "rec_Q2",
    "xB": "rec_xB",
    "t": "rec_minus_t",
    "t_pi0": "rec_minus_t_pi0",
    "trentoPhi": "rec_trento_phi",
    "pDet": "rec_proton_detector",
    "m_gg": "rec_m_gg",
    "m2_miss": "rec_m2_miss",
    "m2_epX": "rec_m2_epX",
    "m_eggX": "rec_m_eggX",
    "E_miss": "rec_E_miss",
    "pT_miss": "rec_pT_miss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join all generated EPPI0 events to selected reconstructed candidates."
    )
    parser.add_argument("matched_root", type=Path, help="Converter ROOT file")
    parser.add_argument("selected_root", type=Path, help="Output from post_process")
    parser.add_argument("output", type=Path, help="Compact event-level .npz output")
    parser.add_argument("--beam-energy", type=float,
                        help="Required only for legacy particle-level GEN input")
    parser.add_argument("--dictionary", type=Path, help="Optional ROOT dictionary shared library")
    parser.add_argument("--tree", default="Events")
    parser.add_argument("--generated-tree", default="GeneratedEvents")
    parser.add_argument(
        "--matched-only",
        action="store_true",
        help=(
            "Reverse the join for visualization: stream GeneratedEvents and write only "
            "selected REC candidates with a valid generated-event match"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="GeneratedEvents rows per chunk in --matched-only mode (default: 250000)",
    )
    parser.add_argument(
        "--progress-chunks",
        type=int,
        default=4,
        help="Print matched-only progress every N chunks; use 0 to disable (default: 4)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.progress_chunks < 0:
        raise ValueError("--progress-chunks must be non-negative")
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if args.dictionary:
        status = ROOT.gSystem.Load(str(args.dictionary.resolve()))
        if status < 0:
            raise RuntimeError(f"Could not load ROOT dictionary: {args.dictionary}")

    matched_path = str(args.matched_root.resolve())
    input_file = ROOT.TFile.Open(matched_path, "READ")
    if not input_file or input_file.IsZombie():
        raise RuntimeError(f"Could not open converter ROOT file: {matched_path}")
    generated_tree = input_file.Get(args.generated_tree)
    has_generated_tree = bool(generated_tree)
    has_generated_source_key = has_generated_tree and all(
        generated_tree.GetBranch(name) for name in GENERATED_SOURCE_COLUMNS
    )
    generated_event_particle_columns = [
        name for name in GENERATED_EVENT_PARTICLE_COLUMNS
        if has_generated_tree and generated_tree.GetBranch(name)
    ]
    generated_entries = int(generated_tree.GetEntries()) if has_generated_tree else 0
    input_file.Close()

    selected_path = str(args.selected_root.resolve())
    selected_file = ROOT.TFile.Open(selected_path, "READ")
    if not selected_file or selected_file.IsZombie():
        raise RuntimeError(f"Could not open selected ROOT file: {selected_path}")
    selected_tree = selected_file.Get(args.tree)
    if not selected_tree:
        selected_file.Close()
        raise RuntimeError(f"Could not find tree {args.tree} in {selected_path}")
    has_source_key = all(selected_tree.GetBranch(name) for name in REC_SOURCE_COLUMNS)
    requested_rec_columns = scalar_branch_names(selected_tree)
    selected_particle_columns = [
        name for name in SELECTED_PARTICLE_COLUMNS if selected_tree.GetBranch(name)
    ]
    selected_file.Close()
    missing_keys = [name for name in ("runNum", "eventNum") if name not in requested_rec_columns]
    if missing_keys:
        raise RuntimeError(f"Selected tree is missing required branches: {missing_keys}")
    if has_source_key:
        for name in REC_SOURCE_COLUMNS:
            if name not in requested_rec_columns:
                requested_rec_columns.append(name)
    rec_input_columns = requested_rec_columns + [
        name for name in selected_particle_columns if name not in requested_rec_columns
    ]
    rec = ROOT.RDataFrame(args.tree, selected_path).AsNumpy(rec_input_columns)
    rec_values = reconstructed_columns(rec, requested_rec_columns)
    rec_values.update(expand_selected_particle_columns(rec))

    if args.matched_only:
        if not has_generated_tree:
            raise ValueError("--matched-only requires the compact GeneratedEvents tree")
        if not has_generated_source_key or not has_source_key:
            raise ValueError(
                "--matched-only requires sourceFileId and sourceEventIndex in both ROOT trees"
            )
        requested_gen_columns = (
            GENERATED_EVENT_COLUMNS + GENERATED_SOURCE_COLUMNS + generated_event_particle_columns
        )
        chunks = generated_tree_chunks(
            ROOT,
            args.generated_tree,
            matched_path,
            requested_gen_columns,
            generated_entries,
            args.chunk_size,
            args.progress_chunks,
        )
        sample, stats = reverse_join_selected_events(
            rec,
            rec_values,
            chunks,
            generated_event_particle_columns,
        )
        metadata = {
            "beam_energy": args.beam_energy,
            "generated_source": args.generated_tree,
            "generated_particle_columns": [
                f"gen_{name}" for name in generated_event_particle_columns
            ],
            "matched_root": str(args.matched_root.resolve()),
            "selected_root": str(args.selected_root.resolve()),
            "join_mode": "selected-left matched-only",
            "generated_events_scanned": stats["generated_events_scanned"],
            "valid_generated_events": stats["valid_generated_events"],
            "selected_reconstructed_events": stats["selected_reconstructed_events"],
            "matched_generated_events": stats["matched_generated_events"],
            "unmatched_selected_events": stats["unmatched_selected_events"],
            "invalid_generated_matches": stats["invalid_generated_matches"],
            "reconstructed_columns": sorted(rec_values),
            "schema_version": 6,
        }
        write_sample(args.output, sample, metadata)
        print(f"Generated events scanned: {stats['generated_events_scanned']}")
        print(f"Valid generated events: {stats['valid_generated_events']}")
        print(f"Selected REC candidates: {stats['selected_reconstructed_events']}")
        print(f"Matched rows exported: {stats['matched_generated_events']}")
        print(f"Selected candidates without GEN row: {stats['unmatched_selected_events']}")
        print(f"Selected candidates with invalid GEN topology: {stats['invalid_generated_matches']}")
        print(f"GEN particle variables carried: {len(generated_event_particle_columns)}")
        print(f"REC variables carried: {len(rec_values)}")
        print(f"Wrote {args.output}")
        return 0

    if has_generated_tree:
        requested_gen_columns = (
            GENERATED_EVENT_COLUMNS
            + (GENERATED_SOURCE_COLUMNS if has_generated_source_key else [])
            + generated_event_particle_columns
        )
        gen = ROOT.RDataFrame(args.generated_tree, matched_path).AsNumpy(requested_gen_columns)
        generated_rows = np.asarray(gen["runNum"]).size
        generated_mask = generated_event_mask(gen)
        generated = generated_sample_from_tree(
            gen["sourceFileId"] if has_generated_source_key else np.full(
                generated_rows, np.iinfo(np.uint64).max, dtype=np.uint64
            ),
            gen["sourceEventIndex"] if has_generated_source_key else np.full(
                generated_rows, np.iinfo(np.uint64).max, dtype=np.uint64
            ),
            gen["runNum"],
            gen["eventNum"],
            gen["topologyValid"],
            gen["Q2"],
            gen["xB"],
            gen["minusT"],
            gen["trentoPhi"],
            gen["radiative"],
            gen["weight"],
        )
        generated_source = args.generated_tree
        generated_values = {
            f"gen_{name}": np.asarray(gen[name], dtype=float)[generated_mask]
            for name in generated_event_particle_columns
        }
    else:
        if args.beam_energy is None:
            raise ValueError("--beam-energy is required when GeneratedEvents is absent")
        gen = ROOT.RDataFrame(args.tree, matched_path).AsNumpy(GEN_COLUMNS)
        generated = build_generated_sample(
            gen["event.runNum"],
            gen["event.eventNum"],
            gen["gen.pid"],
            gen["gen.p"],
            gen["gen.theta"],
            gen["gen.phi"],
            args.beam_energy,
        )
        generated_source = f"{args.tree}.gen (legacy)"
        generated_values = {}

    if not generated_values:
        generated_values = read_generated_particle_columns(
            ROOT,
            matched_path,
            args.tree,
            generated,
            prefer_source_keys=has_generated_source_key,
        )

    sample = join_reconstructed(
        generated,
        rec["runNum"],
        rec["eventNum"],
        rec_values,
        rec_source_file_id=rec["sourceFileId"] if has_source_key else None,
        rec_source_event_index=rec["sourceEventIndex"] if has_source_key else None,
    )
    sample.update(generated_values)
    metadata = {
        "beam_energy": args.beam_energy,
        "generated_source": generated_source,
        "generated_particle_columns": sorted(generated_values),
        "matched_root": str(args.matched_root.resolve()),
        "selected_root": str(args.selected_root.resolve()),
        "generated_events": int(generated.run.size),
        "selected_reconstructed_events": int(sample["rec_selected"].sum()),
        "reconstructed_columns": sorted(rec_values),
        "schema_version": 5,
    }
    write_sample(args.output, sample, metadata)
    print(f"Generated events: {generated.run.size}")
    print(f"Selected REC matches: {sample['rec_selected'].sum()}")
    print(f"GEN particle variables carried: {len(generated_values)}")
    print(f"REC variables carried: {len(rec_values)}")
    print(f"Wrote {args.output}")
    return 0


def generated_tree_chunks(
    ROOT,
    tree_name: str,
    root_path: str,
    columns: list[str],
    entries: int,
    chunk_size: int,
    progress_chunks: int,
) -> Iterable[dict[str, np.ndarray]]:
    chunks = (entries + chunk_size - 1) // chunk_size
    for chunk_index, start in enumerate(range(0, entries, chunk_size), start=1):
        stop = min(start + chunk_size, entries)
        yield ROOT.RDataFrame(tree_name, root_path).Range(start, stop).AsNumpy(columns)
        if progress_chunks and (chunk_index % progress_chunks == 0 or chunk_index == chunks):
            print(f"GeneratedEvents chunks: {chunk_index}/{chunks} ({stop}/{entries} rows)")


def reverse_join_selected_events(
    rec: dict[str, np.ndarray],
    rec_values: dict[str, np.ndarray],
    generated_chunks: Iterable[dict[str, np.ndarray]],
    particle_columns: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Left-join chunked GeneratedEvents data onto selected REC candidates."""
    selected_source_file_id = np.asarray(rec["sourceFileId"], dtype=np.uint64)
    selected_source_event_index = np.asarray(rec["sourceEventIndex"], dtype=np.uint64)
    selected_rows = selected_source_file_id.size
    if selected_source_event_index.shape != selected_source_file_id.shape:
        raise ValueError("selected source-event key branches must have equal shapes")
    selected_keys = source_keys(selected_source_file_id, selected_source_event_index)
    if np.unique(selected_keys).size != selected_rows:
        raise ValueError("selected REC sample contains duplicate (sourceFileId,sourceEventIndex) keys")
    selected_order = np.argsort(selected_keys, order=selected_keys.dtype.names)
    sorted_selected_keys = selected_keys[selected_order]

    sample: dict[str, np.ndarray] = {
        "source_file_id": selected_source_file_id.copy(),
        "source_event_index": selected_source_event_index.copy(),
        "run": np.full(selected_rows, -999, dtype=np.int64),
        "event": np.full(selected_rows, -999, dtype=np.int64),
        "gen_Q2": np.full(selected_rows, np.nan),
        "gen_xB": np.full(selected_rows, np.nan),
        "gen_minus_t": np.full(selected_rows, np.nan),
        "gen_trento_phi": np.full(selected_rows, np.nan),
        "gen_radiative": np.zeros(selected_rows, dtype=bool),
        "gen_weight": np.full(selected_rows, np.nan),
        "rec_selected": np.ones(selected_rows, dtype=bool),
    }
    for name in particle_columns:
        sample[f"gen_{name}"] = np.full(selected_rows, np.nan)
    for name, values in rec_values.items():
        values = np.asarray(values)
        if values.shape != selected_source_file_id.shape:
            raise ValueError(f"REC column {name} does not match selected event keys")
        sample[name] = values.copy()

    seen = np.zeros(selected_rows, dtype=bool)
    valid_matches = np.zeros(selected_rows, dtype=bool)
    generated_events_scanned = 0
    valid_generated_events = 0
    branch_map = {
        "runNum": "run",
        "eventNum": "event",
        "Q2": "gen_Q2",
        "xB": "gen_xB",
        "minusT": "gen_minus_t",
        "trentoPhi": "gen_trento_phi",
        "radiative": "gen_radiative",
        "weight": "gen_weight",
    }

    for gen in generated_chunks:
        chunk_rows = np.asarray(gen["sourceFileId"]).size
        generated_events_scanned += chunk_rows
        for name, values in gen.items():
            if np.asarray(values).size != chunk_rows:
                raise ValueError(f"GeneratedEvents chunk column {name} has inconsistent size")
        valid = generated_event_mask(gen)
        valid_generated_events += int(np.count_nonzero(valid))
        chunk_keys = source_keys(gen["sourceFileId"], gen["sourceEventIndex"])
        positions = np.searchsorted(sorted_selected_keys, chunk_keys)
        bounded = positions < sorted_selected_keys.size
        matched = np.zeros(chunk_rows, dtype=bool)
        matched[bounded] = sorted_selected_keys[positions[bounded]] == chunk_keys[bounded]
        if not np.any(matched):
            continue
        target_rows = selected_order[positions[matched]]
        if np.unique(target_rows).size != target_rows.size or np.any(seen[target_rows]):
            raise ValueError("GeneratedEvents contains duplicate selected source-event keys")
        seen[target_rows] = True
        matched_chunk_rows = np.flatnonzero(matched)
        matched_valid = valid[matched_chunk_rows]
        target_rows = target_rows[matched_valid]
        source_rows = matched_chunk_rows[matched_valid]
        valid_matches[target_rows] = True
        for source_name, output_name in branch_map.items():
            sample[output_name][target_rows] = np.asarray(gen[source_name])[source_rows]
        for name in particle_columns:
            sample[f"gen_{name}"][target_rows] = np.asarray(gen[name], dtype=float)[source_rows]

    keep = valid_matches
    filtered = {name: np.asarray(values)[keep] for name, values in sample.items()}
    stats = {
        "generated_events_scanned": generated_events_scanned,
        "valid_generated_events": valid_generated_events,
        "selected_reconstructed_events": selected_rows,
        "matched_generated_events": int(np.count_nonzero(keep)),
        "unmatched_selected_events": int(np.count_nonzero(~seen)),
        "invalid_generated_matches": int(np.count_nonzero(seen & ~valid_matches)),
    }
    return filtered, stats


def source_keys(source_file_id: np.ndarray, source_event_index: np.ndarray) -> np.ndarray:
    keys = np.empty(
        np.asarray(source_file_id).size,
        dtype=[("source_file_id", "<u8"), ("source_event_index", "<u8")],
    )
    keys["source_file_id"] = np.asarray(source_file_id, dtype=np.uint64)
    keys["source_event_index"] = np.asarray(source_event_index, dtype=np.uint64)
    return keys


def write_sample(output: Path, sample: dict[str, np.ndarray], metadata: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **sample, metadata_json=json.dumps(metadata, sort_keys=True))


def generated_event_mask(gen: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.asarray(gen["topologyValid"], dtype=bool)
    for name in ("Q2", "xB", "minusT", "trentoPhi"):
        valid &= np.isfinite(np.asarray(gen[name], dtype=float))
    return valid


def read_generated_particle_columns(ROOT, root_path: str, tree_name: str, generated, prefer_source_keys: bool) -> dict[str, np.ndarray]:
    column_sets = []
    if prefer_source_keys:
        column_sets.append(GEN_SOURCE_COLUMNS + GEN_COLUMNS)
    column_sets.append(GEN_COLUMNS)
    if not prefer_source_keys:
        column_sets.append(GEN_SOURCE_COLUMNS + GEN_COLUMNS)

    last_error: Exception | None = None
    for columns in column_sets:
        try:
            raw = ROOT.RDataFrame(tree_name, root_path).AsNumpy(columns)
        except Exception as exc:  # ROOT raises RuntimeError for missing dictionaries/branches.
            last_error = exc
            continue
        has_source = all(name in raw for name in GEN_SOURCE_COLUMNS)
        return generated_particle_columns(
            raw["event.runNum"],
            raw["event.eventNum"],
            raw["gen.pid"],
            raw["gen.p"],
            raw["gen.theta"],
            raw["gen.phi"],
            generated.run,
            generated.event,
            source_file_id=raw["event.sourceFileId"] if has_source else None,
            source_event_index=raw["event.sourceEventIndex"] if has_source else None,
            target_source_file_id=generated.source_file_id if has_source else None,
            target_source_event_index=generated.source_event_index if has_source else None,
        )
    message = f"Warning: could not read per-particle GEN kinematics from {tree_name}.gen"
    if last_error is not None:
        message += f": {last_error}"
    print(message, file=sys.stderr)
    return {}


def scalar_branch_names(tree) -> list[str]:
    names: list[str] = []
    for branch in tree.GetListOfBranches():
        name = branch.GetName()
        class_name = str(branch.GetClassName() or "")
        if class_name:
            continue
        leaves = branch.GetListOfLeaves()
        if not leaves or leaves.GetEntries() != 1:
            continue
        names.append(name)
    return names


def reconstructed_columns(rec: dict[str, np.ndarray], columns: list[str]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in columns:
        if name in REC_KEY_COLUMNS:
            continue
        if name not in rec:
            continue
        output_name = REC_COLUMN_ALIASES.get(name, f"rec_{name}")
        output.setdefault(output_name, rec[name])
    return output


def expand_selected_particle_columns(rec: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Expand role-aligned selected-particle vectors into symmetric REC scalars."""
    if "selectedRoles" not in rec:
        return {}
    roles_by_row = [vector_values(value) for value in rec["selectedRoles"]]
    row_count = len(roles_by_row)
    available_fields = [field for field in SELECTED_PARTICLE_FIELDS if field[0] in rec]
    for branch_name, _, _, _ in available_fields:
        if len(rec[branch_name]) != row_count:
            raise ValueError(f"REC vector column {branch_name} does not match selectedRoles")

    maximum_occurrences: dict[str, int] = {}
    sanitized_roles: list[list[str]] = []
    for roles in roles_by_row:
        row_roles = [sanitize_role_name(role) for role in roles]
        sanitized_roles.append(row_roles)
        row_counts: dict[str, int] = {}
        for role in row_roles:
            if not role:
                continue
            row_counts[role] = row_counts.get(role, 0) + 1
        for role, count in row_counts.items():
            maximum_occurrences[role] = max(maximum_occurrences.get(role, 0), count)

    output: dict[str, np.ndarray] = {}
    for role, count in maximum_occurrences.items():
        for occurrence in range(1, count + 1):
            role_name = role if count == 1 else f"{role}{occurrence}"
            for _, suffix, dtype, missing in available_fields:
                output[f"rec_{role_name}{suffix}"] = np.full(row_count, missing, dtype=dtype)

    vectors = {
        branch_name: [vector_values(value) for value in rec[branch_name]]
        for branch_name, _, _, _ in available_fields
    }
    for row, roles in enumerate(sanitized_roles):
        occurrences: dict[str, int] = {}
        for index, role in enumerate(roles):
            if not role:
                continue
            occurrence = occurrences.get(role, 0) + 1
            occurrences[role] = occurrence
            role_name = role if maximum_occurrences[role] == 1 else f"{role}{occurrence}"
            for branch_name, suffix, dtype, _ in available_fields:
                values = vectors[branch_name][row]
                if index >= len(values):
                    continue
                try:
                    output[f"rec_{role_name}{suffix}"][row] = dtype(values[index])
                except (TypeError, ValueError, OverflowError):
                    continue
    return output


def vector_values(value) -> list:
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return []


def sanitize_role_name(value) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
