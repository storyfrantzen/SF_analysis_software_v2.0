from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .binning import AnalysisBinning
from .response import ResponseResult, build_response_from_counts


GENERATED_COLUMNS = [
    "sourceFileId",
    "sourceEventIndex",
    "topologyValid",
    "Q2",
    "xB",
    "minusT",
    "trentoPhi",
    "weight",
]

SELECTED_COLUMNS = [
    "sourceFileId",
    "sourceEventIndex",
    "Q2",
    "xB",
    "t",
    "trentoPhi",
]


@dataclass(frozen=True)
class RootResponseSummary:
    response: ResponseResult
    generated_rows: int
    selected_rows: int
    matched_selected_rows: int


def build_response_from_root(
    converter_root: Path,
    selected_root: Path,
    binning: AnalysisBinning,
    dictionary: Path | None = None,
    tree: str = "Events",
    generated_tree: str = "GeneratedEvents",
    chunk_size: int = 1_000_000,
    selection_mask: np.ndarray | None = None,
    progress_chunks: int = 10,
) -> RootResponseSummary:
    """Build a response directly from ROOT files without a dense event-level NPZ."""
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if dictionary is not None:
        status = ROOT.gSystem.Load(str(dictionary.resolve()))
        if status < 0:
            raise RuntimeError(f"Could not load ROOT dictionary: {dictionary}")

    converter_path = str(converter_root.resolve())
    selected_path = str(selected_root.resolve())
    _require_tree(ROOT, converter_path, generated_tree, GENERATED_COLUMNS)
    selected_entries = _require_tree(ROOT, selected_path, tree, SELECTED_COLUMNS)

    selected = ROOT.RDataFrame(tree, selected_path).AsNumpy(SELECTED_COLUMNS)
    selected_count = np.asarray(selected["sourceFileId"]).size
    if selected_count != selected_entries:
        raise RuntimeError("selected tree read returned an unexpected number of rows")
    if selection_mask is not None:
        selection_mask = np.asarray(selection_mask, dtype=bool)
        if selection_mask.shape != (selected_count,):
            raise ValueError(
                f"selection mask has {selection_mask.size} rows; expected {selected_count}"
            )
    else:
        selection_mask = np.ones(selected_count, dtype=bool)

    selected_keys = _source_keys(
        selected["sourceFileId"][selection_mask],
        selected["sourceEventIndex"][selection_mask],
    )
    selected_rec_flat = binning.coordinates_to_flat(
        selected["Q2"][selection_mask],
        selected["xB"][selection_mask],
        selected["t"][selection_mask],
        selected["trentoPhi"][selection_mask],
    )
    if np.unique(selected_keys).size != selected_keys.size:
        raise ValueError("selected ROOT sample contains duplicate source keys")
    order = np.argsort(selected_keys, order=selected_keys.dtype.names)
    selected_keys = selected_keys[order]
    selected_rec_flat = selected_rec_flat[order]

    generated_entries = _tree_entries(ROOT, converter_path, generated_tree)
    number_of_bins = binning.size
    truth_total = np.zeros(number_of_bins, dtype=float)
    reconstructed_total = np.zeros(number_of_bins, dtype=float)
    feed_counts = np.zeros(number_of_bins, dtype=float)
    migration_rows: list[np.ndarray] = []
    migration_cols: list[np.ndarray] = []
    migration_weights: list[np.ndarray] = []
    matched_selected_rows = 0

    for chunk_index, start in enumerate(range(0, generated_entries, chunk_size), start=1):
        stop = min(start + chunk_size, generated_entries)
        chunk = ROOT.RDataFrame(generated_tree, converter_path).Range(start, stop).AsNumpy(
            GENERATED_COLUMNS
        )
        valid = np.asarray(chunk["topologyValid"], dtype=bool)
        truth_flat = binning.coordinates_to_flat(
            chunk["Q2"], chunk["xB"], chunk["minusT"], chunk["trentoPhi"]
        )
        weights = np.asarray(chunk["weight"], dtype=float)
        truth_inside = valid & (truth_flat >= 0) & (truth_flat < number_of_bins)
        truth_total += np.bincount(
            truth_flat[truth_inside], weights=weights[truth_inside], minlength=number_of_bins
        )
        if progress_chunks > 0 and chunk_index % progress_chunks == 0:
            print(
                f"[PROGRESS] generated rows {stop}/{generated_entries} "
                f"({100.0 * stop / max(generated_entries, 1):.1f}%)"
            )

        if selected_keys.size == 0:
            continue
        gen_keys = _source_keys(chunk["sourceFileId"], chunk["sourceEventIndex"])
        positions = np.searchsorted(selected_keys, gen_keys)
        bounded = positions < selected_keys.size
        matched = np.zeros(gen_keys.size, dtype=bool)
        matched[bounded] = selected_keys[positions[bounded]] == gen_keys[bounded]
        if not np.any(matched):
            continue

        matched_positions = positions[matched]
        rec_flat = selected_rec_flat[matched_positions]
        matched_valid = valid[matched]
        rec_inside = matched_valid & (rec_flat >= 0) & (rec_flat < number_of_bins)
        matched_weights = weights[matched]
        matched_truth_flat = truth_flat[matched]
        matched_truth_inside = truth_inside[matched]
        matched_selected_rows += int(np.count_nonzero(rec_inside))

        reconstructed_total += np.bincount(
            rec_flat[rec_inside], weights=matched_weights[rec_inside], minlength=number_of_bins
        )

        migrated = rec_inside & matched_truth_inside
        if np.any(migrated):
            migration_rows.append(rec_flat[migrated])
            migration_cols.append(matched_truth_flat[migrated])
            migration_weights.append(matched_weights[migrated])

        feed_in = rec_inside & ~matched_truth_inside
        if np.any(feed_in):
            feed_counts += np.bincount(
                rec_flat[feed_in], weights=matched_weights[feed_in], minlength=number_of_bins
            )

    response = build_response_from_counts(
        truth_total,
        reconstructed_total,
        _concat_or_empty(migration_rows),
        _concat_or_empty(migration_cols),
        _concat_or_empty(migration_weights, dtype=float),
        feed_counts,
    )
    return RootResponseSummary(
        response=response,
        generated_rows=generated_entries,
        selected_rows=selected_count,
        matched_selected_rows=matched_selected_rows,
    )


def _require_tree(ROOT, path: str, tree_name: str, columns: list[str]) -> int:
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {path}")
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find tree {tree_name} in {path}")
    missing = [name for name in columns if not tree.GetBranch(name)]
    entries = int(tree.GetEntries())
    root_file.Close()
    if missing:
        raise RuntimeError(f"Tree {tree_name} in {path} is missing branches: {missing}")
    return entries


def _tree_entries(ROOT, path: str, tree_name: str) -> int:
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {path}")
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find tree {tree_name} in {path}")
    entries = int(tree.GetEntries())
    root_file.Close()
    return entries


def _source_keys(source_file_id, source_event_index) -> np.ndarray:
    keys = np.empty(
        np.asarray(source_file_id).size,
        dtype=[("source_file_id", "<u8"), ("source_event_index", "<u8")],
    )
    keys["source_file_id"] = np.asarray(source_file_id, dtype=np.uint64)
    keys["source_event_index"] = np.asarray(source_event_index, dtype=np.uint64)
    return keys


def _concat_or_empty(items: list[np.ndarray], dtype=np.int64) -> np.ndarray:
    if not items:
        return np.empty(0, dtype=dtype)
    return np.concatenate(items).astype(dtype, copy=False)
