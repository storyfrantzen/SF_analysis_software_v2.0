from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .binning import AnalysisBinning
from .phase_space import AnalysisPhaseSpace
from .response import ResponseResult, build_response_from_counts
from .topology import detector_topology_id, ft_photon_count


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

SELECTED_TOPOLOGY_COLUMNS = [
    "pDet",
    "g1Det",
    "g2Det",
]


@dataclass(frozen=True)
class RootResponseSummary:
    response: ResponseResult
    generated_rows: int
    selected_rows: int
    matched_selected_rows: int
    reconstructed_topology_ids: np.ndarray
    reconstructed_topology_counts: np.ndarray


def build_response_from_root(
    converter_root: Path,
    selected_root: Path,
    binning: AnalysisBinning,
    dictionary: Path | None = None,
    tree: str = "sEvents",
    generated_tree: str = "gEvents",
    chunk_size: int = 1_000_000,
    selection_mask: np.ndarray | None = None,
    progress_chunks: int = 10,
    phase_space: AnalysisPhaseSpace | None = None,
    beam_energy: float | None = None,
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
    generated_tree = _resolve_selected_tree(ROOT, converter_path, generated_tree)
    _require_tree(ROOT, converter_path, generated_tree, GENERATED_COLUMNS)
    tree = _resolve_selected_tree(ROOT, selected_path, tree)
    selected_entries = _require_tree(ROOT, selected_path, tree, SELECTED_COLUMNS)
    topology_columns_available = _tree_has_columns(
        ROOT, selected_path, tree, SELECTED_TOPOLOGY_COLUMNS
    )
    selected_columns = SELECTED_COLUMNS + (
        SELECTED_TOPOLOGY_COLUMNS if topology_columns_available else []
    )

    selected = ROOT.RDataFrame(tree, selected_path).AsNumpy(selected_columns)
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
    if topology_columns_available:
        selected_topology = detector_topology_id(
            selected["pDet"][selection_mask],
            ft_photon_count(
                selected["g1Det"][selection_mask],
                selected["g2Det"][selection_mask],
            ),
        )
        topology_ids = np.unique(selected_topology)
    else:
        print(
            "Warning: selected ROOT tree lacks pDet/g1Det/g2Det; "
            "topology-resolved response metadata will be omitted"
        )
        selected_topology = np.empty(0, dtype=np.int64)
        topology_ids = np.empty(0, dtype=np.int64)
    if np.unique(selected_keys).size != selected_keys.size:
        raise ValueError("selected ROOT sample contains duplicate source keys")
    order = np.argsort(selected_keys, order=selected_keys.dtype.names)
    selected_keys = selected_keys[order]
    selected_rec_flat = selected_rec_flat[order]
    if topology_columns_available:
        selected_topology = selected_topology[order]

    generated_entries = _tree_entries(ROOT, converter_path, generated_tree)
    number_of_bins = binning.size
    truth_total = np.zeros(number_of_bins, dtype=float)
    reconstructed_total = np.zeros(number_of_bins, dtype=float)
    feed_counts = np.zeros(number_of_bins, dtype=float)
    topology_counts = np.zeros((topology_ids.size, number_of_bins), dtype=float)
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
        truth_inside = _truth_inside_mask(
            valid,
            truth_flat,
            chunk["Q2"],
            chunk["xB"],
            number_of_bins,
            phase_space=phase_space,
            beam_energy=beam_energy,
        )
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
        if topology_columns_available:
            matched_topology = selected_topology[matched_positions]
        matched_selected_rows += int(np.count_nonzero(rec_inside))

        reconstructed_total += np.bincount(
            rec_flat[rec_inside], weights=matched_weights[rec_inside], minlength=number_of_bins
        )
        if topology_columns_available:
            for topology_index, topology_id in enumerate(topology_ids):
                topology_rows = rec_inside & (matched_topology == topology_id)
                if np.any(topology_rows):
                    topology_counts[topology_index] += np.bincount(
                        rec_flat[topology_rows],
                        weights=matched_weights[topology_rows],
                        minlength=number_of_bins,
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
    if topology_columns_available and not np.allclose(
        topology_counts.sum(axis=0), reconstructed_total, rtol=1.0e-10, atol=1.0e-10
    ):
        raise RuntimeError(
            "topology-resolved reconstructed counts do not sum to reconstructed_total"
        )
    return RootResponseSummary(
        response=response,
        generated_rows=generated_entries,
        selected_rows=selected_count,
        matched_selected_rows=matched_selected_rows,
        reconstructed_topology_ids=topology_ids,
        reconstructed_topology_counts=topology_counts,
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


def _tree_has_columns(ROOT, path: str, tree_name: str, columns: list[str]) -> bool:
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {path}")
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find tree {tree_name} in {path}")
    available = all(tree.GetBranch(name) for name in columns)
    root_file.Close()
    return available


def _resolve_selected_tree(ROOT, path: str, tree_name: str) -> str:
    from .root_trees import resolve

    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {path}")
    resolved = resolve(root_file, tree_name)
    if root_file.Get(resolved):
        root_file.Close()
        if resolved != tree_name:
            print(f"Warning: using compatible tree {resolved}")
        return resolved
    root_file.Close()
    return tree_name


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


def _truth_inside_mask(
    topology_valid: np.ndarray,
    truth_flat: np.ndarray,
    q2: np.ndarray,
    xb: np.ndarray,
    number_of_bins: int,
    *,
    phase_space: AnalysisPhaseSpace | None = None,
    beam_energy: float | None = None,
) -> np.ndarray:
    inside = (
        np.asarray(topology_valid, dtype=bool)
        & (np.asarray(truth_flat) >= 0)
        & (np.asarray(truth_flat) < number_of_bins)
    )
    if phase_space is not None and phase_space.enabled:
        if beam_energy is None:
            raise ValueError("beam_energy is required when phase_space is enabled")
        inside &= phase_space.mask(q2, xb, beam_energy)
    return inside
