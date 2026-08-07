#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check legacy and source-aware identities in a gEvents tree."
    )
    parser.add_argument("input", type=Path, help="ROOT file produced by hipo2root")
    parser.add_argument("--tree", default="gEvents")
    return parser.parse_args()


def structured_keys(first: np.ndarray, second: np.ndarray, names: tuple[str, str]) -> np.ndarray:
    keys = np.empty(
        np.asarray(first).size,
        dtype=[(names[0], np.asarray(first).dtype), (names[1], np.asarray(second).dtype)],
    )
    keys[names[0]] = first
    keys[names[1]] = second
    return keys


def duplicate_summary(keys: np.ndarray) -> tuple[int, int]:
    _, counts = np.unique(keys, return_counts=True)
    return int(counts.size), int(np.count_nonzero(counts > 1))


def main() -> int:
    args = parse_args()
    import ROOT  # type: ignore
    from eppi0.root_trees import resolve

    ROOT.gROOT.SetBatch(True)
    path = str(args.input.resolve())
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {path}")
    tree_name = resolve(root_file, args.tree)
    tree = root_file.Get(tree_name)
    if not tree:
        raise RuntimeError(f"Could not find tree {args.tree} in {path}")
    required = ("runNum", "eventNum", "sourceFileId", "sourceEventIndex")
    missing = [name for name in required if not tree.GetBranch(name)]
    root_file.Close()
    if missing:
        raise RuntimeError(f"{tree_name} is missing branches: {', '.join(missing)}")

    arrays = ROOT.RDataFrame(tree_name, path).AsNumpy(list(required))
    legacy = structured_keys(
        np.asarray(arrays["runNum"], dtype=np.int64),
        np.asarray(arrays["eventNum"], dtype=np.int64),
        ("run", "event"),
    )
    source = structured_keys(
        np.asarray(arrays["sourceFileId"], dtype=np.uint64),
        np.asarray(arrays["sourceEventIndex"], dtype=np.uint64),
        ("source_file", "source_event"),
    )
    legacy_unique, legacy_duplicates = duplicate_summary(legacy)
    source_unique, source_duplicates = duplicate_summary(source)

    print(f"Rows:                         {legacy.size}")
    print(f"Unique (run,event):           {legacy_unique}")
    print(f"Duplicated (run,event) keys:  {legacy_duplicates}")
    print(f"Unique source-event keys:     {source_unique}")
    print(f"Duplicated source-event keys: {source_duplicates}")
    if source_duplicates:
        print("ERROR: source-aware event identity is not unique")
        return 2
    print("OK: source-aware event identity is unique")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
