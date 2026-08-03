#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COLUMNS = [
    "runNum", "eventNum", "Q2", "xB", "t", "trentoPhi", "pDet",
    "m_gg", "m2_miss", "m2_epX", "m_eggX", "E_miss", "pT_miss",
]

OPTIONAL_COLUMNS = [
    "t_pi0",
    "eIdx",
    "pIdx",
    "g1Idx",
    "g2Idx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected EPPI0 data to a compact NumPy artifact")
    parser.add_argument("selected_root", type=Path)
    parser.add_argument("processing_root", type=Path, help="Converter ROOT file containing AccumulatedCharge")
    parser.add_argument("output", type=Path)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--tree", default="Events")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if args.dictionary and ROOT.gSystem.Load(str(args.dictionary.resolve())) < 0:
        raise RuntimeError(f"Could not load ROOT dictionary: {args.dictionary}")
    selected_path = str(args.selected_root.resolve())
    columns = columns_in_tree(ROOT, selected_path, args.tree, COLUMNS, OPTIONAL_COLUMNS)
    arrays = ROOT.RDataFrame(args.tree, selected_path).AsNumpy(columns)
    processing_file = ROOT.TFile.Open(str(args.processing_root.resolve()), "READ")
    charge_object = processing_file.Get("AccumulatedCharge")
    if not charge_object:
        raise RuntimeError("processing ROOT file has no AccumulatedCharge metadata")
    beam_charge_c = float(charge_object.GetVal()) * 1.0e-9
    processing_file.Close()
    has_t_pi0 = "t_pi0" in arrays
    metadata = {
        "selected_root": str(args.selected_root.resolve()),
        "processing_root": str(args.processing_root.resolve()),
        "selected_events": int(len(arrays["Q2"])),
        "schema_version": 2 if has_t_pi0 else 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = dict(
        run=arrays["runNum"],
        event=arrays["eventNum"],
        rec_Q2=arrays["Q2"],
        rec_xB=arrays["xB"],
        rec_minus_t=arrays["t"],
        rec_trento_phi=arrays["trentoPhi"],
        rec_proton_detector=arrays["pDet"],
        rec_m_gg=arrays["m_gg"],
        rec_m2_miss=arrays["m2_miss"],
        rec_m2_epX=arrays["m2_epX"],
        rec_m_eggX=arrays["m_eggX"],
        rec_E_miss=arrays["E_miss"],
        rec_pT_miss=arrays["pT_miss"],
        rec_selected=np.ones(len(arrays["Q2"]), dtype=bool),
        beam_charge_c=beam_charge_c,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    if has_t_pi0:
        output["rec_minus_t_pi0"] = arrays["t_pi0"]
    optional_output_names = {
        "eIdx": "rec_electron_index",
        "pIdx": "rec_proton_index",
        "g1Idx": "rec_gamma1_index",
        "g2Idx": "rec_gamma2_index",
    }
    for column, output_name in optional_output_names.items():
        if column in arrays:
            output[output_name] = arrays[column]
    np.savez_compressed(args.output, **output)
    print(f"Selected events: {len(arrays['Q2'])}")
    print(f"Beam charge: {beam_charge_c:.6e} C")
    print(f"Wrote {args.output}")
    return 0


def columns_in_tree(ROOT, path: str, tree_name: str, required: list[str], optional: list[str]) -> list[str]:
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open selected ROOT file: {path}")
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find tree {tree_name} in {path}")
    missing = [name for name in required if not tree.GetBranch(name)]
    if missing:
        root_file.Close()
        raise RuntimeError(f"Tree {tree_name} is missing branches: {missing}")
    columns = list(required) + [name for name in optional if tree.GetBranch(name)]
    root_file.Close()
    return columns


if __name__ == "__main__":
    raise SystemExit(main())
