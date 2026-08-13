#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.binning import from_config
from eppi0.exclusivity import DEFAULT_VARIABLES, apply_cuts, derive_cuts, load_cuts, save_cuts
from eppi0.topology import ft_photon_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive one exclusivity definition and apply it to an event sample."
    )
    parser.add_argument("sample", type=Path, help="Reference sample used to derive cuts")
    parser.add_argument("--apply-to", type=Path, help="Optional second sample; defaults to reference")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cuts", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("npz", "selected-root"),
        default="npz",
        help="Read either a dense event NPZ or the selected-candidate ROOT tree directly",
    )
    parser.add_argument("--dictionary", type=Path, help="ROOT dictionary shared library for selected-root input")
    parser.add_argument("--tree", default="sEvents", help="ROOT tree name for selected-root input")
    parser.add_argument("--global-cuts", action="store_true")
    parser.add_argument("--n-sigma", type=float, default=3.0)
    parser.add_argument("--minimum-events", type=int, default=50)
    parser.add_argument("--reuse-cuts", action="store_true")
    return parser.parse_args()


def arrays(sample) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...]]:
    values = {name: sample[name] for name in DEFAULT_VARIABLES}
    if "rec_ft_photon_count" in sample:
        ft_photons = sample["rec_ft_photon_count"]
    else:
        gamma1_name = _first_available(
            sample, "rec_gamma1_detector", "rec_g1Det", "rec_gamma1Det"
        )
        gamma2_name = _first_available(
            sample, "rec_gamma2_detector", "rec_g2Det", "rec_gamma2Det"
        )
        ft_photons = ft_photon_count(sample[gamma1_name], sample[gamma2_name])
    return values, (
        sample["rec_proton_detector"],
        ft_photons,
        sample["rec_Q2"],
        sample["rec_xB"],
        sample["rec_minus_t"],
    )


def _first_available(sample, *names: str) -> str:
    for name in names:
        if name in sample:
            return name
    raise KeyError(
        "event sample lacks selected-photon detector topology; rebuild or re-export it"
    )


def selected_root_arrays(path: Path, dictionary: Path | None, tree_name: str):
    import ROOT  # type: ignore
    from eppi0.root_trees import resolve

    ROOT.gROOT.SetBatch(True)
    if dictionary:
        status = ROOT.gSystem.Load(str(dictionary.resolve()))
        if status < 0:
            raise RuntimeError(f"Could not load ROOT dictionary: {dictionary}")

    root_path = str(path.resolve())
    root_file = ROOT.TFile.Open(root_path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open selected ROOT file: {root_path}")
    resolved_tree_name = resolve(root_file, tree_name)
    tree = root_file.Get(resolved_tree_name)
    if tree and resolved_tree_name != tree_name:
        print(f"Warning: using compatible tree {resolved_tree_name}", file=sys.stderr)
        tree_name = resolved_tree_name
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find tree {tree_name} in {root_path}")
    columns = [
        "pDet", "g1Det", "g2Det", "Q2", "xB", "t", "m_gg", "pT_miss",
        "m2_epX", "m_eggX", "E_miss", "m2_miss",
    ]
    missing = [name for name in columns if not tree.GetBranch(name)]
    root_file.Close()
    if missing:
        raise RuntimeError(f"Tree {tree_name} in {root_path} is missing branches: {missing}")

    raw = ROOT.RDataFrame(tree_name, root_path).AsNumpy(columns)
    values = {
        "rec_m_gg": raw["m_gg"],
        "rec_pT_miss": raw["pT_miss"],
        "rec_m2_epX": raw["m2_epX"],
        "rec_m_eggX": raw["m_eggX"],
        "rec_E_miss": raw["E_miss"],
        "rec_m2_miss": raw["m2_miss"],
    }
    return values, (
        raw["pDet"],
        ft_photon_count(raw["g1Det"], raw["g2Det"]),
        raw["Q2"],
        raw["xB"],
        raw["t"],
    )


def load_arrays(path: Path, input_format: str, dictionary: Path | None, tree_name: str):
    if input_format == "selected-root":
        return selected_root_arrays(path, dictionary, tree_name)
    sample = np.load(path, allow_pickle=False)
    return arrays(sample)


def main() -> int:
    args = parse_args()
    binning = from_config(args.config)
    reference_values, (detector, ft_photons, q2, xb, minus_t) = load_arrays(
        args.sample, args.format, args.dictionary, args.tree
    )
    iq2, ixb, it, _ = binning.indices(q2, xb, minus_t, np.zeros_like(q2))
    if args.reuse_cuts:
        cuts = load_cuts(str(args.cuts))
    else:
        cuts = derive_cuts(
            reference_values,
            detector,
            ft_photons,
            iq2,
            ixb,
            it,
            n_sigma=args.n_sigma,
            minimum_events=args.minimum_events,
            global_mode=args.global_cuts,
        )
        args.cuts.parent.mkdir(parents=True, exist_ok=True)
        save_cuts(str(args.cuts), cuts)

    target_values, (detector, ft_photons, q2, xb, minus_t) = load_arrays(
        args.apply_to or args.sample, args.format, args.dictionary, args.tree
    )
    iq2, ixb, it, _ = binning.indices(q2, xb, minus_t, np.zeros_like(q2))
    mask = apply_cuts(cuts, target_values, detector, ft_photons, iq2, ixb, it)
    args.mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.mask, mask)
    print(f"Cut groups: {cuts.group_ids.size}")
    print(f"Passing events: {mask.sum()}/{mask.size}")
    if args.format == "selected-root":
        print("Mask rows correspond to selected ROOT candidates; pass this mask to response-root.")
    print(f"Wrote {args.cuts}")
    print(f"Wrote {args.mask}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
