#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy one run from a ROOT TTree into a separate ROOT file."
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("run_number", type=int)
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace output_file if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_file.expanduser().resolve()
    output_path = args.output_file.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input ROOT file does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output ROOT files must be different")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output ROOT file already exists: {output_path}\n"
            "Pass --overwrite only if replacing it is intentional."
        )

    # Candidate trees are flat scalar trees, so this helper intentionally does
    # not load the project's custom event/particle dictionaries. Avoiding that
    # dependency also makes the command usable with any compatible ROOT build.
    import ROOT  # type: ignore
    input_file = ROOT.TFile.Open(str(input_path), "READ")
    if not input_file or input_file.IsZombie():
        raise RuntimeError(f"Could not open input ROOT file: {input_path}")

    tree = input_file.Get(args.tree)
    if not tree or not tree.InheritsFrom("TTree"):
        input_file.Close()
        raise RuntimeError(f"Input file has no TTree named {args.tree!r}")
    if not tree.GetBranch("runNum"):
        input_file.Close()
        raise RuntimeError(f"TTree {args.tree!r} has no runNum branch")

    selection = f"runNum == {args.run_number}"
    selected_entries = int(tree.GetEntries(selection))
    if selected_entries == 0:
        input_file.Close()
        raise RuntimeError(
            f"No entries in TTree {args.tree!r} have runNum {args.run_number}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = ROOT.TFile.Open(str(output_path), "RECREATE")
    if not output_file or output_file.IsZombie():
        input_file.Close()
        raise RuntimeError(f"Could not create output ROOT file: {output_path}")

    output_file.cd()
    selected_tree = tree.CopyTree(selection)
    if not selected_tree:
        output_file.Close()
        input_file.Close()
        raise RuntimeError("ROOT failed to copy the selected entries")
    selected_tree.Write()
    output_file.Close()
    input_file.Close()

    print(
        f"Wrote {selected_entries} entries for run {args.run_number} "
        f"from {args.tree} to {output_path}"
    )


if __name__ == "__main__":
    main()
