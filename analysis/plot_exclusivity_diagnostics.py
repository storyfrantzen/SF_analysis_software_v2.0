#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.exclusivity import load_cuts
from eppi0.exclusivity_diagnostics import (
    render_comparison_diagnostics,
    render_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render fit, model-component, and fixed-window N-1 diagnostics "
            "from a v8 cut table."
        )
    )
    parser.add_argument("cuts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--gemc-cuts",
        type=Path,
        help=(
            "Render a paired topology comparison; the positional cut table is "
            "treated as data and this cut table as GEMC"
        ),
    )
    parser.add_argument("--data-label", default="Data")
    parser.add_argument("--gemc-label", default="GEMC")
    parser.add_argument(
        "--omit-dropped-topologies",
        action="store_true",
        help=(
            "In paired mode, omit the dedicated failed-topology appendix; by "
            "default it is placed after all retained-topology variable pages"
        ),
    )
    parser.add_argument(
        "--group-id",
        type=int,
        action="append",
        help="Render a particular retained group; repeat for multiple groups",
    )
    parser.add_argument(
        "--maximum-groups",
        type=int,
        default=24,
        help="If group IDs are omitted, render this many worst-fit groups (default: 24)",
    )
    args = parser.parse_args()
    cuts = load_cuts(str(args.cuts))
    if args.omit_dropped_topologies and not args.gemc_cuts:
        parser.error("--omit-dropped-topologies requires --gemc-cuts")
    if args.gemc_cuts:
        if args.group_id:
            parser.error("--group-id is not available with --gemc-cuts")
        gemc_cuts = load_cuts(str(args.gemc_cuts))
        variables = render_comparison_diagnostics(
            cuts,
            gemc_cuts,
            args.output,
            data_label=args.data_label,
            gemc_label=args.gemc_label,
            append_dropped_topologies=not args.omit_dropped_topologies,
        )
        print(
            f"Rendered paired diagnostics for {len(variables)} variables "
            f"to {args.output}"
        )
        print("Variables:", " ".join(variables))
        return 0
    rendered = render_diagnostics(
        cuts,
        args.output,
        group_ids=args.group_id,
        maximum_groups=args.maximum_groups,
    )
    print(f"Rendered {len(rendered)} groups to {args.output}")
    print("Group IDs:", " ".join(str(item) for item in rendered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
