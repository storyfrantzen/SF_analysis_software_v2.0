#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.exclusivity import load_cuts
from eppi0.exclusivity_diagnostics import render_diagnostics


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
