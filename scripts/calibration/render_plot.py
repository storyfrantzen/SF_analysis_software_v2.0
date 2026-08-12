#!/usr/bin/env python3
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} FIGURE_PAYLOAD", file=sys.stderr)
        return 2

    # Select the non-interactive backend before importing or unpickling any
    # Matplotlib object.  Deliberately do not import ROOT in this process.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    payload_path = Path(sys.argv[1])
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)

    fig = payload["figure"]
    context = payload["context"]
    if payload["tightLayout"]:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93 if context else 0.95))
    else:
        fig.subplots_adjust(
            left=0.07,
            right=0.96,
            bottom=0.14,
            top=0.78 if context else 0.84,
            wspace=0.42,
        )

    output_path = Path(payload["outputPath"])
    fig.savefig(
        output_path,
        dpi=payload["dpi"],
        metadata=payload["metadata"],
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
