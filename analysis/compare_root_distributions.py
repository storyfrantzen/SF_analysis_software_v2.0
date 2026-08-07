#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COLUMNS = [
    "Q2",
    "xB",
    "t",
    "trentoPhi",
    "W",
    "y",
    "m_gg",
    "E_miss",
    "pT_miss",
    "electronP",
    "pi0_p",
]

DEFAULT_BINS = {
    "Q2": (80, 0.0, 5.5),
    "xB": (80, 0.0, 0.7),
    "t": (80, 0.0, 2.5),
    "trentoPhi": (72, -math.pi, math.pi),
    "W": (80, 1.5, 4.0),
    "y": (80, 0.0, 1.0),
    "m_gg": (80, 0.0, 0.3),
    "E_miss": (80, -2.0, 2.0),
    "pT_miss": (80, 0.0, 2.0),
    "electronP": (80, 0.0, 6.5),
    "pi0_p": (80, 0.0, 5.0),
    "p": (80, 0.0, 6.5),
    "theta": (80, 0.0, math.pi),
    "phi": (72, -math.pi, math.pi),
}

REC_ALIASES = {
    "runNum": "rec.runNum",
    "eventNum": "rec.eventNum",
    "particleIdx": "rec.particleIdx",
    "matchedGenIdx": "rec.matchedGenIdx",
    "matchAngleDeg": "rec.matchAngleDeg",
    "pid": "rec.pid",
    "charge": "rec.charge",
    "status": "rec.status",
    "det": "rec.det",
    "sector": "rec.sector",
    "p": "rec.p",
    "px": "rec.px",
    "py": "rec.py",
    "pz": "rec.pz",
    "theta": "rec.theta",
    "phi": "rec.phi",
    "beta": "rec.beta",
    "chi2pid": "rec.chi2pid",
    "trackChi2": "rec.trackChi2",
    "trackNDF": "rec.trackNDF",
    "trackChi2N": "rec.trackChi2N",
    "vz": "rec.vz",
    "E_PCAL": "rec.E_PCAL",
    "E_ECIN": "rec.E_ECIN",
    "E_ECOUT": "rec.E_ECOUT",
}


def parse_sample(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("sample label must not be empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay ROOT tree distributions for reconstructed aaoRad comparisons."
    )
    parser.add_argument(
        "--sample",
        action="append",
        type=parse_sample,
        required=True,
        help="Sample as LABEL=path.root. May be repeated.",
    )
    parser.add_argument("--tree", default="rParticles")
    parser.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS)
    parser.add_argument("--where", help="Optional ROOT RDataFrame filter expression")
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/aao_rad_compare"))
    parser.add_argument("--density", action="store_true", help="Normalize each histogram to unit area")
    parser.add_argument("--no-ratio", action="store_true", help="Do not draw ratio-to-first-sample panels")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if args.dictionary and ROOT.gSystem.Load(str(args.dictionary.resolve())) < 0:
        raise RuntimeError(f"Could not load ROOT dictionary: {args.dictionary}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    summary = {"tree": args.tree, "where": args.where, "samples": {}, "columns": {}}

    for label, path in args.sample:
        arrays = read_arrays(ROOT, path, args.tree, args.columns, args.where)
        data[label] = arrays
        summary["samples"][label] = {
            "path": str(path.resolve()),
            "rows": int(sample_size(arrays)),
        }

    for column in args.columns:
        column_summary = {}
        for label, arrays in data.items():
            values = clean(np.asarray(arrays[column], dtype=float))
            column_summary[label] = {
                "entries": int(values.size),
                "mean": float(np.mean(values)) if values.size else None,
                "rms": float(np.std(values)) if values.size else None,
                "min": float(np.min(values)) if values.size else None,
                "max": float(np.max(values)) if values.size else None,
            }
        summary["columns"][column] = column_summary
        plot_column(column, data, args.output_dir, density=args.density, ratio=not args.no_ratio)

    summary_path = args.summary or args.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output_dir}")
    print(f"Wrote {summary_path}")
    return 0


def read_arrays(ROOT, path: Path, tree: str, columns: list[str], where: str | None):
    root_path = str(path.resolve())
    tree = resolve_particle_tree(ROOT, root_path, tree)
    aliases = aliases_for_columns(ROOT, root_path, tree, columns)
    frame = ROOT.RDataFrame(tree, root_path)
    if where:
        frame = frame.Filter(where)
    for name, expression in aliases.items():
        frame = frame.Define(name, expression)
    return frame.AsNumpy(columns)


def resolve_particle_tree(ROOT, path: str, tree_name: str) -> str:
    from eppi0.root_trees import resolve

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


def aliases_for_columns(ROOT, path: str, tree_name: str, columns: list[str]) -> dict[str, str]:
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {path}")
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find tree {tree_name} in {path}")
    has_rec = bool(tree.GetBranch("rec"))
    aliases = {
        name: REC_ALIASES[name]
        for name in columns
        if not tree.GetBranch(name) and has_rec and name in REC_ALIASES
    }
    missing = [name for name in columns if not tree.GetBranch(name) and name not in aliases]
    root_file.Close()
    if missing:
        raise RuntimeError(f"Tree {tree_name} in {path} is missing branches: {missing}")
    return aliases


def sample_size(arrays) -> int:
    if not arrays:
        return 0
    first = next(iter(arrays.values()))
    return int(np.asarray(first).size)


def clean(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def binning_for(column: str, values_by_label: dict[str, np.ndarray]) -> np.ndarray:
    if column in DEFAULT_BINS:
        bins, low, high = DEFAULT_BINS[column]
        return np.linspace(low, high, bins + 1)
    all_values = np.concatenate([values for values in values_by_label.values() if values.size])
    if all_values.size == 0:
        return np.linspace(0.0, 1.0, 51)
    low, high = np.percentile(all_values, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.min(all_values)), float(np.max(all_values))
    if low == high:
        high = low + 1.0
    return np.linspace(float(low), float(high), 81)


def plot_column(
    column: str,
    data: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    *,
    density: bool,
    ratio: bool,
) -> None:
    values_by_label = {
        label: clean(np.asarray(arrays[column], dtype=float)) for label, arrays in data.items()
    }
    bins = binning_for(column, values_by_label)
    centers = 0.5 * (bins[:-1] + bins[1:])
    histograms = {}
    for label, values in values_by_label.items():
        counts, _ = np.histogram(values, bins=bins)
        counts = counts.astype(float)
        if density and counts.sum() > 0:
            widths = np.diff(bins)
            counts = counts / (counts.sum() * widths)
        histograms[label] = counts

    labels = list(histograms)
    if ratio and len(labels) > 1:
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(7.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        rax = None

    for label, counts in histograms.items():
        ax.step(centers, counts, where="mid", label=label, linewidth=1.6)
    ax.set_ylabel("Density" if density else "Entries")
    ax.set_title(column)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    if rax is not None:
        reference = histograms[labels[0]]
        for label in labels[1:]:
            counts = histograms[label]
            ratio_values = np.divide(
                counts,
                reference,
                out=np.full_like(counts, np.nan, dtype=float),
                where=reference > 0,
            )
            rax.step(centers, ratio_values, where="mid", label=f"{label}/{labels[0]}", linewidth=1.2)
        rax.axhline(1.0, color="black", linewidth=0.8)
        rax.set_ylabel("Ratio")
        rax.set_xlabel(column)
        rax.grid(alpha=0.25)
        rax.set_ylim(0.0, 2.0)
    else:
        ax.set_xlabel(column)

    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_name(column)}.png", dpi=160)
    plt.close(fig)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
