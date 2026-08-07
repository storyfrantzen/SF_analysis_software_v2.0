#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
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

PROCESSING_SUMMARY_BRANCHES = {
    "total_events": "TotalEvents",
    "failed_qadb": "FailedQADB",
    "failed_final_state": "FailedFinalState",
    "failed_skim": "FailedSkim",
    "written_events": "WrittenEvents",
    "output_rows": "OutputRows",
}

SAMPLE_SUMMARY_FIELDS = [
    "label",
    "sample_path",
    "selected_rows",
    "processing_root",
    *PROCESSING_SUMMARY_BRANCHES,
    "accumulated_charge_nC",
    "selected_rows_per_input_event",
]

SHAPE_METRIC_FIELDS = [
    "reference",
    "comparison",
    "column",
    "reference_entries_in_range",
    "comparison_entries_in_range",
    "jensen_shannon_divergence",
    "total_variation_distance",
]


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
        description="Overlay ROOT distributions and quantify sample-to-reference shape differences."
    )
    parser.add_argument(
        "--sample",
        action="append",
        type=parse_sample,
        required=True,
        help="Sample as LABEL=path.root. May be repeated.",
    )
    parser.add_argument(
        "--processing-root",
        action="append",
        type=parse_sample,
        default=[],
        help=(
            "Optional converter ROOT metadata as LABEL=path.root. Labels must match --sample; "
            "may be repeated."
        ),
    )
    parser.add_argument(
        "--reference",
        help="Reference sample label for ratios and shape metrics (default: first --sample).",
    )
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS)
    parser.add_argument("--where", help="Optional ROOT RDataFrame filter expression")
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/root_comparison"))
    parser.add_argument("--density", action="store_true", help="Normalize each histogram to unit area")
    parser.add_argument("--no-ratio", action="store_true", help="Do not draw ratio-to-reference panels")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path")
    parser.add_argument("--metrics-csv", type=Path, help="Optional shape-metrics CSV path")
    parser.add_argument("--sample-summary-csv", type=Path, help="Optional sample-summary CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if args.dictionary and ROOT.gSystem.Load(str(args.dictionary.resolve())) < 0:
        raise RuntimeError(f"Could not load ROOT dictionary: {args.dictionary}")

    sample_paths = unique_labeled_paths(args.sample, "--sample")
    processing_paths = unique_labeled_paths(args.processing_root, "--processing-root")
    unknown_processing_labels = sorted(set(processing_paths) - set(sample_paths))
    if unknown_processing_labels:
        raise ValueError(
            "--processing-root labels must match --sample labels; unknown: "
            + ", ".join(unknown_processing_labels)
        )
    reference = args.reference or next(iter(sample_paths))
    if reference not in sample_paths:
        raise ValueError(f"Unknown --reference label: {reference}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    sample_rows = []
    summary = {
        "tree": args.tree,
        "where": args.where,
        "reference": reference,
        "samples": {},
        "columns": {},
        "shape_metrics": [],
    }

    for label, path in sample_paths.items():
        arrays = read_arrays(ROOT, path, args.tree, args.columns, args.where)
        data[label] = arrays
        metadata = (
            read_processing_metadata(ROOT, processing_paths[label])
            if label in processing_paths
            else None
        )
        sample_row = make_sample_summary(label, path, sample_size(arrays), metadata)
        sample_rows.append(sample_row)
        sample_json = dict(sample_row)
        sample_json["path"] = sample_row["sample_path"]
        sample_json["rows"] = sample_row["selected_rows"]
        summary["samples"][label] = sample_json

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
        bins, histograms = histogram_counts(column, data)
        summary["shape_metrics"].extend(
            comparison_metrics(column, histograms, reference)
        )
        plot_column(
            column,
            histograms,
            bins,
            args.output_dir,
            reference=reference,
            density=args.density,
            ratio=not args.no_ratio,
        )

    summary_path = args.summary or args.output_dir / "summary.json"
    metrics_path = args.metrics_csv or args.output_dir / "shape_metrics.csv"
    sample_summary_path = args.sample_summary_csv or args.output_dir / "sample_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(metrics_path, SHAPE_METRIC_FIELDS, summary["shape_metrics"])
    write_csv(sample_summary_path, SAMPLE_SUMMARY_FIELDS, sample_rows)
    print(f"Wrote {args.output_dir}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {sample_summary_path}")
    return 0


def unique_labeled_paths(entries: list[tuple[str, Path]], option: str) -> dict[str, Path]:
    result = {}
    for label, path in entries:
        if label in result:
            raise ValueError(f"Duplicate {option} label: {label}")
        result[label] = path
    return result


def read_processing_metadata(ROOT, path: Path) -> dict[str, object]:
    resolved = path.resolve()
    root_file = ROOT.TFile.Open(str(resolved), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open processing ROOT file: {resolved}")
    try:
        tree = root_file.Get("Summary")
        if not tree or tree.GetEntries() < 1:
            raise RuntimeError(f"Processing ROOT file has no populated Summary tree: {resolved}")
        tree.GetEntry(0)
        metadata: dict[str, object] = {"processing_root": str(resolved)}
        for output_name, branch_name in PROCESSING_SUMMARY_BRANCHES.items():
            metadata[output_name] = (
                int(getattr(tree, branch_name)) if tree.GetBranch(branch_name) else None
            )
        charge = root_file.Get("AccumulatedCharge")
        metadata["accumulated_charge_nC"] = float(charge.GetVal()) if charge else None
        return metadata
    finally:
        root_file.Close()


def make_sample_summary(
    label: str,
    sample_path: Path,
    selected_rows: int,
    processing_metadata: dict[str, object] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "label": label,
        "sample_path": str(sample_path.resolve()),
        "selected_rows": int(selected_rows),
        "processing_root": None,
        **{name: None for name in PROCESSING_SUMMARY_BRANCHES},
        "accumulated_charge_nC": None,
        "selected_rows_per_input_event": None,
    }
    if processing_metadata:
        row.update(processing_metadata)
        total_events = processing_metadata.get("total_events")
        if isinstance(total_events, int) and total_events > 0:
            row["selected_rows_per_input_event"] = selected_rows / total_events
    return row


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def histogram_counts(
    column: str,
    data: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values_by_label = {
        label: clean(np.asarray(arrays[column], dtype=float)) for label, arrays in data.items()
    }
    bins = binning_for(column, values_by_label)
    histograms = {}
    for label, values in values_by_label.items():
        counts, _ = np.histogram(values, bins=bins)
        histograms[label] = counts.astype(float)
    return bins, histograms


def normalized_probabilities(counts: np.ndarray) -> np.ndarray | None:
    total = float(np.sum(counts))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return np.asarray(counts, dtype=float) / total


def shape_metrics(
    reference_counts: np.ndarray,
    comparison_counts: np.ndarray,
) -> tuple[float | None, float | None]:
    reference = normalized_probabilities(reference_counts)
    comparison = normalized_probabilities(comparison_counts)
    if reference is None or comparison is None:
        return None, None
    midpoint = 0.5 * (reference + comparison)
    reference_mask = reference > 0.0
    comparison_mask = comparison > 0.0
    kl_reference = np.sum(
        reference[reference_mask] * np.log(reference[reference_mask] / midpoint[reference_mask])
    )
    kl_comparison = np.sum(
        comparison[comparison_mask]
        * np.log(comparison[comparison_mask] / midpoint[comparison_mask])
    )
    jensen_shannon = 0.5 * (kl_reference + kl_comparison)
    total_variation = 0.5 * np.sum(np.abs(reference - comparison))
    return float(jensen_shannon), float(total_variation)


def comparison_metrics(
    column: str,
    histograms: dict[str, np.ndarray],
    reference: str,
) -> list[dict[str, object]]:
    reference_counts = histograms[reference]
    rows = []
    for label, counts in histograms.items():
        if label == reference:
            continue
        jensen_shannon, total_variation = shape_metrics(reference_counts, counts)
        rows.append(
            {
                "reference": reference,
                "comparison": label,
                "column": column,
                "reference_entries_in_range": int(np.sum(reference_counts)),
                "comparison_entries_in_range": int(np.sum(counts)),
                "jensen_shannon_divergence": jensen_shannon,
                "total_variation_distance": total_variation,
            }
        )
    return rows


def display_histograms(
    histograms: dict[str, np.ndarray],
    bins: np.ndarray,
    density: bool,
) -> dict[str, np.ndarray]:
    if not density:
        return histograms
    widths = np.diff(bins)
    result = {}
    for label, counts in histograms.items():
        total = float(np.sum(counts))
        result[label] = counts / (total * widths) if total > 0.0 else counts.copy()
    return result


def plot_column(
    column: str,
    histograms: dict[str, np.ndarray],
    bins: np.ndarray,
    output_dir: Path,
    *,
    reference: str,
    density: bool,
    ratio: bool,
) -> None:
    centers = 0.5 * (bins[:-1] + bins[1:])
    plotted = display_histograms(histograms, bins, density)

    labels = list(plotted)
    if ratio and len(labels) > 1:
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(7.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        rax = None

    for label, counts in plotted.items():
        ax.step(centers, counts, where="mid", label=label, linewidth=1.6)
    ax.set_ylabel("Density" if density else "Entries")
    ax.set_title(column)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    if rax is not None:
        reference_values = plotted[reference]
        for label in labels:
            if label == reference:
                continue
            counts = plotted[label]
            ratio_values = np.divide(
                counts,
                reference_values,
                out=np.full_like(counts, np.nan, dtype=float),
                where=reference_values > 0,
            )
            rax.step(
                centers,
                ratio_values,
                where="mid",
                label=f"{label}/{reference}",
                linewidth=1.2,
            )
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
