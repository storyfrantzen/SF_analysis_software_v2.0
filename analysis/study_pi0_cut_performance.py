#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from eppi0.exclusivity_models import FitEstimate, estimate_model
from eppi0.root_trees import resolve


PI0_MASS_GEV = 0.1349768
TOPOLOGIES = ("FD/FD", "FD/FT", "FT/FT")
SAMPLES = ("data", "GEMC")
COLORS = {"FD/FD": "#0072B2", "FD/FT": "#E69F00", "FT/FT": "#009E73"}

CSV_FIELDS = (
    "sample",
    "topology",
    "working_point",
    "fd_min_p_gev",
    "ft_min_p_gev",
    "baseline_events",
    "retained_events",
    "retained_fraction",
    "fit_status",
    "fit_reason",
    "fit_model",
    "fit_entries",
    "fit_lower_gev",
    "fit_upper_gev",
    "pi0_mean_gev",
    "pi0_width_gev",
    "peak_lower_gev",
    "peak_upper_gev",
    "signal_fraction_fit",
    "observed_peak",
    "fitted_signal_peak",
    "sideband_subtracted_signal_peak",
    "fitted_background_peak",
    "background_fraction_peak",
    "signal_over_sqrt_signal_plus_background",
    "bic",
    "pearson_chi2_per_ndof",
)


@dataclass(frozen=True)
class WorkingPoint:
    label: str
    fd_min_p: float
    ft_min_p: float


def working_point(value: str) -> WorkingPoint:
    pieces = value.split(":")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError(
            "working points must have the form LABEL:FD_MIN_P:FT_MIN_P"
        )
    label = pieces[0].strip()
    if not label:
        raise argparse.ArgumentTypeError("working-point label must not be empty")
    try:
        fd_min_p, ft_min_p = (float(item) for item in pieces[1:])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "working-point thresholds must be numbers in GeV"
        ) from error
    if not all(math.isfinite(item) and item >= 0.0 for item in (fd_min_p, ft_min_p)):
        raise argparse.ArgumentTypeError(
            "working-point thresholds must be finite and nonnegative"
        )
    return WorkingPoint(label, fd_min_p, ft_min_p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pi0 peak quality and retention in selected data and GEMC for "
            "detector-dependent photon-momentum working points."
        )
    )
    parser.add_argument("--data", type=Path, required=True, help="Selected data ROOT file")
    parser.add_argument("--gemc", type=Path, required=True, help="Selected GEMC ROOT file")
    parser.add_argument("--tree", default="sEvents")
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument(
        "--where",
        default="passExclusivity == 1",
        help=(
            "Shared ROOT RDataFrame baseline filter (default: passExclusivity == 1). "
            "Quote the expression in tcsh."
        ),
    )
    parser.add_argument(
        "--require-tag",
        action="append",
        default=[],
        metavar="CUT_NAME",
        help=(
            "Require an evaluatedCuts tag to pass; may be repeated. This reproduces "
            "a visualizer passCut_* filter without cutting on m_gg itself."
        ),
    )
    parser.add_argument(
        "--working-point",
        action="append",
        type=working_point,
        required=True,
        metavar="LABEL:FD_MIN_P:FT_MIN_P",
        help=(
            "Photon momentum working point in GeV; may be repeated. For example, "
            "loose:0.4:0.4 or fd06:0.6:0.4."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-events", type=int, default=100)
    parser.add_argument("--fit-bins", type=int, default=120)
    parser.add_argument("--fit-window-sigma", type=float, default=5.0)
    parser.add_argument("--fit-max-iterations", type=int, default=100)
    parser.add_argument("--fit-convergence", type=float, default=1.0e-5)
    parser.add_argument(
        "--peak-nsigma",
        type=float,
        default=3.0,
        help="Gaussian-equivalent fitted peak window used for S and B (default: 3)",
    )
    parser.add_argument("--maximum-center-deviation", type=float, default=0.04)
    parser.add_argument("--maximum-width", type=float, default=0.04)
    parser.add_argument("--no-continuous-refinement", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)

    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if args.dictionary and ROOT.gSystem.Load(str(args.dictionary.resolve())) < 0:
        raise RuntimeError(f"Could not load ROOT dictionary: {args.dictionary}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = {
        "data": read_sample(ROOT, args.data, args.tree, args.where, args.require_tag),
        "GEMC": read_sample(ROOT, args.gemc, args.tree, args.where, args.require_tag),
    }

    rows: list[dict[str, Any]] = []
    fits: dict[tuple[str, str, str], FitEstimate] = {}
    for sample_label, arrays in samples.items():
        for topology in TOPOLOGIES:
            topology_selection = topology_mask(arrays, topology)
            baseline_events = int(np.count_nonzero(topology_selection))
            for point in args.working_point:
                selected = topology_selection & threshold_mask(arrays, point)
                masses = np.asarray(arrays["m_gg"][selected], dtype=float)
                masses = masses[np.isfinite(masses)]
                estimate, reason = fit_pi0(masses, args)
                row = metric_row(
                    sample_label,
                    topology,
                    point,
                    baseline_events,
                    int(masses.size),
                    estimate,
                    reason,
                )
                rows.append(row)
                if estimate is not None:
                    fits[(sample_label, topology, point.label)] = estimate

    write_csv(args.output_dir / "pi0_cut_metrics.csv", rows)
    write_json(args, rows, samples, args.output_dir / "pi0_cut_metrics.json")
    plot_metric_summary(rows, args.working_point, args.output_dir / "pi0_cut_metrics.png")
    plot_fit_diagnostics(
        rows,
        fits,
        args.working_point,
        args.output_dir / "pi0_cut_fits.pdf",
    )
    print_summary(rows)
    print(f"Wrote {args.output_dir / 'pi0_cut_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'pi0_cut_metrics.json'}")
    print(f"Wrote {args.output_dir / 'pi0_cut_metrics.png'}")
    print(f"Wrote {args.output_dir / 'pi0_cut_fits.pdf'}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    labels = [point.label for point in args.working_point]
    if len(labels) != len(set(labels)):
        raise ValueError("working-point labels must be unique")
    if args.minimum_events < 20:
        raise ValueError("--minimum-events must be at least 20")
    if args.fit_bins < 20:
        raise ValueError("--fit-bins must be at least 20")
    if args.peak_nsigma <= 0.0:
        raise ValueError("--peak-nsigma must be positive")


def read_sample(ROOT, path: Path, requested_tree: str, where: str, tags: list[str]):
    root_path = str(path.resolve())
    root_file = ROOT.TFile.Open(root_path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")
    tree_name = resolve(root_file, requested_tree)
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find tree {requested_tree} in {root_path}")

    columns = ["m_gg", "g1Det", "g2Det", "gamma1P", "gamma2P"]
    if tags:
        columns.extend(("evaluatedCuts", "failedCuts"))
    missing = [name for name in columns if not tree.GetBranch(name)]
    root_file.Close()
    if missing:
        raise RuntimeError(f"Tree {tree_name} in {root_path} is missing branches: {missing}")

    frame = ROOT.RDataFrame(tree_name, root_path)
    if where.strip():
        frame = frame.Filter(where)
    arrays = frame.AsNumpy(columns)
    if tags:
        keep = required_tag_mask(arrays["evaluatedCuts"], arrays["failedCuts"], tags)
        arrays = {name: np.asarray(values)[keep] for name, values in arrays.items()}
    print(f"{path}: baseline rows={len(arrays['m_gg'])}")
    return arrays


def csv_names(value: Any) -> set[str]:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return {item.strip() for item in str(value).split(",") if item.strip()}


def required_tag_mask(evaluated: np.ndarray, failed: np.ndarray, tags: list[str]) -> np.ndarray:
    required = set(tags)
    output = np.zeros(len(evaluated), dtype=bool)
    for index, (evaluated_value, failed_value) in enumerate(
        zip(evaluated, failed, strict=True)
    ):
        evaluated_names = csv_names(evaluated_value)
        failed_names = csv_names(failed_value)
        evaluated_names.update(failed_names)  # Compatibility with failedCuts-only files.
        output[index] = required.issubset(evaluated_names) and required.isdisjoint(
            failed_names
        )
    return output


def topology_mask(arrays: dict[str, np.ndarray], topology: str) -> np.ndarray:
    gamma1 = np.asarray(arrays["g1Det"], dtype=np.int64)
    gamma2 = np.asarray(arrays["g2Det"], dtype=np.int64)
    if topology == "FD/FD":
        return (gamma1 == 1) & (gamma2 == 1)
    if topology == "FT/FT":
        return (gamma1 == 0) & (gamma2 == 0)
    if topology == "FD/FT":
        return ((gamma1 == 1) & (gamma2 == 0)) | ((gamma1 == 0) & (gamma2 == 1))
    raise ValueError(f"Unknown topology: {topology}")


def threshold_mask(arrays: dict[str, np.ndarray], point: WorkingPoint) -> np.ndarray:
    gamma1_detector = np.asarray(arrays["g1Det"], dtype=np.int64)
    gamma2_detector = np.asarray(arrays["g2Det"], dtype=np.int64)
    gamma1_p = np.asarray(arrays["gamma1P"], dtype=float)
    gamma2_p = np.asarray(arrays["gamma2P"], dtype=float)
    supported = np.isin(gamma1_detector, (0, 1)) & np.isin(gamma2_detector, (0, 1))
    gamma1_threshold = np.where(gamma1_detector == 1, point.fd_min_p, point.ft_min_p)
    gamma2_threshold = np.where(gamma2_detector == 1, point.fd_min_p, point.ft_min_p)
    return supported & (gamma1_p >= gamma1_threshold) & (gamma2_p >= gamma2_threshold)


def fit_pi0(values: np.ndarray, args: argparse.Namespace):
    return estimate_model(
        values,
        "rec_m_gg",
        n_sigma=args.peak_nsigma,
        minimum_events=args.minimum_events,
        fit_window_sigma=args.fit_window_sigma,
        max_iterations=args.fit_max_iterations,
        convergence=args.fit_convergence,
        histogram_bins=args.fit_bins,
        minimum_signal_fraction=1.0e-6,
        minimum_peak_significance=1.0e-6,
        expected_center=PI0_MASS_GEV,
        physical_lower=0.0,
        maximum_center_deviation=args.maximum_center_deviation,
        maximum_sigma=args.maximum_width,
        cut_component="core",
        continuous_refinement=not args.no_continuous_refinement,
    )


def metric_row(
    sample: str,
    topology: str,
    point: WorkingPoint,
    baseline_events: int,
    retained_events: int,
    estimate: FitEstimate | None,
    reason: str,
) -> dict[str, Any]:
    retained_fraction = retained_events / baseline_events if baseline_events else math.nan
    row: dict[str, Any] = {
        "sample": sample,
        "topology": topology,
        "working_point": point.label,
        "fd_min_p_gev": point.fd_min_p,
        "ft_min_p_gev": point.ft_min_p,
        "baseline_events": baseline_events,
        "retained_events": retained_events,
        "retained_fraction": retained_fraction,
        "fit_status": "failed",
        "fit_reason": reason,
    }
    for name in CSV_FIELDS:
        row.setdefault(name, math.nan)
    if estimate is None:
        return row

    centers = 0.5 * (estimate.histogram_edges[:-1] + estimate.histogram_edges[1:])
    in_peak = (centers >= estimate.lower) & (centers <= estimate.upper)
    observed_peak = float(np.sum(estimate.observed_counts[in_peak]))
    signal_peak = float(np.sum(estimate.cut_signal_counts[in_peak]))
    background_peak = float(
        np.sum(estimate.background_counts[in_peak])
        + np.sum(estimate.noncut_component_counts[in_peak])
    )
    fitted_peak = signal_peak + background_peak
    row.update(
        fit_status="ok",
        fit_reason="",
        fit_model=estimate.fit_model,
        fit_entries=estimate.fit_entries,
        fit_lower_gev=estimate.fit_lower,
        fit_upper_gev=estimate.fit_upper,
        pi0_mean_gev=estimate.center,
        pi0_width_gev=estimate.sigma,
        peak_lower_gev=estimate.lower,
        peak_upper_gev=estimate.upper,
        signal_fraction_fit=estimate.signal_fraction,
        observed_peak=observed_peak,
        fitted_signal_peak=signal_peak,
        sideband_subtracted_signal_peak=observed_peak - background_peak,
        fitted_background_peak=background_peak,
        background_fraction_peak=(
            background_peak / fitted_peak if fitted_peak > 0.0 else math.nan
        ),
        signal_over_sqrt_signal_plus_background=estimate.peak_significance,
        bic=estimate.bic,
        pearson_chi2_per_ndof=estimate.pearson_chi2 / max(estimate.fit_ndof, 1),
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({name: row[name] for name in CSV_FIELDS} for row in rows)


def json_value(value: Any):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_json(args, rows, samples, path: Path) -> None:
    document = {
        "data": str(args.data.resolve()),
        "gemc": str(args.gemc.resolve()),
        "tree": args.tree,
        "where": args.where,
        "required_tags": args.require_tag,
        "baseline_rows": {name: int(len(values["m_gg"])) for name, values in samples.items()},
        "peak_nsigma": args.peak_nsigma,
        "pi0_mass_gev": PI0_MASS_GEV,
        "metrics": [
            {name: json_value(row[name]) for name in CSV_FIELDS}
            for row in rows
        ],
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def plot_metric_summary(rows, points, output: Path) -> None:
    metrics = (
        ("retained_fraction", "retained fraction", 1.0),
        ("pi0_mean_gev", r"fitted $\mu$ (MeV)", 1000.0),
        ("pi0_width_gev", r"fitted $\sigma$ (MeV)", 1000.0),
        ("signal_fraction_fit", "signal fraction in fit domain", 1.0),
        ("background_fraction_peak", "background fraction under peak", 1.0),
        (
            "signal_over_sqrt_signal_plus_background",
            r"$S/\sqrt{S+B}$",
            1.0,
        ),
    )
    labels = [point.label for point in points]
    positions = np.arange(len(labels))
    lookup = {
        (row["sample"], row["topology"], row["working_point"]): row for row in rows
    }
    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white"}):
        figure, axes = plt.subplots(2, 3, figsize=(14, 8))
        for axis, (metric, ylabel, scale) in zip(axes.flat, metrics, strict=True):
            for topology in TOPOLOGIES:
                for sample in SAMPLES:
                    values = np.asarray(
                        [lookup[(sample, topology, label)][metric] for label in labels],
                        dtype=float,
                    )
                    axis.plot(
                        positions,
                        scale * values,
                        marker=("o" if sample == "data" else "s"),
                        linestyle=("-" if sample == "data" else "--"),
                        color=COLORS[topology],
                        label=f"{sample} {topology}",
                    )
            axis.set_xticks(positions, labels, rotation=20, ha="right")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=6,
        )
        figure.suptitle(r"$\pi^0\to\gamma\gamma$ cut-performance summary", y=0.995)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        figure.savefig(output, dpi=180, facecolor="white")
        plt.close(figure)


def plot_fit_diagnostics(rows, fits, points, output: Path) -> None:
    lookup = {
        (row["sample"], row["topology"], row["working_point"]): row for row in rows
    }
    with PdfPages(output) as pdf:
        for point in points:
            figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
            figure.suptitle(
                f"{point.label}: FD p >= {point.fd_min_p:g} GeV, "
                f"FT p >= {point.ft_min_p:g} GeV",
                y=0.995,
            )
            for row_index, sample in enumerate(SAMPLES):
                for column_index, topology in enumerate(TOPOLOGIES):
                    axis = axes[row_index, column_index]
                    key = (sample, topology, point.label)
                    row = lookup[key]
                    estimate = fits.get(key)
                    axis.set_title(
                        f"{sample} {topology}; N={row['retained_events']:,}; "
                        f"retained={row['retained_fraction']:.3f}"
                    )
                    if estimate is None:
                        axis.text(
                            0.5,
                            0.5,
                            f"fit failed\n{row['fit_reason']}",
                            ha="center",
                            va="center",
                            transform=axis.transAxes,
                            wrap=True,
                        )
                        continue
                    centers = 0.5 * (
                        estimate.histogram_edges[:-1] + estimate.histogram_edges[1:]
                    )
                    axis.errorbar(
                        centers,
                        estimate.observed_counts,
                        yerr=np.sqrt(np.maximum(estimate.observed_counts, 1.0)),
                        fmt=".",
                        color="#202020",
                        markersize=3,
                        linewidth=0.7,
                        label="observed",
                    )
                    axis.plot(centers, estimate.expected_counts, color="black", label="total")
                    axis.plot(
                        centers,
                        estimate.cut_signal_counts,
                        color="#0072B2",
                        label="Gaussian signal",
                    )
                    axis.plot(
                        centers,
                        estimate.background_counts + estimate.noncut_component_counts,
                        color="#D55E00",
                        label="background",
                    )
                    axis.axvline(estimate.lower, color="#009E73", linestyle="--")
                    axis.axvline(estimate.upper, color="#009E73", linestyle="--")
                    axis.text(
                        0.03,
                        0.96,
                        (
                            f"mu={1000.0 * estimate.center:.2f} MeV\n"
                            f"sigma={1000.0 * estimate.sigma:.2f} MeV\n"
                            f"S/sqrt(S+B)={estimate.peak_significance:.1f}\n"
                            f"B/(S+B)={row['background_fraction_peak']:.3f}"
                        ),
                        transform=axis.transAxes,
                        va="top",
                        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
                    )
                    axis.set_xlabel(r"$m_{\gamma\gamma}$ (GeV)")
                    axis.set_ylabel("entries / bin")
                    axis.grid(alpha=0.25)
            handles: list[Any] = []
            labels: list[str] = []
            for axis in axes.flat:
                axis_handles, axis_labels = axis.get_legend_handles_labels()
                if axis_handles:
                    handles, labels = axis_handles, axis_labels
                    break
            if handles:
                figure.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.955),
                    ncol=4,
                )
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
            pdf.savefig(figure, facecolor="white")
            plt.close(figure)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        "sample topology point retained/fraction mean[MeV] sigma[MeV] "
        "B/(S+B) S/sqrt(S+B)"
    )
    for row in rows:
        if row["fit_status"] != "ok":
            print(
                f"{row['sample']:5s} {row['topology']:5s} {row['working_point']:12s} "
                f"{row['retained_events']:8d}/{row['retained_fraction']:.4f} FIT FAILED: "
                f"{row['fit_reason']}"
            )
            continue
        print(
            f"{row['sample']:5s} {row['topology']:5s} {row['working_point']:12s} "
            f"{row['retained_events']:8d}/{row['retained_fraction']:.4f} "
            f"{1000.0 * row['pi0_mean_gev']:9.3f} "
            f"{1000.0 * row['pi0_width_gev']:10.3f} "
            f"{row['background_fraction_peak']:8.4f} "
            f"{row['signal_over_sqrt_signal_plus_background']:11.2f}"
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
