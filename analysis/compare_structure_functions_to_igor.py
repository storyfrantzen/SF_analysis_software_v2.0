#!/usr/bin/env python3
"""Compare RGA structure functions with the Igor Pass2 v1 extraction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.reference_comparison import STRUCTURE_FUNCTIONS, load_reference_table


DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "reference"
    / "igor_pass2_v1_structure_functions.csv"
)
IGOR_EDGES = {
    "q2": np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.6, 6.0, 7.0, 8.5]),
    "xb": np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.38, 0.48, 0.58, 0.65]),
    "t": np.array([0.09, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0, 1.5, 2.0]),
}
COLORS = {"combined": "black", "torus_plus_1": "#0072B2", "torus_minus_1": "#D55E00"}
LABELS = {
    "combined": "combined",
    "torus_plus_1": "torus+1",
    "torus_minus_1": "torus-1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("combined", type=Path)
    parser.add_argument("--torus-plus-1", required=True, type=Path)
    parser.add_argument("--torus-minus-1", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--campaign-label",
        default="RGA Fall 2018 certified combined unit-weight result",
    )
    parser.add_argument("--reference-label", default="Igor Pass2 v1")
    return parser.parse_args()


def quantiles(values: np.ndarray) -> list[float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return [None, None, None]
    return [float(value) for value in np.quantile(finite, [0.1, 0.5, 0.9])]


def bin_indices(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.searchsorted(edges, values, side="right") - 1
    inside = np.isfinite(values) & (values >= edges[0]) & (values < edges[-1])
    return np.where(inside, result, -1).astype(np.int64)


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        names = tuple(str(item) for item in artifact["structure_function_names"])
        if names != STRUCTURE_FUNCTIONS:
            raise ValueError(f"unexpected structure-function ordering in {path}: {names}")
        keys = (
            "structure_functions",
            "uncertainties",
            "valid",
            "epsilon",
            "q2_reference",
            "xb_reference",
            "minus_t_coordinate",
            "q2_edges",
            "xb_edges",
            "t_edges",
        )
        return {key: np.asarray(artifact[key]) for key in keys}


def map_artifact(table, artifact: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    iq = bin_indices(artifact["q2_edges"], table.q2)
    ix = bin_indices(artifact["xb_edges"], table.xb)
    it = bin_indices(artifact["t_edges"], table.minus_t)
    inside = (iq >= 0) & (ix >= 0) & (it >= 0)
    count = table.q2.size
    values = np.full((count, 3), np.nan)
    uncertainties = np.full((count, 3), np.nan)
    valid = np.zeros(count, dtype=bool)
    q2_reference = np.full(count, np.nan)
    xb_reference = np.full(count, np.nan)
    minus_t_coordinate = np.full(count, np.nan)
    rows = np.flatnonzero(inside)
    index = (iq[rows], ix[rows], it[rows])
    values[rows] = artifact["structure_functions"][index]
    uncertainties[rows] = artifact["uncertainties"][index]
    valid[rows] = artifact["valid"][index]
    q2_reference[rows] = artifact["q2_reference"][index]
    xb_reference[rows] = artifact["xb_reference"][index]
    minus_t_coordinate[rows] = artifact["minus_t_coordinate"][it[rows]]
    return {
        "iq": iq,
        "ix": ix,
        "it": it,
        "inside": inside,
        "values": values,
        "uncertainties": uncertainties,
        "valid": valid,
        "q2_reference": q2_reference,
        "xb_reference": xb_reference,
        "minus_t_coordinate": minus_t_coordinate,
    }


def strict_edge_mask(table, artifact, mapping) -> np.ndarray:
    iq = bin_indices(IGOR_EDGES["q2"], table.q2)
    ix = bin_indices(IGOR_EDGES["xb"], table.xb)
    it = bin_indices(IGOR_EDGES["t"], table.minus_t)
    usable = mapping["inside"] & (iq >= 0) & (ix >= 0) & (it >= 0)
    result = np.zeros(table.q2.size, dtype=bool)
    for row in np.flatnonzero(usable):
        comparisons = (
            (iq[row], mapping["iq"][row], IGOR_EDGES["q2"], artifact["q2_edges"]),
            (ix[row], mapping["ix"][row], IGOR_EDGES["xb"], artifact["xb_edges"]),
            (it[row], mapping["it"][row], IGOR_EDGES["t"], artifact["t_edges"]),
        )
        result[row] = all(
            np.isclose(reference_edges[reference_index], campaign_edges[campaign_index])
            and np.isclose(
                reference_edges[reference_index + 1],
                campaign_edges[campaign_index + 1],
            )
            for reference_index, campaign_index, reference_edges, campaign_edges in comparisons
        )
    return result


def component_metrics(table, mapping, mask: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    for component, name in enumerate(STRUCTURE_FUNCTIONS):
        use = (
            mask
            & mapping["valid"]
            & np.isfinite(table.values[:, component])
            & np.isfinite(mapping["values"][:, component])
        )
        ratios = np.divide(
            mapping["values"][use, component],
            table.values[use, component],
            out=np.full(np.count_nonzero(use), np.nan),
            where=table.values[use, component] != 0.0,
        )
        available = np.hypot(
            mapping["uncertainties"][use, component],
            table.statistical[use, component],
        )
        pulls = np.divide(
            mapping["values"][use, component] - table.values[use, component],
            available,
            out=np.full(available.shape, np.nan),
            where=available > 0.0,
        )
        result[name] = {
            "matched_bins": int(np.count_nonzero(use)),
            "campaign_over_Igor_q10_median_q90": quantiles(ratios),
            "same_sign": int(
                np.count_nonzero(
                    np.sign(mapping["values"][use, component])
                    == np.sign(table.values[use, component])
                )
            ),
            "statistical_only_pull_q10_median_q90": quantiles(pulls),
            "statistical_only_absolute_pull_gt_2": int(
                np.count_nonzero(np.abs(pulls) > 2.0)
            ),
        }
    return result


def write_matches(path: Path, table, mappings, strict: np.ndarray) -> None:
    header = [
        "row",
        "strict_identical_3d_bin",
        "reference_Q2",
        "reference_xB",
        "reference_minus_t",
    ]
    for label in mappings:
        header.extend(
            (
                f"{label}_valid",
                f"{label}_Q2_reference",
                f"{label}_xB_reference",
                f"{label}_minus_t_coordinate",
            )
        )
        for name in STRUCTURE_FUNCTIONS:
            header.extend((f"{label}_{name}", f"{label}_{name}_uncertainty"))
    for name in STRUCTURE_FUNCTIONS:
        header.extend((f"reference_{name}", f"reference_{name}_stat"))
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in range(table.q2.size):
            output = [row + 1, int(strict[row]), table.q2[row], table.xb[row], table.minus_t[row]]
            for mapping in mappings.values():
                output.extend(
                    (
                        int(mapping["valid"][row]),
                        mapping["q2_reference"][row],
                        mapping["xb_reference"][row],
                        mapping["minus_t_coordinate"][row],
                    )
                )
                for component in range(3):
                    output.extend(
                        (
                            mapping["values"][row, component],
                            mapping["uncertainties"][row, component],
                        )
                    )
            for component in range(3):
                output.extend((table.values[row, component], table.statistical[row, component]))
            writer.writerow(output)


def render_plots(args, table, artifacts, mappings, strict, scale: float) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    combined = mappings["combined"]
    primary = strict & combined["valid"]
    positive = primary & (table.values[:, 0] > 0.0) & (combined["values"][:, 0] > 0.0)
    ratio = combined["values"][positive, 0] / table.values[positive, 0]
    summary_pdf = args.output_dir / "comparison_summary.pdf"
    with PdfPages(summary_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
        axes[0].scatter(table.values[positive, 0], combined["values"][positive, 0], s=28, alpha=0.8)
        high = max(np.max(table.values[positive, 0]), np.max(combined["values"][positive, 0])) * 1.04
        axes[0].plot([0.0, high], [0.0, high], "--", color="0.4", label="equal")
        axes[0].plot([0.0, high], [0.0, high * scale], color="#CC79A7", label=f"median scale = {scale:.3f}")
        axes[0].set(xlabel=rf"{args.reference_label} $\sigma_U$", ylabel=rf"{args.campaign_label} $\sigma_U$", xlim=(0.0, high), ylim=(0.0, high))
        axes[0].legend(fontsize=8)
        axes[1].hist(ratio, bins=24, color="#0072B2", alpha=0.8)
        axes[1].axvline(1.0, color="0.4", linestyle="--")
        axes[1].axvline(np.median(ratio), color="black", label=f"median={np.median(ratio):.3f}")
        axes[1].set(xlabel=rf"campaign / {args.reference_label} $\sigma_U$", ylabel="bins")
        axes[1].legend()
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.suptitle("Strict identical-bin comparison")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
        for axis, coordinates, label in zip(
            axes,
            (table.q2, table.xb, table.minus_t),
            (r"$Q^2$ (GeV$^2$)", r"$x_B$", r"$-t$ (GeV$^2$)"),
        ):
            axis.scatter(coordinates[positive], ratio, s=25, alpha=0.8)
            axis.axhline(1.0, color="0.4", linestyle="--")
            axis.axhline(scale, color="#CC79A7")
            axis.set(xlabel=label, ylabel=rf"campaign / {args.reference_label} $\sigma_U$")
            axis.grid(alpha=0.2)
        fig.suptitle("Normalization difference across kinematics")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
        for axis, component, title in (
            (axes[0], 1, r"$\sigma_{LT}/\sigma_U$"),
            (axes[1], 2, r"$\sigma_{TT}/\sigma_U$"),
        ):
            reference_ratio = table.values[primary, component] / table.values[primary, 0]
            campaign_ratio = combined["values"][primary, component] / combined["values"][primary, 0]
            finite = np.isfinite(reference_ratio) & np.isfinite(campaign_ratio)
            axis.scatter(reference_ratio[finite], campaign_ratio[finite], s=28, alpha=0.8)
            low = min(np.min(reference_ratio[finite]), np.min(campaign_ratio[finite]))
            high = max(np.max(reference_ratio[finite]), np.max(campaign_ratio[finite]))
            axis.plot([low, high], [low, high], "--", color="0.4")
            axis.set(xlabel=f"{args.reference_label} {title}", ylabel=f"campaign {title}")
            axis.grid(alpha=0.2)
        fig.suptitle("Scale-independent harmonic ratios")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(11.0, 6.5))
        for key, mapping in mappings.items():
            use = strict & mapping["valid"] & (table.values[:, 0] > 0.0) & (mapping["values"][:, 0] > 0.0)
            axis.scatter(
                table.minus_t[use],
                mapping["values"][use, 0] / table.values[use, 0],
                s=22,
                alpha=0.65,
                label=LABELS[key],
                color=COLORS[key],
            )
        axis.axhline(1.0, color="0.4", linestyle="--")
        axis.set(xlabel=r"$-t$ (GeV$^2$)", ylabel=rf"campaign / {args.reference_label} $\sigma_U$")
        axis.grid(alpha=0.2)
        axis.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    structure_pdf = args.output_dir / "structure_functions_vs_t.pdf"
    with PdfPages(structure_pdf) as pdf:
        groups = sorted(set(zip(combined["iq"][strict], combined["ix"][strict])))
        for iq, ix in groups:
            rows = np.flatnonzero(strict & (combined["iq"] == iq) & (combined["ix"] == ix))
            if not np.any(combined["valid"][rows]):
                continue
            fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharex=True)
            for component, (name, axis) in enumerate(zip(STRUCTURE_FUNCTIONS, axes)):
                axis.errorbar(
                    table.minus_t[rows],
                    table.values[rows, component],
                    yerr=table.statistical[rows, component],
                    fmt="s",
                    color="#D55E00",
                    capsize=2,
                    label=f"{args.reference_label} (stat.)",
                )
                shown = rows[combined["valid"][rows]]
                axis.errorbar(
                    combined["minus_t_coordinate"][shown],
                    combined["values"][shown, component],
                    yerr=combined["uncertainties"][shown, component],
                    fmt="o",
                    color="black",
                    capsize=2,
                    label=args.campaign_label,
                )
                axis.axhline(0.0, color="0.75", linewidth=0.8)
                axis.grid(alpha=0.2)
                axis.set_xlabel(r"$-t$ (GeV$^2$)")
                axis.set_ylabel(name.replace("_", " ") + r" (nb/GeV$^2$)")
            q2_edges = artifacts["combined"]["q2_edges"]
            xb_edges = artifacts["combined"]["xb_edges"]
            fig.suptitle(
                rf"$Q^2\in[{q2_edges[iq]:g},{q2_edges[iq + 1]:g}]$ GeV$^2$, "
                rf"$x_B\in[{xb_edges[ix]:g},{xb_edges[ix + 1]:g}]$"
            )
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
            fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.93))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {summary_pdf}")
    print(f"Wrote {structure_pdf}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = load_reference_table(args.reference)
    artifacts = {
        "combined": load_artifact(args.combined),
        "torus_plus_1": load_artifact(args.torus_plus_1),
        "torus_minus_1": load_artifact(args.torus_minus_1),
    }
    combined_edges = tuple(artifacts["combined"][name] for name in ("q2_edges", "xb_edges", "t_edges"))
    for label, artifact in artifacts.items():
        edges = tuple(artifact[name] for name in ("q2_edges", "xb_edges", "t_edges"))
        if not all(np.array_equal(left, right) for left, right in zip(combined_edges, edges)):
            raise ValueError(f"{label} uses incompatible analysis binning")
    mappings = {label: map_artifact(table, artifact) for label, artifact in artifacts.items()}
    strict = strict_edge_mask(table, artifacts["combined"], mappings["combined"])
    write_matches(args.output_dir / "matched_bins.csv", table, mappings, strict)

    combined = mappings["combined"]
    positive = strict & combined["valid"] & (table.values[:, 0] > 0.0) & (combined["values"][:, 0] > 0.0)
    if not np.any(positive):
        raise ValueError("no positive strict identical-bin sigma_U matches")
    sigma_u_ratio = combined["values"][positive, 0] / table.values[positive, 0]
    scale = float(np.median(sigma_u_ratio))
    normalized_ratio = sigma_u_ratio / scale
    summary = {
        "reference_table": str(args.reference.resolve()),
        "reference_label": args.reference_label,
        "campaign_label": args.campaign_label,
        "artifacts": {
            "combined": str(args.combined.resolve()),
            "torus_plus_1": str(args.torus_plus_1.resolve()),
            "torus_minus_1": str(args.torus_minus_1.resolve()),
        },
        "reference_rows": int(table.q2.size),
        "strict_identical_bin_rows": int(np.count_nonzero(strict)),
        "combined_valid_strict_bins": int(np.count_nonzero(strict & combined["valid"])),
        "comparison_definition": "identical Q2, xB, and -t bin edges; campaign production quality mask",
        "uncertainty_limitation": "Igor table uncertainties are statistical only; campaign systematic covariance is pending",
        "components": {
            label: component_metrics(table, mapping, strict)
            for label, mapping in mappings.items()
        },
        "strict_sigma_U_shape_normalization": {
            "campaign_over_Igor_q10_median_q90": quantiles(sigma_u_ratio),
            "robust_campaign_over_Igor_scale": scale,
            "scale_normalized_ratio_q10_median_q90": quantiles(normalized_ratio),
            "scale_normalized_bins_within_20_percent": int(np.count_nonzero(np.abs(normalized_ratio - 1.0) < 0.2)),
            "scale_normalized_fraction_within_20_percent": float(np.mean(np.abs(normalized_ratio - 1.0) < 0.2)),
        },
    }
    summary_path = args.output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    render_plots(args, table, artifacts, mappings, strict, scale)
    readme = (
        f"# {args.campaign_label} versus {args.reference_label}\n\n"
        f"- Reference rows: {table.q2.size}\n"
        f"- Strict identical-bin rows: {np.count_nonzero(strict)}\n"
        f"- Combined production-quality strict matches: {np.count_nonzero(strict & combined['valid'])}\n"
        f"- Campaign / Igor sigma_U q10, median, q90: {quantiles(sigma_u_ratio)}\n"
        f"- Robust campaign / Igor normalization scale: {scale:.6f}\n"
        "- Igor uncertainties are statistical only; campaign systematic covariance is pending.\n"
        "- The Igor values already contain the reported global factor of 1.3.\n"
    )
    (args.output_dir / "README.md").write_text(readme)
    print(f"Strict identical-bin rows: {np.count_nonzero(strict)}")
    print(f"Combined production-quality strict matches: {np.count_nonzero(strict & combined['valid'])}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
