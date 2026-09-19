#!/usr/bin/env python3

"""Overlay campaign structure functions with a published reference table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.reference_comparison import (
    STRUCTURE_FUNCTIONS,
    load_reference_table,
    match_reference_bins,
)
from eppi0.structure_functions import epsilon_from_xb_q2


DEFAULT_REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "reference"
    / "clas6_bedlinskiy_2014_structure_functions.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match a campaign structure-function artifact to a published table "
            "by configured Q2, xB, and -t bin and render an auditable overlay."
        )
    )
    parser.add_argument("structure_functions", type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign-label", default="campaign")
    parser.add_argument("--reference-label", default="CLAS6 Bedlinskiy et al. (2014)")
    parser.add_argument("--reference-beam-energy", type=float, default=5.75)
    return parser.parse_args()


def _require_shape(name: str, values: np.ndarray, expected: tuple[int, ...]) -> None:
    if values.shape != expected:
        raise ValueError(f"{name} has shape {values.shape}; expected {expected}")


def main() -> int:
    args = parse_args()
    artifact = np.load(args.structure_functions, allow_pickle=False)
    table = load_reference_table(args.reference)
    q2_edges = np.asarray(artifact["q2_edges"], dtype=float)
    xb_edges = np.asarray(artifact["xb_edges"], dtype=float)
    t_edges = np.asarray(artifact["t_edges"], dtype=float)
    shape = (q2_edges.size - 1, xb_edges.size - 1, t_edges.size - 1)
    values = np.asarray(artifact["structure_functions"], dtype=float)
    uncertainties = np.asarray(artifact["uncertainties"], dtype=float)
    valid = np.asarray(artifact["valid"], dtype=bool)
    q2_reference = np.asarray(artifact["q2_reference"], dtype=float)
    xb_reference = np.asarray(artifact["xb_reference"], dtype=float)
    minus_t_coordinate = np.asarray(artifact["minus_t_coordinate"], dtype=float)
    epsilon = np.asarray(artifact["epsilon"], dtype=float)
    _require_shape("structure_functions", values, shape + (3,))
    _require_shape("uncertainties", uncertainties, shape + (3,))
    _require_shape("valid", valid, shape)
    _require_shape("q2_reference", q2_reference, shape)
    _require_shape("xb_reference", xb_reference, shape)
    _require_shape("epsilon", epsilon, shape)
    _require_shape("minus_t_coordinate", minus_t_coordinate, (shape[2],))

    names = tuple(str(item) for item in np.asarray(artifact["structure_function_names"]))
    if names != STRUCTURE_FUNCTIONS:
        raise ValueError(
            f"unexpected structure-function ordering {names}; expected {STRUCTURE_FUNCTIONS}"
        )
    match = match_reference_bins(
        table,
        q2_edges=q2_edges,
        xb_edges=xb_edges,
        t_edges=t_edges,
    )
    index = (match.q2_index, match.xb_index, match.t_index)
    campaign_values = values[index]
    campaign_uncertainties = uncertainties[index]
    campaign_valid = valid[index]
    campaign_q2 = q2_reference[index]
    campaign_xb = xb_reference[index]
    campaign_t = minus_t_coordinate[match.t_index]
    campaign_epsilon = epsilon[index]
    reference_epsilon = epsilon_from_xb_q2(
        table.q2, table.xb, args.reference_beam_energy
    )
    reference_total = np.hypot(table.statistical, table.systematic)
    available_uncertainty = np.sqrt(campaign_uncertainties**2 + reference_total**2)
    residual = campaign_values - table.values
    standardized_residual = np.divide(
        residual,
        available_uncertainty,
        out=np.full_like(residual, np.nan),
        where=campaign_valid[:, None]
        & np.isfinite(available_uncertainty)
        & (available_uncertainty > 0.0),
    )
    sigma_u_ratio = np.divide(
        campaign_values[:, 0],
        table.values[:, 0],
        out=np.full(table.q2.shape, np.nan),
        where=campaign_valid
        & np.isfinite(campaign_values[:, 0])
        & (table.values[:, 0] > 0.0),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "reference_comparison.csv"
    csv_header = [
        "iq2",
        "ixb",
        "it",
        "campaign_valid",
        "reference_Q2",
        "reference_xB",
        "reference_minus_t",
        "campaign_Q2_reference",
        "campaign_xB_reference",
        "campaign_minus_t_coordinate",
        "reference_epsilon",
        "campaign_epsilon",
    ]
    for name in STRUCTURE_FUNCTIONS:
        csv_header.extend(
            (
                f"reference_{name}",
                f"reference_{name}_stat",
                f"reference_{name}_sys",
                f"campaign_{name}",
                f"campaign_{name}_uncertainty",
                f"campaign_minus_reference_{name}",
                f"{name}_residual_over_available_uncertainty",
            )
        )
    csv_header.append("campaign_over_reference_sigma_U")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(csv_header)
        for row in range(table.q2.size):
            output = [
                int(match.q2_index[row]),
                int(match.xb_index[row]),
                int(match.t_index[row]),
                int(campaign_valid[row]),
                table.q2[row],
                table.xb[row],
                table.minus_t[row],
                campaign_q2[row],
                campaign_xb[row],
                campaign_t[row],
                reference_epsilon[row],
                campaign_epsilon[row],
            ]
            for component in range(3):
                output.extend(
                    (
                        table.values[row, component],
                        table.statistical[row, component],
                        table.systematic[row, component],
                        campaign_values[row, component],
                        campaign_uncertainties[row, component],
                        residual[row, component],
                        standardized_residual[row, component],
                    )
                )
            output.append(sigma_u_ratio[row])
            writer.writerow(output)

    valid_ratio = sigma_u_ratio[np.isfinite(sigma_u_ratio) & (sigma_u_ratio > 0.0)]
    summary = {
        "campaign_structure_functions": str(args.structure_functions.resolve()),
        "reference_table": str(args.reference.resolve()),
        "campaign_label": args.campaign_label,
        "reference_label": args.reference_label,
        "reference_beam_energy_GeV": args.reference_beam_energy,
        "campaign_beam_energy_GeV": float(np.asarray(artifact["beam_energy"]).item()),
        "reference_rows": int(table.q2.size),
        "unique_reference_analysis_bins": int(table.q2.size),
        "campaign_valid_matched_bins": int(np.count_nonzero(campaign_valid)),
        "comparison_definition": (
            "same configured Q2, xB, and -t bin; no interpolation or bin-center "
            "translation between experiments"
        ),
        "uncertainty_limitation": (
            "reference statistical and systematic uncertainties are combined in "
            "quadrature; campaign uncertainty does not include the pending campaign "
            "systematic covariance"
        ),
        "sigma_U_limitation": (
            "sigma_U=sigma_T+epsilon*sigma_L uses a different epsilon at each beam "
            "energy and is not an identical observable without an L/T model or separation"
        ),
        "sigma_U_campaign_over_reference_quantiles": (
            np.quantile(valid_ratio, [0.1, 0.5, 0.9]).tolist()
            if valid_ratio.size
            else [None, None, None]
        ),
    }
    summary_path = args.output_dir / "reference_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = args.output_dir / "reference_comparison.pdf"
    pages = 0
    with PdfPages(pdf_path) as pdf:
        groups = sorted(set(zip(match.q2_index, match.xb_index)))
        display_names = {
            "sigma_U": r"$\sigma_U$",
            "sigma_LT": r"$\sigma_{LT}$",
            "sigma_TT": r"$\sigma_{TT}$",
        }
        for iq2, ixb in groups:
            selected = (match.q2_index == iq2) & (match.xb_index == ixb)
            rows = np.flatnonzero(selected)
            fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharex=True)
            for component, (name, axis) in enumerate(zip(STRUCTURE_FUNCTIONS, axes)):
                axis.errorbar(
                    table.minus_t[rows],
                    table.values[rows, component],
                    yerr=reference_total[rows, component],
                    fmt="s",
                    color="#0072B2",
                    ecolor="#0072B2",
                    capsize=2,
                    label=f"{args.reference_label}: total error",
                )
                axis.errorbar(
                    table.minus_t[rows],
                    table.values[rows, component],
                    yerr=table.statistical[rows, component],
                    fmt="none",
                    color="#56B4E9",
                    ecolor="#56B4E9",
                    capsize=2,
                    label=f"{args.reference_label}: statistical",
                )
                shown = rows[campaign_valid[rows]]
                axis.errorbar(
                    campaign_t[shown],
                    campaign_values[shown, component],
                    yerr=campaign_uncertainties[shown, component],
                    fmt="o",
                    color="black",
                    ecolor="black",
                    capsize=2,
                    label=f"{args.campaign_label}: available error",
                )
                axis.axhline(0.0, color="0.75", linewidth=0.8)
                axis.set_xlabel(r"$-t$ (GeV$^2$)")
                axis.set_ylabel(f"{display_names[name]} (nb/GeV$^2$)")
                axis.grid(alpha=0.2)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.075),
                ncol=3,
                fontsize=8,
            )
            fig.suptitle(
                f"Configured bin: {q2_edges[iq2]:g} <= $Q^2$ < "
                f"{q2_edges[iq2 + 1]:g} GeV$^2$, "
                f"{xb_edges[ixb]:g} <= $x_B$ < {xb_edges[ixb + 1]:g}"
            )
            fig.text(
                0.5,
                0.025,
                "Same-bin overlay without kinematic interpolation. sigma_U uses "
                "beam-energy-dependent epsilon; campaign systematic covariance is pending.",
                ha="center",
                fontsize=8,
            )
            fig.tight_layout(rect=(0.0, 0.20, 1.0, 0.92))
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1

    print(f"Reference rows: {table.q2.size}")
    print(f"Campaign-valid matched bins: {int(np.count_nonzero(campaign_valid))}")
    if valid_ratio.size:
        print(
            "Campaign/reference sigma_U q10/median/q90:",
            np.quantile(valid_ratio, [0.1, 0.5, 0.9]),
        )
    print(f"Wrote {pdf_path} ({pages} pages)")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(
        "WARNING: sigma_U differs in epsilon between beam energies; treat its raw "
        "ratio as diagnostic until an L/T model or separation is applied."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
