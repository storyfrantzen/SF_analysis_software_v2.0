#!/usr/bin/env python3
"""Plot campaign phi-bin cross sections against a reference SF reconstruction."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROTON_MASS_GEV = 0.9382720813
CAMPAIGN_COLOR = "#0072B2"
REFERENCE_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-section", required=True, type=Path)
    parser.add_argument("--harmonics", required=True, type=Path)
    parser.add_argument("--structure-functions", required=True, type=Path)
    parser.add_argument("--matched-bins", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-beam-energy", type=float, default=10.6)
    return parser.parse_args()


def scalar_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def reference_epsilon(q2: float, xb: float, beam_energy: float) -> float:
    y = q2 / (2.0 * PROTON_MASS_GEV * beam_energy * xb)
    q2_over_4e2 = q2 / (4.0 * beam_energy**2)
    return (1.0 - y - q2_over_4e2) / (
        1.0 - y + 0.5 * y**2 + q2_over_4e2
    )


def reconstruct_reference(
    row: dict[str, str], phi_rad: np.ndarray, beam_energy: float
) -> tuple[np.ndarray, np.ndarray, float]:
    q2 = scalar_float(row, "reference_Q2")
    xb = scalar_float(row, "reference_xB")
    epsilon = reference_epsilon(q2, xb, beam_energy)
    sigma_u = scalar_float(row, "reference_sigma_U")
    sigma_lt = scalar_float(row, "reference_sigma_LT")
    sigma_tt = scalar_float(row, "reference_sigma_TT")
    uncertainty_u = scalar_float(row, "reference_sigma_U_stat")
    uncertainty_lt = scalar_float(row, "reference_sigma_LT_stat")
    uncertainty_tt = scalar_float(row, "reference_sigma_TT_stat")

    lt_factor = np.sqrt(2.0 * epsilon * (1.0 + epsilon)) * np.cos(phi_rad)
    tt_factor = epsilon * np.cos(2.0 * phi_rad)
    central = (sigma_u + lt_factor * sigma_lt + tt_factor * sigma_tt) / (
        2.0 * np.pi
    )
    # The reference table does not publish the structure-function covariance.
    variance = (
        uncertainty_u**2
        + (lt_factor * uncertainty_lt) ** 2
        + (tt_factor * uncertainty_tt) ** 2
    ) / (2.0 * np.pi) ** 2
    return central, np.sqrt(np.maximum(variance, 0.0)), epsilon


def campaign_fit(
    parameters: np.ndarray, covariance: np.ndarray, phi_rad: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack(
        [np.ones(phi_rad.size), np.cos(phi_rad), np.cos(2.0 * phi_rad)]
    )
    central = design @ parameters
    variance = np.einsum("ij,jk,ik->i", design, covariance, design)
    return central, np.sqrt(np.maximum(variance, 0.0))


def locate_campaign_bin(
    row: dict[str, str], structure_functions: np.lib.npyio.NpzFile
) -> tuple[int, int, int]:
    q2 = scalar_float(row, "combined_Q2_reference")
    xb = scalar_float(row, "combined_xB_reference")
    minus_t = scalar_float(row, "combined_minus_t_coordinate")
    matches = (
        np.isclose(structure_functions["q2_reference"], q2, rtol=0.0, atol=2e-9)
        & np.isclose(
            structure_functions["xb_reference"], xb, rtol=0.0, atol=2e-9
        )
        & np.isclose(
            structure_functions["minus_t_coordinate"][None, None, :],
            minus_t,
            rtol=0.0,
            atol=2e-9,
        )
        & np.asarray(structure_functions["valid"], dtype=bool)
    )
    indices = np.argwhere(matches)
    if indices.shape != (1, 3):
        raise ValueError(
            f"reference row {row['row']} mapped to {len(indices)} campaign bins"
        )
    return tuple(int(value) for value in indices[0])


def panel_limits(*arrays: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate(
        [np.asarray(array, dtype=float)[np.isfinite(array)] for array in arrays]
    )
    low = min(0.0, float(np.min(finite)))
    high = float(np.max(finite))
    span = max(high - low, 1e-9)
    return low - 0.08 * span, high + 0.16 * span


def cover_page(pdf: PdfPages, rows: list[dict[str, str]], group_count: int) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(
        0.07,
        0.91,
        r"RGA Fall 2018 certified combined unit-weight result vs. Igor Pass2 v1",
        fontsize=20,
        weight="bold",
    )
    fig.text(
        0.07,
        0.855,
        r"Reduced cross section $d^2\sigma/(dt\,d\phi)$ versus $\phi$",
        fontsize=16,
        color="#333333",
    )
    body = (
        f"This document contains {len(rows)} strict matched 3D bins in {group_count} "
        "$Q^2$–$x_B$ cells.  Blue markers are the campaign's stored 20-bin combined "
        "torus-polarity cross sections.  The blue line is the corresponding production-quality "
        "campaign harmonic fit.\n\n"
        "The Igor report publishes structure functions rather than numerical phi-bin cross "
        "sections.  The orange line therefore reconstructs its phi dependence from the printed "
        r"$\sigma_U$, $\sigma_{LT}$, and $\sigma_{TT}$ values using Eq. (24) of the report. "
        "It is not a transcription of unpublished Igor phi-bin points.  The orange band "
        "propagates the printed statistical errors as independent because their covariance is "
        "not available.\n\n"
        "Only rows with identical configured $Q^2$, $x_B$, and $-t$ bin edges and a valid "
        "combined campaign harmonic fit are included.  The Igor values already contain its "
        "reported global factor of 1.3.  Campaign error bars are the uncertainties stored in the "
        "combined cross-section artifact; campaign systematic covariance remains incomplete.\n\n"
        "Panel vertical ranges use central values and fitted curves.  Error bars and uncertainty "
        "bands may be clipped at the panel boundary so a single large uncertainty does not set "
        "the visual scale."
    )
    fig.text(0.07, 0.76, body, fontsize=12, va="top", linespacing=1.45, wrap=True)
    formula = (
        r"$\frac{d^2\sigma}{dt\,d\phi}=\frac{1}{2\pi}"
        r"\left[\sigma_U+\sqrt{2\epsilon(1+\epsilon)}\,\sigma_{LT}\cos\phi"
        r"+\epsilon\,\sigma_{TT}\cos(2\phi)\right]$"
    )
    fig.text(0.5, 0.27, formula, fontsize=15, ha="center")
    fig.text(
        0.07,
        0.11,
        "Reference: I. Korover et al., IGOR_Pi0_Analysis_Pass2_version1.pdf (October 2025).\n"
        "Campaign: RGA Fall 2018 certified torus+1/torus−1 BLUE combination, unit current weights.",
        fontsize=10.5,
        color="#444444",
        va="bottom",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cross_section = np.load(args.cross_section, allow_pickle=False)
    harmonics = np.load(args.harmonics, allow_pickle=False)
    structure_functions = np.load(args.structure_functions, allow_pickle=False)
    with args.matched_bins.open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))

    rows: list[dict[str, str]] = []
    grouped: dict[tuple[int, int], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row in source_rows:
        if row["strict_identical_3d_bin"] != "1" or row["combined_valid"] != "1":
            continue
        iq, ix, it = locate_campaign_bin(row, structure_functions)
        enriched = dict(row)
        enriched["campaign_iq"] = str(iq)
        enriched["campaign_ix"] = str(ix)
        enriched["campaign_it"] = str(it)
        rows.append(enriched)
        grouped[(iq, ix)].append((it, enriched))

    if not rows:
        raise ValueError("no strict valid Igor matches were found")

    phi_edges = np.asarray(cross_section["phi_edges"], dtype=float)
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    phi_dense = np.linspace(0.0, 360.0, 721)
    phi_dense_rad = np.deg2rad(phi_dense)
    q2_edges = np.asarray(cross_section["q2_edges"], dtype=float)
    xb_edges = np.asarray(cross_section["xb_edges"], dtype=float)
    t_edges = np.asarray(cross_section["t_edges"], dtype=float)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=CAMPAIGN_COLOR,
            marker="o",
            linewidth=0,
            markersize=5,
            label="combined campaign phi bins",
        ),
        Line2D(
            [0],
            [0],
            color=CAMPAIGN_COLOR,
            linewidth=1.7,
            label="campaign harmonic fit",
        ),
        Line2D(
            [0],
            [0],
            color=REFERENCE_COLOR,
            linewidth=1.7,
            linestyle="--",
            label="Igor SF reconstruction",
        ),
        Patch(
            facecolor=REFERENCE_COLOR,
            edgecolor="none",
            alpha=0.17,
            label="Igor statistical band (diagonal)",
        ),
    ]

    with PdfPages(args.output) as pdf:
        info = pdf.infodict()
        info["Title"] = "RGA combined unit-weight vs Igor Pass2: cross sections versus phi"
        info["Author"] = "SF analysis comparison"
        info["Subject"] = "Strict-bin comparison using campaign phi bins and reconstructed reference harmonics"
        cover_page(pdf, rows, len(grouped))

        for page_number, ((iq, ix), entries) in enumerate(sorted(grouped.items()), start=1):
            fig, axes = plt.subplots(2, 4, figsize=(11.0, 8.5), sharex=True)
            axes_flat = axes.ravel()
            entries.sort(key=lambda item: item[0])
            for panel, (it, row) in enumerate(entries):
                ax = axes_flat[panel]
                valid = np.asarray(
                    cross_section["final_validity_mask"][iq, ix, it], dtype=bool
                )
                values = np.asarray(
                    cross_section["reduced_cross_section"][iq, ix, it], dtype=float
                )
                uncertainties = np.asarray(
                    cross_section["uncertainty"][iq, ix, it], dtype=float
                )
                campaign_central, campaign_band = campaign_fit(
                    np.asarray(harmonics["parameters"][iq, ix, it], dtype=float),
                    np.asarray(harmonics["covariance"][iq, ix, it], dtype=float),
                    phi_dense_rad,
                )
                reference_central, reference_band, reference_eps = reconstruct_reference(
                    row, phi_dense_rad, args.reference_beam_energy
                )

                ax.errorbar(
                    phi_centers[valid],
                    values[valid],
                    yerr=uncertainties[valid],
                    fmt="o",
                    markersize=3.0,
                    capsize=1.5,
                    elinewidth=0.75,
                    color=CAMPAIGN_COLOR,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    zorder=4,
                )
                ax.plot(phi_dense, campaign_central, color=CAMPAIGN_COLOR, linewidth=1.4)
                ax.fill_between(
                    phi_dense,
                    campaign_central - campaign_band,
                    campaign_central + campaign_band,
                    color=CAMPAIGN_COLOR,
                    alpha=0.08,
                    linewidth=0,
                )
                ax.plot(
                    phi_dense,
                    reference_central,
                    color=REFERENCE_COLOR,
                    linewidth=1.5,
                    linestyle="--",
                )
                ax.fill_between(
                    phi_dense,
                    reference_central - reference_band,
                    reference_central + reference_band,
                    color=REFERENCE_COLOR,
                    alpha=0.17,
                    linewidth=0,
                )

                y_low, y_high = panel_limits(
                    values[valid], campaign_central, reference_central
                )
                ax.set_ylim(y_low, y_high)
                ax.set_xlim(0.0, 360.0)
                ax.set_xticks([0, 90, 180, 270, 360])
                ax.grid(True, alpha=0.22, linewidth=0.5)
                ax.tick_params(labelsize=7)
                ax.set_title(
                    rf"$-t\in[{t_edges[it]:g},{t_edges[it+1]:g}]$ GeV$^2$",
                    fontsize=9,
                    pad=3,
                )
                campaign_coordinates = (
                    scalar_float(row, "combined_Q2_reference"),
                    scalar_float(row, "combined_xB_reference"),
                    scalar_float(row, "combined_minus_t_coordinate"),
                )
                reference_coordinates = (
                    scalar_float(row, "reference_Q2"),
                    scalar_float(row, "reference_xB"),
                    scalar_float(row, "reference_minus_t"),
                )
                ax.text(
                    0.03,
                    0.97,
                    "campaign ref: "
                    + f"({campaign_coordinates[0]:.3g}, {campaign_coordinates[1]:.3g}, {campaign_coordinates[2]:.3g})\n"
                    + "Igor ref: "
                    + f"({reference_coordinates[0]:.3g}, {reference_coordinates[1]:.3g}, {reference_coordinates[2]:.3g}); "
                    + rf"$\epsilon={reference_eps:.3f}$",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=5.8,
                    color="#333333",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
                )
                if panel // 4 == 1:
                    ax.set_xlabel(r"$\phi$ (deg)", fontsize=8)
                if panel % 4 == 0:
                    ax.set_ylabel(
                        r"$d^2\sigma/(dt\,d\phi)$ [nb/(GeV$^2$ rad)]", fontsize=7.5
                    )

            for ax in axes_flat[len(entries) :]:
                ax.axis("off")

            fig.suptitle(
                rf"Strict matched bin: $Q^2\in[{q2_edges[iq]:g},{q2_edges[iq+1]:g}]$ GeV$^2$, "
                rf"$x_B\in[{xb_edges[ix]:g},{xb_edges[ix+1]:g}]$"
                + f"   ({page_number}/{len(grouped)})",
                fontsize=13,
                weight="bold",
                y=0.978,
            )
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.945),
                ncol=4,
                frameon=False,
                fontsize=8,
            )
            fig.text(
                0.5,
                0.012,
                "Igor line reconstructed from printed structure functions (including its 1.3 scale); "
                "it is not an Igor phi-bin data series.",
                ha="center",
                fontsize=7.4,
                color="#444444",
            )
            fig.subplots_adjust(
                left=0.075, right=0.985, bottom=0.075, top=0.89, wspace=0.30, hspace=0.30
            )
            pdf.savefig(fig)
            plt.close(fig)

    cross_section.close()
    harmonics.close()
    structure_functions.close()
    print(f"Strict matched phi panels: {len(rows)}")
    print(f"Q2-xB pages: {len(grouped)}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
