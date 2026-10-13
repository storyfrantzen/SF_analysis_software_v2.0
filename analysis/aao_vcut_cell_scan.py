#!/usr/bin/env python3
"""Scan final-LUND inelasticity cuts in one AAO kinematic cell.

The scan uses ``v = MX2(e'p) - m_pi0**2`` formed from the nominal beam and
the final LUND electron and proton.  Every curve shares the uncut radiative
normalization; selected tails are never renormalized back to unit area.
Exact consecutive mode-3 multiplicity copies are kept as one proposal cluster
for the statistical uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aao_soft_hard_radiative_correction import (
    _cell_indices,
    _chunk_phi_indices,
    cluster_chunks,
    histogram_cell,
    integrated_cross_sections,
    manifest_paths,
    ratio_and_error,
)
from analysis.eppi0.binning import from_config
from analysis.eppi0.phase_space import AnalysisPhaseSpace
from analysis.exclurad_phi_radiative_correction import inelasticity_v


def parse_cuts(value: str) -> tuple[float, ...]:
    result: list[float] = []
    for token in value.split(","):
        token = token.strip().lower()
        cut = np.inf if token in {"none", "inf", "uncut"} else float(token)
        if cut <= 0.0:
            raise ValueError("finite v cuts must be positive")
        result.append(cut)
    if not result or len(set(result)) != len(result):
        raise ValueError("v cuts must be a nonempty unique list")
    return tuple(result)


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--born-manifest", required=True)
    parser.add_argument("--radiative-manifest", required=True)
    parser.add_argument("--normalization-artifact")
    parser.add_argument("--born-integrated-cross-section", type=float)
    parser.add_argument("--radiative-integrated-cross-section", type=float)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="output prefix")
    parser.add_argument("--expected-files", type=int, default=2000)
    parser.add_argument("--expected-born-files", type=int)
    parser.add_argument("--expected-radiative-files", type=int)
    parser.add_argument("--v-cuts", default="0.05,0.10,0.17,0.20,0.30,0.50,none")
    parser.add_argument("--q2", type=float, default=1.5)
    parser.add_argument("--xb", type=float, default=0.30)
    parser.add_argument("--minus-t", type=float, default=0.30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = arguments(argv)
    cuts = parse_cuts(args.v_cuts)
    expected_born = args.expected_born_files or args.expected_files
    expected_radiative = args.expected_radiative_files or args.expected_files
    born_files = manifest_paths(args.born_manifest, expected_born)
    rad_files = manifest_paths(args.radiative_manifest, expected_radiative)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    binning = from_config(args.config)
    phase_space = AnalysisPhaseSpace.from_config(config)
    beam_energy = float(config["beam_energy"])
    point = (float(args.q2), float(args.xb), float(args.minus_t))
    cell = _cell_indices(binning, point)

    born = histogram_cell(
        born_files,
        binning,
        phase_space,
        beam_energy=beam_energy,
        point=point,
        radiative=False,
        photon_threshold=0.010,
    )
    shape = (len(cuts), binning.phi_edges.size - 1)
    rad_counts = np.zeros(shape, dtype=float)
    rad_sumw2 = np.zeros(shape, dtype=float)
    total_rad_events = 0
    total_rad_clusters = 0
    maximum_multiplicity = 0
    cell_uncut_events = 0
    cell_v_values: list[np.ndarray] = []
    cell_v_weights: list[np.ndarray] = []

    for electron, proton, _photon, multiplicity in cluster_chunks(
        rad_files, radiative=True
    ):
        total_rad_events += int(np.sum(multiplicity))
        total_rad_clusters += multiplicity.size
        maximum_multiplicity = max(maximum_multiplicity, int(np.max(multiplicity)))
        selected_cell, iphi = _chunk_phi_indices(
            electron, proton, binning, phase_space, beam_energy, cell
        )
        v = inelasticity_v(electron, proton, beam_energy)
        valid_cell = selected_cell & np.isfinite(v)
        cell_uncut_events += int(np.sum(multiplicity[valid_cell]))
        if np.any(valid_cell):
            cell_v_values.append(v[valid_cell])
            cell_v_weights.append(multiplicity[valid_cell])
        for index, cut in enumerate(cuts):
            selected = valid_cell & (v < cut)
            rad_counts[index] += np.bincount(
                iphi[selected], weights=multiplicity[selected], minlength=shape[1]
            )
            rad_sumw2[index] += np.bincount(
                iphi[selected],
                weights=np.square(multiplicity[selected]),
                minlength=shape[1],
            )

    supplied_cross_sections = (
        args.born_integrated_cross_section,
        args.radiative_integrated_cross_section,
    )
    if args.normalization_artifact:
        if any(value is not None for value in supplied_cross_sections):
            raise ValueError(
                "normalization artifact cannot be combined with explicit cross sections"
            )
        sigma_born, sigma_rad = integrated_cross_sections(
            args.normalization_artifact
        )
    else:
        if any(value is None for value in supplied_cross_sections):
            raise ValueError(
                "provide either --normalization-artifact or both integrated cross sections"
            )
        sigma_born, sigma_rad = (float(value) for value in supplied_cross_sections)
        if (
            not np.isfinite(sigma_born)
            or sigma_born <= 0.0
            or not np.isfinite(sigma_rad)
            or sigma_rad <= 0.0
        ):
            raise ValueError("integrated cross sections must be positive and finite")
    if total_rad_events == 0 or cell_uncut_events == 0:
        raise ValueError("radiative sample has no usable events in the requested cell")
    normalization = (
        sigma_rad / sigma_born
        * born.stats.topology_events / total_rad_events
    )
    corrections = np.zeros(shape, dtype=float)
    uncertainties = np.zeros(shape, dtype=float)
    for index in range(len(cuts)):
        corrections[index], uncertainties[index] = ratio_and_error(
            rad_counts[index],
            rad_sumw2[index],
            born.counts,
            born.sumw2,
            normalization,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    phi_edges = binning.phi_edges
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    labels = ["uncut" if np.isinf(cut) else f"v < {cut:g}" for cut in cuts]
    cell_v = np.concatenate(cell_v_values) if cell_v_values else np.asarray([])
    cell_weights = (
        np.concatenate(cell_v_weights) if cell_v_weights else np.asarray([])
    )

    np.savez_compressed(
        output.with_suffix(".npz"),
        v_cuts_GeV2=np.asarray(cuts),
        labels=np.asarray(labels),
        phi_edges_deg=phi_edges,
        C_rad=corrections,
        delta_C=uncertainties,
        H_born=born.counts,
        H_rad=rad_counts,
        H_born_cluster_sumw2=born.sumw2,
        H_rad_cluster_sumw2=rad_sumw2,
        normalization_ratio=normalization,
        born_integrated_cross_section=sigma_born,
        radiative_integrated_cross_section=sigma_rad,
        born_topology_events=born.stats.topology_events,
        radiative_topology_events=total_rad_events,
        cell_uncut_radiative_events=cell_uncut_events,
    )

    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["v_cut_GeV2", "phi_low_deg", "phi_high_deg", "C_rad", "delta_C", "born_events", "radiative_events"]
        )
        for iv, cut in enumerate(cuts):
            for iphi in range(phi.size):
                writer.writerow(
                    [cut, phi_edges[iphi], phi_edges[iphi + 1], corrections[iv, iphi], uncertainties[iv, iphi], born.counts[iphi], rad_counts[iv, iphi]]
                )

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 8.5), sharex=False)
    for index, label in enumerate(labels):
        axes[0].errorbar(
            phi,
            corrections[index],
            yerr=uncertainties[index],
            marker="o",
            linewidth=1.2,
            markersize=3.5,
            label=label,
        )
    axes[0].axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    axes[0].set_xlabel(r"Trento $\phi$ [deg]")
    axes[0].set_ylabel(r"$C_{rad}=\sigma_{rad}(v<v_{max})/\sigma_{Born}$")
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=8)
    if cell_v.size:
        histogram_low = min(-0.1, float(np.min(cell_v)))
        histogram_high = min(1.0, float(np.max(cell_v)))
        if histogram_high <= histogram_low:
            histogram_high = histogram_low + 0.1
        axes[1].hist(
            cell_v,
            bins=np.linspace(histogram_low, histogram_high, 100),
            weights=cell_weights,
            histtype="step",
            color="black",
        )
        for cut in cuts:
            if np.isfinite(cut):
                axes[1].axvline(cut, linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel(r"final-LUND $v=M_X^2(e'p)-m_{\pi^0}^2$ [GeV$^2$]")
    axes[1].set_ylabel("emitted events")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.2)
    figure.suptitle(
        "AAO final-LUND v-cut scan\n"
        rf"$Q^2\in[{binning.q2_edges[cell[0]]:g},{binning.q2_edges[cell[0]+1]:g})$, "
        rf"$x_B\in[{binning.xb_edges[cell[1]]:g},{binning.xb_edges[cell[1]+1]:g})$, "
        rf"$-t\in[{binning.t_edges[cell[2]]:g},{binning.t_edges[cell[2]+1]:g})$"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(figure)

    summary = {
        "definition": "v = MX2(e'p final LUND) - m_pi0^2; one-sided v < v_max",
        "normalization": "all cuts retain the same uncut radiative normalization",
        "v_cuts_GeV2": [None if np.isinf(cut) else cut for cut in cuts],
        "born_files": len(born_files),
        "radiative_files": len(rad_files),
        "born_topology_events": born.stats.topology_events,
        "radiative_topology_events": total_rad_events,
        "radiative_topology_clusters": total_rad_clusters,
        "maximum_radiative_cluster_multiplicity": maximum_multiplicity,
        "cell_uncut_radiative_events": cell_uncut_events,
        "cell_retention": {
            label: float(np.sum(rad_counts[index]) / cell_uncut_events)
            for index, label in enumerate(labels)
        },
        "normalization_ratio": normalization,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    for suffix in (".npz", ".csv", ".pdf", ".png", ".json"):
        print(f"wrote {output.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
