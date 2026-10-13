#!/usr/bin/env python3
"""Test whether the AAO phi peak is caused by finite-cell migration.

The program streams existing Born and radiative LUND samples once and forms
``C_rad(phi)`` in several nested ``(Q2, xB, -t)`` windows around one point.
Radiative LUND stochastic multiplicities are retained as cluster weights, and
the soft (10-keV placeholder) and hard photon contributions share the same
normalization, exactly as in ``aao_soft_hard_radiative_correction.py``.

If the central peak is mainly finite-bin averaging or radiative feed-in, it
should decrease as the windows contract.  This is an output-level diagnostic;
it does not replace a later comparison using AAO's internal hard-scattering
coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.aao_soft_hard_radiative_correction import (
    cluster_chunks,
    integrated_cross_sections,
    manifest_paths,
    ratio_and_error,
)
from analysis.eppi0.binning import bin_indices
from analysis.eppi0.event_sample import _dis, _minus_t, _trento_phi
from analysis.eppi0.phase_space import AnalysisPhaseSpace


Array = np.ndarray


@dataclass(frozen=True)
class Cell:
    label: str
    q2: tuple[float, float]
    xb: tuple[float, float]
    minus_t: tuple[float, float]

    @property
    def volume(self) -> float:
        return (
            (self.q2[1] - self.q2[0])
            * (self.xb[1] - self.xb[0])
            * (self.minus_t[1] - self.minus_t[0])
        )

    def contains(self, point: tuple[float, float, float]) -> bool:
        return all(
            low <= value < high
            for value, (low, high) in zip(
                point, (self.q2, self.xb, self.minus_t), strict=True
            )
        )


@dataclass
class Histograms:
    counts: Array
    sumw2: Array
    soft_counts: Array
    soft_sumw2: Array
    hard_counts: Array
    hard_sumw2: Array
    topology_events: int
    clusters: int
    maximum_multiplicity: int


DEFAULT_CELLS = (
    "broad:1.0:2.0:0.25:0.38:0.20:0.40",
    "medium:1.25:1.75:0.27:0.33:0.25:0.35",
    "narrow:1.40:1.60:0.285:0.315:0.275:0.325",
)


def parse_cell(specification: str) -> Cell:
    fields = specification.split(":")
    if len(fields) != 7:
        raise ValueError(
            "cell must be LABEL:Q2LOW:Q2HIGH:XBLOW:XBHIGH:TLOW:THIGH"
        )
    label = fields[0].strip()
    if not label:
        raise ValueError("cell label must not be empty")
    try:
        values = tuple(float(value) for value in fields[1:])
    except ValueError as error:
        raise ValueError(f"cell has a nonnumeric boundary: {specification}") from error
    pairs = ((values[0], values[1]), (values[2], values[3]), (values[4], values[5]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"cell has a non-finite boundary: {specification}")
    if any(high <= low for low, high in pairs):
        raise ValueError(f"cell upper edges must exceed lower edges: {specification}")
    return Cell(label=label, q2=pairs[0], xb=pairs[1], minus_t=pairs[2])


def validate_nested_cells(
    cells: Sequence[Cell], point: tuple[float, float, float]
) -> None:
    labels = [cell.label for cell in cells]
    if len(labels) != len(set(labels)):
        raise ValueError("cell labels must be unique")
    for cell in cells:
        if not cell.contains(point):
            raise ValueError(f"cell {cell.label!r} does not contain the reference point")
    for outer, inner in zip(cells, cells[1:]):
        bounds = zip(
            (outer.q2, outer.xb, outer.minus_t),
            (inner.q2, inner.xb, inner.minus_t),
            strict=True,
        )
        if not all(olo <= ilo and ihi <= ohi for (olo, ohi), (ilo, ihi) in bounds):
            raise ValueError(f"cell {inner.label!r} is not nested inside {outer.label!r}")


def _empty_histograms(number_cells: int, number_phi: int) -> Histograms:
    shape = (number_cells, number_phi)
    return Histograms(
        counts=np.zeros(shape),
        sumw2=np.zeros(shape),
        soft_counts=np.zeros(shape),
        soft_sumw2=np.zeros(shape),
        hard_counts=np.zeros(shape),
        hard_sumw2=np.zeros(shape),
        topology_events=0,
        clusters=0,
        maximum_multiplicity=0,
    )


def wrapped_phi_bin_indices(phi_rad: Array, phi_edges_deg: Array) -> Array:
    """Bin signed Trento angles after wrapping them onto [0, 360) degrees."""
    phi_deg = np.mod(np.asarray(phi_rad, dtype=float), 2.0 * np.pi)
    phi_deg *= 180.0 / np.pi
    return bin_indices(phi_deg, phi_edges_deg)


def accumulate(
    files: Sequence[Path],
    cells: Sequence[Cell],
    phi_edges: Array,
    phase_space: AnalysisPhaseSpace,
    *,
    beam_energy: float,
    radiative: bool,
    photon_threshold: float,
) -> Histograms:
    result = _empty_histograms(len(cells), len(phi_edges) - 1)
    for electron, proton, photon_energy, multiplicity in cluster_chunks(
        files, radiative=radiative
    ):
        result.topology_events += int(np.sum(multiplicity))
        result.clusters += int(multiplicity.size)
        result.maximum_multiplicity = max(
            result.maximum_multiplicity, int(np.max(multiplicity))
        )
        q2, xb = _dis(electron, beam_energy)
        minus_t = _minus_t(proton)
        phi_rad = _trento_phi(electron, proton, beam_energy)
        iphi = wrapped_phi_bin_indices(phi_rad, phi_edges)
        base = phase_space.mask(q2, xb, beam_energy) & (iphi >= 0)
        is_soft = photon_energy <= photon_threshold if radiative else None
        for index, cell in enumerate(cells):
            selected = base.copy()
            selected &= (q2 >= cell.q2[0]) & (q2 < cell.q2[1])
            selected &= (xb >= cell.xb[0]) & (xb < cell.xb[1])
            selected &= (minus_t >= cell.minus_t[0]) & (minus_t < cell.minus_t[1])
            if not np.any(selected):
                continue
            if not radiative:
                result.counts[index] += np.bincount(
                    iphi[selected],
                    weights=multiplicity[selected],
                    minlength=len(phi_edges) - 1,
                )
                result.sumw2[index] += np.bincount(
                    iphi[selected],
                    weights=np.square(multiplicity[selected]),
                    minlength=len(phi_edges) - 1,
                )
                continue
            assert is_soft is not None
            selected_soft = selected & is_soft
            selected_hard = selected & ~is_soft
            result.soft_counts[index] += np.bincount(
                iphi[selected_soft],
                weights=multiplicity[selected_soft],
                minlength=len(phi_edges) - 1,
            )
            result.soft_sumw2[index] += np.bincount(
                iphi[selected_soft],
                weights=np.square(multiplicity[selected_soft]),
                minlength=len(phi_edges) - 1,
            )
            result.hard_counts[index] += np.bincount(
                iphi[selected_hard],
                weights=multiplicity[selected_hard],
                minlength=len(phi_edges) - 1,
            )
            result.hard_sumw2[index] += np.bincount(
                iphi[selected_hard],
                weights=np.square(multiplicity[selected_hard]),
                minlength=len(phi_edges) - 1,
            )
    return result


def central_summary(
    phi_edges: Array, values: Array, errors: Array
) -> dict[str, float]:
    central = (phi_edges[:-1] >= 150.0) & (phi_edges[1:] <= 210.0)
    edge = (phi_edges[1:] <= 30.0) | (phi_edges[:-1] >= 330.0)
    if not np.any(central) or not np.any(edge):
        raise ValueError("phi edges must include 150-210 and the 0/360 edge bins")
    central_value = float(np.nanmean(values[central]))
    edge_value = float(np.nanmean(values[edge]))
    central_error = float(np.sqrt(np.nansum(np.square(errors[central]))) / np.sum(central))
    edge_error = float(np.sqrt(np.nansum(np.square(errors[edge]))) / np.sum(edge))
    return {
        "central_mean": central_value,
        "central_error": central_error,
        "edge_mean": edge_value,
        "edge_error": edge_error,
        "central_over_edge": central_value / edge_value,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--born-manifest", required=True)
    parser.add_argument("--radiative-manifest", required=True)
    parser.add_argument("--normalization-artifact", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="output prefix without suffix")
    parser.add_argument("--expected-files", type=int, default=2000)
    parser.add_argument("--photon-threshold", type=float, default=0.010)
    parser.add_argument("--q2", type=float, default=1.5)
    parser.add_argument("--xb", type=float, default=0.30)
    parser.add_argument("--minus-t", type=float, default=0.30)
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        metavar="LABEL:Q2LO:Q2HI:XBLO:XBHI:TLO:THI",
        help="nested cell; repeat from broadest to narrowest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    point = (float(args.q2), float(args.xb), float(args.minus_t))
    cells = tuple(parse_cell(spec) for spec in (args.cell or DEFAULT_CELLS))
    validate_nested_cells(cells, point)
    born_files = manifest_paths(args.born_manifest, args.expected_files)
    radiative_files = manifest_paths(args.radiative_manifest, args.expected_files)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    beam_energy = float(config["beam_energy"])
    phi_edges = np.asarray(config["binning"]["phi_deg"], dtype=float)
    phase_space = AnalysisPhaseSpace.from_config(config)
    born = accumulate(
        born_files,
        cells,
        phi_edges,
        phase_space,
        beam_energy=beam_energy,
        radiative=False,
        photon_threshold=args.photon_threshold,
    )
    radiative = accumulate(
        radiative_files,
        cells,
        phi_edges,
        phase_space,
        beam_energy=beam_energy,
        radiative=True,
        photon_threshold=args.photon_threshold,
    )
    sigma_born, sigma_rad = integrated_cross_sections(args.normalization_artifact)
    normalization = (
        sigma_rad / sigma_born
        * born.topology_events / radiative.topology_events
    )
    radiative_counts = radiative.soft_counts + radiative.hard_counts
    radiative_sumw2 = radiative.soft_sumw2 + radiative.hard_sumw2
    shape = born.counts.shape
    c_total = np.full(shape, np.nan)
    dc_total = np.full(shape, np.nan)
    c_soft = np.full(shape, np.nan)
    dc_soft = np.full(shape, np.nan)
    c_hard = np.full(shape, np.nan)
    dc_hard = np.full(shape, np.nan)
    summaries: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        c_total[index], dc_total[index] = ratio_and_error(
            radiative_counts[index], radiative_sumw2[index],
            born.counts[index], born.sumw2[index], normalization,
        )
        c_soft[index], dc_soft[index] = ratio_and_error(
            radiative.soft_counts[index], radiative.soft_sumw2[index],
            born.counts[index], born.sumw2[index], normalization,
        )
        c_hard[index], dc_hard[index] = ratio_and_error(
            radiative.hard_counts[index], radiative.hard_sumw2[index],
            born.counts[index], born.sumw2[index], normalization,
        )
        metric = central_summary(phi_edges, c_total[index], dc_total[index])
        summaries.append(
            {
                "label": cell.label,
                "Q2": list(cell.q2),
                "xB": list(cell.xb),
                "minus_t": list(cell.minus_t),
                "volume": cell.volume,
                "volume_relative_to_broad": cell.volume / cells[0].volume,
                "born_events_in_cell": int(np.sum(born.counts[index])),
                "radiative_events_in_cell": int(np.sum(radiative_counts[index])),
                "soft_events_in_cell": int(np.sum(radiative.soft_counts[index])),
                "hard_events_in_cell": int(np.sum(radiative.hard_counts[index])),
                "integrated_C_rad_in_cell": float(
                    normalization
                    * np.sum(radiative_counts[index])
                    / np.sum(born.counts[index])
                ),
                **metric,
            }
        )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cell_bounds = np.asarray(
        [(*cell.q2, *cell.xb, *cell.minus_t) for cell in cells], dtype=float
    )
    np.savez_compressed(
        output.with_suffix(".npz"),
        labels=np.asarray([cell.label for cell in cells]),
        cell_bounds=cell_bounds,
        phi_edges_deg=phi_edges,
        born_counts=born.counts,
        born_cluster_sumw2=born.sumw2,
        soft_counts=radiative.soft_counts,
        soft_cluster_sumw2=radiative.soft_sumw2,
        hard_counts=radiative.hard_counts,
        hard_cluster_sumw2=radiative.hard_sumw2,
        c_soft=c_soft,
        delta_c_soft=dc_soft,
        c_hard=c_hard,
        delta_c_hard=dc_hard,
        c_total=c_total,
        delta_c_total=dc_total,
        normalization_ratio=normalization,
        born_integrated_cross_section=sigma_born,
        radiative_integrated_cross_section=sigma_rad,
    )
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "cell", "phi_low_deg", "phi_high_deg", "born_events",
                "soft_events", "hard_events", "C_soft", "delta_C_soft",
                "C_hard", "delta_C_hard", "C_total", "delta_C_total",
            ]
        )
        for icell, cell in enumerate(cells):
            for iphi in range(len(phi_edges) - 1):
                writer.writerow(
                    [
                        cell.label, phi_edges[iphi], phi_edges[iphi + 1],
                        born.counts[icell, iphi], radiative.soft_counts[icell, iphi],
                        radiative.hard_counts[icell, iphi], c_soft[icell, iphi],
                        dc_soft[icell, iphi], c_hard[icell, iphi],
                        dc_hard[icell, iphi], c_total[icell, iphi],
                        dc_total[icell, iphi],
                    ]
                )
    summary = {
        "schema": "aao-nested-cell-radiative-correction-v1",
        "purpose": "finite-cell and final-coordinate migration diagnostic",
        "reference_point": {"Q2": point[0], "xB": point[1], "minus_t": point[2]},
        "photon_threshold_GeV": float(args.photon_threshold),
        "born_files": len(born_files),
        "radiative_files": len(radiative_files),
        "born_topology_events": born.topology_events,
        "radiative_topology_events": radiative.topology_events,
        "normalization_ratio": normalization,
        "cells": summaries,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 9.0), sharex=True)
    for index, cell in enumerate(cells):
        axes[0].errorbar(
            centers, c_total[index], yerr=dc_total[index], marker="o",
            capsize=2, label=f"{cell.label} ({int(np.sum(born.counts[index])):,} Born)",
        )
        born_shape = len(centers) * born.counts[index] / np.sum(born.counts[index])
        rad_shape = len(centers) * radiative_counts[index] / np.sum(radiative_counts[index])
        axes[1].plot(
            centers,
            np.divide(rad_shape, born_shape, out=np.full_like(rad_shape, np.nan), where=born_shape > 0),
            marker="o",
            label=cell.label,
        )
    axes[0].axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    axes[0].set_ylabel(r"$C_{rad}$")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2)
    axes[1].axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    axes[1].set_xlabel(r"Trento $\phi$ [deg]")
    axes[1].set_ylabel("unit-normalized radiative / Born shape")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.2)
    figure.suptitle(
        "AAO final-LUND nested-cell migration test\n"
        rf"reference point $Q^2={point[0]:g}$ GeV$^2$, $x_B={point[1]:g}$, $-t={point[2]:g}$ GeV$^2$"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    for suffix in (".npz", ".csv", ".json", ".pdf", ".png"):
        print(f"wrote {output.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
