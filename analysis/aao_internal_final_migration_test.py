#!/usr/bin/env python3
"""Compare AAO hard-parent and final-LUND coordinates in one target cell.

The input is one or more AAO sampling-mode-1 fixed-trial survey directories.
Unlike an unweighted LUND sample, each survey row preserves both the Born-like
hard-parent coordinates and the coordinates reconstructed from the completed
final state.  The per-trial cross-section contribution is used as the weight.

This diagnostic answers a narrow question: do events ending in the observed
central-phi excess originate in the same hard ``(Q2, xB, -t, phi)`` region, or
are they fed into it by radiative/external-loss migration?  Native ``intreg=6``
is reported as the soft branch; ``intreg=1..5`` is the resolved/hard branch.
It does not generate LUND events and does not change AAO physics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray
CATEGORY_LABELS = ("all", "soft_intreg6", "hard_intreg1to5")
DEFAULT_CELLS = (
    "broad:1.0:2.0:0.25:0.38:0.20:0.40",
    "medium:1.25:1.75:0.27:0.33:0.25:0.35",
    "narrow:1.40:1.60:0.285:0.315:0.275:0.325",
)
SURVEY_SCHEMAS = {
    "# schema=aao-rad-survey-v1",
    "# schema=aao-rad-survey-v2",
}
REQUIRED_COLUMNS = {
    "final_valid",
    "candidate_status",
    "intreg",
    "energy_e_lund",
    "energy_gamma",
    "q2_hard",
    "xb_hard",
    "minus_t_hard",
    "phi_cm_deg",
    "q2_observed",
    "xb_observed",
    "minus_t_observed",
    "phi_observed_deg",
    "w_observed",
    "trial_xsec_observed_microbarn",
}


@dataclass(frozen=True)
class Cell:
    label: str
    q2: tuple[float, float]
    xb: tuple[float, float]
    minus_t: tuple[float, float]

    def contains(self, q2: float, xb: float, minus_t: float) -> bool:
        return (
            self.q2[0] <= q2 < self.q2[1]
            and self.xb[0] <= xb < self.xb[1]
            and self.minus_t[0] <= minus_t < self.minus_t[1]
        )


@dataclass(frozen=True)
class PhaseSpace:
    q2_min: float
    w_min: float
    electron_p_min: float
    y_max: float | None
    beam_energy: float

    def contains(self, q2: float, w: float, electron_energy: float) -> bool:
        if q2 < self.q2_min or w < self.w_min or electron_energy < self.electron_p_min:
            return False
        if self.y_max is not None:
            y = 1.0 - electron_energy / self.beam_energy
            if y > self.y_max:
                return False
        return True


@dataclass
class Accumulator:
    rows: Array
    sumw: Array
    sumw2: Array
    observed_phi: Array
    hard_phi: Array
    migration: Array
    delta_phi: Array
    central_source_phi: Array
    central_sumw: Array
    edge_sumw: Array
    central_from_hard_central: Array
    central_from_same_hard_cell: Array
    central_from_same_phi_bin: Array
    central_from_same_hard_cell_and_phi_bin: Array
    residual_sum: Array
    residual_sum2: Array
    delta_cos: Array
    delta_sin: Array


def parse_cell(specification: str) -> Cell:
    fields = specification.split(":")
    if len(fields) != 7 or not fields[0].strip():
        raise ValueError(
            "cell must be LABEL:Q2LOW:Q2HIGH:XBLOW:XBHIGH:TLOW:THIGH"
        )
    try:
        values = tuple(float(value) for value in fields[1:])
    except ValueError as error:
        raise ValueError(f"cell has a nonnumeric boundary: {specification}") from error
    pairs = ((values[0], values[1]), (values[2], values[3]), (values[4], values[5]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"cell has a non-finite boundary: {specification}")
    if any(high <= low for low, high in pairs):
        raise ValueError(f"cell upper edges must exceed lower edges: {specification}")
    return Cell(fields[0].strip(), pairs[0], pairs[1], pairs[2])


def validate_nested_cells(cells: Sequence[Cell]) -> None:
    if len({cell.label for cell in cells}) != len(cells):
        raise ValueError("cell labels must be unique")
    for outer, inner in zip(cells, cells[1:]):
        axes = zip(
            (outer.q2, outer.xb, outer.minus_t),
            (inner.q2, inner.xb, inner.minus_t),
            strict=True,
        )
        if not all(olo <= ilo and ihi <= ohi for (olo, ohi), (ilo, ihi) in axes):
            raise ValueError(f"cell {inner.label!r} is not nested in {outer.label!r}")


def wrap_degrees(values: Array | float) -> Array:
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def phi_index(phi_deg: float, edges: Array) -> int:
    wrapped = phi_deg % 360.0
    index = int(np.searchsorted(edges, wrapped, side="right") - 1)
    return index if 0 <= index < edges.size - 1 else -1


def _empty(number_cells: int, number_phi: int, number_delta: int) -> Accumulator:
    scalar = (number_cells, len(CATEGORY_LABELS))
    phi = scalar + (number_phi,)
    migration = scalar + (number_phi, number_phi)
    residual = scalar + (4,)
    return Accumulator(
        rows=np.zeros(scalar, dtype=np.int64),
        sumw=np.zeros(scalar),
        sumw2=np.zeros(scalar),
        observed_phi=np.zeros(phi),
        hard_phi=np.zeros(phi),
        migration=np.zeros(migration),
        delta_phi=np.zeros(scalar + (number_delta,)),
        central_source_phi=np.zeros(phi),
        central_sumw=np.zeros(scalar),
        edge_sumw=np.zeros(scalar),
        central_from_hard_central=np.zeros(scalar),
        central_from_same_hard_cell=np.zeros(scalar),
        central_from_same_phi_bin=np.zeros(scalar),
        central_from_same_hard_cell_and_phi_bin=np.zeros(scalar),
        residual_sum=np.zeros(residual),
        residual_sum2=np.zeros(residual),
        delta_cos=np.zeros(scalar),
        delta_sin=np.zeros(scalar),
    )


def _norm_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip().split()[0]
    return values


def _survey_signature(directory: Path, norm: dict[str, str]) -> str | None:
    config = directory / "survey_analysis_config.json"
    if not config.is_file():
        return None
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    settings = tuple(
        norm.get(name, "missing")
        for name in (
            "ebeam",
            "survey_schema",
            "survey_proposal",
            "survey_legacy_fraction",
            "epirea",
            "th_opt",
            "res_opt",
        )
    )
    return json.dumps((digest, settings), separators=(",", ":"))


def _float(row: dict[str, str], name: str, source: Path, line: int) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}:{line}: invalid {name}") from error
    if not math.isfinite(value):
        raise ValueError(f"{source}:{line}: non-finite {name}")
    return value


def accumulate_surveys(
    directories: Sequence[Path],
    cells: Sequence[Cell],
    phi_edges: Array,
    delta_edges: Array,
    phase_space: PhaseSpace,
) -> tuple[Accumulator, dict[str, object]]:
    result = _empty(len(cells), phi_edges.size - 1, delta_edges.size - 1)
    total_proposals = 0
    csv_rows = 0
    final_valid_rows = 0
    weighted_rows = 0
    signatures: set[str] = set()
    survey_records: list[dict[str, object]] = []
    for directory in directories:
        directory = directory.expanduser().resolve()
        csv_path = directory / "aao_rad.survey.csv"
        norm_path = directory / "aao_rad.norm"
        if not csv_path.is_file() or not norm_path.is_file():
            raise FileNotFoundError(f"{directory}: missing survey CSV or norm")
        norm = _norm_values(norm_path)
        try:
            proposals = int(norm["ntries"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"{norm_path}: missing valid ntries") from error
        requested = int(norm.get("survey_ntrials_requested", proposals))
        if proposals <= 0 or requested != proposals:
            raise ValueError(
                f"{norm_path}: ntries={proposals} differs from requested={requested}"
            )
        signature = _survey_signature(directory, norm)
        if signature is not None:
            signatures.add(signature)
        total_proposals += proposals
        rows_this = 0
        valid_this = 0
        weighted_this = 0
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            schema = stream.readline().rstrip("\n")
            if schema not in SURVEY_SCHEMAS:
                raise ValueError(f"{csv_path}: unsupported schema {schema!r}")
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
                missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
                raise ValueError(f"{csv_path}: missing columns {missing}")
            for line_number, row in enumerate(reader, start=3):
                rows_this += 1
                csv_rows += 1
                try:
                    final_valid = int(row["final_valid"])
                    status = int(row["candidate_status"])
                    intreg = int(row["intreg"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{csv_path}:{line_number}: malformed integer") from error
                if final_valid != 1 or status != 0:
                    continue
                valid_this += 1
                final_valid_rows += 1
                if intreg < 1 or intreg > 6:
                    raise ValueError(f"{csv_path}:{line_number}: intreg outside 1..6")
                weight = _float(
                    row, "trial_xsec_observed_microbarn", csv_path, line_number
                )
                if weight < 0.0:
                    raise ValueError(f"{csv_path}:{line_number}: negative contribution")
                if weight == 0.0:
                    continue
                weighted_this += 1
                weighted_rows += 1
                e_lund = _float(row, "energy_e_lund", csv_path, line_number)
                q2o = _float(row, "q2_observed", csv_path, line_number)
                xbo = _float(row, "xb_observed", csv_path, line_number)
                mto = _float(row, "minus_t_observed", csv_path, line_number)
                phio = _float(row, "phi_observed_deg", csv_path, line_number) % 360.0
                wo = _float(row, "w_observed", csv_path, line_number)
                if not phase_space.contains(q2o, wo, e_lund):
                    continue
                q2h = _float(row, "q2_hard", csv_path, line_number)
                xbh = _float(row, "xb_hard", csv_path, line_number)
                mth = _float(row, "minus_t_hard", csv_path, line_number)
                phih = _float(row, "phi_cm_deg", csv_path, line_number) % 360.0
                io = phi_index(phio, phi_edges)
                ih = phi_index(phih, phi_edges)
                delta = float(wrap_degrees(phio - phih))
                idelta = int(np.searchsorted(delta_edges, delta, side="right") - 1)
                if io < 0 or ih < 0 or not 0 <= idelta < delta_edges.size - 1:
                    raise ValueError(f"{csv_path}:{line_number}: angle outside histogram")
                category = 1 if intreg == 6 else 2
                categories = (0, category)
                central = 150.0 <= phio < 210.0
                edge = phio < 30.0 or phio >= 330.0
                hard_central = 150.0 <= phih < 210.0
                residual = np.asarray((q2o - q2h, xbo - xbh, mto - mth, delta))
                for icell, cell in enumerate(cells):
                    if not cell.contains(q2o, xbo, mto):
                        continue
                    same_hard_cell = cell.contains(q2h, xbh, mth)
                    for icategory in categories:
                        result.rows[icell, icategory] += 1
                        result.sumw[icell, icategory] += weight
                        result.sumw2[icell, icategory] += weight * weight
                        result.observed_phi[icell, icategory, io] += weight
                        result.hard_phi[icell, icategory, ih] += weight
                        result.migration[icell, icategory, ih, io] += weight
                        result.delta_phi[icell, icategory, idelta] += weight
                        result.residual_sum[icell, icategory] += weight * residual
                        result.residual_sum2[icell, icategory] += weight * residual**2
                        angle = math.radians(delta)
                        result.delta_cos[icell, icategory] += weight * math.cos(angle)
                        result.delta_sin[icell, icategory] += weight * math.sin(angle)
                        if edge:
                            result.edge_sumw[icell, icategory] += weight
                        if not central:
                            continue
                        result.central_sumw[icell, icategory] += weight
                        result.central_source_phi[icell, icategory, ih] += weight
                        if hard_central:
                            result.central_from_hard_central[icell, icategory] += weight
                        if same_hard_cell:
                            result.central_from_same_hard_cell[icell, icategory] += weight
                        if ih == io:
                            result.central_from_same_phi_bin[icell, icategory] += weight
                        if same_hard_cell and ih == io:
                            result.central_from_same_hard_cell_and_phi_bin[
                                icell, icategory
                            ] += weight
        survey_records.append(
            {
                "directory": str(directory),
                "proposals": proposals,
                "csv_rows": rows_this,
                "final_valid_rows": valid_this,
                "positive_observed_rows": weighted_this,
            }
        )
        print(
            f"read {directory}: proposals={proposals} rows={rows_this} "
            f"valid={valid_this} positive={weighted_this}",
            flush=True,
        )
    if len(signatures) > 1:
        raise ValueError("survey directories have different configurations/settings")
    if total_proposals <= 1:
        raise ValueError("combined proposal count must exceed one")
    metadata: dict[str, object] = {
        "total_proposals": total_proposals,
        "csv_rows": csv_rows,
        "final_valid_rows": final_valid_rows,
        "positive_observed_rows": weighted_rows,
        "survey_signature": next(iter(signatures), None),
        "surveys": survey_records,
    }
    return result, metadata


def _safe_fraction(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else math.nan


def summaries(
    result: Accumulator, cells: Sequence[Cell], proposals: int
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for icell, cell in enumerate(cells):
        categories: list[dict[str, object]] = []
        for icategory, label in enumerate(CATEGORY_LABELS):
            total = float(result.sumw[icell, icategory])
            total2 = float(result.sumw2[icell, icategory])
            central = float(result.central_sumw[icell, icategory])
            edge = float(result.edge_sumw[icell, icategory])
            mean = np.divide(
                result.residual_sum[icell, icategory],
                total,
                out=np.full(4, np.nan),
                where=total > 0.0,
            )
            variance = np.divide(
                result.residual_sum2[icell, icategory],
                total,
                out=np.full(4, np.nan),
                where=total > 0.0,
            ) - mean**2
            resultant = _safe_fraction(
                math.hypot(
                    result.delta_cos[icell, icategory],
                    result.delta_sin[icell, icategory],
                ),
                total,
            )
            delta_mean = math.degrees(
                math.atan2(
                    result.delta_sin[icell, icategory],
                    result.delta_cos[icell, icategory],
                )
            ) if total > 0.0 else math.nan
            categories.append(
                {
                    "category": label,
                    "rows": int(result.rows[icell, icategory]),
                    "cross_section_microbarn": total / proposals,
                    "cross_section_sem_microbarn": math.sqrt(
                        max(total2 - total * total / proposals, 0.0)
                        / (proposals * (proposals - 1))
                    ),
                    "effective_weighted_rows": total * total / total2 if total2 > 0 else 0.0,
                    "final_central_fraction": _safe_fraction(central, total),
                    "final_edge_fraction": _safe_fraction(edge, total),
                    "central_over_edge": _safe_fraction(central, edge),
                    "central_from_hard_central_fraction": _safe_fraction(
                        result.central_from_hard_central[icell, icategory], central
                    ),
                    "central_from_same_hard_cell_fraction": _safe_fraction(
                        result.central_from_same_hard_cell[icell, icategory], central
                    ),
                    "central_from_same_phi_bin_fraction": _safe_fraction(
                        result.central_from_same_phi_bin[icell, icategory], central
                    ),
                    "central_from_same_hard_cell_and_phi_bin_fraction": _safe_fraction(
                        result.central_from_same_hard_cell_and_phi_bin[
                            icell, icategory
                        ],
                        central,
                    ),
                    "weighted_mean_final_minus_hard": {
                        "Q2_GeV2": float(mean[0]),
                        "xB": float(mean[1]),
                        "minus_t_GeV2": float(mean[2]),
                        "phi_wrapped_deg": delta_mean,
                    },
                    "weighted_rms_final_minus_hard": {
                        "Q2_GeV2": float(math.sqrt(max(variance[0], 0.0))),
                        "xB": float(math.sqrt(max(variance[1], 0.0))),
                        "minus_t_GeV2": float(math.sqrt(max(variance[2], 0.0))),
                    },
                    "phi_delta_resultant": resultant,
                }
            )
        output.append(
            {
                "label": cell.label,
                "Q2": list(cell.q2),
                "xB": list(cell.xb),
                "minus_t": list(cell.minus_t),
                "categories": categories,
            }
        )
    return output


def write_artifacts(
    output: Path,
    result: Accumulator,
    cells: Sequence[Cell],
    phi_edges: Array,
    delta_edges: Array,
    metadata: dict[str, object],
    phase_space: PhaseSpace,
) -> dict[str, object]:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    proposals = int(metadata["total_proposals"])
    cell_summary = summaries(result, cells, proposals)
    payload = {
        "schema": "aao-internal-final-migration-test-v1",
        "diagnostic_only": True,
        "coordinate_definitions": {
            "hard_parent": "q2_hard, xb_hard, minus_t_hard, phi_cm_deg",
            "observed": (
                "q2_observed, xb_observed, minus_t_observed, "
                "phi_observed_deg from final LUND particles"
            ),
            "soft": "native AAO intreg=6",
            "hard": "native AAO intreg=1..5",
            "weight": "trial_xsec_observed_microbarn",
        },
        "phase_space": {
            "Q2_min": phase_space.q2_min,
            "W_min": phase_space.w_min,
            "electron_p_min": phase_space.electron_p_min,
            "y_max": phase_space.y_max,
            "beam_energy": phase_space.beam_energy,
        },
        **metadata,
        "cells": cell_summary,
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output.with_suffix(".npz"),
        category_labels=np.asarray(CATEGORY_LABELS),
        cell_labels=np.asarray([cell.label for cell in cells]),
        cell_bounds=np.asarray(
            [(*cell.q2, *cell.xb, *cell.minus_t) for cell in cells]
        ),
        phi_edges_deg=phi_edges,
        delta_phi_edges_deg=delta_edges,
        rows=result.rows,
        sumw=result.sumw,
        sumw2=result.sumw2,
        observed_phi_sumw=result.observed_phi,
        hard_phi_sumw=result.hard_phi,
        migration_sumw=result.migration,
        delta_phi_sumw=result.delta_phi,
        central_source_phi_sumw=result.central_source_phi,
        central_sumw=result.central_sumw,
        edge_sumw=result.edge_sumw,
        total_proposals=proposals,
    )
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "cell", "category", "phi_low_deg", "phi_high_deg",
                "observed_cross_section_microbarn", "hard_parent_cross_section_microbarn",
                "central_final_source_cross_section_microbarn",
            ]
        )
        for icell, cell in enumerate(cells):
            for icategory, category in enumerate(CATEGORY_LABELS):
                for iphi in range(phi_edges.size - 1):
                    writer.writerow(
                        [
                            cell.label,
                            category,
                            phi_edges[iphi],
                            phi_edges[iphi + 1],
                            result.observed_phi[icell, icategory, iphi] / proposals,
                            result.hard_phi[icell, icategory, iphi] / proposals,
                            result.central_source_phi[icell, icategory, iphi] / proposals,
                        ]
                    )
    centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    delta_centers = 0.5 * (delta_edges[:-1] + delta_edges[1:])
    figure, axes = plt.subplots(
        len(cells), 4, figsize=(18.0, max(4.0, 4.2 * len(cells))), squeeze=False
    )
    for icell, cell in enumerate(cells):
        observed = result.observed_phi[icell, 0]
        hard = result.hard_phi[icell, 0]
        axes[icell, 0].stairs(
            observed / np.sum(observed), phi_edges, label="final observed", linewidth=1.7
        )
        axes[icell, 0].stairs(
            hard / np.sum(hard), phi_edges, label="hard parent", linewidth=1.4
        )
        axes[icell, 0].set_ylabel(f"{cell.label}\nweighted fraction")
        axes[icell, 0].legend(fontsize=8)

        migration = result.migration[icell, 0]
        migration_fraction = 100.0 * migration / np.sum(migration)
        mesh = axes[icell, 1].pcolormesh(
            phi_edges, phi_edges, migration_fraction.T, shading="auto", cmap="viridis"
        )
        axes[icell, 1].plot([0, 360], [0, 360], "w--", linewidth=0.8)
        axes[icell, 1].set_xlabel(r"hard $\phi$ [deg]")
        axes[icell, 1].set_ylabel(r"final $\phi$ [deg]")
        figure.colorbar(mesh, ax=axes[icell, 1], label="cell cross section [%]")

        central_total = np.sum(result.central_source_phi[icell, 0])
        for category, color in ((1, "tab:blue"), (2, "tab:orange")):
            axes[icell, 2].stairs(
                result.central_source_phi[icell, category] / central_total,
                phi_edges,
                label=CATEGORY_LABELS[category],
                color=color,
            )
        axes[icell, 2].axvspan(150, 210, color="0.8", alpha=0.35)
        axes[icell, 2].set_ylabel("fraction of final-central cross section")
        axes[icell, 2].legend(fontsize=8)

        for category, color in ((1, "tab:blue"), (2, "tab:orange")):
            values = result.delta_phi[icell, category]
            if np.sum(values) > 0:
                axes[icell, 3].plot(
                    delta_centers,
                    values / np.sum(values),
                    drawstyle="steps-mid",
                    label=CATEGORY_LABELS[category],
                    color=color,
                )
        axes[icell, 3].axvline(0.0, color="0.4", linestyle="--", linewidth=0.8)
        axes[icell, 3].set_ylabel("weighted fraction")
        axes[icell, 3].legend(fontsize=8)
        for axis in axes[icell]:
            axis.grid(alpha=0.18)
    axes[0, 0].set_title("hard-parent and final phi")
    axes[0, 1].set_title("hard-to-final migration")
    axes[0, 2].set_title(r"hard-parent sources of final $150^\circ\!<\phi<210^\circ$")
    axes[0, 3].set_title(r"wrap($\phi_{final}-\phi_{hard}$)")
    for axis in axes[-1, (0, 2)]:
        axis.set_xlabel(r"$\phi$ [deg]")
    axes[-1, 3].set_xlabel(r"$\Delta\phi$ [deg]")
    figure.suptitle("AAO fixed-trial hard-parent versus final-coordinate migration")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(figure)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--survey", action="append", required=True,
        help="AAO fixed-trial survey directory; repeat for independent replicas",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="output prefix")
    parser.add_argument(
        "--cell", action="append", default=[],
        metavar="LABEL:Q2LO:Q2HI:XBLO:XBHI:TLO:THI",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cells = tuple(parse_cell(value) for value in (args.cell or DEFAULT_CELLS))
    validate_nested_cells(cells)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    phi_edges = np.asarray(config["binning"]["phi_deg"], dtype=float)
    if (
        phi_edges.ndim != 1
        or phi_edges.size < 2
        or phi_edges[0] != 0.0
        or phi_edges[-1] != 360.0
        or np.any(np.diff(phi_edges) <= 0.0)
    ):
        raise ValueError("config binning.phi_deg must increase from 0 to 360")
    phase = config.get("phase_space", {})
    phase_space = PhaseSpace(
        q2_min=float(phase.get("Q2_min", -math.inf)),
        w_min=float(phase.get("W_min", -math.inf)),
        electron_p_min=float(phase.get("electron_p_min", -math.inf)),
        y_max=float(phase["y_max"]) if "y_max" in phase else None,
        beam_energy=float(config["beam_energy"]),
    )
    delta_edges = np.linspace(-180.0, 180.0, 73)
    result, metadata = accumulate_surveys(
        [Path(value) for value in args.survey],
        cells,
        phi_edges,
        delta_edges,
        phase_space,
    )
    payload = write_artifacts(
        Path(args.output), result, cells, phi_edges, delta_edges, metadata, phase_space
    )
    print(json.dumps(payload, indent=2))
    for suffix in (".npz", ".csv", ".json", ".pdf", ".png"):
        print(f"wrote {Path(args.output).with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
