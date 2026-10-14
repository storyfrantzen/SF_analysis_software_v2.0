#!/usr/bin/env python3
"""Build an analysis-binned EXCLURAD radiative-correction artifact.

The output NPZ is compatible with ``run_analysis.py
radiative-correction-plots``.  Unlike the generic AAO path, every EXCLURAD
LUND file is normalized by the ``sigma_nb`` in its matching ``config.json``
and the radiative sample receives the offline inelasticity cut.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Support both ``python -m analysis...`` and direct execution from the repository.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.eppi0.binning import AnalysisBinning, from_config
from analysis.eppi0.phase_space import AnalysisPhaseSpace
from analysis.eppi0.radiative_correction import support_status_codes
from analysis.exclurad_phi_radiative_correction import (
    audit_samples,
    inelasticity_v,
    read_lund,
    ratio_and_error,
    run_index,
    sigma_nb,
    standard_error,
    trento_phi,
)


PROTON_MASS_GEV = 0.93827208816


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build C_rad(Q2,xB,-t,phi) from EXCLURAD LUND/config pairs with "
            "an offline radiative-v cut"
        )
    )
    parser.add_argument("production_dir", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--v-max", type=float, default=0.2)
    parser.add_argument("--min-counts", type=float, default=5.0)
    parser.add_argument("--expected-runs", type=int, default=200)
    return parser.parse_args()


def dis_kinematics(electron: np.ndarray, beam_energy: float) -> tuple[np.ndarray, np.ndarray]:
    q_energy = beam_energy - electron[:, 3]
    qx = -electron[:, 0]
    qy = -electron[:, 1]
    qz = beam_energy - electron[:, 2]
    q2 = qx**2 + qy**2 + qz**2 - q_energy**2
    xb = np.divide(
        q2,
        2.0 * PROTON_MASS_GEV * q_energy,
        out=np.full_like(q2, np.nan),
        where=q_energy != 0.0,
    )
    return q2, xb


def minus_t(proton: np.ndarray) -> np.ndarray:
    delta_energy = PROTON_MASS_GEV - proton[:, 3]
    return -(delta_energy**2 - np.sum(proton[:, :3] ** 2, axis=1))


def weighted_effective_count(sum_weights: np.ndarray, sum_weight_squares: np.ndarray) -> np.ndarray:
    return np.divide(
        sum_weights**2,
        sum_weight_squares,
        out=np.zeros_like(sum_weights),
        where=sum_weight_squares > 0.0,
    )


def mode_estimates_4d(
    production_dir: Path,
    mode: str,
    binning: AnalysisBinning,
    phase_space: AnalysisPhaseSpace,
    v_max: float,
    expected_runs: int,
) -> dict[str, Any]:
    lund_dir = production_dir / mode / "lund_osg"
    provenance_dir = production_dir / mode / "prov"
    files = sorted(lund_dir.glob("*.lund"), key=run_index)
    if expected_runs and len(files) != expected_runs:
        raise ValueError(f"{mode}: expected {expected_runs} LUND files, found {len(files)}")
    if not files:
        raise ValueError(f"{mode}: no LUND files found under {lund_dir}")

    estimates: list[np.ndarray] = []
    counts_total = np.zeros(binning.size, dtype=float)
    sum_weights = np.zeros(binning.size, dtype=float)
    sum_weight_squares = np.zeros(binning.size, dtype=float)
    configs: list[dict[str, Any]] = []
    run_indices: list[int] = []
    sigma_values: list[float] = []
    generated_total = 0
    selected_total = 0
    in_range_total = 0

    for file_number, lund_path in enumerate(files, start=1):
        index = run_index(lund_path)
        config_path = provenance_dir / f"{mode}_{index:03d}.config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing provenance for {lund_path}: {config_path}")
        config = json.loads(config_path.read_text())
        card = config["card"]
        requested_events = int(card["nev"])
        beam_energy = float(card["ebeam"])
        electrons, protons, observed_events = read_lund(lund_path)
        if observed_events != requested_events:
            raise ValueError(
                f"{lund_path}: observed {observed_events} events, card requests {requested_events}"
            )

        q2, xb = dis_kinematics(electrons, beam_energy)
        mt = minus_t(protons)
        phi = trento_phi(electrons, protons, beam_energy)
        flat = binning.coordinates_to_flat(q2, xb, mt, phi)
        selected = np.isfinite(q2) & np.isfinite(xb) & np.isfinite(mt) & np.isfinite(phi)
        if mode == "rad":
            v = inelasticity_v(electrons, protons, beam_energy)
            selected &= np.isfinite(v) & (v < v_max)
        selected_total += int(np.count_nonzero(selected))
        selected &= phase_space.mask(q2, xb, beam_energy)
        inside = selected & (flat >= 0) & (flat < binning.size)
        counts = np.bincount(flat[inside], minlength=binning.size).astype(float)
        cross_section = sigma_nb(config)
        event_weight = cross_section / requested_events
        estimates.append(event_weight * counts)
        counts_total += counts
        sum_weights += event_weight * counts
        sum_weight_squares += event_weight**2 * counts
        in_range_total += int(np.count_nonzero(inside))
        generated_total += observed_events
        configs.append(config)
        run_indices.append(index)
        sigma_values.append(cross_section)
        if file_number % 25 == 0 or file_number == len(files):
            print(
                f"{mode}: {file_number}/{len(files)} files, generated={generated_total}, "
                f"v-selected={selected_total}, analysis-bin={in_range_total}",
                flush=True,
            )

    matrix = np.asarray(estimates)
    return {
        "matrix": matrix,
        "mean": np.mean(matrix, axis=0),
        "sem": standard_error(matrix),
        "counts": counts_total,
        "effective": weighted_effective_count(sum_weights, sum_weight_squares),
        "configs": configs,
        "run_indices": run_indices,
        "sigma_values": np.asarray(sigma_values),
        "generated": generated_total,
        "selected": selected_total,
        "in_range": in_range_total,
    }


def json_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def main() -> None:
    args = arguments()
    if args.min_counts < 0.0:
        raise ValueError("--min-counts must be nonnegative")
    if not math.isfinite(args.v_max):
        raise ValueError("--v-max must be finite")

    config = json.loads(args.config.read_text())
    binning = from_config(args.config)
    phase_space = AnalysisPhaseSpace.from_config(config)
    born = mode_estimates_4d(
        args.production_dir,
        "born",
        binning,
        phase_space,
        args.v_max,
        args.expected_runs,
    )
    rad = mode_estimates_4d(
        args.production_dir,
        "rad",
        binning,
        phase_space,
        args.v_max,
        args.expected_runs,
    )
    audit = audit_samples(born, rad)

    correction_raw, correction_sem_raw = ratio_and_error(
        rad["mean"], rad["sem"], born["mean"], born["sem"]
    )
    support_overlap = (born["counts"] > 0.0) & (rad["counts"] > 0.0)
    reliable = (
        support_overlap
        & (born["effective"] >= args.min_counts)
        & (rad["effective"] >= args.min_counts)
        & np.isfinite(correction_raw)
        & np.isfinite(correction_sem_raw)
    )
    correction = np.where(reliable, correction_raw, 1.0)
    correction_sem = np.where(reliable, correction_sem_raw, 1.0)
    support_status = support_status_codes(
        born["counts"],
        rad["counts"],
        int(math.ceil(args.min_counts)),
        born_effective_counts=born["effective"],
        radiative_effective_counts=rad["effective"],
    )

    shape = binning.shape
    unflatten = binning.unflatten
    born_sigma_mean = float(np.mean(born["sigma_values"]))
    rad_sigma_mean = float(np.mean(rad["sigma_values"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        C_rad=unflatten(correction),
        delta_C=unflatten(correction_sem),
        C_rad_raw=unflatten(correction_raw),
        delta_C_raw=unflatten(correction_sem_raw),
        reliable=unflatten(reliable),
        support_overlap=unflatten(support_overlap),
        support_status=unflatten(support_status),
        H_born=unflatten(born["counts"]),
        H_rad=unflatten(rad["counts"]),
        H_born_effective=unflatten(born["effective"]),
        H_rad_effective=unflatten(rad["effective"]),
        born_cross_section_nb=unflatten(born["mean"]),
        radiative_vcut_cross_section_nb=unflatten(rad["mean"]),
        born_cross_section_sem_nb=unflatten(born["sem"]),
        radiative_vcut_cross_section_sem_nb=unflatten(rad["sem"]),
        normalization_ratio=rad_sigma_mean / born_sigma_mean,
        born_integrated_cross_section=born_sigma_mean,
        radiative_integrated_cross_section=rad_sigma_mean,
        min_counts=args.min_counts,
        beam_energy=audit["beam_energy_GeV"],
        q2_edges=binning.q2_edges,
        xb_edges=binning.xb_edges,
        t_edges=binning.t_edges,
        phi_edges=binning.phi_edges,
        born_files=len(born["configs"]),
        radiative_files=len(rad["configs"]),
        born_events_seen=born["generated"],
        radiative_events_seen=rad["generated"],
        born_in_range=born["in_range"],
        radiative_in_range=rad["in_range"],
        v_max_GeV2=args.v_max,
        git_commit=audit["git_commit"],
        uncertainty_model="independent-run standard error with sigma_nb-weighted Kish support",
        reliability_count_definition="sigma_nb-weighted Kish effective count per 4D bin",
        phi_convention="electron-proton Trento plane",
        phase_space_definition=(
            "4D analysis bin"
            if not phase_space.enabled
            else f"4D analysis bin and {phase_space.description()}"
        ),
        **phase_space.as_npz_fields(),
    )

    good = reliable & np.isfinite(correction_raw)
    summary = {
        **audit,
        "artifact": str(args.output),
        "analysis_config": str(args.config),
        "shape_Q2_xB_t_phi": list(shape),
        "total_bins": int(np.prod(shape)),
        "reliable_bins": int(np.count_nonzero(reliable)),
        "overlap_bins": int(np.count_nonzero(support_overlap)),
        "min_effective_count": args.min_counts,
        "v_definition": "MX2(e'p) - m_pi0^2",
        "v_max_GeV2": args.v_max,
        "radiative_histogram_renormalized_after_v_cut": False,
        "normalization": "per-run sigma_nb * selected_bin_events / generated_events, averaged over runs",
        "phase_space": phase_space.description(),
        "born_generated_events": born["generated"],
        "radiative_generated_events": rad["generated"],
        "born_analysis_bin_events": born["in_range"],
        "radiative_vcut_analysis_bin_events": rad["in_range"],
        "radiative_vcut_retention_before_analysis_binning": rad["selected"] / rad["generated"],
        "born_sigma_nb_mean": born_sigma_mean,
        "radiative_sigma_nb_mean": rad_sigma_mean,
        "normalization_ratio_no_offline_cut": rad_sigma_mean / born_sigma_mean,
        "reliable_C_rad_mean": json_number(np.mean(correction_raw[good])) if np.any(good) else None,
        "reliable_C_rad_median": json_number(np.median(correction_raw[good])) if np.any(good) else None,
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
