#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import load_npz, save_npz


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.binning import AnalysisBinning, from_config
from eppi0.cross_section import (
    Target,
    integrated_luminosity_fb,
    physical_bin_volumes,
    reduced_cross_section,
)
from eppi0.response import build_response
from eppi0.harmonics import fit_grid
from eppi0.unfolding import bootstrap_uncertainty, iterative_bayes, subtract_feed_in


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Efficient EPPI0 response, unfolding, and cross-section pipeline")
    commands = root.add_subparsers(dest="command", required=True)

    response = commands.add_parser("response", help="Build a sparse response from an event-level MC sample")
    response.add_argument("sample", type=Path)
    response.add_argument("--config", type=Path, required=True)
    response.add_argument("--output-dir", type=Path, required=True)
    response.add_argument("--selection-mask", type=Path)

    unfold = commands.add_parser("unfold", help="Unfold a compact selected-data sample")
    unfold.add_argument("data", type=Path)
    unfold.add_argument("response_matrix", type=Path)
    unfold.add_argument("response_meta", type=Path)
    unfold.add_argument("--config", type=Path, required=True)
    unfold.add_argument("--output", type=Path, required=True)
    unfold.add_argument("--selection-mask", type=Path)
    unfold.add_argument("--iterations", type=int, default=25)
    unfold.add_argument("--bootstrap", type=int, default=200)
    unfold.add_argument("--seed", type=int, default=12345)
    unfold.add_argument("--radiative-correction", type=Path,
                        help="Legacy-compatible NPZ containing C_rad, delta_C, and reliable")

    xsec = commands.add_parser("cross-section", help="Normalize unfolded yields")
    xsec.add_argument("unfolding_result", type=Path)
    xsec.add_argument("--config", type=Path, required=True)
    xsec.add_argument("--output", type=Path, required=True)
    xsec.add_argument("--global-normalization", type=float, default=1.0)

    harmonics = commands.add_parser("fit-harmonics", help="Fit A + B cos(phi) + C cos(2 phi)")
    harmonics.add_argument("cross_section", type=Path)
    harmonics.add_argument("--output", type=Path, required=True)
    return root


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def command_response(args: argparse.Namespace) -> None:
    sample = np.load(args.sample, allow_pickle=False)
    binning = from_config(args.config)
    truth_flat = binning.coordinates_to_flat(
        sample["gen_Q2"], sample["gen_xB"], sample["gen_minus_t"], sample["gen_trento_phi"]
    )
    rec_flat = binning.coordinates_to_flat(
        sample["rec_Q2"], sample["rec_xB"], sample["rec_minus_t"], sample["rec_trento_phi"]
    )
    selected = np.asarray(sample["rec_selected"], dtype=bool)
    if args.selection_mask:
        selected &= _load_mask(args.selection_mask, selected.size)
    weights = sample["gen_weight"] if "gen_weight" in sample.files else None
    response = build_response(truth_flat, rec_flat, selected, binning.size, weights=weights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "response_matrix.npz"
    metadata_path = args.output_dir / "response_meta.npz"
    save_npz(matrix_path, response.core)
    np.savez_compressed(
        metadata_path,
        truth_total=response.truth_total,
        reconstructed_total=response.reconstructed_total,
        efficiency=response.efficiency,
        feed_in_fraction=response.feed_in_fraction,
        feed_in_shape=response.feed_in_shape,
        response_variance_sum=response.response_variance_sum,
        q2_edges=binning.q2_edges,
        xb_edges=binning.xb_edges,
        t_edges=binning.t_edges,
        phi_edges=binning.phi_edges,
    )
    print(f"Truth events in range: {response.truth_total.sum():.0f}")
    print(f"Selected REC events in range: {response.reconstructed_total.sum():.0f}")
    print(f"Feed-in fraction: {response.feed_in_fraction:.6f}")
    print(f"Wrote {matrix_path}")
    print(f"Wrote {metadata_path}")


def command_unfold(args: argparse.Namespace) -> None:
    data = np.load(args.data, allow_pickle=False)
    metadata = np.load(args.response_meta, allow_pickle=False)
    response = load_npz(args.response_matrix).tocsr()
    binning = from_config(args.config)
    config = load_config(args.config)
    minimum_acceptance = float(config.get("minimum_acceptance", 0.005))
    rec_flat = binning.coordinates_to_flat(
        data["rec_Q2"], data["rec_xB"], data["rec_minus_t"], data["rec_trento_phi"]
    )
    selected = rec_flat >= 0
    if "rec_selected" in data.files:
        selected &= np.asarray(data["rec_selected"], dtype=bool)
    if args.selection_mask:
        selected &= _load_mask(args.selection_mask, selected.size)
    measured = np.bincount(rec_flat[selected], minlength=binning.size).astype(float)
    efficiency = metadata["efficiency"]
    corrected = subtract_feed_in(
        measured, float(metadata["feed_in_fraction"]), metadata["feed_in_shape"]
    )
    acceptance_corrected = np.divide(
        measured, efficiency, out=np.zeros_like(measured), where=efficiency > minimum_acceptance
    )
    if args.iterations == 0:
        unfolded = acceptance_corrected.copy()
        kl = np.empty(0)
        sigma_stat = np.divide(
            np.sqrt(measured), efficiency, out=np.zeros_like(measured), where=efficiency > minimum_acceptance
        )
    else:
        result = iterative_bayes(
            response,
            corrected,
            efficiency,
            args.iterations,
            prior=acceptance_corrected,
            minimum_acceptance=minimum_acceptance,
        )
        unfolded, kl = result.unfolded, result.kl_divergence
        _, sigma_stat = bootstrap_uncertainty(
            response,
            measured,
            efficiency,
            args.iterations,
            acceptance_corrected,
            minimum_acceptance=minimum_acceptance,
            experiments=args.bootstrap,
            seed=args.seed,
            feed_in_fraction=float(metadata["feed_in_fraction"]),
            feed_in_shape=metadata["feed_in_shape"],
        )
    sensitivity = np.divide(
        unfolded, efficiency, out=np.zeros_like(unfolded), where=efficiency > minimum_acceptance
    )
    sigma_mc = sensitivity * np.sqrt(metadata["response_variance_sum"])
    sigma_total = np.hypot(sigma_stat, sigma_mc)
    corrected_yield = unfolded.copy()
    corrected_uncertainty = sigma_total.copy()
    radiative_reliable = np.ones(binning.size, dtype=bool)
    if args.radiative_correction:
        correction = np.load(args.radiative_correction, allow_pickle=False)
        factor = binning.flatten_values(correction["C_rad"])
        factor_uncertainty = binning.flatten_values(correction["delta_C"])
        radiative_reliable = binning.flatten_values(correction["reliable"]).astype(bool)
        radiative_valid = radiative_reliable & (efficiency > minimum_acceptance)
        corrected_yield = np.where(radiative_valid, unfolded * factor, 0.0)
        radiative_sigma = unfolded * factor_uncertainty
        corrected_uncertainty = np.where(
            radiative_valid, np.hypot(sigma_total, radiative_sigma), 0.0
        )

    means = binning.bin_means(
        rec_flat[selected],
        {
            "Q2": data["rec_Q2"][selected],
            "xB": data["rec_xB"][selected],
            "minus_t": data["rec_minus_t"][selected],
            "phi": np.mod(data["rec_trento_phi"][selected], 2.0 * np.pi) * 180.0 / np.pi,
        },
    )
    beam_charge = float(data["beam_charge_c"]) if "beam_charge_c" in data.files else np.nan
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        measured=measured,
        acceptance_corrected=acceptance_corrected,
        unfolded=unfolded,
        sigma_stat=sigma_stat,
        sigma_mc=sigma_mc,
        sigma_total=sigma_total,
        corrected_yield=corrected_yield,
        corrected_uncertainty=corrected_uncertainty,
        radiative_reliable=radiative_reliable,
        kl_divergence=kl,
        efficiency=efficiency,
        beam_charge_c=beam_charge,
        Q2_mean=means["Q2"],
        xB_mean=means["xB"],
        minus_t_mean=means["minus_t"],
        phi_mean=means["phi"],
        q2_edges=binning.q2_edges,
        xb_edges=binning.xb_edges,
        t_edges=binning.t_edges,
        phi_edges=binning.phi_edges,
        iterations=args.iterations,
        bootstrap=args.bootstrap,
        random_seed=args.seed,
    )
    print(f"Measured in-range events: {measured.sum():.0f}")
    print(f"Wrote {args.output}")


def command_cross_section(args: argparse.Namespace) -> None:
    result = np.load(args.unfolding_result, allow_pickle=False)
    config = load_config(args.config)
    binning = from_config(args.config)
    beam_energy = float(config["beam_energy"])
    target = Target(
        float(config["target_length_cm"]),
        float(config["target_density_g_cm3"]),
        float(config["target_molar_mass_g"]),
    )
    beam_charge = float(result["beam_charge_c"])
    if not np.isfinite(beam_charge) or beam_charge <= 0:
        raise ValueError("unfolding result does not contain a positive beam_charge_c")
    luminosity = integrated_luminosity_fb(beam_charge, target)
    volumes = binning.flatten_values(physical_bin_volumes(binning, beam_energy))
    # Construct center arrays through the binning itself to avoid order mistakes.
    iq2, ixb, it, iphi = np.indices(binning.shape)
    flat_positions = binning.flatten(iq2, ixb, it, iphi)
    q2_fallback = np.empty(binning.size)
    xb_fallback = np.empty(binning.size)
    q2_fallback[flat_positions.ravel()] = ((binning.q2_edges[:-1] + binning.q2_edges[1:]) / 2.0)[iq2].ravel()
    xb_fallback[flat_positions.ravel()] = ((binning.xb_edges[:-1] + binning.xb_edges[1:]) / 2.0)[ixb].ravel()
    q2_means = np.where(np.isfinite(result["Q2_mean"]), result["Q2_mean"], q2_fallback)
    xb_means = np.where(np.isfinite(result["xB_mean"]), result["xB_mean"], xb_fallback)
    valid = result["efficiency"] > float(config.get("minimum_acceptance", 0.005))
    yields = result["corrected_yield"] if "corrected_yield" in result.files else result["unfolded"]
    yield_uncertainty = (
        result["corrected_uncertainty"]
        if "corrected_uncertainty" in result.files
        else result["sigma_total"]
    )
    values, errors = reduced_cross_section(
        yields,
        yield_uncertainty,
        q2_means,
        xb_means,
        volumes,
        luminosity,
        beam_energy,
        branching_ratio=float(config["pi0_to_gg_branching_ratio"]),
        valid=valid,
    )
    values /= args.global_normalization
    errors /= args.global_normalization
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        reduced_cross_section=binning.unflatten(values),
        uncertainty=binning.unflatten(errors),
        flux_q2_mean=q2_means,
        flux_xb_mean=xb_means,
        bin_volume=binning.unflatten(volumes),
        luminosity_fb=luminosity,
        global_normalization=args.global_normalization,
        q2_edges=binning.q2_edges,
        xb_edges=binning.xb_edges,
        t_edges=binning.t_edges,
        phi_edges=binning.phi_edges,
    )
    print(f"Integrated luminosity: {luminosity:.6g} fb^-1")
    print(f"Wrote {args.output}")


def command_harmonics(args: argparse.Namespace) -> None:
    cross_section = np.load(args.cross_section, allow_pickle=False)
    fits = fit_grid(
        cross_section["reduced_cross_section"],
        cross_section["uncertainty"],
        cross_section["phi_edges"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **fits,
        coefficient_names=np.asarray(["A", "B", "C"]),
        q2_edges=cross_section["q2_edges"],
        xb_edges=cross_section["xb_edges"],
        t_edges=cross_section["t_edges"],
    )
    print(f"Successful fits: {np.isfinite(fits['chi2_ndf']).sum()}")
    print(f"Wrote {args.output}")


def _load_mask(path: Path, expected: int) -> np.ndarray:
    mask = np.asarray(np.load(path), dtype=bool)
    if mask.shape != (expected,):
        raise ValueError(f"selection mask has {mask.size} rows; expected {expected}")
    return mask


def main() -> int:
    args = parser().parse_args()
    if args.command == "response":
        command_response(args)
    elif args.command == "unfold":
        command_unfold(args)
    elif args.command == "cross-section":
        command_cross_section(args)
    elif args.command == "fit-harmonics":
        command_harmonics(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
