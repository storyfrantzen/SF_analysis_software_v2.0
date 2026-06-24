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
from eppi0.root_response import build_response_from_root
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

    response_root = commands.add_parser(
        "response-root",
        help="Build a sparse response directly from converter and selected ROOT files",
    )
    response_root.add_argument("converter_root", type=Path)
    response_root.add_argument("selected_root", type=Path)
    response_root.add_argument("--config", type=Path, required=True)
    response_root.add_argument("--output-dir", type=Path, required=True)
    response_root.add_argument("--dictionary", type=Path)
    response_root.add_argument("--tree", default="Events")
    response_root.add_argument("--generated-tree", default="GeneratedEvents")
    response_root.add_argument("--chunk-size", type=int, default=1_000_000)
    response_root.add_argument("--progress-chunks", type=int, default=10)
    response_root.add_argument(
        "--selection-mask",
        type=Path,
        help="Optional boolean mask with one row per selected ROOT candidate",
    )

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

    acceptance = commands.add_parser("acceptance-plots", help="Plot acceptance diagnostics from response metadata")
    acceptance.add_argument("response_meta", type=Path)
    acceptance.add_argument("--output-dir", type=Path, required=True)
    acceptance.add_argument("--minimum-acceptance", type=float, default=0.005)
    acceptance.add_argument(
        "--phi-min-passing-bins",
        type=int,
        default=1,
        help="Minimum number of above-threshold phi bins required to include a 3D bin in the phi PDF",
    )
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


def command_response_root(args: argparse.Namespace) -> None:
    binning = from_config(args.config)
    mask = None
    if args.selection_mask:
        mask = np.asarray(np.load(args.selection_mask), dtype=bool)
    summary = build_response_from_root(
        args.converter_root,
        args.selected_root,
        binning,
        dictionary=args.dictionary,
        tree=args.tree,
        generated_tree=args.generated_tree,
        chunk_size=args.chunk_size,
        selection_mask=mask,
        progress_chunks=args.progress_chunks,
    )
    response = summary.response

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
        generated_rows=summary.generated_rows,
        selected_rows=summary.selected_rows,
        matched_selected_rows=summary.matched_selected_rows,
    )
    print(f"Generated rows scanned: {summary.generated_rows}")
    print(f"Selected rows read: {summary.selected_rows}")
    print(f"Matched selected rows: {summary.matched_selected_rows}")
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


def command_acceptance_plots(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    metadata = np.load(args.response_meta, allow_pickle=False)
    efficiency = np.asarray(metadata["efficiency"], dtype=float)
    truth = np.asarray(metadata["truth_total"], dtype=float)
    q2_edges = np.asarray(metadata["q2_edges"], dtype=float)
    xb_edges = np.asarray(metadata["xb_edges"], dtype=float)
    t_edges = np.asarray(metadata["t_edges"], dtype=float)
    phi_edges = np.asarray(metadata["phi_edges"], dtype=float)
    shape = (
        q2_edges.size - 1,
        xb_edges.size - 1,
        t_edges.size - 1,
        phi_edges.size - 1,
    )
    if efficiency.size != int(np.prod(shape)) or truth.size != efficiency.size:
        raise ValueError("response metadata arrays do not match bin-edge dimensions")

    eff4 = _unflatten_response(efficiency, shape)
    truth4 = _unflatten_response(truth, shape)
    populated = truth4 > 0
    zero = populated & (eff4 == 0)
    low = populated & (eff4 > 0) & (eff4 < args.minimum_acceptance)
    passing = populated & (eff4 >= args.minimum_acceptance)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    populated_eff = eff4[populated]
    positive_eff = populated_eff[populated_eff > 0]
    _plot_acceptance_histograms(
        populated_eff,
        positive_eff,
        args.minimum_acceptance,
        args.output_dir,
    )
    _plot_acceptance_projection(
        "Q2",
        _edge_labels(q2_edges),
        zero.sum(axis=(1, 2, 3)),
        low.sum(axis=(1, 2, 3)),
        passing.sum(axis=(1, 2, 3)),
        args.output_dir / "acceptance_projection_q2.png",
    )
    _plot_acceptance_projection(
        "xB",
        _edge_labels(xb_edges),
        zero.sum(axis=(0, 2, 3)),
        low.sum(axis=(0, 2, 3)),
        passing.sum(axis=(0, 2, 3)),
        args.output_dir / "acceptance_projection_xb.png",
    )
    _plot_acceptance_projection(
        "-t",
        _edge_labels(t_edges),
        zero.sum(axis=(0, 1, 3)),
        low.sum(axis=(0, 1, 3)),
        passing.sum(axis=(0, 1, 3)),
        args.output_dir / "acceptance_projection_t.png",
    )
    _plot_pass_fraction_map(
        "Q2",
        "xB",
        _edge_labels(q2_edges),
        _edge_labels(xb_edges),
        _pass_fraction(populated, passing, axes=(2, 3)),
        args.output_dir / "acceptance_pass_fraction_q2_xb.png",
    )
    _plot_pass_fraction_map(
        "Q2",
        "-t",
        _edge_labels(q2_edges),
        _edge_labels(t_edges),
        _pass_fraction(populated, passing, axes=(1, 3)),
        args.output_dir / "acceptance_pass_fraction_q2_t.png",
    )
    _plot_pass_fraction_map(
        "xB",
        "-t",
        _edge_labels(xb_edges),
        _edge_labels(t_edges),
        _pass_fraction(populated, passing, axes=(0, 3)),
        args.output_dir / "acceptance_pass_fraction_xb_t.png",
    )
    phi_pages = _plot_acceptance_vs_phi(
        eff4,
        truth4,
        q2_edges,
        xb_edges,
        t_edges,
        phi_edges,
        args.minimum_acceptance,
        args.phi_min_passing_bins,
        args.output_dir / "acceptance_vs_phi_by_3d_bin.pdf",
        args.output_dir / "acceptance_vs_phi_by_3d_bin.csv",
    )

    print(f"Truth-populated bins: {int(populated.sum())}")
    print(f"Zero-acceptance bins: {int(zero.sum())}")
    print(f"Positive sub-threshold bins: {int(low.sum())}")
    print(f"Passing bins: {int(passing.sum())}")
    print(f"3D phi pages: {phi_pages}")
    print(f"Wrote acceptance plots under {args.output_dir}")


def _load_mask(path: Path, expected: int) -> np.ndarray:
    mask = np.asarray(np.load(path), dtype=bool)
    if mask.shape != (expected,):
        raise ValueError(f"selection mask has {mask.size} rows; expected {expected}")
    return mask


def _unflatten_response(values: np.ndarray, shape: tuple[int, int, int, int]) -> np.ndarray:
    nq2, nxb, nt, nphi = shape
    return values.reshape(nxb, nq2, nphi, nt).transpose(1, 0, 3, 2)


def _edge_labels(edges: np.ndarray) -> list[str]:
    return [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]


def _pass_fraction(populated: np.ndarray, passing: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    total = populated.sum(axis=axes)
    good = passing.sum(axis=axes)
    return np.divide(good, total, out=np.full(total.shape, np.nan), where=total > 0)


def _plot_acceptance_histograms(
    populated_eff: np.ndarray,
    positive_eff: np.ndarray,
    minimum_acceptance: float,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(populated_eff, bins=100, histtype="step", linewidth=1.5)
    ax.axvline(minimum_acceptance, color="red", linestyle="--",
               label=f"minimum_acceptance = {minimum_acceptance:g}")
    ax.set_xlabel("Acceptance / efficiency")
    ax.set_ylabel("Number of 4D bins")
    ax.set_title("Acceptance over truth-populated 4D bins")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "acceptance_hist.png", dpi=200)
    plt.close(fig)

    if positive_eff.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(positive_eff.min()), np.log10(positive_eff.max()), 100)
    ax.hist(positive_eff, bins=bins, histtype="step", linewidth=1.5)
    ax.axvline(minimum_acceptance, color="red", linestyle="--",
               label=f"minimum_acceptance = {minimum_acceptance:g}")
    ax.set_xscale("log")
    ax.set_xlabel("Acceptance / efficiency")
    ax.set_ylabel("Number of 4D bins")
    ax.set_title("Acceptance over positive-acceptance 4D bins")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "acceptance_hist_log.png", dpi=200)
    plt.close(fig)


def _plot_acceptance_projection(
    axis_name: str,
    labels: list[str],
    zero: np.ndarray,
    low: np.ndarray,
    passing: np.ndarray,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels)), 5))
    ax.bar(x, zero, label="zero", color="#9aa0a6")
    ax.bar(x, low, bottom=zero, label="positive < threshold", color="#d95f02")
    ax.bar(x, passing, bottom=zero + low, label=">= threshold", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Number of truth-populated 4D bins")
    ax.set_title(f"Acceptance categories projected onto {axis_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_pass_fraction_map(
    y_name: str,
    x_name: str,
    y_labels: list[str],
    x_labels: list[str],
    values: np.ndarray,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(x_labels)), max(5, 0.45 * len(y_labels))))
    image = ax.imshow(values, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title("Fraction of truth-populated 4D bins above acceptance threshold")
    fig.colorbar(image, ax=ax, label="Passing fraction")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_acceptance_vs_phi(
    efficiency: np.ndarray,
    truth: np.ndarray,
    q2_edges: np.ndarray,
    xb_edges: np.ndarray,
    t_edges: np.ndarray,
    phi_edges: np.ndarray,
    minimum_acceptance: float,
    min_passing_bins: int,
    pdf_path: Path,
    csv_path: Path,
) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    pages = 0
    csv_lines = [
        "iq2,q2_low,q2_high,ixb,xb_low,xb_high,it,t_low,t_high,"
        "truth_phi_bins,passing_phi_bins,zero_phi_bins,low_phi_bins,"
        "truth_sum,rec_sum,mean_positive_acceptance,max_acceptance"
    ]

    with PdfPages(pdf_path) as pdf:
        for iq2 in range(efficiency.shape[0]):
            for ixb in range(efficiency.shape[1]):
                for it in range(efficiency.shape[2]):
                    eff_phi = efficiency[iq2, ixb, it, :]
                    truth_phi = truth[iq2, ixb, it, :]
                    populated = truth_phi > 0
                    passing = populated & (eff_phi >= minimum_acceptance)
                    if np.count_nonzero(passing) < min_passing_bins:
                        continue

                    zero = populated & (eff_phi == 0)
                    low = populated & (eff_phi > 0) & (eff_phi < minimum_acceptance)
                    positive = populated & (eff_phi > 0)
                    rec_phi = eff_phi * truth_phi

                    csv_lines.append(
                        ",".join(
                            str(item)
                            for item in (
                                iq2,
                                q2_edges[iq2],
                                q2_edges[iq2 + 1],
                                ixb,
                                xb_edges[ixb],
                                xb_edges[ixb + 1],
                                it,
                                t_edges[it],
                                t_edges[it + 1],
                                int(populated.sum()),
                                int(passing.sum()),
                                int(zero.sum()),
                                int(low.sum()),
                                float(truth_phi[populated].sum()),
                                float(rec_phi[populated].sum()),
                                float(np.nanmean(eff_phi[positive])) if np.any(positive) else np.nan,
                                float(np.nanmax(eff_phi[populated])) if np.any(populated) else np.nan,
                            )
                        )
                    )

                    fig, ax = plt.subplots(figsize=(8, 5))
                    if np.any(zero):
                        ax.scatter(phi_centers[zero], eff_phi[zero], color="#9aa0a6",
                                   label="zero", zorder=3)
                    if np.any(low):
                        ax.scatter(phi_centers[low], eff_phi[low], color="#d95f02",
                                   label="positive < threshold", zorder=4)
                    ax.scatter(phi_centers[passing], eff_phi[passing], color="#1b9e77",
                               label=">= threshold", zorder=5)
                    ax.plot(phi_centers[populated], eff_phi[populated],
                            color="#4c78a8", linewidth=1.0, alpha=0.7)
                    ax.axhline(
                        minimum_acceptance,
                        color="red",
                        linestyle="--",
                        linewidth=1.2,
                        label=f"threshold = {minimum_acceptance:g}",
                    )
                    ax.set_xlim(float(phi_edges[0]), float(phi_edges[-1]))
                    ax.set_ylim(bottom=0.0)
                    ax.set_xlabel("phi bin center [deg]")
                    ax.set_ylabel("Acceptance / efficiency")
                    ax.set_title(
                        "Acceptance vs phi\n"
                        f"Q2 {q2_edges[iq2]:g}-{q2_edges[iq2 + 1]:g}, "
                        f"xB {xb_edges[ixb]:g}-{xb_edges[ixb + 1]:g}, "
                        f"-t {t_edges[it]:g}-{t_edges[it + 1]:g}"
                    )
                    ax.grid(True, alpha=0.25)
                    ax.legend(loc="best", fontsize="small")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                    pages += 1

    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return pages


def main() -> int:
    args = parser().parse_args()
    if args.command == "response":
        command_response(args)
    elif args.command == "response-root":
        command_response_root(args)
    elif args.command == "unfold":
        command_unfold(args)
    elif args.command == "cross-section":
        command_cross_section(args)
    elif args.command == "fit-harmonics":
        command_harmonics(args)
    elif args.command == "acceptance-plots":
        command_acceptance_plots(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
