#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.sparse import load_npz, save_npz


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.binning import from_config
from eppi0.bin_centering import AaoExecutableEvaluator, compute_bin_centering
from eppi0.cross_section import (
    Target,
    integrated_luminosity_fb,
    physical_bin_volumes,
    reduced_cross_section,
)
from eppi0.response import build_response
from eppi0.radiative_correction import compute_radiative_correction
from eppi0.root_response import build_response_from_root
from eppi0.harmonics import fit_grid
from eppi0.phase_space import AnalysisPhaseSpace
from eppi0.unfolding import bootstrap_uncertainty, iterative_bayes, subtract_feed_in


C_RAD_DIAGNOSTIC_PLOT_RANGE = (0.0, 2.0)


@dataclass(frozen=True)
class GeneratorNormalizationRecord:
    path: Path
    sig_sum: float
    events: float | None = None
    ntries: float | None = None
    sig_int: float | None = None
    nevent: float | None = None
    mcall_max: float | None = None
    sigr_max: float | None = None
    generator: str = ""
    units: str = ""


@dataclass(frozen=True)
class GeneratorNormalizationSummary:
    integrated_cross_section: float | None
    records: tuple[GeneratorNormalizationRecord, ...] = ()
    method: str = "none"
    source: str = "none"


def _npz_string(data, key: str, default: str) -> str:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    return str(value)


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

    radcorr = commands.add_parser(
        "radiative-correction",
        help="Compute a native C_rad artifact from Born and radiative LUND samples",
    )
    radcorr.add_argument("born", type=Path, help="Born LUND glob or directory")
    radcorr.add_argument("radiative", type=Path, help="Radiative LUND glob or directory")
    radcorr.add_argument("--config", type=Path, required=True)
    radcorr.add_argument("--output", type=Path, required=True)
    radcorr.add_argument("--chunk-size", type=int, default=200_000)
    radcorr.add_argument("--max-events", type=int)
    radcorr.add_argument(
        "--max-files",
        type=int,
        help="Use at most this many LUND files from each input for quick smoke tests",
    )
    radcorr.add_argument("--min-counts", type=int, default=5)
    radcorr.add_argument(
        "--progress-chunks",
        type=int,
        default=1,
        help="Print one progress line every N LUND chunks; use 0 to disable",
    )
    radcorr.add_argument(
        "--diagnostic-pdf",
        type=Path,
        help="Optional multi-page PDF with C_rad vs phi and support diagnostics",
    )
    radcorr.add_argument(
        "--diagnostic-csv",
        type=Path,
        help="Optional CSV summary for the diagnostic PDF phi pages",
    )
    radcorr.add_argument(
        "--normalization-ratio",
        type=float,
        help="Override all automatic normalization with this global factor",
    )
    radcorr.add_argument(
        "--born-integrated-cross-section",
        type=float,
        help="Born generator integrated cross section, usually sig_sum from aao_norad.sum/.norm",
    )
    radcorr.add_argument(
        "--radiative-integrated-cross-section",
        type=float,
        help="Radiative generator integrated cross section, usually sig_sum from aao_rad.sum/.norm",
    )
    radcorr.add_argument(
        "--born-normalization-file",
        type=Path,
        help="Born generator .norm or .sum file containing sig_sum",
    )
    radcorr.add_argument(
        "--radiative-normalization-file",
        type=Path,
        help="Radiative generator .norm or .sum file containing sig_sum",
    )
    radcorr.add_argument(
        "--max-normalization-files",
        type=int,
        help="Use at most this many .norm/.sum sidecars from each normalization directory",
    )

    radcorr_plots = commands.add_parser(
        "radiative-correction-plots",
        help="Create a diagnostic PDF from a radiative-correction NPZ",
    )
    radcorr_plots.add_argument("correction", type=Path)
    radcorr_plots.add_argument("--output", type=Path, required=True)
    radcorr_plots.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV summary for the per-(Q2,xB,-t) phi pages",
    )

    xsec = commands.add_parser("cross-section", help="Normalize unfolded yields")
    xsec.add_argument("unfolding_result", type=Path)
    xsec.add_argument("--config", type=Path, required=True)
    xsec.add_argument("--output", type=Path, required=True)
    xsec.add_argument("--global-normalization", type=float, default=1.0)
    xsec.add_argument(
        "--bin-centering",
        type=Path,
        help="Optional NPZ from bin-centering; output cross sections are divided by C_BC",
    )

    bin_centering = commands.add_parser(
        "bin-centering",
        help="Compute AAO model bin-centering corrections over the analysis bins",
    )
    bin_centering.add_argument("--config", type=Path, required=True)
    bin_centering.add_argument("--output", type=Path, required=True)
    bin_centering.add_argument("--exe", type=Path, required=True, help="Path to the aao_xsec executable")
    bin_centering.add_argument("--N", type=int, default=4, help="Midpoint samples per dimension")
    bin_centering.add_argument("--workers", type=int, default=None)
    bin_centering.add_argument("--chunk-size", type=int, default=64)
    bin_centering.add_argument(
        "--progress-chunks",
        type=int,
        default=10,
        help="Print one progress line every N assigned 3D bins; use 0 to disable",
    )
    bin_centering.add_argument(
        "--bin-start",
        type=int,
        default=0,
        help="First flattened 3D (Q2,xB,-t) bin to compute, inclusive",
    )
    bin_centering.add_argument(
        "--bin-stop",
        type=int,
        help="Flattened 3D (Q2,xB,-t) bin at which to stop, exclusive",
    )
    bin_centering.add_argument(
        "--bin-chunks",
        type=int,
        help="Split flattened 3D bins into this many chunks for array jobs",
    )
    bin_centering.add_argument(
        "--bin-chunk-index",
        type=int,
        help="Zero-based chunk index to compute when --bin-chunks is set",
    )
    bin_centering.add_argument("--theory", type=int, default=5)
    bin_centering.add_argument("--channel", type=int, default=1)
    bin_centering.add_argument("--resonance", type=int, default=0)
    bin_centering.add_argument(
        "--max-failure-fraction",
        type=float,
        default=0.0,
        help="Maximum failed/non-positive AAO fraction allowed before a bin is marked unreliable",
    )
    bin_centering.add_argument("--verbose-failures", action="store_true")

    bin_centering_merge = commands.add_parser(
        "bin-centering-merge",
        help="Merge partial bin-centering NPZ artifacts from array jobs",
    )
    bin_centering_merge.add_argument("partials", nargs="+", type=Path)
    bin_centering_merge.add_argument("--output", type=Path, required=True)

    harmonics = commands.add_parser("fit-harmonics", help="Fit A + B cos(phi) + C cos(2 phi)")
    harmonics.add_argument("cross_section", type=Path)
    harmonics.add_argument("--output", type=Path, required=True)

    harmonic_plots = commands.add_parser("harmonic-plots", help="Plot harmonic-fit diagnostics")
    harmonic_plots.add_argument("harmonics", type=Path)
    harmonic_plots.add_argument("--output-dir", type=Path, required=True)
    harmonic_plots.add_argument("--min-points", type=int, default=4)
    harmonic_plots.add_argument(
        "--quilt",
        action="store_true",
        help="Append a stitched Q2-by-xB quilt of A, B, and C vs -t to the coefficient PDF",
    )
    harmonic_plots.add_argument(
        "--quilt-scale-percentile",
        type=float,
        default=98.0,
        help="Robust absolute-value percentile used for the stitched quilt y scale",
    )
    harmonic_plots.add_argument(
        "--quilt-scale-mode",
        choices=("global", "panel"),
        default="global",
        help=(
            "Use one percentile-based y scale for the quilt or independently scale "
            "each panel like its corresponding coefficient page (default: global)"
        ),
    )

    xsec_plots = commands.add_parser("cross-section-plots", help="Plot reduced cross section vs phi with harmonic fits")
    xsec_plots.add_argument("cross_section", type=Path)
    xsec_plots.add_argument("harmonics", type=Path)
    xsec_plots.add_argument("--output-dir", type=Path, required=True)
    xsec_plots.add_argument("--min-points", type=int, default=4)
    xsec_plots.add_argument(
        "--quilt",
        action="store_true",
        help="Prepend one stitched Q2-by-xB reduced-cross-section quilt per -t bin",
    )
    xsec_plots.add_argument(
        "--quilt-scale-mode",
        choices=("global", "panel"),
        default="panel",
        help="Use one y scale per -t quilt or independently scale every panel (default: panel)",
    )

    acceptance = commands.add_parser("acceptance-plots", help="Plot acceptance diagnostics from response metadata")
    acceptance.add_argument("response_meta", type=Path)
    acceptance.add_argument("--output-dir", type=Path, required=True)
    acceptance.add_argument(
        "--response-matrix",
        type=Path,
        help="Optional response_matrix.npz used to add migration diagnostics from the response diagonal",
    )
    acceptance.add_argument(
        "--include-purity",
        action="store_true",
        help="Include P_i purity in the overlaid acceptance histograms and phi PDF",
    )
    acceptance.add_argument("--minimum-acceptance", type=float, default=0.005)
    acceptance.add_argument(
        "--phi-min-passing-bins",
        type=int,
        default=1,
        help="Minimum number of above-threshold phi bins required to include a 3D bin in the phi PDF",
    )
    response_plots = commands.add_parser(
        "response-plots",
        help="Visualize the sparse IBU response matrix and migration by kinematic variable",
    )
    response_plots.add_argument("response_matrix", type=Path)
    response_plots.add_argument("response_meta", type=Path)
    response_plots.add_argument("--output", type=Path, required=True)
    response_plots.add_argument(
        "--max-points",
        type=int,
        default=500_000,
        help="Maximum nonzero entries to draw in the global sparse response image",
    )
    response_plots.add_argument("--seed", type=int, default=12345)
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
        radiative_valid = (
            radiative_reliable
            & (efficiency > minimum_acceptance)
            & np.isfinite(factor)
            & (factor > 0.0)
        )
        corrected_yield = np.divide(
            unfolded,
            factor,
            out=np.zeros_like(unfolded),
            where=radiative_valid,
        )
        radiative_sigma = np.divide(
            unfolded * factor_uncertainty,
            factor * factor,
            out=np.zeros_like(unfolded),
            where=radiative_valid,
        )
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


def command_radiative_correction(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    binning = from_config(args.config)
    phase_space = AnalysisPhaseSpace.from_config(config)
    born_normalization = _resolve_normalization_summary(
        args.born_integrated_cross_section,
        args.born_normalization_file,
        "born",
        max_files=args.max_normalization_files,
    )
    radiative_normalization = _resolve_normalization_summary(
        args.radiative_integrated_cross_section,
        args.radiative_normalization_file,
        "radiative",
        max_files=args.max_normalization_files,
    )
    result = compute_radiative_correction(
        args.born,
        args.radiative,
        binning,
        beam_energy=float(config["beam_energy"]),
        chunk_size=args.chunk_size,
        max_events=args.max_events,
        max_files=args.max_files,
        min_counts=args.min_counts,
        normalization_ratio=args.normalization_ratio,
        born_integrated_cross_section=born_normalization.integrated_cross_section,
        radiative_integrated_cross_section=radiative_normalization.integrated_cross_section,
        progress_chunks=args.progress_chunks,
        phase_space=phase_space,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        C_rad=result.c_rad,
        delta_C=result.delta_c,
        reliable=result.reliable,
        support_overlap=result.support_overlap,
        support_status=result.support_status,
        H_born=binning.unflatten(result.born.counts),
        H_rad=binning.unflatten(result.radiative.counts),
        born_q2_min=result.born.q2_min,
        born_q2_max=result.born.q2_max,
        born_eprime_min=result.born.eprime_min,
        born_eprime_max=result.born.eprime_max,
        radiative_q2_min=result.radiative.q2_min,
        radiative_q2_max=result.radiative.q2_max,
        radiative_eprime_min=result.radiative.eprime_min,
        radiative_eprime_max=result.radiative.eprime_max,
        normalization_ratio=result.normalization_ratio,
        born_integrated_cross_section=(
            np.nan if result.born_integrated_cross_section is None
            else result.born_integrated_cross_section
        ),
        radiative_integrated_cross_section=(
            np.nan if result.radiative_integrated_cross_section is None
            else result.radiative_integrated_cross_section
        ),
        min_counts=args.min_counts,
        max_events=-1 if args.max_events is None else args.max_events,
        max_files=-1 if args.max_files is None else args.max_files,
        max_normalization_files=(
            -1 if args.max_normalization_files is None else args.max_normalization_files
        ),
        beam_energy=float(config["beam_energy"]),
        q2_edges=binning.q2_edges,
        xb_edges=binning.xb_edges,
        t_edges=binning.t_edges,
        phi_edges=binning.phi_edges,
        born_files=result.born.files,
        radiative_files=result.radiative.files,
        born_events_seen=result.born.events_seen,
        radiative_events_seen=result.radiative.events_seen,
        born_topology_events=result.born.topology_events,
        radiative_topology_events=result.radiative.topology_events,
        born_in_range=result.born.in_range,
        radiative_in_range=result.radiative.in_range,
        born_generated_q2_range=np.asarray([result.born.generated_q2_min, result.born.generated_q2_max]),
        radiative_generated_q2_range=np.asarray([
            result.radiative.generated_q2_min, result.radiative.generated_q2_max,
        ]),
        born_generated_eprime_range=np.asarray([
            result.born.generated_eprime_min, result.born.generated_eprime_max,
        ]),
        radiative_generated_eprime_range=np.asarray([
            result.radiative.generated_eprime_min, result.radiative.generated_eprime_max,
        ]),
        phi_convention="electron-proton trento plane",
        phase_space_definition=(
            "4D bin"
            if not phase_space.enabled
            else f"4D bin and {phase_space.description()}"
        ),
        **phase_space.as_npz_fields(),
        **_normalization_npz_fields("born", born_normalization),
        **_normalization_npz_fields("radiative", radiative_normalization),
    )
    if args.diagnostic_pdf:
        pages = _plot_radiative_correction_diagnostics(
            args.output,
            args.diagnostic_pdf,
            csv_path=args.diagnostic_csv,
        )
        print(f"Wrote radiative-correction diagnostic PDF with {pages} pages: {args.diagnostic_pdf}")
    reliable_bins = int(np.count_nonzero(result.reliable))
    total_bins = int(result.reliable.size)
    support_counts = np.bincount(result.support_status.ravel(), minlength=7)
    print(
        "Born events: "
        f"seen={result.born.events_seen}, topology={result.born.topology_events}, "
        f"in-range={result.born.in_range}"
    )
    print(
        "Radiative events: "
        f"seen={result.radiative.events_seen}, topology={result.radiative.topology_events}, "
        f"in-range={result.radiative.in_range}"
    )
    print(
        "Born generated ranges: "
        f"Q2={result.born.generated_q2_min:.6g}-{result.born.generated_q2_max:.6g}, "
        f"Eprime={result.born.generated_eprime_min:.6g}-{result.born.generated_eprime_max:.6g}"
    )
    print(
        "Radiative generated ranges: "
        f"Q2={result.radiative.generated_q2_min:.6g}-{result.radiative.generated_q2_max:.6g}, "
        f"Eprime={result.radiative.generated_eprime_min:.6g}-{result.radiative.generated_eprime_max:.6g}"
    )
    print(f"Reliable bins: {reliable_bins}/{total_bins} with min_counts={args.min_counts}")
    print(
        "Support bins: "
        f"overlap={int(np.count_nonzero(result.support_overlap))}, "
        f"both_empty={int(support_counts[1])}, born_only={int(support_counts[2])}, "
        f"radiative_only={int(support_counts[3])}, "
        f"low_born={int(support_counts[4])}, low_radiative={int(support_counts[5])}, "
        f"low_both={int(support_counts[6])}"
    )
    print(f"Normalization ratio: {result.normalization_ratio:.8g}")
    if phase_space.enabled:
        print(f"Analysis phase space: 4D bin and {phase_space.description()}")
    if result.born_integrated_cross_section is not None:
        print(
            "Integrated cross sections: "
            f"born={result.born_integrated_cross_section:.8g}, "
            f"radiative={result.radiative_integrated_cross_section:.8g}"
        )
    print(f"Wrote {args.output}")


def _resolve_integrated_cross_section(
    value: float | None,
    path: Path | None,
    label: str,
    *,
    max_files: int | None = None,
) -> float | None:
    if value is not None and path is not None:
        raise ValueError(
            f"Use either --{label}-integrated-cross-section or "
            f"--{label}-normalization-file, not both"
        )
    if path is None:
        return value
    return _read_generator_integrated_cross_section(path, max_files=max_files)


def _resolve_normalization_summary(
    value: float | None,
    path: Path | None,
    label: str,
    *,
    max_files: int | None = None,
) -> GeneratorNormalizationSummary:
    if value is not None and path is not None:
        raise ValueError(
            f"Use either --{label}-integrated-cross-section or "
            f"--{label}-normalization-file, not both"
        )
    if path is not None:
        return _read_generator_normalization_summary(path, max_files=max_files)
    if value is None:
        return GeneratorNormalizationSummary(None)
    value = _positive_finite_float(str(value), f"{label}_integrated_cross_section", Path("<manual>"))
    return GeneratorNormalizationSummary(
        value,
        method="manual",
        source="manual",
    )


def _read_generator_integrated_cross_section(path: Path, *, max_files: int | None = None) -> float:
    summary = _read_generator_normalization_summary(path, max_files=max_files)
    if summary.integrated_cross_section is None:
        raise ValueError(f"No integrated cross section found in {path}")
    return summary.integrated_cross_section


def _read_generator_normalization_summary(
    path: Path,
    *,
    max_files: int | None = None,
) -> GeneratorNormalizationSummary:
    if max_files is not None and max_files <= 0:
        raise ValueError("--max-normalization-files must be positive when provided")
    if path.is_dir():
        norm_files = _normalization_sidecar_files(path, ".norm", max_files=max_files)
        sum_files = [] if norm_files else _normalization_sidecar_files(path, ".sum", max_files=max_files)
        files = norm_files or sum_files
        if not files:
            raise ValueError(f"No .norm or .sum files found under {path}")
        records = [_read_generator_normalization_record(item) for item in files]
        values = np.array([record.sig_sum for record in records], dtype=float)
        weights = [record.events for record in records]
        have_weight = [weight is not None for weight in weights]
        if all(have_weight):
            return GeneratorNormalizationSummary(
                float(np.average(values, weights=np.array(weights, dtype=float))),
                records=tuple(records),
                method="events_weighted_mean_sig_sum",
                source=".norm_directory" if norm_files else ".sum_directory",
            )
        if any(have_weight):
            raise ValueError(
                f"Mixed weighted and unweighted normalization files under {path}; "
                "use a directory containing only .norm files with events metadata "
                "or only legacy .sum files"
            )
        return GeneratorNormalizationSummary(
            float(np.mean(values)),
            records=tuple(records),
            method="unweighted_mean_sig_sum",
            source=".norm_directory" if norm_files else ".sum_directory",
        )

    record = _read_generator_normalization_record(path)
    return GeneratorNormalizationSummary(
        record.sig_sum,
        records=(record,),
        method="single_file_sig_sum",
        source=path.suffix.lower() or "file",
    )


def _normalization_sidecar_files(path: Path, suffix: str, *, max_files: int | None) -> list[Path]:
    pattern = f"*{suffix}"
    if max_files is None:
        return [
            item for item in sorted(path.rglob(pattern))
            if item.is_file() and item.suffix.lower() == suffix
        ]

    files: list[Path] = []
    for item in path.rglob(pattern):
        if item.is_file() and item.suffix.lower() == suffix:
            files.append(item)
            if len(files) >= max_files:
                break
    return files


def _read_generator_normalization_record(path: Path) -> GeneratorNormalizationRecord:
    text = path.read_text(encoding="utf-8", errors="replace")
    fields = _parse_generator_key_values(text)
    return GeneratorNormalizationRecord(
        path=path,
        sig_sum=_parse_generator_integrated_cross_section(text, path),
        events=_optional_positive_generator_float(fields, "events", path),
        ntries=_optional_positive_generator_float(fields, "ntries", path),
        sig_int=_optional_positive_generator_float(fields, "sig_int", path),
        nevent=_optional_positive_generator_float(fields, "nevent", path),
        mcall_max=_optional_positive_generator_float(fields, "mcall_max", path),
        sigr_max=_optional_positive_generator_float(fields, "sigr_max", path),
        generator=fields.get("generator", ""),
        units=fields.get("integrated_cross_section_units", ""),
    )


def _parse_generator_integrated_cross_section(text: str, path: Path) -> float:
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)"
    key_match = re.search(
        r"(?im)^\s*(?:sig_sum|integrated_cross_section_sig_sum)\s*=\s*"
        + number + r"\s*$",
        text,
    )
    if key_match:
        return _positive_finite_float(key_match.group(1), "sig_sum", path)
    line_match = re.search(
        r"(?im)^\s*Integrated\s+cross\s+section(?:\s+\([^)]*\))?\s*=\s*"
        + number + r"\s+" + number,
        text,
    )
    if line_match:
        return _positive_finite_float(line_match.group(2), "sig_sum", path)
    raise ValueError(f"Could not find sig_sum integrated cross section in {path}")


def _parse_generator_key_values(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in re.finditer(r"(?im)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", text):
        fields[match.group(1).lower()] = match.group(2).strip()
    return fields


def _optional_positive_generator_float(
    fields: dict[str, str],
    key: str,
    path: Path,
) -> float | None:
    value = fields.get(key)
    if value is None:
        return None
    return _positive_finite_float(value, key, path)


def _parse_generator_events(text: str, path: Path) -> float | None:
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)"
    event_match = re.search(r"(?im)^\s*events\s*=\s*" + number + r"\s*$", text)
    if event_match is None:
        return None
    return _positive_finite_float(event_match.group(1), "events", path)


def _positive_finite_float(value: str, label: str, path: Path) -> float:
    parsed = float(value.replace("D", "E").replace("d", "e"))
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} in {path} must be positive and finite")
    return parsed


def _normalization_npz_fields(
    label: str,
    summary: GeneratorNormalizationSummary,
) -> dict[str, np.ndarray]:
    records = summary.records

    def floats(name: str) -> np.ndarray:
        return np.asarray(
            [
                getattr(record, name) if getattr(record, name) is not None else np.nan
                for record in records
            ],
            dtype=float,
        )

    return {
        f"{label}_normalization_source": np.asarray(summary.source),
        f"{label}_normalization_method": np.asarray(summary.method),
        f"{label}_normalization_record_count": np.asarray(len(records), dtype=np.int64),
        f"{label}_normalization_files": np.asarray([str(record.path) for record in records]),
        f"{label}_normalization_generators": np.asarray([record.generator for record in records]),
        f"{label}_normalization_units": np.asarray([record.units for record in records]),
        f"{label}_normalization_sig_sum": floats("sig_sum"),
        f"{label}_normalization_sig_int": floats("sig_int"),
        f"{label}_normalization_events": floats("events"),
        f"{label}_normalization_ntries": floats("ntries"),
        f"{label}_normalization_nevent": floats("nevent"),
        f"{label}_normalization_mcall_max": floats("mcall_max"),
        f"{label}_normalization_sigr_max": floats("sigr_max"),
    }


def command_radiative_correction_plots(args: argparse.Namespace) -> None:
    pages = _plot_radiative_correction_diagnostics(args.correction, args.output, csv_path=args.csv)
    print(f"Wrote radiative-correction diagnostic PDF with {pages} pages: {args.output}")


def _plot_radiative_correction_diagnostics(
    correction_path: Path,
    pdf_path: Path,
    csv_path: Path | None = None,
) -> int:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    correction = np.load(correction_path, allow_pickle=False)
    c_rad = np.asarray(correction["C_rad"], dtype=float)
    delta_c = np.asarray(correction["delta_C"], dtype=float)
    reliable = np.asarray(correction["reliable"], dtype=bool)
    h_born = np.asarray(correction["H_born"], dtype=float)
    h_rad = np.asarray(correction["H_rad"], dtype=float)
    q2_edges = _npz_first(correction, "q2_edges", "Q2_edges")
    xb_edges = _npz_first(correction, "xb_edges", "Xb_edges")
    t_edges = _npz_first(correction, "t_edges")
    phi_edges = _npz_first(correction, "phi_edges")
    min_counts = int(np.asarray(correction["min_counts"]).item()) if "min_counts" in correction.files else 5
    if "support_status" in correction.files:
        support_status = np.asarray(correction["support_status"], dtype=np.uint8)
    else:
        support_status = _support_status_codes_for_plotting(h_born, h_rad, min_counts)
    if "support_overlap" in correction.files:
        support_overlap = np.asarray(correction["support_overlap"], dtype=bool)
    else:
        support_overlap = (h_born > 0.0) & (h_rad > 0.0)

    expected_shape = (
        q2_edges.size - 1,
        xb_edges.size - 1,
        t_edges.size - 1,
        phi_edges.size - 1,
    )
    for name, values in (
        ("C_rad", c_rad),
        ("delta_C", delta_c),
        ("reliable", reliable),
        ("H_born", h_born),
        ("H_rad", h_rad),
        ("support_status", support_status),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}; expected {expected_shape}")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    status_labels = (
        "reliable",
        "both empty",
        "born only",
        "radiative only",
        "low born",
        "low radiative",
        "low both",
    )
    status_colors = {
        0: "#1b9e77",
        1: "#c7c7c7",
        2: "#7570b3",
        3: "#e7298a",
        4: "#d95f02",
        5: "#e6ab02",
        6: "#666666",
    }
    csv_lines = [
        "iq2,q2_low,q2_high,ixb,xb_low,xb_high,it,t_low,t_high,"
        "reliable_phi_bins,overlap_phi_bins,born_sum,radiative_sum,"
        "mean_c_rad,median_c_rad,min_c_rad,max_c_rad,mean_delta_c"
    ]

    pages = 0
    with PdfPages(pdf_path) as pdf:
        _plot_radcorr_summary_page(
            pdf,
            correction_path,
            c_rad,
            delta_c,
            reliable,
            support_overlap,
            support_status,
            h_born,
            h_rad,
            correction,
            min_counts,
        )
        pages += 1

        _plot_radcorr_status_page(pdf, support_status, status_labels, status_colors)
        pages += 1

        _plot_radcorr_histograms(pdf, c_rad, delta_c, reliable)
        pages += 1

        _plot_radcorr_q2_xb_projection(pdf, c_rad, reliable, q2_edges, xb_edges)
        pages += 1

        _plot_radcorr_t_phi_projection(pdf, c_rad, reliable, t_edges, phi_edges)
        pages += 1

        for iq2 in range(c_rad.shape[0]):
            for ixb in range(c_rad.shape[1]):
                for it in range(c_rad.shape[2]):
                    born_phi = h_born[iq2, ixb, it, :]
                    rad_phi = h_rad[iq2, ixb, it, :]
                    if not np.any((born_phi > 0.0) | (rad_phi > 0.0)):
                        continue
                    reliable_phi = reliable[iq2, ixb, it, :]
                    overlap_phi = support_overlap[iq2, ixb, it, :]
                    values = c_rad[iq2, ixb, it, :]
                    errors = delta_c[iq2, ixb, it, :]
                    status_phi = support_status[iq2, ixb, it, :]
                    _plot_radcorr_phi_page(
                        pdf,
                        phi_centers,
                        values,
                        errors,
                        reliable_phi,
                        status_phi,
                        born_phi,
                        rad_phi,
                        q2_edges,
                        xb_edges,
                        t_edges,
                        iq2,
                        ixb,
                        it,
                        status_labels,
                        status_colors,
                    )
                    pages += 1
                    good = reliable_phi & np.isfinite(values)
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
                                int(np.count_nonzero(reliable_phi)),
                                int(np.count_nonzero(overlap_phi)),
                                float(np.sum(born_phi)),
                                float(np.sum(rad_phi)),
                                float(np.nanmean(values[good])) if np.any(good) else np.nan,
                                float(np.nanmedian(values[good])) if np.any(good) else np.nan,
                                float(np.nanmin(values[good])) if np.any(good) else np.nan,
                                float(np.nanmax(values[good])) if np.any(good) else np.nan,
                                float(np.nanmean(errors[good])) if np.any(good) else np.nan,
                            )
                        )
                    )

    if csv_path is not None:
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return pages


def _npz_first(data, *keys: str) -> np.ndarray:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key], dtype=float)
    raise KeyError(f"NPZ is missing one of: {', '.join(keys)}")


def _support_status_codes_for_plotting(
    born_counts: np.ndarray,
    radiative_counts: np.ndarray,
    min_counts: int,
) -> np.ndarray:
    born_nonzero = born_counts > 0.0
    rad_nonzero = radiative_counts > 0.0
    born_low = born_counts < min_counts
    rad_low = radiative_counts < min_counts
    status = np.zeros(born_counts.shape, dtype=np.uint8)
    status[~born_nonzero & ~rad_nonzero] = 1
    status[born_nonzero & ~rad_nonzero] = 2
    status[~born_nonzero & rad_nonzero] = 3
    both = born_nonzero & rad_nonzero
    status[both & born_low & ~rad_low] = 4
    status[both & ~born_low & rad_low] = 5
    status[both & born_low & rad_low] = 6
    return status


def _plot_radcorr_summary_page(
    pdf,
    correction_path: Path,
    c_rad: np.ndarray,
    delta_c: np.ndarray,
    reliable: np.ndarray,
    support_overlap: np.ndarray,
    support_status: np.ndarray,
    h_born: np.ndarray,
    h_rad: np.ndarray,
    correction,
    min_counts: int,
) -> None:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    good = reliable & np.isfinite(c_rad)
    lines = [
        "Radiative-correction diagnostics",
        "",
        f"Source: {correction_path}",
        f"4D shape: {c_rad.shape}",
        f"Total bins: {c_rad.size}",
        f"Reliable phi bins: {np.count_nonzero(reliable)}",
        f"Overlap phi bins: {np.count_nonzero(support_overlap)}",
        f"min_counts: {min_counts}",
        f"Born total in-range count: {np.sum(h_born):.8g}",
        f"Radiative total in-range count: {np.sum(h_rad):.8g}",
        f"Normalization ratio: {_optional_scalar(correction, 'normalization_ratio'):.8g}",
        f"Born integrated cross section: {_optional_scalar(correction, 'born_integrated_cross_section'):.8g}",
        f"Radiative integrated cross section: {_optional_scalar(correction, 'radiative_integrated_cross_section'):.8g}",
        f"Mean C_rad reliable: {np.nanmean(c_rad[good]) if np.any(good) else np.nan:.8g}",
        f"Median C_rad reliable: {np.nanmedian(c_rad[good]) if np.any(good) else np.nan:.8g}",
        f"Mean delta_C reliable: {np.nanmean(delta_c[good]) if np.any(good) else np.nan:.8g}",
        "",
        _range_line(correction, "born_generated_q2_range", "Born generated Q2"),
        _range_line(correction, "radiative_generated_q2_range", "Radiative generated Q2"),
        _range_line(correction, "born_generated_eprime_range", "Born generated Eprime"),
        _range_line(correction, "radiative_generated_eprime_range", "Radiative generated Eprime"),
        "",
        "support_status codes: 0 reliable, 1 both empty, 2 born only,",
        "3 radiative only, 4 low born, 5 low radiative, 6 low both.",
    ]
    counts = np.bincount(support_status.ravel(), minlength=7)
    lines.append("")
    lines.append("Status counts: " + ", ".join(f"{idx}={int(count)}" for idx, count in enumerate(counts[:7])))
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.94, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def _plot_radcorr_status_page(pdf, support_status, status_labels, status_colors) -> None:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    counts = np.bincount(support_status.ravel(), minlength=len(status_labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(status_labels))
    ax.bar(x, counts, color=[status_colors[index] for index in range(len(status_labels))])
    ax.set_xticks(x)
    ax.set_xticklabels(status_labels, rotation=25, ha="right")
    ax.set_ylabel("Phi bins")
    ax.set_title("Radiative-correction support status")
    ax.grid(True, axis="y", alpha=0.25)
    for index, count in enumerate(counts):
        ax.text(index, count, str(int(count)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_radcorr_histograms(pdf, c_rad, delta_c, reliable) -> None:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    good = reliable & np.isfinite(c_rad) & np.isfinite(delta_c)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    if np.any(good):
        axes[0].hist(
            c_rad[good],
            bins=80,
            range=C_RAD_DIAGNOSTIC_PLOT_RANGE,
            histtype="stepfilled",
            color="#4c78a8",
            alpha=0.75,
        )
        axes[1].hist(delta_c[good], bins=80, histtype="stepfilled", color="#f58518", alpha=0.75)
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlim(*C_RAD_DIAGNOSTIC_PLOT_RANGE)
    axes[0].set_xlabel("C_rad")
    axes[0].set_ylabel("Reliable phi bins")
    axes[0].set_title("C_rad distribution")
    axes[1].set_xlabel("delta_C")
    axes[1].set_title("Correction uncertainty distribution")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_radcorr_q2_xb_projection(pdf, c_rad, reliable, q2_edges, xb_edges) -> None:
    median = _nanmedian_where(c_rad, reliable, axis=(2, 3))
    fraction = np.mean(reliable, axis=(2, 3))
    _plot_radcorr_projection_page(
        pdf,
        median,
        fraction,
        x_edges=xb_edges,
        y_edges=q2_edges,
        x_label="xB",
        y_label="Q2 [GeV^2]",
        title="Radiative correction coverage in Q2 and xB",
    )


def _plot_radcorr_t_phi_projection(pdf, c_rad, reliable, t_edges, phi_edges) -> None:
    median = _nanmedian_where(c_rad, reliable, axis=(0, 1))
    fraction = np.mean(reliable, axis=(0, 1))
    _plot_radcorr_projection_page(
        pdf,
        median,
        fraction,
        x_edges=phi_edges,
        y_edges=t_edges,
        x_label="phi [deg]",
        y_label="-t [GeV^2]",
        title="Radiative correction coverage in phi and -t",
    )


def _plot_radcorr_projection_page(
    pdf,
    median_c_rad,
    reliable_fraction,
    *,
    x_edges,
    y_edges,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    c_image = axes[0].pcolormesh(
        x_edges,
        y_edges,
        median_c_rad,
        vmin=C_RAD_DIAGNOSTIC_PLOT_RANGE[0],
        vmax=C_RAD_DIAGNOSTIC_PLOT_RANGE[1],
        cmap="viridis",
        shading="flat",
    )
    fig.colorbar(c_image, ax=axes[0], label="Median reliable C_rad (clipped 0-2)")
    axes[0].set_aspect("auto")
    axes[0].set_title("Median C_rad")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    _annotate_radcorr_projection(
        axes[0],
        median_c_rad,
        x_edges,
        y_edges,
        value_kind="c_rad",
    )

    f_image = axes[1].pcolormesh(
        x_edges,
        y_edges,
        reliable_fraction,
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
        shading="flat",
    )
    fig.colorbar(f_image, ax=axes[1], label="Reliable fraction")
    axes[1].set_aspect("auto")
    axes[1].set_title("Reliable-bin fraction")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    _annotate_radcorr_projection(
        axes[1],
        reliable_fraction,
        x_edges,
        y_edges,
        value_kind="fraction",
    )

    fig.suptitle(title)
    pdf.savefig(fig)
    plt.close(fig)


def _annotate_radcorr_projection(ax, values, x_edges, y_edges, *, value_kind: str) -> None:
    values = np.asarray(values, dtype=float)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    rows, cols = values.shape
    fontsize = 6.5 if rows * cols <= 100 else 4.8
    for iy, ix in np.ndindex(values.shape):
        value = values[iy, ix]
        if not np.isfinite(value) or value <= 0.0:
            continue
        label = _format_projection_value(value, value_kind)
        color = _projection_text_color(value, value_kind)
        ax.text(
            x_centers[ix],
            y_centers[iy],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
        )


def _format_projection_value(value: float, value_kind: str) -> str:
    if value_kind == "fraction":
        return f"{value:.2f}"
    if abs(value) < 10.0:
        return f"{value:.2f}"
    if abs(value) < 100.0:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _projection_text_color(value: float, value_kind: str) -> str:
    if not np.isfinite(value):
        return "#555555"
    if value_kind == "fraction":
        return "white" if value < 0.55 else "black"
    clipped = min(max(value, C_RAD_DIAGNOSTIC_PLOT_RANGE[0]), C_RAD_DIAGNOSTIC_PLOT_RANGE[1])
    midpoint = 0.5 * sum(C_RAD_DIAGNOSTIC_PLOT_RANGE)
    return "white" if clipped < midpoint else "black"


def _nanmedian_where(values, mask, axis):
    import warnings

    masked = np.where(mask & np.isfinite(values), values, np.nan)
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(masked, axis=axis)


def _plot_radcorr_phi_page(
    pdf,
    phi_centers,
    c_rad,
    delta_c,
    reliable,
    status,
    born_counts,
    rad_counts,
    q2_edges,
    xb_edges,
    t_edges,
    iq2,
    ixb,
    it,
    status_labels,
    status_colors,
) -> None:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    fig, (ax, cax) = plt.subplots(
        2,
        1,
        figsize=(8.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    if np.any(reliable):
        ax.errorbar(
            phi_centers[reliable],
            c_rad[reliable],
            yerr=delta_c[reliable],
            fmt="o",
            color=status_colors[0],
            ecolor=status_colors[0],
            elinewidth=0.8,
            capsize=2,
            markersize=4,
            label=status_labels[0],
        )
    for code in range(1, len(status_labels)):
        mask = status == code
        if np.any(mask):
            ax.scatter(
                phi_centers[mask],
                c_rad[mask],
                s=26,
                color=status_colors[code],
                label=status_labels[code],
                zorder=3,
            )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_ylabel("C_rad")
    ax.set_title(
        "Radiative correction vs phi\n"
        f"Q2 {q2_edges[iq2]:g}-{q2_edges[iq2 + 1]:g}, "
        f"xB {xb_edges[ixb]:g}-{xb_edges[ixb + 1]:g}, "
        f"-t {t_edges[it]:g}-{t_edges[it + 1]:g}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="x-small", ncol=2)

    cax.plot(phi_centers, born_counts, "o-", color="#4c78a8", markersize=3, linewidth=1.0, label="born")
    cax.plot(phi_centers, rad_counts, "s-", color="#f58518", markersize=3, linewidth=1.0, label="radiative")
    if np.nanmax(np.r_[born_counts, rad_counts]) > 0:
        cax.set_yscale("symlog", linthresh=1.0)
    cax.set_xlabel("phi bin center [deg]")
    cax.set_ylabel("Generated counts")
    cax.grid(True, alpha=0.25)
    cax.legend(loc="best", fontsize="small")
    cax.set_xlim(float(phi_centers[0]), float(phi_centers[-1]))
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _optional_scalar(data, key: str) -> float:
    if key not in data.files:
        return np.nan
    value = np.asarray(data[key])
    return float(value.item()) if value.shape == () else float(value.ravel()[0])


def _range_line(data, key: str, label: str) -> str:
    if key not in data.files:
        return f"{label}: not stored"
    values = np.asarray(data[key], dtype=float).ravel()
    if values.size < 2:
        return f"{label}: not stored"
    return f"{label}: {values[0]:.8g} to {values[1]:.8g}"


def _prepare_matplotlib_cache() -> None:
    import os
    import tempfile

    cache_root = Path(tempfile.gettempdir())
    mpl_dir = cache_root / "sf_analysis_matplotlib"
    xdg_dir = cache_root / "sf_analysis_xdg_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))


def command_bin_centering(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    binning = from_config(args.config)
    beam_energy = float(config["beam_energy"])
    phase_space = AnalysisPhaseSpace.from_config(config)
    if not args.exe.is_file():
        raise FileNotFoundError(f"aao_xsec executable not found: {args.exe}")
    if args.N <= 0:
        raise ValueError("--N must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    bin_start, bin_stop, total_3d_bins = _bin_centering_range(args, binning)

    evaluator = AaoExecutableEvaluator(
        exe=args.exe,
        beam_energy=beam_energy,
        theory=args.theory,
        channel=args.channel,
        resonance=args.resonance,
        workers=args.workers,
        chunk_size=args.chunk_size,
        verbose_failures=args.verbose_failures,
    )
    with evaluator:
        result = compute_bin_centering(
            binning,
            beam_energy,
            evaluator,
            samples_per_dimension=args.N,
            max_failure_fraction=args.max_failure_fraction,
            bin_start=bin_start,
            bin_stop=bin_stop,
            progress_chunks=args.progress_chunks,
            phase_space=phase_space,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _save_bin_centering_artifact(
        args.output,
        binning,
        result,
        beam_energy=beam_energy,
        samples_per_dimension=args.N,
        max_failure_fraction=args.max_failure_fraction,
        theory=args.theory,
        channel=args.channel,
        resonance=args.resonance,
        bin_start=bin_start,
        bin_stop=bin_stop,
        total_3d_bins=total_3d_bins,
        phase_space=phase_space,
    )
    reliable_bins = int(np.count_nonzero(result.reliable & result.computed))
    computed_bins = int(np.count_nonzero(result.computed))
    physical_bins = int(np.count_nonzero(result.n_physical))
    print(f"3D bin range: [{bin_start}, {bin_stop}) / {total_3d_bins}")
    print(f"Computed phi bins: {computed_bins}/{result.computed.size}")
    print(f"Physical phi bins: {physical_bins}/{result.n_physical.size}")
    print(f"Reliable C_BC bins: {reliable_bins}/{result.reliable.size}")
    if reliable_bins:
        print(f"Mean C_BC reliable: {np.nanmean(result.c_bc[result.reliable & result.computed]):.6g}")
    if phase_space.enabled:
        print(f"Analysis phase space: physical bin and {phase_space.description()}")
    print(f"Wrote {args.output}")


def _bin_centering_range(args: argparse.Namespace, binning) -> tuple[int, int, int]:
    total_3d_bins = int(np.prod(binning.shape[:3]))
    if (args.bin_chunks is None) != (args.bin_chunk_index is None):
        raise ValueError("--bin-chunks and --bin-chunk-index must be used together")
    if args.bin_chunks is not None:
        if args.bin_chunks <= 0:
            raise ValueError("--bin-chunks must be positive")
        if args.bin_chunk_index < 0 or args.bin_chunk_index >= args.bin_chunks:
            raise ValueError("--bin-chunk-index must satisfy 0 <= index < --bin-chunks")
        if args.bin_start != 0 or args.bin_stop is not None:
            raise ValueError("use either --bin-start/--bin-stop or --bin-chunks/--bin-chunk-index, not both")
        edges = np.linspace(0, total_3d_bins, args.bin_chunks + 1, dtype=int)
        return int(edges[args.bin_chunk_index]), int(edges[args.bin_chunk_index + 1]), total_3d_bins
    bin_start = int(args.bin_start)
    bin_stop = total_3d_bins if args.bin_stop is None else int(args.bin_stop)
    if bin_start < 0 or bin_stop < bin_start or bin_stop > total_3d_bins:
        raise ValueError(f"invalid 3D bin range [{bin_start}, {bin_stop}) for {total_3d_bins} bins")
    return bin_start, bin_stop, total_3d_bins


def _save_bin_centering_artifact(
    path: Path,
    binning,
    result,
    *,
    beam_energy: float,
    samples_per_dimension: int,
    max_failure_fraction: float,
    theory: int,
    channel: int,
    resonance: int,
    bin_start: int,
    bin_stop: int,
    total_3d_bins: int,
    phase_space: AnalysisPhaseSpace | None = None,
) -> None:
    if phase_space is None:
        phase_space = AnalysisPhaseSpace()
    np.savez_compressed(
        path,
        C_BC=result.c_bc,
        reliable=result.reliable,
        computed=result.computed,
        average_d4sigma=result.average_d4sigma,
        center_d4sigma=result.center_d4sigma,
        xB_center=result.xB_center,
        q2_center=result.q2_center,
        minus_t_center=result.minus_t_center,
        phi_center=result.phi_center,
        n_physical=result.n_physical,
        n_valid=result.n_valid,
        n_failed=result.n_failed,
        physical_fraction=result.physical_fraction,
        failure_fraction=result.failure_fraction,
        q2_edges=binning.q2_edges,
        xb_edges=binning.xb_edges,
        t_edges=binning.t_edges,
        phi_edges=binning.phi_edges,
        beam_energy=beam_energy,
        samples_per_dimension=samples_per_dimension,
        max_failure_fraction=max_failure_fraction,
        theory=theory,
        channel=channel,
        resonance=resonance,
        bin_start=bin_start,
        bin_stop=bin_stop,
        total_3d_bins=total_3d_bins,
        convention=(
            "C_BC = <d4sigma>_physical_selected_bin / "
            "d4sigma(physical selected midpoint-grid centroid)"
        ),
        apply_as="centered_cross_section = bin_averaged_cross_section / C_BC",
        t_convention="positive -t externally; signed t passed to aao_xsec",
        phase_space_definition=(
            "exclusive physical bin"
            if not phase_space.enabled
            else f"exclusive physical bin and {phase_space.description()}"
        ),
        **phase_space.as_npz_fields(),
    )


def command_bin_centering_merge(args: argparse.Namespace) -> None:
    partials = [np.load(path, allow_pickle=False) for path in args.partials]
    if not partials:
        raise ValueError("at least one partial artifact is required")
    first = partials[0]
    q2_edges = np.asarray(first["q2_edges"], dtype=float)
    xb_edges = np.asarray(first["xb_edges"], dtype=float)
    t_edges = np.asarray(first["t_edges"], dtype=float)
    phi_edges = np.asarray(first["phi_edges"], dtype=float)
    shape = (
        q2_edges.size - 1,
        xb_edges.size - 1,
        t_edges.size - 1,
        phi_edges.size - 1,
    )
    fields = {
        "C_BC": np.ones(shape, dtype=float),
        "reliable": np.zeros(shape, dtype=bool),
        "computed": np.zeros(shape, dtype=bool),
        "average_d4sigma": np.full(shape, np.nan, dtype=float),
        "center_d4sigma": np.full(shape, np.nan, dtype=float),
        "xB_center": np.full(shape, np.nan, dtype=float),
        "q2_center": np.full(shape, np.nan, dtype=float),
        "minus_t_center": np.full(shape, np.nan, dtype=float),
        "phi_center": np.full(shape, np.nan, dtype=float),
        "n_physical": np.zeros(shape, dtype=np.int64),
        "n_valid": np.zeros(shape, dtype=np.int64),
        "n_failed": np.zeros(shape, dtype=np.int64),
        "physical_fraction": np.zeros(shape, dtype=float),
        "failure_fraction": np.ones(shape, dtype=float),
    }
    metadata_keys = (
        "beam_energy",
        "samples_per_dimension",
        "max_failure_fraction",
        "theory",
        "channel",
        "resonance",
        "phase_space_Q2_min",
        "phase_space_W_min",
        "phase_space_y_max",
    )

    for path, partial in zip(args.partials, partials):
        for edge_name, expected in (
            ("q2_edges", q2_edges),
            ("xb_edges", xb_edges),
            ("t_edges", t_edges),
            ("phi_edges", phi_edges),
        ):
            values = np.asarray(partial[edge_name], dtype=float)
            if values.shape != expected.shape or not np.allclose(values, expected):
                raise ValueError(f"{path} {edge_name} does not match the first partial")
        computed = (
            np.asarray(partial["computed"], dtype=bool)
            if "computed" in partial.files
            else np.ones(shape, dtype=bool)
        )
        if computed.shape != shape:
            raise ValueError(f"{path} computed mask has shape {computed.shape}; expected {shape}")
        overlap = fields["computed"] & computed
        if np.any(overlap):
            raise ValueError(f"{path} overlaps previously merged bin-centering bins")
        for name, target in fields.items():
            if name == "computed":
                continue
            values = np.asarray(partial[name])
            if values.shape != shape:
                raise ValueError(f"{path} {name} has shape {values.shape}; expected {shape}")
            target[computed] = values[computed]
        fields["computed"][computed] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(fields)
    payload.update(
        q2_edges=q2_edges,
        xb_edges=xb_edges,
        t_edges=t_edges,
        phi_edges=phi_edges,
        bin_start=0,
        bin_stop=shape[0] * shape[1] * shape[2],
        total_3d_bins=shape[0] * shape[1] * shape[2],
        convention=_npz_string(first, "convention", "C_BC = <d4sigma>_physical_bin / d4sigma(center)"),
        apply_as=_npz_string(first, "apply_as", "centered_cross_section = bin_averaged_cross_section / C_BC"),
        t_convention=_npz_string(first, "t_convention", "positive -t externally; signed t passed to aao_xsec"),
        phase_space_definition=_npz_string(first, "phase_space_definition", "exclusive physical bin"),
    )
    for key in metadata_keys:
        if key in first.files:
            payload[key] = np.asarray(first[key]).item()
    np.savez_compressed(args.output, **payload)
    computed_phi = int(np.count_nonzero(fields["computed"]))
    reliable_phi = int(np.count_nonzero(fields["computed"] & fields["reliable"]))
    print(f"Merged partial files: {len(args.partials)}")
    print(f"Computed phi bins: {computed_phi}/{fields['computed'].size}")
    print(f"Reliable C_BC bins: {reliable_phi}/{fields['reliable'].size}")
    print(f"Wrote {args.output}")


def command_cross_section(args: argparse.Namespace) -> None:
    result = np.load(args.unfolding_result, allow_pickle=False)
    config = load_config(args.config)
    binning = from_config(args.config)
    beam_energy = float(config["beam_energy"])
    phase_space = AnalysisPhaseSpace.from_config(config)
    target = Target(
        float(config["target_length_cm"]),
        float(config["target_density_g_cm3"]),
        float(config["target_molar_mass_g"]),
    )
    beam_charge = float(result["beam_charge_c"])
    if not np.isfinite(beam_charge) or beam_charge <= 0:
        raise ValueError("unfolding result does not contain a positive beam_charge_c")
    luminosity = integrated_luminosity_fb(beam_charge, target)
    volumes = binning.flatten_values(
        physical_bin_volumes(
            binning,
            beam_energy,
            q2_minimum=1.0 if phase_space.q2_min is None else phase_space.q2_min,
            w_minimum=2.0 if phase_space.w_min is None else phase_space.w_min,
            y_maximum=0.8 if phase_space.y_max is None else phase_space.y_max,
        )
    )
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
    bin_centering_cbc = None
    bin_centering_reliable = None
    if args.bin_centering:
        bin_centering = np.load(args.bin_centering, allow_pickle=False)
        bin_centering_cbc = np.asarray(bin_centering["C_BC"], dtype=float)
        bin_centering_reliable = np.asarray(bin_centering["reliable"], dtype=bool)
        if bin_centering_cbc.shape != binning.shape:
            raise ValueError(f"bin-centering C_BC has shape {bin_centering_cbc.shape}; expected {binning.shape}")
        if bin_centering_reliable.shape != binning.shape:
            raise ValueError(
                f"bin-centering reliable has shape {bin_centering_reliable.shape}; expected {binning.shape}"
            )
        cbc_flat = binning.flatten_values(bin_centering_cbc)
        reliable_flat = binning.flatten_values(bin_centering_reliable)
        apply_mask = reliable_flat & np.isfinite(cbc_flat) & (cbc_flat > 0.0)
        values = np.divide(values, cbc_flat, out=np.zeros_like(values), where=apply_mask)
        errors = np.divide(errors, cbc_flat, out=np.zeros_like(errors), where=apply_mask)
    values /= args.global_normalization
    errors /= args.global_normalization
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        reduced_cross_section=binning.unflatten(values),
        uncertainty=binning.unflatten(errors),
        flux_q2_mean=q2_means,
        flux_xb_mean=xb_means,
        bin_volume=binning.unflatten(volumes),
        reduced_cross_section_units="nb/(GeV^2 rad)",
        luminosity_fb=luminosity,
        global_normalization=args.global_normalization,
        q2_edges=binning.q2_edges,
        xb_edges=binning.xb_edges,
        t_edges=binning.t_edges,
        phi_edges=binning.phi_edges,
    )
    payload.update(
        phase_space_definition=(
            "4D bin"
            if not phase_space.enabled
            else f"4D bin and {phase_space.description()}"
        ),
        **phase_space.as_npz_fields(),
    )
    if bin_centering_cbc is not None and bin_centering_reliable is not None:
        payload.update(
            bin_centering_C_BC=bin_centering_cbc,
            bin_centering_reliable=bin_centering_reliable,
            bin_centering_path=str(args.bin_centering),
            bin_centering_application="reduced_cross_section and uncertainty divided by C_BC",
        )
    np.savez_compressed(args.output, **payload)
    print(f"Integrated luminosity: {luminosity:.6g} fb^-1")
    if args.bin_centering:
        print(f"Applied bin-centering correction: {args.bin_centering}")
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


def command_harmonic_plots(args: argparse.Namespace) -> None:
    harmonics = np.load(args.harmonics, allow_pickle=False)
    parameters = np.asarray(harmonics["parameters"], dtype=float)
    covariance = np.asarray(harmonics["covariance"], dtype=float)
    chi2_ndf = np.asarray(harmonics["chi2_ndf"], dtype=float)
    points = np.asarray(harmonics["points"], dtype=int)
    q2_edges = np.asarray(harmonics["q2_edges"], dtype=float)
    xb_edges = np.asarray(harmonics["xb_edges"], dtype=float)
    t_edges = np.asarray(harmonics["t_edges"], dtype=float)
    if "coefficient_names" in harmonics.files:
        names = tuple(str(item) for item in harmonics["coefficient_names"])
    else:
        names = ("A", "B", "C")
    if parameters.ndim != 4 or parameters.shape[-1] != 3:
        raise ValueError("harmonic parameters must have shape (Q2, xB, t, 3)")
    expected_shape = (q2_edges.size - 1, xb_edges.size - 1, t_edges.size - 1)
    if parameters.shape[:-1] != expected_shape:
        raise ValueError("harmonic arrays do not match bin-edge dimensions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit_mask = np.isfinite(chi2_ndf) & (points >= args.min_points)
    _plot_harmonic_overview_maps(
        fit_mask,
        chi2_ndf,
        points,
        q2_edges,
        xb_edges,
        args.output_dir,
    )
    pages = _plot_harmonic_coefficients_vs_t(
        parameters,
        covariance,
        chi2_ndf,
        points,
        fit_mask,
        q2_edges,
        xb_edges,
        t_edges,
        names,
        args.output_dir / "harmonic_coefficients_vs_t.pdf",
        args.output_dir / "harmonic_coefficients_summary.csv",
        include_quilt=args.quilt,
        quilt_scale_percentile=args.quilt_scale_percentile,
        quilt_scale_mode=args.quilt_scale_mode,
    )
    print(f"Successful fits: {int(fit_mask.sum())}")
    print(f"Coefficient pages: {pages}")
    if args.quilt:
        print("Coefficient PDF includes one stitched quilt page")
    print(f"Wrote harmonic plots under {args.output_dir}")


def command_cross_section_plots(args: argparse.Namespace) -> None:
    cross_section = np.load(args.cross_section, allow_pickle=False)
    harmonics = np.load(args.harmonics, allow_pickle=False)
    values = np.asarray(cross_section["reduced_cross_section"], dtype=float)
    uncertainties = np.asarray(cross_section["uncertainty"], dtype=float)
    units = _npz_string(cross_section, "reduced_cross_section_units", "nb/(GeV^2 rad)")
    phi_edges = np.asarray(cross_section["phi_edges"], dtype=float)
    parameters = np.asarray(harmonics["parameters"], dtype=float)
    chi2_ndf = np.asarray(harmonics["chi2_ndf"], dtype=float)
    points = np.asarray(harmonics["points"], dtype=int)
    q2_edges = np.asarray(harmonics["q2_edges"], dtype=float)
    xb_edges = np.asarray(harmonics["xb_edges"], dtype=float)
    t_edges = np.asarray(harmonics["t_edges"], dtype=float)

    if values.shape != uncertainties.shape or values.ndim != 4:
        raise ValueError("cross-section values and uncertainties must be equal 4D arrays")
    if values.shape[:-1] != parameters.shape[:-1] or parameters.shape[-1] != 3:
        raise ValueError("cross-section and harmonic-fit dimensions do not match")
    if phi_edges.size != values.shape[-1] + 1:
        raise ValueError("phi edges do not match cross-section bins")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = _plot_cross_section_vs_phi(
        values,
        uncertainties,
        units,
        phi_edges,
        parameters,
        chi2_ndf,
        points,
        q2_edges,
        xb_edges,
        t_edges,
        args.min_points,
        args.output_dir / "reduced_cross_section_vs_phi_with_fits.pdf",
        args.output_dir / "reduced_cross_section_vs_phi_summary.csv",
        include_quilt=args.quilt,
        quilt_scale_mode=args.quilt_scale_mode,
    )
    print(f"Cross-section phi pages: {pages}")
    print(f"Wrote cross-section plots under {args.output_dir}")


def command_acceptance_plots(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    metadata = np.load(args.response_meta, allow_pickle=False)
    efficiency = np.asarray(metadata["efficiency"], dtype=float)
    truth = np.asarray(metadata["truth_total"], dtype=float)
    reconstructed = np.asarray(metadata["reconstructed_total"], dtype=float)
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
    if (
        efficiency.size != int(np.prod(shape))
        or truth.size != efficiency.size
        or reconstructed.size != efficiency.size
    ):
        raise ValueError("response metadata arrays do not match bin-edge dimensions")

    response_matrix = args.response_matrix
    if response_matrix is None:
        candidate = args.response_meta.parent / "response_matrix.npz"
        response_matrix = candidate if candidate.exists() else None
    same_bin_efficiency = _response_diagonal(response_matrix, efficiency.size) if response_matrix else None

    eff4 = _unflatten_response(efficiency, shape)
    truth4 = _unflatten_response(truth, shape)
    rec4 = _unflatten_response(reconstructed, shape)
    acceptance4 = np.divide(
        rec4,
        truth4,
        out=np.zeros_like(rec4),
        where=truth4 > 0,
    )
    same_bin4 = (
        _unflatten_response(same_bin_efficiency, shape)
        if same_bin_efficiency is not None else None
    )
    purity4 = None
    if args.include_purity and same_bin4 is not None:
        same_counts4 = same_bin4 * truth4
        purity4 = np.divide(
            same_counts4,
            rec4,
            out=np.zeros_like(same_counts4),
            where=rec4 > 0,
        )
    populated = truth4 > 0
    zero = populated & (eff4 == 0)
    low = populated & (eff4 > 0) & (eff4 < args.minimum_acceptance)
    passing = populated & (eff4 >= args.minimum_acceptance)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _plot_acceptance_histograms(
        {
            "A_i bin-by-bin acceptance": acceptance4[populated],
            "epsilon_i IBU total efficiency": eff4[populated],
            **({
                "E_i same-bin efficiency": same_bin4[populated],
            } if same_bin4 is not None else {}),
            **({"P_i purity": purity4[populated]} if purity4 is not None else {}),
        },
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
        acceptance4,
        purity4,
        same_bin4,
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
    if response_matrix:
        print(f"Migration diagnostic source: {response_matrix}")
        if not args.include_purity:
            print("Purity overlay: disabled; pass --include-purity to show P_i")
    else:
        print("Migration diagnostic source: unavailable; pass --response-matrix to include E_i/P_i")
    print(f"Wrote acceptance plots under {args.output_dir}")


def command_response_plots(args: argparse.Namespace) -> None:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    metadata = np.load(args.response_meta, allow_pickle=False)
    matrix = load_npz(args.response_matrix).tocsr()
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
    number_of_bins = int(np.prod(shape))
    if matrix.shape[0] != number_of_bins or matrix.shape[1] < number_of_bins:
        raise ValueError(
            f"response matrix shape {matrix.shape} does not match {number_of_bins} analysis bins"
        )
    core = matrix[:, :number_of_bins].tocoo()
    efficiency = np.asarray(metadata["efficiency"], dtype=float)
    truth = np.asarray(metadata["truth_total"], dtype=float)
    if efficiency.size != number_of_bins or truth.size != number_of_bins:
        raise ValueError("response metadata arrays do not match bin-edge dimensions")

    rec_indices = _flat_indices_for_shape(core.row, shape)
    truth_indices = _flat_indices_for_shape(core.col, shape)
    migration = _response_migration_diagnostics(
        core,
        rec_indices,
        truth_indices,
        shape,
        efficiency,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        _plot_response_summary_page(
            pdf,
            args.response_matrix,
            args.response_meta,
            matrix,
            core,
            efficiency,
            truth,
        )
        _plot_response_sparse_global_page(pdf, core, number_of_bins, args.max_points, args.seed)
        _plot_response_variable_matrices_page(pdf, migration["collapsed_matrices"], shape)
        _plot_response_variable_histograms_page(
            pdf,
            efficiency,
            migration["same_bin_probability"],
            migration["different_probability"],
            migration["mean_abs_delta"],
        )
        _plot_response_projection_page(
            pdf,
            migration["different_probability"],
            shape,
            q2_edges,
            xb_edges,
            projection="q2_xb",
        )
        _plot_response_projection_page(
            pdf,
            migration["different_probability"],
            shape,
            t_edges,
            phi_edges,
            projection="t_phi",
        )

    print(f"Wrote response diagnostics PDF: {args.output}")


def _load_mask(path: Path, expected: int) -> np.ndarray:
    mask = np.asarray(np.load(path), dtype=bool)
    if mask.shape != (expected,):
        raise ValueError(f"selection mask has {mask.size} rows; expected {expected}")
    return mask


def _unflatten_response(values: np.ndarray, shape: tuple[int, int, int, int]) -> np.ndarray:
    nq2, nxb, nt, nphi = shape
    return values.reshape(nxb, nq2, nphi, nt).transpose(1, 0, 3, 2)


def _response_diagonal(response_matrix: Path, number_of_bins: int) -> np.ndarray:
    matrix = load_npz(response_matrix).tocsr()
    if matrix.shape[0] != number_of_bins or matrix.shape[1] < number_of_bins:
        raise ValueError(
            f"response matrix shape {matrix.shape} does not match {number_of_bins} analysis bins"
        )
    return matrix[:, :number_of_bins].diagonal()


RESPONSE_VARIABLES = ("Q2", "xB", "-t", "phi")
RESPONSE_LABEL_CELL_LIMIT = 180


def _flat_indices_for_shape(flat: np.ndarray, shape: tuple[int, int, int, int]) -> tuple[np.ndarray, ...]:
    nq2, nxb, nt, nphi = shape
    flat = np.asarray(flat, dtype=np.int64)
    it = flat % nt
    tmp = flat // nt
    iphi = tmp % nphi
    tmp = tmp // nphi
    iq2 = tmp % nq2
    ixb = tmp // nq2
    return iq2, ixb, it, iphi


def _response_migration_diagnostics(
    core,
    rec_indices: tuple[np.ndarray, ...],
    truth_indices: tuple[np.ndarray, ...],
    shape: tuple[int, int, int, int],
    efficiency: np.ndarray,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    number_of_bins = int(np.prod(shape))
    data = np.asarray(core.data, dtype=float)
    columns = np.asarray(core.col, dtype=np.int64)

    same_all = np.ones(data.size, dtype=bool)
    for rec, truth in zip(rec_indices, truth_indices):
        same_all &= rec == truth
    same_bin_probability = np.bincount(
        columns[same_all],
        weights=data[same_all],
        minlength=number_of_bins,
    ).astype(float)

    different_probability = np.zeros((4, number_of_bins), dtype=float)
    mean_abs_delta = np.zeros((4, number_of_bins), dtype=float)
    collapsed_matrices: list[np.ndarray] = []
    for ivar, (rec, truth) in enumerate(zip(rec_indices, truth_indices)):
        delta = np.abs(rec - truth).astype(float)
        different = delta > 0
        different_probability[ivar] = np.bincount(
            columns[different],
            weights=data[different],
            minlength=number_of_bins,
        ).astype(float)
        delta_sum = np.bincount(columns, weights=data * delta, minlength=number_of_bins).astype(float)
        mean_abs_delta[ivar] = np.divide(
            delta_sum,
            efficiency,
            out=np.zeros(number_of_bins, dtype=float),
            where=efficiency > 0,
        )
        size = shape[ivar]
        collapsed = np.zeros((size, size), dtype=float)
        np.add.at(collapsed, (rec, truth), data)
        truth_totals = collapsed.sum(axis=0)
        collapsed = np.divide(
            collapsed,
            truth_totals[np.newaxis, :],
            out=np.zeros_like(collapsed),
            where=truth_totals[np.newaxis, :] > 0,
        )
        collapsed_matrices.append(collapsed)

    return {
        "same_bin_probability": same_bin_probability,
        "different_probability": different_probability,
        "mean_abs_delta": mean_abs_delta,
        "collapsed_matrices": collapsed_matrices,
    }


def _positive_heatmap_scale(values: np.ndarray, percentile: float = 95.0) -> float:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(positive, percentile))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(positive))
    return max(vmax, 1.0e-12)


def _heatmap_label(value: float) -> str:
    if value >= 0.1:
        return f"{value:.2f}"
    if value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.1e}"


def _annotate_heatmap_cells(
    ax,
    values: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    *,
    max_cells: int = RESPONSE_LABEL_CELL_LIMIT,
) -> None:
    if values.size > max_cells:
        return
    for iy, y in enumerate(y_centers):
        for ix, x in enumerate(x_centers):
            value = float(values[iy, ix])
            if not np.isfinite(value) or value <= 0.0:
                continue
            ax.text(
                x,
                y,
                _heatmap_label(value),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value < 0.55 else "black",
            )


def _plot_response_summary_page(
    pdf,
    response_matrix: Path,
    response_meta: Path,
    matrix,
    core,
    efficiency: np.ndarray,
    truth: np.ndarray,
) -> None:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    populated = truth > 0
    lines = [
        "Response-matrix diagnostics",
        "",
        f"Response matrix: {response_matrix}",
        f"Response metadata: {response_meta}",
        f"Matrix shape: {matrix.shape}",
        f"Core nonzero entries: {core.nnz}",
        f"Analysis bins: {efficiency.size}",
        f"Truth-populated bins: {int(np.count_nonzero(populated))}",
        f"Mean efficiency, populated: {np.nanmean(efficiency[populated]) if np.any(populated) else np.nan:.8g}",
        f"Median efficiency, populated: {np.nanmedian(efficiency[populated]) if np.any(populated) else np.nan:.8g}",
        f"Max efficiency: {np.nanmax(efficiency) if efficiency.size else np.nan:.8g}",
        "",
        "Rows are reconstructed bins; columns are truth bins.",
        "Each core column sums to the IBU reconstruction efficiency for that truth bin.",
        "Collapsed variable matrices are column-normalized within the reconstructed sample.",
    ]
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.94, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def _plot_response_sparse_global_page(pdf, core, number_of_bins: int, max_points: int, seed: int) -> None:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    if max_points <= 0:
        raise ValueError("--max-points must be positive")
    rng = np.random.default_rng(seed)
    rows = np.asarray(core.row)
    cols = np.asarray(core.col)
    values = np.asarray(core.data, dtype=float)
    if values.size > max_points:
        choice = rng.choice(values.size, size=max_points, replace=False)
        rows = rows[choice]
        cols = cols[choice]
        values = values[choice]
        sampled = True
    else:
        sampled = False
    colors = np.log10(np.clip(values, 1.0e-12, None))
    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    scatter = ax.scatter(cols, rows, c=colors, s=0.15, marker="s", linewidths=0, cmap="viridis")
    fig.colorbar(scatter, ax=ax, label="log10 R[reco, truth]")
    ax.plot([0, number_of_bins], [0, number_of_bins], color="white", linewidth=0.6, alpha=0.7)
    ax.set_xlim(0, number_of_bins)
    ax.set_ylim(number_of_bins, 0)
    ax.set_xlabel("Truth flat bin")
    ax.set_ylabel("Reconstructed flat bin")
    title = "Sparse global response matrix"
    if sampled:
        title += f" ({max_points:g} sampled nonzeros)"
    ax.set_title(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_response_variable_matrices_page(pdf, matrices: list[np.ndarray], shape: tuple[int, ...]) -> None:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    for ax, label, matrix, size in zip(axes.ravel(), RESPONSE_VARIABLES, matrices, shape):
        shown = np.where(matrix > 0, matrix, np.nan)
        image = ax.imshow(shown, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        fig.colorbar(image, ax=ax, label="P(reco bin | truth bin, reconstructed)")
        _annotate_heatmap_cells(
            ax,
            matrix,
            np.arange(matrix.shape[1]),
            np.arange(matrix.shape[0]),
            max_cells=RESPONSE_LABEL_CELL_LIMIT,
        )
        ax.plot([-0.5, size - 0.5], [-0.5, size - 0.5], color="white", linewidth=0.8, alpha=0.8)
        ax.set_xlabel(f"Truth {label} bin")
        ax.set_ylabel(f"Reco {label} bin")
        ax.set_title(f"Collapsed migration in {label}")
    fig.suptitle("Variable-wise collapsed response matrices")
    pdf.savefig(fig)
    plt.close(fig)


def _plot_response_variable_histograms_page(
    pdf,
    efficiency: np.ndarray,
    same_bin_probability: np.ndarray,
    different_probability: np.ndarray,
    mean_abs_delta: np.ndarray,
) -> None:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    populated = efficiency > 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].hist(efficiency[populated], bins=80, histtype="step", linewidth=1.5, label="epsilon total")
    axes[0].hist(
        same_bin_probability[populated],
        bins=80,
        histtype="step",
        linewidth=1.5,
        label="same 4D bin",
    )
    for ivar, label in enumerate(RESPONSE_VARIABLES):
        axes[0].hist(
            different_probability[ivar, populated],
            bins=80,
            histtype="step",
            linewidth=1.2,
            label=f"different {label}",
        )
    axes[0].set_xlabel("Unconditional response probability per truth bin")
    axes[0].set_ylabel("Truth bins")
    axes[0].set_title("Migration probability distributions")
    axes[0].legend(fontsize="x-small")
    axes[0].grid(True, alpha=0.25)

    means = [
        np.nanmean(mean_abs_delta[ivar, populated])
        if np.any(populated) else np.nan
        for ivar in range(len(RESPONSE_VARIABLES))
    ]
    axes[1].bar(RESPONSE_VARIABLES, means, color="#4c78a8")
    axes[1].set_ylabel("Mean |reco index - truth index| among reconstructed")
    axes[1].set_title("Average migration distance by variable")
    axes[1].grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(means):
        axes[1].text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)

    pdf.savefig(fig)
    plt.close(fig)


def _plot_response_projection_page(
    pdf,
    different_probability: np.ndarray,
    shape: tuple[int, int, int, int],
    y_edges: np.ndarray,
    x_edges: np.ndarray,
    *,
    projection: str,
) -> None:
    _prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    if projection == "q2_xb":
        title = "Median per-truth-bin migration probability in Q2 and xB"
        x_label, y_label = "xB", "Q2 [GeV^2]"
        axes_to_reduce = (2, 3)
    elif projection == "t_phi":
        title = "Median per-truth-bin migration probability in phi and -t"
        x_label, y_label = "phi [deg]", "-t [GeV^2]"
        axes_to_reduce = (0, 1)
    else:
        raise ValueError(f"unknown response projection: {projection}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    for ax, label, values in zip(axes.ravel(), RESPONSE_VARIABLES, different_probability):
        values4 = _unflatten_response(values, shape)
        projected = _nanmedian_where(values4, values4 > 0.0, axis=axes_to_reduce)
        vmax = _positive_heatmap_scale(projected)
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("#eeeeee")
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(projected),
            vmin=0.0,
            vmax=vmax,
            cmap=cmap,
            shading="flat",
        )
        fig.colorbar(image, ax=ax, label=f"P(reco {label} != truth {label})")
        _annotate_heatmap_cells(
            ax,
            projected,
            0.5 * (x_edges[:-1] + x_edges[1:]),
            0.5 * (y_edges[:-1] + y_edges[1:]),
            max_cells=RESPONSE_LABEL_CELL_LIMIT,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Migration in {label} (scale max {vmax:.3g})")
    fig.suptitle(title)
    pdf.savefig(fig)
    plt.close(fig)


def _edge_labels(edges: np.ndarray) -> list[str]:
    return [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]


def _pass_fraction(populated: np.ndarray, passing: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    total = populated.sum(axis=axes)
    good = passing.sum(axis=axes)
    return np.divide(good, total, out=np.full(total.shape, np.nan), where=total > 0)


def _plot_acceptance_histograms(
    values_by_label: dict[str, np.ndarray],
    minimum_acceptance: float,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in values_by_label.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        ax.hist(finite, bins=100, histtype="step", linewidth=1.5, label=label)
    ax.axvline(minimum_acceptance, color="red", linestyle="--",
               label=f"minimum_acceptance = {minimum_acceptance:g}")
    ax.set_xlabel("Acceptance-like quantity")
    ax.set_ylabel("Number of 4D bins")
    ax.set_title("Acceptance diagnostics over truth-populated 4D bins")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "acceptance_hist.png", dpi=200)
    plt.close(fig)

    positive_sets = {}
    for label, values in values_by_label.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite) & (finite > 0)]
        if finite.size > 0:
            positive_sets[label] = finite
    if not positive_sets:
        return
    positive_min = min(values.min() for values in positive_sets.values())
    positive_max = max(values.max() for values in positive_sets.values())
    if positive_min <= 0 or positive_max <= positive_min:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(positive_min), np.log10(positive_max), 100)
    for label, values in positive_sets.items():
        ax.hist(values, bins=bins, histtype="step", linewidth=1.5, label=label)
    ax.axvline(minimum_acceptance, color="red", linestyle="--",
               label=f"minimum_acceptance = {minimum_acceptance:g}")
    ax.set_xscale("log")
    ax.set_xlabel("Acceptance-like quantity")
    ax.set_ylabel("Number of 4D bins")
    ax.set_title("Positive acceptance diagnostics over truth-populated 4D bins")
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


def _plot_harmonic_overview_maps(
    fit_mask: np.ndarray,
    chi2_ndf: np.ndarray,
    points: np.ndarray,
    q2_edges: np.ndarray,
    xb_edges: np.ndarray,
    output_dir: Path,
) -> None:
    fit_counts = fit_mask.sum(axis=2)
    chi2_median = np.full(fit_counts.shape, np.nan, dtype=float)
    points_median = np.full(fit_counts.shape, np.nan, dtype=float)
    for iq2, ixb in zip(*np.nonzero(fit_counts)):
        mask = fit_mask[iq2, ixb, :]
        chi2_median[iq2, ixb] = np.nanmedian(chi2_ndf[iq2, ixb, mask])
        points_median[iq2, ixb] = np.nanmedian(points[iq2, ixb, mask])
    _plot_heatmap(
        fit_counts,
        "Q2",
        "xB",
        _edge_labels(q2_edges),
        _edge_labels(xb_edges),
        "Successful harmonic fits per (Q2, xB)",
        "fit count over -t bins",
        output_dir / "harmonic_fit_count_q2_xb.png",
        vmin=0.0,
        vmax=float(fit_mask.shape[2]),
    )
    _plot_heatmap(
        chi2_median,
        "Q2",
        "xB",
        _edge_labels(q2_edges),
        _edge_labels(xb_edges),
        "Median harmonic fit chi2/ndf per (Q2, xB)",
        "median chi2/ndf",
        output_dir / "harmonic_chi2_median_q2_xb.png",
    )
    _plot_heatmap(
        points_median,
        "Q2",
        "xB",
        _edge_labels(q2_edges),
        _edge_labels(xb_edges),
        "Median phi points used per harmonic fit",
        "median points",
        output_dir / "harmonic_points_median_q2_xb.png",
        vmin=0.0,
    )


def _plot_harmonic_coefficients_vs_t(
    parameters: np.ndarray,
    covariance: np.ndarray,
    chi2_ndf: np.ndarray,
    points: np.ndarray,
    fit_mask: np.ndarray,
    q2_edges: np.ndarray,
    xb_edges: np.ndarray,
    t_edges: np.ndarray,
    names: tuple[str, ...],
    pdf_path: Path,
    csv_path: Path,
    include_quilt: bool = False,
    quilt_scale_percentile: float = 98.0,
    quilt_scale_mode: str = "global",
) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = ("#1b9e77", "#d95f02", "#7570b3")
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    errors = np.sqrt(np.where(covariance >= 0.0, covariance, np.nan))
    coeff_errors = np.stack([errors[..., i, i] for i in range(3)], axis=-1)
    pages = 0
    csv_lines = [
        "iq2,q2_low,q2_high,ixb,xb_low,xb_high,it,t_low,t_high,"
        "points,chi2_ndf,A,A_err,B,B_err,C,C_err,B_over_A,C_over_A"
    ]

    with PdfPages(pdf_path) as pdf:
        if include_quilt:
            pages += _plot_harmonic_coefficient_quilt_vs_t(
                pdf,
                parameters,
                coeff_errors,
                fit_mask,
                q2_edges,
                xb_edges,
                t_edges,
                names,
                colors,
                quilt_scale_percentile,
                quilt_scale_mode,
            )
        for iq2 in range(parameters.shape[0]):
            for ixb in range(parameters.shape[1]):
                mask = fit_mask[iq2, ixb, :]
                if not np.any(mask):
                    continue
                fig, ax = plt.subplots(figsize=(8, 5))
                for coeff_index in range(3):
                    y = parameters[iq2, ixb, :, coeff_index]
                    yerr = coeff_errors[iq2, ixb, :, coeff_index]
                    ax.errorbar(
                        t_centers[mask],
                        y[mask],
                        yerr=yerr[mask],
                        fmt="o",
                        capsize=2,
                        linewidth=1.0,
                        markersize=4,
                        color=colors[coeff_index],
                        label=names[coeff_index] if coeff_index < len(names) else f"p{coeff_index}",
                    )
                ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
                ax.set_xlabel("-t bin center [GeV^2]")
                ax.set_ylabel("Harmonic coefficient [nb/(GeV^2 rad)]")
                ax.set_title(
                    "Harmonic coefficients vs -t\n"
                    f"Q2 {q2_edges[iq2]:g}-{q2_edges[iq2 + 1]:g}, "
                    f"xB {xb_edges[ixb]:g}-{xb_edges[ixb + 1]:g}"
                )
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best", fontsize="small")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                pages += 1

                for it in np.flatnonzero(mask):
                    a, b, c = parameters[iq2, ixb, it, :]
                    ea, eb, ec = coeff_errors[iq2, ixb, it, :]
                    b_over_a = b / a if np.isfinite(a) and a != 0.0 else np.nan
                    c_over_a = c / a if np.isfinite(a) and a != 0.0 else np.nan
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
                                int(points[iq2, ixb, it]),
                                float(chi2_ndf[iq2, ixb, it]),
                                float(a),
                                float(ea),
                                float(b),
                                float(eb),
                                float(c),
                                float(ec),
                                float(b_over_a),
                                float(c_over_a),
                            )
                        )
                    )

    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return pages


def _plot_harmonic_coefficient_quilt_vs_t(
    pdf,
    parameters: np.ndarray,
    coeff_errors: np.ndarray,
    fit_mask: np.ndarray,
    q2_edges: np.ndarray,
    xb_edges: np.ndarray,
    t_edges: np.ndarray,
    names: tuple[str, ...],
    colors: tuple[str, str, str],
    scale_percentile: float,
    scale_mode: str,
) -> int:
    import matplotlib.pyplot as plt

    nq2, nxb, nt = parameters.shape[:3]
    if not np.any(fit_mask):
        return 0
    if not 0.0 < scale_percentile <= 100.0:
        raise ValueError("--quilt-scale-percentile must be in the range (0, 100]")
    if scale_mode not in ("global", "panel"):
        raise ValueError("--quilt-scale-mode must be global or panel")

    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    q2_labels = _edge_labels(q2_edges)
    xb_labels = _edge_labels(xb_edges)
    visible_abs = []
    for coeff_index in range(3):
        values = np.where(fit_mask, parameters[..., coeff_index], np.nan)
        errors = np.where(fit_mask, coeff_errors[..., coeff_index], np.nan)
        for item in (values - errors, values, values + errors):
            finite = np.abs(item[np.isfinite(item)])
            if finite.size:
                visible_abs.append(finite.ravel())
    finite_abs = np.concatenate(visible_abs) if visible_abs else np.asarray([], dtype=float)
    if finite_abs.size:
        limit = float(np.nanpercentile(finite_abs, scale_percentile))
        if not np.isfinite(limit) or limit <= 0.0:
            limit = float(np.nanmax(finite_abs))
        limit = 1.08 * max(limit, 1.0e-12)
        ylim = (-limit, limit)
    else:
        ylim = (-1.0, 1.0)

    fig, axes = plt.subplots(
        nq2,
        nxb,
        figsize=(max(9.0, 1.65 * nxb), max(7.0, 1.2 * nq2)),
        sharex=True,
        sharey=scale_mode == "global",
        squeeze=False,
    )
    for iq2 in range(nq2):
        for ixb in range(nxb):
            ax = axes[nq2 - 1 - iq2, ixb]
            mask = fit_mask[iq2, ixb, :]
            if not np.any(mask):
                ax.set_axis_off()
                continue
            ax.axhline(0.0, color="black", linewidth=0.45, alpha=0.35)
            ax.set_xlim(float(t_edges[0]), float(t_edges[-1]))
            if scale_mode == "global":
                ax.set_ylim(*ylim)
            else:
                local_low = []
                local_high = []
                for coeff_index in range(3):
                    local_values = parameters[iq2, ixb, mask, coeff_index]
                    local_errors = coeff_errors[iq2, ixb, mask, coeff_index]
                    finite = np.isfinite(local_values) & np.isfinite(local_errors)
                    if np.any(finite):
                        local_low.extend((local_values[finite] - local_errors[finite]).tolist())
                        local_high.extend((local_values[finite] + local_errors[finite]).tolist())
                if local_low and local_high:
                    lower = min(0.0, float(np.min(local_low)))
                    upper = max(0.0, float(np.max(local_high)))
                    span = upper - lower
                    padding = 0.08 * span if span > 0.0 else max(abs(lower), abs(upper), 1.0) * 0.08
                    ax.set_ylim(lower - padding, upper + padding)
            ax.grid(True, alpha=0.16, linewidth=0.45)
            for coeff_index in range(3):
                ax.errorbar(
                    t_centers[mask],
                    parameters[iq2, ixb, mask, coeff_index],
                    yerr=coeff_errors[iq2, ixb, mask, coeff_index],
                    fmt="o-",
                    capsize=1.0,
                    linewidth=0.75,
                    markersize=2.4,
                    color=colors[coeff_index],
                )
            ax.tick_params(axis="both", labelsize=6, length=2, labelleft=True)
            if scale_mode == "panel":
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
                ax.yaxis.get_offset_text().set_fontsize(5)
            if iq2 == 0:
                ax.set_xlabel("-t [GeV^2]", fontsize=7)
            if ixb == 0:
                ax.set_ylabel(f"Q2 {q2_labels[iq2]}", fontsize=7)
            if iq2 == nq2 - 1:
                ax.set_title(f"xB {xb_labels[ixb]}", fontsize=7)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[index],
            marker="o",
            linewidth=1.2,
            markersize=3,
            label=names[index] if index < len(names) else f"p{index}",
        )
        for index in range(3)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize="small")
    fig.suptitle(
        "Harmonic coefficients vs -t quilt\n"
        "Q2 increases bottom to top; xB increases left to right; "
        + (
            "each panel uses its corresponding page scale"
            if scale_mode == "panel"
            else f"y scale uses p{scale_percentile:g}"
        ),
        y=0.985,
    )
    fig.text(
        0.012,
        0.5,
        "Harmonic coefficient [nb/(GeV^2 rad)]",
        rotation="vertical",
        va="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.052,
        right=0.995,
        bottom=0.045,
        top=0.915,
        wspace=0.16,
        hspace=0.18,
    )
    pdf.savefig(fig)
    plt.close(fig)
    return 1


def _plot_cross_section_vs_phi(
    values: np.ndarray,
    uncertainties: np.ndarray,
    units: str,
    phi_edges: np.ndarray,
    parameters: np.ndarray,
    chi2_ndf: np.ndarray,
    points: np.ndarray,
    q2_edges: np.ndarray,
    xb_edges: np.ndarray,
    t_edges: np.ndarray,
    min_points: int,
    pdf_path: Path,
    csv_path: Path,
    include_quilt: bool = False,
    quilt_scale_mode: str = "panel",
) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    phi_curve = np.linspace(float(phi_edges[0]), float(phi_edges[-1]), 361)
    radians = np.deg2rad(phi_curve)
    pages = 0
    csv_lines = [
        "iq2,q2_low,q2_high,ixb,xb_low,xb_high,it,t_low,t_high,"
        "points,chi2_ndf,A,B,C,finite_phi_bins,min_cross_section,max_cross_section"
    ]

    with PdfPages(pdf_path) as pdf:
        if include_quilt:
            pages += _plot_cross_section_quilts_vs_phi(
                pdf,
                values,
                uncertainties,
                units,
                phi_edges,
                parameters,
                chi2_ndf,
                points,
                q2_edges,
                xb_edges,
                t_edges,
                min_points,
                quilt_scale_mode,
            )
        for iq2 in range(values.shape[0]):
            for ixb in range(values.shape[1]):
                for it in range(values.shape[2]):
                    y = values[iq2, ixb, it, :]
                    yerr = uncertainties[iq2, ixb, it, :]
                    p = parameters[iq2, ixb, it, :]
                    valid = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0.0)
                    fit_valid = (
                        np.all(np.isfinite(p))
                        and np.isfinite(chi2_ndf[iq2, ixb, it])
                        and points[iq2, ixb, it] >= min_points
                    )
                    if not fit_valid:
                        continue
                    fit_curve = p[0] + p[1] * np.cos(radians) + p[2] * np.cos(2.0 * radians)
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
                                int(points[iq2, ixb, it]),
                                float(chi2_ndf[iq2, ixb, it]),
                                float(p[0]),
                                float(p[1]),
                                float(p[2]),
                                int(valid.sum()),
                                float(np.nanmin(y[valid])) if np.any(valid) else np.nan,
                                float(np.nanmax(y[valid])) if np.any(valid) else np.nan,
                            )
                        )
                    )

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.errorbar(
                        phi_centers[valid],
                        y[valid],
                        yerr=yerr[valid],
                        fmt="o",
                        capsize=2,
                        markersize=4,
                        linewidth=1.0,
                        label="reduced cross section",
                    )
                    ax.plot(
                        phi_curve,
                        fit_curve,
                        color="#d95f02",
                        linewidth=1.7,
                        label=(
                            "fit: A + B cos(phi) + C cos(2phi)\n"
                            f"A={p[0]:.4g}, B={p[1]:.4g}, C={p[2]:.4g}"
                        ),
                    )
                    ax.set_xlim(float(phi_edges[0]), float(phi_edges[-1]))
                    ax.set_xlabel("phi [deg]")
                    ax.set_ylabel(f"Reduced cross section [{units}]")
                    ax.set_title(
                        "Reduced cross section vs phi\n"
                        f"Q2 {q2_edges[iq2]:g}-{q2_edges[iq2 + 1]:g}, "
                        f"xB {xb_edges[ixb]:g}-{xb_edges[ixb + 1]:g}, "
                        f"-t {t_edges[it]:g}-{t_edges[it + 1]:g}\n"
                        f"chi2/ndf={chi2_ndf[iq2, ixb, it]:.3g}, points={points[iq2, ixb, it]}"
                    )
                    ax.grid(True, alpha=0.25)
                    ax.legend(loc="best", fontsize="small")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                    pages += 1

    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return pages


def _plot_cross_section_quilts_vs_phi(
    pdf,
    values: np.ndarray,
    uncertainties: np.ndarray,
    units: str,
    phi_edges: np.ndarray,
    parameters: np.ndarray,
    chi2_ndf: np.ndarray,
    points: np.ndarray,
    q2_edges: np.ndarray,
    xb_edges: np.ndarray,
    t_edges: np.ndarray,
    min_points: int,
    scale_mode: str,
) -> int:
    import matplotlib.pyplot as plt

    if scale_mode not in ("global", "panel"):
        raise ValueError("--quilt-scale-mode must be global or panel")
    nq2, nxb, nt = values.shape[:3]
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    phi_curve = np.linspace(float(phi_edges[0]), float(phi_edges[-1]), 361)
    radians = np.deg2rad(phi_curve)
    q2_labels = _edge_labels(q2_edges)
    xb_labels = _edge_labels(xb_edges)
    pages = 0

    for it in range(nt):
        panel_data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        global_low: list[float] = []
        global_high: list[float] = []
        for iq2 in range(nq2):
            for ixb in range(nxb):
                y = values[iq2, ixb, it, :]
                yerr = uncertainties[iq2, ixb, it, :]
                p = parameters[iq2, ixb, it, :]
                valid = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0.0)
                fit_valid = (
                    np.all(np.isfinite(p))
                    and np.isfinite(chi2_ndf[iq2, ixb, it])
                    and points[iq2, ixb, it] >= min_points
                )
                if not fit_valid or not np.any(valid):
                    continue
                fit_curve = p[0] + p[1] * np.cos(radians) + p[2] * np.cos(2.0 * radians)
                panel_data[(iq2, ixb)] = (valid, y, fit_curve)
                finite_fit = fit_curve[np.isfinite(fit_curve)]
                global_low.extend((y[valid] - yerr[valid]).tolist())
                global_high.extend((y[valid] + yerr[valid]).tolist())
                global_low.extend(finite_fit.tolist())
                global_high.extend(finite_fit.tolist())
        if not panel_data:
            continue

        global_ylim = _padded_plot_limits(global_low, global_high, include_zero=True)
        fig, axes = plt.subplots(
            nq2,
            nxb,
            figsize=(max(10.0, 1.8 * nxb), max(7.5, 1.3 * nq2)),
            sharex=True,
            sharey=scale_mode == "global",
            squeeze=False,
        )
        for iq2 in range(nq2):
            for ixb in range(nxb):
                ax = axes[nq2 - 1 - iq2, ixb]
                item = panel_data.get((iq2, ixb))
                if item is None:
                    ax.set_axis_off()
                    continue
                valid, y, fit_curve = item
                yerr = uncertainties[iq2, ixb, it, :]
                ax.errorbar(
                    phi_centers[valid],
                    y[valid],
                    yerr=yerr[valid],
                    fmt="o",
                    capsize=1.0,
                    linewidth=0.7,
                    markersize=2.2,
                    color="#1f78b4",
                )
                ax.plot(phi_curve, fit_curve, color="#d95f02", linewidth=0.9)
                ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.3)
                ax.set_xlim(float(phi_edges[0]), float(phi_edges[-1]))
                if scale_mode == "global":
                    ax.set_ylim(*global_ylim)
                else:
                    finite_fit = fit_curve[np.isfinite(fit_curve)]
                    local_low = (y[valid] - yerr[valid]).tolist() + finite_fit.tolist()
                    local_high = (y[valid] + yerr[valid]).tolist() + finite_fit.tolist()
                    ax.set_ylim(*_padded_plot_limits(local_low, local_high, include_zero=True))
                ax.grid(True, alpha=0.16, linewidth=0.45)
                ax.tick_params(axis="both", labelsize=6, length=2, labelleft=True)
                if scale_mode == "panel":
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
                    ax.yaxis.get_offset_text().set_fontsize(5)
                if iq2 == 0:
                    ax.set_xlabel("phi [deg]", fontsize=7)
                if ixb == 0:
                    ax.set_ylabel(f"Q2 {q2_labels[iq2]}", fontsize=7)
                if iq2 == nq2 - 1:
                    ax.set_title(f"xB {xb_labels[ixb]}", fontsize=7)

        handles = [
            plt.Line2D([0], [0], color="#1f78b4", marker="o", linewidth=0.8,
                       markersize=3, label="reduced cross section"),
            plt.Line2D([0], [0], color="#d95f02", linewidth=1.1,
                       label="A + B cos(phi) + C cos(2phi)"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, fontsize="small")
        fig.suptitle(
            "Reduced cross section vs phi quilt\n"
            f"-t {t_edges[it]:g}-{t_edges[it + 1]:g} GeV^2; "
            "Q2 increases bottom to top; xB increases left to right; "
            + ("independent panel scales" if scale_mode == "panel" else "shared page scale"),
            y=0.987,
        )
        fig.text(
            0.012,
            0.5,
            f"Reduced cross section [{units}]",
            rotation="vertical",
            va="center",
            fontsize=9,
        )
        fig.subplots_adjust(
            left=0.052,
            right=0.995,
            bottom=0.045,
            top=0.91,
            wspace=0.16,
            hspace=0.18,
        )
        pdf.savefig(fig)
        plt.close(fig)
        pages += 1
    return pages


def _padded_plot_limits(
    lower_values: list[float],
    upper_values: list[float],
    *,
    include_zero: bool,
) -> tuple[float, float]:
    finite_lower = np.asarray(lower_values, dtype=float)
    finite_upper = np.asarray(upper_values, dtype=float)
    finite_lower = finite_lower[np.isfinite(finite_lower)]
    finite_upper = finite_upper[np.isfinite(finite_upper)]
    if not finite_lower.size or not finite_upper.size:
        return (-1.0, 1.0)
    lower = float(np.min(finite_lower))
    upper = float(np.max(finite_upper))
    if include_zero:
        lower = min(0.0, lower)
        upper = max(0.0, upper)
    span = upper - lower
    padding = 0.08 * span if span > 0.0 else max(abs(lower), abs(upper), 1.0) * 0.08
    return lower - padding, upper + padding


def _plot_heatmap(
    values: np.ndarray,
    y_name: str,
    x_name: str,
    y_labels: list[str],
    x_labels: list[str],
    title: str,
    colorbar_label: str,
    output: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(x_labels)), max(5, 0.45 * len(y_labels))))
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_acceptance_vs_phi(
    efficiency: np.ndarray,
    truth: np.ndarray,
    acceptance: np.ndarray,
    purity: np.ndarray | None,
    same_bin_efficiency: np.ndarray | None,
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
        "truth_sum,epsilon_rec_sum,mean_positive_A,max_A,"
        "mean_positive_epsilon,max_epsilon,max_epsilon_stat_error,"
        "median_epsilon_relative_stat_error,mean_positive_P,max_P,"
        "mean_positive_E,max_E"
    ]

    with PdfPages(pdf_path) as pdf:
        for iq2 in range(efficiency.shape[0]):
            for ixb in range(efficiency.shape[1]):
                for it in range(efficiency.shape[2]):
                    eff_phi = efficiency[iq2, ixb, it, :]
                    truth_phi = truth[iq2, ixb, it, :]
                    acceptance_phi = acceptance[iq2, ixb, it, :]
                    purity_phi = purity[iq2, ixb, it, :] if purity is not None else None
                    same_bin_phi = (
                        same_bin_efficiency[iq2, ixb, it, :]
                        if same_bin_efficiency is not None else None
                    )
                    populated = truth_phi > 0
                    passing = populated & (eff_phi >= minimum_acceptance)
                    if np.count_nonzero(passing) < min_passing_bins:
                        continue

                    zero = populated & (eff_phi == 0)
                    low = populated & (eff_phi > 0) & (eff_phi < minimum_acceptance)
                    positive = populated & (eff_phi > 0)
                    rec_phi = eff_phi * truth_phi
                    variance = np.divide(
                        eff_phi * (1.0 - eff_phi),
                        truth_phi,
                        out=np.zeros_like(eff_phi),
                        where=truth_phi > 0,
                    )
                    stat_error = np.sqrt(np.maximum(variance, 0.0))
                    relative_stat_error = np.divide(
                        stat_error,
                        eff_phi,
                        out=np.full_like(stat_error, np.nan),
                        where=eff_phi > 0,
                    )
                    positive_acceptance = populated & (acceptance_phi > 0)
                    positive_purity = (
                        populated & (purity_phi > 0)
                        if purity_phi is not None else np.zeros_like(populated, dtype=bool)
                    )
                    positive_same_bin = (
                        populated & (same_bin_phi > 0)
                        if same_bin_phi is not None else np.zeros_like(populated, dtype=bool)
                    )

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
                                float(np.nanmean(acceptance_phi[positive_acceptance]))
                                if np.any(positive_acceptance) else np.nan,
                                float(np.nanmax(acceptance_phi[populated]))
                                if np.any(populated) else np.nan,
                                float(np.nanmean(eff_phi[positive])) if np.any(positive) else np.nan,
                                float(np.nanmax(eff_phi[populated])) if np.any(populated) else np.nan,
                                float(np.nanmax(stat_error[populated])) if np.any(populated) else np.nan,
                                float(np.nanmedian(relative_stat_error[positive]))
                                if np.any(positive) else np.nan,
                                float(np.nanmean(purity_phi[positive_purity]))
                                if purity_phi is not None and np.any(positive_purity) else np.nan,
                                float(np.nanmax(purity_phi[populated]))
                                if purity_phi is not None and np.any(populated) else np.nan,
                                float(np.nanmean(same_bin_phi[positive_same_bin]))
                                if same_bin_phi is not None and np.any(positive_same_bin) else np.nan,
                                float(np.nanmax(same_bin_phi[populated]))
                                if same_bin_phi is not None and np.any(populated) else np.nan,
                            )
                        )
                    )

                    fig, ax = plt.subplots(figsize=(8, 5))
                    if np.any(zero):
                        ax.errorbar(
                            phi_centers[zero],
                            eff_phi[zero],
                            yerr=stat_error[zero],
                            fmt="o",
                            color="#9aa0a6",
                            ecolor="#9aa0a6",
                            elinewidth=0.9,
                            capsize=2,
                            label="zero",
                            zorder=3,
                        )
                    if np.any(low):
                        ax.errorbar(
                            phi_centers[low],
                            eff_phi[low],
                            yerr=stat_error[low],
                            fmt="o",
                            color="#d95f02",
                            ecolor="#d95f02",
                            elinewidth=0.9,
                            capsize=2,
                            label="positive < threshold",
                            zorder=4,
                        )
                    ax.errorbar(
                        phi_centers[passing],
                        eff_phi[passing],
                        yerr=stat_error[passing],
                        fmt="o",
                        color="#1b9e77",
                        ecolor="#1b9e77",
                        elinewidth=0.9,
                        capsize=2,
                        label=">= threshold",
                        zorder=5,
                    )
                    ax.plot(
                        phi_centers[populated],
                        eff_phi[populated],
                        color="#4c78a8",
                        linewidth=1.1,
                        alpha=0.8,
                        label="epsilon_i total IBU efficiency",
                    )
                    ax.plot(
                        phi_centers[populated],
                        acceptance_phi[populated],
                        color="#7b3294",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.85,
                        label="A_i bin-by-bin acceptance",
                    )
                    if purity_phi is not None:
                        ax.plot(
                            phi_centers[populated],
                            purity_phi[populated],
                            color="#f58518",
                            linestyle="-.",
                            linewidth=1.2,
                            alpha=0.9,
                            label="P_i purity",
                        )
                    if same_bin_phi is not None:
                        ax.plot(
                            phi_centers[populated],
                            same_bin_phi[populated],
                            color="#222222",
                            linestyle=":",
                            linewidth=1.4,
                            alpha=0.9,
                            label="E_i same-bin efficiency",
                        )
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
                    ax.set_ylabel("Migration/acceptance diagnostic")
                    ax.set_title(
                        "Acceptance diagnostics vs phi\n"
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
    elif args.command == "radiative-correction":
        command_radiative_correction(args)
    elif args.command == "radiative-correction-plots":
        command_radiative_correction_plots(args)
    elif args.command == "bin-centering":
        command_bin_centering(args)
    elif args.command == "bin-centering-merge":
        command_bin_centering_merge(args)
    elif args.command == "cross-section":
        command_cross_section(args)
    elif args.command == "fit-harmonics":
        command_harmonics(args)
    elif args.command == "harmonic-plots":
        command_harmonic_plots(args)
    elif args.command == "cross-section-plots":
        command_cross_section_plots(args)
    elif args.command == "acceptance-plots":
        command_acceptance_plots(args)
    elif args.command == "response-plots":
        command_response_plots(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
