#!/usr/bin/env python3

"""Extract the exclusive-ep-pi0 beam-spin asymmetry and sin(phi) amplitude."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.background_subtraction import METHOD as BACKGROUND_METHOD
from eppi0.background_subtraction import estimate_mgg_background
from eppi0.beam_spin import (
    extract_beam_spin,
    fit_sine_amplitudes,
    load_helicity_audit,
    load_polarization_manifest,
)
from eppi0.binning import from_config
from eppi0.current_efficiency import load_current_efficiency_correction
from eppi0.exclusivity import load_cuts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="compact selected-data NPZ")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-mask", type=Path, required=True)
    parser.add_argument("--background-cuts", type=Path, required=True)
    parser.add_argument("--helicity-audit-dir", type=Path, required=True)
    parser.add_argument("--polarization", type=Path, required=True)
    parser.add_argument(
        "--run-selection-artifact",
        type=Path,
        required=True,
        help="current-efficiency artifact used only for its zero-weight run selection",
    )
    parser.add_argument(
        "--exclude-run",
        type=int,
        action="append",
        default=[],
        help="run to exclude from this extraction (repeatable; recorded in output)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--background-alpha-bootstrap", type=int, default=200)
    parser.add_argument("--minimum-fit-points", type=int, default=8)
    parser.add_argument("--maximum-chi2-ndf", type=float, default=5.0)
    parser.add_argument("--maximum-amplitude-uncertainty", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mask(path: Path, size: int) -> np.ndarray:
    mask = np.asarray(np.load(path, allow_pickle=False), dtype=bool)
    if mask.shape != (size,):
        raise ValueError(f"selection mask has shape {mask.shape}; expected {(size,)}")
    return mask


def render_plots(
    artifact: dict[str, np.ndarray], output_dir: Path
) -> tuple[int, int]:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir.mkdir(parents=True, exist_ok=True)
    bsa = artifact["beam_spin_asymmetry"]
    error = artifact["beam_spin_statistical_uncertainty"]
    valid = artifact["beam_spin_valid"]
    amplitude = artifact["sin_phi_amplitude"]
    amplitude_error = artifact["sin_phi_amplitude_uncertainty"]
    amplitude_polarization_error = artifact[
        "sin_phi_amplitude_polarization_uncertainty"
    ]
    fit_quality = artifact["sin_phi_fit_quality"]
    q2_edges = artifact["q2_edges"]
    xb_edges = artifact["xb_edges"]
    t_edges = artifact["t_edges"]
    phi_edges = artifact["phi_edges"]
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    pages = 0
    with PdfPages(output_dir / "beam_spin_asymmetry_vs_phi.pdf") as pdf:
        for iq2, ixb, it in np.ndindex(bsa.shape[:-1]):
            keep = valid[iq2, ixb, it]
            if np.count_nonzero(keep) < 2:
                continue
            figure, axis = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
            axis.axhline(0.0, color="0.55", linewidth=0.8)
            axis.errorbar(
                phi[keep], bsa[iq2, ixb, it, keep],
                yerr=error[iq2, ixb, it, keep], fmt="o", markersize=4,
                color="#1f77b4", capsize=2, label="data (stat.)",
            )
            if np.isfinite(amplitude[iq2, ixb, it]):
                dense = np.linspace(0.0, 360.0, 721)
                axis.plot(
                    dense,
                    amplitude[iq2, ixb, it] * np.sin(np.deg2rad(dense)),
                    color="#d62728", linewidth=1.5,
                    label=(
                        rf"$A_{{LU}}^{{\sin\phi}}={amplitude[iq2, ixb, it]:.3f}"
                        rf"\pm{amplitude_error[iq2, ixb, it]:.3f}\ \mathrm{{(stat.)}}"
                        rf"\pm{amplitude_polarization_error[iq2, ixb, it]:.3f}"
                        rf"\ \mathrm{{(pol.)}}$"
                    ),
                )
            axis.set_xlim(0.0, 360.0)
            axis.set_ylim(-1.05, 1.05)
            axis.set_xlabel(r"$\phi$ (deg)")
            axis.set_ylabel(r"$A_{LU}$")
            axis.set_title(
                rf"$Q^2\in[{q2_edges[iq2]:g},{q2_edges[iq2+1]:g}]$, "
                rf"$x_B\in[{xb_edges[ixb]:g},{xb_edges[ixb+1]:g}]$, "
                rf"$-t\in[{t_edges[it]:g},{t_edges[it+1]:g}]$"
            )
            axis.grid(alpha=0.2)
            axis.legend(loc="best", fontsize=8)
            pdf.savefig(figure)
            plt.close(figure)
            pages += 1

    coefficient_pages = 0
    t = 0.5 * (t_edges[:-1] + t_edges[1:])
    with PdfPages(output_dir / "sin_phi_amplitude_vs_t.pdf") as pdf:
        for iq2, ixb in np.ndindex(amplitude.shape[:2]):
            keep = fit_quality[iq2, ixb]
            if not np.any(keep):
                continue
            figure, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
            axis.axhline(0.0, color="0.55", linewidth=0.8)
            axis.errorbar(
                t[keep], amplitude[iq2, ixb, keep],
                yerr=amplitude_error[iq2, ixb, keep], fmt="o", capsize=2,
                color="#1f77b4",
            )
            axis.set_ylim(-1.05, 1.05)
            axis.set_xlabel(r"$-t$ (GeV$^2$)")
            axis.set_ylabel(r"$A_{LU}^{\sin\phi}$")
            axis.set_title(
                rf"$Q^2\in[{q2_edges[iq2]:g},{q2_edges[iq2+1]:g}]$, "
                rf"$x_B\in[{xb_edges[ixb]:g},{xb_edges[ixb+1]:g}]$"
            )
            axis.grid(alpha=0.2)
            pdf.savefig(figure)
            plt.close(figure)
            coefficient_pages += 1
    return pages, coefficient_pages


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    config_path = args.config.resolve()
    selection_mask_path = args.selection_mask.resolve()
    cuts_path = args.background_cuts.resolve()
    audit_dir = args.helicity_audit_dir.resolve()
    polarization_path = args.polarization.resolve()
    run_selection_path = args.run_selection_artifact.resolve()
    data = np.load(data_path, allow_pickle=False)
    required = {
        "run", "event", "helicity_raw", "rec_Q2", "rec_xB", "rec_minus_t",
        "rec_trento_phi", "rec_proton_detector", "rec_ft_photon_count",
    }
    cuts = load_cuts(str(cuts_path))
    required.update(cuts.variables)
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(
            "compact data is missing beam-spin inputs: " + ", ".join(missing)
            + "; rerun export_selected_data.py with the current repository"
        )

    binning = from_config(config_path)
    rec_flat = binning.coordinates_to_flat(
        data["rec_Q2"], data["rec_xB"], data["rec_minus_t"], data["rec_trento_phi"]
    )
    base = rec_flat >= 0
    if "rec_selected" in data.files:
        base &= np.asarray(data["rec_selected"], dtype=bool)
    run_selection = load_current_efficiency_correction(run_selection_path)
    event_runs = np.asarray(data["run"], dtype=np.int64)
    run_weights = run_selection.event_weights(event_runs)
    base &= run_weights > 0.0
    explicit_excluded_runs = np.asarray(sorted(set(args.exclude_run)), dtype=np.int64)
    if explicit_excluded_runs.size:
        base &= ~np.isin(event_runs, explicit_excluded_runs)
    selection_mask = load_mask(selection_mask_path, base.size)
    iq2, ixb, it, _ = binning.indices(
        data["rec_Q2"], data["rec_xB"], data["rec_minus_t"], data["rec_trento_phi"]
    )
    background = estimate_mgg_background(
        cuts=cuts,
        values={name: data[name] for name in cuts.variables},
        proton_detector=data["rec_proton_detector"],
        ft_photons=data["rec_ft_photon_count"],
        iq2=iq2,
        ixb=ixb,
        it=it,
        rec_flat=rec_flat,
        base_mask=base,
        event_weights=np.ones(base.size, dtype=float),
        number_of_bins=binning.size,
        alpha_bootstrap=args.background_alpha_bootstrap,
        seed=args.seed,
    )
    expected_signal = base & selection_mask
    if not np.array_equal(expected_signal, background.signal_region_mask):
        disagreement = int(np.count_nonzero(expected_signal != background.signal_region_mask))
        raise ValueError(
            "selection mask disagrees with the background-cut signal region for "
            f"{disagreement} events"
        )
    active = background.signal_region_mask | background.sideband_mask
    audit = load_helicity_audit(audit_dir)
    with (audit_dir / "audit_summary.json").open(encoding="utf-8") as source:
        audit_summary = json.load(source)
    polarization = load_polarization_manifest(polarization_path)
    result = extract_beam_spin(
        flat_bins=rec_flat,
        event_runs=event_runs,
        event_numbers=data["event"],
        raw_helicity=data["helicity_raw"],
        net_event_weights=background.net_event_weights,
        active_events=active,
        audit=audit,
        polarization=polarization,
        number_of_bins=binning.size,
    )
    bsa = binning.unflatten(result.asymmetry)
    statistical = binning.unflatten(result.statistical_uncertainty)
    polarization_uncertainty = binning.unflatten(result.polarization_uncertainty)
    valid = binning.unflatten(result.valid)
    phi_centers = 0.5 * (binning.phi_edges[:-1] + binning.phi_edges[1:])
    fit = fit_sine_amplitudes(
        bsa,
        statistical,
        valid,
        phi_centers,
        polarization_shifts=binning.unflatten(
            result.polarization_asymmetry_shifts
        ),
        minimum_points=args.minimum_fit_points,
        maximum_chi2_ndf=args.maximum_chi2_ndf,
        maximum_uncertainty=args.maximum_amplitude_uncertainty,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "beam_spin_asymmetry.npz"
    payload = dict(
        schema_version=np.asarray(1),
        method=np.asarray("charge-balanced QADB-corrected helicity asymmetry"),
        fit_model=np.asarray("A_LU(phi) = A_sin_phi * sin(phi)"),
        beam_spin_asymmetry=bsa,
        beam_spin_statistical_uncertainty=statistical,
        beam_spin_polarization_uncertainty=polarization_uncertainty,
        polarization_period_labels=result.polarization_period_labels,
        polarization_asymmetry_shifts=binning.unflatten(
            result.polarization_asymmetry_shifts
        ),
        beam_spin_valid=valid,
        asymmetry_numerator=binning.unflatten(result.numerator),
        asymmetry_denominator=binning.unflatten(result.denominator),
        plus_background_subtracted_yield=binning.unflatten(result.plus_yield),
        minus_background_subtracted_yield=binning.unflatten(result.minus_yield),
        plus_charge_balanced_yield=binning.unflatten(result.plus_charge_balanced_yield),
        minus_charge_balanced_yield=binning.unflatten(result.minus_charge_balanced_yield),
        plus_event_count=binning.unflatten(result.plus_event_count),
        minus_event_count=binning.unflatten(result.minus_event_count),
        event_count=binning.unflatten(result.event_count),
        sin_phi_amplitude=fit.amplitude,
        sin_phi_amplitude_uncertainty=fit.uncertainty,
        sin_phi_amplitude_polarization_uncertainty=fit.polarization_uncertainty,
        sin_phi_fit_chi2=fit.chi2,
        sin_phi_fit_ndof=fit.ndof,
        sin_phi_fit_point_count=fit.point_count,
        sin_phi_fit_valid=fit.valid,
        sin_phi_fit_quality=fit.quality,
        q2_edges=binning.q2_edges,
        xb_edges=binning.xb_edges,
        t_edges=binning.t_edges,
        phi_edges=binning.phi_edges,
        excluded_runs=np.asarray(
            sorted(set(run_selection.excluded_runs).union(args.exclude_run)),
            dtype=np.int64,
        ),
        explicitly_excluded_runs=explicit_excluded_runs,
        background_group_ids=background.group_ids,
        background_alpha=background.alpha,
        background_alpha_uncertainty=background.alpha_uncertainty,
        background_method=np.asarray(BACKGROUND_METHOD),
        data_artifact=np.asarray(str(data_path)),
        analysis_config_artifact=np.asarray(str(config_path)),
        selection_mask_artifact=np.asarray(str(selection_mask_path)),
        background_cuts_artifact=np.asarray(str(cuts_path)),
        helicity_audit_directory=np.asarray(str(audit_dir)),
        helicity_audit_summary_json=np.asarray(
            json.dumps(audit_summary, sort_keys=True)
        ),
        polarization_manifest=np.asarray(str(polarization_path)),
        run_selection_artifact=np.asarray(str(run_selection_path)),
        polarization_manifest_json=np.asarray(json.dumps(polarization.payload, sort_keys=True)),
    )
    np.savez_compressed(artifact_path, **payload)

    csv_path = args.output_dir / "sin_phi_amplitudes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(
            ["iq2", "ixb", "it", "q2_low", "q2_high", "xb_low", "xb_high",
             "minus_t_low", "minus_t_high", "amplitude", "statistical_uncertainty",
             "polarization_uncertainty",
             "chi2", "ndof", "point_count", "fit_valid", "quality"]
        )
        for iq2_index, ixb_index, it_index in np.ndindex(fit.amplitude.shape):
            writer.writerow(
                [iq2_index, ixb_index, it_index,
                 binning.q2_edges[iq2_index], binning.q2_edges[iq2_index + 1],
                 binning.xb_edges[ixb_index], binning.xb_edges[ixb_index + 1],
                 binning.t_edges[it_index], binning.t_edges[it_index + 1],
                 fit.amplitude[iq2_index, ixb_index, it_index],
                 fit.uncertainty[iq2_index, ixb_index, it_index],
                 fit.polarization_uncertainty[iq2_index, ixb_index, it_index],
                 fit.chi2[iq2_index, ixb_index, it_index],
                 fit.ndof[iq2_index, ixb_index, it_index],
                 fit.point_count[iq2_index, ixb_index, it_index],
                 int(fit.valid[iq2_index, ixb_index, it_index]),
                 int(fit.quality[iq2_index, ixb_index, it_index])]
            )

    pages, coefficient_pages = render_plots(payload, args.output_dir / "diagnostics")
    summary = {
        "schema_version": 1,
        "observable": "A_LU and its sin(phi) amplitude",
        "definition": (
            "charge-balanced background-subtracted yields with physical helicity "
            "h = raw helicity * QADB CorrectHelicitySign and positive polarization magnitude"
        ),
        "data": {"path": str(data_path), "sha256": sha256(data_path)},
        "analysis_config": {
            "path": str(config_path), "sha256": sha256(config_path)
        },
        "selection_mask": {
            "path": str(selection_mask_path), "sha256": sha256(selection_mask_path)
        },
        "background_cuts": {"path": str(cuts_path), "sha256": sha256(cuts_path)},
        "helicity_audit": {
            "directory": str(audit_dir),
            "run_charge_sha256": sha256(audit_dir / "run_helicity_charge.tsv"),
            "sign_intervals_sha256": sha256(
                audit_dir / "helicity_sign_intervals.tsv"
            ),
            "summary_sha256": sha256(audit_dir / "audit_summary.json"),
            "summary": audit_summary,
        },
        "polarization": {
            "path": str(polarization_path), "sha256": sha256(polarization_path)
        },
        "run_selection": {
            "path": str(run_selection_path), "sha256": sha256(run_selection_path),
            "excluded_runs": list(run_selection.excluded_runs),
            "explicitly_excluded_runs": explicit_excluded_runs.tolist(),
        },
        "used_signal_or_sideband_events": result.used_event_count,
        "rejected_unknown_helicity_events": result.rejected_helicity_event_count,
        "valid_4d_bins": int(np.count_nonzero(valid)),
        "numerically_successful_sine_fits": int(np.count_nonzero(fit.valid)),
        "production_quality_sine_fits": int(np.count_nonzero(fit.quality)),
        "phi_plot_pages": pages,
        "amplitude_plot_pages": coefficient_pages,
        "uncertainty_limitation": (
            "statistical errors include signal and sideband event counting; polarization "
            "uncertainty is stored separately; sideband-transfer, detector, selection, and "
            "other campaign systematic covariances remain to be evaluated"
        ),
    }
    summary_path = args.output_dir / "beam_spin_asymmetry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Signal-region events: {int(np.count_nonzero(background.signal_region_mask))}")
    print(f"Sideband events: {int(np.count_nonzero(background.sideband_mask))}")
    print(f"QADB-helicity accepted signal/sideband events: {result.used_event_count}")
    print(f"Unknown/invalid helicity events rejected: {result.rejected_helicity_event_count}")
    print(f"Valid BSA bins: {int(np.count_nonzero(valid))} / {binning.size}")
    print(f"Numerically successful sine fits: {int(np.count_nonzero(fit.valid))}")
    print(f"Production-quality sine fits: {int(np.count_nonzero(fit.quality))}")
    print(f"Wrote {artifact_path}")
    print(f"Wrote diagnostics under {args.output_dir / 'diagnostics'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
