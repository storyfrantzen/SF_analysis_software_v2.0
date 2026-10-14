#!/usr/bin/env python3

"""Scan helicity-blind BSA binning candidates and recommend a production grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.background_subtraction import estimate_mgg_background
from eppi0.beam_spin import (
    corrected_helicity,
    load_helicity_audit,
    load_polarization_manifest,
    values_by_run,
)
from eppi0.beam_spin_binning import (
    BinningThresholds,
    PreparedBinningSample,
    automatic_bin_counts,
    candidate_binnings,
    evaluate_sample,
    pareto_frontier,
)
from eppi0.binning import AnalysisBinning, from_config
from eppi0.current_efficiency import load_current_efficiency_correction
from eppi0.exclusivity import load_cuts


SAMPLE_FIELDS = (
    "label", "data", "selection_mask", "background_cuts", "helicity_audit_dir",
    "polarization", "run_selection_artifact",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--sample",
        action="append",
        nargs=7,
        required=True,
        metavar=SAMPLE_FIELDS,
        help=(
            "repeatable sample definition: LABEL DATA MASK CUTS AUDIT_DIR "
            "POLARIZATION RUN_SELECTION"
        ),
    )
    parser.add_argument(
        "--exclude-run", action="append", default=[], metavar="LABEL:RUN",
        help="sample-specific run exclusion (repeatable)",
    )
    parser.add_argument(
        "--response-meta", action="append", default=[], metavar="LABEL=PATH",
        help=(
            "optional response_meta.npz used to require "
            "generated/reconstructed support"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--q2-bin-counts", type=int, nargs="+")
    parser.add_argument("--xb-bin-counts", type=int, nargs="+")
    parser.add_argument("--t-bin-counts", type=int, nargs="+")
    parser.add_argument("--phi-bin-counts", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--minimum-production-phi-bins", type=int, default=12)
    parser.add_argument("--minimum-effective-signal-events", type=float, default=400.0)
    parser.add_argument("--maximum-projected-uncertainty", type=float, default=0.10)
    parser.add_argument("--minimum-phi-coverage-fraction", type=float, default=0.80)
    parser.add_argument("--maximum-phi-gap-deg", type=float, default=60.0)
    parser.add_argument("--maximum-background-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-response-coverage-fraction", type=float, default=0.80)
    parser.add_argument("--background-alpha-bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_label_values(
    specifications: list[str], labels: set[str], separator: str
) -> dict[str, str]:
    output: dict[str, str] = {}
    for specification in specifications:
        if separator not in specification:
            raise ValueError(f"expected LABEL{separator}VALUE: {specification}")
        label, value = specification.split(separator, 1)
        if label not in labels:
            raise ValueError(f"unknown sample label {label!r}")
        if label in output:
            raise ValueError(f"duplicate value for sample {label!r}")
        output[label] = value
    return output


def response_support(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "truth_total", "reconstructed_total", "q2_edges", "xb_edges",
            "t_edges", "phi_edges",
        }
        missing = sorted(required.difference(saved.files))
        if missing:
            raise ValueError(f"response metadata is missing {missing}: {path}")
        binning = AnalysisBinning(
            saved["q2_edges"], saved["xb_edges"], saved["t_edges"], saved["phi_edges"]
        )
        truth = np.asarray(saved["truth_total"], dtype=float).reshape(-1)
        accepted = np.asarray(saved["reconstructed_total"], dtype=float).reshape(-1)
    if truth.size != binning.size or accepted.size != binning.size:
        raise ValueError(f"response histogram size disagrees with stored edges: {path}")
    q2, xb, minus_t, phi = np.meshgrid(
        0.5 * (binning.q2_edges[:-1] + binning.q2_edges[1:]),
        0.5 * (binning.xb_edges[:-1] + binning.xb_edges[1:]),
        0.5 * (binning.t_edges[:-1] + binning.t_edges[1:]),
        0.5 * (binning.phi_edges[:-1] + binning.phi_edges[1:]),
        indexing="ij",
    )
    return {
        "response_q2": binning.flatten_values(q2).reshape(-1),
        "response_xb": binning.flatten_values(xb).reshape(-1),
        "response_minus_t": binning.flatten_values(minus_t).reshape(-1),
        "response_phi_rad": np.deg2rad(
            binning.flatten_values(phi).reshape(-1)
        ),
        "response_truth": truth,
        "response_accepted": accepted,
    }


def prepare_sample(
    specification: list[str],
    source_binning: AnalysisBinning,
    excluded_runs: set[int],
    response_path: Path | None,
    *,
    alpha_bootstrap: int,
    seed: int,
) -> tuple[PreparedBinningSample, dict[str, object]]:
    label, *path_text = specification
    data_path, mask_path, cuts_path, audit_dir, polarization_path, run_selection_path = [
        Path(value).resolve() for value in path_text
    ]
    with np.load(data_path, allow_pickle=False) as saved:
        required = {
            "run", "event", "helicity_raw", "rec_Q2", "rec_xB", "rec_minus_t",
            "rec_trento_phi", "rec_proton_detector", "rec_ft_photon_count",
        }
        cuts = load_cuts(str(cuts_path))
        required.update(cuts.variables)
        missing = sorted(required.difference(saved.files))
        if missing:
            raise ValueError(f"{label} compact data is missing: {', '.join(missing)}")
        arrays = {name: np.asarray(saved[name]) for name in required}
        if "rec_selected" in saved.files:
            arrays["rec_selected"] = np.asarray(saved["rec_selected"], dtype=bool)

    size = arrays["run"].size
    selection_mask = np.asarray(np.load(mask_path, allow_pickle=False), dtype=bool)
    if selection_mask.shape != (size,):
        raise ValueError(f"{label} selection mask has shape {selection_mask.shape}")
    rec_flat = source_binning.coordinates_to_flat(
        arrays["rec_Q2"], arrays["rec_xB"], arrays["rec_minus_t"],
        arrays["rec_trento_phi"],
    )
    base = rec_flat >= 0
    if "rec_selected" in arrays:
        base &= arrays["rec_selected"]
    run_selection = load_current_efficiency_correction(run_selection_path)
    runs = np.asarray(arrays["run"], dtype=np.int64)
    base &= run_selection.event_weights(runs) > 0.0
    if excluded_runs:
        base &= ~np.isin(runs, sorted(excluded_runs))
    iq2, ixb, it, _ = source_binning.indices(
        arrays["rec_Q2"], arrays["rec_xB"], arrays["rec_minus_t"],
        arrays["rec_trento_phi"],
    )
    background = estimate_mgg_background(
        cuts=cuts,
        values={name: arrays[name] for name in cuts.variables},
        proton_detector=arrays["rec_proton_detector"],
        ft_photons=arrays["rec_ft_photon_count"],
        iq2=iq2,
        ixb=ixb,
        it=it,
        rec_flat=rec_flat,
        base_mask=base,
        event_weights=np.ones(size, dtype=float),
        number_of_bins=source_binning.size,
        alpha_bootstrap=alpha_bootstrap,
        seed=seed,
    )
    expected_signal = base & selection_mask
    if not np.array_equal(expected_signal, background.signal_region_mask):
        disagreement = int(
            np.count_nonzero(expected_signal != background.signal_region_mask)
        )
        raise ValueError(
            f"{label} selection mask disagrees with its cuts for {disagreement} events"
        )

    audit = load_helicity_audit(audit_dir)
    polarization = load_polarization_manifest(polarization_path)
    helicity, helicity_ok = corrected_helicity(
        runs, arrays["event"], arrays["helicity_raw"], audit
    )
    included_runs = np.intersect1d(audit.runs, polarization.runs)
    active = background.signal_region_mask | background.sideband_mask
    missing_runs = np.setdiff1d(np.unique(runs[active]), included_runs)
    if missing_runs.size:
        raise ValueError(
            f"{label} active events have no usable audit/polarization entry for runs "
            f"{missing_runs.tolist()}; pass --exclude-run {label}:RUN"
        )
    used = active & helicity_ok & np.isin(runs, included_runs)
    used_runs = runs[used]
    qplus = values_by_run(
        used_runs, audit.runs, audit.plus_charge_nc, name="positive-helicity charge"
    )
    qminus = values_by_run(
        used_runs, audit.runs, audit.minus_charge_nc, name="negative-helicity charge"
    )
    polarization_values = values_by_run(
        used_runs, polarization.runs, polarization.values, name="polarization"
    )
    h = helicity[used].astype(float)
    charge_for_state = np.where(h > 0.0, qplus, qminus)
    normalization = 0.5 * (qplus + qminus) / charge_for_state
    net = background.net_event_weights[used]
    denominator_weight = net * normalization
    response = response_support(response_path) if response_path is not None else {}
    prepared = PreparedBinningSample(
        label=label,
        q2=np.asarray(arrays["rec_Q2"], dtype=float)[used],
        xb=np.asarray(arrays["rec_xB"], dtype=float)[used],
        minus_t=np.asarray(arrays["rec_minus_t"], dtype=float)[used],
        phi_rad=np.asarray(arrays["rec_trento_phi"], dtype=float)[used],
        denominator_weight=denominator_weight,
        denominator_variance_weight=denominator_weight**2,
        numerator_variance_weight=(denominator_weight / polarization_values) ** 2,
        signal_region_weight=background.signal_region_mask[used] * normalization,
        background_weight=(
            background.sideband_mask[used] * (-net) * normalization
        ),
        plus_signal=background.signal_region_mask[used] & (h > 0.0),
        minus_signal=background.signal_region_mask[used] & (h < 0.0),
        **response,
    )
    provenance = {
        "label": label,
        "data": file_record(data_path),
        "selection_mask": file_record(mask_path),
        "background_cuts": file_record(cuts_path),
        "helicity_audit_dir": str(audit_dir),
        "polarization": file_record(polarization_path),
        "run_selection_artifact": file_record(run_selection_path),
        "explicitly_excluded_runs": sorted(excluded_runs),
        "active_signal_or_sideband_events": int(np.count_nonzero(active)),
        "usable_signal_or_sideband_events": int(np.count_nonzero(used)),
        "response_meta": file_record(response_path) if response_path else None,
    }
    return prepared, provenance


def file_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": sha256(path)}


def finite_quantile(values: np.ndarray, probability: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, probability)) if finite.size else float("inf")


def render_report(
    rows: list[dict[str, object]],
    recommended: dict[str, object],
    recommended_metrics: dict[str, object],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output.parent.mkdir(parents=True, exist_ok=True)
    good = np.asarray([row["common_good_cells"] for row in rows], dtype=float)
    uncertainty = np.asarray(
        [row["common_projected_uncertainty_median"] for row in rows], dtype=float
    )
    phi_bins = np.asarray([row["phi_bins"] for row in rows], dtype=float)
    finite = np.isfinite(uncertainty)
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
        scatter = axis.scatter(
            good[finite], uncertainty[finite], c=phi_bins[finite], cmap="viridis",
            alpha=0.75, edgecolor="none",
        )
        axis.scatter(
            [recommended["common_good_cells"]],
            [recommended["common_projected_uncertainty_median"]],
            marker="*", s=220, color="#d62728", label="recommended",
        )
        axis.set_xlabel("3D cells passing every sample")
        axis.set_ylabel(r"Median projected $\sigma(A_{LU}^{\sin\phi})$")
        axis.set_title("Helicity-blind beam-spin binning scan")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.colorbar(scatter, ax=axis, label="phi bins")
        pdf.savefig(figure)
        plt.close(figure)

        for label, metrics in recommended_metrics.items():
            good_mask = metrics.good
            counts = np.count_nonzero(good_mask, axis=-1)
            figure, axis = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
            image = axis.imshow(counts.T, origin="lower", aspect="auto", cmap="Blues")
            axis.set_xlabel("Q2-bin index")
            axis.set_ylabel("xB-bin index")
            axis.set_title(
                f"{label}: passing -t bins for {recommended['candidate_id']}"
            )
            figure.colorbar(image, ax=axis, label="passing -t bins")
            pdf.savefig(figure)
            plt.close(figure)

            values = metrics.projected_uncertainty[good_mask]
            figure, axis = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
            axis.hist(values[np.isfinite(values)], bins=30, color="#1f77b4", alpha=0.8)
            axis.set_xlabel(r"Projected $\sigma(A_{LU}^{\sin\phi})$")
            axis.set_ylabel("Passing 3D cells")
            axis.set_title(f"{label}: recommended-grid projected precision")
            axis.grid(alpha=0.2)
            pdf.savefig(figure)
            plt.close(figure)


def main() -> int:
    args = parse_args()
    source_binning = from_config(args.config)
    labels = [sample[0] for sample in args.sample]
    if len(labels) != len(set(labels)):
        raise ValueError("sample labels must be unique")
    label_set = set(labels)
    response_values = parse_label_values(args.response_meta, label_set, "=")
    excluded: dict[str, set[int]] = {label: set() for label in labels}
    for specification in args.exclude_run:
        parsed = parse_label_values([specification], label_set, ":")
        label, run = next(iter(parsed.items()))
        excluded[label].add(int(run))

    thresholds = BinningThresholds(
        minimum_effective_signal_events=args.minimum_effective_signal_events,
        maximum_projected_uncertainty=args.maximum_projected_uncertainty,
        minimum_phi_coverage_fraction=args.minimum_phi_coverage_fraction,
        maximum_phi_gap_deg=args.maximum_phi_gap_deg,
        maximum_background_fraction=args.maximum_background_fraction,
        minimum_response_coverage_fraction=args.minimum_response_coverage_fraction,
    )
    prepared: list[PreparedBinningSample] = []
    provenance: list[dict[str, object]] = []
    for sample_index, specification in enumerate(args.sample):
        label = specification[0]
        item, record = prepare_sample(
            specification,
            source_binning,
            excluded[label],
            Path(response_values[label]).resolve() if label in response_values else None,
            alpha_bootstrap=args.background_alpha_bootstrap,
            seed=args.seed + sample_index,
        )
        prepared.append(item)
        provenance.append(record)
        print(
            f"Prepared {label}: {record['usable_signal_or_sideband_events']} "
            "usable signal/sideband events"
        )

    q2_counts = args.q2_bin_counts or automatic_bin_counts(source_binning.shape[0])
    xb_counts = args.xb_bin_counts or automatic_bin_counts(source_binning.shape[1])
    t_counts = args.t_bin_counts or automatic_bin_counts(source_binning.shape[2])
    candidates = candidate_binnings(
        source_binning, q2_counts, xb_counts, t_counts, args.phi_bin_counts
    )
    rows: list[dict[str, object]] = []
    metrics_by_candidate: dict[str, dict[str, object]] = {}
    binning_by_candidate = dict(candidates)
    for candidate_index, (candidate_id, binning) in enumerate(candidates, start=1):
        sample_metrics = {
            sample.label: evaluate_sample(sample, binning, thresholds)
            for sample in prepared
        }
        common_good = np.logical_and.reduce(
            [metrics.good for metrics in sample_metrics.values()]
        )
        worst_uncertainty = np.maximum.reduce(
            [metrics.projected_uncertainty for metrics in sample_metrics.values()]
        )
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "q2_bins": binning.shape[0],
            "xb_bins": binning.shape[1],
            "t_bins": binning.shape[2],
            "phi_bins": binning.shape[3],
            "total_3d_cells": int(np.prod(binning.shape[:3])),
            "total_4d_bins": binning.size,
            "common_good_cells": int(np.count_nonzero(common_good)),
            "common_good_fraction": float(np.mean(common_good)),
            "common_projected_uncertainty_median": finite_quantile(
                worst_uncertainty[common_good], 0.5
            ),
            "common_projected_uncertainty_q90": finite_quantile(
                worst_uncertainty[common_good], 0.9
            ),
        }
        for label, metrics in sample_metrics.items():
            row[f"{label}_good_cells"] = int(np.count_nonzero(metrics.good))
            row[f"{label}_good_fraction"] = float(np.mean(metrics.good))
            row[f"{label}_projected_uncertainty_median"] = finite_quantile(
                metrics.projected_uncertainty[metrics.good], 0.5
            )
        rows.append(row)
        metrics_by_candidate[candidate_id] = sample_metrics
        print(
            f"[{candidate_index}/{len(candidates)}] {candidate_id}: "
            f"common good={row['common_good_cells']}"
        )

    frontier = pareto_frontier(
        np.asarray([row["common_good_cells"] for row in rows]),
        np.asarray([row["common_projected_uncertainty_median"] for row in rows]),
        np.asarray([row["total_3d_cells"] for row in rows]),
    )
    for row, value in zip(rows, frontier):
        row["pareto_frontier"] = bool(value)
    eligible = [
        row for row in rows
        if int(row["phi_bins"]) >= args.minimum_production_phi_bins
        and int(row["common_good_cells"]) > 0
    ]
    if not eligible:
        raise RuntimeError("no production-eligible candidate has a common passing cell")
    maximum_good = max(int(row["common_good_cells"]) for row in eligible)
    finalists = [row for row in eligible if int(row["common_good_cells"]) == maximum_good]
    recommended = min(
        finalists,
        key=lambda row: (
            float(row["common_projected_uncertainty_median"]),
            int(row["total_4d_bins"]),
        ),
    )
    recommended_id = str(recommended["candidate_id"])
    recommended_binning = binning_by_candidate[recommended_id]
    recommended_metrics = metrics_by_candidate[recommended_id]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scan_csv = args.output_dir / "binning_scan.csv"
    with scan_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    config_document = json.loads(args.config.read_text(encoding="utf-8"))
    config_document["binning"] = {
        "Q2": recommended_binning.q2_edges.tolist(),
        "xB": recommended_binning.xb_edges.tolist(),
        "minus_t": recommended_binning.t_edges.tolist(),
        "phi_deg": recommended_binning.phi_edges.tolist(),
    }
    recommended_config = args.output_dir / "recommended_analysis_config.json"
    recommended_config.write_text(
        json.dumps(config_document, indent=2) + "\n", encoding="utf-8"
    )

    cell_csv = args.output_dir / "recommended_cells.csv"
    with cell_csv.open("w", newline="", encoding="utf-8") as target:
        fields = [
            "sample", "iq2", "ixb", "it", "q2_low", "q2_high", "xb_low",
            "xb_high", "minus_t_low", "minus_t_high", "effective_signal_events",
            "projected_uncertainty", "populated_phi_bins", "maximum_phi_gap_deg",
            "background_fraction", "response_coverage_fraction", "passes",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for label, metrics in recommended_metrics.items():
            for iq2, ixb, it in np.ndindex(metrics.good.shape):
                writer.writerow(
                    {
                        "sample": label, "iq2": iq2, "ixb": ixb, "it": it,
                        "q2_low": recommended_binning.q2_edges[iq2],
                        "q2_high": recommended_binning.q2_edges[iq2 + 1],
                        "xb_low": recommended_binning.xb_edges[ixb],
                        "xb_high": recommended_binning.xb_edges[ixb + 1],
                        "minus_t_low": recommended_binning.t_edges[it],
                        "minus_t_high": recommended_binning.t_edges[it + 1],
                        "effective_signal_events": metrics.effective_signal_events[iq2, ixb, it],
                        "projected_uncertainty": metrics.projected_uncertainty[iq2, ixb, it],
                        "populated_phi_bins": metrics.populated_phi_bins[iq2, ixb, it],
                        "maximum_phi_gap_deg": metrics.maximum_phi_gap_deg[iq2, ixb, it],
                        "background_fraction": metrics.background_fraction[iq2, ixb, it],
                        "response_coverage_fraction": (
                            metrics.response_coverage_fraction[iq2, ixb, it]
                        ),
                        "passes": int(metrics.good[iq2, ixb, it]),
                    }
                )

    summary = {
        "schema_version": 1,
        "method": (
            "helicity-blind scan using charge-balanced background-subtracted "
            "effective statistics and null-asymmetry projected sine-fit precision"
        ),
        "selection_rule": (
            "maximize the number of 3D cells passing every sample among candidates "
            "with at least minimum-production-phi-bins; break ties by projected "
            "precision and then smaller artifact size"
        ),
        "source_analysis_config": file_record(args.config.resolve()),
        "samples": provenance,
        "thresholds": thresholds.__dict__,
        "minimum_production_phi_bins": args.minimum_production_phi_bins,
        "candidate_count": len(rows),
        "recommended": recommended,
        "recommended_edges": config_document["binning"],
        "outputs": {
            "scan_csv": str(scan_csv.resolve()),
            "recommended_cells_csv": str(cell_csv.resolve()),
            "recommended_analysis_config": str(recommended_config.resolve()),
        },
        "limitations": (
            "the optional response requirement measures generated and reconstructed "
            "support, "
            "not migration purity; final binning still requires split-sample, null-harmonic, "
            "and migration validation after the edges are frozen"
        ),
    }
    summary_path = args.output_dir / "binning_scan_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = args.output_dir / "binning_scan.pdf"
    render_report(rows, recommended, recommended_metrics, report_path)
    print(f"Recommended candidate: {recommended_id}")
    print(f"Common passing 3D cells: {recommended['common_good_cells']}")
    print(f"Wrote {scan_csv}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {recommended_config}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
