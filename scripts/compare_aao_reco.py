#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POST_FILENAME = "aao_6.535_eppi0_selected.root"
PROCESSING_FILENAME = "aao_6.535_eppi0_matched.root"

SAMPLES = [
    (11221, "q2_0.7_ep_1.00_eg_0.005", r"$Q^2_g=0.7$, $E'_g=1.00$, $E_{\gamma g}=5$ MeV", "#C94F37", "-"),
    (11222, "q2_0.7_ep_1.00_eg_0.010", r"$Q^2_g=0.7$, $E'_g=1.00$, $E_{\gamma g}=10$ MeV", "#C94F37", "--"),
    (11223, "q2_0.7_ep_1.00_eg_0.015", r"$Q^2_g=0.7$, $E'_g=1.00$, $E_{\gamma g}=15$ MeV", "#C94F37", ":"),
    (11224, "q2_0.9_ep_1.15_eg_0.005", r"$Q^2_g=0.9$, $E'_g=1.15$, $E_{\gamma g}=5$ MeV", "#167D8D", "-"),
    (11225, "q2_0.9_ep_1.15_eg_0.010", r"$Q^2_g=0.9$, $E'_g=1.15$, $E_{\gamma g}=10$ MeV", "#167D8D", "--"),
]

HISTOGRAMS = [
    ("electronP", "electronP", (70, 1.0, 6.6), r"$p_e$ [GeV]"),
    ("electronTheta", "electronTheta * 180.0 / TMath::Pi()", (70, 5.0, 45.0), r"$\theta_e$ [deg]"),
    ("electronPhi", "electronPhi * 180.0 / TMath::Pi()", (72, -180.0, 180.0), r"$\phi_e$ [deg]"),
    ("Q2", "Q2", (70, 0.8, 5.5), r"$Q^2$ [GeV$^2$]"),
    ("W", "W", (70, 1.8, 3.5), r"$W$ [GeV]"),
    ("xB", "xB", (70, 0.0, 0.8), r"$x_B$"),
    ("minusT", "t", (70, 0.0, 3.5), r"$-t$ [GeV$^2$]"),
    ("protonP", "selectedP[1]", (70, 0.3, 3.5), r"$p_p$ [GeV]"),
    ("protonTheta", "selectedTheta[1] * 180.0 / TMath::Pi()", (70, 25.0, 90.0), r"$\theta_p$ [deg]"),
    ("protonPhi", "selectedPhi[1] * 180.0 / TMath::Pi()", (72, -180.0, 180.0), r"$\phi_p$ [deg]"),
    ("pi0P", "pi0_p", (70, 0.0, 5.0), r"$p_{\pi^0}$ [GeV]"),
    ("pi0Theta", "pi0_theta * 180.0 / TMath::Pi()", (70, 0.0, 45.0), r"$\theta_{\pi^0}$ [deg]"),
    ("pi0Phi", "pi0_phi * 180.0 / TMath::Pi()", (72, -180.0, 180.0), r"$\phi_{\pi^0}$ [deg]"),
    ("mgg", "m_gg", (70, 0.04, 0.24), r"$M_{\gamma\gamma}$ [GeV]"),
    ("missingEnergy", "E_miss", (70, -1.5, 1.5), r"$E_{miss}$ [GeV]"),
    ("missingPt", "pT_miss", (70, 0.0, 1.2), r"$p_{T,miss}$ [GeV]"),
    ("missingMass2", "m2_miss", (70, -1.5, 1.5), r"$M^2_{miss}$ [GeV$^2$]"),
    ("pi0Cone", "pi0_thetaX * 180.0 / TMath::Pi()", (70, 0.0, 4.0), r"$\theta_{\pi^0X}$ [deg]"),
    ("trentoPhi", "trentoPhi * 180.0 / TMath::Pi()", (72, -180.0, 180.0), r"$\phi_{Trento}$ [deg]"),
    ("samplingFraction", "(electronEPCAL + electronEECIN + electronEECOUT) / electronP", (70, 0.10, 0.40), r"electron $E_{cal}/p$"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reconstructed eppi0 shapes across AAORAD settings and optional RGK data."
    )
    parser.add_argument("input_dir", type=Path, help="Output directory made by run_aao_osg_comparison.csh")
    parser.add_argument("--data-root", type=Path, help="Post-processed RGK data ROOT file")
    parser.add_argument("--data-processing-root", type=Path, help="Pre-post-processing RGK ROOT file with charge metadata")
    return parser.parse_args()


def normalized_histogram(hist) -> tuple[np.ndarray, np.ndarray]:
    nbins = hist.GetNbinsX()
    edges = np.array([hist.GetXaxis().GetBinLowEdge(i) for i in range(1, nbins + 2)])
    counts = np.array([hist.GetBinContent(i) for i in range(1, nbins + 1)], dtype=float)
    total = counts.sum()
    if total:
        counts /= total
    return edges, counts


def processing_metadata(ROOT, path: Path | None) -> tuple[int, float]:
    if path is None or not path.is_file():
        return 0, math.nan
    root_file = ROOT.TFile.Open(str(path), "READ")
    summary = root_file.Get("Summary")
    total_events = 0
    if summary and summary.GetEntries():
        summary.GetEntry(0)
        total_events = int(summary.TotalEvents)
    charge_object = root_file.Get("AccumulatedCharge")
    charge = float(charge_object.GetVal()) if charge_object else math.nan
    root_file.Close()
    return total_events, charge


def shape_metrics(mc: np.ndarray, data: np.ndarray) -> tuple[float, float]:
    mask = (mc + data) > 0.0
    if not np.any(mask):
        return math.nan, math.nan
    p, q = mc[mask], data[mask]
    midpoint = 0.5 * (p + q)
    positive_p = p > 0.0
    positive_q = q > 0.0
    kl_p = np.sum(p[positive_p] * np.log(p[positive_p] / midpoint[positive_p]))
    kl_q = np.sum(q[positive_q] * np.log(q[positive_q] / midpoint[positive_q]))
    return float(0.5 * (kl_p + kl_q)), float(0.5 * np.sum(np.abs(p - q)))


def load_sample(ROOT, tag: str, label: str, color: str, linestyle: str, path: Path):
    frame = ROOT.RDataFrame("Events", str(path))
    actions = {}
    for name, expression, (bins, low, high), _ in HISTOGRAMS:
        column = name if name == expression else f"plot_{name}"
        node = frame if name == expression else frame.Define(column, expression)
        actions[name] = node.Histo1D((f"h_{tag}_{name}", "", bins, low, high), column)
    count_action = frame.Count()
    histograms = {name: normalized_histogram(action.GetValue()) for name, action in actions.items()}
    return {
        "tag": tag,
        "label": label,
        "color": color,
        "linestyle": linestyle,
        "histograms": histograms,
        "selected": int(count_action.GetValue()),
    }


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    plot_dir = input_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    results = []
    for osg_id, tag, label, color, linestyle in SAMPLES:
        selected_path = input_dir / tag / POST_FILENAME
        if not selected_path.is_file():
            print(f"[SKIP] Missing post-processed file: {selected_path}")
            continue
        sample = load_sample(ROOT, tag, label, color, linestyle, selected_path)
        total, _ = processing_metadata(ROOT, input_dir / tag / PROCESSING_FILENAME)
        sample.update(osg_id=osg_id, total_events=total, charge_nc=math.nan, kind="gemc")
        results.append(sample)

    data_sample = None
    if args.data_root:
        data_path = args.data_root.resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"Data ROOT file does not exist: {data_path}")
        data_sample = load_sample(ROOT, "rgk_data", "RGK data", "black", "-", data_path)
        total, charge = processing_metadata(
            ROOT, args.data_processing_root.resolve() if args.data_processing_root else None
        )
        data_sample.update(osg_id="", total_events=total, charge_nc=charge, kind="data")
        results.insert(0, data_sample)

    if not results:
        raise RuntimeError(f"No {POST_FILENAME} files found below {input_dir}")

    fig, axes = plt.subplots(4, 5, figsize=(19, 14))
    for axis, (name, _, _, xlabel) in zip(axes.flat, HISTOGRAMS):
        for sample in results:
            edges, counts = sample["histograms"][name]
            width = 2.5 if sample["kind"] == "data" else 1.8
            axis.stairs(counts, edges, label=sample["label"], color=sample["color"],
                        linestyle=sample["linestyle"], linewidth=width)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Fraction / bin")
        axis.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, frameon=False, fontsize=9)
    title = r"AAORAD GEMC reconstructed $ep\pi^0$ comparison"
    if data_sample:
        title += " to RGK data"
    fig.suptitle(title, y=0.995, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(plot_dir / "aao_eppi0_reconstructed_distributions.png", dpi=180)
    fig.savefig(plot_dir / "aao_eppi0_reconstructed_distributions.pdf")
    plt.close(fig)

    summary_rows = []
    metric_rows = []
    for sample in results:
        total = sample["total_events"]
        selected = sample["selected"]
        summary_rows.append({
            "sample": sample["tag"],
            "kind": sample["kind"],
            "osg_id": sample["osg_id"],
            "input_events": total,
            "selected_candidates": selected,
            "selection_fraction": selected / total if total else math.nan,
            "accumulated_charge_nc": sample["charge_nc"],
        })
        if data_sample and sample["kind"] == "gemc":
            for name, _, _, _ in HISTOGRAMS:
                _, mc_counts = sample["histograms"][name]
                _, data_counts = data_sample["histograms"][name]
                js_divergence, total_variation = shape_metrics(mc_counts, data_counts)
                metric_rows.append({
                    "sample": sample["tag"],
                    "observable": name,
                    "jensen_shannon_divergence": js_divergence,
                    "total_variation_distance": total_variation,
                })

    summary_path = plot_dir / "aao_eppi0_yield_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    if metric_rows:
        metrics_path = plot_dir / "aao_eppi0_data_shape_metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=metric_rows[0].keys())
            writer.writeheader()
            writer.writerows(metric_rows)
        print(f"[DONE] {metrics_path}")

    print(f"[DONE] {plot_dir / 'aao_eppi0_reconstructed_distributions.png'}")
    print(f"[DONE] {plot_dir / 'aao_eppi0_reconstructed_distributions.pdf'}")
    print(f"[DONE] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
