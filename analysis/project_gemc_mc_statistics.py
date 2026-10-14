#!/usr/bin/env python3
"""Project GEMC Monte-Carlo statistics from Valery's 50-float ntuples.

The input format is the little-endian float32 record written by
``production/gemc_check/hipo_to_ntuple.groovy`` in ``exclurad_py``.  This is a
planning diagnostic: a reconstructed event passes when the generated electron,
proton, and two pi0-decay photons all have correctly identified MC::GenMatch
partners and the generated/reconstructed event lies in the configured analysis
phase space.  It does not replace the maintained post-processing selection used
to build a production response.

For every sample and requested analysis binning the script writes maps of

* truth occupancy and matched-particle efficiency;
* supported phi-bin fraction;
* the total generated-event count required to give a chosen counting precision
  in a requested fraction of phi bins; and
* three-dimensional-cell coverage versus projected generated sample size.

Radiative samples may apply an offline ``v = MX2(e'p)-m_pi0^2`` cut.  The
projected total still refers to events generated before that cut, matching the
production quantity the user controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.binning import AnalysisBinning, from_config  # noqa: E402


NFLD = 50
RECORD_BYTES = NFLD * np.dtype("<f4").itemsize
ELECTRON_MASS = 0.00051099895
PROTON_MASS = 0.9382720813
PI0_MASS = 0.1349768


@dataclass(frozen=True)
class SampleSpec:
    label: str
    path: Path
    v_max: float | None


@dataclass
class Accumulation:
    raw_events: int
    events_after_vcut: int
    truth: np.ndarray
    accepted_truth: np.ndarray
    reconstructed: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        action="append",
        nargs=3,
        required=True,
        metavar=("LABEL", "F32_PATH", "V_MAX"),
        help="repeatable sample; use 'none' for V_MAX when no offline cut applies",
    )
    parser.add_argument(
        "--binning",
        action="append",
        nargs=2,
        required=True,
        metavar=("LABEL", "CONFIG_JSON"),
        help="repeatable analysis binning/configuration",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beam-energy", type=float, default=10.604)
    parser.add_argument("--minimum-acceptance", type=float, default=0.005)
    parser.add_argument("--minimum-phi-coverage", type=float, default=0.80)
    parser.add_argument(
        "--precision-targets", type=float, nargs="+", default=(0.10, 0.05)
    )
    parser.add_argument(
        "--projected-events",
        type=float,
        nargs="+",
        default=(1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8),
    )
    return parser.parse_args()


def parse_samples(raw: list[list[str]]) -> list[SampleSpec]:
    result: list[SampleSpec] = []
    labels: set[str] = set()
    for label, path, v_text in raw:
        if label in labels:
            raise ValueError(f"duplicate sample label: {label}")
        labels.add(label)
        v_max = None if v_text.lower() in {"none", "null", "off"} else float(v_text)
        if v_max is not None and not np.isfinite(v_max):
            raise ValueError(f"non-finite v cut for {label}")
        result.append(SampleSpec(label, Path(path).resolve(), v_max))
    return result


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(item for item in path.rglob("*.f32") if item.is_file())
    else:
        raise FileNotFoundError(path)
    if not files:
        raise ValueError(f"no .f32 inputs under {path}")
    malformed = [item for item in files if item.stat().st_size % RECORD_BYTES]
    if malformed:
        raise ValueError(f"incomplete 50-float record in {malformed[0]}")
    return files


def four(momentum: np.ndarray, mass: float) -> np.ndarray:
    p = np.asarray(momentum, dtype=np.float64)
    energy = np.sqrt(np.sum(p * p, axis=1) + mass * mass)
    return np.column_stack((p, energy))


def unit(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1)
    return np.divide(
        vectors,
        norm[:, None],
        out=np.full_like(vectors, np.nan, dtype=np.float64),
        where=norm[:, None] > 0.0,
    )


def kinematics(
    electron3: np.ndarray, proton3: np.ndarray, beam_energy: float
) -> dict[str, np.ndarray]:
    electron = four(electron3, ELECTRON_MASS)
    proton = four(proton3, PROTON_MASS)
    beam_pz = math.sqrt(beam_energy * beam_energy - ELECTRON_MASS * ELECTRON_MASS)
    q_energy = beam_energy - electron[:, 3]
    q3 = np.column_stack((-electron[:, 0], -electron[:, 1], beam_pz - electron[:, 2]))
    q2 = np.sum(q3 * q3, axis=1) - q_energy * q_energy
    xb = np.divide(
        q2,
        2.0 * PROTON_MASS * q_energy,
        out=np.full_like(q2, np.nan),
        where=q_energy != 0.0,
    )
    w2 = PROTON_MASS * PROTON_MASS + 2.0 * PROTON_MASS * q_energy - q2
    w = np.sqrt(np.maximum(w2, 0.0))
    delta_energy = PROTON_MASS - proton[:, 3]
    minus_t = np.sum(proton[:, :3] ** 2, axis=1) - delta_energy * delta_energy

    beam3 = np.zeros_like(electron3, dtype=np.float64)
    beam3[:, 2] = beam_pz
    lepton_normal = unit(np.cross(beam3, electron[:, :3]))
    hadron_normal = unit(np.cross(proton[:, :3], q3))
    cosine = np.sum(lepton_normal * hadron_normal, axis=1)
    sine = np.sum(unit(q3) * np.cross(lepton_normal, hadron_normal), axis=1)
    trento_phi = np.arctan2(sine, cosine)

    missing_energy = beam_energy + PROTON_MASS - electron[:, 3] - proton[:, 3]
    missing3 = np.column_stack(
        (-electron[:, 0] - proton[:, 0],
         -electron[:, 1] - proton[:, 1],
         beam_pz - electron[:, 2] - proton[:, 2])
    )
    missing_mass2 = missing_energy * missing_energy - np.sum(missing3 * missing3, axis=1)
    return {
        "Q2": q2,
        "xB": xb,
        "minus_t": minus_t,
        "phi": trento_phi,
        "W": w,
        "electron_p": np.linalg.norm(electron3, axis=1),
        "v": missing_mass2 - PI0_MASS * PI0_MASS,
    }


def identified_mask(records: np.ndarray) -> np.ndarray:
    expected_pid = (11, 2212, 22, 22)
    expected_charge = (-1, 1, 0, 0)
    selected = np.ones(records.shape[0], dtype=bool)
    for index, (pid, charge) in enumerate(zip(expected_pid, expected_charge)):
        base = 13 + 8 * index
        momentum = records[:, base + 5 : base + 8]
        selected &= records[:, base] > 0.0
        selected &= records[:, base + 1] == pid
        selected &= records[:, base + 2] == charge
        selected &= np.linalg.norm(momentum, axis=1) > 0.0
    return selected


def rec_momenta(records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    electron_base = 13
    proton_base = 21
    return (
        np.asarray(records[:, electron_base + 5 : electron_base + 8], dtype=np.float64),
        np.asarray(records[:, proton_base + 5 : proton_base + 8], dtype=np.float64),
    )


def phase_mask(values: dict[str, np.ndarray], config: dict) -> np.ndarray:
    phase = config.get("phase_space", {})
    result = np.ones(values["Q2"].shape, dtype=bool)
    if "Q2_min" in phase:
        result &= values["Q2"] >= float(phase["Q2_min"])
    if "W_min" in phase:
        result &= values["W"] >= float(phase["W_min"])
    if "electron_p_min" in phase:
        result &= values["electron_p"] >= float(phase["electron_p_min"])
    return result


def accumulate(
    spec: SampleSpec,
    binnings: dict[str, tuple[AnalysisBinning, dict]],
    beam_energy: float,
) -> dict[str, Accumulation]:
    outputs = {
        name: Accumulation(
            raw_events=0,
            events_after_vcut=0,
            truth=np.zeros(binning.size, dtype=float),
            accepted_truth=np.zeros(binning.size, dtype=float),
            reconstructed=np.zeros(binning.size, dtype=float),
        )
        for name, (binning, _) in binnings.items()
    }
    files = input_files(spec.path)
    for file_index, path in enumerate(files, start=1):
        raw = np.memmap(path, dtype="<f4", mode="r")
        records = raw.reshape(-1, NFLD)
        generated = kinematics(records[:, 1:4], records[:, 4:7], beam_energy)
        reconstructed = kinematics(*rec_momenta(records), beam_energy)
        identified = identified_mask(records)
        vmask = (
            np.ones(records.shape[0], dtype=bool)
            if spec.v_max is None
            else generated["v"] < spec.v_max
        )
        for name, (binning, config) in binnings.items():
            output = outputs[name]
            output.raw_events += records.shape[0]
            output.events_after_vcut += int(np.count_nonzero(vmask))
            truth_flat = binning.coordinates_to_flat(
                generated["Q2"], generated["xB"], generated["minus_t"], generated["phi"]
            )
            rec_flat = binning.coordinates_to_flat(
                reconstructed["Q2"], reconstructed["xB"],
                reconstructed["minus_t"], reconstructed["phi"]
            )
            truth_inside = vmask & phase_mask(generated, config) & (truth_flat >= 0)
            rec_inside = identified & vmask & phase_mask(reconstructed, config) & (rec_flat >= 0)
            accepted = truth_inside & rec_inside
            output.truth += np.bincount(
                truth_flat[truth_inside], minlength=binning.size
            )
            output.accepted_truth += np.bincount(
                truth_flat[accepted], minlength=binning.size
            )
            output.reconstructed += np.bincount(
                rec_flat[rec_inside], minlength=binning.size
            )
        del records, raw
        if file_index % 25 == 0 or file_index == len(files):
            print(f"{spec.label}: {file_index}/{len(files)} files")
    return outputs


def bin_metrics(
    accumulation: Accumulation,
    binning: AnalysisBinning,
    minimum_acceptance: float,
) -> dict[str, np.ndarray]:
    truth = binning.unflatten(accumulation.truth)
    accepted = binning.unflatten(accumulation.accepted_truth)
    acceptance = np.divide(
        accepted, truth, out=np.full_like(accepted, np.nan), where=truth > 0.0
    )
    relative_counting = np.divide(
        1.0,
        np.sqrt(accepted),
        out=np.full_like(accepted, np.inf),
        where=accepted > 0.0,
    )
    relative_acceptance = np.sqrt(
        np.divide(
            np.maximum(1.0 - acceptance, 0.0),
            truth * acceptance,
            out=np.full_like(acceptance, np.inf),
            where=(truth > 0.0) & (acceptance > 0.0),
        )
    )
    supported = (truth > 0.0) & (accepted > 0.0) & (acceptance >= minimum_acceptance)
    return {
        "truth": truth,
        "accepted": accepted,
        "acceptance": acceptance,
        "relative_counting": relative_counting,
        "relative_acceptance": relative_acceptance,
        "supported": supported,
    }


def required_events_by_cell(
    truth: np.ndarray,
    accepted: np.ndarray,
    supported: np.ndarray,
    raw_events: int,
    precision: float,
    coverage_fraction: float,
) -> np.ndarray:
    required_phi = int(math.ceil(coverage_fraction * accepted.shape[-1]))
    required = np.divide(
        float(raw_events),
        precision * precision * accepted,
        out=np.full_like(accepted, np.inf, dtype=float),
        where=accepted > 0.0,
    )
    required = np.where(supported, required, np.inf)
    ordered = np.sort(required, axis=-1)
    result = ordered[..., required_phi - 1]
    # A cell without enough truth-populated phi bins is outside the current
    # generator support rather than an acceptance failure.  Keep it out of the
    # coverage denominator with NaN.  A truth-supported cell with too few
    # accepted bins remains +inf and therefore counts as a failure.
    truth_eligible = np.count_nonzero(truth > 0.0, axis=-1) >= required_phi
    return np.where(truth_eligible, result, np.nan)


def coverage_at_events(
    required: np.ndarray, projected_events: np.ndarray
) -> np.ndarray:
    eligible_cells = ~np.isnan(required)
    denominator = int(np.count_nonzero(eligible_cells))
    if denominator == 0:
        return np.zeros(projected_events.shape, dtype=float)
    return np.asarray(
        [np.count_nonzero(eligible_cells & (required <= total)) / denominator
         for total in projected_events],
        dtype=float,
    )


def events_for_cell_coverage(required: np.ndarray, coverage_fraction: float) -> float:
    """Return the event count at which the requested eligible-cell fraction passes."""
    if not 0.0 < coverage_fraction <= 1.0:
        raise ValueError("coverage fraction must lie in (0, 1]")
    eligible = np.asarray(required, dtype=float)[~np.isnan(required)]
    if eligible.size == 0:
        return float("nan")
    ordered = np.sort(eligible)
    index = int(math.ceil(coverage_fraction * ordered.size)) - 1
    return float(ordered[index])


def draw_t_maps(
    pdf: PdfPages,
    values: np.ndarray,
    binning: AnalysisBinning,
    title: str,
    colorbar_label: str,
    *,
    log: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    nt = values.shape[2]
    columns = min(4, nt)
    rows = int(math.ceil(nt / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.7 * rows), squeeze=False)
    finite_values = values[np.isfinite(values) & (values > 0.0 if log else np.ones(values.shape, dtype=bool))]
    if log and finite_values.size:
        from matplotlib.colors import LogNorm

        norm = LogNorm(
            vmin=vmin or max(float(np.nanpercentile(finite_values, 2.0)), np.finfo(float).tiny),
            vmax=vmax or float(np.nanpercentile(finite_values, 98.0)),
        )
    else:
        norm = None
        if vmin is None and finite_values.size:
            vmin = float(np.nanpercentile(finite_values, 2.0))
        if vmax is None and finite_values.size:
            vmax = float(np.nanpercentile(finite_values, 98.0))
    image = None
    for it, axis in enumerate(axes.flat):
        if it >= nt:
            axis.axis("off")
            continue
        panel = np.asarray(values[:, :, it], dtype=float).T
        image = axis.pcolormesh(
            binning.q2_edges,
            binning.xb_edges,
            np.ma.masked_invalid(panel),
            shading="auto",
            cmap="viridis",
            norm=norm,
            vmin=None if norm else vmin,
            vmax=None if norm else vmax,
        )
        axis.set_xlabel(r"$Q^2$ [GeV$^2$]")
        axis.set_ylabel(r"$x_B$")
        axis.set_title(
            rf"${binning.t_edges[it]:g} < -t < {binning.t_edges[it + 1]:g}$ GeV$^2$"
        )
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=colorbar_label, shrink=0.88)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.90, wspace=0.30, hspace=0.35, right=0.90)
    pdf.savefig(fig)
    plt.close(fig)


def render_sample(
    pdf: PdfPages,
    sample_label: str,
    binning_label: str,
    accumulation: Accumulation,
    binning: AnalysisBinning,
    metrics: dict[str, np.ndarray],
    precision_targets: list[float],
    projected_events: np.ndarray,
    minimum_phi_coverage: float,
) -> tuple[list[dict[str, object]], dict[str, object], dict[float, np.ndarray]]:
    truth3 = np.sum(metrics["truth"], axis=-1)
    accepted3 = np.sum(metrics["accepted"], axis=-1)
    acceptance3 = np.divide(
        accepted3, truth3, out=np.full_like(accepted3, np.nan), where=truth3 > 0.0
    )
    phi_fraction = np.mean(metrics["supported"], axis=-1)

    prefix = f"{sample_label} — {binning_label} {binning.shape}"
    draw_t_maps(pdf, truth3, binning, f"{prefix}: truth occupancy", "events", log=True)
    draw_t_maps(
        pdf, acceptance3, binning, f"{prefix}: matched-particle efficiency",
        "accepted / generated", vmin=0.0, vmax=1.0,
    )
    draw_t_maps(
        pdf, phi_fraction, binning, f"{prefix}: supported phi-bin fraction",
        "supported fraction", vmin=0.0, vmax=1.0,
    )

    rows: list[dict[str, object]] = []
    required_by_precision: dict[float, np.ndarray] = {}
    for precision in precision_targets:
        required = required_events_by_cell(
            metrics["truth"], metrics["accepted"], metrics["supported"],
            accumulation.raw_events,
            precision, minimum_phi_coverage,
        )
        required_by_precision[precision] = required
        draw_t_maps(
            pdf,
            required / 1.0e6,
            binning,
            f"{prefix}: generated events required for {100 * precision:g}% counting precision\n"
            f"in at least {100 * minimum_phi_coverage:g}% of phi bins",
            r"required generated events [$10^6$]",
            log=True,
        )

    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    for precision, required in required_by_precision.items():
        axis.plot(
            projected_events / 1.0e6,
            100.0 * coverage_at_events(required, projected_events),
            marker="o",
            label=f"{100 * precision:g}% per-phi-bin precision",
        )
    axis.set_xscale("log")
    axis.set_ylim(0.0, 102.0)
    axis.set_xlabel(r"projected generated events [$10^6$]")
    axis.set_ylabel("eligible 3D-cell coverage [%]")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    axis.set_title(prefix)
    pdf.savefig(fig)
    plt.close(fig)

    for iq2, ixb, it in np.ndindex(truth3.shape):
        row: dict[str, object] = {
            "sample": sample_label,
            "binning": binning_label,
            "iq2": iq2,
            "ixb": ixb,
            "it": it,
            "q2_low": binning.q2_edges[iq2],
            "q2_high": binning.q2_edges[iq2 + 1],
            "xb_low": binning.xb_edges[ixb],
            "xb_high": binning.xb_edges[ixb + 1],
            "minus_t_low": binning.t_edges[it],
            "minus_t_high": binning.t_edges[it + 1],
            "truth_events": truth3[iq2, ixb, it],
            "accepted_truth_events": accepted3[iq2, ixb, it],
            "integrated_efficiency": acceptance3[iq2, ixb, it],
            "supported_phi_fraction": phi_fraction[iq2, ixb, it],
        }
        for precision, required in required_by_precision.items():
            row[f"required_generated_events_{precision:g}"] = required[iq2, ixb, it]
        rows.append(row)

    summary: dict[str, object] = {
        "sample": sample_label,
        "binning": binning_label,
        "shape": list(binning.shape),
        "raw_generated_events": accumulation.raw_events,
        "events_after_vcut": accumulation.events_after_vcut,
        "truth_events_in_analysis_box": float(np.sum(metrics["truth"])),
        "accepted_truth_events": float(np.sum(metrics["accepted"])),
        "global_efficiency": (
            float(np.sum(metrics["accepted"]) / np.sum(metrics["truth"]))
            if np.sum(metrics["truth"]) > 0.0 else 0.0
        ),
        "truth_populated_4d_bins": int(np.count_nonzero(metrics["truth"] > 0.0)),
        "supported_4d_bins": int(np.count_nonzero(metrics["supported"])),
    }
    for precision, required in required_by_precision.items():
        finite = required[np.isfinite(required)]
        summary[f"required_events_{precision:g}_median"] = (
            float(np.median(finite)) if finite.size else None
        )
        summary[f"required_events_{precision:g}_q90"] = (
            float(np.quantile(finite, 0.90)) if finite.size else None
        )
        for coverage in (0.50, 0.80, 0.90, 0.95):
            value = events_for_cell_coverage(required, coverage)
            summary[
                f"required_events_{precision:g}_for_{coverage:g}_cell_coverage"
            ] = value if np.isfinite(value) else None
    return rows, summary, required_by_precision


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.minimum_acceptance <= 1.0:
        raise ValueError("--minimum-acceptance must lie in (0, 1]")
    if not 0.0 < args.minimum_phi_coverage <= 1.0:
        raise ValueError("--minimum-phi-coverage must lie in (0, 1]")
    precision_targets = [float(value) for value in args.precision_targets]
    if any(not 0.0 < value < 1.0 for value in precision_targets):
        raise ValueError("precision targets must lie in (0, 1)")
    projected_events = np.sort(np.asarray(args.projected_events, dtype=float))
    if np.any(~np.isfinite(projected_events)) or np.any(projected_events <= 0.0):
        raise ValueError("projected event counts must be finite and positive")

    samples = parse_samples(args.sample)
    binnings: dict[str, tuple[AnalysisBinning, dict]] = {}
    for label, config_text in args.binning:
        if label in binnings:
            raise ValueError(f"duplicate binning label: {label}")
        config_path = Path(config_text).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        binnings[label] = (from_config(config_path), config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "mc_statistics_projection.pdf"
    csv_path = args.output_dir / "mc_statistics_cells.csv"
    summary_path = args.output_dir / "mc_statistics_summary.json"
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    with PdfPages(pdf_path) as pdf:
        for sample in samples:
            accumulated = accumulate(sample, binnings, args.beam_energy)
            for binning_label, (binning, _) in binnings.items():
                metrics = bin_metrics(
                    accumulated[binning_label], binning, args.minimum_acceptance
                )
                rows, summary, _ = render_sample(
                    pdf,
                    sample.label,
                    binning_label,
                    accumulated[binning_label],
                    binning,
                    metrics,
                    precision_targets,
                    projected_events,
                    args.minimum_phi_coverage,
                )
                all_rows.extend(rows)
                summary["input"] = str(sample.path)
                summary["v_max_GeV2"] = sample.v_max
                summaries.append(summary)
                print(json.dumps(summary, indent=2))

    write_csv(csv_path, all_rows)
    document = {
        "schema_version": 1,
        "method": "provisional-matched-particle-f32-statistics-projection-v1",
        "selection": (
            "correctly identified MC::GenMatch electron, proton, and two photons; "
            "generated and reconstructed events in the configured phase space"
        ),
        "limitations": (
            "planning diagnostic only; does not apply the maintained candidate, "
            "fiducial, or exclusivity selection and therefore does not replace a "
            "production response study"
        ),
        "beam_energy_GeV": args.beam_energy,
        "minimum_acceptance": args.minimum_acceptance,
        "minimum_phi_coverage": args.minimum_phi_coverage,
        "precision_targets": precision_targets,
        "projected_generated_events": projected_events.tolist(),
        "samples": summaries,
        "outputs": {
            "pdf": str(pdf_path.resolve()),
            "cell_csv": str(csv_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
