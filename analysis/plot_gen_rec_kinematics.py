#!/usr/bin/env python3
"""Render generated-versus-selected EPPI0 diagnostics from an event-sample NPZ.

The input is produced by ``analysis/build_event_sample.py`` after the HIPO files
have passed through ``hipo2root`` and ``post_process``.  Consequently
``rec_selected`` means that the repository's configured candidate construction
and any cuts declared by that post-processing configuration found and retained
one candidate.  This is particularly useful for AAO radiative input, whose
generator record is e' p pi0 gamma rather than e' p gamma gamma.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROTON_MASS = 0.93827208816
PI0_MASS = 0.1349768
BLUE = "#0072B2"
BLUE_FILL = "#9ecae1"
ORANGE = "#D55E00"

OUTPUT_NAMES = (
    "gen_kinematics.png",
    "rec_kinematics.png",
    "gen_vs_rec.png",
    "gen_rec_overlay.png",
    "efficiency.png",
    "header_nu_check.png",
    "results.md",
)


@dataclass(frozen=True)
class Variable:
    key: str
    label: str
    gen_key: str
    rec_key: str | None
    degrees: bool = False
    fixed_range: tuple[float, float] | None = None


COMPARISON_VARIABLES = (
    Variable("Q2", r"$Q^2$ [GeV$^2$]", "gen_Q2", "rec_Q2"),
    Variable("W", r"$W$ [GeV]", "gen_W", "rec_W"),
    Variable("xB", r"$x_B$", "gen_xB", "rec_xB", fixed_range=(0.0, 1.0)),
    Variable("minus_t", r"$-t$ [GeV$^2$]", "gen_minus_t", "rec_minus_t"),
    Variable(
        "trento_phi",
        r"Trento $\phi$ [deg]",
        "gen_trento_phi",
        "rec_trento_phi",
        degrees=True,
        fixed_range=(-180.0, 180.0),
    ),
    Variable("electronP", r"$e'$ $|p|$ [GeV]", "gen_electronP", "rec_electronP"),
    Variable(
        "electronTheta",
        r"$e'$ $\theta$ [deg]",
        "gen_electronTheta",
        "rec_electronTheta",
        degrees=True,
    ),
    Variable("protonP", r"$p$ $|p|$ [GeV]", "gen_protonP", "rec_protonP"),
    Variable(
        "protonTheta",
        r"$p$ $\theta$ [deg]",
        "gen_protonTheta",
        "rec_protonTheta",
        degrees=True,
    ),
    Variable("pi0P", r"$\pi^0$ $|p|$ [GeV]", "gen_pi0P", "rec_pi0_p"),
    Variable(
        "pi0Theta",
        r"$\pi^0$ $\theta$ [deg]",
        "gen_pi0Theta",
        "rec_pi0_theta",
        degrees=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="NPZ from build_event_sample.py")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--label", default="AAO radiative RGA Fall 2018 inbending")
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument(
        "--beam-energy",
        type=float,
        help="Beam energy in GeV; defaults to the event-sample metadata",
    )
    return parser.parse_args()


def load_sample(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "metadata_json"
        }
        raw_metadata = archive["metadata_json"].item() if "metadata_json" in archive else "{}"
    metadata = json.loads(str(raw_metadata))
    required = {
        "gen_Q2",
        "gen_xB",
        "gen_minus_t",
        "gen_trento_phi",
        "gen_electronP",
        "gen_electronTheta",
        "gen_protonP",
        "gen_protonTheta",
        "gen_pi0P",
        "gen_pi0Theta",
        "rec_selected",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"event sample is missing required arrays: {missing}")
    size = arrays["rec_selected"].size
    bad_shapes = sorted(name for name, values in arrays.items() if values.ndim and values.size != size)
    if bad_shapes:
        raise ValueError(f"event-sample arrays have inconsistent sizes: {bad_shapes}")
    add_derived_arrays(arrays)
    return arrays, metadata


def resolve_beam_energy(
    metadata: dict[str, object], override: float | None = None
) -> float:
    raw = override if override is not None else metadata.get("beam_energy")
    if raw is None:
        raise ValueError(
            "beam energy is absent from the event-sample metadata; "
            "supply --beam-energy"
        )
    beam_energy = float(raw)
    if not np.isfinite(beam_energy) or beam_energy <= 0.0:
        raise ValueError("beam energy must be finite and positive")
    return beam_energy


def add_derived_arrays(arrays: dict[str, np.ndarray]) -> None:
    for prefix in ("gen", "rec"):
        q2_key, xb_key, w_key = f"{prefix}_Q2", f"{prefix}_xB", f"{prefix}_W"
        if w_key not in arrays and q2_key in arrays and xb_key in arrays:
            q2 = np.asarray(arrays[q2_key], dtype=float)
            xb = np.asarray(arrays[xb_key], dtype=float)
            w2 = PROTON_MASS**2 + np.divide(
                q2 * (1.0 - xb),
                xb,
                out=np.full_like(q2, np.nan),
                where=xb > 0,
            )
            arrays[w_key] = np.sqrt(np.maximum(w2, 0.0))


def values(arrays: dict[str, np.ndarray], key: str, degrees: bool = False) -> np.ndarray:
    result = np.asarray(arrays.get(key, np.full(arrays["rec_selected"].size, np.nan)), dtype=float)
    if degrees:
        result = np.rad2deg(result)
    return result


def finite(values_: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    selected = np.isfinite(values_)
    if mask is not None:
        selected &= mask
    return values_[selected]


def display_range(*samples: np.ndarray, fixed: tuple[float, float] | None = None) -> tuple[float, float]:
    if fixed is not None:
        return fixed
    combined = np.concatenate([finite(np.asarray(sample, dtype=float)) for sample in samples])
    if combined.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(combined, [0.2, 99.8])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.nanmedian(combined)) if combined.size else 0.0
        return (center - 0.5, center + 0.5)
    pad = 0.04 * (hi - lo)
    return (float(lo - pad), float(hi + pad))


def one_hist(
    axis,
    sample: np.ndarray,
    label: str,
    bins: int,
    range_: tuple[float, float] | None = None,
) -> None:
    clean = finite(sample)
    axis.hist(
        clean,
        bins=bins,
        range=range_,
        histtype="stepfilled",
        color=BLUE_FILL,
        edgecolor=BLUE,
        linewidth=1.0,
    )
    axis.set_xlabel(label)
    axis.set_ylabel("events")


def hist2d(axis, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, bins: int) -> None:
    keep = np.isfinite(x) & np.isfinite(y)
    if not np.any(keep):
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        return
    xrange = display_range(x[keep])
    yrange = display_range(y[keep])
    axis.hist2d(x[keep], y[keep], bins=bins, range=[xrange, yrange], cmap="viridis")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def four_vector(momentum: np.ndarray, theta: np.ndarray, phi: np.ndarray, mass: float) -> np.ndarray:
    momentum = np.asarray(momentum, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    sin_theta = np.sin(theta)
    px = momentum * sin_theta * np.cos(phi)
    py = momentum * sin_theta * np.sin(phi)
    pz = momentum * np.cos(theta)
    energy = np.sqrt(np.maximum(momentum * momentum + mass * mass, 0.0))
    return np.column_stack([energy, px, py, pz])


def mass2(vector: np.ndarray) -> np.ndarray:
    return vector[:, 0] ** 2 - np.sum(vector[:, 1:] ** 2, axis=1)


def particle(arrays: dict[str, np.ndarray], prefix: str, rec_pi0: bool = False) -> dict[str, np.ndarray]:
    if rec_pi0:
        keys = ("rec_pi0_p", "rec_pi0_theta", "rec_pi0_phi")
    else:
        keys = (f"{prefix}P", f"{prefix}Theta", f"{prefix}Phi")
    return {
        "p": values(arrays, keys[0]),
        "theta": values(arrays, keys[1]),
        "phi": values(arrays, keys[2]),
    }


def page_data(
    arrays: dict[str, np.ndarray], level: str, beam_energy: float
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    if level == "gen":
        parts = {
            "electron": particle(arrays, "gen_electron"),
            "proton": particle(arrays, "gen_proton"),
            "pi0": particle(arrays, "gen_pi0"),
        }
        q2 = values(arrays, "gen_Q2")
        xb = values(arrays, "gen_xB")
        minus_t = values(arrays, "gen_minus_t")
        w = values(arrays, "gen_W")
        trento_phi = values(arrays, "gen_trento_phi")
        pi_mass = np.full(q2.size, PI0_MASS)
    else:
        parts = {
            "electron": particle(arrays, "rec_electron"),
            "proton": particle(arrays, "rec_proton"),
            "pi0": particle(arrays, "rec_pi0", rec_pi0=True),
        }
        q2 = values(arrays, "rec_Q2")
        xb = values(arrays, "rec_xB")
        minus_t = values(arrays, "rec_minus_t")
        w = values(arrays, "rec_W")
        trento_phi = values(arrays, "rec_trento_phi")
        pi_mass = values(arrays, "rec_m_gg")

    electron4 = four_vector(
        parts["electron"]["p"], parts["electron"]["theta"], parts["electron"]["phi"], 0.0
    )
    proton4 = four_vector(
        parts["proton"]["p"], parts["proton"]["theta"], parts["proton"]["phi"], PROTON_MASS
    )
    beam = np.array([beam_energy, 0.0, 0.0, beam_energy])
    target = np.array([PROTON_MASS, 0.0, 0.0, 0.0])
    mm2_ep = mass2(beam[None, :] + target[None, :] - electron4 - proton4)
    event = {
        "Q2": q2,
        "xB": xb,
        "minus_t": minus_t,
        "W": w,
        "trento_phi": trento_phi,
        "MM2": mm2_ep,
        "pi_mass": pi_mass,
    }
    return parts, event


def plot_kinematics_page(
    arrays: dict[str, np.ndarray],
    output: Path,
    label: str,
    bins: int,
    level: str,
    beam_energy: float,
) -> None:
    selected = np.asarray(arrays["rec_selected"], dtype=bool)
    mask = np.ones(selected.size, dtype=bool) if level == "gen" else selected
    parts, event = page_data(arrays, level, beam_energy)
    # Retain the momentum-versus-theta diagnostics while also showing all three
    # one-dimensional particle coordinates.  The final two rows contain the
    # event variables, including Trento phi explicitly.
    fig, axes = plt.subplots(5, 4, figsize=(17, 16))
    for row, (name, symbol) in enumerate(
        (("electron", r"$e'$"), ("proton", r"$p$"), ("pi0", r"$\pi^0$"))
    ):
        momentum = finite(parts[name]["p"], mask)
        theta = finite(np.rad2deg(parts[name]["theta"]), mask)
        phi = finite(np.rad2deg(parts[name]["phi"]), mask)
        paired = mask & np.isfinite(parts[name]["p"]) & np.isfinite(parts[name]["theta"])
        one_hist(
            axes[row, 0], momentum, rf"{symbol} $|p|$ [GeV]", bins,
            display_range(momentum),
        )
        one_hist(
            axes[row, 1], theta, rf"{symbol} $\theta$ [deg]", bins,
            display_range(theta),
        )
        one_hist(
            axes[row, 2], phi, rf"{symbol} $\phi$ [deg]", bins,
            (-180.0, 180.0),
        )
        hist2d(
            axes[row, 3],
            parts[name]["p"][paired],
            np.rad2deg(parts[name]["theta"][paired]),
            rf"{symbol} $|p|$ [GeV]",
            rf"{symbol} $\theta$ [deg]",
            bins,
        )

    one_hist(axes[3, 0], finite(event["Q2"], mask), r"$Q^2$ [GeV$^2$]", bins)
    one_hist(axes[3, 1], finite(event["W"], mask), r"$W$ [GeV]", bins)
    one_hist(axes[3, 2], finite(event["xB"], mask), r"$x_B$", bins, (0.0, 1.0))
    one_hist(
        axes[3, 3], finite(np.rad2deg(event["trento_phi"]), mask),
        r"Trento $\phi$ [deg]", bins, (-180.0, 180.0),
    )
    one_hist(axes[4, 0], finite(event["minus_t"], mask), r"$-t$ [GeV$^2$]", bins)
    one_hist(
        axes[4, 1], finite(event["MM2"], mask),
        r"$M_X^2(e'p)$ [GeV$^2$]  (missing $\pi^0$)", bins,
    )
    axes[4, 1].axvline(PI0_MASS**2, color="grey", linestyle=":", linewidth=1)
    mass_label = r"generated $M(\pi^0)$ [GeV]" if level == "gen" else r"$M(\gamma\gamma)$ [GeV]"
    one_hist(axes[4, 2], finite(event["pi_mass"], mask), mass_label, bins, (0.0, 0.3))
    axes[4, 2].axvline(PI0_MASS, color="grey", linestyle=":", linewidth=1)
    axes[4, 3].axis("off")
    kind = "generator level" if level == "gen" else "reconstructed candidate level"
    fig.suptitle(f"{label}: {kind}, {np.count_nonzero(mask):,} events", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_valery_overlay(
    arrays: dict[str, np.ndarray],
    output: Path,
    label: str,
    bins: int,
    beam_energy: float,
) -> None:
    """Draw Valery-style rec/same-event-gen/all-gen overlays in event counts."""
    selected = np.asarray(arrays["rec_selected"], dtype=bool)
    gen_parts, gen_event = page_data(arrays, "gen", beam_energy)
    rec_parts, rec_event = page_data(arrays, "rec", beam_energy)

    panels: list[tuple[str, np.ndarray, np.ndarray, tuple[float, float] | None]] = [
        (r"$Q^2$ [GeV$^2$]", gen_event["Q2"], rec_event["Q2"], None),
        (r"$W$ [GeV]", gen_event["W"], rec_event["W"], None),
        (r"$x_B$", gen_event["xB"], rec_event["xB"], (0.0, 1.0)),
        (r"$-t$ [GeV$^2$]", gen_event["minus_t"], rec_event["minus_t"], None),
        (r"$M_X^2(e'p)$ [GeV$^2$]", gen_event["MM2"], rec_event["MM2"], None),
        (
            r"gen $M(\pi^0)$ / rec $M(\gamma\gamma)$ [GeV]",
            gen_event["pi_mass"], rec_event["pi_mass"], (0.0, 0.3),
        ),
    ]
    for name, symbol in (
        ("electron", r"$e'$"),
        ("proton", r"$p$"),
        ("pi0", r"$\pi^0$"),
    ):
        panels.extend(
            [
                (rf"{symbol} $|p|$ [GeV]", gen_parts[name]["p"], rec_parts[name]["p"], None),
                (
                    rf"{symbol} $\theta$ [deg]",
                    np.rad2deg(gen_parts[name]["theta"]),
                    np.rad2deg(rec_parts[name]["theta"]),
                    None,
                ),
                (
                    rf"{symbol} $\phi$ [deg]",
                    np.rad2deg(gen_parts[name]["phi"]),
                    np.rad2deg(rec_parts[name]["phi"]),
                    (-180.0, 180.0),
                ),
            ]
        )

    fig, axes = plt.subplots(5, 3, figsize=(13, 17))
    for axis, (xlabel, gen_values, rec_values, fixed_range) in zip(axes.flat, panels):
        raw_gen = np.asarray(gen_values, dtype=float)
        raw_rec = np.asarray(rec_values, dtype=float)
        gen_all = finite(raw_gen)
        gen_same = finite(raw_gen, selected)
        rec = finite(raw_rec, selected)
        range_ = display_range(gen_all, gen_same, rec, fixed=fixed_range)
        edges = np.linspace(range_[0], range_[1], bins + 1)
        rec_counts, same_counts, scaled_all_counts, counts = overlay_histograms(
            raw_gen, raw_rec, selected, edges
        )

        axis.stairs(
            rec_counts, edges, fill=True, alpha=0.32, color=ORANGE,
            linewidth=1.0, label=f"rec ({counts['rec']:,})",
        )
        axis.stairs(
            same_counts, edges, color=BLUE, linewidth=1.15,
            label=f"gen, same events ({counts['gen_same']:,})",
        )
        axis.stairs(
            scaled_all_counts, edges, color="#009E73", linewidth=1.15,
            label="gen, all scaled",
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("events")
        axis.legend(fontsize=6.5)

    fig.suptitle(
        f"{label}: reconstructed vs generated (unweighted); "
        r"rows 1–2 event variables, rows 3–5 $e'$, $p$, and $\pi^0$",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(output, dpi=140)
    plt.close(fig)


def overlay_histograms(
    generated: np.ndarray,
    reconstructed: np.ndarray,
    selected: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Return rec, same-event GEN, and integral-matched all-GEN histograms."""
    generated = np.asarray(generated, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    edges = np.asarray(edges, dtype=float)
    if generated.shape != reconstructed.shape or generated.shape != selected.shape:
        raise ValueError("generated, reconstructed, and selected arrays must align")
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("histogram edges must be a strictly increasing 1D array")

    generated_all = generated[np.isfinite(generated)]
    generated_same = generated[selected & np.isfinite(generated)]
    reconstructed_selected = reconstructed[selected & np.isfinite(reconstructed)]
    rec_counts, _ = np.histogram(reconstructed_selected, bins=edges)
    same_counts, _ = np.histogram(generated_same, bins=edges)
    all_counts, _ = np.histogram(generated_all, bins=edges)
    scale = generated_same.size / generated_all.size if generated_all.size else 0.0
    counts = {
        "rec": int(reconstructed_selected.size),
        "gen_same": int(generated_same.size),
        "gen_all": int(generated_all.size),
    }
    return rec_counts, same_counts, all_counts * scale, counts


def gen_variables() -> list[Variable]:
    particle_variables = []
    for name, symbol in (
        ("electron", r"$e'$"),
        ("proton", r"$p$"),
        ("pi0", r"$\pi^0$"),
        ("gamma1", r"$\gamma_{rad}$"),
    ):
        particle_variables.extend(
            [
                Variable(f"{name}P", rf"{symbol} $|p|$ [GeV]", f"gen_{name}P", None),
                Variable(
                    f"{name}Theta",
                    rf"{symbol} $\theta$ [deg]",
                    f"gen_{name}Theta",
                    None,
                    degrees=True,
                ),
                Variable(
                    f"{name}Phi",
                    rf"{symbol} $\phi$ [deg]",
                    f"gen_{name}Phi",
                    None,
                    degrees=True,
                    fixed_range=(-180.0, 180.0),
                ),
            ]
        )
    particle_variables.extend(COMPARISON_VARIABLES[:5])
    return particle_variables


def rec_variables() -> list[tuple[str, str, bool, tuple[float, float] | None]]:
    result = []
    for name, prefix, symbol in (
        ("electron", "electron", r"$e'$"),
        ("proton", "proton", r"$p$"),
        ("pi0", "pi0_", r"$\pi^0$"),
        ("gamma1", "gamma1", r"$\gamma_1$"),
        ("gamma2", "gamma2", r"$\gamma_2$"),
    ):
        result.extend(
            [
                (f"rec_{prefix}P" if prefix != "pi0_" else "rec_pi0_p", rf"{symbol} $|p|$ [GeV]", False, None),
                (f"rec_{prefix}Theta" if prefix != "pi0_" else "rec_pi0_theta", rf"{symbol} $\theta$ [deg]", True, None),
                (f"rec_{prefix}Phi" if prefix != "pi0_" else "rec_pi0_phi", rf"{symbol} $\phi$ [deg]", True, (-180.0, 180.0)),
            ]
        )
    result.extend(
        [
            ("rec_Q2", r"$Q^2$ [GeV$^2$]", False, None),
            ("rec_xB", r"$x_B$", False, (0.0, 1.0)),
            ("rec_minus_t", r"$-t$ [GeV$^2$]", False, None),
            ("rec_W", r"$W$ [GeV]", False, None),
            ("rec_trento_phi", r"Trento $\phi$ [deg]", True, (-180.0, 180.0)),
            ("rec_m_gg", r"$m_{\gamma\gamma}$ [GeV]", False, (0.0, 0.3)),
        ]
    )
    return result


def plot_comparison(arrays: dict[str, np.ndarray], output: Path, label: str, bins: int) -> None:
    selected = np.asarray(arrays["rec_selected"], dtype=bool)
    fig, axes = plt.subplots(4, 3, figsize=(13, 13))
    for axis, variable in zip(axes.flat, COMPARISON_VARIABLES):
        gen = values(arrays, variable.gen_key, variable.degrees)
        rec = values(arrays, variable.rec_key or "", variable.degrees)
        range_ = display_range(gen, finite(rec, selected), fixed=variable.fixed_range)
        axis.hist(
            finite(gen), bins=bins, range=range_, density=True,
            histtype="step", color=BLUE, linewidth=1.5, label="generated",
        )
        axis.hist(
            finite(rec, selected), bins=bins, range=range_, density=True,
            histtype="step", color=ORANGE, linewidth=1.5,
            label="selected reconstructed candidate",
        )
        axis.set_xlabel(variable.label)
        axis.set_ylabel("unit-normalized density")
        axis.legend(fontsize=7)
    for axis in axes.flat[len(COMPARISON_VARIABLES):]:
        axis.axis("off")
    fraction = np.count_nonzero(selected) / selected.size if selected.size else 0.0
    fig.suptitle(
        f"{label}: generated vs selected reconstructed shapes "
        f"(event retention {100.0 * fraction:.2f}%)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output, dpi=140)
    plt.close(fig)


def plot_header_nu(
    arrays: dict[str, np.ndarray],
    output: Path,
    label: str,
    bins: int,
    beam_energy: float,
) -> None:
    electron_p = values(arrays, "gen_electronP")
    nu = beam_energy - electron_p
    weight = np.asarray(arrays.get("gen_weight", np.ones(nu.size)), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    one_hist(axes[0], finite(weight), "gEvents generator weight", bins)
    keep = np.isfinite(nu) & np.isfinite(weight)
    hist2d(
        axes[1], nu[keep], weight[keep],
        r"$\nu=E-E'$ from generated $e'$ [GeV]", "gEvents generator weight", bins,
    )
    one_hist(axes[2], finite(weight - 1.0), "gEvents generator weight $-1$", bins)
    fig.suptitle(
        f"{label}: generator-weight/nu check; AAO radiative header payload is not used as a weight",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=130)
    plt.close(fig)


def binned_efficiency(
    coordinates: np.ndarray,
    selected: np.ndarray,
    weights: np.ndarray,
    bins: int,
    range_: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(coordinates) & np.isfinite(weights) & (weights >= 0)
    denominator, edges = np.histogram(coordinates[valid], bins=bins, range=range_, weights=weights[valid])
    numerator, _ = np.histogram(
        coordinates[valid & selected], bins=edges, weights=weights[valid & selected]
    )
    efficiency = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    # Binomial bars are exact for the unit-weight AAO samples and remain a useful
    # visual guide if a later sample supplies non-negative generator weights.
    error = np.sqrt(
        np.divide(
            efficiency * (1.0 - efficiency),
            denominator,
            out=np.full_like(efficiency, np.nan),
            where=denominator > 0,
        )
    )
    return 0.5 * (edges[1:] + edges[:-1]), efficiency, error


def plot_efficiency(arrays: dict[str, np.ndarray], output: Path, label: str) -> None:
    selected = np.asarray(arrays["rec_selected"], dtype=bool)
    weights = np.asarray(arrays.get("gen_weight", np.ones(selected.size)), dtype=float)
    variables = COMPARISON_VARIABLES
    fig, axes = plt.subplots(4, 3, figsize=(13, 14))
    for axis, variable in zip(axes.flat, variables):
        gen = values(arrays, variable.gen_key, variable.degrees)
        range_ = display_range(gen, fixed=variable.fixed_range)
        centers, efficiency, error = binned_efficiency(gen, selected, weights, 28, range_)
        axis.errorbar(centers, efficiency, yerr=error, fmt="o", ms=2.5, lw=0.8, color=BLUE)
        axis.set_xlabel(f"generated {variable.label}")
        axis.set_ylabel("selected / generated")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(variables):]:
        axis.axis("off")
    fig.suptitle(f"{label}: full configured candidate-selection efficiency", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output, dpi=140)
    plt.close(fig)


def write_results(
    arrays: dict[str, np.ndarray], metadata: dict[str, object], output: Path, label: str
) -> None:
    selected = np.asarray(arrays["rec_selected"], dtype=bool)
    generated = selected.size
    accepted = int(np.count_nonzero(selected))
    fraction = accepted / generated if generated else 0.0
    mgg = finite(values(arrays, "rec_m_gg"), selected)
    radiative = np.asarray(arrays.get("gen_radiative", np.zeros(generated)), dtype=bool)
    lines = [
        f"# {label}",
        "",
        "The reconstructed numerator uses the repository's configured EPPI0 "
        "candidate construction. It is not daughter-photon truth matching.",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| valid generated events | {generated:,} |",
        f"| generated events marked radiative | {np.count_nonzero(radiative):,} |",
        f"| selected reconstructed candidates | {accepted:,} |",
        f"| selected/generated | {fraction:.6f} |",
    ]
    if mgg.size:
        lines.extend(
            [
                f"| median selected $m_{{\\gamma\\gamma}}$ [GeV] | {np.median(mgg):.6f} |",
                f"| selected $0.08 < m_{{\\gamma\\gamma}} < 0.20$ | {np.count_nonzero((mgg > 0.08) & (mgg < 0.20)):,} |",
            ]
        )
    lines.extend(["", "## Event-sample provenance", "", "```json", json.dumps(metadata, indent=2, sort_keys=True), "```", ""])
    output.write_text("\n".join(lines))


def render_diagnostics(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    output_dir: Path,
    label: str,
    bins: int,
    beam_energy: float,
) -> tuple[Path, ...]:
    if bins <= 0:
        raise ValueError("--bins must be positive")
    if not np.isfinite(beam_energy) or beam_energy <= 0.0:
        raise ValueError("beam energy must be finite and positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_kinematics_page(
        arrays, output_dir / "gen_kinematics.png", label, bins, "gen", beam_energy
    )
    plot_kinematics_page(
        arrays, output_dir / "rec_kinematics.png", label, bins, "rec", beam_energy
    )
    plot_comparison(arrays, output_dir / "gen_vs_rec.png", label, bins)
    plot_valery_overlay(
        arrays, output_dir / "gen_rec_overlay.png", label, bins, beam_energy
    )
    plot_efficiency(arrays, output_dir / "efficiency.png", label)
    plot_header_nu(
        arrays, output_dir / "header_nu_check.png", label, bins, beam_energy
    )
    write_results(arrays, metadata, output_dir / "results.md", label)
    return tuple(output_dir / name for name in OUTPUT_NAMES)


def main() -> int:
    args = parse_args()
    arrays, metadata = load_sample(args.sample)
    beam_energy = resolve_beam_energy(metadata, args.beam_energy)
    outputs = render_diagnostics(
        arrays,
        metadata,
        args.output_dir,
        args.label,
        args.bins,
        beam_energy,
    )
    selected = np.asarray(arrays["rec_selected"], dtype=bool)
    print(f"generated events: {selected.size}")
    print(f"selected candidates: {np.count_nonzero(selected)}")
    print(f"beam energy: {beam_energy:g} GeV")
    for output in outputs:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
