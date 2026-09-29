#!/usr/bin/env python3
"""Compute the generator-level EXCLURAD radiative correction versus phi.

Each LUND file is paired with its production ``config.json``.  For run ``i``
and phi bin ``b`` the cross-section estimate is

    sigma_i * n_i,b / N_i,

where ``N_i`` is the number of generated events in the file.  The radiative
count alone is restricted to ``v = MX2(e'p) - m_pi0**2 < v_max``.  In
particular, the surviving radiative histogram is *not* renormalized after the
cut: rejecting the hard tail is part of the radiative correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROTON_MASS_GEV = 0.93827208816
PI0_MASS_GEV = 0.1349768
ELECTRON_MASS_GEV = 0.00051099895
RUN_INDEX = re.compile(r"_(\d+)\.lund$")
PHYSICS_CARD_FIELDS = (
    "model",
    "ebeam",
    "W2",
    "Q2",
    "t",
    "vcut",
    "t_nucl",
    "ihel",
    "idecay",
    "y",
    "xB",
    "iacc",
    "vmin",
    "itarg",
    "vz",
    "ptarg",
    "fermi",
    "theta_e",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXCLURAD generator-level C_rad(phi), normalized per config sigma_nb"
    )
    parser.add_argument(
        "production_dir",
        type=Path,
        help="Production root containing born/ and rad/",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--v-max", type=float, default=0.2)
    parser.add_argument("--phi-bins", type=int, default=12)
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=200,
        help="Require this many files per mode; use 0 to disable",
    )
    return parser.parse_args()


def nested_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        if key in value:
            found.append(value[key])
        for child in value.values():
            found.extend(nested_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(nested_values(child, key))
    return found


def sigma_nb(config: dict[str, Any]) -> float:
    values = [float(value) for value in nested_values(config, "sigma_nb")]
    if not values:
        raise ValueError("config contains no sigma_nb")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(f"invalid sigma_nb value(s): {values}")
    if not np.allclose(values, values[0], rtol=1e-12, atol=0.0):
        raise ValueError(f"config contains inconsistent sigma_nb values: {values}")
    return values[0]


def run_index(path: Path) -> int:
    match = RUN_INDEX.search(path.name)
    if match is None:
        raise ValueError(f"cannot extract run index from {path.name}")
    return int(match.group(1))


def unit(vectors: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(vectors, axis=1)
    return np.divide(
        vectors,
        magnitude[:, None],
        out=np.full_like(vectors, np.nan),
        where=magnitude[:, None] > 0.0,
    )


def trento_phi(electron: np.ndarray, proton: np.ndarray, beam_energy: float) -> np.ndarray:
    beam = np.zeros_like(electron[:, :3])
    beam[:, 2] = beam_energy
    q = beam - electron[:, :3]
    lepton_normal = unit(np.cross(beam, electron[:, :3]))
    hadron_normal = unit(np.cross(proton[:, :3], q))
    q_hat = unit(q)
    cosine = np.sum(lepton_normal * hadron_normal, axis=1)
    sine = np.sum(q_hat * np.cross(lepton_normal, hadron_normal), axis=1)
    return np.mod(np.arctan2(sine, cosine), 2.0 * np.pi)


def inelasticity_v(
    electron: np.ndarray,
    proton: np.ndarray,
    beam_energy: float,
) -> np.ndarray:
    beam_pz = math.sqrt(beam_energy**2 - ELECTRON_MASS_GEV**2)
    missing_energy = beam_energy + PROTON_MASS_GEV - electron[:, 3] - proton[:, 3]
    missing_px = -electron[:, 0] - proton[:, 0]
    missing_py = -electron[:, 1] - proton[:, 1]
    missing_pz = beam_pz - electron[:, 2] - proton[:, 2]
    missing_mass2 = (
        missing_energy**2 - missing_px**2 - missing_py**2 - missing_pz**2
    )
    return missing_mass2 - PI0_MASS_GEV**2


def read_lund(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    electrons: list[tuple[float, float, float, float]] = []
    protons: list[tuple[float, float, float, float]] = []
    events = 0
    with path.open("r", errors="replace") as source:
        while True:
            header = source.readline()
            if not header:
                break
            fields = header.split()
            if not fields:
                continue
            try:
                n_particles = int(fields[0])
            except ValueError:
                continue
            events += 1
            electron = None
            proton = None
            complete = True
            for _ in range(n_particles):
                row = source.readline()
                if not row:
                    complete = False
                    break
                parts = row.split()
                if len(parts) < 10:
                    complete = False
                    continue
                try:
                    pid = int(parts[3])
                    momentum = tuple(float(parts[index]) for index in (6, 7, 8, 9))
                except ValueError:
                    complete = False
                    continue
                if pid == 11 and electron is None:
                    electron = momentum
                elif pid == 2212 and proton is None:
                    proton = momentum
            if not complete or electron is None or proton is None:
                raise ValueError(f"incomplete/non-e'p event {events} in {path}")
            electrons.append(electron)
            protons.append(proton)
    return np.asarray(electrons, dtype=float), np.asarray(protons, dtype=float), events


def canonical(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(canonical(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, canonical(child)) for key, child in value.items()))
    return value


def card_signature(card: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    missing = [key for key in PHYSICS_CARD_FIELDS if key not in card]
    if missing:
        raise ValueError(f"card is missing required fields: {missing}")
    return tuple((key, canonical(card[key])) for key in PHYSICS_CARD_FIELDS)


def standard_error(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.full(values.shape[1:], np.nan)
    return np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])


def mode_estimates(
    production_dir: Path,
    mode: str,
    edges: np.ndarray,
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
    integrated: list[float] = []
    configs: list[dict[str, Any]] = []
    selected_total = 0
    generated_total = 0
    sigma_values: list[float] = []
    run_indices: list[int] = []

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
        phi = trento_phi(electrons, protons, beam_energy)
        finite = np.isfinite(phi)
        if mode == "rad":
            v = inelasticity_v(electrons, protons, beam_energy)
            finite &= np.isfinite(v) & (v < v_max)
        counts, _ = np.histogram(phi[finite], bins=edges)
        cross_section = sigma_nb(config)
        per_run = cross_section * counts.astype(float) / requested_events
        estimates.append(per_run)
        integrated.append(cross_section * int(np.count_nonzero(finite)) / requested_events)
        selected_total += int(np.count_nonzero(finite))
        generated_total += observed_events
        sigma_values.append(cross_section)
        run_indices.append(index)
        configs.append(config)
        if file_number % 25 == 0 or file_number == len(files):
            print(
                f"{mode}: {file_number}/{len(files)} files, "
                f"generated={generated_total}, selected={selected_total}",
                flush=True,
            )

    matrix = np.asarray(estimates)
    integrated_values = np.asarray(integrated)
    return {
        "matrix": matrix,
        "mean": np.mean(matrix, axis=0),
        "sem": standard_error(matrix),
        "integrated_values": integrated_values,
        "integrated_mean": float(np.mean(integrated_values)),
        "integrated_sem": float(standard_error(integrated_values[:, None])[0]),
        "generated": generated_total,
        "selected": selected_total,
        "sigma_values": np.asarray(sigma_values),
        "run_indices": run_indices,
        "configs": configs,
    }


def audit_samples(born: dict[str, Any], rad: dict[str, Any]) -> dict[str, Any]:
    born_cards = {card_signature(config["card"]) for config in born["configs"]}
    rad_cards = {card_signature(config["card"]) for config in rad["configs"]}
    if len(born_cards) != 1 or len(rad_cards) != 1:
        raise ValueError("card settings vary between runs within a mode")
    if born_cards != rad_cards:
        raise ValueError("Born and radiative samples do not share one physics card/box")

    born_commits = {config["stamp"]["git_commit"] for config in born["configs"]}
    rad_commits = {config["stamp"]["git_commit"] for config in rad["configs"]}
    if born_commits != rad_commits or len(born_commits) != 1:
        raise ValueError(
            f"Born/radiative commit mismatch: born={born_commits}, rad={rad_commits}"
        )
    if born["run_indices"] != rad["run_indices"]:
        raise ValueError("Born and radiative run-index sets differ")

    card = born["configs"][0]["card"]
    return {
        "git_commit": next(iter(born_commits)),
        "git_branch": born["configs"][0]["stamp"].get("git_branch"),
        "model": card["model"],
        "beam_energy_GeV": float(card["ebeam"]),
        "W2_box_GeV2": list(card["W2"]),
        "Q2_box_GeV2": list(card["Q2"]),
        "t_box_GeV2": list(card["t"]),
        "generation_vcut_GeV2": float(card["vcut"]),
    }


def ratio_and_error(
    numerator: np.ndarray,
    numerator_sem: np.ndarray,
    denominator: np.ndarray,
    denominator_sem: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )
    relative_variance = np.divide(
        numerator_sem**2,
        numerator**2,
        out=np.zeros_like(numerator),
        where=numerator > 0.0,
    )
    relative_variance += np.divide(
        denominator_sem**2,
        denominator**2,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return ratio, ratio * np.sqrt(relative_variance)


def write_csv(
    path: Path,
    edges_deg: np.ndarray,
    born: dict[str, Any],
    rad: dict[str, Any],
    correction: np.ndarray,
    correction_sem: np.ndarray,
) -> None:
    with path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            (
                "phi_low_deg",
                "phi_high_deg",
                "phi_center_deg",
                "born_cross_section_nb",
                "born_sem_nb",
                "rad_vcut_cross_section_nb",
                "rad_vcut_sem_nb",
                "C_rad_rad_over_born",
                "C_rad_sem",
                "delta_rad_percent",
                "delta_rad_sem_percent",
            )
        )
        for index in range(len(correction)):
            writer.writerow(
                (
                    edges_deg[index],
                    edges_deg[index + 1],
                    0.5 * (edges_deg[index] + edges_deg[index + 1]),
                    born["mean"][index],
                    born["sem"][index],
                    rad["mean"][index],
                    rad["sem"][index],
                    correction[index],
                    correction_sem[index],
                    100.0 * (correction[index] - 1.0),
                    100.0 * correction_sem[index],
                )
            )


def make_plot(
    path: Path,
    edges_deg: np.ndarray,
    born: dict[str, Any],
    rad: dict[str, Any],
    correction: np.ndarray,
    correction_sem: np.ndarray,
    v_max: float,
    audit: dict[str, Any],
) -> None:
    centers = 0.5 * (edges_deg[:-1] + edges_deg[1:])
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    axes[0].errorbar(
        centers,
        born["mean"],
        yerr=born["sem"],
        fmt="o-",
        capsize=2,
        label="Born",
    )
    axes[0].errorbar(
        centers,
        rad["mean"],
        yerr=rad["sem"],
        fmt="s-",
        capsize=2,
        label=rf"Radiative, $v<{v_max:g}$ GeV$^2$",
    )
    axes[0].set_ylabel("Bin-integrated cross section [nb]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        centers,
        correction,
        yerr=correction_sem,
        fmt="o-",
        capsize=2,
        color="black",
    )
    axes[1].axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel=r"Trento $\phi$ [deg]",
        ylabel=r"$C_{\rm rad}=\sigma_{\rm rad}(v<0.2)/\sigma_{\rm Born}$",
        xlim=(0.0, 360.0),
    )
    axes[1].grid(alpha=0.25)
    fig.suptitle(
        "EXCLURAD generator-level radiative correction\n"
        f"{audit['model']}, E={audit['beam_energy_GeV']:g} GeV, "
        f"commit {audit['git_commit']}, v<{v_max:g} GeV²"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = arguments()
    if args.phi_bins <= 0:
        raise ValueError("--phi-bins must be positive")
    if not math.isfinite(args.v_max):
        raise ValueError("--v-max must be finite")

    edges = np.linspace(0.0, 2.0 * np.pi, args.phi_bins + 1)
    born = mode_estimates(
        args.production_dir, "born", edges, args.v_max, args.expected_runs
    )
    rad = mode_estimates(
        args.production_dir, "rad", edges, args.v_max, args.expected_runs
    )
    audit = audit_samples(born, rad)
    correction, correction_sem = ratio_and_error(
        rad["mean"], rad["sem"], born["mean"], born["sem"]
    )
    integrated_correction, integrated_sem = ratio_and_error(
        np.asarray([rad["integrated_mean"]]),
        np.asarray([rad["integrated_sem"]]),
        np.asarray([born["integrated_mean"]]),
        np.asarray([born["integrated_sem"]]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges_deg = np.rad2deg(edges)
    csv_path = args.output_dir / "radiative_correction_phi.csv"
    png_path = args.output_dir / "radiative_correction_phi.png"
    pdf_path = args.output_dir / "radiative_correction_phi.pdf"
    npz_path = args.output_dir / "radiative_correction_phi.npz"
    summary_path = args.output_dir / "summary.json"

    write_csv(csv_path, edges_deg, born, rad, correction, correction_sem)
    make_plot(png_path, edges_deg, born, rad, correction, correction_sem, args.v_max, audit)
    make_plot(pdf_path, edges_deg, born, rad, correction, correction_sem, args.v_max, audit)
    np.savez_compressed(
        npz_path,
        phi_edges_deg=edges_deg,
        born_cross_section_nb=born["mean"],
        born_sem_nb=born["sem"],
        rad_vcut_cross_section_nb=rad["mean"],
        rad_vcut_sem_nb=rad["sem"],
        C_rad=correction,
        C_rad_sem=correction_sem,
        v_max_GeV2=args.v_max,
    )

    summary = {
        **audit,
        "definition": "C_rad = sigma_rad(v < v_max) / sigma_Born",
        "percent_definition": "100 * (C_rad - 1)",
        "normalization": "mean over runs of sigma_nb_i * selected_bin_events_i / generated_events_i",
        "radiative_histogram_renormalized_after_v_cut": False,
        "v_definition": "MX2(e'p) - m_pi0^2",
        "v_max_GeV2": args.v_max,
        "phi_convention": "electron-proton Trento plane, wrapped to [0, 360) degrees",
        "phi_bins": args.phi_bins,
        "born_runs": len(born["configs"]),
        "radiative_runs": len(rad["configs"]),
        "born_generated_events": born["generated"],
        "radiative_generated_events": rad["generated"],
        "born_selected_events": born["selected"],
        "radiative_selected_events_vcut": rad["selected"],
        "radiative_vcut_retention": rad["selected"] / rad["generated"],
        "born_sigma_nb_mean": float(np.mean(born["sigma_values"])),
        "born_sigma_nb_min": float(np.min(born["sigma_values"])),
        "born_sigma_nb_max": float(np.max(born["sigma_values"])),
        "radiative_sigma_nb_mean": float(np.mean(rad["sigma_values"])),
        "radiative_sigma_nb_min": float(np.min(rad["sigma_values"])),
        "radiative_sigma_nb_max": float(np.max(rad["sigma_values"])),
        "born_integrated_cross_section_nb": born["integrated_mean"],
        "born_integrated_sem_nb": born["integrated_sem"],
        "radiative_vcut_integrated_cross_section_nb": rad["integrated_mean"],
        "radiative_vcut_integrated_sem_nb": rad["integrated_sem"],
        "integrated_C_rad": float(integrated_correction[0]),
        "integrated_C_rad_sem": float(integrated_sem[0]),
        "integrated_delta_percent": float(100.0 * (integrated_correction[0] - 1.0)),
        "integrated_delta_sem_percent": float(100.0 * integrated_sem[0]),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
