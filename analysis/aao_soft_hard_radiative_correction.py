#!/usr/bin/env python3
"""Split an AAO generator-level radiative correction into soft and hard parts.

AAO radiative LUND records contain an undecayed pi0 and one radiative photon.
For the soft/no-hard-photon branch the photon is a 10-keV placeholder; for the
hard branch its energy is above the generator separator Delta (10 MeV in the
RGA mode-3 production).  This utility keeps the original stochastic
multiplicity: consecutive byte-identical LUND events are one generated
proposal with multiplicity ``m`` and contribute ``m**2`` to the variance.

The two radiative components deliberately share one normalization.  Therefore

    C_rad(total) = C_rad(soft) + C_rad(hard)

bin by bin.  Independently normalizing the components would destroy this
identity and obscure which branch produces a structure in phi.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.eppi0.binning import AnalysisBinning, bin_indices, from_config
from analysis.eppi0.event_sample import _dis, _minus_t, _trento_phi
from analysis.eppi0.phase_space import AnalysisPhaseSpace


Array = np.ndarray


@dataclass
class StreamStats:
    files: int = 0
    events: int = 0
    topology_events: int = 0
    clusters: int = 0
    maximum_multiplicity: int = 0
    soft_events: int = 0
    hard_events: int = 0
    soft_photon_energy_min: float = np.inf
    soft_photon_energy_max: float = -np.inf
    hard_photon_energy_min: float = np.inf
    hard_photon_energy_max: float = -np.inf


@dataclass
class CellHistogram:
    counts: Array
    sumw2: Array
    stats: StreamStats


@dataclass
class RadiativeCellHistogram:
    soft_counts: Array
    soft_sumw2: Array
    hard_counts: Array
    hard_sumw2: Array
    stats: StreamStats


def manifest_paths(path: str | Path, expected_files: int | None = None) -> list[Path]:
    manifest = Path(path)
    files = [
        Path(line.strip())
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    missing = [item for item in files if not item.is_file() or item.stat().st_size <= 0]
    if missing:
        preview = "\n".join(str(item) for item in missing[:10])
        raise FileNotFoundError(f"manifest contains missing/empty files:\n{preview}")
    resolved = [item.resolve() for item in files]
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"manifest contains duplicate source files: {manifest}")
    if expected_files is not None and len(files) != expected_files:
        raise ValueError(
            f"expected {expected_files} files in {manifest}, found {len(files)}"
        )
    return files


def integrated_cross_sections(path: str | Path) -> tuple[float, float]:
    with np.load(path, allow_pickle=False) as artifact:
        born = float(np.asarray(artifact["born_integrated_cross_section"]).reshape(-1)[0])
        rad = float(
            np.asarray(artifact["radiative_integrated_cross_section"]).reshape(-1)[0]
        )
    if not np.isfinite(born) or born <= 0.0 or not np.isfinite(rad) or rad <= 0.0:
        raise ValueError("normalization artifact has invalid integrated cross sections")
    return born, rad


def _particle(row: str) -> tuple[int, tuple[float, float, float, float]] | None:
    fields = row.split()
    if len(fields) < 10:
        return None
    try:
        pid = int(fields[3])
        vector = tuple(float(fields[index]) for index in (6, 7, 8, 9))
    except ValueError:
        return None
    return pid, vector


def lund_clusters(
    files: Sequence[Path],
    *,
    radiative: bool,
    progress_files: int = 100,
) -> Iterator[tuple[Array, Array, float, int]]:
    """Yield electron, proton, photon energy, and exact-record multiplicity."""

    stats = StreamStats(files=len(files))
    for file_number, filename in enumerate(files, start=1):
        previous_signature: tuple[str, ...] | None = None
        previous_value: tuple[Array, Array, float] | None = None
        multiplicity = 0

        def flush() -> Iterator[tuple[Array, Array, float, int]]:
            nonlocal previous_signature, previous_value, multiplicity
            if previous_signature is not None and previous_value is not None:
                electron, proton, photon_energy = previous_value
                yield electron, proton, photon_energy, multiplicity
            previous_signature = None
            previous_value = None
            multiplicity = 0

        with filename.open("r", encoding="utf-8", errors="replace") as source:
            while True:
                header = source.readline()
                if not header:
                    break
                fields = header.split()
                if not fields:
                    continue
                try:
                    number_particles = int(fields[0])
                except ValueError:
                    continue
                stats.events += 1
                rows = [source.readline() for _ in range(number_particles)]
                if any(not row for row in rows):
                    yield from flush()
                    break
                electron = None
                proton = None
                photons: list[tuple[float, float, float, float]] = []
                has_pi0 = False
                valid = True
                for row in rows:
                    parsed = _particle(row)
                    if parsed is None:
                        valid = False
                        continue
                    pid, vector = parsed
                    if pid == 11 and electron is None:
                        electron = vector
                    elif pid == 2212 and proton is None:
                        proton = vector
                    elif pid == 111:
                        has_pi0 = True
                    elif pid == 22:
                        photons.append(vector)
                if radiative:
                    topology = valid and electron is not None and proton is not None and has_pi0
                    topology = topology and len(photons) == 1
                    photon_energy = photons[0][3] if topology else np.nan
                else:
                    topology = valid and electron is not None and proton is not None
                    topology = topology and (has_pi0 or len(photons) >= 2)
                    photon_energy = np.nan
                if not topology:
                    yield from flush()
                    continue
                stats.topology_events += 1
                signature = tuple([header] + rows)
                if signature == previous_signature:
                    multiplicity += 1
                else:
                    yield from flush()
                    previous_signature = signature
                    previous_value = (
                        np.asarray(electron, dtype=float),
                        np.asarray(proton, dtype=float),
                        float(photon_energy),
                    )
                    multiplicity = 1
        yield from flush()
        if progress_files and (file_number % progress_files == 0 or file_number == len(files)):
            print(
                f"[PROGRESS] {'radiative' if radiative else 'Born'} "
                f"files={file_number}/{len(files)} events={stats.events} "
                f"topology={stats.topology_events}",
                flush=True,
            )
    del stats


def cluster_chunks(
    files: Sequence[Path],
    *,
    radiative: bool,
    chunk_events: int = 200_000,
) -> Iterator[tuple[Array, Array, Array, Array]]:
    """Batch clusters for vectorized kinematic calculations."""

    electrons: list[Array] = []
    protons: list[Array] = []
    photon_energies: list[float] = []
    multiplicities: list[int] = []
    buffered_events = 0
    for electron, proton, photon_energy, multiplicity in lund_clusters(
        files, radiative=radiative
    ):
        electrons.append(electron)
        protons.append(proton)
        photon_energies.append(photon_energy)
        multiplicities.append(multiplicity)
        buffered_events += multiplicity
        if buffered_events >= chunk_events:
            yield (
                np.asarray(electrons), np.asarray(protons),
                np.asarray(photon_energies), np.asarray(multiplicities, dtype=float),
            )
            electrons.clear()
            protons.clear()
            photon_energies.clear()
            multiplicities.clear()
            buffered_events = 0
    if electrons:
        yield (
            np.asarray(electrons), np.asarray(protons),
            np.asarray(photon_energies), np.asarray(multiplicities, dtype=float),
        )


def _cell_indices(
    binning: AnalysisBinning, point: tuple[float, float, float]
) -> tuple[int, int, int]:
    result = (
        int(bin_indices(np.asarray([point[0]]), binning.q2_edges)[0]),
        int(bin_indices(np.asarray([point[1]]), binning.xb_edges)[0]),
        int(bin_indices(np.asarray([point[2]]), binning.t_edges)[0]),
    )
    if min(result) < 0:
        raise ValueError("requested point lies outside the configured binning")
    return result


def _chunk_phi_indices(
    electron: Array,
    proton: Array,
    binning: AnalysisBinning,
    phase_space: AnalysisPhaseSpace,
    beam_energy: float,
    cell: tuple[int, int, int],
) -> tuple[Array, Array]:
    q2, xb = _dis(electron, beam_energy)
    mt = _minus_t(proton)
    phi = _trento_phi(electron, proton, beam_energy)
    indices = binning.indices(q2, xb, mt, phi)
    selected = phase_space.mask(q2, xb, beam_energy)
    selected &= indices[0] == cell[0]
    selected &= indices[1] == cell[1]
    selected &= indices[2] == cell[2]
    selected &= indices[3] >= 0
    return selected, indices[3]


def histogram_cell(
    files: Sequence[Path],
    binning: AnalysisBinning,
    phase_space: AnalysisPhaseSpace,
    *,
    beam_energy: float,
    point: tuple[float, float, float],
    radiative: bool,
    photon_threshold: float,
) -> CellHistogram:
    cell = _cell_indices(binning, point)
    counts = np.zeros(binning.phi_edges.size - 1, dtype=float)
    sumw2 = np.zeros_like(counts)
    stats = StreamStats(files=len(files))
    for electron, proton, photon_energy, multiplicity in cluster_chunks(
        files, radiative=radiative
    ):
        stats.events += int(np.sum(multiplicity))
        stats.topology_events += int(np.sum(multiplicity))
        stats.clusters += multiplicity.size
        stats.maximum_multiplicity = max(
            stats.maximum_multiplicity, int(np.max(multiplicity))
        )
        selected, iphi = _chunk_phi_indices(
            electron, proton, binning, phase_space, beam_energy, cell
        )
        counts += np.bincount(
            iphi[selected], weights=multiplicity[selected], minlength=counts.size
        )
        sumw2 += np.bincount(
            iphi[selected], weights=np.square(multiplicity[selected]), minlength=counts.size
        )
    return CellHistogram(counts=counts, sumw2=sumw2, stats=stats)


def histogram_radiative_cell(
    files: Sequence[Path],
    binning: AnalysisBinning,
    phase_space: AnalysisPhaseSpace,
    *,
    beam_energy: float,
    point: tuple[float, float, float],
    photon_threshold: float,
) -> RadiativeCellHistogram:
    cell = _cell_indices(binning, point)
    nphi = binning.phi_edges.size - 1
    soft_counts = np.zeros(nphi, dtype=float)
    soft_sumw2 = np.zeros(nphi, dtype=float)
    hard_counts = np.zeros(nphi, dtype=float)
    hard_sumw2 = np.zeros(nphi, dtype=float)
    stats = StreamStats(files=len(files))
    for electron, proton, photon_energy, multiplicity in cluster_chunks(
        files, radiative=True
    ):
        number_events = int(np.sum(multiplicity))
        stats.events += number_events
        stats.topology_events += number_events
        stats.clusters += multiplicity.size
        stats.maximum_multiplicity = max(
            stats.maximum_multiplicity, int(np.max(multiplicity))
        )
        is_soft = photon_energy <= photon_threshold
        stats.soft_events += int(np.sum(multiplicity[is_soft]))
        stats.hard_events += int(np.sum(multiplicity[~is_soft]))
        if np.any(is_soft):
            stats.soft_photon_energy_min = min(
                stats.soft_photon_energy_min, float(np.min(photon_energy[is_soft]))
            )
            stats.soft_photon_energy_max = max(
                stats.soft_photon_energy_max, float(np.max(photon_energy[is_soft]))
            )
        if np.any(~is_soft):
            stats.hard_photon_energy_min = min(
                stats.hard_photon_energy_min, float(np.min(photon_energy[~is_soft]))
            )
            stats.hard_photon_energy_max = max(
                stats.hard_photon_energy_max, float(np.max(photon_energy[~is_soft]))
            )
        selected, iphi = _chunk_phi_indices(
            electron, proton, binning, phase_space, beam_energy, cell
        )
        selected_soft = selected & is_soft
        selected_hard = selected & ~is_soft
        soft_counts += np.bincount(
            iphi[selected_soft], weights=multiplicity[selected_soft], minlength=nphi
        )
        soft_sumw2 += np.bincount(
            iphi[selected_soft],
            weights=np.square(multiplicity[selected_soft]),
            minlength=nphi,
        )
        hard_counts += np.bincount(
            iphi[selected_hard], weights=multiplicity[selected_hard], minlength=nphi
        )
        hard_sumw2 += np.bincount(
            iphi[selected_hard],
            weights=np.square(multiplicity[selected_hard]),
            minlength=nphi,
        )
    return RadiativeCellHistogram(
        soft_counts=soft_counts,
        soft_sumw2=soft_sumw2,
        hard_counts=hard_counts,
        hard_sumw2=hard_sumw2,
        stats=stats,
    )


def ratio_and_error(
    numerator: Array,
    numerator_sumw2: Array,
    denominator: Array,
    denominator_sumw2: Array,
    normalization: float,
) -> tuple[Array, Array]:
    ratio = np.divide(
        normalization * numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0.0,
    )
    variance = np.full_like(ratio, np.nan)
    valid = (numerator > 0.0) & (denominator > 0.0)
    variance[valid] = np.square(ratio[valid]) * (
        numerator_sumw2[valid] / np.square(numerator[valid])
        + denominator_sumw2[valid] / np.square(denominator[valid])
    )
    return ratio, np.sqrt(variance)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--born-manifest", required=True)
    parser.add_argument("--radiative-manifest", required=True)
    parser.add_argument("--normalization-artifact", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="output prefix (no suffix)")
    parser.add_argument("--expected-files", type=int, default=2000)
    parser.add_argument("--photon-threshold", type=float, default=0.010)
    parser.add_argument("--q2", type=float, default=1.5)
    parser.add_argument("--xb", type=float, default=0.30)
    parser.add_argument("--minus-t", type=float, default=0.30)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    born_files = manifest_paths(args.born_manifest, args.expected_files)
    rad_files = manifest_paths(args.radiative_manifest, args.expected_files)
    with Path(args.config).open(encoding="utf-8") as source:
        config = json.load(source)
    binning = from_config(args.config)
    phase_space = AnalysisPhaseSpace.from_config(config)
    beam_energy = float(config["beam_energy"])
    point = (float(args.q2), float(args.xb), float(args.minus_t))

    born = histogram_cell(
        born_files,
        binning,
        phase_space,
        beam_energy=beam_energy,
        point=point,
        radiative=False,
        photon_threshold=args.photon_threshold,
    )
    radiative = histogram_radiative_cell(
        rad_files,
        binning,
        phase_space,
        beam_energy=beam_energy,
        point=point,
        photon_threshold=args.photon_threshold,
    )
    if radiative.stats.soft_events == 0 or radiative.stats.hard_events == 0:
        raise ValueError(
            "the requested photon threshold did not produce both soft and hard events"
        )
    sigma_born, sigma_rad = integrated_cross_sections(args.normalization_artifact)
    normalization = (
        sigma_rad / sigma_born
        * born.stats.topology_events / radiative.stats.topology_events
    )
    total_counts = radiative.soft_counts + radiative.hard_counts
    total_sumw2 = radiative.soft_sumw2 + radiative.hard_sumw2
    c_soft, dc_soft = ratio_and_error(
        radiative.soft_counts,
        radiative.soft_sumw2,
        born.counts,
        born.sumw2,
        normalization,
    )
    c_hard, dc_hard = ratio_and_error(
        radiative.hard_counts,
        radiative.hard_sumw2,
        born.counts,
        born.sumw2,
        normalization,
    )
    c_total, dc_total = ratio_and_error(
        total_counts, total_sumw2, born.counts, born.sumw2, normalization
    )
    if not np.allclose(c_total, c_soft + c_hard, equal_nan=True, atol=1e-12):
        raise RuntimeError("soft + hard decomposition failed")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    phi_edges = binning.phi_edges
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    hard_fraction = np.divide(
        radiative.hard_counts,
        total_counts,
        out=np.full_like(total_counts, np.nan),
        where=total_counts > 0,
    )
    np.savez_compressed(
        output.with_suffix(".npz"),
        phi_edges_deg=phi_edges,
        born_counts=born.counts,
        born_cluster_sumw2=born.sumw2,
        soft_counts=radiative.soft_counts,
        soft_cluster_sumw2=radiative.soft_sumw2,
        hard_counts=radiative.hard_counts,
        hard_cluster_sumw2=radiative.hard_sumw2,
        c_soft=c_soft,
        delta_c_soft=dc_soft,
        c_hard=c_hard,
        delta_c_hard=dc_hard,
        c_total=c_total,
        delta_c_total=dc_total,
        hard_fraction=hard_fraction,
        normalization_ratio=normalization,
        born_integrated_cross_section=sigma_born,
        radiative_integrated_cross_section=sigma_rad,
        photon_threshold_GeV=float(args.photon_threshold),
        point_Q2_GeV2=point[0],
        point_xB=point[1],
        point_minus_t_GeV2=point[2],
        born_topology_events=born.stats.topology_events,
        radiative_topology_events=radiative.stats.topology_events,
        radiative_soft_events=radiative.stats.soft_events,
        radiative_hard_events=radiative.stats.hard_events,
    )

    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "phi_low_deg", "phi_high_deg", "born_events", "soft_events",
                "hard_events", "C_soft", "delta_C_soft", "C_hard",
                "delta_C_hard", "C_total", "delta_C_total", "hard_fraction",
            ]
        )
        for index in range(phi.size):
            writer.writerow(
                [
                    phi_edges[index], phi_edges[index + 1], born.counts[index],
                    radiative.soft_counts[index], radiative.hard_counts[index], c_soft[index],
                    dc_soft[index], c_hard[index], dc_hard[index], c_total[index],
                    dc_total[index], hard_fraction[index],
                ]
            )

    figure, axes = plt.subplots(3, 1, figsize=(9.0, 10.5), sharex=True)
    axes[0].stairs(born.counts, phi_edges, label="Born", color="black", linewidth=1.7)
    axes[0].stairs(
        radiative.soft_counts,
        phi_edges,
        label=r"radiative soft ($E_\gamma\leq\Delta$)",
    )
    axes[0].stairs(
        radiative.hard_counts,
        phi_edges,
        label=r"radiative hard ($E_\gamma>\Delta$)",
    )
    axes[0].set_ylabel("emitted events")
    axes[0].legend(ncol=3, fontsize=9)
    axes[0].grid(alpha=0.2)
    axes[1].errorbar(phi, c_total, yerr=dc_total, fmt="o-", label="total", color="black")
    axes[1].errorbar(phi, c_soft, yerr=dc_soft, fmt="o-", label="soft contribution")
    axes[1].errorbar(phi, c_hard, yerr=dc_hard, fmt="o-", label="hard contribution")
    axes[1].axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    axes[1].set_ylabel(r"contribution to $C_{rad}$")
    axes[1].legend(ncol=3, fontsize=9)
    axes[1].grid(alpha=0.2)
    axes[2].plot(phi, hard_fraction, "o-", color="tab:red")
    axes[2].set_ylabel("hard fraction of\nradiative numerator")
    axes[2].set_xlabel(r"Trento $\phi$ [deg]")
    axes[2].set_ylim(bottom=0.0)
    axes[2].grid(alpha=0.2)
    q2_index = int(bin_indices(np.asarray([point[0]]), binning.q2_edges)[0])
    xb_index = int(bin_indices(np.asarray([point[1]]), binning.xb_edges)[0])
    t_index = int(bin_indices(np.asarray([point[2]]), binning.t_edges)[0])
    figure.suptitle(
        "AAO soft/hard radiative decomposition\n"
        rf"$Q^2\in[{binning.q2_edges[q2_index]:g},{binning.q2_edges[q2_index+1]:g})$ GeV$^2$, "
        rf"$x_B\in[{binning.xb_edges[xb_index]:g},{binning.xb_edges[xb_index+1]:g})$, "
        rf"$-t\in[{binning.t_edges[t_index]:g},{binning.t_edges[t_index+1]:g})$ GeV$^2$; "
        rf"$\Delta={1000.0*args.photon_threshold:g}$ MeV"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(figure)

    summary = {
        "born_files": len(born_files),
        "radiative_files": len(rad_files),
        "born_topology_events": born.stats.topology_events,
        "radiative_topology_events": radiative.stats.topology_events,
        "radiative_soft_events": radiative.stats.soft_events,
        "radiative_hard_events": radiative.stats.hard_events,
        "radiative_soft_fraction": (
            radiative.stats.soft_events / radiative.stats.topology_events
        ),
        "radiative_hard_fraction": (
            radiative.stats.hard_events / radiative.stats.topology_events
        ),
        "soft_photon_energy_range_GeV": [
            radiative.stats.soft_photon_energy_min,
            radiative.stats.soft_photon_energy_max,
        ],
        "hard_photon_energy_range_GeV": [
            radiative.stats.hard_photon_energy_min,
            radiative.stats.hard_photon_energy_max,
        ],
        "maximum_born_cluster_multiplicity": born.stats.maximum_multiplicity,
        "maximum_radiative_cluster_multiplicity": (
            radiative.stats.maximum_multiplicity
        ),
        "normalization_ratio": normalization,
        "born_integrated_cross_section": sigma_born,
        "radiative_integrated_cross_section": sigma_rad,
        "photon_threshold_GeV": float(args.photon_threshold),
        "cell": {
            "Q2": [binning.q2_edges[q2_index], binning.q2_edges[q2_index + 1]],
            "xB": [binning.xb_edges[xb_index], binning.xb_edges[xb_index + 1]],
            "minus_t": [binning.t_edges[t_index], binning.t_edges[t_index + 1]],
        },
        "identity_max_abs_C_total_minus_soft_minus_hard": float(
            np.nanmax(np.abs(c_total - c_soft - c_hard))
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    for suffix in (".npz", ".csv", ".pdf", ".png", ".json"):
        print(f"wrote {output.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
