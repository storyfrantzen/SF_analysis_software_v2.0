from __future__ import annotations

from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Iterable

import numpy as np

from .binning import AnalysisBinning
from .event_sample import _dis, _minus_t, _trento_phi


Array = np.ndarray


@dataclass(frozen=True)
class LundHistogramResult:
    counts: Array
    files: int
    events_seen: int
    topology_events: int
    in_range: int


@dataclass(frozen=True)
class RadiativeCorrectionResult:
    c_rad: Array
    delta_c: Array
    reliable: Array
    born: LundHistogramResult
    radiative: LundHistogramResult
    normalization_ratio: float


def compute_radiative_correction(
    born: str | Path,
    radiative: str | Path,
    binning: AnalysisBinning,
    *,
    beam_energy: float,
    chunk_size: int = 200_000,
    max_events: int | None = None,
    min_counts: int = 5,
    normalization_ratio: float | None = None,
) -> RadiativeCorrectionResult:
    """Compute bin-by-bin radiative corrections from Born and radiative LUND samples.

    Histograms are accumulated in the native flat bin order and unflattened only at
    output time.  The trento phi convention is inherited from the analysis core:
    electron-proton planes through ``event_sample._trento_phi``.
    """
    if min_counts < 0:
        raise ValueError("min_counts must be non-negative")
    born_result = histogram_lund(
        born,
        binning,
        beam_energy=beam_energy,
        chunk_size=chunk_size,
        max_events=max_events,
    )
    radiative_result = histogram_lund(
        radiative,
        binning,
        beam_energy=beam_energy,
        chunk_size=chunk_size,
        max_events=max_events,
    )
    if born_result.topology_events == 0:
        raise ValueError(f"Born sample has no valid generated e p pi0 events: {born}")
    if radiative_result.topology_events == 0:
        raise ValueError(f"Radiative sample has no valid generated e p pi0 events: {radiative}")

    if normalization_ratio is None:
        normalization_ratio = radiative_result.topology_events / born_result.topology_events
    normalization_ratio = float(normalization_ratio)
    if not np.isfinite(normalization_ratio) or normalization_ratio <= 0.0:
        raise ValueError("normalization_ratio must be positive and finite")

    lambda_born = born_result.counts + 0.5
    lambda_rad = radiative_result.counts + 0.5
    c_rad_flat = normalization_ratio * lambda_born / lambda_rad
    delta_flat = c_rad_flat * np.sqrt((1.0 / lambda_born) + (1.0 / lambda_rad))
    reliable_flat = (born_result.counts >= min_counts) & (radiative_result.counts >= min_counts)

    c_rad_safe = np.where(reliable_flat, c_rad_flat, 1.0)
    delta_safe = np.where(reliable_flat, delta_flat, 1.0)

    return RadiativeCorrectionResult(
        c_rad=binning.unflatten(c_rad_safe),
        delta_c=binning.unflatten(delta_safe),
        reliable=binning.unflatten(reliable_flat),
        born=born_result,
        radiative=radiative_result,
        normalization_ratio=normalization_ratio,
    )


def histogram_lund(
    pattern_or_dir: str | Path,
    binning: AnalysisBinning,
    *,
    beam_energy: float,
    chunk_size: int = 200_000,
    max_events: int | None = None,
) -> LundHistogramResult:
    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be positive when provided")
    files = _lund_files(pattern_or_dir)
    if not files:
        raise FileNotFoundError(f"No LUND text files matched: {pattern_or_dir}")
    counts = np.zeros(binning.size, dtype=float)
    stats = _LundStats(files=len(files))

    for electron, proton in _iter_lund_chunks(files, chunk_size, max_events, stats):
        q2, xb = _dis(electron, beam_energy)
        minus_t = _minus_t(proton)
        phi = _trento_phi(electron, proton, beam_energy)
        flat = binning.coordinates_to_flat(q2, xb, minus_t, phi)
        inside = (flat >= 0) & (flat < binning.size)
        counts += np.bincount(flat[inside], minlength=binning.size)
        stats.in_range += int(np.count_nonzero(inside))

    return LundHistogramResult(
        counts=counts,
        files=stats.files,
        events_seen=stats.events_seen,
        topology_events=stats.topology_events,
        in_range=stats.in_range,
    )


@dataclass
class _LundStats:
    files: int
    events_seen: int = 0
    topology_events: int = 0
    in_range: int = 0


def _iter_lund_chunks(
    files: Iterable[Path],
    chunk_size: int,
    max_events: int | None,
    stats: _LundStats,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    electron_rows: list[tuple[float, float, float, float]] = []
    proton_rows: list[tuple[float, float, float, float]] = []

    for filename in files:
        with filename.open("r", errors="replace") as source:
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

                stats.events_seen += 1
                electron = None
                proton = None
                photon_count = 0
                has_pi0 = False
                valid = True
                for _ in range(n_particles):
                    row = source.readline()
                    if not row:
                        valid = False
                        break
                    parts = row.split()
                    if len(parts) < 10:
                        valid = False
                        continue
                    try:
                        pid = int(parts[3])
                        px = float(parts[6])
                        py = float(parts[7])
                        pz = float(parts[8])
                        energy = float(parts[9])
                    except ValueError:
                        valid = False
                        continue
                    if pid == 11 and electron is None:
                        electron = (px, py, pz, energy)
                    elif pid == 2212 and proton is None:
                        proton = (px, py, pz, energy)
                    elif pid == 111:
                        has_pi0 = True
                    elif pid == 22:
                        photon_count += 1

                if not valid or electron is None or proton is None:
                    continue
                if not (has_pi0 or photon_count >= 2):
                    continue

                electron_rows.append(electron)
                proton_rows.append(proton)
                stats.topology_events += 1

                if len(electron_rows) >= chunk_size:
                    yield np.asarray(electron_rows, dtype=float), np.asarray(proton_rows, dtype=float)
                    electron_rows.clear()
                    proton_rows.clear()

                if max_events is not None and stats.topology_events >= max_events:
                    if electron_rows:
                        yield np.asarray(electron_rows, dtype=float), np.asarray(proton_rows, dtype=float)
                    return

    if electron_rows:
        yield np.asarray(electron_rows, dtype=float), np.asarray(proton_rows, dtype=float)


def _lund_files(pattern_or_dir: str | Path) -> list[Path]:
    path = Path(pattern_or_dir)
    if path.is_dir():
        files = sorted(item for item in path.rglob("*.txt") if item.is_file())
    else:
        files = sorted(Path(item) for item in glob.glob(str(pattern_or_dir)))
    return [item for item in files if item.stat().st_size > 0 and _looks_text(item)]


def _looks_text(path: Path, nbytes: int = 4096) -> bool:
    try:
        with path.open("rb") as source:
            chunk = source.read(nbytes)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True
