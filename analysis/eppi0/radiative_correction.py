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
    q2_min: Array
    q2_max: Array
    eprime_min: Array
    eprime_max: Array
    files: int
    events_seen: int
    topology_events: int
    in_range: int
    generated_q2_min: float
    generated_q2_max: float
    generated_eprime_min: float
    generated_eprime_max: float


@dataclass(frozen=True)
class RadiativeCorrectionResult:
    c_rad: Array
    delta_c: Array
    reliable: Array
    support_overlap: Array
    support_status: Array
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
    support_overlap = (born_result.counts > 0.0) & (radiative_result.counts > 0.0)
    support_status = support_status_codes(born_result.counts, radiative_result.counts, min_counts)

    return RadiativeCorrectionResult(
        c_rad=binning.unflatten(c_rad_safe),
        delta_c=binning.unflatten(delta_safe),
        reliable=binning.unflatten(reliable_flat),
        support_overlap=binning.unflatten(support_overlap),
        support_status=binning.unflatten(support_status),
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
    q2_min = np.full(binning.size, np.inf)
    q2_max = np.full(binning.size, -np.inf)
    eprime_min = np.full(binning.size, np.inf)
    eprime_max = np.full(binning.size, -np.inf)
    stats = _LundStats(files=len(files))

    for electron, proton in _iter_lund_chunks(files, chunk_size, max_events, stats):
        q2, xb = _dis(electron, beam_energy)
        eprime = electron[:, 3]
        _update_global_ranges(stats, q2, eprime)
        minus_t = _minus_t(proton)
        phi = _trento_phi(electron, proton, beam_energy)
        flat = binning.coordinates_to_flat(q2, xb, minus_t, phi)
        inside = (flat >= 0) & (flat < binning.size)
        inside_flat = flat[inside]
        counts += np.bincount(inside_flat, minlength=binning.size)
        if inside_flat.size:
            np.minimum.at(q2_min, inside_flat, q2[inside])
            np.maximum.at(q2_max, inside_flat, q2[inside])
            np.minimum.at(eprime_min, inside_flat, eprime[inside])
            np.maximum.at(eprime_max, inside_flat, eprime[inside])
        stats.in_range += int(np.count_nonzero(inside))

    q2_min = np.where(np.isfinite(q2_min), q2_min, np.nan)
    q2_max = np.where(np.isfinite(q2_max), q2_max, np.nan)
    eprime_min = np.where(np.isfinite(eprime_min), eprime_min, np.nan)
    eprime_max = np.where(np.isfinite(eprime_max), eprime_max, np.nan)

    return LundHistogramResult(
        counts=counts,
        q2_min=binning.unflatten(q2_min),
        q2_max=binning.unflatten(q2_max),
        eprime_min=binning.unflatten(eprime_min),
        eprime_max=binning.unflatten(eprime_max),
        files=stats.files,
        events_seen=stats.events_seen,
        topology_events=stats.topology_events,
        in_range=stats.in_range,
        generated_q2_min=_finite_or_nan(stats.generated_q2_min),
        generated_q2_max=_finite_or_nan(stats.generated_q2_max),
        generated_eprime_min=_finite_or_nan(stats.generated_eprime_min),
        generated_eprime_max=_finite_or_nan(stats.generated_eprime_max),
    )


def support_status_codes(born_counts: Array, radiative_counts: Array, min_counts: int) -> Array:
    """Classify per-bin correction support.

    Codes:
      0 reliable;
      1 both samples are empty;
      2 born only;
      3 radiative only;
      4 both populated but born is below min_counts;
      5 both populated but radiative is below min_counts;
      6 both populated but both are below min_counts.
    """
    born_counts = np.asarray(born_counts, dtype=float)
    radiative_counts = np.asarray(radiative_counts, dtype=float)
    if born_counts.shape != radiative_counts.shape:
        raise ValueError("born and radiative count arrays must have matching shapes")
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


@dataclass
class _LundStats:
    files: int
    events_seen: int = 0
    topology_events: int = 0
    in_range: int = 0
    generated_q2_min: float = np.inf
    generated_q2_max: float = -np.inf
    generated_eprime_min: float = np.inf
    generated_eprime_max: float = -np.inf


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


def _update_global_ranges(stats: _LundStats, q2: Array, eprime: Array) -> None:
    valid_q2 = np.asarray(q2)[np.isfinite(q2)]
    if valid_q2.size:
        stats.generated_q2_min = min(stats.generated_q2_min, float(np.min(valid_q2)))
        stats.generated_q2_max = max(stats.generated_q2_max, float(np.max(valid_q2)))
    valid_eprime = np.asarray(eprime)[np.isfinite(eprime)]
    if valid_eprime.size:
        stats.generated_eprime_min = min(stats.generated_eprime_min, float(np.min(valid_eprime)))
        stats.generated_eprime_max = max(stats.generated_eprime_max, float(np.max(valid_eprime)))


def _finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan
