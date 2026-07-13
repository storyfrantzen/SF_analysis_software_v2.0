from __future__ import annotations

from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .binning import AnalysisBinning
from .event_sample import _dis, _minus_t, _trento_phi


Array = np.ndarray
LUND_TEXT_SUFFIXES = {".lund", ".txt"}


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
    born_integrated_cross_section: float | None = None
    radiative_integrated_cross_section: float | None = None


def compute_radiative_correction(
    born: str | Path,
    radiative: str | Path,
    binning: AnalysisBinning,
    *,
    beam_energy: float,
    chunk_size: int = 200_000,
    max_events: int | None = None,
    max_files: int | None = None,
    min_counts: int = 5,
    normalization_ratio: float | None = None,
    born_integrated_cross_section: float | None = None,
    radiative_integrated_cross_section: float | None = None,
    progress_chunks: int = 0,
) -> RadiativeCorrectionResult:
    """Compute bin-by-bin radiative corrections from Born and radiative LUND samples.

    Histograms are accumulated in the native flat bin order and unflattened only at
    output time.  The trento phi convention is inherited from the analysis core:
    electron-proton planes through ``event_sample._trento_phi``.
    The stored correction is the radiative-to-Born cross-section ratio; downstream
    unfolding divides by it.
    """
    if min_counts < 0:
        raise ValueError("min_counts must be non-negative")
    born_files = _limited_lund_files(born, max_files)
    radiative_files = _limited_lund_files(radiative, max_files)
    if progress_chunks > 0:
        limit_text = f", max_files={max_files}" if max_files is not None else ""
        events_text = f", max_events={max_events}" if max_events is not None else ""
        print(
            f"[PROGRESS] Radiative correction preflight: "
            f"born_files={len(born_files)}, radiative_files={len(radiative_files)}"
            f"{limit_text}{events_text}"
        )
    born_result = histogram_lund(
        born_files,
        binning,
        beam_energy=beam_energy,
        chunk_size=chunk_size,
        max_events=max_events,
        progress_chunks=progress_chunks,
        progress_label="Born",
    )
    radiative_result = histogram_lund(
        radiative_files,
        binning,
        beam_energy=beam_energy,
        chunk_size=chunk_size,
        max_events=max_events,
        progress_chunks=progress_chunks,
        progress_label="Radiative",
    )
    if born_result.topology_events == 0:
        raise ValueError(f"Born sample has no valid generated e p pi0 events: {born}")
    if radiative_result.topology_events == 0:
        raise ValueError(f"Radiative sample has no valid generated e p pi0 events: {radiative}")

    if (born_integrated_cross_section is None) != (radiative_integrated_cross_section is None):
        raise ValueError(
            "born_integrated_cross_section and radiative_integrated_cross_section "
            "must be provided together"
        )
    if normalization_ratio is not None and born_integrated_cross_section is not None:
        raise ValueError(
            "normalization_ratio cannot be combined with integrated cross sections"
        )

    if born_integrated_cross_section is not None:
        born_integrated_cross_section = _positive_finite(
            born_integrated_cross_section, "born_integrated_cross_section"
        )
        radiative_integrated_cross_section = _positive_finite(
            radiative_integrated_cross_section, "radiative_integrated_cross_section"
        )
        normalization_ratio = (
            radiative_integrated_cross_section
            / born_integrated_cross_section
            * born_result.topology_events
            / radiative_result.topology_events
        )
    elif normalization_ratio is None:
        normalization_ratio = born_result.topology_events / radiative_result.topology_events
    normalization_ratio = float(normalization_ratio)
    if not np.isfinite(normalization_ratio) or normalization_ratio <= 0.0:
        raise ValueError("normalization_ratio must be positive and finite")

    lambda_born = born_result.counts + 0.5
    lambda_rad = radiative_result.counts + 0.5
    c_rad_flat = normalization_ratio * lambda_rad / lambda_born
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
        born_integrated_cross_section=born_integrated_cross_section,
        radiative_integrated_cross_section=radiative_integrated_cross_section,
    )


def _positive_finite(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def histogram_lund(
    pattern_or_dir: str | Path | Sequence[Path],
    binning: AnalysisBinning,
    *,
    beam_energy: float,
    chunk_size: int = 200_000,
    max_events: int | None = None,
    max_files: int | None = None,
    progress_chunks: int = 0,
    progress_label: str = "LUND",
) -> LundHistogramResult:
    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be positive when provided")
    files = _limited_lund_files(pattern_or_dir, max_files)
    counts = np.zeros(binning.size, dtype=float)
    q2_min = np.full(binning.size, np.inf)
    q2_max = np.full(binning.size, -np.inf)
    eprime_min = np.full(binning.size, np.inf)
    eprime_max = np.full(binning.size, -np.inf)
    stats = _LundStats(files=len(files))

    if progress_chunks > 0:
        print(
            f"[PROGRESS] {progress_label} LUND files found: {len(files)}; "
            f"chunk_size={chunk_size}, progress_chunks={progress_chunks}"
        )

    for chunk_index, (electron, proton) in enumerate(
        _iter_lund_chunks(files, chunk_size, max_events, stats, progress_chunks, progress_label),
        start=1,
    ):
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
        if progress_chunks > 0 and chunk_index % progress_chunks == 0:
            print(
                f"[PROGRESS] {progress_label} LUND chunks {chunk_index}: "
                f"seen={stats.events_seen}, topology={stats.topology_events}, "
                f"in-range={stats.in_range}"
            )

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
    files: list[Path],
    chunk_size: int,
    max_events: int | None,
    stats: _LundStats,
    progress_chunks: int = 0,
    progress_label: str = "LUND",
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    electron_rows: list[tuple[float, float, float, float]] = []
    proton_rows: list[tuple[float, float, float, float]] = []
    progress_events = chunk_size * progress_chunks if progress_chunks > 0 else 0
    next_event_progress = progress_events

    for file_index, filename in enumerate(files, start=1):
        if progress_chunks > 0 and (file_index == 1 or file_index % progress_chunks == 0):
            print(
                f"[PROGRESS] {progress_label} LUND file {file_index}/{len(files)}: {filename}"
            )
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
                if progress_events > 0 and stats.events_seen >= next_event_progress:
                    print(
                        f"[PROGRESS] {progress_label} LUND events: "
                        f"seen={stats.events_seen}, topology={stats.topology_events}, "
                        f"pending-topology={len(electron_rows)}"
                    )
                    next_event_progress += progress_events
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


def _limited_lund_files(pattern_or_dir: str | Path | Sequence[Path], max_files: int | None) -> list[Path]:
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive when provided")
    files = _lund_files(pattern_or_dir, max_files=max_files)
    if not files:
        raise FileNotFoundError(
            f"No LUND-like text files matched: {pattern_or_dir}. "
            "Directories are scanned recursively for text files with LUND event headers."
        )
    return files


def _lund_files(pattern_or_dir: str | Path | Sequence[Path], max_files: int | None = None) -> list[Path]:
    if not isinstance(pattern_or_dir, str | Path):
        return _filter_lund_files(pattern_or_dir, max_files=max_files)
    path = Path(pattern_or_dir)
    if path.is_dir():
        if max_files is not None:
            return _filter_lund_files(path.rglob("*"), max_files=max_files)
        candidates = sorted(
            item for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in LUND_TEXT_SUFFIXES
        )
    else:
        candidates = sorted(Path(item) for item in glob.glob(str(pattern_or_dir)))
    return _filter_lund_files(candidates, max_files=max_files)


def _filter_lund_files(candidates: Iterable[Path], max_files: int | None = None) -> list[Path]:
    files: list[Path] = []
    for item in candidates:
        if not item.is_file() or item.suffix.lower() not in LUND_TEXT_SUFFIXES:
            continue
        if item.stat().st_size <= 0:
            continue
        if item.suffix.lower() == ".lund":
            files.append(item)
            if max_files is not None and len(files) >= max_files:
                break
            continue
        if (
            _looks_text(item)
            and _looks_lund_header(item)
        ):
            files.append(item)
            if max_files is not None and len(files) >= max_files:
                break
    return files


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


def _looks_lund_header(path: Path, max_lines: int = 50) -> bool:
    try:
        with path.open("r", errors="replace") as source:
            for _ in range(max_lines):
                line = source.readline()
                if not line:
                    return False
                fields = line.split()
                if not fields:
                    continue
                try:
                    return int(fields[0]) > 0
                except ValueError:
                    continue
    except OSError:
        return False
    return False


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
