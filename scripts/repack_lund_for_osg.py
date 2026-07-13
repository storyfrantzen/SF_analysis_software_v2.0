#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import sys


@dataclass
class RepackStats:
    input_files: int = 0
    output_files: int = 0
    events: int = 0
    lines: int = 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Repack LUND files into event-count-limited chunks suitable for OSG simulation jobs. "
            "The original input files are only read, never modified."
        )
    )
    root.add_argument("input_dir", type=Path, help="Directory containing original .lund files")
    root.add_argument("output_dir", type=Path, help="Directory to receive repacked .lund chunks")
    root.add_argument("--glob", default="*.lund", help="Input filename glob inside input_dir")
    root.add_argument(
        "--events-per-file",
        type=int,
        default=5000,
        help="Maximum LUND events per repacked output file",
    )
    root.add_argument(
        "--jobs-per-submission",
        type=int,
        default=10000,
        help="Maximum entries per manifest file",
    )
    root.add_argument("--prefix", default="aao_rad_osg", help="Output LUND filename prefix")
    root.add_argument(
        "--relative-manifest-paths",
        action="store_true",
        help="Write paths relative to output_dir in manifest lists instead of absolute paths",
    )
    root.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count input events without writing repacked files",
    )
    root.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only rebuild manifest lists from existing output chunks in output_dir",
    )
    root.add_argument(
        "--fixed-lines-per-event",
        type=int,
        help=(
            "Use a faster path for LUND samples with a fixed number of lines per event. "
            "AAO 4-particle output uses 5 lines per event."
        ),
    )
    root.add_argument(
        "--progress-files",
        type=int,
        default=100,
        help="Print progress every N input files; use 0 to disable",
    )
    return root


def main() -> int:
    args = parser().parse_args()
    if args.events_per_file <= 0:
        raise SystemExit("--events-per-file must be positive")
    if args.jobs_per_submission <= 0:
        raise SystemExit("--jobs-per-submission must be positive")
    if args.fixed_lines_per_event is not None and args.fixed_lines_per_event <= 0:
        raise SystemExit("--fixed-lines-per-event must be positive")
    if args.manifest_only:
        manifests = write_manifests(
            args.output_dir,
            prefix=args.prefix,
            jobs_per_submission=args.jobs_per_submission,
            relative_paths=args.relative_manifest_paths,
        )
        chunks = sorted(args.output_dir.glob(f"{args.prefix}_*.lund"))
        print(f"Rebuilt {manifests} manifest files for {len(chunks)} chunks under {args.output_dir}")
        return 0
    input_files = sorted(path for path in args.input_dir.glob(args.glob) if path.is_file())
    if not input_files:
        raise SystemExit(f"No input files matched {args.input_dir / args.glob}")
    if args.output_dir.exists() and any(args.output_dir.glob("*.lund")) and not args.dry_run:
        raise SystemExit(
            f"Refusing to write into {args.output_dir}: it already contains .lund files. "
            "Choose a new output directory."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.fixed_lines_per_event is None:
        stats = repack_lund_files(
            input_files,
            args.output_dir,
            events_per_file=args.events_per_file,
            prefix=args.prefix,
            dry_run=args.dry_run,
            progress_files=args.progress_files,
        )
    else:
        stats = repack_fixed_line_lund_files(
            input_files,
            args.output_dir,
            events_per_file=args.events_per_file,
            lines_per_event=args.fixed_lines_per_event,
            prefix=args.prefix,
            dry_run=args.dry_run,
            progress_files=args.progress_files,
        )
    stats.input_files = len(input_files)
    if not args.dry_run:
        write_manifests(
            args.output_dir,
            prefix=args.prefix,
            jobs_per_submission=args.jobs_per_submission,
            relative_paths=args.relative_manifest_paths,
        )
    print_summary(stats, args)
    return 0


def repack_lund_files(
    input_files: list[Path],
    output_dir: Path,
    *,
    events_per_file: int,
    prefix: str,
    dry_run: bool,
    progress_files: int = 0,
) -> RepackStats:
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    stats = RepackStats()
    output = None
    events_in_output = 0
    output_index = 0
    try:
        total_files = len(input_files)
        for file_index, input_file in enumerate(input_files, start=1):
            with input_file.open("r", encoding="utf-8") as handle:
                while True:
                    event = read_lund_event(handle, input_file)
                    if event is None:
                        break
                    if events_in_output == 0 and not dry_run:
                        output_index += 1
                        output_path = output_dir / f"{prefix}_{output_index:06d}.lund"
                        output = output_path.open("w", encoding="utf-8")
                        stats.output_files = output_index
                    if not dry_run:
                        assert output is not None
                        output.writelines(event)
                    stats.events += 1
                    stats.lines += len(event)
                    events_in_output += 1
                    if events_in_output >= events_per_file:
                        if output is not None:
                            output.close()
                            output = None
                        events_in_output = 0
            print_progress(file_index, total_files, stats, progress_files, events_per_file)
        if dry_run:
            stats.output_files = (stats.events + events_per_file - 1) // events_per_file
    finally:
        if output is not None:
            output.close()
    return stats


def repack_fixed_line_lund_files(
    input_files: list[Path],
    output_dir: Path,
    *,
    events_per_file: int,
    lines_per_event: int,
    prefix: str,
    dry_run: bool,
    progress_files: int = 0,
) -> RepackStats:
    if dry_run:
        return count_fixed_line_lund_files(
            input_files,
            events_per_file=events_per_file,
            lines_per_event=lines_per_event,
            progress_files=progress_files,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = RepackStats()
    output = None
    events_in_output = 0
    output_index = 0
    total_files = len(input_files)
    try:
        for file_index, input_file in enumerate(input_files, start=1):
            with input_file.open("rb") as handle:
                while True:
                    lines_needed = (events_per_file - events_in_output) * lines_per_event
                    lines = list(islice(handle, lines_needed))
                    if not lines:
                        break
                    if len(lines) % lines_per_event != 0:
                        raise ValueError(
                            f"{input_file} ended after {len(lines)} lines in a fixed-line block; "
                            f"expected a multiple of {lines_per_event}"
                        )
                    if output is None:
                        output_index += 1
                        output_path = output_dir / f"{prefix}_{output_index:06d}.lund"
                        output = output_path.open("wb")
                        stats.output_files = output_index
                    assert output is not None
                    output.writelines(lines)
                    events = len(lines) // lines_per_event
                    stats.events += events
                    stats.lines += len(lines)
                    events_in_output += events
                    if events_in_output >= events_per_file:
                        output.close()
                        output = None
                        events_in_output = 0
            print_progress(file_index, total_files, stats, progress_files, events_per_file)
    finally:
        if output is not None:
            output.close()
    return stats


def count_fixed_line_lund_files(
    input_files: list[Path],
    *,
    events_per_file: int,
    lines_per_event: int,
    progress_files: int = 0,
) -> RepackStats:
    stats = RepackStats()
    total_files = len(input_files)
    for file_index, input_file in enumerate(input_files, start=1):
        lines = count_lines(input_file)
        if lines % lines_per_event != 0:
            raise ValueError(
                f"{input_file} has {lines} lines, which is not divisible by "
                f"--fixed-lines-per-event={lines_per_event}"
            )
        stats.lines += lines
        stats.events += lines // lines_per_event
        print_progress(file_index, total_files, stats, progress_files, events_per_file)
    stats.output_files = (stats.events + events_per_file - 1) // events_per_file
    return stats


def count_lines(path: Path, chunk_size: int = 16 * 1024 * 1024) -> int:
    count = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def print_progress(
    file_index: int,
    total_files: int,
    stats: RepackStats,
    progress_files: int,
    events_per_file: int,
) -> None:
    if progress_files <= 0:
        return
    if file_index % progress_files != 0 and file_index != total_files:
        return
    print(
        f"[PROGRESS] files={file_index}/{total_files} events={stats.events} "
        f"output_chunks~={(stats.events + events_per_file - 1) // events_per_file}",
        file=sys.stderr,
        flush=True,
    )


def read_lund_event(handle, input_file: Path) -> list[str] | None:
    header = handle.readline()
    if header == "":
        return None
    stripped = header.strip()
    if not stripped:
        raise ValueError(f"Unexpected blank line in {input_file}")
    try:
        particles = int(stripped.split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Malformed LUND header in {input_file}: {header.rstrip()}") from exc
    if particles < 0:
        raise ValueError(f"Negative particle count in {input_file}: {header.rstrip()}")
    event = [header]
    for _ in range(particles):
        particle = handle.readline()
        if particle == "":
            raise ValueError(f"Unexpected EOF inside event in {input_file}")
        event.append(particle)
    return event


def write_manifests(
    output_dir: Path,
    *,
    prefix: str,
    jobs_per_submission: int,
    relative_paths: bool,
) -> int:
    chunks = sorted(output_dir.glob(f"{prefix}_*.lund"))
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for old_manifest in manifest_dir.glob(f"{prefix}_submission_*.list"):
        old_manifest.unlink()
    count = 0
    for index, start in enumerate(range(0, len(chunks), jobs_per_submission), start=1):
        count += 1
        manifest = manifest_dir / f"{prefix}_submission_{index:03d}.list"
        group = chunks[start : start + jobs_per_submission]
        with manifest.open("w", encoding="utf-8") as handle:
            for path in group:
                item = path.name if relative_paths else str(path.resolve())
                handle.write(f"{item}\n")
    return count


def print_summary(stats: RepackStats, args: argparse.Namespace) -> None:
    submissions = (stats.output_files + args.jobs_per_submission - 1) // args.jobs_per_submission
    mode = "DRY RUN" if args.dry_run else "WROTE"
    print(f"{mode}: {stats.input_files} input files")
    print(f"Events: {stats.events}")
    print(f"Lines: {stats.lines}")
    print(f"Output chunks: {stats.output_files}")
    print(f"Events per output chunk: <= {args.events_per_file}")
    print(f"OSG submissions at <= {args.jobs_per_submission} jobs each: {submissions}")
    if not args.dry_run:
        print(f"Output directory: {args.output_dir}")
        print(f"Manifest directory: {args.output_dir / 'manifests'}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
