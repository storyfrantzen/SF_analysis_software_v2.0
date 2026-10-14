#!/usr/bin/env python3
"""Stage AAO normalization sidecars matching a selected LUND manifest.

The report-input manifests contain ``index``, ``events``, and ``source``
columns.  Born sources normally still sit beside their ``.norm`` sidecars.
Radiative sources may instead be frozen links; in that case an original-source
list maps each LUND basename back to the directory containing its sidecar.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--original-source-list",
        type=Path,
        help="Absolute original LUND paths, one per line, keyed by basename",
    )
    return parser.parse_args()


def source_lookup(path: Path | None) -> dict[str, Path] | None:
    if path is None:
        return None
    mapping: dict[str, Path] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        source = Path(text)
        previous = mapping.get(source.name)
        if previous is not None and previous != source:
            raise ValueError(
                f"{path}:{line_number}: duplicate basename {source.name!r}: "
                f"{previous} and {source}"
            )
        mapping[source.name] = source
    if not mapping:
        raise ValueError(f"original-source list is empty: {path}")
    return mapping


def selected_sidecars(
    manifest: Path,
    original_sources: dict[str, Path] | None,
) -> list[tuple[int, int, Path, Path]]:
    selected: list[tuple[int, int, Path, Path]] = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        required = {"index", "events", "source"}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError(
                f"{manifest}: expected tab-separated columns {sorted(required)}"
            )
        for row in rows:
            index = int(row["index"])
            events = int(row["events"])
            selected_lund = Path(row["source"])
            if original_sources is None:
                original_lund = selected_lund
            else:
                try:
                    original_lund = original_sources[selected_lund.name]
                except KeyError as error:
                    raise ValueError(
                        f"{manifest}: no original source for {selected_lund.name}"
                    ) from error
            norm = original_lund.with_suffix(".norm")
            if not norm.is_file() or norm.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"missing nonempty .norm for manifest index {index}: {norm}"
                )
            selected.append((index, events, selected_lund, norm))
    if not selected:
        raise ValueError(f"selection manifest is empty: {manifest}")
    indices = [item[0] for item in selected]
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate selection index in {manifest}")
    norm_paths = [item[3] for item in selected]
    if len(norm_paths) != len(set(norm_paths)):
        raise ValueError(f"duplicate normalization sidecar selected from {manifest}")
    return selected


def stage(selected: list[tuple[int, int, Path, Path]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records = ["index\tevents\tlund_source\tnorm_source\tstaged"]
    for index, events, lund, norm in selected:
        destination = output / f"{index:05d}_{norm.name}"
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or Path(os.path.realpath(destination)) != norm.resolve():
                raise FileExistsError(
                    f"refusing to replace existing nonmatching path: {destination}"
                )
        else:
            destination.symlink_to(norm.resolve())
        records.append(
            "\t".join(
                (str(index), str(events), str(lund), str(norm), str(destination))
            )
        )
    (output / "manifest.tsv").write_text("\n".join(records) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    lookup = source_lookup(args.original_source_list)
    selected = selected_sidecars(args.manifest, lookup)
    stage(selected, args.output)
    print(f"staged {len(selected)} matching AAO .norm sidecars -> {args.output}")


if __name__ == "__main__":
    main()
