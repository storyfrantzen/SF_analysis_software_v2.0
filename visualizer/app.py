#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np


M_PROTON_GEV = 0.9382720813
M_PI0_GEV = 0.1349768


ROOT_VECTOR_BRANCHES = (
    "selectedRoles",
    "selectedIdx",
    "selectedPid",
    "selectedDet",
    "selectedSector",
    "selectedP",
    "selectedTheta",
    "selectedPhi",
    "topologyPids",
    "topologyPidCounts",
    "topologyPidCountsFT",
    "topologyPidCountsFD",
    "topologyPidCountsCD",
    "topologyPidCountsOther",
)

ROOT_PREFERRED_BRANCHES = (
    "sourceFileId",
    "sourceEventIndex",
    "runNum",
    "eventNum",
    "helicity",
    "charge",
    "passTopology",
    "nPid11",
    "nPid11FT",
    "nPid11FD",
    "nPid11CD",
    "nPid2212",
    "nPid2212FT",
    "nPid2212FD",
    "nPid2212CD",
    "nPid22",
    "nPid22FT",
    "nPid22FD",
    "nPid22CD",
    "electronDet",
    "electronSector",
    "electronP",
    "electronTheta",
    "electronPhi",
    "electronEPCAL",
    "electronEECIN",
    "electronEECOUT",
    "eDet",
    "pDet",
    "g1Det",
    "g2Det",
    "eIdx",
    "pIdx",
    "g1Idx",
    "g2Idx",
    "eSector",
    "pSector",
    "g1Sector",
    "g2Sector",
    "passFiducial",
    "electronPassFiducial",
    "protonPassFiducial",
    "gamma1PassFiducial",
    "gamma2PassFiducial",
    "passSamplingFraction",
    "passExclusivity",
    "Q2",
    "nu",
    "xB",
    "y",
    "W",
    "t",
    "trentoPhi",
    "pi0_p",
    "pi0_theta",
    "pi0_phi",
    "pi0_deltaPhi",
    "pi0_thetaX",
    "m_gg",
    "m2_miss",
    "m2_epX",
    "m2_epi0X",
    "m_eggX",
    "E_miss",
    "pT_miss",
    "theta_e_g1",
    "theta_e_g2",
    "theta_g1_g2",
    "evaluatedCuts",
    "failedCuts",
) + ROOT_VECTOR_BRANCHES

EVENT_OBJECT_FIELDS = (
    "sourceFileId",
    "sourceEventIndex",
    "runNum",
    "eventNum",
    "helicity",
    "charge",
)

REC_OBJECT_FIELDS = (
    "runNum",
    "eventNum",
    "particleIdx",
    "matchedGenIdx",
    "matchAngleDeg",
    "pid",
    "charge",
    "status",
    "det",
    "sector",
    "p",
    "px",
    "py",
    "pz",
    "theta",
    "phi",
    "p_raw",
    "theta_raw",
    "phi_raw",
    "delta_p",
    "delta_theta",
    "delta_phi",
    "beta",
    "chi2pid",
    "trackChi2",
    "trackNDF",
    "trackChi2N",
    "vx",
    "vy",
    "vz",
    "time",
    "xFT",
    "yFT",
    "xDC1",
    "yDC1",
    "xDC2",
    "yDC2",
    "xDC3",
    "yDC3",
    "edgeDC1",
    "edgeDC2",
    "edgeDC3",
    "xPCAL",
    "yPCAL",
    "uPCAL",
    "vPCAL",
    "wPCAL",
    "E_PCAL",
    "uECIN",
    "vECIN",
    "wECIN",
    "E_ECIN",
    "uECOUT",
    "vECOUT",
    "wECOUT",
    "E_ECOUT",
    "edge_cvt1",
    "edge_cvt3",
    "edge_cvt5",
    "edge_cvt7",
    "edge_cvt12",
    "theta_cvt",
    "phi_cvt",
)

GEN_OBJECT_FIELDS = (
    "runNum",
    "eventNum",
    "particleIdx",
    "pid",
    "p",
    "theta",
    "phi",
)

REDUNDANT_COLUMNS = (
    ("minus_t", "t"),
    ("electronIdx", "eIdx"),
    ("electronDet", "eDet"),
    ("eSector", "electronSector"),
    ("protonIdx", "pIdx"),
    ("protonDet", "pDet"),
    ("protonSector", "pSector"),
    ("gammaIdx", "gamma1Idx"),
    ("gammaDet", "gamma1Det"),
    ("gammaSector", "gamma1Sector"),
    ("gamma1Idx", "g1Idx"),
    ("gamma2Idx", "g2Idx"),
    ("gamma1Det", "g1Det"),
    ("gamma2Det", "g2Det"),
    ("gamma1Sector", "g1Sector"),
    ("gamma2Sector", "g2Sector"),
    ("rec_proton_detector", "pDet"),
    ("rec_eIdx", "rec_electronIdx"),
    ("rec_eDet", "rec_electronDet"),
    ("rec_eSector", "rec_electronSector"),
    ("rec_pIdx", "rec_protonIdx"),
    ("rec_pDet", "rec_protonDet"),
    ("rec_proton_detector", "rec_protonDet"),
    ("rec_pSector", "rec_protonSector"),
    ("rec_g1Idx", "rec_gamma1Idx"),
    ("rec_g1Det", "rec_gamma1Det"),
    ("rec_g1Sector", "rec_gamma1Sector"),
    ("rec_g2Idx", "rec_gamma2Idx"),
    ("rec_g2Det", "rec_gamma2Det"),
    ("rec_g2Sector", "rec_gamma2Sector"),
    ("gen_eIdx", "gen_electronIdx"),
    ("gen_eDet", "gen_electronDet"),
    ("gen_eSector", "gen_electronSector"),
    ("gen_pIdx", "gen_protonIdx"),
    ("gen_pDet", "gen_protonDet"),
    ("gen_pSector", "gen_protonSector"),
    ("gen_g1Idx", "gen_gamma1Idx"),
    ("gen_g1Det", "gen_gamma1Det"),
    ("gen_g1Sector", "gen_gamma1Sector"),
    ("gen_g2Idx", "gen_gamma2Idx"),
    ("gen_g2Det", "gen_gamma2Det"),
    ("gen_g2Sector", "gen_gamma2Sector"),
)


PARTICLE_DISPLAY_PREFIXES = (
    ("electron", "electron"),
    ("proton", "proton"),
    ("gamma1", "gamma 1"),
    ("gamma2", "gamma 2"),
    ("gamma", "gamma"),
    ("pi0", "pi0"),
    ("g1", "gamma 1"),
    ("g2", "gamma 2"),
    ("e", "electron"),
    ("p", "proton"),
)


PARTICLE_QUANTITY_DISPLAY_NAMES = {
    "idx": "index",
    "index": "index",
    "particleidx": "index",
    "matchedgenidx": "matched GEN index",
    "matchangledeg": "match angle deg",
    "pid": "pid",
    "charge": "charge",
    "status": "status",
    "det": "detector",
    "detector": "detector",
    "sector": "sector",
    "p": "p",
    "px": "px",
    "py": "py",
    "pz": "pz",
    "praw": "p raw",
    "theta": "theta",
    "thetadeg": "theta deg",
    "thetaraw": "theta raw",
    "thetarawdeg": "theta raw deg",
    "phi": "phi",
    "phideg": "phi deg",
    "phiraw": "phi raw",
    "phirawdeg": "phi raw deg",
    "deltap": "delta p",
    "deltatheta": "delta theta",
    "deltaphi": "delta phi",
    "beta": "beta",
    "chi2pid": "chi2 pid",
    "trackchi2": "track chi2",
    "trackndf": "track NDF",
    "trackchi2n": "track chi2/NDF",
    "vx": "vx",
    "vy": "vy",
    "vz": "vz",
    "time": "time",
    "epcal": "E PCAL",
    "eecin": "E ECIN",
    "eecout": "E ECOUT",
    "samplingfraction": "SF",
    "samplingfractionpcal": "SF PCAL",
    "samplingfractionecin": "SF ECIN",
    "samplingfractionecout": "SF ECOUT",
    "samplingfractionecal": "SF ECAL",
    "passfiducial": "fiducial",
}


DISPLAY_NAMES = {
    "run": "run",
    "runNum": "run",
    "rec_runNum": "REC run",
    "gen_runNum": "GEN run",
    "Q2": "Q2",
    "rec_Q2": "REC Q2",
    "gen_Q2": "GEN Q2",
    "xB": "xB",
    "rec_xB": "REC xB",
    "gen_xB": "GEN xB",
    "t": "-t",
    "t_pi0": "pi0 -t",
    "minus_t": "-t",
    "rec_minus_t": "REC -t",
    "rec_minus_t_pi0": "REC pi0 -t",
    "gen_minus_t": "GEN -t",
    "signed_t": "t",
    "t_min": "-t_min",
    "signed_t_min": "t_min",
    "t_prime": "t'",
    "t_pi0_prime": "pi0 t'",
    "rec_t_min": "REC -t_min",
    "rec_signed_t_min": "REC t_min",
    "rec_t_prime": "REC t'",
    "rec_t_pi0_prime": "REC pi0 t'",
    "gen_t_min": "GEN -t_min",
    "gen_signed_t_min": "GEN t_min",
    "gen_t_prime": "GEN t'",
    "trentoPhi": "phi",
    "trentoPhi_deg": "phi deg",
    "rec_trento_phi": "REC phi",
    "gen_trento_phi": "GEN phi",
    "rec_trento_phi_deg": "REC phi deg",
    "gen_trento_phi_deg": "GEN phi deg",
    "pDet": "proton detector",
    "rec_proton_detector": "REC proton detector",
    "passFiducial": "fiducial",
    "passSamplingFraction": "passes SF cut",
    "passExclusivity": "loose exclusivity",
    "rec_passFiducial": "REC fiducial",
    "rec_passSamplingFraction": "REC passes SF cut",
    "rec_passExclusivity": "REC loose exclusivity",
    "electronSamplingFraction": "electron SF",
    "electronSamplingFractionPCAL": "electron SF PCAL",
    "electronSamplingFractionECIN": "electron SF ECIN",
    "electronSamplingFractionECOUT": "electron SF ECOUT",
    "electronSamplingFractionECAL": "electron SF ECAL",
    "rec_electronSamplingFraction": "REC electron SF",
    "rec_electronSamplingFractionPCAL": "REC electron SF PCAL",
    "rec_electronSamplingFractionECIN": "REC electron SF ECIN",
    "rec_electronSamplingFractionECOUT": "REC electron SF ECOUT",
    "rec_electronSamplingFractionECAL": "REC electron SF ECAL",
    "eIdx": "electron index",
    "pIdx": "proton index",
    "g1Idx": "gamma 1 index",
    "g2Idx": "gamma 2 index",
    "eSector": "electron sector",
    "pSector": "proton sector",
    "g1Sector": "gamma 1 sector",
    "g2Sector": "gamma 2 sector",
    "protonTheta_deg": "protonTheta deg",
    "protonIdx": "proton index",
    "protonSector": "proton sector",
    "gammaIdx": "gamma index",
    "gammaSector": "gamma sector",
    "gamma1Idx": "gamma 1 index",
    "gamma1Sector": "gamma 1 sector",
    "gamma2Idx": "gamma 2 index",
    "gamma2Sector": "gamma 2 sector",
    "electronTheta_deg": "electronTheta deg",
    "electronIdx": "electron index",
    "pi0_theta_deg": "pi0_theta deg",
    "rec_pid": "REC pid",
    "rec_det": "REC detector",
    "rec_p": "REC p",
    "rec_theta": "REC theta",
    "rec_theta_deg": "REC theta deg",
    "rec_phi": "REC phi",
    "rec_phi_deg": "REC phi deg",
    "gen_pid": "GEN pid",
    "gen_p": "GEN p",
    "gen_theta": "GEN theta",
    "gen_theta_deg": "GEN theta deg",
    "gen_phi": "GEN phi",
    "gen_phi_deg": "GEN phi deg",
    "gen_electronP": "GEN electron p",
    "gen_electronTheta": "GEN electron theta",
    "gen_electronTheta_deg": "GEN electron theta deg",
    "gen_electronPhi": "GEN electron phi",
    "gen_electronPhi_deg": "GEN electron phi deg",
    "gen_protonP": "GEN proton p",
    "gen_protonTheta": "GEN proton theta",
    "gen_protonTheta_deg": "GEN proton theta deg",
    "gen_protonPhi": "GEN proton phi",
    "gen_protonPhi_deg": "GEN proton phi deg",
    "gen_gamma1P": "GEN gamma 1 p",
    "gen_gamma1Theta": "GEN gamma 1 theta",
    "gen_gamma1Theta_deg": "GEN gamma 1 theta deg",
    "gen_gamma1Phi": "GEN gamma 1 phi",
    "gen_gamma1Phi_deg": "GEN gamma 1 phi deg",
    "gen_gamma2P": "GEN gamma 2 p",
    "gen_gamma2Theta": "GEN gamma 2 theta",
    "gen_gamma2Theta_deg": "GEN gamma 2 theta deg",
    "gen_gamma2Phi": "GEN gamma 2 phi",
    "gen_gamma2Phi_deg": "GEN gamma 2 phi deg",
    "gen_pi0P": "GEN pi0 p",
    "gen_pi0Theta": "GEN pi0 theta",
    "gen_pi0Theta_deg": "GEN pi0 theta deg",
    "gen_pi0Phi": "GEN pi0 phi",
    "gen_pi0Phi_deg": "GEN pi0 phi deg",
    "electronSector": "electron sector",
    "rec_sector": "REC sector",
    "sector": "sector",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standalone interactive histogram/2D-histogram visualizer "
            "from EPPI0 NPZ artifacts or selected ROOT trees."
        )
    )
    parser.add_argument("input", type=Path, help=".npz sample or selected .root file")
    parser.add_argument("--output", type=Path, required=True, help="Output .html path")
    parser.add_argument("--format", choices=("auto", "npz", "root"), default="auto")
    parser.add_argument(
        "--tree",
        help=(
            "ROOT tree name; defaults to sEvents, rParticles, rEvents, then gEvents"
        ),
    )
    parser.add_argument("--dictionary", type=Path, help="Optional ROOT dictionary shared library")
    parser.add_argument(
        "--root-filter",
        help=(
            "Optional ROOT RDataFrame expression applied before sampling. Converter object "
            "fields use their qualified branch names, such as event.runNum and rec.pid."
        ),
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Optional ROOT branch allow-list. NPZ input always reads all numeric arrays.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=250_000,
        help=(
            "Maximum rows embedded in the HTML. Larger NPZ and ROOT inputs are sampled "
            "deterministically using --seed; use 0 to read all rows."
        ),
    )
    parser.add_argument(
        "--max-source-events",
        type=int,
        default=None,
        help=(
            "Maximum distinct source events embedded from a ROOT input. All particle rows "
            "belonging to each sampled event are retained. This overrides the --max-events "
            "row limit; use 0 to read every source event."
        ),
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--title", default=None, help="Title shown inside the visualizer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_format = args.format
    if input_format == "auto":
        suffix = args.input.suffix.lower()
        input_format = "npz" if suffix == ".npz" else "root" if suffix == ".root" else ""
        if not input_format:
            raise ValueError("Could not infer input format; pass --format npz or --format root")

    if input_format == "npz":
        if args.root_filter:
            raise ValueError("--root-filter requires a ROOT input")
        if args.max_source_events is not None:
            raise ValueError("--max-source-events currently requires a ROOT input")
        log(f"Reading NPZ input {args.input}")
        arrays, metadata = load_npz(args.input)
    else:
        arrays, metadata = load_root(
            args.input,
            args.tree,
            args.dictionary,
            args.columns,
            max_events=args.max_events,
            max_source_events=args.max_source_events,
            root_filter=args.root_filter,
            seed=args.seed,
        )

    log("Preparing embedded data")
    arrays = add_derived_quantities(arrays)
    arrays = normalize_visual_columns(arrays)
    arrays = rectangular_numeric_and_text(arrays)
    row_limit = 0 if args.max_source_events is not None else args.max_events
    arrays, downsample = downsample_arrays(arrays, row_limit, args.seed)
    if input_format == "root" and metadata.get("root_rows_total", 0) > len(next(iter(arrays.values()))):
        downsample = {
            "originalRows": int(metadata["root_rows_total"]),
            "embeddedRows": int(len(next(iter(arrays.values())))),
            "sampled": True,
            "strategy": "deterministic-random",
            "seed": int(args.seed),
        }
        if metadata.get("sampling_unit") == "source-events":
            downsample.update(
                {
                    "unit": "source-events",
                    "originalEvents": int(metadata["root_events_total"]),
                    "embeddedEvents": int(metadata["root_events_read"]),
                }
            )
    if args.root_filter:
        downsample["filter"] = args.root_filter
    payload = build_payload(args.input, arrays, metadata, downsample, args.title)
    log(f"Writing {args.output}")
    html = render_html(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Rows embedded: {payload['rowCount']}")
    print(f"Variables: {len(payload['variables'])}")
    print(f"Wrote {args.output}")
    return 0


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_npz(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    arrays: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for name in data.files:
        value = data[name]
        if name == "metadata_json":
            metadata.update(parse_metadata_json(value))
        elif value.ndim <= 1:
            arrays[name] = value
    metadata.setdefault("format", "npz")
    return arrays, metadata


def load_root(
    path: Path,
    tree_name: str | None,
    dictionary: Path | None,
    requested_columns: list[str] | None,
    max_events: int,
    max_source_events: int | None,
    root_filter: str | None,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if ROOT.IsImplicitMTEnabled():
        ROOT.DisableImplicitMT()
    log(f"Opening ROOT input {path}")
    loaded_dictionary = load_root_dictionary(ROOT, dictionary)

    root_path = str(path.resolve())
    root_file = ROOT.TFile.Open(root_path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")
    tree_name = resolve_root_tree_name(root_file, tree_name)
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find tree {tree_name} in {root_path}")

    entries = int(tree.GetEntries())
    branches = {branch.GetName(): branch for branch in tree.GetListOfBranches()}
    available = set(branches)
    object_aliases = object_branch_aliases(available)
    if requested_columns:
        columns = [name for name in requested_columns if name in available or name in object_aliases]
        missing = [
            name for name in requested_columns
            if name not in available and name not in object_aliases
        ]
        if missing:
            raise RuntimeError(f"Tree {tree_name} is missing requested branches: {missing}")
    else:
        columns = [name for name in ROOT_PREFERRED_BRANCHES if name in available]
        columns.extend(
            sorted(
                name for name, branch in branches.items()
                if name not in columns and is_plain_root_branch(branch)
            )
        )
        columns.extend(name for name in object_aliases if name not in columns)
    root_file.Close()

    filtered_entries = entries
    eligible_entries: np.ndarray | None = None
    event_starts: np.ndarray | None = None
    key_expressions = None
    if max_source_events is not None:
        key_expressions = root_event_key_expressions(available, object_aliases)
        if key_expressions is None:
            raise RuntimeError(
                "--max-source-events requires sourceFileId/sourceEventIndex or runNum/eventNum "
                f"event identifiers in tree {tree_name}"
            )
    if root_filter:
        log(f"Applying ROOT filter: {root_filter}")
        eligible_entries, event_starts = read_filtered_root_selection(
            ROOT,
            root_path,
            tree_name,
            root_filter,
            object_aliases,
            key_expressions,
        )
        filtered_entries = int(eligible_entries.size)
        if filtered_entries == 0:
            raise ValueError(f"ROOT filter selected no rows: {root_filter}")
        log(f"ROOT filter selected {filtered_entries} of {entries} rows")

    event_count: int | None = None
    event_read_count: int | None = None
    if max_source_events is not None:
        log(f"Scanning {filtered_entries} qualifying rows for distinct source events")
        if event_starts is None:
            event_starts = read_root_event_starts(
                ROOT,
                root_path,
                tree_name,
                key_expressions,
            )
        sampled_positions, event_count = sample_event_row_indices(
            event_starts,
            filtered_entries,
            max_source_events,
            seed,
        )
        sample_indices = (
            sampled_positions
            if eligible_entries is None or sampled_positions is None
            else eligible_entries[sampled_positions]
        )
        event_read_count = (
            event_count
            if sampled_positions is None
            else min(max_source_events, event_count)
        )
    else:
        sampled_positions = sample_row_indices(filtered_entries, max_events, seed)
        sample_indices = (
            sampled_positions
            if eligible_entries is None or sampled_positions is None
            else eligible_entries[sampled_positions]
        )
    read_count = filtered_entries if sample_indices is None else len(sample_indices)
    if max_source_events is not None:
        if sample_indices is None:
            log(
                f"Reading {len(columns)} ROOT columns from all {event_count} source events "
                f"({filtered_entries} rows)"
            )
        else:
            log(
                f"Reading {len(columns)} ROOT columns from a deterministic random sample "
                f"of {event_read_count} of {event_count} source events "
                f"({read_count} of {filtered_entries} qualifying rows, seed {seed})"
            )
    elif sample_indices is not None:
        log(
            f"Reading {len(columns)} ROOT columns from a deterministic random sample "
            f"of {read_count} of {filtered_entries} qualifying rows (seed {seed})"
        )
    else:
        log(f"Reading {len(columns)} ROOT columns from {filtered_entries} rows")
    raw = read_root_arrays(
        ROOT,
        root_path,
        tree_name,
        columns,
        aliases=object_branch_aliases(available),
        sample_indices=sample_indices,
        root_filter=root_filter,
        strict=bool(requested_columns),
    )
    arrays = {name: raw[name] for name in columns}
    arrays.update(extract_selected_particle_quantities(raw))
    metadata = {"format": "root", "tree": tree_name}
    if loaded_dictionary:
        metadata["dictionary"] = str(loaded_dictionary)
    if root_filter:
        metadata["root_filter"] = root_filter
        metadata["root_rows_before_filter"] = entries
    if max_source_events is None and max_events > 0 and filtered_entries > max_events:
        metadata["root_rows_total"] = filtered_entries
        metadata["root_rows_read"] = read_count
        metadata["sampling_seed"] = seed
        metadata["sampling_strategy"] = "deterministic-random"
    if max_source_events is not None:
        metadata["root_rows_total"] = filtered_entries
        metadata["root_rows_read"] = read_count
        metadata["root_events_total"] = event_count
        metadata["root_events_read"] = event_read_count
        metadata["sampling_unit"] = "source-events"
        if sample_indices is not None:
            metadata["sampling_seed"] = seed
            metadata["sampling_strategy"] = "deterministic-random"
    return arrays, metadata


def resolve_root_tree_name(root_file: Any, requested: str | None) -> str:
    if requested and root_file.Get(requested):
        return requested
    aliases = {
        "sEvents": ("SelectedEvents", "Events"),
        "SelectedEvents": ("sEvents", "Events"),
        "rParticles": ("ReconstructedParticles", "Events"),
        "ReconstructedParticles": ("rParticles", "Events"),
        "rEvents": ("ReconstructedEvents",),
        "ReconstructedEvents": ("rEvents",),
        "gEvents": ("GeneratedEvents",),
        "GeneratedEvents": ("gEvents",),
    }
    for candidate in aliases.get(requested, ()):
        if root_file.Get(candidate):
            log(f"Warning: using compatible tree {candidate} instead of {requested}")
            return candidate
    if requested:
        return requested
    for candidate in (
        "sEvents",
        "rParticles",
        "rEvents",
        "gEvents",
        "SelectedEvents",
        "ReconstructedParticles",
        "ReconstructedEvents",
        "GeneratedEvents",
        "Events",
    ):
        if root_file.Get(candidate):
            return candidate
    return "sEvents"


def load_root_dictionary(ROOT: Any, dictionary: Path | None) -> Path | None:
    if dictionary is None:
        return None
    candidates = [dictionary]
    if dictionary.suffix == ".dylib":
        candidates.append(dictionary.with_suffix(".so"))
    elif dictionary.suffix == ".so":
        candidates.append(dictionary.with_suffix(".dylib"))

    found_candidate = False
    for candidate in candidates:
        if not candidate.exists():
            continue
        found_candidate = True
        if ROOT.gSystem.Load(str(candidate.resolve())) >= 0:
            if candidate != dictionary:
                print(f"Using ROOT dictionary {candidate} instead of {dictionary}", file=sys.stderr)
            return candidate
        print(f"Warning: could not load ROOT dictionary {candidate}; trying alternatives", file=sys.stderr)

    tried = ", ".join(str(candidate) for candidate in candidates)
    problem = "could not be loaded" if found_candidate else "was not found"
    print(f"Warning: ROOT dictionary {problem} ({tried}); continuing without it", file=sys.stderr)
    return None


def is_plain_root_branch(branch: Any) -> bool:
    class_name = str(branch.GetClassName() or "")
    if class_name:
        return False
    leaves = branch.GetListOfLeaves()
    return bool(leaves and leaves.GetEntries() == 1)


def object_branch_aliases(available: set[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if "event" in available:
        aliases.update({f"event.{field}": f"event.{field}" for field in EVENT_OBJECT_FIELDS})
    if "rec" in available:
        aliases.update({f"rec.{field}": f"rec.{field}" for field in REC_OBJECT_FIELDS})
    if "gen" in available:
        aliases.update({f"gen.{field}": f"gen.{field}" for field in GEN_OBJECT_FIELDS})
    return aliases


def root_event_key_expressions(
    available: set[str],
    aliases: dict[str, str],
) -> tuple[tuple[str, str], tuple[str, str]] | None:
    candidates = (
        ("event.sourceFileId", "event.sourceEventIndex"),
        ("sourceFileId", "sourceEventIndex"),
        ("event.runNum", "event.eventNum"),
        ("runNum", "eventNum"),
        ("rec.runNum", "rec.eventNum"),
    )
    for primary, secondary in candidates:
        if primary not in available and primary not in aliases:
            continue
        if secondary not in available and secondary not in aliases:
            continue
        return (
            (primary, aliases.get(primary, primary)),
            (secondary, aliases.get(secondary, secondary)),
        )
    return None


def define_root_aliases(frame: Any, aliases: dict[str, str]) -> Any:
    for name, expression in aliases.items():
        if name != expression:
            frame = frame.Define(name, expression)
    return frame


def read_filtered_root_selection(
    ROOT: Any,
    root_path: str,
    tree_name: str,
    root_filter: str,
    aliases: dict[str, str],
    key_expressions: tuple[tuple[str, str], tuple[str, str]] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    frame = define_root_aliases(ROOT.RDataFrame(tree_name, root_path), aliases)
    output_names = ["rdfentry_"]
    key_names = ("__sf_filtered_event_key_primary", "__sf_filtered_event_key_secondary")
    if key_expressions is not None:
        for output_name, (logical_name, expression) in zip(key_names, key_expressions):
            resolved_expression = logical_name if logical_name in aliases else expression
            frame = frame.Define(
                output_name,
                f"static_cast<ULong64_t>({resolved_expression})",
            )
        output_names.extend(key_names)
    try:
        values = frame.Filter(root_filter).AsNumpy(output_names)
    except RuntimeError as error:
        raise RuntimeError(f"ROOT filter failed ({root_filter!r}): {error}") from error

    entries = np.asarray(values["rdfentry_"], dtype=np.uint64)
    if key_expressions is None or entries.size == 0:
        return entries, None
    primary = np.asarray(values[key_names[0]])
    secondary = np.asarray(values[key_names[1]])
    boundaries = np.empty(entries.size, dtype=bool)
    boundaries[0] = True
    boundaries[1:] = (primary[1:] != primary[:-1]) | (secondary[1:] != secondary[:-1])
    return entries, np.flatnonzero(boundaries).astype(np.uint64, copy=False)


def read_root_event_starts(
    ROOT: Any,
    root_path: str,
    tree_name: str,
    key_expressions: tuple[tuple[str, str], tuple[str, str]],
) -> np.ndarray:
    if not hasattr(ROOT, "SFVisualizerEventBoundaries"):
        declared = ROOT.gInterpreter.Declare(
            """
            namespace SFVisualizerEventBoundaries {
            bool initialized = false;
            ULong64_t previous_primary = 0;
            ULong64_t previous_secondary = 0;
            void reset() {
                initialized = false;
                previous_primary = 0;
                previous_secondary = 0;
            }
            bool starts_event(ULong64_t primary, ULong64_t secondary) {
                const bool starts = !initialized
                    || primary != previous_primary
                    || secondary != previous_secondary;
                initialized = true;
                previous_primary = primary;
                previous_secondary = secondary;
                return starts;
            }
            }
            """
        )
        if not declared:
            raise RuntimeError("Could not initialize ROOT source-event boundary scanner")

    ROOT.SFVisualizerEventBoundaries.reset()
    frame = ROOT.RDataFrame(tree_name, root_path)
    output_names = ("__sf_event_key_primary", "__sf_event_key_secondary")
    for output_name, (_, expression) in zip(output_names, key_expressions):
        frame = frame.Define(output_name, f"static_cast<ULong64_t>({expression})")
    try:
        starts = frame.Filter(
            "SFVisualizerEventBoundaries::starts_event("
            f"{output_names[0]}, {output_names[1]})"
        ).AsNumpy(["rdfentry_"])
        return np.asarray(starts["rdfentry_"], dtype=np.uint64)
    finally:
        ROOT.SFVisualizerEventBoundaries.reset()


def read_root_arrays(
    ROOT: Any,
    root_path: str,
    tree_name: str,
    columns: list[str],
    *,
    aliases: dict[str, str],
    sample_indices: np.ndarray | None,
    root_filter: str | None,
    strict: bool,
) -> dict[str, Any]:
    remaining = list(columns)
    if sample_indices is not None:
        set_root_sample_entries(ROOT, sample_indices)
    try:
        while remaining:
            try:
                frame = ROOT.RDataFrame(tree_name, root_path)
                defined_aliases: set[str] = set()
                if root_filter:
                    frame = define_root_aliases(frame, aliases)
                    defined_aliases.update(aliases)
                    frame = frame.Filter(root_filter)
                if sample_indices is not None:
                    frame = frame.Filter("SFVisualizerRootSampling::includes(rdfentry_)")
                for name in remaining:
                    if name in aliases and name not in defined_aliases:
                        frame = frame.Define(name, aliases[name])
                return frame.AsNumpy(remaining)
            except RuntimeError as error:
                match = re.search(r'The column named "([^"]+)"', str(error))
                if strict or not match or match.group(1) not in remaining:
                    raise
                column = match.group(1)
                print(
                    f"Warning: skipping ROOT branch {column!r}; its type needs a dictionary",
                    file=sys.stderr,
                )
                remaining.remove(column)
                columns[:] = remaining
    finally:
        if sample_indices is not None:
            ROOT.SFVisualizerRootSampling.clear_entries()
    raise RuntimeError(f"No readable scalar or vector branches found in {tree_name}")


def set_root_sample_entries(ROOT: Any, indices: np.ndarray) -> None:
    if not hasattr(ROOT, "SFVisualizerRootSampling"):
        declared = ROOT.gInterpreter.Declare(
            """
            #include <unordered_set>
            #include <vector>
            namespace SFVisualizerRootSampling {
            std::unordered_set<ULong64_t> entries;
            void set_entries(const std::vector<ULong64_t>& values) {
                entries.clear();
                entries.reserve(values.size());
                entries.insert(values.begin(), values.end());
            }
            void clear_entries() {
                entries.clear();
                entries.rehash(0);
            }
            bool includes(ULong64_t entry) {
                return entries.find(entry) != entries.end();
            }
            }
            """
        )
        if not declared:
            raise RuntimeError("Could not initialize the ROOT entry sampler")
    ROOT.SFVisualizerRootSampling.set_entries(np.asarray(indices, dtype=np.uint64))


def extract_selected_particle_quantities(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    needed = {"selectedRoles", "selectedP", "selectedTheta", "selectedPhi"}
    if not needed.issubset(raw):
        return {}
    roles = raw["selectedRoles"]
    size = len(roles)
    output: dict[str, np.ndarray] = {}
    for role in ("electron", "proton", "gamma", "gamma1", "gamma2"):
        output[f"{role}Idx"] = np.full(size, -999, dtype=np.int64)
        output[f"{role}P"] = np.full(size, np.nan, dtype=float)
        output[f"{role}Theta"] = np.full(size, np.nan, dtype=float)
        output[f"{role}Phi"] = np.full(size, np.nan, dtype=float)
        output[f"{role}Det"] = np.full(size, -999, dtype=np.int64)
        output[f"{role}Sector"] = np.full(size, -999, dtype=np.int64)
    selected_idx = raw.get("selectedIdx")
    selected_det = raw.get("selectedDet")
    selected_sector = raw.get("selectedSector")
    for row in range(size):
        row_roles = vector_to_list(roles[row])
        row_idx = vector_to_list(selected_idx[row]) if selected_idx is not None else []
        row_p = vector_to_list(raw["selectedP"][row])
        row_theta = vector_to_list(raw["selectedTheta"][row])
        row_phi = vector_to_list(raw["selectedPhi"][row])
        row_det = vector_to_list(selected_det[row]) if selected_det is not None else []
        row_sector = vector_to_list(selected_sector[row]) if selected_sector is not None else []
        seen: dict[str, int] = {}
        for index, role_value in enumerate(row_roles):
            role = str(role_value)
            if role not in ("electron", "proton", "gamma"):
                continue
            count = seen.get(role, 0) + 1
            seen[role] = count
            output_roles = [role]
            if role == "gamma":
                if count == 1:
                    output_roles.append("gamma1")
                elif count == 2:
                    output_roles = ["gamma2"]
                else:
                    continue
            elif count > 1:
                continue
            for output_role in output_roles:
                if index < len(row_idx):
                    output[f"{output_role}Idx"][row] = int(as_float(row_idx[index]))
                if index < len(row_p):
                    output[f"{output_role}P"][row] = as_float(row_p[index])
                if index < len(row_theta):
                    output[f"{output_role}Theta"][row] = as_float(row_theta[index])
                if index < len(row_phi):
                    output[f"{output_role}Phi"][row] = as_float(row_phi[index])
                if index < len(row_det):
                    output[f"{output_role}Det"][row] = int(as_float(row_det[index]))
                if index < len(row_sector):
                    output[f"{output_role}Sector"][row] = int(as_float(row_sector[index]))
    return output


def vector_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return []


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_metadata_json(value: np.ndarray) -> dict[str, Any]:
    try:
        raw = value.item() if value.shape == () else str(value)
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def add_derived_quantities(arrays: dict[str, Any]) -> dict[str, Any]:
    derived = dict(arrays)
    if "t" in derived and "signed_t" not in derived:
        derived["signed_t"] = -np.asarray(derived["t"], dtype=float)
    for source, target in (
        ("t", "minus_t"),
        ("rec_minus_t", "rec_t_signed"),
        ("gen_minus_t", "gen_t_signed"),
    ):
        if source in derived and target not in derived:
            values = np.asarray(derived[source], dtype=float)
            derived[target] = values if source == "t" else -values
    add_tmin_quantities(derived, "", "t", "t_pi0")
    add_tmin_quantities(derived, "rec_", "rec_minus_t", "rec_minus_t_pi0")
    add_tmin_quantities(derived, "gen_", "gen_minus_t")
    for source in list(derived):
        lower = source.lower()
        if ("theta" in lower or "phi" in lower) and not source.endswith("_deg"):
            values = np.asarray(derived[source])
            if values.dtype.kind in "fiu" and finite_fraction(values) > 0:
                finite = values[np.isfinite(values.astype(float))]
                if finite.size and np.nanmax(np.abs(finite)) <= 2.0 * math.pi + 1.0e-6:
                    degrees = np.asarray(values, dtype=float) * 180.0 / math.pi
                    if is_wrapped_phi_column_name(source):
                        degrees = np.mod(degrees, 360.0)
                    derived[f"{source}_deg"] = degrees
    if "rec_proton_detector" in derived and "pDet" not in derived:
        derived["pDet"] = derived["rec_proton_detector"]
    if "rec_selected" in derived:
        derived["rec_not_selected"] = ~np.asarray(derived["rec_selected"], dtype=bool)
    add_sampling_fraction_quantities(derived, "")
    add_sampling_fraction_quantities(derived, "rec_")
    add_cut_result_quantities(derived)
    return derived


def add_cut_result_quantities(arrays: dict[str, Any]) -> None:
    """Expand evaluated/failed CSV cut sets into filterable passCut_* arrays."""
    for prefix in ("", "rec_"):
        evaluated_name = f"{prefix}evaluatedCuts"
        failed_name = f"{prefix}failedCuts"
        if evaluated_name not in arrays and failed_name not in arrays:
            continue

        reference = arrays.get(evaluated_name, arrays.get(failed_name))
        rows = len(np.asarray(reference))
        # PyROOT may expose std::string branches as per-row character arrays
        # rather than a rectangular NumPy string array.  Keep the rows in their
        # original representation and normalize each value individually.
        evaluated_values = arrays.get(evaluated_name, np.full(rows, "", dtype=str))
        failed_values = arrays.get(failed_name, np.full(rows, "", dtype=str))
        evaluated_sets: list[set[str]] = []
        failed_sets: list[set[str]] = []
        cut_names: set[str] = set()
        for row in range(rows):
            evaluated = csv_name_set(root_text_value(evaluated_values[row]))
            failed = csv_name_set(root_text_value(failed_values[row]))
            evaluated.update(failed)  # Supports older files with failedCuts only.
            evaluated_sets.append(evaluated)
            failed_sets.append(failed)
            cut_names.update(evaluated)

        used_names: set[str] = set(arrays)
        for cut_name in sorted(cut_names):
            base = f"{prefix}passCut_{sanitize_cut_name(cut_name)}"
            quantity_name = base
            suffix = 2
            while quantity_name in used_names:
                quantity_name = f"{base}_{suffix}"
                suffix += 1
            used_names.add(quantity_name)
            values = np.full(rows, np.nan, dtype=float)
            for row, evaluated in enumerate(evaluated_sets):
                if cut_name in evaluated:
                    values[row] = 0.0 if cut_name in failed_sets[row] else 1.0
            arrays[quantity_name] = values


def csv_name_set(value: Any) -> set[str]:
    return {name.strip() for name in str(value).split(",") if name.strip()}


def root_text_value(value: Any) -> str:
    """Normalize scalar text and PyROOT's per-row string representations."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return root_text_value(value.item())
    try:
        items = list(value)
    except TypeError:
        return str(value)
    if not items:
        return ""
    if all(isinstance(item, (int, np.integer)) for item in items):
        return bytes(int(item) for item in items if int(item) != 0).decode(
            "utf-8", errors="replace"
        )
    text_items = [root_text_value(item) for item in items]
    separator = "" if all(len(item) <= 1 for item in text_items) else ","
    return separator.join(text_items)


def sanitize_cut_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return sanitized or "unnamed"


def add_tmin_quantities(
    arrays: dict[str, Any],
    prefix: str,
    minus_t_name: str,
    pi0_minus_t_name: str | None = None,
) -> None:
    q2_name = f"{prefix}Q2"
    xb_name = f"{prefix}xB"
    if q2_name not in arrays or xb_name not in arrays:
        return
    t_min = positive_t_min(np.asarray(arrays[q2_name], dtype=float), np.asarray(arrays[xb_name], dtype=float))
    if not np.count_nonzero(np.isfinite(t_min)):
        return
    arrays.setdefault(f"{prefix}t_min", t_min)
    arrays.setdefault(f"{prefix}signed_t_min", -t_min)
    if minus_t_name in arrays:
        arrays.setdefault(f"{prefix}t_prime", np.asarray(arrays[minus_t_name], dtype=float) - t_min)
    if pi0_minus_t_name and pi0_minus_t_name in arrays:
        arrays.setdefault(f"{prefix}t_pi0_prime", np.asarray(arrays[pi0_minus_t_name], dtype=float) - t_min)


def positive_t_min(q2: np.ndarray, xb: np.ndarray) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    xb = np.asarray(xb, dtype=float)
    result = np.full_like(q2, np.nan, dtype=float)
    valid = np.isfinite(q2) & np.isfinite(xb) & (q2 > 0.0) & (xb > 0.0) & (xb < 1.0)
    if not np.any(valid):
        return result

    mp2 = M_PROTON_GEV * M_PROTON_GEV
    mpi2 = M_PI0_GEV * M_PI0_GEV
    with np.errstate(divide="ignore", invalid="ignore"):
        w2 = mp2 + q2 * (1.0 / xb - 1.0)
    valid &= np.isfinite(w2) & (w2 > (M_PROTON_GEV + M_PI0_GEV) ** 2)
    if not np.any(valid):
        return result

    w = np.sqrt(w2[valid])
    q2_valid = q2[valid]
    w2_valid = w2[valid]
    q0_cm = (w2_valid - mp2 - q2_valid) / (2.0 * w)
    q_cm2 = q0_cm * q0_cm + q2_valid
    pi0_energy_cm = (w2_valid + mpi2 - mp2) / (2.0 * w)
    pi0_p_cm2 = kallen(w2_valid, mpi2, mp2) / (4.0 * w2_valid)
    physical = np.isfinite(q_cm2) & np.isfinite(pi0_p_cm2) & (q_cm2 > 0.0) & (pi0_p_cm2 > 0.0)
    valid_indices = np.flatnonzero(valid)
    if not np.any(physical):
        return result

    q_cm = np.sqrt(q_cm2[physical])
    pi0_p_cm = np.sqrt(pi0_p_cm2[physical])
    q0_cm = q0_cm[physical]
    pi0_energy_cm = pi0_energy_cm[physical]
    q2_physical = q2_valid[physical]
    t_forward = mpi2 - q2_physical - 2.0 * q0_cm * pi0_energy_cm + 2.0 * q_cm * pi0_p_cm
    t_backward = mpi2 - q2_physical - 2.0 * q0_cm * pi0_energy_cm - 2.0 * q_cm * pi0_p_cm
    signed_t_min = np.maximum(t_forward, t_backward)
    result[valid_indices[physical]] = -signed_t_min
    return result


def kallen(x: np.ndarray, y: float, z: float) -> np.ndarray:
    return x * x + y * y + z * z - 2.0 * x * y - 2.0 * x * z - 2.0 * y * z


def add_sampling_fraction_quantities(arrays: dict[str, Any], prefix: str) -> None:
    momentum_name = f"{prefix}electronP"
    if momentum_name not in arrays:
        return
    momentum = np.asarray(arrays[momentum_name], dtype=float)
    pcal_name = f"{prefix}electronEPCAL"
    ecin_name = f"{prefix}electronEECIN"
    ecout_name = f"{prefix}electronEECOUT"
    pcal = np.asarray(arrays[pcal_name], dtype=float) if pcal_name in arrays else None
    ecin = np.asarray(arrays[ecin_name], dtype=float) if ecin_name in arrays else None
    ecout = np.asarray(arrays[ecout_name], dtype=float) if ecout_name in arrays else None
    if pcal is not None:
        arrays.setdefault(f"{prefix}electronSamplingFractionPCAL", safe_ratio(pcal, momentum))
    if ecin is not None:
        arrays.setdefault(f"{prefix}electronSamplingFractionECIN", safe_ratio(ecin, momentum))
    if ecout is not None:
        arrays.setdefault(f"{prefix}electronSamplingFractionECOUT", safe_ratio(ecout, momentum))
    parts = [part for part in (pcal, ecin, ecout) if part is not None]
    if parts:
        total = np.zeros_like(momentum, dtype=float)
        for part in parts:
            total = total + part
        arrays.setdefault(f"{prefix}electronSamplingFraction", safe_ratio(total, momentum))
        arrays.setdefault(f"{prefix}electronSamplingFractionECAL", safe_ratio(total, momentum))


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(denominator, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def normalize_visual_columns(arrays: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arrays)
    for duplicate, canonical in REDUNDANT_COLUMNS:
        if duplicate in normalized and canonical in normalized:
            normalized.pop(duplicate)

    for name in list(normalized):
        if name.endswith("_deg"):
            continue
        if is_angle_column_name(name) and f"{name}_deg" in normalized:
            normalized.pop(name)
    if has_split_gamma_quantities(normalized):
        for name in list(normalized):
            if is_generic_gamma_quantity(name):
                normalized.pop(name)
    return normalized


def has_split_gamma_quantities(arrays: dict[str, Any]) -> bool:
    return any(name.startswith(("gamma1", "gamma2", "g1", "g2")) for name in arrays)


def is_generic_gamma_quantity(name: str) -> bool:
    return bool(re.fullmatch(r"gamma(?:Idx|Index|P|Theta|Theta_deg|Phi|Phi_deg|Det|Sector)", name))


def is_angle_column_name(name: str) -> bool:
    lower = name.lower()
    return "theta" in lower or "phi" in lower


def is_wrapped_phi_column_name(name: str) -> bool:
    lower = name.lower()
    return "phi" in lower and "delta" not in lower


def finite_fraction(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    try:
        return float(np.count_nonzero(np.isfinite(values.astype(float)))) / float(values.size)
    except (TypeError, ValueError):
        return 0.0


def rectangular_numeric_and_text(arrays: dict[str, Any]) -> dict[str, np.ndarray]:
    lengths = [np.asarray(value).shape[0] for value in arrays.values() if np.asarray(value).ndim == 1]
    if not lengths:
        raise ValueError("No one-dimensional arrays found")
    length = max(set(lengths), key=lengths.count)
    cleaned: dict[str, np.ndarray] = {}
    for name, raw in arrays.items():
        value = np.asarray(raw)
        if value.ndim != 1 or value.shape[0] != length:
            continue
        if value.dtype.kind in "biuf":
            cleaned[name] = value
        elif value.dtype.kind in "US":
            cleaned[name] = value.astype(str)
        elif value.dtype.kind == "O":
            if all(is_scalar_text(item) for item in value[: min(value.size, 100)]):
                cleaned[name] = value.astype(str)
    if not any(array.dtype.kind in "biuf" for array in cleaned.values()):
        raise ValueError("No numeric scalar columns found")
    return cleaned


def is_scalar_text(value: Any) -> bool:
    return isinstance(value, (str, bytes, np.str_))


def sample_row_indices(row_count: int, max_events: int, seed: int) -> np.ndarray | None:
    if max_events <= 0 or row_count <= max_events:
        return None
    rng = random.Random(seed)
    return np.asarray(sorted(rng.sample(range(row_count), max_events)), dtype=np.uint64)


def sample_event_row_indices(
    event_starts: np.ndarray,
    row_count: int,
    max_source_events: int,
    seed: int,
) -> tuple[np.ndarray | None, int]:
    starts = np.asarray(event_starts, dtype=np.int64)
    if starts.ndim != 1:
        raise ValueError("ROOT event boundaries must be a one-dimensional array")
    if row_count < 0:
        raise ValueError("ROOT row count cannot be negative")
    if row_count == 0:
        if starts.size:
            raise ValueError("An empty ROOT tree cannot contain event boundaries")
        return None, 0
    if starts.size == 0 or starts[0] != 0 or np.any(starts[1:] <= starts[:-1]) or starts[-1] >= row_count:
        raise ValueError("ROOT event boundaries must start at row zero and increase within the tree")
    event_count = int(starts.size)
    if max_source_events <= 0 or event_count <= max_source_events:
        return None, event_count

    rng = random.Random(seed)
    selected_events = np.zeros(event_count, dtype=bool)
    selected_events[rng.sample(range(event_count), max_source_events)] = True
    event_row_counts = np.diff(np.append(starts, row_count))
    selected_rows = np.flatnonzero(np.repeat(selected_events, event_row_counts))
    return selected_rows.astype(np.uint64, copy=False), event_count


def downsample_arrays(
    arrays: dict[str, np.ndarray],
    max_events: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    row_count = next(iter(arrays.values())).shape[0]
    indices = sample_row_indices(row_count, max_events, seed)
    if indices is None:
        return arrays, {"originalRows": int(row_count), "embeddedRows": int(row_count), "sampled": False}
    sampled = {name: value[indices] for name, value in arrays.items()}
    return sampled, {
        "originalRows": int(row_count),
        "embeddedRows": int(max_events),
        "sampled": True,
        "strategy": "deterministic-random",
        "seed": int(seed),
    }


def build_payload(
    input_path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    downsample: dict[str, Any],
    title: str | None,
) -> dict[str, Any]:
    variables: list[dict[str, Any]] = []
    encoded_columns: dict[str, Any] = {}
    categorical_filters: list[dict[str, Any]] = []
    text_filters: list[dict[str, Any]] = []
    sector_splits = sector_split_candidates(arrays)

    for name in sorted(arrays, key=sort_key):
        values = arrays[name]
        if values.dtype.kind in "biuf":
            numeric = numeric_array(values)
            finite = numeric[np.isfinite(numeric)]
            if finite.size == 0:
                continue
            display_min, display_max = default_display_range(name, finite)
            variables.append(
                {
                    "name": name,
                    "label": label_for(name),
                    "min": display_min,
                    "max": display_max,
                    "mean": float(np.mean(finite)),
                    "finite": int(finite.size),
                    "integer": bool(is_integer_category(name)),
                    "group": variable_group_label(name),
                }
            )
            encoded_columns[name] = encode_float32(numeric)
            filter_info = categorical_filter_info(name, numeric)
            if filter_info:
                categorical_filters.append(filter_info)
        elif values.dtype.kind in "US":
            unique = sorted({str(item) for item in values if str(item)})
            if 0 < len(unique) <= 40:
                text_filters.append({"name": name, "label": label_for(name), "values": unique[:40]})
            encoded_columns[name] = values.astype(str).tolist()

    categorical_filters.sort(key=categorical_filter_sort_key)
    text_filters.sort(key=lambda item: label_for(str(item["name"])).lower())

    preferred_x = first_present(
        variables,
        ("rec_minus_t", "t", "minus_t", "gen_minus_t", "Q2", "rec_Q2"),
    )
    preferred_y = first_present(
        variables,
        ("protonTheta_deg", "theta_p_deg", "rec_theta_deg", "gen_theta_deg", "rec_Q2", "Q2", "xB"),
        fallback_index=1,
    )
    return {
        "title": title or f"Interactive histograms: {input_path.name}",
        "source": str(input_path),
        "metadata": metadata,
        "downsample": downsample,
        "rowCount": int(next(iter(arrays.values())).shape[0]),
        "variables": variables,
        "columns": encoded_columns,
        "categoricalFilters": categorical_filters,
        "textFilters": text_filters,
        "sectorSplits": sector_splits,
        "defaultX": preferred_x,
        "defaultY": preferred_y,
    }


def sector_split_candidates(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for name, values in arrays.items():
        if "sector" not in name.lower():
            continue
        if values.dtype.kind not in "biuf":
            continue
        numeric = np.asarray(values, dtype=float)
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0 or not np.all(np.isclose(finite, np.rint(finite))):
            continue
        unique = np.unique(finite.astype(np.int64))
        if unique.size >= 2 and set(unique.tolist()).issubset({0, 1, 2, 3, 4, 5, 6}):
            candidates.append({"name": name, "label": label_for(name)})
    return sorted(candidates, key=lambda item: sort_key(str(item["name"])))


def default_display_range(name: str, finite: np.ndarray) -> tuple[float, float]:
    if is_wrapped_phi_degree_column(name):
        return 0.0, 360.0

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    if is_detector_code_column(name):
        return data_min - 0.5, data_max + 0.5

    if is_missing_mass_squared_column(name) and finite.size >= 20:
        robust_min, robust_max = np.quantile(finite, (0.01, 0.99))
        robust_min = float(robust_min)
        robust_max = float(robust_max)
        if np.isfinite(robust_min) and np.isfinite(robust_max) and robust_max > robust_min:
            padding = 0.05 * (robust_max - robust_min)
            return robust_min - padding, robust_max + padding

    return data_min, data_max


def is_detector_code_column(name: str) -> bool:
    canonical = canonical_variable_name(name)
    return canonical == "det" or canonical.endswith("det") or canonical.endswith("detector")


def is_missing_mass_squared_column(name: str) -> bool:
    canonical = canonical_variable_name(name)
    return canonical in {"m2miss", "m2epx", "m2epi0x"} or canonical.startswith("missingmass2")


def is_wrapped_phi_degree_column(name: str) -> bool:
    return name.endswith("_deg") and is_wrapped_phi_column_name(name[:-4])


def sort_key(name: str) -> tuple[int, int, int, str, str]:
    return (
        variable_group_rank(name),
        quantity_sort_rank(name),
        source_sort_rank(name),
        label_for(name).lower(),
        name.lower(),
    )


def variable_group_rank(name: str) -> int:
    lowered = name.lower()
    canonical = canonical_variable_name(name)
    if lowered.startswith("derived_"):
        return 13
    if is_event_variable(name):
        return 0
    if is_global_kinematic_variable(name):
        return 1
    if (
        is_pass_flag(name)
        or lowered in {"rec_selected", "rec_not_selected"}
        or canonical in {"passtopology", "failedcuts", "selected", "notselected"}
    ):
        return 2
    particle_rank = particle_variable_group_rank(name)
    if particle_rank is not None:
        return particle_rank
    if is_mass_or_exclusivity_variable(name):
        return 9
    if lowered.startswith(("rec_", "rec.")):
        return 10
    if lowered.startswith(("gen_", "gen.")):
        return 11
    if is_detector_geometry_variable(name):
        return 12
    return 14


def variable_group_label(name: str) -> str:
    labels = {
        0: "Event",
        1: "Kinematics",
        2: "Selections",
        3: "Electron",
        4: "Proton",
        5: "Gamma 1",
        6: "Gamma 2",
        7: "Gamma",
        8: "Pi0",
        9: "Masses / Exclusivity",
        10: "REC particle",
        11: "GEN particle",
        12: "Detector / Geometry",
        13: "Derived",
    }
    return labels.get(variable_group_rank(name), "Other")


def canonical_variable_name(name: str) -> str:
    lowered = name.lower()
    for prefix in ("rec_", "gen_", "rec.", "gen.", "event."):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):]
            break
    return lowered.replace("_", "").replace(".", "")


def particle_variable_group_rank(name: str) -> int | None:
    canonical = canonical_variable_name(name)
    if canonical.startswith("electron") or canonical in {"edet", "esector", "eidx"}:
        return 3
    if canonical.startswith("proton") or canonical in {"pdet", "psector", "pidx"}:
        return 4
    if canonical.startswith("gamma1") or canonical.startswith("g1"):
        return 5
    if canonical.startswith("gamma2") or canonical.startswith("g2"):
        return 6
    if canonical.startswith("gamma"):
        return 7
    if canonical.startswith("pi0"):
        return 8
    return None


def is_event_variable(name: str) -> bool:
    lowered = name.lower()
    canonical = canonical_variable_name(name)
    if is_run_number_column(name) or canonical in {"sourcefileid", "sourceeventindex", "eventnum"}:
        return True
    if lowered.startswith(("rec_", "gen_", "rec.", "gen.")):
        return False
    return canonical in {
        "helicity",
        "charge",
    }


def is_global_kinematic_variable(name: str) -> bool:
    canonical = canonical_variable_name(name)
    return canonical in {
        "q2",
        "nu",
        "xb",
        "y",
        "w",
        "t",
        "signedt",
        "minust",
        "tmin",
        "signedtmin",
        "tprime",
        "tpi0",
        "minustpi0",
        "tpi0prime",
        "trentophi",
        "trentophideg",
    }


def is_mass_or_exclusivity_variable(name: str) -> bool:
    canonical = canonical_variable_name(name)
    return (
        canonical.startswith("m2")
        or canonical.startswith("mgg")
        or "miss" in canonical
        or canonical in {"meggx", "thetaeg1", "thetaeg2", "thetag1g2"}
    )


def is_detector_geometry_variable(name: str) -> bool:
    canonical = canonical_variable_name(name)
    return (
        canonical.endswith("det")
        or canonical.endswith("detector")
        or "sector" in canonical
        or canonical.startswith(("xdc", "ydc", "edge", "xft", "yft", "xpcal", "ypcal"))
        or canonical.startswith(("upcal", "vpcal", "wpcal", "uecin", "vecin", "wecin", "uecout", "vecout", "wecout"))
    )


def source_sort_rank(name: str) -> int:
    lowered = name.lower()
    if lowered.startswith(("rec_", "rec.")):
        return 1
    if lowered.startswith(("gen_", "gen.")):
        return 2
    return 0


def quantity_sort_rank(name: str) -> int:
    canonical = canonical_variable_name(name)
    core = particle_quantity_core(canonical)
    exact_order = {
        "sourcefileid": 0,
        "run": 1,
        "runnum": 1,
        "eventnum": 2,
        "sourceeventindex": 3,
        "helicity": 4,
        "charge": 5,
        "passtopology": 10,
        "passfiducial": 11,
        "passsamplingfraction": 12,
        "passexclusivity": 13,
        "recselected": 14,
        "selected": 14,
        "recnotselected": 15,
        "notselected": 15,
        "failedcuts": 16,
        "q2": 20,
        "nu": 21,
        "xb": 22,
        "y": 23,
        "w": 24,
        "t": 25,
        "minust": 25,
        "tmin": 26,
        "signedtmin": 27,
        "tprime": 28,
        "tpi0": 29,
        "minustpi0": 29,
        "tpi0prime": 30,
        "signedt": 31,
        "trentophi": 32,
        "trentophideg": 32,
        "mgg": 40,
        "m2miss": 41,
        "m2epx": 42,
        "m2epi0x": 43,
        "meggx": 44,
        "emiss": 45,
        "ptmiss": 46,
        "thetaeg1": 47,
        "thetaeg2": 48,
        "thetag1g2": 49,
    }
    if canonical in exact_order:
        return exact_order[canonical]
    if core in exact_order:
        return exact_order[core]
    if core in {"pid", "particleidx", "idx", "index"}:
        return 100
    if core in {"matchedgenidx", "matchangledeg"}:
        return 101
    if core in {"charge", "status", "det", "detector", "sector"}:
        return 102
    if core in {"p", "praw"}:
        return 110
    if core in {"px", "py", "pz"}:
        return 111
    if core in {"theta", "thetadeg", "thetaraw", "thetarawdeg"}:
        return 120
    if core in {"phi", "phideg", "phiraw", "phirawdeg"}:
        return 121
    if core.startswith("delta"):
        return 122
    if core in {"beta", "chi2pid", "trackchi2", "trackndf", "trackchi2n"}:
        return 130
    if core.startswith("e") or "samplingfraction" in core:
        return 140
    if core in {"vx", "vy", "vz", "time"}:
        return 150
    if core.startswith(("x", "y", "u", "v", "w", "edge")):
        return 160
    return 900


def particle_quantity_core(canonical: str) -> str:
    for prefix in ("electron", "proton", "gamma1", "gamma2", "gamma", "pi0", "g1", "g2"):
        if canonical.startswith(prefix):
            return canonical[len(prefix):]
    if canonical in {"edet", "esector", "eidx", "pdet", "psector", "pidx"}:
        return canonical[1:]
    return canonical


def numeric_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "b":
        return values.astype(np.float32)
    return values.astype(np.float32, copy=False)


def categorical_filter_info(name: str, values: np.ndarray) -> dict[str, Any] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    integers = np.all(np.isclose(finite, np.rint(finite)))
    if integers:
        if np.min(finite) < np.iinfo(np.int64).min or np.max(finite) > np.iinfo(np.int64).max:
            return None
        unique = np.unique(finite.astype(np.int64))
    else:
        unique = np.unique(finite)
    max_categories = 500 if is_run_number_column(name) else 40 if is_index_column(name) else 12
    if 1 < unique.size <= max_categories and (
        integers
        or is_pass_flag(name)
        or name.endswith("Det")
        or is_index_column(name)
        or is_run_number_column(name)
    ):
        labels = [category_label(name, item) for item in unique.tolist()]
        return {
            "name": name,
            "label": label_for(name),
            "group": category_group_label(name),
            "values": [int(item) if float(item).is_integer() else float(item) for item in unique.tolist()],
            "labels": labels,
        }
    return None


def categorical_filter_sort_key(filter_info: dict[str, Any]) -> tuple[int, int, str, str]:
    name = str(filter_info["name"])
    label = str(filter_info.get("label") or label_for(name)).lower()
    return (category_group_rank(name), category_kind_rank(name), label, name.lower())


def category_group_rank(name: str) -> int:
    lowered = name.lower()
    canonical = canonical_variable_name(name)
    if is_run_number_column(name) or canonical in {"sourcefileid", "sourceeventindex", "eventnum", "helicity", "charge"}:
        return 0
    if is_pass_flag(name) or lowered in {"rec_selected", "rec_not_selected"} or "selected" in canonical:
        return 1
    if canonical.startswith("electron") or canonical in {"edet", "esector", "eidx"}:
        return 2
    if canonical.startswith("proton") or canonical in {"pdet", "psector", "pidx"}:
        return 3
    if canonical.startswith("gamma1") or canonical.startswith("g1"):
        return 4
    if canonical.startswith("gamma2") or canonical.startswith("g2"):
        return 5
    if canonical.startswith("gamma"):
        return 6
    if canonical.startswith("pi0"):
        return 7
    if "sector" in canonical:
        return 8
    if canonical.endswith("det") or canonical.endswith("detector"):
        return 9
    if is_index_column(name):
        return 10
    return 11


def category_group_label(name: str) -> str:
    labels = {
        0: "Event",
        1: "Selections",
        2: "Electron",
        3: "Proton",
        4: "Gamma 1",
        5: "Gamma 2",
        6: "Gamma",
        7: "Pi0",
        8: "Sectors",
        9: "Detectors",
        10: "Indices",
    }
    return labels.get(category_group_rank(name), "Other")


def category_kind_rank(name: str) -> int:
    canonical = canonical_variable_name(name)
    if is_run_number_column(name):
        return 0
    if canonical.endswith("det") or canonical.endswith("detector"):
        return 1
    if "sector" in canonical:
        return 2
    if is_index_column(name):
        return 3
    if is_pass_flag(name):
        return 4
    return 5


def category_label(name: str, value: Any) -> str:
    if is_run_number_column(name):
        return str(int(value))
    if name in {"pDet", "rec_proton_detector", "protonDet", "protonDetector"}:
        return {1: "FD", 2: "CD", 0: "FT", -999: "missing"}.get(int(value), str(value))
    if "sector" in name.lower():
        sector = int(value)
        return "missing" if sector < 0 else f"sector {sector}"
    if is_pass_flag(name) or name in {"rec_selected", "rec_not_selected"}:
        flag = int(value)
        if flag == 1:
            return "pass"
        if flag == 0:
            return "fail"
        return "missing"
    return str(value)


def is_pass_flag(name: str) -> bool:
    core = name
    for prefix in ("rec_", "gen_", "rec.", "gen.", "event."):
        if core.lower().startswith(prefix):
            core = core[len(prefix):]
            break
    return core.startswith("pass") or bool(
        re.match(r"^(?:electron|proton|gamma1|gamma2|gamma|pi0)Pass", core)
    )


def is_index_column(name: str) -> bool:
    return name.endswith("Idx") or name.endswith("Index")


def is_run_number_column(name: str) -> bool:
    return name in {"run", "runNum", "rec_runNum", "gen_runNum"} or name.lower().endswith("runnum")


def is_integer_category(name: str) -> bool:
    return is_run_number_column(name) or is_index_column(name) or "sector" in name.lower() or name.endswith("Det")


def label_for(name: str) -> str:
    particle_label = particle_display_label(name)
    if particle_label is not None:
        return particle_label
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    return name.replace("_", " ")


def particle_display_label(name: str) -> str | None:
    lowered = name.lower()
    source = ""
    base = name
    if lowered.startswith("rec_"):
        source = "REC"
        base = name[4:]
    elif lowered.startswith("gen_"):
        source = "GEN"
        base = name[4:]
    elif lowered.startswith("rec."):
        source = "REC"
        base = name[4:]
    elif lowered.startswith("gen."):
        source = "GEN"
        base = name[4:]

    canonical = base.replace("_", "").replace(".", "").lower()
    if source and canonical in PARTICLE_QUANTITY_DISPLAY_NAMES:
        return f"{source} {PARTICLE_QUANTITY_DISPLAY_NAMES[canonical]}"
    for prefix, particle in PARTICLE_DISPLAY_PREFIXES:
        if not canonical.startswith(prefix):
            continue
        quantity = PARTICLE_QUANTITY_DISPLAY_NAMES.get(canonical[len(prefix):])
        if quantity is None:
            continue
        return " ".join(part for part in (source, particle, quantity) if part)
    return None


def first_present(
    variables: list[dict[str, Any]],
    names: tuple[str, ...],
    fallback_index: int = 0,
) -> str:
    available = {item["name"] for item in variables}
    for name in names:
        if name in available:
            return name
    if not variables:
        raise ValueError("No variables available")
    return variables[min(fallback_index, len(variables) - 1)]["name"]


def encode_float32(values: np.ndarray) -> dict[str, str]:
    compact = values.astype("<f4", copy=False)
    raw = compact.tobytes(order="C")
    return {"dtype": "float32", "data": base64.b64encode(raw).decode("ascii")}


def render_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_escape(payload["title"])}</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: Canvas;
  --fg: CanvasText;
  --muted: color-mix(in srgb, CanvasText 66%, Canvas);
  --panel: color-mix(in srgb, Canvas 92%, CanvasText);
  --border: color-mix(in srgb, CanvasText 20%, Canvas);
  --accent: Highlight;
  --accent-text: HighlightText;
  --filter-alert: color-mix(in srgb, red 78%, CanvasText);
  --mark: color-mix(in srgb, Highlight 78%, CanvasText);
  --ghost: color-mix(in srgb, orange 82%, CanvasText);
  --reference: color-mix(in srgb, LinkText 86%, CanvasText);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 13px/1.38 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.startup-loading {{
  position: fixed;
  inset: 0;
  z-index: 100;
  display: grid;
  place-items: center;
  padding: 24px;
  background: var(--bg);
  opacity: 1;
  transition: opacity 160ms ease;
}}
.startup-loading.complete {{
  opacity: 0;
  pointer-events: none;
}}
.startup-loading-content {{
  display: grid;
  justify-items: center;
  gap: 8px;
  color: var(--fg);
  text-align: center;
}}
.startup-loading-spinner {{
  width: 30px;
  height: 30px;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: startup-spin 720ms linear infinite;
}}
.startup-loading-content strong {{ font-size: 16px; }}
.startup-loading-content span {{ color: var(--muted); }}
.startup-loading-progress {{
  width: min(320px, 72vw);
  height: 8px;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--panel);
}}
.startup-loading-progress-bar {{
  width: 0;
  height: 100%;
  border-radius: inherit;
  background: var(--accent);
  transition: width 120ms ease-out;
}}
.startup-loading-percent {{
  min-width: 4ch;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
}}
@keyframes startup-spin {{ to {{ transform: rotate(360deg); }} }}
@media (prefers-reduced-motion: reduce) {{
  .startup-loading {{ transition: none; }}
  .startup-loading-spinner {{ animation: none; }}
}}
main {{
  display: grid;
  grid-template-columns: minmax(220px, 270px) minmax(0, 1fr);
  min-height: 100vh;
}}
aside {{
  border-right: 1px solid var(--border);
  padding: 10px;
  background: var(--panel);
  overflow: auto;
}}
section {{
  padding: 12px;
  min-width: 0;
}}
.control-deck {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px 16px;
  align-items: start;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}}
.control-panel {{
  min-width: 0;
}}
.control-panel h2:first-child {{
  margin-top: 0;
}}
.control-panel.wide {{
  grid-column: 1 / -1;
}}
.analysis-tools {{
  grid-template-columns: 1fr;
  gap: 12px;
}}
.analysis-tools .control-panel {{
  min-width: 0;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
}}
.analysis-tools .control-panel > h2 {{
  margin-top: 0;
}}
.analysis-tools .text-panel {{
  grid-column: 1;
}}
.analysis-tools .fit-panel {{ order: 1; }}
.analysis-tools .text-panel {{ order: 2; }}
h1 {{
  font-size: 16px;
  font-weight: 600;
  margin: 1px 0 3px;
  overflow-wrap: anywhere;
}}
.dataset-heading {{
  position: relative;
  margin-bottom: 12px;
  padding: 9px 10px 9px 13px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  box-shadow: 0 3px 12px color-mix(in srgb, CanvasText 8%, transparent);
}}
.dataset-heading::before {{
  content: "";
  position: absolute;
  inset: 7px auto 7px 0;
  width: 3px;
  border-radius: 0 3px 3px 0;
  background: var(--mark);
}}
.dataset-kicker {{
  color: var(--muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}
.dataset-heading #source {{
  margin-top: 3px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.dataset-heading #source[hidden] {{ display: none; }}
h2 {{
  font-size: 12px;
  font-weight: 600;
  margin: 14px 0 6px;
}}
.subtle {{ color: var(--muted); font-size: 12px; }}
.row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.slice-controls {{
  display: grid;
  grid-template-columns: minmax(74px, 0.45fr) minmax(0, 1.55fr);
  gap: 0 8px;
  margin-top: -3px;
}}
.slice-controls[hidden] {{ display: none; }}
.slice-status {{ grid-column: 1 / -1; min-height: 0; }}
label {{ display: grid; gap: 4px; margin: 8px 0; }}
select, input, button {{
  font: inherit;
  color: inherit;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 4px 6px;
  min-width: 0;
}}
button {{ cursor: pointer; width: auto; white-space: nowrap; }}
button.active {{
  background: var(--accent);
  color: var(--accent-text);
  border-color: var(--accent);
}}
.segmented {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }}
.segmented button {{ flex: 0 0 auto; min-width: 54px; padding-inline: 8px; }}
.chips {{ display: flex; gap: 5px; flex-wrap: wrap; }}
.chip {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 3px 7px;
  background: var(--bg);
}}
.chip input {{ margin: 0; }}
.fit-tools {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
  margin: 8px 0 6px;
}}
.fit-model-grid {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px 10px;
  align-items: end;
}}
.fit-model-grid label {{ margin: 0; width: auto; }}
.fit-model-grid select {{ width: auto; max-width: 190px; }}
.fit-tools button {{
  flex: 0 0 auto;
}}
.fit-panel-layout {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
  align-items: start;
}}
.fit-panel-layout.engaged {{
  grid-template-columns: minmax(470px, 0.98fr) minmax(0, 1.02fr);
}}
.fit-controls {{
  min-width: 0;
}}
.fit-range-summary {{
  margin-top: 8px;
  padding: 7px 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
}}
.fit-summary {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 6px;
  min-width: 0;
}}
.fit-summary.sector {{
  grid-template-columns: repeat(3, minmax(0, 1fr));
}}
.fit-summary.sector .fit-summary-item:last-child:nth-child(7) {{
  grid-column: 2;
}}
.fit-summary-item {{
  min-width: 0;
  padding: 6px 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  overflow-wrap: anywhere;
}}
.fit-summary:not(.multi) .fit-summary-item {{
  grid-column: 1 / -1;
}}
.fit-summary-label {{
  display: block;
  margin-bottom: 2px;
  color: var(--fg);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.fit-summary-detail {{
  display: block;
  color: var(--muted);
  font-size: 11.5px;
  line-height: 1.35;
}}
canvas.fit-range-picker {{
  cursor: crosshair;
}}
.quick-category {{
  display: grid;
  gap: 6px;
  margin: 8px 0 10px;
}}
.quick-category-head {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 6px;
  align-items: end;
}}
.quick-category-head label {{ margin: 0; }}
.collapse-button {{
  width: 30px;
  min-width: 30px;
  padding: 5px 0;
}}
.quick-category .chips {{
  max-height: 118px;
  overflow: auto;
  padding-right: 2px;
}}
.quick-category.collapsed #quickCategoryBody {{ display: none; }}
.constraints-panel {{
  margin: 10px 0 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}}
.constraints-panel h2 {{
  margin-top: 0;
}}
.sidebar-derived {{
  margin-top: 8px;
}}
.sidebar-derived h2 {{
  margin: 0 0 6px;
}}
.filter-details {{
  border-top: 1px solid var(--border);
  padding: 6px 0;
}}
.category-group-title {{
  color: var(--fg);
  font-size: 12px;
  font-weight: 600;
  margin-top: 10px;
}}
.axis-control {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 6px;
  align-items: end;
  margin: 8px 0;
}}
.axis-control label,
.extra-variable label {{ margin: 0; }}
.axis-button {{
  width: 30px;
  min-width: 30px;
  padding: 5px 0;
  text-align: center;
}}
.plot-panel-controls {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}}
.panel-view-options {{
  margin: 0;
  gap: 0;
  padding: 2px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--panel);
}}
.panel-view-options button {{
  border-color: transparent;
  background: transparent;
}}
.panel-view-options button + button {{
  border-left-color: var(--border);
  border-radius: 0 4px 4px 0;
}}
.panel-view-options button:first-child {{
  border-radius: 4px 0 0 4px;
}}
.panel-view-options button.active {{
  border-color: var(--accent);
  background: var(--accent);
}}
.panel-tabs {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
  margin: 0;
}}
.panel-tabs button {{
  flex: 0 1 auto;
}}
.panel-tab.active {{
  background: var(--accent);
  color: var(--accent-text);
  border-color: var(--accent);
}}
.extra-variable {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 6px;
  align-items: end;
  margin: 6px 0 8px;
}}
.filter-details summary {{
  cursor: pointer;
  color: var(--muted);
  font-size: 12px;
}}
.filter-details[open] summary {{
  margin-bottom: 6px;
}}
#categoryFilters {{
  column-width: 260px;
  column-gap: 18px;
}}
#categoryFilters .category-group-title,
#categoryFilters .filter-details {{
  break-inside: avoid;
}}
.filter-row {{
  display: grid;
  grid-template-columns: minmax(74px, 1.3fr) minmax(48px, 0.7fr) minmax(48px, 0.7fr) auto;
  gap: 4px;
  margin: 6px 0;
  align-items: center;
}}
.filter-row > :first-child {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; }}
.filter-row input {{ width: 100%; }}
.constraint-status {{
  margin: -2px 0 4px;
}}
.constraint-status:empty {{ display: none; }}
.operation-grid {{
  display: flex;
  flex-direction: column;
  gap: 6px;
}}
.operation-builder {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
  align-items: end;
}}
.operation-builder label {{
  margin: 0;
  min-width: 0;
}}
.operation-builder button {{
  white-space: nowrap;
  width: 100%;
}}
.operation-status {{
  margin-top: 2px;
  font-size: 11px;
}}
.operation-status:empty {{ display: none; }}
.toolbar-tile {{
  display: inline-flex;
  align-items: center;
  gap: 2px;
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 3px;
  background: var(--panel);
}}
.action-tile button {{
  border: 0;
  border-radius: 4px;
  background: transparent;
}}
.action-tile button + button {{ border-left: 1px solid var(--border); border-radius: 0 4px 4px 0; }}
.header-utility-stack {{
  display: flex;
  align-items: stretch;
  gap: 6px;
  margin-left: auto;
}}
.header-utility-stack .toolbar-tile {{
  justify-content: center;
}}
.header-utility-stack #loadFiles {{ white-space: nowrap; }}
.count-tile {{ gap: 0; }}
.count-stat {{
  display: inline-flex;
  gap: 4px;
  align-items: baseline;
  padding: 2px 7px;
}}
.count-stat + .count-stat {{ border-left: 1px solid var(--border); }}
.count-stat .subtle {{ font-size: 10px; }}
.count-stat strong {{ font-size: 12px; font-weight: 650; }}
.canvas-toolbar {{
  display: grid;
  grid-template-columns: minmax(470px, 1.7fr) repeat(2, minmax(240px, 1fr));
  align-items: stretch;
  gap: 10px;
  width: 100%;
  min-height: 34px;
  margin: 0 auto;
}}
.canvas-toolbar[hidden] {{ display: none; }}
.canvas-toolbar .toolbar-tile {{
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  min-height: 34px;
  justify-content: center;
}}
.canvas-toolbar .axis-tile {{
  display: grid;
  grid-template-columns: repeat(5, minmax(70px, 1fr));
  gap: 4px 6px;
  padding: 6px;
}}
.axis-tile label {{
  gap: 2px;
  margin: 0;
  color: var(--muted);
  font-size: 10px;
}}
.axis-tile input {{
  box-sizing: border-box;
  width: 100%;
  min-height: 26px;
}}
.axis-tile input[type="range"] {{
  padding-inline: 0;
}}
.axis-range-pair {{ display: contents; }}
.axis-range-pair.axis-range-hidden {{ display: none; }}
.axis-y-ticks {{ grid-column: 4; grid-row: 2; }}
.axis-y-label {{ grid-column: 5; grid-row: 2; }}
.canvas-toolbar-slot {{
  width: 100%;
  min-width: 0;
  margin-bottom: 6px;
}}
.canvas-toolbar-slot:empty {{ display: none; }}
.canvas-toolbar-slot.controls-collapsed {{ margin-bottom: 0; }}
.plot-grid:not(.compare) .canvas-toolbar-slot.controls-collapsed {{ display: none; }}
.display-tile .chip {{
  border: 0;
  border-radius: 4px;
  padding: 2px 6px;
  background: transparent;
}}
.canvas-toolbar .display-tile {{
  display: grid;
  grid-template-columns: repeat(3, max-content);
  align-content: center;
  gap: 3px 6px;
}}
.canvas-toolbar .action-tile {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  align-content: center;
  gap: 2px;
}}
.canvas-toolbar .action-tile button {{ min-width: 0; white-space: nowrap; }}
.display-tile .aspect-control {{
  grid-column: 1 / -1;
  gap: 1px;
  margin: 0;
  color: var(--muted);
  font-size: 10px;
}}
.display-tile .aspect-control input {{ width: 100%; padding-inline: 0; }}
canvas {{
  display: block;
  width: 100%;
  margin-inline: auto;
  height: min(70vh, 700px);
  min-height: 420px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
}}
.hover-overlay {{
  position: absolute;
  z-index: 2;
  display: none;
  pointer-events: none;
  min-height: 0;
  border: 0;
  background: transparent;
}}
.plot-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
}}
.plot-grid.compare {{
  grid-template-columns: repeat(2, minmax(0, 1fr));
}}
.plot-grid.compare .canvas-toolbar-slot {{ display: block; }}
.plot-grid.compare .canvas-toolbar {{
  grid-template-columns: repeat(2, minmax(0, 1fr));
  width: 100%;
}}
.plot-grid.compare .canvas-toolbar .axis-tile {{ grid-column: 1 / -1; }}
.plot-pane {{
  position: relative;
}}
.plot-pane.hidden {{ display: none; }}
.canvas-context-menu {{
  position: fixed;
  z-index: 40;
  display: grid;
  min-width: 148px;
  gap: 2px;
  padding: 4px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--panel);
  box-shadow: 0 8px 24px color-mix(in srgb, CanvasText 22%, transparent);
}}
.canvas-context-menu[hidden] {{ display: none; }}
.canvas-context-menu button {{
  width: 100%;
  border: 0;
  text-align: left;
  background: transparent;
}}
.canvas-context-menu button:hover:not(:disabled) {{
  background: color-mix(in srgb, var(--accent) 18%, transparent);
}}
.canvas-context-menu button:disabled {{
  color: var(--muted);
  cursor: default;
}}
.reference-editor {{
  position: fixed;
  inset: 0;
  z-index: 50;
  display: grid;
  place-items: center;
  padding: 16px;
  background: color-mix(in srgb, CanvasText 26%, transparent);
}}
.reference-editor[hidden] {{ display: none; }}
.reference-editor-panel {{
  width: min(460px, 100%);
  max-height: min(680px, calc(100vh - 32px));
  overflow: auto;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 9px;
  background: var(--panel);
  box-shadow: 0 12px 36px color-mix(in srgb, CanvasText 24%, transparent);
}}
.reference-editor-head,
.reference-editor-actions,
.reference-curve-item {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}}
.reference-editor-head h2 {{ margin: 0; font-size: 14px; }}
.reference-editor-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0 8px;
}}
.reference-editor-grid .reference-expression {{ grid-column: 1 / -1; }}
.reference-editor-actions {{ justify-content: flex-end; margin-top: 8px; }}
.reference-curve-list {{ display: grid; gap: 5px; margin-top: 12px; }}
.reference-curve-item {{
  padding: 6px 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
}}
.reference-curve-item span {{ min-width: 0; overflow-wrap: anywhere; }}
.filter-badge {{
  display: none;
  align-items: center;
  gap: 8px;
  margin: 6px 0 7px;
  padding: 5px 8px;
  border: 1px solid var(--border);
  border-left: 4px solid var(--filter-alert);
  border-radius: 6px;
  background: var(--bg);
  color: var(--fg);
  font-size: 12px;
}}
.filter-badge strong {{
  font-weight: 600;
  white-space: nowrap;
}}
.filter-badge span {{
  color: var(--muted);
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.color-scale-hover {{
  position: absolute;
  z-index: 3;
  display: none;
  align-items: center;
  gap: 5px;
  pointer-events: none;
  transform: translateY(-50%);
  color: var(--fg);
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 2px 5px;
  font-size: 11px;
  line-height: 1.1;
  box-shadow: 0 4px 12px rgba(0,0,0,0.12);
  white-space: nowrap;
}}
.scale-slider {{
  width: var(--marker-width, 18px);
  height: 6px;
  border: 1px solid var(--fg);
  background: var(--marker-color, currentColor);
}}
.scale-name {{
  color: var(--muted);
}}
.plot-toolbar {{
  display: flex;
  justify-content: flex-start;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: 10px;
}}
.plot-toolbar .chips {{
  align-items: center;
}}
.plot-actions {{
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}}
.plot-actions button {{
  flex: 0 0 auto;
}}
.load-browser {{
  position: fixed;
  inset: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 18px;
  background: rgba(0,0,0,0.24);
}}
.load-browser.hidden {{
  display: none;
}}
.load-browser-panel {{
  width: min(760px, 96vw);
  max-height: min(720px, 90vh);
  display: grid;
  grid-template-rows: auto auto auto minmax(180px, 1fr);
  gap: 10px;
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  color: var(--fg);
  box-shadow: 0 18px 52px rgba(0,0,0,0.22);
}}
.load-browser-head, .load-browser-actions {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}}
.load-browser-head strong {{
  display: block;
  margin-bottom: 2px;
}}
.remote-file-list {{
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 6px;
  background: var(--panel);
}}
.remote-entry {{
  width: 100%;
  min-height: 30px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 7px;
  border-radius: 6px;
  color: var(--fg);
}}
.remote-entry:hover {{
  background: var(--chip);
}}
button.remote-entry {{
  text-align: left;
  background: transparent;
  border: 0;
  cursor: pointer;
}}
.remote-entry input {{
  width: auto;
}}
.remote-entry span {{
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.quantity-banner {{
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 9px;
  flex-wrap: wrap;
  text-align: center;
  margin: 0 0 7px;
  padding: 6px 9px;
  border: 1px solid var(--border);
  border-left: 4px solid var(--mark);
  border-radius: 7px;
  background: var(--panel);
  color: var(--fg);
}}
.quantity-banner .quantity-mode {{
  font-size: 11px;
  font-weight: 700;
  color: var(--muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.quantity-banner strong {{
  min-width: 0;
  font-size: 15px;
  font-weight: 700;
  overflow-wrap: anywhere;
}}
.quantity-banner .quantity-detail {{
  min-width: 0;
  color: var(--muted);
  font-size: 12px;
  overflow-wrap: anywhere;
}}
.plot-grid.compare canvas {{
  height: min(64vh, 660px);
  min-height: 390px;
}}
.hover-info {{
  display: none;
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 12px;
}}
.table-wrap {{
  margin-top: 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: auto;
  max-height: 220px;
}}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border-bottom: 1px solid var(--border); padding: 5px 7px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
@media (max-width: 1250px) {{
  .fit-panel-layout.engaged {{ grid-template-columns: 1fr; }}
  .canvas-toolbar {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .canvas-toolbar .axis-tile {{ grid-column: 1 / -1; }}
}}
@media (max-width: 820px) {{
  main {{ grid-template-columns: 1fr; }}
  aside {{ border-right: 0; border-bottom: 1px solid var(--border); }}
  .control-deck {{ grid-template-columns: 1fr; }}
  .header-utility-stack {{ margin-left: 0; }}
  #categoryFilters {{ column-width: auto; }}
  .plot-grid.compare {{ grid-template-columns: 1fr; }}
  canvas {{ min-height: 340px; height: 58vh; }}
}}
@media (max-width: 620px) {{
  .canvas-toolbar {{ grid-template-columns: 1fr; }}
  .canvas-toolbar .axis-tile {{ grid-column: auto; }}
  .fit-summary.sector {{ grid-template-columns: 1fr; }}
  .fit-summary.sector .fit-summary-item:last-child:nth-child(7) {{ grid-column: 1; }}
}}
@media (max-width: 420px) {{
  .canvas-toolbar .axis-tile {{ grid-template-columns: repeat(2, minmax(70px, 1fr)); }}
  .axis-y-ticks,
  .axis-y-label {{ grid-column: auto; grid-row: auto; }}
  .reference-editor-grid {{ grid-template-columns: 1fr; }}
  .reference-editor-grid .reference-expression {{ grid-column: 1; }}
}}
</style>
</head>
<body aria-busy="true">
<div class="startup-loading" id="startupLoading" role="status" aria-live="polite" aria-label="Loading visualizer data">
  <div class="startup-loading-content">
    <div class="startup-loading-spinner" aria-hidden="true"></div>
    <strong>Loading visualizer</strong>
    <div class="startup-loading-progress" role="progressbar" aria-label="Visualizer startup progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
      <div class="startup-loading-progress-bar" id="startupProgressBar"></div>
    </div>
    <span class="startup-loading-percent" id="startupProgressPercent">0%</span>
    <span id="startupProgressStage">Reading embedded data…</span>
  </div>
</div>
<main id="app">
  <aside>
    <div class="dataset-heading">
      <div class="dataset-kicker" id="datasetKicker">Dataset</div>
      <h1></h1>
      <div class="subtle" id="source" hidden></div>
    </div>
    <h2>Plot</h2>
    <div class="segmented">
      <button type="button" id="mode1d">1D</button>
      <button type="button" id="mode2d">2D</button>
    </div>
    <div class="axis-control" id="yAxisControl">
      <label>Y <select id="yvar"></select></label>
      <button type="button" class="axis-button" id="addYVar" aria-label="Add Y quantity">+</button>
    </div>
    <div class="extra-variable" id="extraYControls">
      <label>Additional Y <select id="y2var"></select></label>
      <button type="button" class="axis-button" id="removeYVar" aria-label="Remove additional Y quantity">-</button>
    </div>
    <div class="axis-control">
      <label>X <select id="xvar"></select></label>
      <button type="button" class="axis-button" id="addXVar" aria-label="Add X quantity">+</button>
    </div>
    <div class="extra-variable" id="extraXControls">
      <label>Additional X <select id="x2var"></select></label>
      <button type="button" class="axis-button" id="removeXVar" aria-label="Remove additional X quantity">-</button>
    </div>
    <label id="splitLabel">Split by <select id="splitVar"></select></label>
    <div class="slice-controls" id="sliceControls" hidden>
      <label>Slices <input id="sliceBins" type="number" min="1" max="24" step="1" value="6"></label>
      <label>Manual edges (optional)<input id="sliceEdges" type="text" placeholder="e.g. 0, 0.2, 0.5, 1"></label>
      <div class="subtle slice-status" id="sliceStatus" role="status" aria-live="polite"></div>
    </div>
    <div class="quick-category collapsed" id="quickCategoryBlock">
      <div class="quick-category-head">
        <label>Filter topology <select id="quickCategoryFilter"></select></label>
        <button type="button" class="collapse-button" id="toggleTopology" aria-expanded="false" aria-controls="quickCategoryBody" aria-label="Expand filter topology">&gt;</button>
      </div>
      <div id="quickCategoryBody">
        <div class="chips" id="quickCategoryChips"></div>
        <div class="segmented">
          <button type="button" id="quickCategoryAll">All</button>
          <button type="button" id="quickCategoryNone">None</button>
        </div>
        <div class="subtle" id="quickCategorySummary"></div>
      </div>
    </div>
    <div class="constraints-panel">
      <h2>Constraints</h2>
      <div class="filter-row constraint-builder">
        <select id="rangeVar"></select>
        <input id="rangeMin" type="number" step="any" placeholder="min">
        <input id="rangeMax" type="number" step="any" placeholder="max">
        <button type="button" id="addRange">Add</button>
      </div>
      <div class="subtle constraint-status" id="constraintStatus" role="status" aria-live="polite"></div>
      <div id="rangeFilters"></div>
      <div class="sidebar-derived">
        <h2>Derived Operations</h2>
        <div class="operation-grid">
          <div class="operation-builder">
            <label>Left <select id="opLeft"></select></label>
            <label>Right <select id="opRight"></select></label>
            <label>Operator <select id="opKind">
              <option value="subtract">left - right</option>
              <option value="add">left + right</option>
              <option value="ratio">left / right</option>
              <option value="fractional">(left - right) / right</option>
            </select></label>
            <button type="button" id="addDerived">Add derived</button>
          </div>
          <div class="subtle operation-status" id="opStatus"></div>
        </div>
      </div>
    </div>
  </aside>
  <section>
    <div class="plot-toolbar">
      <div class="plot-panel-controls">
        <div class="panel-tabs" id="panelTabs"></div>
        <button type="button" id="addPanel">+ panel</button>
        <div class="segmented panel-view-options" aria-label="Panel view options">
          <button type="button" id="splitView" aria-pressed="false">split view</button>
          <button type="button" id="sharedPanelFilters" class="active" aria-pressed="true" title="Apply the same topology, constraints, and text filters to both panels">shared panel filters</button>
        </div>
        <button type="button" id="toggleCanvasToolbar" aria-expanded="true" aria-controls="canvasToolbar">Hide plot controls</button>
      </div>
      <div class="header-utility-stack">
        <span class="subtle" id="samplingNote"></span>
        <span class="subtle" id="datasetStatus" role="status" aria-live="polite"></span>
        <button type="button" id="loadFiles">Load File(s)</button>
        <input id="loadFileInput" type="file" accept=".html,text/html" multiple hidden>
        <div class="toolbar-tile count-tile" aria-label="Dataset counts">
          <div class="count-stat"><span class="subtle">samples</span><strong id="sampleCount">1</strong></div>
          <div class="count-stat"><span class="subtle">selected</span><strong id="selectedCount">0</strong></div>
          <div class="count-stat"><span class="subtle">embedded</span><strong id="embeddedCount">0</strong></div>
        </div>
      </div>
      <span id="meanX" hidden>-</span>
      <span id="meanY" hidden>-</span>
    </div>
    <div class="load-browser hidden" id="loadBrowser" role="dialog" aria-modal="true" aria-labelledby="loadBrowserTitle">
      <div class="load-browser-panel">
        <div class="load-browser-head">
          <div>
            <strong id="loadBrowserTitle">Load Farm-Side Visualizer HTML</strong>
            <div class="subtle" id="loadBrowserPath"></div>
          </div>
          <button type="button" id="closeLoadBrowser">Close</button>
        </div>
        <div class="load-browser-actions">
          <div class="plot-actions">
            <button type="button" id="loadBrowserRoot">Root</button>
            <button type="button" id="loadBrowserUp">Up</button>
            <button type="button" id="loadBrowserRefresh">Refresh</button>
          </div>
          <button type="button" id="loadSelectedRemote">Load selected</button>
        </div>
        <div class="subtle" id="loadBrowserMessage"></div>
        <div class="remote-file-list" id="remoteFileList"></div>
      </div>
    </div>
    <div class="canvas-toolbar" id="canvasToolbar" aria-label="Plot controls">
      <div class="toolbar-tile axis-tile" aria-label="Axis labels, binning, ranges, and ticks">
        <label><span>X bins</span><input id="xbins" type="number" min="5" max="400" value="80"></label>
        <label><span>X min</span><input id="xmin" type="number" step="any"></label>
        <label><span>X max</span><input id="xmax" type="number" step="any"></label>
        <label><span>X ticks <span id="xtickValue"></span></span><input id="xticks" type="range" min="1" max="40" step="0.5" value="6"></label>
        <label class="axis-label-control"><span>X label</span><input id="xAxisLabel" type="text" placeholder="auto"></label>
        <label><span>Y bins</span><input id="ybins" type="number" min="5" max="300" value="80"></label>
        <span class="axis-range-pair" id="yrange">
          <label><span>Y min</span><input id="ymin" type="number" step="any"></label>
          <label><span>Y max</span><input id="ymax" type="number" step="any"></label>
        </span>
        <label class="axis-y-ticks"><span>Y ticks <span id="ytickValue"></span></span><input id="yticks" type="range" min="1" max="40" step="0.5" value="6"></label>
        <label class="axis-label-control axis-y-label"><span>Y label</span><input id="yAxisLabel" type="text" placeholder="auto"></label>
      </div>
      <div class="toolbar-tile display-tile" aria-label="Display options">
        <label class="chip" id="logzChip"><input id="logz" type="checkbox"> log color</label>
        <label class="chip"><input id="density" type="checkbox"> density</label>
        <label class="chip" id="colorScaleChip"><input id="colorScale" type="checkbox"> color scale</label>
        <label class="aspect-control"><span>plot height <span id="plotHeightValue">50%</span></span><input id="plotHeight" type="range" min="0.25" max="1" step="0.01" value="0.5"></label>
        <label class="aspect-control"><span>plot width <span id="plotWidthValue">100%</span></span><input id="plotWidth" type="range" min="0.5" max="1" step="0.01" value="1"></label>
      </div>
      <div class="toolbar-tile action-tile" aria-label="Plot actions">
        <button type="button" id="resetFilters">Reset filters</button>
        <button type="button" id="resetRanges">Reset axes</button>
        <button type="button" id="plotTools" aria-haspopup="menu" aria-expanded="false">Plot tools…</button>
        <button type="button" id="savePng">Save PNG</button>
        <button type="button" id="saveWorkspace">Save workspace</button>
        <button type="button" id="restoreWorkspace">Restore saved</button>
      </div>
    </div>
    <div class="plot-grid" id="plotGrid">
      <div class="plot-pane" id="plotPaneA">
        <div class="quantity-banner" id="quantityBannerA"><span class="quantity-mode"></span><strong></strong><span class="quantity-detail"></span></div>
        <div class="filter-badge" id="filterBadgeA"><strong></strong><span></span></div>
        <div class="canvas-toolbar-slot" id="canvasToolbarSlotA"></div>
        <canvas id="plotA" width="1200" height="780"></canvas>
        <canvas class="hover-overlay" id="hoverOverlayA" aria-hidden="true"></canvas>
        <div class="color-scale-hover" id="colorScaleHoverAPrimary"><span class="scale-slider"></span><span class="scale-name"></span><span class="scale-value"></span></div>
        <div class="color-scale-hover" id="colorScaleHoverAOverlay"><span class="scale-slider"></span><span class="scale-name"></span><span class="scale-value"></span></div>
        <div class="hover-info" id="hoverInfoA"></div>
      </div>
      <div class="plot-pane hidden" id="plotPaneB">
        <div class="quantity-banner" id="quantityBannerB"><span class="quantity-mode"></span><strong></strong><span class="quantity-detail"></span></div>
        <div class="filter-badge" id="filterBadgeB"><strong></strong><span></span></div>
        <div class="canvas-toolbar-slot" id="canvasToolbarSlotB"></div>
        <canvas id="plotB" width="1200" height="780"></canvas>
        <canvas class="hover-overlay" id="hoverOverlayB" aria-hidden="true"></canvas>
        <div class="color-scale-hover" id="colorScaleHoverBPrimary"><span class="scale-slider"></span><span class="scale-name"></span><span class="scale-value"></span></div>
        <div class="color-scale-hover" id="colorScaleHoverBOverlay"><span class="scale-slider"></span><span class="scale-name"></span><span class="scale-value"></span></div>
        <div class="hover-info" id="hoverInfoB"></div>
      </div>
    </div>
    <div class="control-deck analysis-tools">
      <div class="control-panel fit-panel">
        <h2>Fit</h2>
        <div class="fit-panel-layout">
          <div class="fit-controls">
            <div class="fit-model-grid">
              <label>Signal S <select id="signalModel">
                <option value="none">none</option>
                <option value="gaussian">Gaussian</option>
                <option value="crystalball">Crystal Ball</option>
              </select></label>
              <label>Background B <select id="backgroundModel">
                <option value="none">none</option>
                <option value="poly0">constant</option>
                <option value="poly1">Polynomial degree 1</option>
                <option value="poly2">Polynomial degree 2</option>
                <option value="poly3">Polynomial degree 3</option>
                <option value="poly4">Polynomial degree 4</option>
                <option value="poly5">Polynomial degree 5</option>
              </select></label>
              <label>Method <select id="fitMethod">
                <option value="unweighted">Ordinary LS</option>
                <option value="poisson">Poisson WLS (Pearson)</option>
                <option value="unbinned">Unbinned likelihood</option>
              </select></label>
            </div>
            <div class="fit-tools">
              <label class="chip"><input id="fitRangeClick" type="checkbox"> click endpoints</label>
              <button type="button" id="clearFitRange">Clear range</button>
              <button type="button" id="toggleFitAnnotations" aria-pressed="true">Hide canvas fit results</button>
            </div>
            <label id="fitScanDetailControl">
              <span>Unbinned scan detail: <strong id="fitScanDetailValue">balanced</strong></span>
              <input id="fitScanDetail" type="range" min="1" max="5" step="1" value="3">
            </label>
            <div class="subtle" id="fitMethodNote"></div>
          <div class="subtle fit-range-summary" id="fitRangeSummary" hidden>Fit range: full X range</div>
          </div>
          <div class="fit-summary" id="fitSummary" aria-live="polite" hidden></div>
        </div>
      </div>
      <div class="control-panel text-panel" id="textFilterPanel">
        <h2>Text Filters</h2>
        <div id="textFilters"></div>
      </div>
    </div>
    <div class="table-wrap"><table id="preview"></table></div>
    <div class="control-deck">
      <div class="control-panel wide">
        <h2>All Category Filters</h2>
        <div id="categoryFilters"></div>
      </div>
    </div>
  </section>
</main>
<div class="canvas-context-menu" id="canvasContextMenu" role="menu" hidden>
  <button type="button" id="makeGhost" role="menuitem">Make ghost</button>
  <button type="button" id="clearGhost" role="menuitem">Clear ghost</button>
  <button type="button" id="toggleCanvasToolbarContext" role="menuitem">Hide plot controls</button>
  <button type="button" id="toggleMeanGuides" role="menuitemcheckbox" aria-checked="false">Show mean guides</button>
  <button type="button" id="profileX" role="menuitem">Profile X</button>
  <button type="button" id="profileY" role="menuitem">Profile Y</button>
  <button type="button" id="addFunctionCurve" role="menuitem">Add function curve…</button>
  <button type="button" id="manageReferenceCurves" role="menuitem">Manage reference curves…</button>
</div>
<div class="reference-editor" id="referenceCurveEditor" role="dialog" aria-modal="true" aria-labelledby="referenceCurveTitle" hidden>
  <div class="reference-editor-panel">
    <div class="reference-editor-head">
      <h2 id="referenceCurveTitle">Reference curves</h2>
      <button type="button" id="closeReferenceCurveEditor">Close</button>
    </div>
    <div class="subtle" id="referenceCurveAxes"></div>
    <div class="reference-editor-grid">
      <label>Graph <select id="referenceCurveDirection">
        <option value="y-of-x">y = f(x)</option>
        <option value="x-of-y">x = f(y)</option>
      </select></label>
      <label>Line style <select id="referenceLineStyle">
        <option value="solid">Solid</option>
        <option value="dashed">Dashed</option>
        <option value="dotted">Dotted</option>
        <option value="dash-dot">Dash-dot</option>
      </select></label>
      <label class="reference-expression"><span id="referenceExpressionLabel">f(x)</span><input id="referenceCurveExpression" type="text" value="x" spellcheck="false"></label>
      <label>Domain minimum (optional)<input id="referenceDomainMin" type="number" step="any"></label>
      <label>Domain maximum (optional)<input id="referenceDomainMax" type="number" step="any"></label>
      <label>Line width <input id="referenceLineWidth" type="number" min="0.5" max="3" step="0.25" value="1.25"></label>
      <label>Label (optional)<input id="referenceCurveLabel" type="text" placeholder="auto"></label>
    </div>
    <div class="subtle">Use numbers, x or y, pi, e, + − * / ^, parentheses, and functions such as sqrt, sin, cos, abs, exp, log, min, max, and pow.</div>
    <div class="subtle" id="referenceCurveStatus" role="status" aria-live="polite"></div>
    <div class="reference-editor-actions">
      <button type="button" id="clearReferenceCurves">Clear all</button>
      <button type="button" id="saveReferenceCurve">Add curve</button>
    </div>
    <div class="reference-curve-list" id="referenceCurveList"></div>
  </div>
</div>
<script>
const payload = {payload_json};
const columns = {{}};
const textColumns = {{}};
let rowCount = payload.rowCount;
const variables = payload.variables;
const byName = Object.fromEntries(variables.map(v => [v.name, v]));
const integerVariables = new Set(variables.filter(v => v.integer).map(v => v.name));
const SAMPLE_COLUMN = "__sampleId";
const WORKSPACE_STORAGE_VERSION = 3;
const WORKSPACE_STORAGE_KEY = `sf-visualizer:${{payload.source || payload.title}}`;
const loadedSamples = [{{id: 0, label: sampleLabel(payload.source || "sample 1"), rows: rowCount}}];
let remoteDirectoryUrl = null;
const remoteSelections = new Map();
const panelKeys = ["A", "B"];
const panelLabels = {{A: "Panel 1", B: "Panel 2"}};
let enabledPanels = ["A"];
let activePanel = "A";
let compareMode = false;
let contextMenuPanelKey = "A";
let contextMenuProfileBin = null;
let referenceCurveId = 0;
let topologyCollapsed = true;
let canvasToolbarCollapsed = false;
let canvasToolbarExpandedHeight = 0;
let workspaceSaveTimer = null;
let updateFrame = null;
const sharedFilterState = makeFilterState();
let sharedPanelFilters = true;
let activeRanges = sharedFilterState.ranges;
let categoryState = sharedFilterState.categories;
const panels = {{
  A: makePanel("A", payload.defaultX, payload.defaultY),
  B: makePanel("B", comparisonDefaultX(), payload.defaultY)
}};

const el = id => document.getElementById(id);
const fmt = value => Number.isFinite(value) ? (Math.abs(value) >= 1000 || Math.abs(value) < 0.01 ? value.toExponential(3) : value.toPrecision(4)) : "-";
const fmtColumn = (name, value) => integerVariables.has(name) && Number.isFinite(value) ? String(Math.round(value)) : fmt(value);
const fmtTickTarget = value => Number.isInteger(value) ? String(value) : value.toFixed(1);
const MAX_PLOT_HEIGHT_TO_WIDTH = 2 / 3;
const canonicalPlotHeight = value => clamp(Number(value) || 0.5, 0.25, 1);
const plotHeightLabel = value => `${{Math.round(canonicalPlotHeight(value) * 100)}}%`;
const canonicalPlotWidth = value => clamp(Number(value) || 1, 0.5, 1);
const plotWidthLabel = value => `${{Math.round(canonicalPlotWidth(value) * 100)}}%`;

function setStartupProgress(percent, stage) {{
  const normalized = Math.max(0, Math.min(100, Math.round(percent)));
  const overlay = document.getElementById("startupLoading");
  const bar = document.getElementById("startupProgressBar");
  const percentLabel = document.getElementById("startupProgressPercent");
  const stageLabel = document.getElementById("startupProgressStage");
  if (bar) bar.style.width = `${{normalized}}%`;
  if (percentLabel) percentLabel.textContent = `${{normalized}}%`;
  if (stageLabel && stage) stageLabel.textContent = stage;
  const progress = overlay?.querySelector('[role="progressbar"]');
  if (progress) progress.setAttribute("aria-valuenow", String(normalized));
}}

function yieldStartupFrame() {{
  return new Promise(resolve => requestAnimationFrame(() => resolve()));
}}

async function decodeInitialPayloadColumns() {{
  const entries = Object.entries(payload.columns || {{}});
  setStartupProgress(2, `Preparing ${{entries.length.toLocaleString()}} columns…`);
  await yieldStartupFrame();
  for (let index = 0; index < entries.length; index++) {{
    const [name, value] = entries[index];
    if (value && value.dtype === "float32") {{
      const binary = atob(value.data);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      columns[name] = new Float32Array(bytes.buffer);
    }} else {{
      textColumns[name] = value;
    }}
    setStartupProgress(
      5 + 55 * (index + 1) / Math.max(entries.length, 1),
      `Decoding column ${{index + 1}} of ${{entries.length}}: ${{name}}`
    );
    await yieldStartupFrame();
  }}
}}

function sampleLabel(source) {{
  const clean = String(source || "").split(/[\\\\/]/).pop() || "sample";
  return clean.replace(/\\.html?$/i, "") || clean;
}}

function ensureSampleColumn() {{
  if (!columns[SAMPLE_COLUMN]) {{
    columns[SAMPLE_COLUMN] = filledFloat32(rowCount, 0);
  }}
  if (!byName[SAMPLE_COLUMN]) {{
    const variable = {{
      name: SAMPLE_COLUMN,
      label: "Sample",
      min: 0,
      max: 0,
      mean: 0,
      finite: rowCount,
      integer: true,
      group: "Samples"
    }};
    variables.unshift(variable);
    byName[SAMPLE_COLUMN] = variable;
    integerVariables.add(SAMPLE_COLUMN);
  }}
}}

function filledFloat32(length, value) {{
  const array = new Float32Array(length);
  array.fill(value);
  return array;
}}

function concatFloat32(left, right) {{
  const merged = new Float32Array(left.length + right.length);
  merged.set(left, 0);
  merged.set(right, left.length);
  return merged;
}}

function concatText(left, right) {{
  return Array.from(left || []).concat(Array.from(right || []));
}}

function decodePayloadColumns(nextPayload) {{
  const numeric = {{}};
  const text = {{}};
  for (const [name, value] of Object.entries(nextPayload.columns || {{}})) {{
    if (value && value.dtype === "float32") {{
      const binary = atob(value.data);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      numeric[name] = new Float32Array(bytes.buffer);
    }} else if (Array.isArray(value)) {{
      text[name] = value.map(item => String(item));
    }}
  }}
  return {{numeric, text}};
}}

function parseVisualizerPayload(html) {{
  const match = String(html).match(/const payload = (.*?);\\nconst columns/s);
  if (!match) throw new Error("No embedded visualizer payload found.");
  return JSON.parse(match[1]);
}}

function readTextFile(file) {{
  return new Promise((resolve, reject) => {{
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Could not read file."));
    reader.readAsText(file);
  }});
}}

function canFetchRemoteFiles() {{
  return window.location.protocol === "http:" || window.location.protocol === "https:";
}}

function remoteRootUrl() {{
  return new URL("/", window.location.href);
}}

function normalizeDirectoryUrl(url) {{
  const normalized = new URL(url, window.location.href);
  normalized.hash = "";
  normalized.search = "";
  if (!normalized.pathname.endsWith("/")) {{
    normalized.pathname = normalized.pathname.replace(/[^/]*$/, "");
  }}
  return normalized;
}}

function cleanRemoteLabel(label) {{
  return String(label || "").replace(/\\/$/, "") || "/";
}}

function safeDecodeRemote(value) {{
  try {{
    return decodeURIComponent(String(value || ""));
  }} catch (error) {{
    return String(value || "");
  }}
}}

function sameRemoteFile(url, href) {{
  const other = new URL(href, window.location.href);
  return url.origin === other.origin && url.pathname === other.pathname;
}}

function remoteFileName(url) {{
  const parts = safeDecodeRemote(new URL(url, window.location.href).pathname).split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "visualizer.html";
}}

function parseRemoteDirectoryListing(html, baseUrl) {{
  const documentView = new DOMParser().parseFromString(String(html), "text/html");
  const entries = [];
  const seen = new Set();
  for (const link of documentView.querySelectorAll("a")) {{
    const href = link.getAttribute("href");
    if (!href || href.startsWith("#") || href.startsWith("?")) continue;
    let url;
    try {{
      url = new URL(href, baseUrl);
    }} catch (error) {{
      continue;
    }}
    url.hash = "";
    if (url.origin !== window.location.origin) continue;
    const text = safeDecodeRemote(link.textContent.trim() || "");
    const isParent = href === "../" || text === "../";
    const isDirectory = isParent || href.endsWith("/") || url.pathname.endsWith("/");
    const isHtml = /\\.html?$/i.test(url.pathname);
    if (!isDirectory && !isHtml) continue;
    if (!isParent && isHtml && sameRemoteFile(url, window.location.href)) continue;
    const key = url.href;
    if (seen.has(key)) continue;
    seen.add(key);
    entries.push({{
      url,
      label: isParent ? ".." : cleanRemoteLabel(text || remoteFileName(url)),
      type: isDirectory ? "directory" : "file",
      parent: isParent
    }});
  }}
  entries.sort((left, right) => {{
    if (left.parent !== right.parent) return left.parent ? -1 : 1;
    if (left.type !== right.type) return left.type === "directory" ? -1 : 1;
    return left.label.localeCompare(right.label);
  }});
  return entries;
}}

function remoteDirectoryLabel(url) {{
  const decoded = safeDecodeRemote(url.pathname || "/");
  return decoded || "/";
}}

function updateRemoteSelectionMessage() {{
  const count = remoteSelections.size;
  el("loadBrowserMessage").textContent = count
    ? `${{count}} selected`
    : "Select generated visualizer HTML files from the served farm directory.";
}}

function renderRemoteDirectory(entries) {{
  const target = el("remoteFileList");
  target.innerHTML = "";
  if (!entries.length) {{
    target.textContent = "No subdirectories or generated visualizer HTML files found here.";
    updateRemoteSelectionMessage();
    return;
  }}
  for (const entry of entries) {{
    if (entry.type === "directory") {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "remote-entry";
      const label = document.createElement("span");
      label.textContent = entry.parent ? "../" : `${{entry.label}}/`;
      button.appendChild(label);
      button.addEventListener("click", () => showRemoteDirectory(entry.url));
      target.appendChild(button);
    }} else {{
      const label = document.createElement("label");
      label.className = "remote-entry";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = remoteSelections.has(entry.url.href);
      checkbox.addEventListener("input", () => {{
        if (checkbox.checked) {{
          remoteSelections.set(entry.url.href, {{url: entry.url.href, label: entry.label}});
        }} else {{
          remoteSelections.delete(entry.url.href);
        }}
        updateRemoteSelectionMessage();
      }});
      const text = document.createElement("span");
      text.textContent = entry.label;
      label.append(checkbox, text);
      target.appendChild(label);
    }}
  }}
  updateRemoteSelectionMessage();
}}

async function showRemoteDirectory(url) {{
  remoteDirectoryUrl = normalizeDirectoryUrl(url);
  el("loadBrowserPath").textContent = remoteDirectoryLabel(remoteDirectoryUrl);
  el("remoteFileList").innerHTML = "";
  el("loadBrowserMessage").textContent = "Loading directory...";
  try {{
    const response = await fetch(remoteDirectoryUrl.href, {{cache: "no-store"}});
    if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
    const html = await response.text();
    renderRemoteDirectory(parseRemoteDirectoryListing(html, remoteDirectoryUrl));
  }} catch (error) {{
    el("loadBrowserMessage").textContent = `Could not read directory: ${{error.message || error}}`;
  }}
}}

async function openRemoteLoadBrowser() {{
  if (!canFetchRemoteFiles()) {{
    el("datasetStatus").textContent = "Serve the visualizer over HTTP to browse farm-side files.";
    el("loadFileInput").click();
    return;
  }}
  remoteSelections.clear();
  el("loadBrowser").classList.remove("hidden");
  await showRemoteDirectory(remoteRootUrl());
}}

function closeRemoteLoadBrowser() {{
  el("loadBrowser").classList.add("hidden");
}}

function goRemoteParentDirectory() {{
  if (!remoteDirectoryUrl) return;
  showRemoteDirectory(new URL("../", remoteDirectoryUrl));
}}

async function loadSelectedRemoteFiles() {{
  const selected = Array.from(remoteSelections.values());
  if (!selected.length) {{
    el("loadBrowserMessage").textContent = "Select one or more generated visualizer HTML files first.";
    return;
  }}
  const status = el("datasetStatus");
  let loaded = 0;
  for (const item of selected) {{
    try {{
      status.textContent = `Loading ${{item.label}}...`;
      el("loadBrowserMessage").textContent = `Loading ${{item.label}}...`;
      const response = await fetch(item.url, {{cache: "no-store"}});
      if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
      const html = await response.text();
      const nextPayload = parseVisualizerPayload(html);
      const decoded = decodePayloadColumns(nextPayload);
      mergeVisualizerPayload(nextPayload, decoded, item.label || remoteFileName(item.url));
      loaded++;
    }} catch (error) {{
      const message = `Could not load ${{item.label}}: ${{error.message || error}}`;
      status.textContent = message;
      el("loadBrowserMessage").textContent = message;
      console.error(error);
      break;
    }}
  }}
  if (loaded) {{
    remoteSelections.clear();
    closeRemoteLoadBrowser();
    finishLoadedVisualizers(loaded);
  }}
}}

function payloadRowCount(nextPayload, decoded) {{
  if (Number.isFinite(nextPayload.rowCount) && nextPayload.rowCount > 0) return Number(nextPayload.rowCount);
  const numericColumn = Object.values(decoded.numeric)[0];
  if (numericColumn) return numericColumn.length;
  const textColumn = Object.values(decoded.text)[0];
  return textColumn ? textColumn.length : 0;
}}

function ensureVariableFromPayload(name, nextPayload) {{
  if (byName[name]) return;
  const source = (nextPayload.variables || []).find(variable => variable.name === name) || {{}};
  const variable = {{
    name,
    label: source.label || name,
    min: Number.isFinite(source.min) ? source.min : NaN,
    max: Number.isFinite(source.max) ? source.max : NaN,
    mean: Number.isFinite(source.mean) ? source.mean : NaN,
    finite: Number.isFinite(source.finite) ? source.finite : 0,
    integer: Boolean(source.integer),
    group: source.group || "Other"
  }};
  variables.push(variable);
  byName[name] = variable;
  if (variable.integer) integerVariables.add(name);
}}

function refreshVariableStats() {{
  for (const variable of variables) {{
    const values = columns[variable.name];
    if (!values) continue;
    let finite = 0, sum = 0, min = Infinity, max = -Infinity;
    for (const value of values) {{
      if (!Number.isFinite(value)) continue;
      finite++;
      sum += value;
      if (value < min) min = value;
      if (value > max) max = value;
    }}
    variable.finite = finite;
    variable.mean = finite ? sum / finite : NaN;
    variable.min = finite ? min : NaN;
    variable.max = finite ? max : NaN;
  }}
}}

function mergeTextFilters(nextPayload, decoded) {{
  if (!payload.textFilters) payload.textFilters = [];
  const incomingFilters = nextPayload.textFilters || [];
  for (const [name, values] of Object.entries(decoded.text)) {{
    const incoming = incomingFilters.find(filter => filter.name === name) || {{}};
    let filter = payload.textFilters.find(item => item.name === name);
    if (!filter) {{
      filter = {{name, label: incoming.label || name, values: []}};
      payload.textFilters.push(filter);
    }}
    const merged = new Set([...(filter.values || []), ...values.filter(value => value).slice(0, 40)]);
    filter.values = Array.from(merged).sort().slice(0, 40);
  }}
  payload.textFilters.sort((left, right) => (left.label || left.name).localeCompare(right.label || right.name));
}}

function mergeVisualizerPayload(nextPayload, decoded, fileName) {{
  ensureSampleColumn();
  const addedRows = payloadRowCount(nextPayload, decoded);
  if (!addedRows) throw new Error("Loaded payload has no rows.");
  const oldRows = rowCount;
  const sampleId = loadedSamples.length;
  loadedSamples.push({{id: sampleId, label: sampleLabel(fileName || nextPayload.source || `sample ${{sampleId + 1}}`), rows: addedRows}});

  for (const [name] of Object.entries(decoded.numeric)) {{
    ensureVariableFromPayload(name, nextPayload);
  }}

  const existingNumeric = Object.keys(columns);
  for (const name of existingNumeric) {{
    const addition = name === SAMPLE_COLUMN
      ? filledFloat32(addedRows, sampleId)
      : decoded.numeric[name] || filledFloat32(addedRows, NaN);
    columns[name] = concatFloat32(columns[name], addition);
  }}
  for (const [name, values] of Object.entries(decoded.numeric)) {{
    if (existingNumeric.includes(name)) continue;
    columns[name] = concatFloat32(filledFloat32(oldRows, NaN), values);
  }}

  const existingText = Object.keys(textColumns);
  for (const name of existingText) {{
    textColumns[name] = concatText(textColumns[name], decoded.text[name] || Array(addedRows).fill(""));
  }}
  for (const [name, values] of Object.entries(decoded.text)) {{
    if (existingText.includes(name)) continue;
    textColumns[name] = concatText(Array(oldRows).fill(""), values);
  }}

  rowCount += addedRows;
  payload.rowCount = rowCount;
  refreshVariableStats();
  mergeTextFilters(nextPayload, decoded);
  rebuildCategoricalFilters(true);
  rebuildSplitOptions();
  updateDatasetStatus();
}}

async function loadVisualizerFiles(files) {{
  const chosen = Array.from(files || []);
  if (!chosen.length) return;
  const status = el("datasetStatus");
  let loaded = 0;
  for (const file of chosen) {{
    try {{
      status.textContent = `Loading ${{file.name}}...`;
      const html = await readTextFile(file);
      const nextPayload = parseVisualizerPayload(html);
      const decoded = decodePayloadColumns(nextPayload);
      mergeVisualizerPayload(nextPayload, decoded, file.name);
      loaded++;
    }} catch (error) {{
      status.textContent = `Could not load ${{file.name}}: ${{error.message || error}}`;
      console.error(error);
      break;
    }}
  }}
  finishLoadedVisualizers(loaded);
}}

function finishLoadedVisualizers(loaded) {{
  if (loaded) {{
    updateDatasetStatus();
    for (const panel of Object.values(panels)) resetAxisRanges(panel);
    fillSelect(el("rangeVar"), payload.defaultX);
    fillOperationSelects();
    renderCategoryFilters();
    renderQuickCategoryOptions();
    renderQuickCategory();
    renderTextFilters();
    syncControlsFromPanel();
    update();
  }}
}}

function updateDatasetStatus() {{
  const status = el("datasetStatus");
  if (status) status.textContent = "";
  el("sampleCount").textContent = loadedSamples.length.toLocaleString();
  el("embeddedCount").textContent = rowCount.toLocaleString();
  const multiple = loadedSamples.length > 1;
  const labels = loadedSamples.map(sample => sample.label);
  const fullSourceSummary = labels.join(" + ");
  const compactSourceSummary = labels.length > 3
    ? `${{labels.slice(0, 2).join(" + ")}} + ${{labels.length - 2}} more`
    : fullSourceSummary;
  el("datasetKicker").textContent = multiple ? "Workspace" : "Dataset";
  document.querySelector("h1").textContent = multiple
    ? `${{loadedSamples.length.toLocaleString()}} samples combined`
    : payload.title;
  el("source").textContent = multiple ? compactSourceSummary : payload.source;
  el("source").hidden = !multiple;
  el("source").title = multiple ? fullSourceSummary : payload.source;
  document.querySelector(".dataset-heading").title = el("source").title;
}}

function comparisonDefaultX() {{
  for (const name of ["rec_minus_t_pi0", "t_pi0", "gen_minus_t", payload.defaultX]) {{
    if (name && columns[name]) return name;
  }}
  return payload.defaultX;
}}

function makePanel(key, xvar, yvar) {{
  const xInfo = byName[xvar] || variables[0];
  const yInfo = byName[yvar] || variables[1] || xInfo;
  return {{
    key,
    mode: "2d",
    xvar: xInfo.name,
    x2var: "",
    yvar: yInfo.name,
    y2var: "",
    xLabel: "",
    yLabel: "",
    splitVar: "",
    sliceBins: 6,
    sliceEdges: "",
    xbins: 80,
    ybins: 80,
    xticks: 6,
    yticks: 6,
    xmin: xInfo.min,
    xmax: xInfo.max,
    ymin: yInfo.min,
    ymax: yInfo.max,
    logz: true,
    density: false,
    colorScale: true,
    plotHeightFraction: 0.5,
    plotWidthFraction: 1,
    fitModel: "none",
    signalModel: "none",
    backgroundModel: "none",
    fitMethod: "unweighted",
    fitScanDetail: 3,
    fitRangeClick: false,
    showFitAnnotations: true,
    showMeanGuides: false,
    fitRangeMin: NaN,
    fitRangeMax: NaN,
    fitSummary: "No fit",
    profile: null,
    filterState: makeFilterState(),
    ghostPlot: null,
    referenceCurves: [],
    pinnedMarkers: [],
    nextPinnedMarkerColor: 0,
    lastPlot: null,
    stats: {{selected: 0, meanX: NaN, meanY: NaN}}
  }};
}}

function makeFilterState() {{
  return {{categories: {{}}, ranges: [], text: {{}}}};
}}

const persistedPanelKeys = [
  "mode", "xvar", "x2var", "yvar", "y2var", "xLabel", "yLabel", "splitVar",
  "sliceBins", "sliceEdges", "xbins", "ybins", "xticks", "yticks", "xmin", "xmax",
  "ymin", "ymax", "logz", "density", "colorScale", "plotHeightFraction", "plotWidthFraction", "signalModel",
  "backgroundModel", "fitMethod", "fitScanDetail", "fitRangeClick", "showFitAnnotations",
  "showMeanGuides", "fitRangeMin", "fitRangeMax", "referenceCurves"
];

function serializableFilterState(state) {{
  return {{
    categories: Object.fromEntries(
      Object.entries(state.categories).map(([name, values]) => [name, Array.from(values)])
    ),
    ranges: state.ranges.map(filter => ({{...filter}})),
    text: {{...state.text}}
  }};
}}

function serializablePanel(panel) {{
  const saved = Object.fromEntries(persistedPanelKeys.map(key => [key, panel[key]]));
  saved.filterState = serializableFilterState(panel.filterState);
  return saved;
}}

function workspaceSnapshot() {{
  return {{
    version: WORKSPACE_STORAGE_VERSION,
    enabledPanels: [...enabledPanels],
    activePanel,
    compareMode,
    sharedPanelFilters,
    topologyCollapsed,
    canvasToolbarCollapsed,
    sharedFilterState: serializableFilterState(sharedFilterState),
    panels: Object.fromEntries(panelKeys.map(key => [key, serializablePanel(panels[key])]))
  }};
}}

function applySavedFilterState(target, saved) {{
  if (!saved || typeof saved !== "object") return;
  target.categories = Object.fromEntries(
    Object.entries(saved.categories || {{}}).map(([name, values]) => [name, new Set(values.map(Number))])
  );
  target.ranges = Array.isArray(saved.ranges)
    ? saved.ranges.filter(filter => columns[filter.name]).map(filter => ({{...filter}}))
    : [];
  target.text = {{...(saved.text || {{}})}};
}}

function restoreWorkspace(showStatus = false) {{
  let saved;
  try {{
    saved = JSON.parse(localStorage.getItem(WORKSPACE_STORAGE_KEY) || "null");
  }} catch (error) {{
    if (showStatus) el("datasetStatus").textContent = "Saved workspace could not be read.";
    return false;
  }}
  if (!saved || ![1, 2, WORKSPACE_STORAGE_VERSION].includes(saved.version)) {{
    if (showStatus) el("datasetStatus").textContent = "No saved workspace for this dataset.";
    return false;
  }}
  applySavedFilterState(sharedFilterState, saved.sharedFilterState);
  for (const key of panelKeys) {{
    const savedPanel = saved.panels?.[key];
    if (!savedPanel) continue;
    if (savedPanel.plotHeightFraction === undefined) {{
      if (saved.version === 1 && savedPanel.plotAspect !== undefined) {{
        savedPanel.plotHeightFraction = 1 / Number(savedPanel.plotAspect) / MAX_PLOT_HEIGHT_TO_WIDTH;
      }} else if (saved.version === 2 && savedPanel.plotHeightScale !== undefined) {{
        savedPanel.plotHeightFraction = Number(savedPanel.plotHeightScale) / MAX_PLOT_HEIGHT_TO_WIDTH;
      }}
    }}
    for (const name of persistedPanelKeys) {{
      if (savedPanel[name] !== undefined) panels[key][name] = savedPanel[name];
    }}
    if (!columns[panels[key].xvar]) panels[key].xvar = payload.defaultX;
    if (!columns[panels[key].yvar]) panels[key].yvar = payload.defaultY;
    if (panels[key].x2var && !columns[panels[key].x2var]) panels[key].x2var = "";
    if (panels[key].y2var && !columns[panels[key].y2var]) panels[key].y2var = "";
    if (panels[key].splitVar && !columns[panels[key].splitVar]) panels[key].splitVar = "";
    for (const curve of panels[key].referenceCurves || []) {{
      referenceCurveId = Math.max(referenceCurveId, Number(curve.id) || 0);
    }}
    applySavedFilterState(panels[key].filterState, savedPanel.filterState);
  }}
  enabledPanels = (saved.enabledPanels || ["A"]).filter(key => panelKeys.includes(key));
  if (!enabledPanels.length) enabledPanels = ["A"];
  activePanel = enabledPanels.includes(saved.activePanel) ? saved.activePanel : enabledPanels[0];
  compareMode = Boolean(saved.compareMode && enabledPanels.length > 1);
  sharedPanelFilters = saved.sharedPanelFilters !== false;
  topologyCollapsed = saved.topologyCollapsed !== false;
  canvasToolbarCollapsed = Boolean(saved.canvasToolbarCollapsed);
  useActiveFilterState();
  initializeCategoryState();
  if (showStatus) el("datasetStatus").textContent = "Saved workspace restored.";
  return true;
}}

function saveWorkspace(showStatus = false) {{
  try {{
    localStorage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify(workspaceSnapshot()));
    if (showStatus) el("datasetStatus").textContent = "Workspace saved in this browser.";
    return true;
  }} catch (error) {{
    if (showStatus) el("datasetStatus").textContent = "Workspace could not be saved in this browser.";
    return false;
  }}
}}

function scheduleWorkspaceSave() {{
  window.clearTimeout(workspaceSaveTimer);
  workspaceSaveTimer = window.setTimeout(() => saveWorkspace(false), 250);
}}

function scheduleUpdate() {{
  if (updateFrame !== null) cancelAnimationFrame(updateFrame);
  updateFrame = requestAnimationFrame(() => {{
    updateFrame = null;
    update();
  }});
}}

function allFilterStates() {{
  return [sharedFilterState, ...panelKeys.map(key => panels[key].filterState)];
}}

function filterStateForPanel(key = activePanel) {{
  return sharedPanelFilters ? sharedFilterState : panels[key].filterState;
}}

function useActiveFilterState() {{
  const state = filterStateForPanel(activePanel);
  categoryState = state.categories;
  activeRanges = state.ranges;
  return state;
}}

function copyFilterState(target, source) {{
  target.categories = Object.fromEntries(
    Object.entries(source.categories).map(([name, values]) => [name, new Set(values)])
  );
  target.ranges = source.ranges.map(filter => ({{...filter}}));
  target.text = {{...source.text}};
}}

function currentPanel() {{
  return panels[activePanel];
}}

function appendGroupedOptions(select, items, valueFor, labelFor, groupFor) {{
  select.innerHTML = "";
  let currentGroup = "";
  let groupNode = null;
  for (const item of items) {{
    const group = groupFor(item) || "Other";
    if (group !== currentGroup) {{
      currentGroup = group;
      groupNode = document.createElement("optgroup");
      groupNode.label = group;
      select.appendChild(groupNode);
    }}
    const option = document.createElement("option");
    const label = labelFor(item);
    option.value = valueFor(item);
    option.textContent = label;
    option.title = `${{group}}: ${{label}}`;
    groupNode.appendChild(option);
  }}
}}

function fillSelect(select, selected) {{
  appendGroupedOptions(
    select,
    variables,
    variable => variable.name,
    variable => variable.label,
    variable => variable.group,
  );
  select.value = selected;
}}

function fillOverlaySelect(select, selected) {{
  fillSelect(select, selected);
  select.value = selected && byName[selected] ? selected : "";
}}

function firstAdditionalVariable(primary) {{
  const candidate = variables.find(variable => variable.name !== primary);
  return candidate ? candidate.name : "";
}}

function fillOperationSelects() {{
  const left = el("opLeft");
  const right = el("opRight");
  const currentLeft = left.value || firstPresent(["rec_theta_deg", "recTheta_deg", "rec_theta", "theta_deg", payload.defaultY]);
  const currentRight = right.value || matchingGeneratedName(currentLeft) || firstPresent(["gen_theta_deg", "gen_theta", payload.defaultX]);
  fillSelect(left, currentLeft);
  fillSelect(right, currentRight);
}}

function firstPresent(names) {{
  for (const name of names) {{
    if (name && columns[name]) return name;
  }}
  return variables[0]?.name || "";
}}

function matchingGeneratedName(name) {{
  if (!name) return "";
  const candidates = [];
  if (name.startsWith("rec_")) candidates.push("gen_" + name.slice(4));
  if (name.startsWith("rec")) candidates.push("gen" + name.slice(3));
  candidates.push(name.replace(/^rec_?/, "gen_"));
  for (const candidate of candidates) {{
    if (columns[candidate]) return candidate;
  }}
  return "";
}}

function addDerivedVariable() {{
  const leftName = el("opLeft").value;
  const rightName = el("opRight").value;
  const kind = el("opKind").value;
  const left = columns[leftName];
  const right = columns[rightName];
  if (!left || !right || leftName === rightName) {{
    el("opStatus").textContent = "Choose two different numeric variables.";
    return;
  }}
  const values = new Float32Array(rowCount);
  let finite = 0, sum = 0, min = Infinity, max = -Infinity;
  for (let i = 0; i < rowCount; i++) {{
    const a = left[i];
    const b = right[i];
    let value = NaN;
    if (Number.isFinite(a) && Number.isFinite(b)) {{
      if (kind === "subtract") value = a - b;
      else if (kind === "add") value = a + b;
      else if (kind === "ratio") value = b !== 0 ? a / b : NaN;
      else if (kind === "fractional") value = b !== 0 ? (a - b) / b : NaN;
    }}
    values[i] = value;
    if (Number.isFinite(value)) {{
      finite++;
      sum += value;
      if (value < min) min = value;
      if (value > max) max = value;
    }}
  }}
  if (!finite) {{
    el("opStatus").textContent = "No finite values were produced.";
    return;
  }}
  const label = derivedLabel(leftName, rightName, kind);
  const name = uniqueDerivedName(label);
  columns[name] = values;
  const variable = {{name, label, min, max, mean: sum / finite, finite, group: "Derived"}};
  variables.push(variable);
  byName[name] = variable;
  fillSelect(el("rangeVar"), name);
  fillOperationSelects();
  for (const key of panelKeys) {{
    if (panels[key].xvar === leftName) {{
      panels[key].xvar = name;
      panels[key].xmin = min;
      panels[key].xmax = max;
      break;
    }}
  }}
  currentPanel().xvar = name;
  currentPanel().xmin = min;
  currentPanel().xmax = max;
  el("opStatus").textContent = `Added ${{label}}`;
  syncControlsFromPanel();
  update();
}}

function derivedLabel(leftName, rightName, kind) {{
  const left = byName[leftName]?.label || leftName;
  const right = byName[rightName]?.label || rightName;
  if (kind === "add") return `${{left}} + ${{right}}`;
  if (kind === "ratio") return `${{left}} / ${{right}}`;
  if (kind === "fractional") return `(${{left}} - ${{right}}) / ${{right}}`;
  return `${{left}} - ${{right}}`;
}}

function uniqueDerivedName(label) {{
  const base = "derived_" + label.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 70);
  let name = base || "derived_value";
  let suffix = 2;
  while (columns[name] || textColumns[name]) {{
    name = `${{base}}_${{suffix++}}`;
  }}
  return name;
}}

function fillSplitSelect(selected) {{
  const label = el("splitLabel");
  const select = el("splitVar");
  select.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "none";
  select.appendChild(none);
  const categorical = document.createElement("optgroup");
  categorical.label = "Categories";
  for (const split of payload.sectorSplits || []) {{
    const option = document.createElement("option");
    option.value = split.name;
    option.textContent = split.label;
    categorical.appendChild(option);
  }}
  if (categorical.children.length) select.appendChild(categorical);
  const categoricalNames = new Set((payload.sectorSplits || []).map(split => split.name));
  const numeric = document.createElement("optgroup");
  numeric.label = "Numeric slices";
  for (const variable of variables) {{
    if (!columns[variable.name] || categoricalNames.has(variable.name)) continue;
    const option = document.createElement("option");
    option.value = variable.name;
    option.textContent = variable.label;
    numeric.appendChild(option);
  }}
  if (numeric.children.length) select.appendChild(numeric);
  select.value = selected || "";
  if (select.value !== (selected || "")) select.value = "";
  label.style.display = select.options.length > 1 ? "" : "none";
}}

function splitFacets(splitName) {{
  const filter = payload.categoricalFilters.find(item => item.name === splitName);
  if (filter) {{
    const definitions = filter.values.map((value, index) => ({{
      value: Number(value),
      label: isProtonSectorSplit(splitName) && Math.round(Number(value)) === 0
        ? "CD proton (sector 0)"
        : filter.labels[index] || String(value),
      shortLabel: shortFacetLabel(filter, value, index)
    }}));
    return orderSplitFacets(splitName, definitions);
  }}
  const values = columns[splitName];
  if (!values) return [];
  const unique = Array.from(new Set(Array.from(values).filter(Number.isFinite).map(value => Math.round(value)))).sort((a, b) => a - b).slice(0, 12);
  return orderSplitFacets(splitName, unique.map(value => ({{
    value,
    label: isProtonSectorSplit(splitName) && value === 0 ? "CD proton (sector 0)" : String(value),
    shortLabel: isProtonSectorSplit(splitName) && value === 0 ? "CD" : String(value)
  }})));
}}

function isCategoricalSplit(splitName) {{
  return (payload.sectorSplits || []).some(split => split.name === splitName);
}}

function parseManualSliceEdges(text) {{
  const source = String(text || "").trim();
  if (!source) return {{edges: null, error: ""}};
  const parts = source.split(/[\\s,;]+/).filter(Boolean);
  const edges = parts.map(Number);
  if (edges.length < 2 || edges.some(value => !Number.isFinite(value))) {{
    return {{edges: null, error: "Enter at least two finite, comma-separated edges."}};
  }}
  if (edges.length > 25) return {{edges: null, error: "Use no more than 25 edges (24 slices)."}};
  for (let index = 1; index < edges.length; index++) {{
    if (!(edges[index] > edges[index - 1])) {{
      return {{edges: null, error: "Manual edges must be strictly increasing."}};
    }}
  }}
  return {{edges, error: ""}};
}}

function numericSliceConfiguration(panel, splitName) {{
  const manual = parseManualSliceEdges(panel.sliceEdges);
  if (manual.edges) return {{edges: manual.edges, manual: true, error: ""}};
  const variable = byName[splitName];
  const minimum = Number(variable?.min);
  const maximum = Number(variable?.max);
  if (!(maximum > minimum)) {{
    return {{edges: [], manual: false, error: manual.error || "This quantity has no finite slicing range."}};
  }}
  const bins = clamp(Math.round(Number(panel.sliceBins) || 6), 1, 24);
  const edges = Array.from({{length: bins + 1}}, (_, index) => minimum + (maximum - minimum) * index / bins);
  return {{edges, manual: false, error: manual.error}};
}}

function numericSliceFacets(panel, splitName) {{
  const configuration = numericSliceConfiguration(panel, splitName);
  const label = variableLabel(splitName);
  const definitions = [];
  for (let index = 0; index + 1 < configuration.edges.length; index++) {{
    const lower = configuration.edges[index];
    const upper = configuration.edges[index + 1];
    const last = index === configuration.edges.length - 2;
    const interval = `[${{formatAxisTick(lower)}}, ${{formatAxisTick(upper)}}${{last ? "]" : ")"}}`;
    definitions.push({{
      value: index,
      lower,
      upper,
      last,
      numericSlice: true,
      label: `${{label}} ${{interval}}`,
      shortLabel: interval
    }});
  }}
  return definitions;
}}

function facetDefinitionsForPanel(panel, splitName) {{
  return isCategoricalSplit(splitName) ? splitFacets(splitName) : numericSliceFacets(panel, splitName);
}}

function valueMatchesFacet(value, definition) {{
  if (definition.numericSlice) {{
    return Number.isFinite(value) && value >= definition.lower && (definition.last ? value <= definition.upper : value < definition.upper);
  }}
  return Math.round(value) === Number(definition.value);
}}

function panelSplitSignature(panel, splitName) {{
  if (!splitName) return "";
  if (isCategoricalSplit(splitName)) return `category:${{splitName}}`;
  const configuration = numericSliceConfiguration(panel, splitName);
  return `numeric:${{splitName}}:${{configuration.edges.map(value => Number(value).toPrecision(12)).join(",")}}`;
}}

function updateSliceControls(panel) {{
  const numeric = Boolean(panel.splitVar) && !isCategoricalSplit(panel.splitVar);
  el("sliceControls").hidden = !numeric;
  if (!numeric) {{ el("sliceStatus").textContent = ""; return; }}
  const configuration = numericSliceConfiguration(panel, panel.splitVar);
  if (configuration.error) {{
    el("sliceStatus").textContent = `${{configuration.error}} Using equal-width slices until the edges are valid.`;
  }} else if (configuration.manual) {{
    el("sliceStatus").textContent = `${{configuration.edges.length - 1}} manual slices; values outside the edge range are omitted.`;
  }} else {{
    el("sliceStatus").textContent = `${{configuration.edges.length - 1}} equal-width slices over the embedded ${{variableLabel(panel.splitVar)}} range.`;
  }}
}}

function isProtonSectorSplit(name) {{
  const canonical = String(name || "").toLowerCase().replace(/^(?:rec|gen)[_.]/, "").replace(/[_.]/g, "");
  return canonical === "psector" || canonical === "protonsector";
}}

function orderSplitFacets(splitName, definitions) {{
  if (!isProtonSectorSplit(splitName)) return definitions;
  return definitions.slice().sort((left, right) => {{
    const leftValue = Math.round(Number(left.value));
    const rightValue = Math.round(Number(right.value));
    if (leftValue === 0 && rightValue !== 0) return 1;
    if (rightValue === 0 && leftValue !== 0) return -1;
    return leftValue - rightValue;
  }});
}}

function shortFacetLabel(filter, value, index) {{
  if (filter.name === SAMPLE_COLUMN) return `sample ${{Number(value) + 1}}`;
  if (isProtonSectorSplit(filter.name) && Math.round(Number(value)) === 0) return "CD";
  if (filter.name.toLowerCase().includes("sector")) return `S${{Math.round(Number(value))}}`;
  return filter.labels[index] || String(value);
}}

async function init() {{
  setStartupProgress(64, "Building dataset controls…");
  await yieldStartupFrame();
  document.querySelector("h1").textContent = payload.title;
  el("source").textContent = payload.source;
  el("source").title = payload.source;
  document.querySelector(".dataset-heading").title = payload.source;
  el("embeddedCount").textContent = rowCount.toLocaleString();
  const samplingNotes = [];
  if (payload.downsample.sampled) {{
    const seedNote = Number.isInteger(payload.downsample.seed)
      ? `, seed ${{payload.downsample.seed}}`
      : "";
    if (payload.downsample.unit === "source-events") {{
      samplingNotes.push(`sampled ${{payload.downsample.embeddedEvents.toLocaleString()}} of ${{payload.downsample.originalEvents.toLocaleString()}} source events (${{payload.downsample.embeddedRows.toLocaleString()}} rows)${{seedNote}}`);
    }} else {{
      samplingNotes.push(`sampled ${{payload.downsample.embeddedRows.toLocaleString()}} of ${{payload.downsample.originalRows.toLocaleString()}} rows${{seedNote}}`);
    }}
  }}
  if (payload.downsample.filter) samplingNotes.push(`filter: ${{payload.downsample.filter}}`);
  el("samplingNote").textContent = samplingNotes.join("; ");
  ensureSampleColumn();
  rebuildCategoricalFilters(false);
  rebuildSplitOptions();
  updateDatasetStatus();
  setStartupProgress(74, "Preparing filters and quantities…");
  await yieldStartupFrame();
  fillSelect(el("rangeVar"), payload.defaultX);
  fillOperationSelects();
  initializeCategoryState();
  restoreWorkspace(false);
  renderCategoryFilters();
  renderQuickCategoryOptions();
  renderQuickCategory();
  renderTextFilters();
  attachEvents();
  renderPanelTabs();
  syncControlsFromPanel();
  setStartupProgress(90, "Rendering the initial histogram…");
  await yieldStartupFrame();
  update();
  setStartupProgress(100, "Ready");
  await yieldStartupFrame();
  const startupLoading = el("startupLoading");
  document.body.removeAttribute("aria-busy");
  requestAnimationFrame(() => {{
    startupLoading.classList.add("complete");
    window.setTimeout(() => startupLoading.remove(), 180);
  }});
}}

function initializeCategoryState() {{
  for (const state of allFilterStates()) {{
    for (const filter of payload.categoricalFilters) {{
      if (!state.categories[filter.name]) {{
        state.categories[filter.name] = new Set(filter.values.map(value => Number(value)));
      }}
    }}
  }}
  useActiveFilterState();
}}

function rebuildCategoricalFilters(preserveSelections = true) {{
  const previousValues = {{}};
  const previousByState = new Map();
  if (preserveSelections) {{
    for (const filter of payload.categoricalFilters || []) {{
      previousValues[filter.name] = new Set(filter.values.map(value => Number(value)));
    }}
    for (const state of allFilterStates()) {{
      previousByState.set(state, Object.fromEntries(
        Object.entries(state.categories).map(([name, values]) => [name, new Set(values)])
      ));
    }}
  }}
  const filters = [];
  for (const variable of variables) {{
    const values = columns[variable.name];
    if (!values) continue;
    const filter = categoricalFilterFromColumn(variable, values);
    if (filter) filters.push(filter);
  }}
  filters.sort(compareCategoricalFilters);
  payload.categoricalFilters = filters;
  for (const state of allFilterStates()) {{
    const previous = previousByState.get(state) || {{}};
    for (const filter of filters) {{
      const prior = previous[filter.name];
      const knownValues = previousValues[filter.name];
      const next = new Set();
      for (const value of filter.values) {{
        const numeric = Number(value);
        if (!prior || prior.has(numeric) || !knownValues || !knownValues.has(numeric)) next.add(numeric);
      }}
      state.categories[filter.name] = next;
    }}
    for (const name of Object.keys(state.categories)) {{
      if (!filters.some(filter => filter.name === name)) delete state.categories[name];
    }}
  }}
  useActiveFilterState();
}}

function categoricalFilterFromColumn(variable, values) {{
  const finite = [];
  for (const value of values) if (Number.isFinite(value)) finite.push(value);
  if (!finite.length) return null;
  const integers = finite.every(value => Math.abs(value - Math.round(value)) < 1.0e-6);
  const unique = Array.from(new Set(finite.map(value => integers ? Math.round(value) : value))).sort((a, b) => a - b);
  const maxCategories = variable.name === SAMPLE_COLUMN ? 100 : isRunNumberName(variable.name) ? 500 : isIndexName(variable.name) ? 40 : 12;
  const categorical = variable.name === SAMPLE_COLUMN || integers || isPassFlagName(variable.name) || variable.name.endsWith("Det") || isIndexName(variable.name) || isRunNumberName(variable.name);
  if (unique.length <= 1 || unique.length > maxCategories || !categorical) return null;
  return {{
    name: variable.name,
    label: variable.label || variable.name,
    group: categoricalGroup(variable.name, variable.group),
    values: unique,
    labels: unique.map(value => categoryValueLabel(variable.name, value))
  }};
}}

function compareCategoricalFilters(left, right) {{
  return categoricalGroupRank(left.group) - categoricalGroupRank(right.group)
    || categoricalKindRank(left.name) - categoricalKindRank(right.name)
    || left.label.localeCompare(right.label)
    || left.name.localeCompare(right.name);
}}

function categoricalGroup(name, variableGroup) {{
  if (name === SAMPLE_COLUMN) return "Samples";
  return variableGroup || "Other";
}}

function categoricalGroupRank(group) {{
  const order = ["Samples", "Event", "Selections", "Electron", "Proton", "Gamma 1", "Gamma 2", "Gamma", "Pi0", "Sectors", "Detectors", "Indices", "Kinematics", "Masses / Exclusivity", "REC particle", "GEN particle", "Detector / Geometry", "Derived", "Other"];
  const index = order.indexOf(group || "Other");
  return index >= 0 ? index : order.length;
}}

function categoricalKindRank(name) {{
  if (name === SAMPLE_COLUMN) return 0;
  if (isRunNumberName(name)) return 1;
  const canonical = name.toLowerCase().replace(/^(?:rec|gen|event)[_.]/, "").replace(/[_.]/g, "");
  if (canonical.endsWith("det") || canonical.endsWith("detector")) return 2;
  if (canonical.includes("sector")) return 3;
  if (isIndexName(name)) return 4;
  if (isPassFlagName(name)) return 5;
  return 6;
}}

function categoryValueLabel(name, value) {{
  if (name === SAMPLE_COLUMN) {{
    const sample = loadedSamples.find(item => item.id === Number(value));
    return sample ? sample.label : `sample ${{Number(value) + 1}}`;
  }}
  if (isRunNumberName(name)) return String(Math.round(value));
  if (["pDet", "rec_proton_detector", "protonDet", "protonDetector"].includes(name)) {{
    return {{1: "FD", 2: "CD", 0: "FT", "-999": "missing"}}[String(Math.round(value))] || String(value);
  }}
  if (name.toLowerCase().includes("sector")) {{
    const sector = Math.round(value);
    return sector < 0 ? "missing" : `sector ${{sector}}`;
  }}
  if (isPassFlagName(name) || name === "rec_selected" || name === "rec_not_selected") {{
    const flag = Math.round(value);
    if (flag === 1) return "pass";
    if (flag === 0) return "fail";
    return "missing";
  }}
  return String(value);
}}

function isPassFlagName(name) {{
  return name.startsWith("pass") || name.startsWith("rec_pass");
}}

function isIndexName(name) {{
  return name.endsWith("Idx") || name.endsWith("Index");
}}

function isRunNumberName(name) {{
  return ["run", "runNum", "rec_runNum", "gen_runNum"].includes(name) || name.toLowerCase().endsWith("runnum");
}}

function rebuildSplitOptions() {{
  const splits = [];
  const addSplit = filter => {{
    if (filter && columns[filter.name] && !splits.some(split => split.name === filter.name)) {{
      splits.push({{name: filter.name, label: filter.label}});
    }}
  }};
  addSplit(payload.categoricalFilters.find(filter => filter.name === SAMPLE_COLUMN));
  for (const filter of payload.categoricalFilters) {{
    if (filter.name.toLowerCase().includes("sector")) addSplit(filter);
  }}
  payload.sectorSplits = splits;
}}

function renderCategoryFilters() {{
  const target = el("categoryFilters");
  target.innerHTML = "";
  if (!payload.categoricalFilters.length) {{
    target.textContent = "No categorical filters available.";
    return;
  }}
  let currentGroup = "";
  for (const filter of payload.categoricalFilters) {{
    const group = filter.group || "Other";
    if (group !== currentGroup) {{
      currentGroup = group;
      const groupTitle = document.createElement("div");
      groupTitle.className = "category-group-title";
      groupTitle.textContent = group;
      target.appendChild(groupTitle);
    }}
    const block = document.createElement("details");
    block.className = "filter-details";
    const title = document.createElement("summary");
    title.textContent = categorySummaryText(filter);
    const chips = document.createElement("div");
    chips.className = "chips";
    filter.values.forEach((value, index) => {{
      chips.appendChild(categoryChip(filter, value, filter.labels[index], () => {{
        renderCategoryFilters();
        renderQuickCategory();
      }}));
    }});
    block.appendChild(title);
    block.appendChild(chips);
    target.appendChild(block);
  }}
}}

function renderQuickCategoryOptions() {{
  const block = el("quickCategoryBlock");
  const select = el("quickCategoryFilter");
  if (!payload.categoricalFilters.length) {{
    block.style.display = "none";
    return;
  }}
  block.style.display = "";
  const previous = select.value;
  appendGroupedOptions(
    select,
    payload.categoricalFilters,
    filter => filter.name,
    filter => filter.label,
    filter => filter.group,
  );
  select.value = payload.categoricalFilters.some(filter => filter.name === previous)
    ? previous
    : payload.categoricalFilters[0].name;
  syncTopologyCollapse();
}}

function renderQuickCategory() {{
  const select = el("quickCategoryFilter");
  const chips = el("quickCategoryChips");
  const summary = el("quickCategorySummary");
  chips.innerHTML = "";
  const filter = payload.categoricalFilters.find(item => item.name === select.value);
  if (!filter) {{
    summary.textContent = "";
    return;
  }}
  filter.values.forEach((value, index) => {{
    chips.appendChild(categoryChip(filter, value, filter.labels[index], () => {{
      renderQuickCategory();
      renderCategoryFilters();
    }}));
  }});
  summary.textContent = categorySummaryText(filter);
}}

function toggleTopology() {{
  topologyCollapsed = !topologyCollapsed;
  syncTopologyCollapse();
}}

function syncTopologyCollapse() {{
  const block = el("quickCategoryBlock");
  const button = el("toggleTopology");
  if (!block || !button) return;
  block.classList.toggle("collapsed", topologyCollapsed);
  button.textContent = topologyCollapsed ? ">" : "v";
  button.setAttribute("aria-expanded", topologyCollapsed ? "false" : "true");
  button.setAttribute("aria-label", topologyCollapsed ? "Expand filter topology" : "Collapse filter topology");
}}

function categoryChip(filter, value, text, afterChange) {{
  const label = document.createElement("label");
  label.className = "chip";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = categoryState[filter.name]?.has(Number(value)) ?? true;
  input.addEventListener("input", () => {{
    setCategoryValue(filter.name, value, input.checked);
    afterChange();
    update();
  }});
  label.appendChild(input);
  label.appendChild(document.createTextNode(text));
  return label;
}}

function setCategoryValue(name, value, enabled) {{
  if (!categoryState[name]) categoryState[name] = new Set();
  const numeric = Number(value);
  if (enabled) categoryState[name].add(numeric);
  else categoryState[name].delete(numeric);
}}

function setCurrentCategoryValues(enabled) {{
  const filter = payload.categoricalFilters.find(item => item.name === el("quickCategoryFilter").value);
  if (!filter) return;
  categoryState[filter.name] = enabled ? new Set(filter.values.map(value => Number(value))) : new Set();
  renderQuickCategory();
  renderCategoryFilters();
  update();
}}

function categorySummaryText(filter) {{
  const selected = categoryState[filter.name]?.size ?? filter.values.length;
  return `${{filter.label}}: ${{selected}}/${{filter.values.length}} selected`;
}}

function renderTextFilters() {{
  const panel = el("textFilterPanel");
  const target = el("textFilters");
  const state = filterStateForPanel(activePanel);
  target.innerHTML = "";
  panel.style.display = payload.textFilters.length ? "" : "none";
  for (const filter of payload.textFilters) {{
    const label = document.createElement("label");
    label.textContent = filter.label;
    const input = document.createElement("input");
    input.type = "search";
    input.placeholder = "contains...";
    input.dataset.textFilter = filter.name;
    input.value = state.text[filter.name] || "";
    input.addEventListener("input", () => {{
      state.text[filter.name] = input.value;
      update();
    }});
    label.appendChild(input);
    target.appendChild(label);
  }}
}}

function renderActiveFilterControls() {{
  useActiveFilterState();
  renderCategoryFilters();
  renderQuickCategory();
  renderTextFilters();
  renderRangeFilters();
}}

function attachEvents() {{
  ["x2var","y2var","xAxisLabel","yAxisLabel","splitVar","sliceBins","sliceEdges","xbins","ybins","xticks","yticks","xmin","xmax","ymin","ymax","logz","density","colorScale","plotHeight","plotWidth","signalModel","backgroundModel","fitMethod","fitScanDetail","fitRangeClick"].forEach(id => {{
    el(id).addEventListener("input", () => {{ readControlsToPanel(); scheduleUpdate(); }});
  }});
  el("xvar").addEventListener("change", () => {{ setPanelVariable("x"); update(); }});
  el("yvar").addEventListener("change", () => {{ setPanelVariable("y"); update(); }});
  el("loadFiles").addEventListener("click", openRemoteLoadBrowser);
  el("loadFileInput").addEventListener("change", event => {{
    loadVisualizerFiles(event.target.files);
    event.target.value = "";
  }});
  el("closeLoadBrowser").addEventListener("click", closeRemoteLoadBrowser);
  el("loadBrowser").addEventListener("click", event => {{
    if (event.target === el("loadBrowser")) closeRemoteLoadBrowser();
  }});
  el("loadBrowserRoot").addEventListener("click", () => showRemoteDirectory(remoteRootUrl()));
  el("loadBrowserUp").addEventListener("click", goRemoteParentDirectory);
  el("loadBrowserRefresh").addEventListener("click", () => {{
    if (remoteDirectoryUrl) showRemoteDirectory(remoteDirectoryUrl);
  }});
  el("loadSelectedRemote").addEventListener("click", loadSelectedRemoteFiles);
  ["opLeft","opRight","opKind"].forEach(id => {{
    el(id).addEventListener("change", () => {{
      el("opStatus").textContent = "";
    }});
  }});
  el("addDerived").addEventListener("click", addDerivedVariable);
  el("addPanel").addEventListener("click", addPanelTab);
  el("splitView").addEventListener("click", () => {{
    const next = !compareMode;
    if (next && enabledPanels.length < panelKeys.length) addPanelTab(false);
    compareMode = next && enabledPanels.length > 1;
    renderPanelTabs();
    update();
  }});
  el("sharedPanelFilters").addEventListener("click", toggleSharedPanelFilters);
  el("toggleCanvasToolbar").addEventListener("click", toggleCanvasToolbar);
  el("mode1d").addEventListener("click", () => setMode("1d"));
  el("mode2d").addEventListener("click", () => setMode("2d"));
  el("addXVar").addEventListener("click", () => addAdditionalVariable("x"));
  el("addYVar").addEventListener("click", () => addAdditionalVariable("y"));
  el("removeXVar").addEventListener("click", () => removeAdditionalVariable("x"));
  el("removeYVar").addEventListener("click", () => removeAdditionalVariable("y"));
  el("addRange").addEventListener("click", addRangeFilter);
  ["rangeVar", "rangeMin", "rangeMax"].forEach(id => {{
    el(id).addEventListener("input", () => {{ el("constraintStatus").textContent = ""; }});
  }});
  el("resetFilters").addEventListener("click", resetFilters);
  el("resetRanges").addEventListener("click", () => {{ resetAxisRanges(currentPanel()); syncControlsFromPanel(); update(); }});
  el("plotTools").addEventListener("click", event => {{
    event.stopPropagation();
    const bounds = event.currentTarget.getBoundingClientRect();
    showCanvasContextMenu({{
      preventDefault() {{}},
      clientX: bounds.left,
      clientY: bounds.bottom + 4
    }}, activePanel);
  }});
  el("savePng").addEventListener("click", savePng);
  el("saveWorkspace").addEventListener("click", () => saveWorkspace(true));
  el("restoreWorkspace").addEventListener("click", () => {{
    if (!restoreWorkspace(true)) return;
    renderActiveFilterControls();
    syncControlsFromPanel();
    update();
  }});
  el("clearFitRange").addEventListener("click", () => {{ clearFitRange(currentPanel()); syncControlsFromPanel(); update(); }});
  el("toggleFitAnnotations").addEventListener("click", () => {{
    const panel = currentPanel();
    panel.showFitAnnotations = !(panel.showFitAnnotations !== false);
    syncFitAnnotationButton(panel);
    update();
  }});
  el("quickCategoryFilter").addEventListener("change", renderQuickCategory);
  el("quickCategoryAll").addEventListener("click", () => setCurrentCategoryValues(true));
  el("quickCategoryNone").addEventListener("click", () => setCurrentCategoryValues(false));
  el("toggleTopology").addEventListener("click", toggleTopology);
  el("makeGhost").addEventListener("click", () => {{
    captureGhost(contextMenuPanelKey);
    hideCanvasContextMenu();
  }});
  el("clearGhost").addEventListener("click", () => {{
    clearGhost(contextMenuPanelKey);
    hideCanvasContextMenu();
  }});
  el("toggleCanvasToolbarContext").addEventListener("click", () => {{
    toggleCanvasToolbar();
    hideCanvasContextMenu();
  }});
  el("toggleMeanGuides").addEventListener("click", () => {{
    const panel = panels[contextMenuPanelKey];
    panel.showMeanGuides = !Boolean(panel.showMeanGuides);
    hideCanvasContextMenu();
    update();
  }});
  el("profileX").addEventListener("click", () => {{
    launchBinProfile("x");
    hideCanvasContextMenu();
  }});
  el("profileY").addEventListener("click", () => {{
    launchBinProfile("y");
    hideCanvasContextMenu();
  }});
  el("addFunctionCurve").addEventListener("click", () => {{
    hideCanvasContextMenu();
    openReferenceCurveEditor(contextMenuPanelKey, true);
  }});
  el("manageReferenceCurves").addEventListener("click", () => {{
    hideCanvasContextMenu();
    openReferenceCurveEditor(contextMenuPanelKey, false);
  }});
  el("closeReferenceCurveEditor").addEventListener("click", hideReferenceCurveEditor);
  el("saveReferenceCurve").addEventListener("click", saveReferenceCurve);
  el("clearReferenceCurves").addEventListener("click", clearReferenceCurves);
  el("referenceCurveDirection").addEventListener("change", syncReferenceExpressionControl);
  el("referenceCurveEditor").addEventListener("click", event => {{
    if (event.target === el("referenceCurveEditor")) hideReferenceCurveEditor();
  }});
  for (const key of panelKeys) {{
    el("plot" + key).addEventListener("mousemove", event => showHoverInfo(event, key));
    el("plot" + key).addEventListener("click", event => handleCanvasClick(event, key));
    el("plot" + key).addEventListener("contextmenu", event => showCanvasContextMenu(event, key));
    el("plot" + key).addEventListener("mouseleave", () => {{
      setHoverText(key, pinnedMarkerSummary(key));
      renderPinnedMarkers(key);
      hideColorScaleMarker(key);
    }});
  }}
  document.addEventListener("click", event => {{
    if (!el("canvasContextMenu").contains(event.target)) hideCanvasContextMenu();
  }});
  document.addEventListener("keydown", event => {{
    if (event.key === "Escape") {{
      hideCanvasContextMenu();
      hideReferenceCurveEditor();
    }}
  }});
  window.addEventListener("blur", hideCanvasContextMenu);
  window.addEventListener("resize", scheduleUpdate);
}}

function setActivePanel(key) {{
  if (!enabledPanels.includes(key)) return;
  activePanel = key;
  renderActiveFilterControls();
  syncControlsFromPanel();
  update();
}}

function addPanelTab(activate = true) {{
  const next = panelKeys.find(key => !enabledPanels.includes(key));
  if (!next) return;
  enabledPanels.push(next);
  if (activate) {{
    activePanel = next;
    compareMode = false;
  }}
  renderPanelTabs();
  renderActiveFilterControls();
  syncControlsFromPanel();
  update();
}}

function toggleSharedPanelFilters() {{
  if (sharedPanelFilters) {{
    for (const key of panelKeys) copyFilterState(panels[key].filterState, sharedFilterState);
    sharedPanelFilters = false;
  }} else {{
    copyFilterState(sharedFilterState, panels[activePanel].filterState);
    sharedPanelFilters = true;
  }}
  renderActiveFilterControls();
  renderPanelTabs();
  update();
}}

function syncPanelViewButtons() {{
  const states = {{splitView: compareMode, sharedPanelFilters}};
  for (const [id, active] of Object.entries(states)) {{
    const button = el(id);
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }}
  el("sharedPanelFilters").title = sharedPanelFilters
    ? "Apply the same topology, constraints, and text filters to both panels"
    : "Each panel keeps its own topology, constraints, text filters, and axis ranges";
}}

function toggleCanvasToolbar() {{
  canvasToolbarCollapsed = !canvasToolbarCollapsed;
  update();
}}

function syncCanvasToolbarVisibility() {{
  const toolbar = el("canvasToolbar");
  const button = el("toggleCanvasToolbar");
  if (!toolbar || !button) return;
  if (!toolbar.hidden) {{
    const measuredHeight = Math.ceil(toolbar.getBoundingClientRect().height);
    if (measuredHeight > 0) canvasToolbarExpandedHeight = measuredHeight;
  }}
  toolbar.hidden = canvasToolbarCollapsed;
  const actionLabel = canvasToolbarCollapsed ? "Show plot controls" : "Hide plot controls";
  button.textContent = actionLabel;
  const contextButton = el("toggleCanvasToolbarContext");
  if (contextButton) contextButton.textContent = actionLabel;
  button.setAttribute("aria-expanded", canvasToolbarCollapsed ? "false" : "true");
  button.title = canvasToolbarCollapsed
    ? "Show axis, display, and plot-action controls"
    : "Collapse axis, display, and plot-action controls";
}}

function renderPanelTabs() {{
  const target = el("panelTabs");
  target.innerHTML = "";
  const showPanelTabs = enabledPanels.length > 1;
  target.style.display = showPanelTabs ? "flex" : "none";
  if (showPanelTabs) {{
    for (const key of enabledPanels) {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "panel-tab";
      button.textContent = panelLabels[key] || key;
      button.classList.toggle("active", key === activePanel);
      button.addEventListener("click", () => setActivePanel(key));
      target.appendChild(button);
    }}
  }}
  el("addPanel").style.display = enabledPanels.length < panelKeys.length ? "" : "none";
  syncPanelViewButtons();
}}

function syncControlsFromPanel() {{
  const panel = currentPanel();
  fillSelect(el("xvar"), panel.xvar);
  fillOverlaySelect(el("x2var"), panel.x2var);
  fillSelect(el("yvar"), panel.yvar);
  fillOverlaySelect(el("y2var"), panel.y2var);
  fillSplitSelect(panel.splitVar);
  el("sliceBins").value = panel.sliceBins || 6;
  el("sliceEdges").value = panel.sliceEdges || "";
  updateSliceControls(panel);
  el("xAxisLabel").value = panel.xLabel || "";
  el("yAxisLabel").value = panel.yLabel || "";
  el("xbins").value = panel.xbins;
  el("ybins").value = panel.ybins;
  el("xticks").value = panel.xticks;
  el("yticks").value = panel.yticks;
  el("xtickValue").textContent = fmtTickTarget(panel.xticks);
  el("ytickValue").textContent = fmtTickTarget(panel.yticks);
  el("xmin").value = panel.xmin;
  el("xmax").value = panel.xmax;
  el("ymin").value = panel.ymin;
  el("ymax").value = panel.ymax;
  el("logz").checked = panel.logz;
  el("density").checked = panel.density;
  el("colorScale").checked = panel.colorScale;
  panel.plotHeightFraction = canonicalPlotHeight(panel.plotHeightFraction);
  panel.plotWidthFraction = canonicalPlotWidth(panel.plotWidthFraction);
  el("plotHeight").value = panel.plotHeightFraction;
  el("plotHeightValue").textContent = plotHeightLabel(panel.plotHeightFraction);
  el("plotWidth").value = panel.plotWidthFraction;
  el("plotWidthValue").textContent = plotWidthLabel(panel.plotWidthFraction);
  migratePanelFitSpec(panel);
  el("signalModel").value = panel.signalModel || "none";
  el("backgroundModel").value = panel.backgroundModel || "none";
  if (panel.fitMethod === undefined) panel.fitMethod = canonicalFitMethod(panel.fitWeighting);
  el("fitMethod").value = canonicalFitMethod(panel.fitMethod);
  el("fitMethod").disabled = panel.mode !== "1d";
  el("fitMethod").title = panel.mode === "1d"
    ? "Choose a least-squares or unbinned fit method"
    : "Fit-method selection applies only to 1D histograms";
  panel.fitScanDetail = canonicalFitScanDetail(panel.fitScanDetail);
  el("fitScanDetail").value = panel.fitScanDetail;
  el("fitScanDetailValue").textContent = fitScanDetailLabel(panel.fitScanDetail);
  syncFitMethodControls(panel);
  el("fitRangeClick").checked = panel.fitRangeClick;
  syncFitAnnotationButton(panel);
  el("fitRangeSummary").textContent = fitRangeSummaryText(panel);
  renderFitSummary(panel);
  renderPanelTabs();
  el("mode1d").classList.toggle("active", panel.mode === "1d");
  el("mode2d").classList.toggle("active", panel.mode === "2d");
  el("logzChip").style.display = panel.mode === "2d" ? "" : "none";
  el("colorScaleChip").style.display = panel.mode === "2d" ? "" : "none";
  el("yAxisControl").style.display = panel.mode === "2d" ? "" : "none";
  const showExtraX = Boolean(panel.x2var);
  const showExtraY = panel.mode === "2d" && Boolean(panel.y2var);
  el("extraXControls").style.display = showExtraX ? "" : "none";
  el("addXVar").style.display = !panel.x2var ? "" : "none";
  el("extraYControls").style.display = showExtraY ? "" : "none";
  el("addYVar").style.display = panel.mode === "2d" && !panel.y2var ? "" : "none";
  el("yrange").classList.toggle("axis-range-hidden", panel.mode !== "2d");
  el("ybins").closest("label").style.display = panel.mode === "2d" ? "" : "none";
  updateFitRangePickerCursors();
}}

function addAdditionalVariable(axis) {{
  const panel = currentPanel();
  if (axis === "x") {{
    panel.x2var = panel.x2var || firstAdditionalVariable(panel.xvar);
  }} else {{
    panel.y2var = panel.y2var || firstAdditionalVariable(panel.yvar);
  }}
  syncControlsFromPanel();
  update();
}}

function removeAdditionalVariable(axis) {{
  const panel = currentPanel();
  if (axis === "x") panel.x2var = "";
  else panel.y2var = "";
  syncControlsFromPanel();
  update();
}}

function readControlsToPanel() {{
  const panel = currentPanel();
  panel.xvar = el("xvar").value;
  panel.x2var = el("x2var").value;
  panel.yvar = el("yvar").value;
  panel.y2var = el("y2var").value;
  panel.xLabel = el("xAxisLabel").value.trim();
  panel.yLabel = el("yAxisLabel").value.trim();
  panel.splitVar = el("splitVar").value;
  panel.sliceBins = clamp(Math.round(Number(el("sliceBins").value) || 6), 1, 24);
  panel.sliceEdges = el("sliceEdges").value.trim();
  updateSliceControls(panel);
  panel.xbins = clamp(Number(el("xbins").value) || 80, 5, 400);
  panel.ybins = clamp(Number(el("ybins").value) || 80, 5, 300);
  panel.xticks = clamp(Number(el("xticks").value) || 6, 1, 40);
  panel.yticks = clamp(Number(el("yticks").value) || 6, 1, 40);
  el("xtickValue").textContent = fmtTickTarget(panel.xticks);
  el("ytickValue").textContent = fmtTickTarget(panel.yticks);
  panel.xmin = parseNumber(el("xmin").value);
  panel.xmax = parseNumber(el("xmax").value);
  panel.ymin = parseNumber(el("ymin").value);
  panel.ymax = parseNumber(el("ymax").value);
  panel.logz = el("logz").checked;
  panel.density = el("density").checked;
  panel.colorScale = el("colorScale").checked;
  panel.plotHeightFraction = canonicalPlotHeight(el("plotHeight").value);
  panel.plotWidthFraction = canonicalPlotWidth(el("plotWidth").value);
  el("plotHeightValue").textContent = plotHeightLabel(panel.plotHeightFraction);
  el("plotWidthValue").textContent = plotWidthLabel(panel.plotWidthFraction);
  panel.signalModel = canonicalSignalModel(el("signalModel").value);
  panel.backgroundModel = canonicalBackgroundModel(el("backgroundModel").value);
  panel.fitMethod = canonicalFitMethod(el("fitMethod").value);
  panel.fitScanDetail = canonicalFitScanDetail(el("fitScanDetail").value);
  el("fitScanDetailValue").textContent = fitScanDetailLabel(panel.fitScanDetail);
  syncFitMethodControls(panel);
  panel.fitModel = fitSpecKey(fitSpecFromPanel(panel));
  panel.fitRangeClick = el("fitRangeClick").checked;
  el("fitRangeSummary").textContent = fitRangeSummaryText(panel);
  updateFitRangePickerCursors();
}}

function canonicalFitModel(model) {{
  if (model === "linear") return "poly1";
  if (model === "quadratic") return "poly2";
  if (model === "constant") return "poly0";
  return model || "none";
}}

function canonicalFitMethod(method) {{
  if (method === "poisson" || method === "unbinned") return method;
  return "unweighted";
}}

function fitMethodLabel(method) {{
  const value = canonicalFitMethod(method);
  if (value === "poisson") return "Poisson WLS";
  if (value === "unbinned") return "Unbinned ML";
  return "OLS";
}}

function canonicalFitScanDetail(value) {{
  return clamp(Math.round(Number(value) || 3), 1, 5);
}}

function fitScanDetailLabel(value) {{
  return ["fastest", "coarse", "balanced", "fine", "finest"][canonicalFitScanDetail(value) - 1];
}}

function syncFitMethodControls(panel) {{
  const showUnbinnedControls = panel.mode === "1d" && canonicalFitMethod(panel.fitMethod) === "unbinned";
  el("fitScanDetailControl").style.display = showUnbinnedControls ? "" : "none";
  el("fitMethodNote").style.display = showUnbinnedControls ? "" : "none";
  el("fitMethodNote").textContent = showUnbinnedControls
    ? "Background degree 0-5 uses the same selector as binned fits; unbinned PDFs use a positive Bernstein polynomial of that degree."
    : "";
}}

function syncFitAnnotationButton(panel) {{
  const button = el("toggleFitAnnotations");
  const visible = panel.showFitAnnotations !== false;
  button.textContent = visible ? "Hide canvas fit results" : "Show canvas fit results";
  button.setAttribute("aria-pressed", String(visible));
  button.title = visible
    ? "Hide fit optimization result boxes without disabling the fit"
    : "Show fit optimization result boxes on the canvas";
}}

function renderFitSummary(panel) {{
  const target = el("fitSummary");
  const rangeSummary = el("fitRangeSummary");
  const layout = target.closest(".fit-panel-layout");
  const engaged = panelHasFit(panel);
  layout?.classList.toggle("engaged", engaged);
  target.hidden = !engaged;
  rangeSummary.hidden = !engaged;
  if (!engaged) {{
    target.innerHTML = "";
    target.classList.remove("multi", "sector");
    return;
  }}
  const summary = panel?.fitSummary || "No fit";
  const entries = summary.split(/\\s+\\|\\s+/).map(value => value.trim()).filter(Boolean);
  const items = entries.length ? entries : ["No fit"];
  const multi = items.length > 1;
  const sector = multi && isProtonSectorSplit(panel?.splitVar);
  target.innerHTML = "";
  target.classList.toggle("multi", multi);
  target.classList.toggle("sector", sector);
  for (const text of items) {{
    const item = document.createElement("div");
    item.className = "fit-summary-item";
    const separator = multi ? text.indexOf(":") : -1;
    if (separator > 0 && separator < 24) {{
      const label = document.createElement("span");
      label.className = "fit-summary-label";
      label.textContent = text.slice(0, separator);
      const detail = document.createElement("span");
      detail.className = "fit-summary-detail";
      detail.textContent = text.slice(separator + 1).trim();
      item.append(label, detail);
    }} else {{
      item.classList.add("fit-summary-detail");
      item.textContent = text;
    }}
    target.appendChild(item);
  }}
}}

function fitModelInfo(model) {{
  const value = canonicalFitModel(model);
  if (value === "none") return {{kind: "none", label: "No fit"}};
  if (value === "gaussian") return {{kind: "gaussian", label: "Gaussian", parameters: 4}};
  if (value === "crystalball") return {{kind: "crystalball", label: "Crystal Ball", parameters: 6}};
  const match = String(value).match(/^poly([0-5])$/);
  if (match) {{
    const degree = Number(match[1]);
    const label = degree === 0 ? "constant" : `Polynomial degree ${{degree}}`;
    return {{kind: "polynomial", degree, label, parameters: degree + 1}};
  }}
  return {{kind: "none", label: "No fit"}};
}}

function canonicalSignalModel(model) {{
  const value = canonicalFitModel(model);
  return value === "gaussian" || value === "crystalball" ? value : "none";
}}

function canonicalBackgroundModel(model) {{
  const value = canonicalFitModel(model);
  return fitModelInfo(value).kind === "polynomial" ? value : "none";
}}

function legacyFitSpec(model) {{
  const value = canonicalFitModel(model);
  const info = fitModelInfo(value);
  if (info.kind === "gaussian" || info.kind === "crystalball") return {{signal: value, background: "none"}};
  if (info.kind === "polynomial") return {{signal: "none", background: value}};
  return {{signal: "none", background: "none"}};
}}

function fitSpecFromPanel(panel, fallbackModel = null) {{
  if (!panel) return legacyFitSpec(fallbackModel);
  migratePanelFitSpec(panel);
  return {{
    signal: canonicalSignalModel(panel.signalModel),
    background: canonicalBackgroundModel(panel.backgroundModel)
  }};
}}

function fitSpecKey(spec) {{
  return spec.signal === "none" && spec.background === "none"
    ? "none"
    : `S:${{spec.signal}}|B:${{spec.background}}`;
}}

function migratePanelFitSpec(panel) {{
  if (!panel) return;
  if (panel.signalModel === undefined || panel.backgroundModel === undefined) {{
    const legacy = legacyFitSpec(panel.fitModel);
    if (panel.signalModel === undefined) panel.signalModel = legacy.signal;
    if (panel.backgroundModel === undefined) panel.backgroundModel = legacy.background;
  }}
  panel.signalModel = canonicalSignalModel(panel.signalModel);
  panel.backgroundModel = canonicalBackgroundModel(panel.backgroundModel);
  panel.fitModel = fitSpecKey({{signal: panel.signalModel, background: panel.backgroundModel}});
}}

function panelHasFit(panel) {{
  const spec = fitSpecFromPanel(panel);
  return spec.signal !== "none" || spec.background !== "none";
}}

function fitSpecLabel(spec) {{
  const parts = [];
  if (spec.signal !== "none") parts.push(`S=${{fitModelInfo(spec.signal).label}}`);
  if (spec.background !== "none") parts.push(`B=${{fitModelInfo(spec.background).label}}`);
  return parts.length ? parts.join(" + ") : "No fit";
}}

function clearFitRange(panel) {{
  panel.fitRangeMin = NaN;
  panel.fitRangeMax = NaN;
}}

function fitRangeBounds(panel) {{
  const a = panel.fitRangeMin;
  const b = panel.fitRangeMax;
  if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return null;
  return [Math.min(a, b), Math.max(a, b)];
}}

function fitRangeSummaryText(panel) {{
  const bounds = fitRangeBounds(panel);
  if (bounds) return `Fit range: ${{fmt(bounds[0])}} to ${{fmt(bounds[1])}}`;
  if (Number.isFinite(panel.fitRangeMin)) return `Fit range start: ${{fmt(panel.fitRangeMin)}}; click a second endpoint`;
  return "Fit range: full X range";
}}

function xInFitRange(panel, x) {{
  const bounds = fitRangeBounds(panel);
  return !bounds || (x >= bounds[0] && x <= bounds[1]);
}}

function updateFitRangePickerCursors() {{
  for (const key of panelKeys) {{
    const canvas = el("plot" + key);
    if (!canvas) continue;
    const panel = panels[key];
    canvas.classList.toggle("fit-range-picker", Boolean(panel.fitRangeClick && panelHasFit(panel)));
  }}
}}

function fitClickArea(lastPlot, px, py) {{
  const areas = (lastPlot.mode === "1d-facet" || lastPlot.mode === "2d-facet")
    ? lastPlot.facets.map(facet => facet.area)
    : [lastPlot.area];
  for (const area of areas) {{
    const pw = area.width - area.left - area.right;
    const ph = area.height - area.top - area.bottom;
    if (px >= area.left && px <= area.left + pw && py >= area.top && py <= area.top + ph) return area;
  }}
  return null;
}}

function handleFitRangeClick(event, key) {{
  const panel = panels[key];
  const lastPlot = panel?.lastPlot;
  if (!panel || !lastPlot || !panel.fitRangeClick || !panelHasFit(panel)) return false;
  const rect = el("plot" + key).getBoundingClientRect();
  const px = event.clientX - rect.left;
  const py = event.clientY - rect.top;
  const area = fitClickArea(lastPlot, px, py);
  if (!area) return true;
  const pw = area.width - area.left - area.right;
  const xValue = lastPlot.xMin + (px - area.left) / pw * (lastPlot.xMax - lastPlot.xMin);
  if (!Number.isFinite(xValue)) return true;
  activePanel = key;
  if (!Number.isFinite(panel.fitRangeMin) || Number.isFinite(panel.fitRangeMax)) {{
    panel.fitRangeMin = xValue;
    panel.fitRangeMax = NaN;
  }} else {{
    panel.fitRangeMax = xValue;
    if (panel.fitRangeMin > panel.fitRangeMax) {{
      const tmp = panel.fitRangeMin;
      panel.fitRangeMin = panel.fitRangeMax;
      panel.fitRangeMax = tmp;
    }}
  }}
  renderActiveFilterControls();
  syncControlsFromPanel();
  update();
  return true;
}}

function handleCanvasClick(event, key) {{
  if (handleFitRangeClick(event, key)) return;
  togglePinnedMarker(event, key);
}}

function setMode(next) {{
  currentPanel().mode = next;
  currentPanel().profile = null;
  syncControlsFromPanel();
  update();
}}

function setPanelVariable(axis) {{
  const panel = currentPanel();
  panel.profile = null;
  const name = el(axis + "var").value;
  const variable = byName[name];
  if (axis === "x") {{
    panel.xvar = name;
    if (panel.x2var === name) panel.x2var = "";
    panel.xmin = variable ? variable.min : "";
    panel.xmax = variable ? variable.max : "";
  }} else {{
    panel.yvar = name;
    if (panel.y2var === name) panel.y2var = "";
    panel.ymin = variable ? variable.min : "";
    panel.ymax = variable ? variable.max : "";
  }}
  syncControlsFromPanel();
}}

function resetAxisRanges(panel) {{
  const xInfo = byName[panel.xvar];
  const yInfo = byName[panel.yvar];
  panel.xmin = xInfo ? xInfo.min : "";
  panel.xmax = xInfo ? xInfo.max : "";
  panel.ymin = yInfo ? yInfo.min : "";
  panel.ymax = yInfo ? yInfo.max : "";
}}

function addRangeFilter() {{
  const name = el("rangeVar").value;
  const min = parseNumber(el("rangeMin").value);
  const max = parseNumber(el("rangeMax").value);
  const status = el("constraintStatus");
  if (!Number.isFinite(min) && !Number.isFinite(max)) {{
    status.textContent = "Enter a minimum, a maximum, or both.";
    return;
  }}
  if (Number.isFinite(min) && Number.isFinite(max) && min > max) {{
    status.textContent = "Minimum cannot exceed maximum.";
    return;
  }}
  activeRanges.push({{name, min, max}});
  status.textContent = "";
  el("rangeMin").value = "";
  el("rangeMax").value = "";
  renderRangeFilters();
  update();
}}

function renderRangeFilters() {{
  const target = el("rangeFilters");
  target.innerHTML = "";
  activeRanges.forEach((filter, index) => {{
    const row = document.createElement("div");
    row.className = "filter-row";
    const name = document.createElement("div");
    name.textContent = `${{byName[filter.name]?.label || filter.name}}${{filter.profile ? " (profile bin)" : ""}}`;
    const min = document.createElement("input");
    min.type = "number";
    min.step = "any";
    min.placeholder = "no minimum";
    min.value = Number.isFinite(filter.min) ? filter.min : "";
    min.addEventListener("input", () => {{
      filter.min = parseNumber(min.value);
      if (filter.profileAxisSlice && currentPanel().profile) currentPanel().profile.min = filter.min;
      update();
    }});
    const max = document.createElement("input");
    max.type = "number";
    max.step = "any";
    max.placeholder = "no maximum";
    max.value = Number.isFinite(filter.max) ? filter.max : "";
    max.addEventListener("input", () => {{
      filter.max = parseNumber(max.value);
      if (filter.profileAxisSlice && currentPanel().profile) currentPanel().profile.max = filter.max;
      update();
    }});
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "x";
    remove.addEventListener("click", () => {{
      if (filter.profileAxisSlice) currentPanel().profile = null;
      activeRanges.splice(index, 1);
      renderRangeFilters();
      update();
    }});
    row.append(name, min, max, remove);
    target.appendChild(row);
  }});
}}

function resetFilters() {{
  const state = filterStateForPanel(activePanel);
  currentPanel().profile = null;
  for (const filter of payload.categoricalFilters) {{
    categoryState[filter.name] = new Set(filter.values.map(value => Number(value)));
  }}
  renderQuickCategory();
  renderCategoryFilters();
  state.text = {{}};
  renderTextFilters();
  activeRanges.length = 0;
  el("constraintStatus").textContent = "";
  renderRangeFilters();
  update();
}}

function parseNumber(value) {{
  const text = value === null || value === undefined ? "" : String(value).trim();
  if (!text) return NaN;
  const number = Number(text);
  return Number.isFinite(number) ? number : NaN;
}}

function valuePassesRange(value, min, max, maxExclusive = false) {{
  return Number.isFinite(value)
    && (!Number.isFinite(min) || value >= min)
    && (!Number.isFinite(max) || (maxExclusive ? value < max : value <= max));
}}

function selectedMask(state = filterStateForPanel(activePanel)) {{
  const mask = new Uint8Array(rowCount);
  mask.fill(1);
  for (const filter of payload.categoricalFilters) {{
    const allowed = state.categories[filter.name];
    const values = columns[filter.name];
    if (!allowed || !values) continue;
    for (let i = 0; i < rowCount; i++) {{
      if (mask[i] && !allowed.has(Math.round(values[i]))) mask[i] = 0;
    }}
  }}
  for (const filter of state.ranges) {{
    const values = columns[filter.name];
    if (!values) continue;
    for (let i = 0; i < rowCount; i++) {{
      const value = values[i];
      if (mask[i] && !valuePassesRange(value, filter.min, filter.max, filter.maxExclusive)) mask[i] = 0;
    }}
  }}
  for (const filter of payload.textFilters) {{
    const needle = String(state.text[filter.name] || "").trim().toLowerCase();
    if (!needle) continue;
    const values = textColumns[filter.name];
    if (!values) continue;
    for (let i = 0; i < rowCount; i++) {{
      if (mask[i] && !String(values[i]).toLowerCase().includes(needle)) mask[i] = 0;
    }}
  }}
  return mask;
}}

function activeFilterSummaries(state = filterStateForPanel(activePanel)) {{
  const summaries = [];
  for (const filter of payload.categoricalFilters) {{
    const selected = state.categories[filter.name]?.size ?? filter.values.length;
    if (selected < filter.values.length) summaries.push(`${{filter.label}} ${{selected}}/${{filter.values.length}}`);
  }}
  for (const filter of state.ranges) {{
    const label = byName[filter.name]?.label || filter.name;
    const bounds = [];
    if (Number.isFinite(filter.min)) bounds.push(`>=${{fmt(filter.min)}}`);
    if (Number.isFinite(filter.max)) bounds.push(`${{filter.maxExclusive ? "<" : "<="}}${{fmt(filter.max)}}`);
    if (bounds.length) summaries.push(`${{label}} ${{bounds.join(" ")}}`);
  }}
  for (const filter of payload.textFilters) {{
    const needle = String(state.text[filter.name] || "").trim();
    if (needle) {{
      summaries.push(`${{filter.label || filter.name}} contains "${{needle}}"`);
    }}
  }}
  return summaries;
}}

function update() {{
  readControlsToPanel();
  updateFilterBadges();
  updatePanelVisibility();
  let activeMask = null;
  for (const key of visiblePanelKeys()) {{
    clearHoverOverlay(key);
    hideColorScaleMarker(key);
    const panel = panels[key];
    if (!columns[panel.xvar]) continue;
    const mask = selectedMask(filterStateForPanel(key));
    if (key === activePanel) activeMask = mask;
    if (panel.mode === "1d") draw1d(panel, mask);
    else draw2d(panel, mask);
    renderPinnedMarkers(key);
  }}
  updateActiveStats();
  renderPreview(activeMask || selectedMask(filterStateForPanel(activePanel)));
  scheduleWorkspaceSave();
}}

function visiblePanelKeys() {{
  return compareMode ? enabledPanels : [activePanel];
}}

function updatePanelVisibility() {{
  const visible = visiblePanelKeys();
  const plotGrid = el("plotGrid");
  const comparing = visible.length > 1;
  plotGrid.classList.toggle("compare", comparing);
  plotGrid.dataset.activePanel = activePanel;
  for (const key of panelKeys) {{
    el("plotPane" + key).classList.toggle("hidden", !visible.includes(key));
  }}
  const canvasToolbar = el("canvasToolbar");
  const toolbarSlot = el("canvasToolbarSlot" + activePanel);
  if (canvasToolbar && toolbarSlot && canvasToolbar.parentElement !== toolbarSlot) {{
    toolbarSlot.appendChild(canvasToolbar);
  }}
  syncCanvasToolbarVisibility();
  syncCanvasToolbarRail(comparing, canvasToolbar);
}}

function syncCanvasToolbarRail(comparing, canvasToolbar) {{
  const visible = visiblePanelKeys();
  for (const key of panelKeys) {{
    const slot = el("canvasToolbarSlot" + key);
    if (!slot) continue;
    slot.style.minHeight = "";
    slot.classList.toggle("controls-collapsed", canvasToolbarCollapsed);
  }}
  if (!comparing) return;
  const toolbarRailHeight = !canvasToolbarCollapsed && canvasToolbar
    ? Math.max(34, Math.ceil(canvasToolbar.getBoundingClientRect().height))
    : 0;
  const headerHeights = Object.fromEntries(visible.map(key => {{
    const pane = el("plotPane" + key);
    const slot = el("canvasToolbarSlot" + key);
    return [key, slot && pane
      ? slot.getBoundingClientRect().top - pane.getBoundingClientRect().top
      : 0];
  }}));
  const tallestHeader = Math.max(0, ...Object.values(headerHeights));
  for (const key of visible) {{
    const slot = el("canvasToolbarSlot" + key);
    if (slot) slot.style.minHeight = `${{toolbarRailHeight + tallestHeader - headerHeights[key]}}px`;
  }}
  alignVisibleCanvasTops(visible);
  requestAnimationFrame(() => {{
    if (compareMode) alignVisibleCanvasTops(visiblePanelKeys());
  }});
}}

function alignVisibleCanvasTops(visible) {{
  if (visible.length < 2) return;
  const canvasTops = Object.fromEntries(visible.map(key => [key, el("plot" + key)?.getBoundingClientRect().top || 0]));
  const lowestCanvasTop = Math.max(0, ...Object.values(canvasTops));
  for (const key of visible) {{
    const slot = el("canvasToolbarSlot" + key);
    const adjustment = lowestCanvasTop - canvasTops[key];
    if (slot && adjustment > 0.5) {{
      slot.style.minHeight = `${{slot.getBoundingClientRect().height + adjustment}}px`;
    }}
  }}
}}

function profileBinAtCanvasPoint(clientX, clientY, key) {{
  const panel = panels[key];
  const lastPlot = panel?.lastPlot;
  if (!lastPlot || (lastPlot.mode !== "2d" && lastPlot.mode !== "2d-facet")) return null;
  const rect = el("plot" + key).getBoundingClientRect();
  const px = clientX - rect.left;
  const py = clientY - rect.top;
  let area = lastPlot.area;
  let facet = null;
  if (lastPlot.mode === "2d-facet") {{
    facet = lastPlot.facets.find(item => {{
      const candidate = item.area;
      const width = candidate.width - candidate.left - candidate.right;
      const height = candidate.height - candidate.top - candidate.bottom;
      return px >= candidate.left && px <= candidate.left + width
        && py >= candidate.top && py <= candidate.top + height;
    }}) || null;
    if (!facet) return null;
    area = facet.area;
  }}
  const plotWidth = area.width - area.left - area.right;
  const plotHeight = area.height - area.top - area.bottom;
  if (px < area.left || px > area.left + plotWidth || py < area.top || py > area.top + plotHeight) return null;
  const xi = clamp(Math.floor((px - area.left) / plotWidth * lastPlot.xBins), 0, lastPlot.xBins - 1);
  const yi = clamp(Math.floor((area.top + plotHeight - py) / plotHeight * lastPlot.yBins), 0, lastPlot.yBins - 1);
  return {{
    sourceKey: key,
    xName: lastPlot.xName,
    yName: lastPlot.yName,
    xMin: lastPlot.xMin,
    xMax: lastPlot.xMax,
    yMin: lastPlot.yMin,
    yMax: lastPlot.yMax,
    xBins: lastPlot.xBins,
    yBins: lastPlot.yBins,
    xi,
    yi,
    x0: lastPlot.xMin + xi / lastPlot.xBins * (lastPlot.xMax - lastPlot.xMin),
    x1: lastPlot.xMin + (xi + 1) / lastPlot.xBins * (lastPlot.xMax - lastPlot.xMin),
    y0: lastPlot.yMin + yi / lastPlot.yBins * (lastPlot.yMax - lastPlot.yMin),
    y1: lastPlot.yMin + (yi + 1) / lastPlot.yBins * (lastPlot.yMax - lastPlot.yMin),
    facet: facet ? {{
      splitName: lastPlot.splitName,
      value: facet.value,
      numericSlice: Boolean(facet.numericSlice),
      lower: facet.lower,
      upper: facet.upper,
      last: Boolean(facet.last),
      label: facet.label
    }} : null
  }};
}}

function profileMenuTitle(axis, hit) {{
  if (!hit) return "Right-click inside a 2D histogram bin";
  const sliceName = axis === "x" ? hit.yName : hit.xName;
  const minimum = axis === "x" ? hit.y0 : hit.x0;
  const maximum = axis === "x" ? hit.y1 : hit.x1;
  const finalBin = axis === "x" ? hit.yi >= hit.yBins - 1 : hit.xi >= hit.xBins - 1;
  return `Plot ${{axis.toUpperCase()}} in Panel 2 for ${{variableLabel(sliceName)}} [${{formatAxisTick(minimum)}}, ${{formatAxisTick(maximum)}}${{finalBin ? "]" : ")"}}`;
}}

function showCanvasContextMenu(event, key) {{
  event.preventDefault();
  contextMenuPanelKey = key;
  contextMenuProfileBin = profileBinAtCanvasPoint(event.clientX, event.clientY, key);
  const panel = panels[key];
  const menu = el("canvasContextMenu");
  const make = el("makeGhost");
  const clear = el("clearGhost");
  const addCurve = el("addFunctionCurve");
  const manageCurves = el("manageReferenceCurves");
  const profileX = el("profileX");
  const profileY = el("profileY");
  const meanGuides = el("toggleMeanGuides");
  el("toggleCanvasToolbarContext").textContent = canvasToolbarCollapsed
    ? "Show plot controls"
    : "Hide plot controls";
  make.textContent = panel.ghostPlot ? "Replace ghost" : "Make ghost";
  make.disabled = !panel.lastPlot;
  clear.disabled = !panel.ghostPlot;
  meanGuides.textContent = panel.showMeanGuides ? "Hide mean guides" : "Show mean guides";
  meanGuides.setAttribute("aria-checked", panel.showMeanGuides ? "true" : "false");
  meanGuides.disabled = !panel.lastPlot;
  profileX.disabled = !contextMenuProfileBin;
  profileY.disabled = !contextMenuProfileBin;
  profileX.title = profileMenuTitle("x", contextMenuProfileBin);
  profileY.title = profileMenuTitle("y", contextMenuProfileBin);
  addCurve.disabled = panel.mode !== "2d";
  addCurve.title = addCurve.disabled ? "Function curves require a 2D plot" : "";
  manageCurves.disabled = !(panel.referenceCurves || []).length;
  menu.hidden = false;
  el("plotTools").setAttribute("aria-expanded", "true");
  const padding = 6;
  menu.style.left = clamp(event.clientX, padding, Math.max(padding, window.innerWidth - menu.offsetWidth - padding)) + "px";
  menu.style.top = clamp(event.clientY, padding, Math.max(padding, window.innerHeight - menu.offsetHeight - padding)) + "px";
}}

function hideCanvasContextMenu() {{
  el("canvasContextMenu").hidden = true;
  el("plotTools").setAttribute("aria-expanded", "false");
}}

function addProfileRange(state, name, minimum, maximum, maxExclusive, profileAxisSlice = false) {{
  state.ranges.push({{
    name,
    min: minimum,
    max: maximum,
    maxExclusive: Boolean(maxExclusive),
    profile: true,
    profileAxisSlice
  }});
}}

function applyProfileFacetConstraint(state, facet) {{
  if (!facet?.splitName) return;
  if (facet.numericSlice) {{
    addProfileRange(state, facet.splitName, facet.lower, facet.upper, !facet.last);
    return;
  }}
  const value = Number(facet.value);
  if (state.categories[facet.splitName]) {{
    state.categories[facet.splitName] = state.categories[facet.splitName].has(value)
      ? new Set([value])
      : new Set();
  }} else {{
    addProfileRange(state, facet.splitName, value, value, false);
  }}
}}

function configureProfilePanel(target, source, hit, axis) {{
  const profileX = axis === "x";
  const sourceSettings = {{
    xLabel: source.xLabel,
    yLabel: source.yLabel,
    xticks: source.xticks,
    yticks: source.yticks,
    density: source.density,
    plotHeightFraction: source.plotHeightFraction,
    plotWidthFraction: source.plotWidthFraction,
    showMeanGuides: source.showMeanGuides
  }};
  const variableName = profileX ? hit.xName : hit.yName;
  const sliceName = profileX ? hit.yName : hit.xName;
  const sliceMin = profileX ? hit.y0 : hit.x0;
  const sliceMax = profileX ? hit.y1 : hit.x1;
  const sliceIndex = profileX ? hit.yi : hit.xi;
  const sliceBins = profileX ? hit.yBins : hit.xBins;
  target.mode = "1d";
  target.xvar = variableName;
  target.x2var = "";
  target.yvar = hit.yName;
  target.y2var = "";
  target.xLabel = profileX ? sourceSettings.xLabel : sourceSettings.yLabel;
  target.yLabel = "";
  target.splitVar = "";
  target.xbins = profileX ? hit.xBins : hit.yBins;
  target.xticks = profileX ? sourceSettings.xticks : sourceSettings.yticks;
  target.yticks = sourceSettings.yticks;
  target.xmin = profileX ? hit.xMin : hit.yMin;
  target.xmax = profileX ? hit.xMax : hit.yMax;
  target.density = sourceSettings.density;
  target.plotHeightFraction = canonicalPlotHeight(sourceSettings.plotHeightFraction);
  target.plotWidthFraction = canonicalPlotWidth(sourceSettings.plotWidthFraction);
  target.showMeanGuides = Boolean(sourceSettings.showMeanGuides);
  target.fitModel = "none";
  target.signalModel = "none";
  target.backgroundModel = "none";
  target.fitMethod = "unweighted";
  target.fitRangeClick = false;
  target.fitRangeMin = NaN;
  target.fitRangeMax = NaN;
  target.fitSummary = "No fit";
  target.profile = {{
    axis,
    sourceKey: hit.sourceKey,
    variableName,
    sliceName,
    min: sliceMin,
    max: sliceMax,
    maxExclusive: sliceIndex < sliceBins - 1
  }};
}}

function launchBinProfile(axis) {{
  const hit = contextMenuProfileBin;
  if (!hit || (axis !== "x" && axis !== "y")) return;
  const source = panels[hit.sourceKey];
  if (sharedPanelFilters) {{
    for (const key of panelKeys) copyFilterState(panels[key].filterState, sharedFilterState);
    sharedPanelFilters = false;
  }}
  if (hit.sourceKey !== "B") copyFilterState(panels.B.filterState, panels[hit.sourceKey].filterState);
  const targetState = panels.B.filterState;
  applyProfileFacetConstraint(targetState, hit.facet);
  const profileX = axis === "x";
  addProfileRange(
    targetState,
    profileX ? hit.yName : hit.xName,
    profileX ? hit.y0 : hit.x0,
    profileX ? hit.y1 : hit.x1,
    profileX ? hit.yi < hit.yBins - 1 : hit.xi < hit.xBins - 1,
    true
  );
  configureProfilePanel(panels.B, source, hit, axis);
  if (!enabledPanels.includes("B")) enabledPanels.push("B");
  activePanel = "B";
  compareMode = true;
  canvasToolbarCollapsed = true;
  renderActiveFilterControls();
  syncControlsFromPanel();
  update();
}}

const MATH_FUNCTIONS = {{
  sin: {{fn: Math.sin, min: 1, max: 1}}, cos: {{fn: Math.cos, min: 1, max: 1}},
  tan: {{fn: Math.tan, min: 1, max: 1}}, asin: {{fn: Math.asin, min: 1, max: 1}},
  acos: {{fn: Math.acos, min: 1, max: 1}}, atan: {{fn: Math.atan, min: 1, max: 1}},
  atan2: {{fn: Math.atan2, min: 2, max: 2}}, sqrt: {{fn: Math.sqrt, min: 1, max: 1}},
  abs: {{fn: Math.abs, min: 1, max: 1}}, exp: {{fn: Math.exp, min: 1, max: 1}},
  log: {{fn: Math.log, min: 1, max: 1}}, ln: {{fn: Math.log, min: 1, max: 1}},
  log10: {{fn: Math.log10, min: 1, max: 1}},
  floor: {{fn: Math.floor, min: 1, max: 1}}, ceil: {{fn: Math.ceil, min: 1, max: 1}},
  round: {{fn: Math.round, min: 1, max: 1}}, min: {{fn: Math.min, min: 2, max: Infinity}},
  max: {{fn: Math.max, min: 2, max: Infinity}}, pow: {{fn: Math.pow, min: 2, max: 2}}
}};

function tokenizeMathExpression(expression) {{
  const source = String(expression || "").replace(/π/gi, "pi").replace(/\\*\\*/g, "^");
  const tokens = [];
  let index = 0;
  while (index < source.length) {{
    const char = source[index];
    if (/\\s/.test(char)) {{ index++; continue; }}
    const number = source.slice(index).match(/^(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:e[+-]?\\d+)?/i);
    if (number) {{
      tokens.push({{type: "number", value: Number(number[0])}});
      index += number[0].length;
      continue;
    }}
    const name = source.slice(index).match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (name) {{
      tokens.push({{type: "name", value: name[0].toLowerCase()}});
      index += name[0].length;
      continue;
    }}
    if ("+-*/^(),".includes(char)) {{
      tokens.push({{type: char, value: char}});
      index++;
      continue;
    }}
    throw new Error(`Unexpected character "${{char}}"`);
  }}
  tokens.push({{type: "end", value: ""}});
  return tokens;
}}

function compileMathExpression(expression, variableName = "x") {{
  const tokens = tokenizeMathExpression(expression);
  let position = 0;
  const peek = () => tokens[position];
  const match = type => peek().type === type ? tokens[position++] : null;
  const expect = type => {{
    const token = match(type);
    if (!token) throw new Error(`Expected "${{type}}"`);
    return token;
  }};
  const combine = (left, right, operation) => value => operation(left(value), right(value));
  function parseExpression() {{
    let left = parseTerm();
    while (peek().type === "+" || peek().type === "-") {{
      const operator = tokens[position++].type;
      const right = parseTerm();
      left = combine(left, right, operator === "+" ? (a, b) => a + b : (a, b) => a - b);
    }}
    return left;
  }}
  function parseTerm() {{
    let left = parseUnary();
    while (peek().type === "*" || peek().type === "/") {{
      const operator = tokens[position++].type;
      const right = parseUnary();
      left = combine(left, right, operator === "*" ? (a, b) => a * b : (a, b) => a / b);
    }}
    return left;
  }}
  function parseUnary() {{
    if (match("+")) return parseUnary();
    if (match("-")) {{ const value = parseUnary(); return input => -value(input); }}
    return parsePower();
  }}
  function parsePower() {{
    const left = parsePrimary();
    if (!match("^")) return left;
    const right = parseUnary();
    return combine(left, right, Math.pow);
  }}
  function parsePrimary() {{
    const number = match("number");
    if (number) return () => number.value;
    const name = match("name");
    if (name) {{
      if (name.value === variableName) return value => value;
      if (name.value === "pi") return () => Math.PI;
      if (name.value === "e") return () => Math.E;
      const definition = MATH_FUNCTIONS[name.value];
      if (!definition) throw new Error(`Unknown name "${{name.value}}"`);
      expect("(");
      const argumentsList = [];
      if (peek().type !== ")") {{
        argumentsList.push(parseExpression());
        while (match(",")) argumentsList.push(parseExpression());
      }}
      expect(")");
      if (argumentsList.length < definition.min || argumentsList.length > definition.max) {{
        throw new Error(`${{name.value}} received ${{argumentsList.length}} argument(s)`);
      }}
      return value => definition.fn(...argumentsList.map(argument => argument(value)));
    }}
    if (match("(")) {{
      const value = parseExpression();
      expect(")");
      return value;
    }}
    throw new Error("Expected a number, variable, function, or parenthesized expression");
  }}
  const evaluate = parseExpression();
  if (peek().type !== "end") throw new Error(`Unexpected token "${{peek().value}}"`);
  return value => {{
    const result = evaluate(value);
    return Number.isFinite(result) ? result : NaN;
  }};
}}

function parseOptionalNumber(value) {{
  return String(value).trim() === "" ? NaN : Number(value);
}}

function syncReferenceExpressionControl() {{
  const variable = el("referenceCurveDirection").value === "x-of-y" ? "y" : "x";
  el("referenceExpressionLabel").textContent = `f(${{variable}})`;
  el("referenceCurveExpression").placeholder = variable;
}}

function automaticReferenceCurveLabel(direction, expression) {{
  const dependent = direction === "x-of-y" ? "x" : "y";
  return truncateText(`${{dependent}} = ${{expression}}`, 48);
}}

function openReferenceCurveEditor(key, focusValue = false) {{
  contextMenuPanelKey = key;
  const panel = panels[key];
  el("referenceCurveAxes").textContent = panel.mode === "2d"
    ? `${{byName[panel.yvar]?.label || panel.yvar}} versus ${{byName[panel.xvar]?.label || panel.xvar}}`
    : "Function curves require a two-dimensional plot.";
  el("referenceCurveStatus").textContent = "";
  el("saveReferenceCurve").disabled = panel.mode !== "2d";
  syncReferenceExpressionControl();
  renderReferenceCurveList(panel);
  el("referenceCurveEditor").hidden = false;
  if (focusValue) {{
    el("referenceCurveExpression").focus();
    el("referenceCurveExpression").select();
  }} else {{
    el("closeReferenceCurveEditor").focus();
  }}
}}

function hideReferenceCurveEditor() {{
  const editor = el("referenceCurveEditor");
  if (!editor || editor.hidden) return;
  editor.hidden = true;
}}

function saveReferenceCurve() {{
  const panel = panels[contextMenuPanelKey];
  const direction = el("referenceCurveDirection").value;
  const variable = direction === "x-of-y" ? "y" : "x";
  const expression = el("referenceCurveExpression").value.trim();
  const domainMin = parseOptionalNumber(el("referenceDomainMin").value);
  const domainMax = parseOptionalNumber(el("referenceDomainMax").value);
  const lineStyle = el("referenceLineStyle").value;
  const lineWidth = Number(el("referenceLineWidth").value);
  if (!expression) {{
    el("referenceCurveStatus").textContent = "Enter a function expression.";
    return;
  }}
  try {{ compileMathExpression(expression, variable); }}
  catch (error) {{ el("referenceCurveStatus").textContent = error.message; return; }}
  if ((Number.isFinite(domainMin) && Number.isFinite(domainMax)) && domainMax <= domainMin) {{
    el("referenceCurveStatus").textContent = "The domain maximum must exceed the minimum.";
    return;
  }}
  if (!(lineWidth >= 0.5 && lineWidth <= 3)) {{
    el("referenceCurveStatus").textContent = "Line width must be between 0.5 and 3 pixels.";
    return;
  }}
  if (!Array.isArray(panel.referenceCurves)) panel.referenceCurves = [];
  panel.referenceCurves.push({{
    id: ++referenceCurveId,
    kind: "function",
    direction,
    expression,
    domainMin,
    domainMax,
    lineStyle,
    lineWidth,
    xName: panel.xvar,
    yName: panel.yvar,
    label: el("referenceCurveLabel").value.trim() || automaticReferenceCurveLabel(direction, expression)
  }});
  el("referenceCurveLabel").value = "";
  el("referenceCurveStatus").textContent = "Added function curve.";
  renderReferenceCurveList(panel);
  update();
}}

function removeReferenceCurve(id) {{
  const panel = panels[contextMenuPanelKey];
  panel.referenceCurves = (panel.referenceCurves || []).filter(curve => curve.id !== id);
  renderReferenceCurveList(panel);
  update();
}}

function clearReferenceCurves() {{
  const panel = panels[contextMenuPanelKey];
  panel.referenceCurves = [];
  el("referenceCurveStatus").textContent = "Cleared reference curves for this panel.";
  renderReferenceCurveList(panel);
  update();
}}

function renderReferenceCurveList(panel) {{
  const target = el("referenceCurveList");
  target.innerHTML = "";
  const curves = panel.referenceCurves || [];
  el("clearReferenceCurves").disabled = !curves.length;
  for (const curve of curves) {{
    const row = document.createElement("div");
    row.className = "reference-curve-item";
    const text = document.createElement("span");
    const relation = curve.direction === "x-of-y" ? "x=f(y)" : "y=f(x)";
    const axes = curve.xName && curve.yName
      ? `${{byName[curve.yName]?.label || curve.yName}} vs ${{byName[curve.xName]?.label || curve.xName}}`
      : "current axes";
    const style = `${{curve.lineStyle || "solid"}}, ${{formatAxisTick(curve.lineWidth || 1.25)}} px`;
    text.textContent = `${{curve.label}} · ${{relation}} · ${{style}} · ${{axes}}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => removeReferenceCurve(curve.id));
    row.append(text, remove);
    target.appendChild(row);
  }}
  if (!curves.length) {{
    const empty = document.createElement("div");
    empty.className = "subtle";
    empty.textContent = "No reference curves on this panel.";
    target.appendChild(empty);
  }}
}}

function copyGhostValues(values) {{
  return values ? Float64Array.from(values) : null;
}}

function captureGhost(key) {{
  const panel = panels[key];
  const source = panel?.lastPlot;
  if (!source) return;
  panel.ghostPlot = {{
    mode: source.mode,
    xName: source.xName || "",
    yName: source.yName || "",
    splitName: source.splitName || "",
    splitSignature: source.splitSignature || "",
    density: Boolean(source.density),
    logz: Boolean(source.logz),
    xMin: source.xMin,
    xMax: source.xMax,
    yMin: source.yMin,
    yMax: source.yMax,
    bins: source.bins,
    xBins: source.xBins,
    yBins: source.yBins,
    selected: source.selected || 0,
    counts: copyGhostValues(source.counts),
    facets: source.facets ? source.facets.map(facet => ({{
      value: facet.value,
      label: facet.label,
      counts: copyGhostValues(facet.counts),
      selected: facet.selected || 0
    }})) : null
  }};
  update();
}}

function clearGhost(key) {{
  if (!panels[key]) return;
  panels[key].ghostPlot = null;
  update();
}}

function compatibleGhost(panel, mode, fields) {{
  const ghost = panel?.ghostPlot;
  if (!ghost || ghost.mode !== mode || ghost.density !== Boolean(panel.density)) return null;
  for (const [name, value] of Object.entries(fields)) {{
    if (String(ghost[name] || "") !== String(value || "")) return null;
  }}
  return ghost;
}}

function ghostFacet(ghost, value) {{
  return ghost?.facets?.find(facet => Number(facet.value) === Number(value)) || null;
}}

function drawGhostLegend(ctx, area, selected) {{
  const c = colors();
  const pw = area.width - area.left - area.right;
  const x = area.left + pw - 108;
  const y = area.top + 10;
  ctx.save();
  ctx.fillStyle = c.bg;
  ctx.globalAlpha = 0.86;
  ctx.fillRect(x - 5, y - 6, 112, 19);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = c.ghost;
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  ctx.moveTo(x, y + 3);
  ctx.lineTo(x + 21, y + 3);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = c.fg;
  ctx.font = "10px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(selected ? `ghost (${{Number(selected).toLocaleString()}})` : "ghost", x + 27, y + 3);
  ctx.restore();
}}

function drawGhost1d(ctx, area, ghost, counts, xMin, xMax, yMax, showLegend = true) {{
  if (!ghost || !counts || !counts.length || !(yMax > 0)) return;
  const c = colors();
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  const bins = counts.length;
  ctx.save();
  ctx.beginPath();
  ctx.rect(area.left, area.top, pw, ph);
  ctx.clip();
  ctx.strokeStyle = c.ghost;
  ctx.globalAlpha = 0.9;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.setLineDash([6, 3]);
  ctx.beginPath();
  for (let index = 0; index < bins; index++) {{
    const value = counts[index];
    const gx0 = ghost.xMin + index / bins * (ghost.xMax - ghost.xMin);
    const gx1 = ghost.xMin + (index + 1) / bins * (ghost.xMax - ghost.xMin);
    const x0 = area.left + (gx0 - xMin) / (xMax - xMin) * pw;
    const x1 = area.left + (gx1 - xMin) / (xMax - xMin) * pw;
    const y = area.top + ph - value / yMax * ph;
    if (index === 0) ctx.moveTo(x0, y);
    else ctx.lineTo(x0, y);
    ctx.lineTo(x1, y);
  }}
  ctx.stroke();
  ctx.restore();
  if (showLegend) drawGhostLegend(ctx, area, ghost.selected);
}}

function drawGhost2d(ctx, area, ghost, counts, xMin, xMax, yMin, yMax, showLegend = true) {{
  if (!ghost || !counts || !counts.length) return;
  const c = colors();
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  const xBins = ghost.xBins;
  const yBins = ghost.yBins;
  const maximum = histogramMax(counts);
  ctx.save();
  ctx.beginPath();
  ctx.rect(area.left, area.top, pw, ph);
  ctx.clip();
  ctx.strokeStyle = c.ghost;
  ctx.lineWidth = 1.2;
  for (let yi = 0; yi < yBins; yi++) {{
    for (let xi = 0; xi < xBins; xi++) {{
      const count = counts[yi * xBins + xi];
      if (!(count > 0)) continue;
      const fraction = ghost.logz ? Math.log1p(count) / Math.log1p(maximum) : count / maximum;
      const gx0 = ghost.xMin + xi / xBins * (ghost.xMax - ghost.xMin);
      const gx1 = ghost.xMin + (xi + 1) / xBins * (ghost.xMax - ghost.xMin);
      const gy0 = ghost.yMin + yi / yBins * (ghost.yMax - ghost.yMin);
      const gy1 = ghost.yMin + (yi + 1) / yBins * (ghost.yMax - ghost.yMin);
      const x0 = area.left + (gx0 - xMin) / (xMax - xMin) * pw;
      const x1 = area.left + (gx1 - xMin) / (xMax - xMin) * pw;
      const py0 = area.top + ph - (gy1 - yMin) / (yMax - yMin) * ph;
      const py1 = area.top + ph - (gy0 - yMin) / (yMax - yMin) * ph;
      ctx.globalAlpha = 0.18 + 0.72 * fraction;
      ctx.strokeRect(x0 + 0.5, py0 + 0.5, Math.max(0, x1 - x0 - 1), Math.max(0, py1 - py0 - 1));
    }}
  }}
  ctx.restore();
  if (showLegend) drawGhostLegend(ctx, area, ghost.selected);
}}

function plotArea(canvas, showColorScale = false) {{
  const colorScaleSlots = typeof showColorScale === "number" ? showColorScale : showColorScale ? 1 : 0;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(240, Math.floor(rect.width * dpr));
  canvas.height = Math.max(180, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  const right = colorScaleSlots > 1 ? 112 : colorScaleSlots > 0 ? 82 : 22;
  return {{ctx, width, height, left: 76, right, top: 18, bottom: 62}};
}}

function preparePanelCanvas(panel) {{
  const canvas = el("plot" + panel.key);
  canvas.style.width = `${{canonicalPlotWidth(panel.plotWidthFraction) * 100}}%`;
  const width = canvas.getBoundingClientRect().width;
  const heightScale = canonicalPlotHeight(panel.plotHeightFraction) * MAX_PLOT_HEIGHT_TO_WIDTH;
  const reclaimedHeight = canvasToolbarCollapsed ? canvasToolbarExpandedHeight : 0;
  canvas.style.minHeight = "0";
  canvas.style.height = `${{clamp(width * heightScale + reclaimedHeight, 160, 2000)}}px`;
}}

function colors() {{
  const style = getComputedStyle(document.documentElement);
  return {{
    fg: style.getPropertyValue("--fg").trim(),
    muted: style.getPropertyValue("--muted").trim(),
    border: style.getPropertyValue("--border").trim(),
    mark: style.getPropertyValue("--mark").trim(),
    alert: style.getPropertyValue("--filter-alert").trim(),
    bg: style.getPropertyValue("--bg").trim(),
    ghost: style.getPropertyValue("--ghost").trim(),
    reference: style.getPropertyValue("--reference").trim()
  }};
}}

function referenceLineDash(style) {{
  if (style === "dashed") return [7, 4];
  if (style === "dotted") return [1.25, 3];
  if (style === "dash-dot") return [8, 3, 1.25, 3];
  return [];
}}

function referenceLineWidth(curve) {{
  const width = Number(curve?.lineWidth);
  return Number.isFinite(width) ? clamp(width, 0.5, 3) : 1.25;
}}

function drawReferenceCurves(ctx, area, panel, xMin, xMax, yMin, yMax, showLegend = true) {{
  const curves = (panel.referenceCurves || []).filter(curve =>
    curve.kind === "function" && curve.xName === panel.xvar && curve.yName === panel.yvar
  );
  if (panel.mode !== "2d" || !curves.length || !(xMax > xMin) || !(yMax > yMin)) return 0;
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  const c = colors();
  const drawn = [];
  ctx.save();
  ctx.beginPath();
  ctx.rect(area.left, area.top, pw, ph);
  ctx.clip();
  for (let curveIndex = 0; curveIndex < curves.length; curveIndex++) {{
    const curve = curves[curveIndex];
    const xOfY = curve.direction === "x-of-y";
    const independentMin = xOfY ? yMin : xMin;
    const independentMax = xOfY ? yMax : xMax;
    const dependentMin = xOfY ? xMin : yMin;
    const dependentMax = xOfY ? xMax : yMax;
    const domainMin = Number.isFinite(curve.domainMin) ? Math.max(independentMin, curve.domainMin) : independentMin;
    const domainMax = Number.isFinite(curve.domainMax) ? Math.min(independentMax, curve.domainMax) : independentMax;
    if (!(domainMax > domainMin)) continue;
    let evaluate;
    try {{ evaluate = compileMathExpression(curve.expression, xOfY ? "y" : "x"); }}
    catch (_) {{ continue; }}
    const points = [];
    const samples = 520;
    const dependentPadding = Math.max(1.0e-12, dependentMax - dependentMin) * 0.08;
    for (let index = 0; index <= samples; index++) {{
      const independent = domainMin + (domainMax - domainMin) * index / samples;
      const dependent = evaluate(independent);
      if (!Number.isFinite(dependent) || dependent < dependentMin - dependentPadding || dependent > dependentMax + dependentPadding) {{
        points.push(null);
        continue;
      }}
      const xValue = xOfY ? dependent : independent;
      const yValue = xOfY ? independent : dependent;
      points.push({{
        x: area.left + (xValue - xMin) / (xMax - xMin) * pw,
        y: area.top + ph - (yValue - yMin) / (yMax - yMin) * ph
      }});
    }}
    if (!points.some(Boolean)) continue;
    const drawPath = () => {{
      ctx.beginPath();
      let previous = null;
      for (const point of points) {{
        if (!point) {{ previous = null; continue; }}
        const discontinuity = previous && (Math.abs(point.x - previous.x) > pw * 0.45 || Math.abs(point.y - previous.y) > ph * 0.45);
        if (!previous || discontinuity) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
        previous = point;
      }}
      ctx.stroke();
    }};
    ctx.setLineDash(referenceLineDash(curve.lineStyle));
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = c.reference;
    ctx.lineWidth = referenceLineWidth(curve);
    drawPath();
    drawn.push(curve);
  }}
  ctx.restore();
  if (showLegend && drawn.length) drawReferenceCurveLegend(ctx, area, drawn);
  return drawn.length;
}}

function drawReferenceCurveLegend(ctx, area, curves) {{
  const c = colors();
  ctx.save();
  ctx.font = "12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const shown = curves.slice(0, 4);
  const lineHeight = 17;
  const longest = Math.max(...shown.map(curve => ctx.measureText(curve.label).width));
  const boxWidth = Math.min(area.width - area.left - area.right - 10, longest + 38);
  const boxHeight = shown.length * lineHeight + 8;
  const x = area.width - area.right - 5;
  const y = area.top + (area.height - area.top - area.bottom) - boxHeight - 5;
  ctx.globalAlpha = 0.84;
  ctx.fillStyle = c.bg;
  ctx.fillRect(x - boxWidth, y, boxWidth, boxHeight);
  ctx.globalAlpha = 1;
  for (let index = 0; index < shown.length; index++) {{
    const curve = shown[index];
    const cy = y + 4 + lineHeight * (index + 0.5);
    ctx.strokeStyle = c.reference;
    ctx.lineWidth = referenceLineWidth(curve);
    ctx.lineCap = "round";
    ctx.setLineDash(referenceLineDash(curve.lineStyle));
    ctx.beginPath();
    ctx.moveTo(x - boxWidth + 7, cy);
    ctx.lineTo(x - boxWidth + 27, cy);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = c.fg;
    ctx.fillText(curve.label, x - 6, cy);
  }}
  ctx.restore();
}}

function filterBadgeText(state = filterStateForPanel(activePanel)) {{
  const summaries = activeFilterSummaries(state);
  if (!summaries.length) return null;
  const detail = summaries.slice(0, 2).join("; ") + (summaries.length > 2 ? `; +${{summaries.length - 2}} more` : "");
  return {{count: summaries.length, detail: truncateText(detail, 96)}};
}}

function updateFilterBadges() {{
  for (const key of panelKeys) {{
    const node = el("filterBadge" + key);
    if (!node) continue;
    const badge = filterBadgeText(filterStateForPanel(key));
    if (!badge) {{
      node.style.display = "none";
      continue;
    }}
    node.style.display = "flex";
    node.querySelector("strong").textContent = `Filters: ${{badge.count}} active`;
    node.querySelector("span").textContent = badge.detail;
  }}
}}

function truncateText(text, maxChars) {{
  return text.length <= maxChars ? text : text.slice(0, Math.max(0, maxChars - 3)) + "...";
}}

function formatAxisTick(value) {{
  if (!Number.isFinite(value)) return "";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1.0e6 || magnitude < 1.0e-4) {{
    return value.toExponential(3).replace(/\\.0+(?=e)/, "").replace(/(\\.\\d*?)0+(?=e)/, "$1");
  }}
  return String(Number(value.toPrecision(6)));
}}

function drawAxes(ctx, area, xMin, xMax, yMin, yMax, xLabel, yLabel, xTickCount, yTickCount, options = null) {{
  const visibility = options || {{}};
  const showXTickLabels = visibility.showXTickLabels !== false;
  const showYTickLabels = visibility.showYTickLabels !== false;
  const showXLabel = visibility.showXLabel !== false;
  const showYLabel = visibility.showYLabel !== false;
  const c = colors();
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  ctx.lineWidth = 1;
  ctx.font = "12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

  ctx.strokeStyle = c.border;
  ctx.fillStyle = c.muted;
  ctx.textBaseline = "middle";
  for (const tick of niceTicks(xMin, xMax, xTickCount)) {{
    const x = area.left + (tick - xMin) / (xMax - xMin) * pw;
    ctx.beginPath();
    ctx.moveTo(x, area.top);
    ctx.lineTo(x, area.top + ph + 5);
    ctx.stroke();
    if (showXTickLabels) {{
      ctx.textAlign = "center";
      ctx.fillText(formatAxisTick(tick), x, area.top + ph + 20);
    }}
  }}
  for (const tick of niceTicks(yMin, yMax, yTickCount)) {{
    const y = area.top + ph - (tick - yMin) / (yMax - yMin) * ph;
    ctx.beginPath();
    ctx.moveTo(area.left - 5, y);
    ctx.lineTo(area.left + pw, y);
    ctx.stroke();
    if (showYTickLabels) {{
      ctx.textAlign = "right";
      ctx.fillText(formatAxisTick(tick), area.left - 8, y);
    }}
  }}

  ctx.strokeStyle = c.fg;
  ctx.beginPath();
  ctx.moveTo(area.left, area.top);
  ctx.lineTo(area.left, area.top + ph);
  ctx.lineTo(area.left + pw, area.top + ph);
  ctx.stroke();
  ctx.fillStyle = c.muted;
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "center";
  if (showXLabel) ctx.fillText(xLabel, area.left + pw / 2, area.top + ph + 38);
  if (showYLabel) {{
    ctx.save();
    ctx.translate(area.left - 40, area.top + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }}
}}

function drawMeanGuides(ctx, area, panel, meanX, meanY, xMin, xMax, yMin, yMax) {{
  if (!panel?.showMeanGuides) return;
  const drawX = Number.isFinite(meanX) && xMax > xMin && meanX >= xMin && meanX <= xMax;
  const drawY = panel.mode === "2d" && Number.isFinite(meanY) && yMax > yMin && meanY >= yMin && meanY <= yMax;
  if (!drawX && !drawY) return;
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  const c = colors();
  ctx.save();
  ctx.beginPath();
  ctx.rect(area.left, area.top, pw, ph);
  ctx.clip();
  ctx.strokeStyle = c.fg;
  ctx.globalAlpha = 0.76;
  ctx.lineWidth = 1.25;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  if (drawX) {{
    const x = area.left + (meanX - xMin) / (xMax - xMin) * pw;
    ctx.moveTo(x, area.top);
    ctx.lineTo(x, area.top + ph);
  }}
  if (drawY) {{
    const y = area.top + ph - (meanY - yMin) / (yMax - yMin) * ph;
    ctx.moveTo(area.left, y);
    ctx.lineTo(area.left + pw, y);
  }}
  ctx.stroke();
  ctx.restore();
}}

function poissonBinError(value, total, density) {{
  if (!(value >= 0)) return 0;
  if (!density) return Math.sqrt(value);
  if (!(total > 0)) return 0;
  return Math.sqrt(Math.max(0, value * total)) / total;
}}

function histogramPointScaleMax(series, ghostCounts = null) {{
  let maximum = 0;
  for (const item of series) {{
    if (!item?.values) continue;
    for (let index = 0; index < item.values.length; index++) {{
      const value = item.values[index];
      if (!Number.isFinite(value) || value < 0) continue;
      maximum = Math.max(maximum, value + poissonBinError(value, item.total, item.density));
    }}
  }}
  if (ghostCounts) {{
    for (let index = 0; index < ghostCounts.length; index++) {{
      const value = ghostCounts[index];
      if (Number.isFinite(value)) maximum = Math.max(maximum, value);
    }}
  }}
  return maximum > 0 ? maximum * 1.06 : 1;
}}

function draw1dPoints(ctx, area, values, yMax, total, density, color, alpha = 1) {{
  if (!values || !values.length || !(yMax > 0)) return;
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  const binWidth = pw / values.length;
  const capHalfWidth = Math.min(4, Math.max(1.5, binWidth * 0.22));
  ctx.save();
  ctx.beginPath();
  ctx.rect(area.left, area.top, pw, ph + 3);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 1.2;
  for (let index = 0; index < values.length; index++) {{
    const value = values[index];
    if (!Number.isFinite(value) || value < 0) continue;
    const error = poissonBinError(value, total, density);
    const x = area.left + (index + 0.5) * binWidth;
    const y = area.top + ph - value / yMax * ph;
    if (error > 0) {{
      const yHigh = area.top + ph - (value + error) / yMax * ph;
      const yLow = area.top + ph - Math.max(0, value - error) / yMax * ph;
      ctx.beginPath();
      ctx.moveTo(x, yHigh);
      ctx.lineTo(x, yLow);
      ctx.moveTo(x - capHalfWidth, yHigh);
      ctx.lineTo(x + capHalfWidth, yHigh);
      ctx.moveTo(x - capHalfWidth, yLow);
      ctx.lineTo(x + capHalfWidth, yLow);
      ctx.stroke();
    }}
    ctx.beginPath();
    ctx.arc(x, y, 2.6, 0, 2 * Math.PI);
    ctx.fill();
  }}
  ctx.restore();
}}

function draw1d(panel, mask) {{
  const splitName = panel.splitVar;
  if (splitName) {{
    draw1dFacets(panel, mask, splitName);
    return;
  }}
  preparePanelCanvas(panel);
  const xName = panel.xvar;
  const x2Name = panel.x2var && panel.x2var !== xName && columns[panel.x2var] ? panel.x2var : "";
  const x = columns[xName];
  const x2 = x2Name ? columns[x2Name] : null;
  const bins = panel.xbins;
  const xMin = panel.xmin;
  const xMax = panel.xmax;
  const counts = new Float64Array(bins);
  const overlayCounts = x2 ? new Float64Array(bins) : null;
  const fitValues = canonicalFitMethod(panel.fitMethod) === "unbinned" && panelHasFit(panel) ? [] : null;
  let selected = 0, overlaySelected = 0, sumX = 0;
  for (let i = 0; i < rowCount; i++) {{
    const xv = x[i];
    if (!mask[i] || xMax <= xMin) continue;
    if (Number.isFinite(xv) && xv >= xMin && xv <= xMax) {{
      const bin = Math.min(bins - 1, Math.max(0, Math.floor((xv - xMin) / (xMax - xMin) * bins)));
      counts[bin] += 1;
      if (fitValues) fitValues.push(xv);
      selected++;
      sumX += xv;
    }}
    if (x2) {{
      const x2v = x2[i];
      if (Number.isFinite(x2v) && x2v >= xMin && x2v <= xMax) {{
        const bin = Math.min(bins - 1, Math.max(0, Math.floor((x2v - xMin) / (xMax - xMin) * bins)));
        overlayCounts[bin] += 1;
        overlaySelected++;
      }}
    }}
  }}
  if (panel.density) {{
    normalizeHistogram(counts, selected);
    normalizeHistogram(overlayCounts, overlaySelected);
  }}
  const ghost = compatibleGhost(panel, "1d", {{xName}});
  const maxCount = histogramPointScaleMax([
    {{values: counts, total: selected, density: panel.density}},
    {{values: overlayCounts, total: overlaySelected, density: panel.density}}
  ], ghost?.counts);
  const canvas = el("plot" + panel.key);
  const area = plotArea(canvas);
  const {{ctx, width, height}} = area;
  const c = colors();
  ctx.clearRect(0, 0, width, height);
  drawAxes(ctx, area, xMin, xMax, 0, maxCount, axisDisplayLabel(panel, "x", byName[xName].label), axisDisplayLabel(panel, "y", panel.density ? "density" : "counts"), panel.xticks, panel.yticks);
  drawMeanGuides(ctx, area, panel, sumX / selected, NaN, xMin, xMax, 0, maxCount);
  draw1dPoints(ctx, area, counts, maxCount, selected, panel.density, c.mark);
  if (overlayCounts) {{
    draw1dPoints(ctx, area, overlayCounts, maxCount, overlaySelected, panel.density, overlayHeatColor(0.82), 0.78);
  }}
  if (x2Name) draw1dOverlayLegend(ctx, area, byName[xName].label, byName[x2Name].label);
  if (ghost) drawGhost1d(ctx, area, ghost, ghost.counts, xMin, xMax, maxCount);
  drawFitRangeIndicator(ctx, area, panel, xMin, xMax);
  panel.fitSummary = draw1dFit(ctx, area, panel, counts, xMin, xMax, 0, maxCount, fitValues);
  panel.lastPlot = {{
    mode: "1d", area, xName, x2Name, xMin, xMax, bins, counts, overlayCounts,
    selected, overlaySelected, density: panel.density, yMax: maxCount
  }};
  setPanelStats(panel, selected, sumX / selected, NaN);
}}

function draw2d(panel, mask) {{
  const splitName = panel.splitVar;
  if (splitName) {{
    draw2dFacets(panel, mask, splitName);
    return;
  }}
  preparePanelCanvas(panel);
  const xName = panel.xvar;
  const yName = panel.yvar;
  const x2Name = panel.x2var && panel.x2var !== xName && columns[panel.x2var] ? panel.x2var : "";
  const y2Name = panel.y2var && panel.y2var !== yName && columns[panel.y2var] ? panel.y2var : "";
  const x = columns[xName];
  const y = columns[yName];
  const x2 = x2Name ? columns[x2Name] : null;
  const y2 = y2Name ? columns[y2Name] : null;
  const xBins = panel.xbins;
  const yBins = panel.ybins;
  const xMin = panel.xmin;
  const xMax = panel.xmax;
  const yMin = panel.ymin;
  const yMax = panel.ymax;
  const counts = new Float64Array(xBins * yBins);
  const overlayCounts = (x2 || y2) ? new Float64Array(xBins * yBins) : null;
  let selected = 0, overlaySelected = 0, sumX = 0, sumY = 0;
  for (let i = 0; i < rowCount; i++) {{
    const xv = x[i], yv = y[i];
    if (!mask[i] || xMax <= xMin || yMax <= yMin) continue;
    if (Number.isFinite(xv) && xv >= xMin && xv <= xMax && Number.isFinite(yv) && yv >= yMin && yv <= yMax) {{
      const xi = Math.min(xBins - 1, Math.max(0, Math.floor((xv - xMin) / (xMax - xMin) * xBins)));
      const yi = Math.min(yBins - 1, Math.max(0, Math.floor((yv - yMin) / (yMax - yMin) * yBins)));
      counts[yi * xBins + xi] += 1;
      selected++;
      sumX += xv;
      sumY += yv;
    }}
    if (overlayCounts) {{
      const x2v = x2 ? x2[i] : xv;
      const y2v = y2 ? y2[i] : yv;
      if (Number.isFinite(x2v) && x2v >= xMin && x2v <= xMax && Number.isFinite(y2v) && y2v >= yMin && y2v <= yMax) {{
        const xi = Math.min(xBins - 1, Math.max(0, Math.floor((x2v - xMin) / (xMax - xMin) * xBins)));
        const yi = Math.min(yBins - 1, Math.max(0, Math.floor((y2v - yMin) / (yMax - yMin) * yBins)));
        overlayCounts[yi * xBins + xi] += 1;
        overlaySelected++;
      }}
    }}
  }}
  if (panel.density) {{
    normalizeHistogram(counts, selected);
    normalizeHistogram(overlayCounts, overlaySelected);
  }}
  const ghost = compatibleGhost(panel, "2d", {{xName, yName}});
  const maxCount = histogramMax(counts);
  const overlayMaxCount = overlayCounts ? histogramMax(overlayCounts) : 1;
  const canvas = el("plot" + panel.key);
  const area = plotArea(canvas, panel.colorScale ? (overlayCounts ? 2 : 1) : 0);
  const {{ctx, width, height, left, right, top, bottom}} = area;
  ctx.clearRect(0, 0, width, height);
  const pw = width - left - right;
  const ph = height - top - bottom;
  for (let yi = 0; yi < yBins; yi++) {{
    for (let xi = 0; xi < xBins; xi++) {{
      const count = counts[yi * xBins + xi];
      if (count <= 0) continue;
      const fraction = panel.logz ? Math.log1p(count) / Math.log1p(maxCount) : count / maxCount;
      ctx.fillStyle = heatColor(fraction);
      const x0 = left + xi / xBins * pw;
      const x1 = left + (xi + 1) / xBins * pw;
      const y0 = top + ph - (yi + 1) / yBins * ph;
      const y1 = top + ph - yi / yBins * ph;
      ctx.fillRect(x0, y0, Math.ceil(x1 - x0), Math.ceil(y1 - y0));
    }}
  }}
  if (overlayCounts) {{
    ctx.save();
    ctx.globalAlpha = 0.58;
    for (let yi = 0; yi < yBins; yi++) {{
      for (let xi = 0; xi < xBins; xi++) {{
        const count = overlayCounts[yi * xBins + xi];
        if (count <= 0) continue;
        const fraction = panel.logz ? Math.log1p(count) / Math.log1p(overlayMaxCount) : count / overlayMaxCount;
        ctx.fillStyle = overlayHeatColor(fraction);
        const x0 = left + xi / xBins * pw;
        const x1 = left + (xi + 1) / xBins * pw;
        const y0 = top + ph - (yi + 1) / yBins * ph;
        const y1 = top + ph - yi / yBins * ph;
        ctx.fillRect(x0, y0, Math.ceil(x1 - x0), Math.ceil(y1 - y0));
      }}
    }}
    ctx.restore();
  }}
  const xAxisLabel = x2Name ? `${{byName[xName].label}} / ${{byName[x2Name].label}}` : byName[xName].label;
  const yAxisLabel = y2Name ? `${{byName[yName].label}} / ${{byName[y2Name].label}}` : byName[yName].label;
  drawAxes(ctx, area, xMin, xMax, yMin, yMax, axisDisplayLabel(panel, "x", xAxisLabel), axisDisplayLabel(panel, "y", yAxisLabel), panel.xticks, panel.yticks);
  drawMeanGuides(ctx, area, panel, sumX / selected, sumY / selected, xMin, xMax, yMin, yMax);
  if (x2Name || y2Name) drawOverlayLegend(ctx, area, `${{byName[yName].label}} vs ${{byName[xName].label}}`, overlay2dLabel({{xName, x2Name, yName, y2Name}}));
  const colorScale = panel.colorScale ? draw2dColorScale(ctx, area, maxCount, overlayCounts ? overlayMaxCount : 0, panel) : null;
  if (ghost) drawGhost2d(ctx, area, ghost, ghost.counts, xMin, xMax, yMin, yMax);
  drawFitRangeIndicator(ctx, area, panel, xMin, xMax);
  panel.fitSummary = draw2dFit(ctx, area, panel, mask, x, y, xMin, xMax, yMin, yMax);
  drawReferenceCurves(ctx, area, panel, xMin, xMax, yMin, yMax);
  panel.lastPlot = {{
    mode: "2d", area, xName, x2Name, yName, y2Name, xMin, xMax, yMin, yMax,
    xBins, yBins, counts, overlayCounts, selected, overlaySelected, density: panel.density,
    logz: panel.logz, colorScale
  }};
  setPanelStats(panel, selected, sumX / selected, sumY / selected);
}}

function draw1dFacets(panel, mask, splitName) {{
  const xName = panel.xvar;
  const x2Name = panel.x2var && panel.x2var !== xName && columns[panel.x2var] ? panel.x2var : "";
  const x = columns[xName];
  const x2 = x2Name ? columns[x2Name] : null;
  const split = columns[splitName];
  const bins = panel.xbins;
  const xMin = panel.xmin;
  const xMax = panel.xmax;
  const facets = [];
  const facetDefinitions = facetDefinitionsForPanel(panel, splitName);
  const splitSignature = panelSplitSignature(panel, splitName);
  let totalSelected = 0, totalOverlaySelected = 0, sumXAll = 0;
  for (const definition of facetDefinitions) {{
    const counts = new Float64Array(bins);
    const overlayCounts = x2 ? new Float64Array(bins) : null;
    const fitValues = canonicalFitMethod(panel.fitMethod) === "unbinned" && panelHasFit(panel) ? [] : null;
    let selected = 0, overlaySelected = 0, sumX = 0;
    for (let i = 0; i < rowCount; i++) {{
      const xv = x[i];
      if (!mask[i] || !valueMatchesFacet(split[i], definition) || xMax <= xMin) continue;
      if (Number.isFinite(xv) && xv >= xMin && xv <= xMax) {{
        const bin = Math.min(bins - 1, Math.max(0, Math.floor((xv - xMin) / (xMax - xMin) * bins)));
        counts[bin] += 1;
        if (fitValues) fitValues.push(xv);
        selected++;
        sumX += xv;
      }}
      if (x2) {{
        const x2v = x2[i];
        if (Number.isFinite(x2v) && x2v >= xMin && x2v <= xMax) {{
          const bin = Math.min(bins - 1, Math.max(0, Math.floor((x2v - xMin) / (xMax - xMin) * bins)));
          overlayCounts[bin] += 1;
          overlaySelected++;
        }}
      }}
    }}
    totalSelected += selected;
    totalOverlaySelected += overlaySelected;
    sumXAll += sumX;
    facets.push({{value: definition.value, label: definition.label, shortLabel: definition.shortLabel, counts, overlayCounts, fitValues, selected, overlaySelected, meanX: sumX / selected}});
  }}
  if (panel.density) {{
    for (const facet of facets) {{
      normalizeHistogram(facet.counts, facet.selected);
      normalizeHistogram(facet.overlayCounts, facet.overlaySelected);
    }}
  }}
  const ghost = compatibleGhost(panel, "1d-facet", {{xName, splitName, splitSignature}});
  const pointSeries = [];
  for (const facet of facets) {{
    pointSeries.push({{values: facet.counts, total: facet.selected, density: panel.density}});
    if (facet.overlayCounts) pointSeries.push({{values: facet.overlayCounts, total: facet.overlaySelected, density: panel.density}});
  }}
  const ghostCounts = (ghost?.facets || []).flatMap(facet => Array.from(facet.counts));
  const maxCount = histogramPointScaleMax(pointSeries, ghostCounts);
  const canvas = el("plot" + panel.key);
  prepareFacetCanvas(canvas, panel, facets.length, splitName, facets);
  const area = plotArea(canvas);
  const {{ctx, width, height}} = area;
  const c = colors();
  ctx.clearRect(0, 0, width, height);
  const layout = facetLayout(area, facets.length, splitName, facets);
  const fitSummaries = [];
  for (let index = 0; index < facets.length; index++) {{
    const facet = facets[index];
    const facetAreaInfo = panelArea(area, layout, index);
    const axisVisibility = facetAxisVisibility(layout, index, facets.length);
    drawAxes(ctx, facetAreaInfo, xMin, xMax, 0, maxCount, axisDisplayLabel(panel, "x", byName[xName].label), axisDisplayLabel(panel, "y", panel.density ? "density" : "counts"), panel.xticks, panel.yticks, axisVisibility);
    drawMeanGuides(ctx, facetAreaInfo, panel, facet.meanX, NaN, xMin, xMax, 0, maxCount);
    draw1dPoints(ctx, facetAreaInfo, facet.counts, maxCount, facet.selected, panel.density, c.mark);
    if (facet.overlayCounts) {{
      draw1dPoints(ctx, facetAreaInfo, facet.overlayCounts, maxCount, facet.overlaySelected, panel.density, overlayHeatColor(0.82), 0.78);
    }}
    if (x2Name && index === 0) draw1dOverlayLegend(ctx, facetAreaInfo, byName[xName].label, byName[x2Name].label);
    const savedFacet = ghostFacet(ghost, facet.value);
    if (savedFacet) drawGhost1d(ctx, facetAreaInfo, {{...ghost, selected: savedFacet.selected}}, savedFacet.counts, xMin, xMax, maxCount, index === 0);
    drawFacetTitle(ctx, facetAreaInfo, `${{facet.label}} (${{facet.selected.toLocaleString()}})`);
    drawFitRangeIndicator(ctx, facetAreaInfo, panel, xMin, xMax);
    if (panelHasFit(panel)) {{
      const fit = make1dFit(facet.counts, xMin, xMax, panel, null, facet.fitValues);
      if (fit.predict) {{
        drawFitResult(ctx, facetAreaInfo, xMin, xMax, 0, maxCount, fit, panel);
        drawFitAnnotation(ctx, facetAreaInfo, fit, panel);
      }}
      fitSummaries.push(`${{facet.shortLabel}}: ${{fit.summary}}`);
    }}
    facet.area = facetAreaInfo;
  }}
  panel.lastPlot = {{
    mode: "1d-facet", area, facets, splitName, splitSignature, xName, x2Name, xMin, xMax, bins,
    selected: totalSelected, overlaySelected: totalOverlaySelected, density: panel.density, yMax: maxCount
  }};
  panel.fitSummary = panelHasFit(panel) ? fitSummaries.join(" | ") : "No fit";
  setPanelStats(panel, totalSelected, sumXAll / totalSelected, NaN);
}}

function draw2dFacets(panel, mask, splitName) {{
  const xName = panel.xvar;
  const yName = panel.yvar;
  const x2Name = panel.x2var && panel.x2var !== xName && columns[panel.x2var] ? panel.x2var : "";
  const y2Name = panel.y2var && panel.y2var !== yName && columns[panel.y2var] ? panel.y2var : "";
  const x = columns[xName];
  const y = columns[yName];
  const x2 = x2Name ? columns[x2Name] : null;
  const y2 = y2Name ? columns[y2Name] : null;
  const split = columns[splitName];
  const xBins = panel.xbins;
  const yBins = panel.ybins;
  const xMin = panel.xmin;
  const xMax = panel.xmax;
  const yMin = panel.ymin;
  const yMax = panel.ymax;
  const facets = [];
  const facetDefinitions = facetDefinitionsForPanel(panel, splitName);
  const splitSignature = panelSplitSignature(panel, splitName);
  const fitSpec = fitSpecFromPanel(panel);
  const collectFitPoints = fitSpec.signal === "none" && fitModelInfo(fitSpec.background).kind === "polynomial";
  let totalSelected = 0, totalOverlaySelected = 0, sumXAll = 0, sumYAll = 0;
  for (const definition of facetDefinitions) {{
    const counts = new Float64Array(xBins * yBins);
    const overlayCounts = (x2 || y2) ? new Float64Array(xBins * yBins) : null;
    const fitXs = collectFitPoints ? [] : null;
    const fitYs = collectFitPoints ? [] : null;
    let selected = 0, overlaySelected = 0, sumX = 0, sumY = 0;
    for (let i = 0; i < rowCount; i++) {{
      const xv = x[i], yv = y[i];
      if (!mask[i] || !valueMatchesFacet(split[i], definition) || xMax <= xMin || yMax <= yMin) continue;
      if (Number.isFinite(xv) && xv >= xMin && xv <= xMax && Number.isFinite(yv) && yv >= yMin && yv <= yMax) {{
        const xi = Math.min(xBins - 1, Math.max(0, Math.floor((xv - xMin) / (xMax - xMin) * xBins)));
        const yi = Math.min(yBins - 1, Math.max(0, Math.floor((yv - yMin) / (yMax - yMin) * yBins)));
        counts[yi * xBins + xi] += 1;
        if (collectFitPoints && xInFitRange(panel, xv)) {{
          fitXs.push(xv);
          fitYs.push(yv);
        }}
        selected++;
        sumX += xv;
        sumY += yv;
      }}
      if (overlayCounts) {{
        const x2v = x2 ? x2[i] : xv;
        const y2v = y2 ? y2[i] : yv;
        if (Number.isFinite(x2v) && x2v >= xMin && x2v <= xMax && Number.isFinite(y2v) && y2v >= yMin && y2v <= yMax) {{
          const xi = Math.min(xBins - 1, Math.max(0, Math.floor((x2v - xMin) / (xMax - xMin) * xBins)));
          const yi = Math.min(yBins - 1, Math.max(0, Math.floor((y2v - yMin) / (yMax - yMin) * yBins)));
          overlayCounts[yi * xBins + xi] += 1;
          overlaySelected++;
        }}
      }}
    }}
    totalSelected += selected;
    totalOverlaySelected += overlaySelected;
    sumXAll += sumX;
    sumYAll += sumY;
    facets.push({{...definition, counts, overlayCounts, selected, overlaySelected, meanX: sumX / selected, meanY: sumY / selected, fitXs, fitYs, maxCount: 1, overlayMaxCount: 0, colorScale: null}});
  }}
  if (panel.density) {{
    for (const facet of facets) {{
      normalizeHistogram(facet.counts, facet.selected);
      normalizeHistogram(facet.overlayCounts, facet.overlaySelected);
    }}
  }}
  for (const facet of facets) {{
    facet.maxCount = histogramMax(facet.counts);
    facet.overlayMaxCount = facet.overlayCounts ? histogramMax(facet.overlayCounts) : 0;
  }}
  const ghost = compatibleGhost(panel, "2d-facet", {{xName, yName, splitName, splitSignature}});
  const canvas = el("plot" + panel.key);
  prepareFacetCanvas(canvas, panel, facets.length, splitName, facets);
  const area = plotArea(canvas, panel.colorScale ? (x2Name || y2Name ? 2 : 1) : 0);
  const {{ctx, width, height}} = area;
  ctx.clearRect(0, 0, width, height);
  const layout = facetLayout(area, facets.length, splitName, facets);
  const fitSummaries = [];
  for (let index = 0; index < facets.length; index++) {{
    const facet = facets[index];
    const facetAreaInfo = panelArea(area, layout, index, panel.colorScale ? (facet.overlayCounts ? 2 : 1) : 0);
    const pw = facetAreaInfo.width - facetAreaInfo.left - facetAreaInfo.right;
    const ph = facetAreaInfo.height - facetAreaInfo.top - facetAreaInfo.bottom;
    for (let yi = 0; yi < yBins; yi++) {{
      for (let xi = 0; xi < xBins; xi++) {{
        const count = facet.counts[yi * xBins + xi];
        if (count <= 0) continue;
        const fraction = panel.logz ? Math.log1p(count) / Math.log1p(facet.maxCount) : count / facet.maxCount;
        ctx.fillStyle = heatColor(fraction);
        const x0 = facetAreaInfo.left + xi / xBins * pw;
        const x1 = facetAreaInfo.left + (xi + 1) / xBins * pw;
        const y0 = facetAreaInfo.top + ph - (yi + 1) / yBins * ph;
        const y1 = facetAreaInfo.top + ph - yi / yBins * ph;
        ctx.fillRect(x0, y0, Math.ceil(x1 - x0), Math.ceil(y1 - y0));
      }}
    }}
    if (facet.overlayCounts) {{
      ctx.save();
      ctx.globalAlpha = 0.58;
      for (let yi = 0; yi < yBins; yi++) {{
        for (let xi = 0; xi < xBins; xi++) {{
          const count = facet.overlayCounts[yi * xBins + xi];
          if (count <= 0) continue;
          const fraction = panel.logz ? Math.log1p(count) / Math.log1p(facet.overlayMaxCount) : count / facet.overlayMaxCount;
          ctx.fillStyle = overlayHeatColor(fraction);
          const x0 = facetAreaInfo.left + xi / xBins * pw;
          const x1 = facetAreaInfo.left + (xi + 1) / xBins * pw;
          const y0 = facetAreaInfo.top + ph - (yi + 1) / yBins * ph;
          const y1 = facetAreaInfo.top + ph - yi / yBins * ph;
          ctx.fillRect(x0, y0, Math.ceil(x1 - x0), Math.ceil(y1 - y0));
        }}
      }}
      ctx.restore();
    }}
    const xAxisLabel = x2Name ? `${{byName[xName].label}} / ${{byName[x2Name].label}}` : byName[xName].label;
    const yAxisLabel = y2Name ? `${{byName[yName].label}} / ${{byName[y2Name].label}}` : byName[yName].label;
    const axisVisibility = facetAxisVisibility(layout, index, facets.length);
    drawAxes(ctx, facetAreaInfo, xMin, xMax, yMin, yMax, axisDisplayLabel(panel, "x", xAxisLabel), axisDisplayLabel(panel, "y", yAxisLabel), panel.xticks, panel.yticks, axisVisibility);
    drawMeanGuides(ctx, facetAreaInfo, panel, facet.meanX, facet.meanY, xMin, xMax, yMin, yMax);
    if ((x2Name || y2Name) && index === 0) drawOverlayLegend(ctx, facetAreaInfo, `${{byName[yName].label}} vs ${{byName[xName].label}}`, overlay2dLabel({{xName, x2Name, yName, y2Name}}));
    const savedFacet = ghostFacet(ghost, facet.value);
    if (savedFacet) drawGhost2d(ctx, facetAreaInfo, {{...ghost, selected: savedFacet.selected}}, savedFacet.counts, xMin, xMax, yMin, yMax, index === 0);
    drawFacetTitle(ctx, facetAreaInfo, `${{facet.label}} (${{facet.selected.toLocaleString()}})`);
    if (panel.colorScale) facet.colorScale = draw2dColorScale(ctx, facetAreaInfo, facet.maxCount, facet.overlayCounts ? facet.overlayMaxCount : 0, panel);
    drawFitRangeIndicator(ctx, facetAreaInfo, panel, xMin, xMax);
    if (collectFitPoints) {{
      const fit = make2dFit(facet.fitXs, facet.fitYs, panel);
      if (fit.predict) {{
        drawFitResult(ctx, facetAreaInfo, xMin, xMax, yMin, yMax, fit, panel);
        drawFitAnnotation(ctx, facetAreaInfo, fit, panel);
      }}
      fitSummaries.push(`${{facet.shortLabel}}: ${{fit.summary}}; n=${{facet.fitXs.length.toLocaleString()}}`);
    }}
    drawReferenceCurves(ctx, facetAreaInfo, panel, xMin, xMax, yMin, yMax, index === 0);
    facet.area = facetAreaInfo;
  }}
  panel.lastPlot = {{
    mode: "2d-facet", area, facets, splitName, splitSignature, xName, x2Name, yName, y2Name, xMin, xMax, yMin, yMax,
    xBins, yBins, selected: totalSelected, overlaySelected: totalOverlaySelected, density: panel.density,
    logz: panel.logz
  }};
  panel.fitSummary = !panelHasFit(panel)
    ? "No fit"
    : !collectFitPoints
      ? `${{fitSpecLabel(fitSpec)}} is available for 1D histograms; choose only B polynomial for 2D trend fits`
      : fitSummaries.join(" | ");
  setPanelStats(panel, totalSelected, sumXAll / totalSelected, sumYAll / totalSelected);
}}

function facetLayout(area, facetCount, splitName = "", facets = null) {{
  const count = Math.max(1, facetCount || 1);
  const protonSectorValues = Array.isArray(facets) ? facets.map(facet => Math.round(Number(facet.value))) : [];
  const centeredCdLayout = isProtonSectorSplit(splitName)
    && count === 7
    && [0, 1, 2, 3, 4, 5, 6].every(value => protonSectorValues.includes(value));
  if (centeredCdLayout) {{
    return {{
      cols: 3,
      rows: 3,
      centeredCd: true,
      positions: [
        {{row: 0, col: 0}}, {{row: 0, col: 1}}, {{row: 0, col: 2}},
        {{row: 1, col: 0}}, {{row: 1, col: 1}}, {{row: 1, col: 2}},
        {{row: 2, col: 1}}
      ],
      gapX: 16, gapY: 30, outerLeft: 8, outerRight: 8, outerTop: 4, outerBottom: 8
    }};
  }}
  const cols = count <= 2 ? count : area.width >= 900 ? 3 : 2;
  const rows = Math.ceil(count / cols);
  return {{cols, rows, gapX: 16, gapY: 30, outerLeft: 8, outerRight: 8, outerTop: 4, outerBottom: 8}};
}}

function prepareFacetCanvas(canvas, panel, facetCount, splitName = "", facets = null) {{
  const count = Math.max(1, facetCount || 1);
  const protonValues = Array.isArray(facets) ? facets.map(facet => Math.round(Number(facet.value))) : [];
  const centeredCd = isProtonSectorSplit(splitName)
    && count === 7
    && [0, 1, 2, 3, 4, 5, 6].every(value => protonValues.includes(value));
  canvas.style.width = `${{canonicalPlotWidth(panel.plotWidthFraction) * 100}}%`;
  const width = canvas.getBoundingClientRect().width;
  const columns = centeredCd ? 3 : count <= 2 ? count : width >= 900 ? 3 : 2;
  const rows = centeredCd ? 3 : Math.ceil(count / Math.max(1, columns));
  const gapWidth = Math.max(0, columns - 1) * 16;
  const cellWidth = Math.max(120, (width - gapWidth - 16) / Math.max(1, columns));
  const plotWidth = Math.max(80, cellWidth - 60);
  const cellHeight = plotWidth * canonicalPlotHeight(panel.plotHeightFraction) * MAX_PLOT_HEIGHT_TO_WIDTH + 64;
  const reclaimedHeight = canvasToolbarCollapsed ? canvasToolbarExpandedHeight : 0;
  canvas.style.minHeight = "0";
  canvas.style.height = `${{clamp(rows * cellHeight + Math.max(0, rows - 1) * 30 + 12 + reclaimedHeight, 240, 4800)}}px`;
}}

function panelArea(area, layout, index, colorScaleSlots = 0) {{
  const cellW = (area.width - layout.outerLeft - layout.outerRight - (layout.cols - 1) * layout.gapX) / layout.cols;
  const cellH = (area.height - layout.outerTop - layout.outerBottom - (layout.rows - 1) * layout.gapY) / layout.rows;
  const position = layout.positions ? layout.positions[index] : null;
  const col = position ? position.col : index % layout.cols;
  const row = position ? position.row : Math.floor(index / layout.cols);
  const cellLeft = layout.outerLeft + col * (cellW + layout.gapX);
  const cellTop = layout.outerTop + row * (cellH + layout.gapY);
  const miniLeft = 52, miniRight = colorScaleSlots > 1 ? 96 : colorScaleSlots > 0 ? 68 : 8, miniTop = 24, miniBottom = 40;
  return {{
    ctx: area.ctx,
    width: area.width,
    height: area.height,
    left: cellLeft + miniLeft,
    right: area.width - (cellLeft + cellW - miniRight),
    top: cellTop + miniTop,
    bottom: area.height - (cellTop + cellH - miniBottom)
  }};
}}

function facetAxisVisibility(layout, index, facetCount) {{
  const position = layout.positions ? layout.positions[index] : {{row: Math.floor(index / layout.cols), col: index % layout.cols}};
  let hasFacetBelow = false;
  for (let other = 0; other < facetCount; other++) {{
    if (other === index) continue;
    const otherPosition = layout.positions
      ? layout.positions[other]
      : {{row: Math.floor(other / layout.cols), col: other % layout.cols}};
    if (otherPosition.col === position.col && otherPosition.row > position.row) {{
      hasFacetBelow = true;
      break;
    }}
  }}
  const centeredCdAxis = Boolean(layout.centeredCd && position.row === layout.rows - 1 && position.col === 1);
  return {{
    showXTickLabels: true,
    showYTickLabels: true,
    showXLabel: !hasFacetBelow,
    showYLabel: position.col === 0 || centeredCdAxis
  }};
}}

function drawFacetTitle(ctx, area, title) {{
  const c = colors();
  const pw = area.width - area.left - area.right;
  ctx.fillStyle = c.fg;
  ctx.font = "12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(title, area.left + pw / 2, area.top - 8);
}}

function drawOverlayLegend(ctx, area, primaryLabel, overlayLabel) {{
  const c = colors();
  const pw = area.width - area.left - area.right;
  const x = area.left + pw - 150;
  let y = area.top + 10;
  ctx.save();
  ctx.font = "12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = heatColor(0.78);
  ctx.fillRect(x, y - 5, 10, 10);
  ctx.fillStyle = c.fg;
  ctx.fillText(primaryLabel, x + 15, y);
  y += 16;
  ctx.globalAlpha = 0.75;
  ctx.fillStyle = overlayHeatColor(0.78);
  ctx.fillRect(x, y - 5, 10, 10);
  ctx.globalAlpha = 1;
  ctx.fillStyle = c.fg;
  ctx.fillText(overlayLabel, x + 15, y);
  ctx.restore();
}}

function draw1dOverlayLegend(ctx, area, primaryLabel, overlayLabel) {{
  const c = colors();
  const pw = area.width - area.left - area.right;
  const x = area.left + pw - 150;
  let y = area.top + 10;
  ctx.save();
  ctx.font = "12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const item of [
    {{label: primaryLabel, color: c.mark, alpha: 1}},
    {{label: overlayLabel, color: overlayHeatColor(0.82), alpha: 0.78}}
  ]) {{
    ctx.globalAlpha = item.alpha;
    ctx.strokeStyle = item.color;
    ctx.fillStyle = item.color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x + 5, y - 5);
    ctx.lineTo(x + 5, y + 5);
    ctx.moveTo(x + 2, y - 5);
    ctx.lineTo(x + 8, y - 5);
    ctx.moveTo(x + 2, y + 5);
    ctx.lineTo(x + 8, y + 5);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x + 5, y, 2.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = c.fg;
    ctx.fillText(item.label, x + 15, y);
    y += 16;
  }}
  ctx.restore();
}}

function draw2dColorScale(ctx, area, maxValue, overlayMaxValue, panel) {{
  const c = colors();
  const ph = area.height - area.top - area.bottom;
  const plotRight = area.width - area.right;
  const barTop = area.top;
  const barHeight = ph;
  const barWidth = 10;
  const primaryX = plotRight + 16;
  const overlayX = overlayMaxValue > 0 ? primaryX + 24 : 0;
  const primary = drawColorBar(ctx, primaryX, barTop, barWidth, barHeight, maxValue, heatColor, panel.logz, panel.density ? "density" : "count", c);
  let overlay = null;
  if (overlayMaxValue > 0) {{
    overlay = drawColorBar(ctx, overlayX, barTop, barWidth, barHeight, overlayMaxValue, overlayHeatColor, panel.logz, "overlay", c);
  }}
  return {{primary, overlay}};
}}

function drawColorBar(ctx, x, y, width, height, maxValue, colorFn, logScale, label, c) {{
  const steps = Math.max(20, Math.floor(height));
  for (let i = 0; i < steps; i++) {{
    const fraction = i / Math.max(1, steps - 1);
    ctx.fillStyle = colorFn(fraction);
    const y0 = y + height - (i + 1) / steps * height;
    const y1 = y + height - i / steps * height;
    ctx.fillRect(x, y0, width, Math.ceil(y1 - y0) + 1);
  }}
  ctx.strokeStyle = c.border;
  ctx.strokeRect(x, y, width, height);
  ctx.fillStyle = c.muted;
  ctx.font = "11px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const mid = logScale ? Math.expm1(Math.log1p(maxValue) * 0.5) : maxValue * 0.5;
  ctx.fillText(fmt(maxValue), x + width + 4, y + 3);
  ctx.fillText(fmt(mid), x + width + 4, y + height / 2);
  ctx.fillText("0", x + width + 4, y + height - 3);
  return {{x, y, width, height, maxValue, logScale, label}};
}}

function overlay2dLabel(plot) {{
  const xName = plot.x2Name || plot.xName;
  const yName = plot.y2Name || plot.yName;
  return `${{byName[yName]?.label || yName}} vs ${{byName[xName]?.label || xName}}`;
}}

function axisDisplayLabel(panel, axis, fallback) {{
  const label = axis === "x" ? panel.xLabel : panel.yLabel;
  return label || fallback;
}}

function variableLabel(name) {{
  return byName[name]?.label || name || "";
}}

function panelPrimaryQuantity(panel) {{
  if (panel.mode === "1d") return variableLabel(panel.xvar);
  return `${{variableLabel(panel.yvar)}} vs ${{variableLabel(panel.xvar)}}`;
}}

function panelOverlayQuantity(panel) {{
  const details = [];
  if (panel.mode === "1d" && panel.profile && panel.profile.variableName === panel.xvar) {{
    const closing = panel.profile.maxExclusive ? ")" : "]";
    details.push(
      `Profile ${{panel.profile.axis.toUpperCase()}} · ${{variableLabel(panel.profile.sliceName)}} `
      + `[${{formatAxisTick(panel.profile.min)}}, ${{formatAxisTick(panel.profile.max)}}${{closing}}`
    );
  }}
  if (panel.mode === "1d") {{
    if (panel.x2var && panel.x2var !== panel.xvar && byName[panel.x2var]) details.push(`overlay: ${{variableLabel(panel.x2var)}}`);
  }} else {{
    const x2Name = panel.x2var && panel.x2var !== panel.xvar && byName[panel.x2var] ? panel.x2var : panel.xvar;
    const y2Name = panel.y2var && panel.y2var !== panel.yvar && byName[panel.y2var] ? panel.y2var : panel.yvar;
    if (x2Name !== panel.xvar || y2Name !== panel.yvar) details.push(`overlay: ${{variableLabel(y2Name)}} vs ${{variableLabel(x2Name)}}`);
  }}
  if (panel.splitVar && byName[panel.splitVar]) {{
    const suffix = isCategoricalSplit(panel.splitVar)
      ? ""
      : ` (${{numericSliceConfiguration(panel, panel.splitVar).edges.length - 1}} slices)`;
    details.push(`split by ${{variableLabel(panel.splitVar)}}${{suffix}}`);
  }}
  return details.join(" | ");
}}

function updateQuantityBanner(panel) {{
  const banner = el("quantityBanner" + panel.key);
  if (!banner) return;
  banner.querySelector(".quantity-mode").textContent = panel.mode === "1d" ? "1D" : "2D";
  banner.querySelector("strong").textContent = panelPrimaryQuantity(panel);
  banner.querySelector(".quantity-detail").textContent = panelOverlayQuantity(panel);
}}

function draw1dFit(ctx, area, panel, counts, xMin, xMax, yMin, yMax, unbinnedValues = null) {{
  const fit = make1dFit(counts, xMin, xMax, panel, null, unbinnedValues);
  if (!fit.predict) return fit.summary;
  drawFitResult(ctx, area, xMin, xMax, yMin, yMax, fit, panel);
  drawFitAnnotation(ctx, area, fit, panel);
  return fit.summary;
}}

function fitSpecFromArgs(modelOrPanel, panel = null) {{
  if (panel) return fitSpecFromPanel(panel);
  if (modelOrPanel && typeof modelOrPanel === "object") return fitSpecFromPanel(modelOrPanel);
  return legacyFitSpec(modelOrPanel);
}}

function make1dFit(counts, xMin, xMax, modelOrPanel, panel = null, unbinnedValues = null) {{
  const ownerPanel = panel || (modelOrPanel && typeof modelOrPanel === "object" ? modelOrPanel : null);
  const spec = fitSpecFromArgs(modelOrPanel, panel);
  const method = canonicalFitMethod(ownerPanel ? (ownerPanel.fitMethod ?? ownerPanel.fitWeighting) : "unweighted");
  if (spec.signal === "none" && spec.background === "none") return {{summary: "No fit"}};
  if (method === "unbinned") {{
    if (!unbinnedValues) return {{summary: "Unbinned likelihood requires selected event values"}};
    return unbinnedLikelihoodFit(unbinnedValues, xMin, xMax, spec, ownerPanel, counts.length);
  }}
  if (method === "poisson" && ownerPanel && ownerPanel.density) {{
    return {{summary: "Poisson WLS requires count bins; turn off density"}};
  }}
  const xs = [];
  const ys = [];
  const binWidth = (xMax - xMin) / counts.length;
  for (let i = 0; i < counts.length; i++) {{
    const y = counts[i];
    if (!Number.isFinite(y)) continue;
    const x = xMin + (i + 0.5) * binWidth;
    if (ownerPanel && !xInFitRange(ownerPanel, x)) continue;
    xs.push(x);
    ys.push(y);
  }}
  const backgroundInfo = fitModelInfo(spec.background);
  const backgroundTerms = backgroundInfo.kind === "polynomial" ? backgroundInfo.degree + 1 : 0;
  const required = Math.max(backgroundTerms + (spec.signal === "none" ? 0 : 3), 2);
  if (xs.length < required) return {{summary: "Not enough bins for fit"}};
  const fit = spec.signal === "none"
    ? backgroundOnlyFit(xs, ys, spec.background, method)
    : signalBackgroundFit(xs, ys, spec.signal, spec.background, method);
  return fit || {{summary: "Fit failed"}};
}}

function draw2dFit(ctx, area, panel, mask, xValues, yValues, xMin, xMax, yMin, yMax) {{
  const spec = fitSpecFromPanel(panel);
  if (spec.signal === "none" && spec.background === "none") return "No fit";
  if (canonicalFitMethod(panel.fitMethod) === "unbinned") return "Unbinned likelihood is available for 1D distributions";
  if (spec.signal !== "none") return `${{fitSpecLabel(spec)}} is available for 1D histograms; choose only B polynomial for 2D trend fits`;
  const info = fitModelInfo(spec.background);
  if (info.kind !== "polynomial") return "Choose a B polynomial for 2D trend fits";
  const xs = [];
  const ys = [];
  for (let i = 0; i < rowCount; i++) {{
    const x = xValues[i];
    const y = yValues[i];
    if (!mask[i] || !Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < xMin || x > xMax || y < yMin || y > yMax) continue;
    if (!xInFitRange(panel, x)) continue;
    xs.push(x);
    ys.push(y);
  }}
  const fit = make2dFit(xs, ys, panel);
  if (!fit.predict) return fit.summary;
  drawFitResult(ctx, area, xMin, xMax, yMin, yMax, fit, panel);
  drawFitAnnotation(ctx, area, fit, panel);
  return `${{fit.summary}}; n=${{xs.length.toLocaleString()}}`;
}}

function make2dFit(xs, ys, modelOrPanel) {{
  const spec = fitSpecFromArgs(modelOrPanel);
  if (spec.signal === "none" && spec.background === "none") return {{summary: "No fit"}};
  if (modelOrPanel && typeof modelOrPanel === "object" && canonicalFitMethod(modelOrPanel.fitMethod) === "unbinned") {{
    return {{summary: "Unbinned likelihood is available for 1D distributions"}};
  }}
  if (spec.signal !== "none") return {{summary: `${{fitSpecLabel(spec)}} is available for 1D histograms`}};
  const info = fitModelInfo(spec.background);
  if (info.kind !== "polynomial") return {{summary: "Choose a B polynomial for 2D trend fits"}};
  if (xs.length < info.degree + 1) return {{summary: "Not enough selected points for fit"}};
  return backgroundOnlyFit(xs, ys, spec.background, "unweighted") || {{summary: "Fit failed"}};
}}

function drawFitResult(ctx, area, xMin, xMax, yMin, yMax, fit, panel) {{
  const c = colors();
  const bounds = fitRangeBounds(panel);
  if (fit.backgroundPredict && fit.signalPredict && fit.hasBackground) {{
    drawFitCurve(ctx, area, xMin, xMax, yMin, yMax, fit.backgroundPredict, bounds, {{strokeStyle: c.alert, lineDash: [2, 3], lineWidth: 1.5, alpha: 0.95}});
  }}
  if (fit.signalPredict && fit.hasBackground) {{
    drawFitCurve(ctx, area, xMin, xMax, yMin, yMax, fit.signalPredict, bounds, {{strokeStyle: c.mark, lineDash: [], lineWidth: 1.6, alpha: 0.95}});
  }}
  drawFitCurve(ctx, area, xMin, xMax, yMin, yMax, fit.predict, bounds, {{strokeStyle: c.fg, lineDash: [7, 4], lineWidth: 2}});
}}

function drawFitCurve(ctx, area, xMin, xMax, yMin, yMax, predict, xBounds = null, options = null) {{
  const c = colors();
  const style = options || {{}};
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  const drawMin = xBounds ? Math.max(xMin, xBounds[0]) : xMin;
  const drawMax = xBounds ? Math.min(xMax, xBounds[1]) : xMax;
  if (!Number.isFinite(drawMin) || !Number.isFinite(drawMax) || drawMax <= drawMin) return;
  ctx.save();
  ctx.strokeStyle = style.strokeStyle || c.fg;
  ctx.lineWidth = style.lineWidth || 2;
  ctx.globalAlpha = style.alpha === undefined ? 1 : style.alpha;
  ctx.setLineDash(style.lineDash || [7, 4]);
  ctx.beginPath();
  let started = false;
  for (let step = 0; step <= 160; step++) {{
    const x = drawMin + (drawMax - drawMin) * step / 160;
    const y = predict(x);
    if (!Number.isFinite(y)) {{
      started = false;
      continue;
    }}
    const px = area.left + (x - xMin) / (xMax - xMin) * pw;
    const py = area.top + ph - (y - yMin) / (yMax - yMin) * ph;
    if (py < area.top - ph || py > area.top + ph * 2) {{
      started = false;
      continue;
    }}
    if (!started) {{
      ctx.moveTo(px, py);
      started = true;
    }} else {{
      ctx.lineTo(px, py);
    }}
  }}
  ctx.stroke();
  ctx.restore();
}}

function drawFitRangeIndicator(ctx, area, panel, xMin, xMax) {{
  if (!panel || !Number.isFinite(xMin) || !Number.isFinite(xMax) || xMax <= xMin) return;
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  const c = colors();
  const toPixel = value => area.left + (value - xMin) / (xMax - xMin) * pw;
  const bounds = fitRangeBounds(panel);
  ctx.save();
  ctx.beginPath();
  ctx.rect(area.left, area.top, pw, ph);
  ctx.clip();
  ctx.strokeStyle = c.alert || c.fg;
  ctx.fillStyle = c.alert || c.fg;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  if (bounds) {{
    const lo = Math.max(bounds[0], xMin);
    const hi = Math.min(bounds[1], xMax);
    if (hi >= lo) {{
      const x0 = toPixel(lo);
      const x1 = toPixel(hi);
      ctx.globalAlpha = 0.08;
      ctx.fillRect(x0, area.top, Math.max(1, x1 - x0), ph);
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.moveTo(x0, area.top);
      ctx.lineTo(x0, area.top + ph);
      ctx.moveTo(x1, area.top);
      ctx.lineTo(x1, area.top + ph);
      ctx.stroke();
    }}
  }} else if (Number.isFinite(panel.fitRangeMin) && panel.fitRangeMin >= xMin && panel.fitRangeMin <= xMax) {{
    const x = toPixel(panel.fitRangeMin);
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    ctx.moveTo(x, area.top);
    ctx.lineTo(x, area.top + ph);
    ctx.stroke();
  }}
  ctx.restore();
}}

function drawFitAnnotation(ctx, area, fit, panel = null) {{
  if (!fit || !fit.predict || panel?.showFitAnnotations === false) return;
  const lines = fit.annotation || [fit.summary];
  const c = colors();
  const x = area.left + 6;
  const y = area.top + 8;
  const lineHeight = 12;
  const width = Math.min(190, Math.max(82, ...lines.map(line => line.length * 5.8)) + 8);
  const height = lineHeight * lines.length + 7;
  ctx.save();
  ctx.globalAlpha = 0.88;
  ctx.fillStyle = c.bg;
  ctx.fillRect(x - 4, y - 3, width, height);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = c.border;
  ctx.strokeRect(x - 4, y - 3, width, height);
  ctx.fillStyle = c.fg;
  ctx.font = "11px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  lines.forEach((line, index) => ctx.fillText(line, x, y + index * lineHeight));
  ctx.restore();
}}

function unbinnedLikelihoodFit(rawValues, xMin, xMax, spec, panel, binCount) {{
  const selectedBounds = fitRangeBounds(panel);
  const fitMin = selectedBounds ? Math.max(xMin, selectedBounds[0]) : xMin;
  const fitMax = selectedBounds ? Math.min(xMax, selectedBounds[1]) : xMax;
  if (!Number.isFinite(fitMin) || !Number.isFinite(fitMax) || fitMax <= fitMin) {{
    return {{summary: "Invalid unbinned fit range"}};
  }}
  const values = rawValues.filter(value => Number.isFinite(value) && value >= fitMin && value <= fitMax);
  if (values.length < 5) return {{summary: "Not enough events for unbinned likelihood fit"}};
  const backgroundInfo = fitModelInfo(spec.background);
  const backgroundDegree = backgroundInfo.kind === "polynomial" ? backgroundInfo.degree : -1;
  const scanDetail = canonicalFitScanDetail(panel ? panel.fitScanDetail : 3);
  let candidates = [null];
  if (spec.signal !== "none") {{
    const seed = unbinnedDistributionSeed(values);
    candidates = unbinnedSignalCandidates(spec.signal, seed, scanDetail);
    if (!candidates.length) return {{summary: "Could not initialize unbinned signal model"}};
  }}
  const previewLimits = [2000, 4000, 8000, 12000, 20000];
  const preview = deterministicFitSample(values, previewLimits[scanDetail - 1]);
  const previewIterations = 4 + scanDetail;
  const ranked = candidates
    .map(candidate => fitUnbinnedMixture(preview, candidate, backgroundDegree, fitMin, fitMax, previewIterations))
    .filter(Boolean)
    .sort((left, right) => left.nll - right.nll);
  if (!ranked.length) return {{summary: "Unbinned likelihood fit failed"}};
  let best = null;
  const finalistCount = Math.min(Math.ceil(scanDetail / 2), ranked.length);
  const fullIterations = 10 + 3 * scanDetail;
  for (const previewFit of ranked.slice(0, finalistCount)) {{
    const fit = fitUnbinnedMixture(values, previewFit.candidate, backgroundDegree, fitMin, fitMax, fullIterations);
    if (fit && (!best || fit.nll < best.nll)) best = fit;
  }}
  if (!best) return {{summary: "Unbinned likelihood fit failed"}};
  return formatUnbinnedLikelihoodFit(best, spec, backgroundInfo, panel, (xMax - xMin) / Math.max(1, binCount), values.length, scanDetail, candidates.length);
}}

function deterministicFitSample(values, limit) {{
  if (values.length <= limit) return values;
  const sampled = [];
  const step = values.length / limit;
  for (let i = 0; i < limit; i++) sampled.push(values[Math.floor(i * step)]);
  return sampled;
}}

function unbinnedDistributionSeed(values) {{
  const sorted = values.slice().sort((left, right) => left - right);
  const windowSize = clamp(Math.round(sorted.length * 0.2), Math.min(5, sorted.length), sorted.length);
  let start = 0;
  let smallestSpan = Infinity;
  for (let i = 0; i + windowSize <= sorted.length; i++) {{
    const span = sorted[i + windowSize - 1] - sorted[i];
    if (span < smallestSpan) {{
      smallestSpan = span;
      start = i;
    }}
  }}
  let mean = 0;
  for (let i = start; i < start + windowSize; i++) mean += sorted[i];
  mean /= windowSize;
  let variance = 0;
  for (let i = start; i < start + windowSize; i++) variance += Math.pow(sorted[i] - mean, 2);
  const fullSpan = Math.max(sorted[sorted.length - 1] - sorted[0], 1.0e-9);
  const sigma = Math.max(Math.sqrt(variance / windowSize), smallestSpan / 4, fullSpan / 1000);
  return {{mean, sigma}};
}}

function evenlySpaced(min, max, count) {{
  if (count <= 1) return [(min + max) / 2];
  return Array.from({{length: count}}, (_, index) => min + (max - min) * index / (count - 1));
}}

function unbinnedSignalCandidates(kind, seed, detail = 3) {{
  const scanDetail = canonicalFitScanDetail(detail);
  const candidates = [];
  const sigmaScales = evenlySpaced(1.0, 4.0, scanDetail + 1);
  const muOffsets = evenlySpaced(-1.0, 1.0, scanDetail + 2);
  if (kind === "gaussian") {{
    for (const sigmaScale of sigmaScales) {{
      for (const muOffset of muOffsets) {{
        const sigma = seed.sigma * sigmaScale;
        const mu = seed.mean + muOffset * seed.sigma;
        candidates.push({{kind, mu, sigma, shape: x => Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2))}});
      }}
    }}
  }} else if (kind === "crystalball") {{
    for (const side of ["left", "right"]) {{
      for (const sigmaScale of sigmaScales) {{
        for (const muOffset of muOffsets) {{
          const sigma = seed.sigma * sigmaScale;
          const mu = seed.mean + muOffset * seed.sigma;
          const alpha = 1.5;
          const n = 3;
          candidates.push({{kind, mu, sigma, alpha, n, side, shape: x => crystalBallShape(x, mu, sigma, alpha, n, side)}});
        }}
      }}
      const centralSigma = seed.sigma * sigmaScales[Math.floor(sigmaScales.length / 2)];
      for (const alpha of evenlySpaced(0.8, 2.8, scanDetail)) {{
        for (const n of evenlySpaced(1.5, 8, scanDetail)) {{
          const mu = seed.mean;
          candidates.push({{kind, mu, sigma: centralSigma, alpha, n, side, shape: x => crystalBallShape(x, mu, centralSigma, alpha, n, side)}});
        }}
      }}
    }}
  }}
  return candidates.filter(candidate => Number.isFinite(candidate.mu) && Number.isFinite(candidate.sigma) && candidate.sigma > 0);
}}

function fitUnbinnedMixture(values, candidate, backgroundDegree, fitMin, fitMax, maxIterations) {{
  const hasSignal = Boolean(candidate);
  const backgroundComponents = backgroundDegree >= 0 ? backgroundDegree + 1 : 0;
  const componentCount = (hasSignal ? 1 : 0) + backgroundComponents;
  if (!componentCount) return null;
  const signalNorm = hasSignal ? integratePositiveShape(candidate.shape, fitMin, fitMax) : NaN;
  if (hasSignal && (!Number.isFinite(signalNorm) || signalNorm <= 0)) return null;
  const pdfs = new Float64Array(values.length * componentCount);
  for (let i = 0; i < values.length; i++) {{
    let component = 0;
    if (hasSignal) pdfs[i * componentCount + component++] = Math.max(candidate.shape(values[i]) / signalNorm, 1.0e-300);
    for (let k = 0; k < backgroundComponents; k++) {{
      pdfs[i * componentCount + component++] = Math.max(bernsteinPdf(values[i], k, backgroundDegree, fitMin, fitMax), 1.0e-300);
    }}
  }}
  let weights;
  if (hasSignal && backgroundComponents) {{
    weights = [0.5, ...Array(backgroundComponents).fill(0.5 / backgroundComponents)];
  }} else {{
    weights = Array(componentCount).fill(1 / componentCount);
  }}
  for (let iteration = 0; iteration < maxIterations; iteration++) {{
    const sums = Array(componentCount).fill(0);
    for (let i = 0; i < values.length; i++) {{
      let denominator = 0;
      for (let component = 0; component < componentCount; component++) {{
        denominator += weights[component] * pdfs[i * componentCount + component];
      }}
      denominator = Math.max(denominator, 1.0e-300);
      for (let component = 0; component < componentCount; component++) {{
        sums[component] += weights[component] * pdfs[i * componentCount + component] / denominator;
      }}
    }}
    const next = sums.map(value => value / values.length);
    const change = next.reduce((largest, value, index) => Math.max(largest, Math.abs(value - weights[index])), 0);
    weights = next;
    if (change < 1.0e-7) break;
  }}
  let nll = 0;
  for (let i = 0; i < values.length; i++) {{
    let density = 0;
    for (let component = 0; component < componentCount; component++) density += weights[component] * pdfs[i * componentCount + component];
    nll -= Math.log(Math.max(density, 1.0e-300));
  }}
  return {{candidate, backgroundDegree, fitMin, fitMax, signalNorm, weights, nll, componentCount}};
}}

function integratePositiveShape(shape, min, max) {{
  const steps = 240;
  const width = (max - min) / steps;
  let sum = 0;
  for (let i = 0; i < steps; i++) sum += Math.max(0, shape(min + (i + 0.5) * width));
  return sum * width;
}}

function binomialCoefficient(n, k) {{
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 1; i <= Math.min(k, n - k); i++) result = result * (n - i + 1) / i;
  return result;
}}

function bernsteinPdf(x, component, degree, min, max) {{
  if (degree < 0 || x < min || x > max || max <= min) return 0;
  const u = clamp((x - min) / (max - min), 0, 1);
  return (degree + 1) * binomialCoefficient(degree, component) * Math.pow(u, component) * Math.pow(1 - u, degree - component) / (max - min);
}}

function formatUnbinnedLikelihoodFit(fit, spec, backgroundInfo, panel, displayBinWidth, eventCount, scanDetail, candidateCount) {{
  const hasSignal = Boolean(fit.candidate);
  const signalFraction = hasSignal ? fit.weights[0] : 0;
  const backgroundOffset = hasSignal ? 1 : 0;
  const backgroundWeights = fit.weights.slice(backgroundOffset);
  const signalPdf = x => hasSignal && x >= fit.fitMin && x <= fit.fitMax
    ? Math.max(0, fit.candidate.shape(x)) / fit.signalNorm
    : 0;
  const backgroundPdf = x => {{
    if (fit.backgroundDegree < 0 || x < fit.fitMin || x > fit.fitMax) return 0;
    return backgroundWeights.reduce((sum, weight, component) => sum + weight * bernsteinPdf(x, component, fit.backgroundDegree, fit.fitMin, fit.fitMax), 0);
  }};
  const displayScale = panel && panel.density ? displayBinWidth : eventCount * displayBinWidth;
  const signalPredict = x => displayScale * signalFraction * signalPdf(x);
  const backgroundPredict = x => displayScale * backgroundPdf(x);
  const predict = x => signalPredict(x) + backgroundPredict(x);
  const signalInfo = fit.candidate ? fitModelInfo(fit.candidate.kind) : null;
  const backgroundLabel = backgroundInfo.kind === "polynomial"
    ? `Bernstein degree ${{backgroundInfo.degree}}`
    : "none";
  const pieces = [];
  if (hasSignal) pieces.push(`fS=${{fmt(signalFraction)}}`, `mu=${{fmt(fit.candidate.mu)}}`, `sigma=${{fmt(fit.candidate.sigma)}}`);
  pieces.push(`NLL=${{fmt(fit.nll)}}`, `n=${{eventCount.toLocaleString()}}`);
  const modelLabel = `${{hasSignal ? signalInfo.label : "No signal"}} + B ${{backgroundLabel}}`;
  return {{
    predict,
    signalPredict,
    backgroundPredict,
    hasBackground: fit.backgroundDegree >= 0,
    backgroundDegree: fit.backgroundDegree,
    method: "unbinned",
    signalFraction,
    mean: hasSignal ? fit.candidate.mu : NaN,
    sigma: hasSignal ? fit.candidate.sigma : NaN,
    nll: fit.nll,
    eventCount,
    scanDetail,
    candidateCount,
    summary: `Unbinned ML ${{modelLabel}} (${{fitScanDetailLabel(scanDetail)}}, ${{candidateCount}} shapes): ${{pieces.join(", ")}}`,
    annotation: [`Unbinned ML ${{modelLabel}}`, `${{fitScanDetailLabel(scanDetail)}} scan, ${{candidateCount}} shapes`, ...pieces.slice(0, 3)]
  }};
}}

function backgroundOnlyFit(xs, ys, backgroundModel, weighting = "unweighted") {{
  const info = fitModelInfo(backgroundModel);
  if (info.kind !== "polynomial") return null;
  const fit = polynomialFit(xs, ys, info.degree, weighting);
  if (!fit) return null;
  const label = info.degree === 0 ? "B constant" : `B poly deg ${{info.degree}}`;
  const coeffText = fit.coeff.slice(0, Math.min(fit.coeff.length, 3)).map((value, index) => `b${{index}}=${{fmt(value)}}`);
  return {{
    ...fit,
    hasBackground: true,
    backgroundPredict: fit.predict,
    summary: `${{fitMethodLabel(weighting)}} ${{label}}: chi2/ndf=${{fmt(fit.quality.reduced)}}; ${{coeffText.join(", ")}}`,
    annotation: [`${{fitMethodLabel(weighting)}} ${{label}}`, `chi2/ndf=${{fmt(fit.quality.reduced)}}`, ...coeffText.slice(0, 2)]
  }};
}}

function signalBackgroundFit(xs, ys, signalModel, backgroundModel, weighting = "unweighted") {{
  const signalInfo = fitModelInfo(signalModel);
  const backgroundInfo = fitModelInfo(backgroundModel);
  if (signalInfo.kind !== "gaussian" && signalInfo.kind !== "crystalball") return backgroundOnlyFit(xs, ys, backgroundModel, weighting);
  const backgroundDegree = backgroundInfo.kind === "polynomial" ? backgroundInfo.degree : -1;
  const seed = distributionSeed(xs, ys) || fallbackDistributionSeed(xs, ys);
  if (!seed) return null;
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const xCenter = (xMin + xMax) / 2;
  const xScale = Math.max((xMax - xMin) / 2, 1.0e-12);
  let best = null;
  for (const candidate of signalShapeCandidates(signalInfo.kind, seed)) {{
    const linear = solveSignalBackgroundLinearFit(xs, ys, candidate, backgroundDegree, xCenter, xScale, weighting);
    if (!linear || linear.signalAmplitude <= 0) continue;
    const parameterCount = linear.coeff.length + candidate.nonlinearParameters;
    const quality = fitQuality(xs, ys, linear.predict, parameterCount, weighting);
    if (!best || quality.reduced < best.quality.reduced) {{
      best = {{...linear, ...candidate, backgroundDegree, quality, parameterCount}};
    }}
  }}
  if (!best) return null;
  return formatSignalBackgroundFit(best, signalInfo, backgroundInfo, weighting);
}}

function signalShapeCandidates(kind, seed) {{
  const candidates = [];
  if (kind === "gaussian") {{
    const sigmaScales = [0.45, 0.65, 0.85, 1.0, 1.2, 1.5, 2.0];
    const muOffsets = [-1.0, -0.5, 0, 0.5, 1.0];
    for (const sigmaScale of sigmaScales) {{
      const sigma = seed.sigma * sigmaScale;
      if (!Number.isFinite(sigma) || sigma <= 0) continue;
      for (const muOffset of muOffsets) {{
        const mu = seed.mean + muOffset * seed.sigma;
        candidates.push({{
          kind,
          mu,
          sigma,
          nonlinearParameters: 2,
          shape: x => Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2))
        }});
      }}
    }}
  }} else if (kind === "crystalball") {{
    const alphaValues = [0.7, 0.9, 1.1, 1.4, 1.8, 2.3, 3.0];
    const nValues = [1.4, 2, 3, 5, 8, 12];
    const sigmaScales = [0.65, 0.85, 1.05, 1.3, 1.65];
    const muOffsets = [-0.6, -0.25, 0, 0.25, 0.6];
    for (const side of ["left", "right"]) {{
      for (const sigmaScale of sigmaScales) {{
        const sigma = seed.sigma * sigmaScale;
        if (!Number.isFinite(sigma) || sigma <= 0) continue;
        for (const muOffset of muOffsets) {{
          const mu = seed.mean + muOffset * seed.sigma;
          for (const alpha of alphaValues) {{
            for (const n of nValues) {{
              candidates.push({{
                kind,
                mu,
                sigma,
                alpha,
                n,
                side,
                nonlinearParameters: 4,
                shape: x => crystalBallShape(x, mu, sigma, alpha, n, side)
              }});
            }}
          }}
        }}
      }}
    }}
  }}
  return candidates;
}}

function solveSignalBackgroundLinearFit(xs, ys, candidate, backgroundDegree, xCenter, xScale, weighting = "unweighted") {{
  const backgroundTerms = backgroundDegree >= 0 ? backgroundDegree + 1 : 0;
  const signalIndex = backgroundTerms;
  const termsByPoint = [];
  const fitYs = [];
  for (let i = 0; i < xs.length; i++) {{
    const signalValue = candidate.shape(xs[i]);
    if (!Number.isFinite(signalValue) || !Number.isFinite(ys[i])) continue;
    const terms = backgroundBasis(xs[i], backgroundDegree, xCenter, xScale);
    terms.push(signalValue);
    termsByPoint.push(terms);
    fitYs.push(ys[i]);
  }}
  const coeff = solveWeightedLinearTerms(termsByPoint, fitYs, weighting);
  if (!coeff || !coeff.every(Number.isFinite)) return null;
  const signalAmplitude = coeff[signalIndex];
  const backgroundPredict = x => {{
    const terms = backgroundBasis(x, backgroundDegree, xCenter, xScale);
    return terms.reduce((sum, term, index) => sum + coeff[index] * term, 0);
  }};
  const signalPredict = x => signalAmplitude * candidate.shape(x);
  const predict = x => backgroundPredict(x) + signalPredict(x);
  return {{coeff, signalAmplitude, backgroundPredict, signalPredict, predict, xCenter, xScale}};
}}

function backgroundBasis(x, degree, xCenter, xScale) {{
  if (degree < 0) return [];
  const scaled = (x - xCenter) / xScale;
  const terms = [1];
  for (let power = 1; power <= degree; power++) terms.push(terms[power - 1] * scaled);
  return terms;
}}

function formatSignalBackgroundFit(fit, signalInfo, backgroundInfo, weighting = "unweighted") {{
  const signalLabel = signalInfo.kind === "crystalball" ? "S Crystal Ball" : "S Gaussian";
  const backgroundLabel = backgroundInfo.kind === "polynomial"
    ? (backgroundInfo.degree === 0 ? "B constant" : `B poly deg ${{backgroundInfo.degree}}`)
    : "B none";
  const tail = fit.kind === "crystalball" ? (fit.side === "left" ? ", left tail" : ", right tail") : "";
  const params = [
    `mu=${{fmt(fit.mu)}}`,
    `sigma=${{fmt(fit.sigma)}}`,
    `A=${{fmt(fit.signalAmplitude)}}`
  ];
  if (fit.kind === "crystalball") {{
    params.push(`alpha=${{fmt(fit.alpha)}}`, `n=${{fmt(fit.n)}}`);
  }}
  const summary = `${{fitMethodLabel(weighting)}} ${{signalLabel}}${{tail}} + ${{backgroundLabel}}: ${{params.join(", ")}}, chi2/ndf=${{fmt(fit.quality.reduced)}}`;
  const annotation = [
    `${{fitMethodLabel(weighting)}} ${{signalLabel}} + ${{backgroundLabel}}`,
    `mu=${{fmt(fit.mu)}}`,
    `sigma=${{fmt(fit.sigma)}}`,
    ...(fit.kind === "crystalball" ? [`alpha=${{fmt(fit.alpha)}}`, `n=${{fmt(fit.n)}}`] : []),
    `chi2/ndf=${{fmt(fit.quality.reduced)}}`
  ];
  return {{...fit, hasBackground: fit.backgroundDegree >= 0, summary, annotation}};
}}

function polynomialFit(xs, ys, degree, weighting = "unweighted") {{
  const n = degree + 1;
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const xCenter = (xMin + xMax) / 2;
  const xScale = Math.max((xMax - xMin) / 2, 1.0e-12);
  const termsByPoint = [];
  const fitYs = [];
  for (let i = 0; i < xs.length; i++) {{
    const x = (xs[i] - xCenter) / xScale;
    const y = ys[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const powers = [1];
    for (let p = 1; p <= degree; p++) powers[p] = powers[p - 1] * x;
    termsByPoint.push(powers);
    fitYs.push(y);
  }}
  const coeff = solveWeightedLinearTerms(termsByPoint, fitYs, weighting);
  if (!coeff) return null;
  const predict = x => {{
    const scaled = (x - xCenter) / xScale;
    return coeff.reduce((sum, value, power) => sum + value * Math.pow(scaled, power), 0);
  }};
  const quality = fitQuality(xs, ys, predict, n, weighting);
  const coeffText = coeff.slice(0, Math.min(coeff.length, 4)).map((value, index) => `c${{index}}=${{fmt(value)}}`);
  const suffix = coeff.length > 4 ? ", ..." : "";
  const summary = `${{fitMethodLabel(weighting)}} poly${{degree}}: chi2/ndf=${{fmt(quality.reduced)}}; ${{coeffText.join(", ")}}${{suffix}}`;
  const annotation = [`${{fitMethodLabel(weighting)}} poly deg ${{degree}}`, `chi2/ndf=${{fmt(quality.reduced)}}`, ...coeffText.slice(0, 2)];
  return {{predict, summary, annotation, coeff, quality, xCenter, xScale}};
}}

function gaussianMomentFit(xs, ys) {{
  const baseline = Math.min(...ys);
  let weightSum = 0, meanSum = 0, peak = 0;
  for (let i = 0; i < xs.length; i++) {{
    const weight = Math.max(0, ys[i] - baseline);
    weightSum += weight;
    meanSum += weight * xs[i];
    if (ys[i] > peak) peak = ys[i];
  }}
  if (weightSum <= 0) return null;
  const mean = meanSum / weightSum;
  let variance = 0;
  for (let i = 0; i < xs.length; i++) {{
    const weight = Math.max(0, ys[i] - baseline);
    variance += weight * Math.pow(xs[i] - mean, 2);
  }}
  const sigma = Math.sqrt(variance / weightSum);
  if (!Number.isFinite(sigma) || sigma <= 0) return null;
  const amplitude = peak - baseline;
  const predict = x => baseline + amplitude * Math.exp(-0.5 * Math.pow((x - mean) / sigma, 2));
  const quality = fitQuality(xs, ys, predict, 4);
  return {{
    predict,
    summary: `Gaussian: mu=${{fmt(mean)}}, sigma=${{fmt(sigma)}}, A=${{fmt(amplitude)}}, chi2/ndf=${{fmt(quality.reduced)}}`,
    annotation: [`mu=${{fmt(mean)}}`, `sigma=${{fmt(sigma)}}`, `chi2/ndf=${{fmt(quality.reduced)}}`],
    mean,
    sigma,
    amplitude,
    baseline,
    quality
  }};
}}

function crystalBallFit(xs, ys) {{
  const seed = distributionSeed(xs, ys);
  if (!seed) return null;
  const alphaValues = [0.7, 0.9, 1.1, 1.4, 1.8, 2.3, 3.0];
  const nValues = [1.4, 2, 3, 5, 8, 12];
  const sigmaScales = [0.7, 0.9, 1.1, 1.4];
  const muOffsets = [-0.35, 0, 0.35];
  let best = null;
  for (const side of ["left", "right"]) {{
    for (const sigmaScale of sigmaScales) {{
      const sigma = seed.sigma * sigmaScale;
      if (!Number.isFinite(sigma) || sigma <= 0) continue;
      for (const muOffset of muOffsets) {{
        const mu = seed.mean + muOffset * seed.sigma;
        for (const alpha of alphaValues) {{
          for (const n of nValues) {{
            const shape = xs.map(x => crystalBallShape(x, mu, sigma, alpha, n, side));
            const linear = solveAmplitudeBaseline(shape, ys);
            if (!linear || linear.amplitude <= 0) continue;
            const predict = x => linear.baseline + linear.amplitude * crystalBallShape(x, mu, sigma, alpha, n, side);
            const quality = fitQuality(xs, ys, predict, 6);
            if (!best || quality.reduced < best.quality.reduced) {{
              best = {{mu, sigma, alpha, n, side, amplitude: linear.amplitude, baseline: linear.baseline, predict, quality}};
            }}
          }}
        }}
      }}
    }}
  }}
  if (!best) return null;
  const tail = best.side === "left" ? "left tail" : "right tail";
  return {{
    ...best,
    summary: `Crystal Ball (${{tail}}): mu=${{fmt(best.mu)}}, sigma=${{fmt(best.sigma)}}, alpha=${{fmt(best.alpha)}}, n=${{fmt(best.n)}}, chi2/ndf=${{fmt(best.quality.reduced)}}`,
    annotation: [`CB ${{tail}}`, `mu=${{fmt(best.mu)}}`, `sigma=${{fmt(best.sigma)}}`, `alpha=${{fmt(best.alpha)}}`, `n=${{fmt(best.n)}}`, `chi2/ndf=${{fmt(best.quality.reduced)}}`]
  }};
}}

function distributionSeed(xs, ys) {{
  const baseline = percentile(ys, 0.1);
  let weightSum = 0, meanSum = 0, peak = -Infinity;
  for (let i = 0; i < xs.length; i++) {{
    const weight = Math.max(0, ys[i] - baseline);
    weightSum += weight;
    meanSum += weight * xs[i];
    if (ys[i] > peak) peak = ys[i];
  }}
  if (weightSum <= 0) return null;
  const mean = meanSum / weightSum;
  let variance = 0;
  for (let i = 0; i < xs.length; i++) {{
    const weight = Math.max(0, ys[i] - baseline);
    variance += weight * Math.pow(xs[i] - mean, 2);
  }}
  const sigma = Math.sqrt(variance / weightSum);
  if (!Number.isFinite(sigma) || sigma <= 0) return null;
  return {{baseline, mean, sigma, peak, amplitude: peak - baseline}};
}}

function fallbackDistributionSeed(xs, ys) {{
  if (!xs.length) return null;
  let peakIndex = 0;
  for (let i = 1; i < ys.length; i++) {{
    if (ys[i] > ys[peakIndex]) peakIndex = i;
  }}
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const span = Math.max(xMax - xMin, 1.0e-6);
  return {{
    baseline: Math.min(...ys),
    mean: xs[peakIndex],
    sigma: span / 6,
    peak: ys[peakIndex],
    amplitude: ys[peakIndex] - Math.min(...ys)
  }};
}}

function percentile(values, fraction) {{
  const finite = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (!finite.length) return 0;
  const index = clamp(Math.floor((finite.length - 1) * fraction), 0, finite.length - 1);
  return finite[index];
}}

function crystalBallShape(x, mu, sigma, alpha, n, side) {{
  const z0 = (x - mu) / sigma;
  const z = side === "right" ? -z0 : z0;
  const absAlpha = Math.max(Math.abs(alpha), 1.0e-6);
  if (z > -absAlpha) return Math.exp(-0.5 * z * z);
  const a = Math.pow(n / absAlpha, n) * Math.exp(-0.5 * absAlpha * absAlpha);
  const b = n / absAlpha - absAlpha;
  return a * Math.pow(Math.max(1.0e-9, b - z), -n);
}}

function solveAmplitudeBaseline(shape, ys) {{
  let s00 = 0, s01 = 0, s11 = 0, t0 = 0, t1 = 0;
  for (let i = 0; i < shape.length; i++) {{
    const s = shape[i];
    const y = ys[i];
    if (!Number.isFinite(s) || !Number.isFinite(y)) continue;
    s00 += 1;
    s01 += s;
    s11 += s * s;
    t0 += y;
    t1 += y * s;
  }}
  const det = s00 * s11 - s01 * s01;
  if (Math.abs(det) < 1.0e-12) return null;
  const baseline = (t0 * s11 - t1 * s01) / det;
  const amplitude = (s00 * t1 - s01 * t0) / det;
  if (!Number.isFinite(baseline) || !Number.isFinite(amplitude)) return null;
  return {{baseline, amplitude}};
}}

function solveWeightedLinearTerms(termsByPoint, ys, weighting = "unweighted") {{
  if (!termsByPoint.length || termsByPoint.length !== ys.length) return null;
  const mode = canonicalFitMethod(weighting);
  let weights = Array(ys.length).fill(1);
  let coeff = solveLinearTerms(termsByPoint, ys, weights);
  if (!coeff || mode !== "poisson") return coeff;
  for (let iteration = 0; iteration < 8; iteration++) {{
    weights = termsByPoint.map(terms => {{
      const expected = terms.reduce((sum, term, index) => sum + term * coeff[index], 0);
      return 1 / Math.max(expected, 1);
    }});
    const next = solveLinearTerms(termsByPoint, ys, weights);
    if (!next) return null;
    const change = next.reduce((largest, value, index) => Math.max(largest, Math.abs(value - coeff[index])), 0);
    const scale = Math.max(1, ...next.map(Math.abs));
    coeff = next;
    if (change <= 1.0e-8 * scale) break;
  }}
  return coeff;
}}

function solveLinearTerms(termsByPoint, ys, weights) {{
  const termCount = termsByPoint[0].length;
  const matrix = Array.from({{length: termCount}}, () => Array(termCount).fill(0));
  const rhs = Array(termCount).fill(0);
  for (let i = 0; i < termsByPoint.length; i++) {{
    const terms = termsByPoint[i];
    const y = ys[i];
    const weight = weights[i];
    if (terms.length !== termCount || !Number.isFinite(y) || !Number.isFinite(weight) || weight <= 0) continue;
    for (let row = 0; row < termCount; row++) {{
      rhs[row] += weight * y * terms[row];
      for (let col = 0; col < termCount; col++) matrix[row][col] += weight * terms[row] * terms[col];
    }}
  }}
  return solveLinearSystem(matrix, rhs);
}}

function fitQuality(xs, ys, predict, parameterCount, weighting = "unweighted") {{
  let chi2 = 0;
  let used = 0;
  for (let i = 0; i < xs.length; i++) {{
    const expected = predict(xs[i]);
    const observed = ys[i];
    if (!Number.isFinite(expected) || !Number.isFinite(observed)) continue;
    const variance = Math.max(expected, 1);
    chi2 += Math.pow(observed - expected, 2) / variance;
    used++;
  }}
  const ndf = Math.max(1, used - parameterCount);
  return {{chi2, ndf, reduced: chi2 / ndf, weighting: canonicalFitMethod(weighting)}};
}}

function solveLinearSystem(matrix, rhs) {{
  const n = rhs.length;
  const a = matrix.map((row, i) => row.concat(rhs[i]));
  for (let col = 0; col < n; col++) {{
    let pivot = col;
    for (let row = col + 1; row < n; row++) {{
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }}
    if (Math.abs(a[pivot][col]) < 1e-12) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const scale = a[col][col];
    for (let j = col; j <= n; j++) a[col][j] /= scale;
    for (let row = 0; row < n; row++) {{
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
    }}
  }}
  return a.map(row => row[n]);
}}

function savePng() {{
  update();
  const keys = visiblePanelKeys();
  const plots = keys.map(key => {{
    const badge = filterBadgeText(filterStateForPanel(key));
    return {{
      key,
      canvas: el("plot" + key),
      title: panelLabels[key] || key,
      summary: [
        panelPrimaryQuantity(panels[key]),
        panelOverlayQuantity(panels[key]),
        badge ? `Filters: ${{badge.count}} active - ${{badge.detail}}` : ""
      ].filter(Boolean).join(" | ")
    }};
  }});
  if (!plots.length) return;
  const horizontal = plots.length > 1;
  const pad = 28;
  const gap = 24;
  const header = 48;
  const width = horizontal
    ? plots.reduce((sum, plot) => sum + plot.canvas.width, 0) + gap * (plots.length - 1) + pad * 2
    : Math.max(...plots.map(plot => plot.canvas.width)) + pad * 2;
  const height = horizontal
    ? Math.max(...plots.map(plot => plot.canvas.height)) + header + pad * 2
    : plots.reduce((sum, plot) => sum + plot.canvas.height + header, 0) + gap * (plots.length - 1) + pad * 2;
  const output = document.createElement("canvas");
  output.width = Math.max(1, width);
  output.height = Math.max(1, height);
  const ctx = output.getContext("2d");
  const c = colors();
  const bodyStyle = getComputedStyle(document.body);
  ctx.fillStyle = bodyStyle.backgroundColor || c.bg || "white";
  ctx.fillRect(0, 0, output.width, output.height);
  ctx.fillStyle = c.fg;
  ctx.strokeStyle = c.border;
  ctx.font = "22px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  let x = pad;
  let y = pad;
  for (const plot of plots) {{
    ctx.fillStyle = c.fg;
    ctx.font = "22px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillText(plot.title, x, y + 18);
    ctx.fillStyle = c.muted;
    ctx.font = "16px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillText(plot.summary, x, y + 40);
    ctx.drawImage(plot.canvas, x, y + header);
    ctx.strokeStyle = c.border;
    ctx.strokeRect(x + 0.5, y + header + 0.5, plot.canvas.width - 1, plot.canvas.height - 1);
    if (horizontal) x += plot.canvas.width + gap;
    else y += plot.canvas.height + header + gap;
  }}
  output.toBlob(blob => {{
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${{safeFilename(payload.title)}}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }}, "image/png");
}}

function safeFilename(value) {{
  const cleaned = String(value || "histograms")
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned || "histograms";
}}

function niceTicks(min, max, target) {{
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [];
  const span = max - min;
  const rawStep = span / Math.max(1, target);
  const power = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const fraction = rawStep / power;
  const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  const step = niceFraction * power;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let value = start; value <= max + step * 0.5; value += step) {{
    if (value >= min - step * 0.5) ticks.push(Number(value.toPrecision(12)));
  }}
  return ticks;
}}

function hoverElement(key) {{
  return el("hoverInfo" + key);
}}

function hoverOverlayElement(key) {{
  return el("hoverOverlay" + key);
}}

function setHoverText(key, text) {{
  const node = hoverElement(key);
  if (!node) return;
  node.textContent = text || "";
  node.style.display = text ? "block" : "none";
}}

function clearHoverOverlay(key) {{
  const overlay = hoverOverlayElement(key);
  if (!overlay) return;
  const ctx = overlay.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  overlay.style.display = "none";
}}

function prepareHoverOverlay(key) {{
  const canvas = el("plot" + key);
  const overlay = hoverOverlayElement(key);
  const pane = el("plotPane" + key);
  if (!canvas || !overlay || !pane) return null;
  const canvasRect = canvas.getBoundingClientRect();
  if (canvasRect.width <= 0 || canvasRect.height <= 0) return null;
  const paneRect = pane.getBoundingClientRect();
  overlay.style.left = (canvasRect.left - paneRect.left) + "px";
  overlay.style.top = (canvasRect.top - paneRect.top) + "px";
  overlay.style.width = canvasRect.width + "px";
  overlay.style.height = canvasRect.height + "px";
  overlay.width = canvas.width;
  overlay.height = canvas.height;
  overlay.style.display = "block";
  const ctx = overlay.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.setTransform(overlay.width / canvasRect.width, 0, 0, overlay.height / canvasRect.height, 0, 0);
  return {{ctx, width: canvasRect.width, height: canvasRect.height}};
}}

const PIN_MARKER_COLORS = [
  "#ff5f57", "#2f9bff", "#20b26b", "#b66dff", "#ff9f1c", "#00a6a6", "#e64980", "#7c8a00"
];

function pinnedMarkerColor(index) {{
  return PIN_MARKER_COLORS[Math.abs(index) % PIN_MARKER_COLORS.length];
}}

function pinnedMarkerLabel(index) {{
  return index < 26 ? String.fromCharCode(65 + index) : `P${{index + 1}}`;
}}

function plotMarkerSignature(lastPlot) {{
  if (!lastPlot) return "";
  return JSON.stringify([
    lastPlot.mode, lastPlot.xName || "", lastPlot.x2Name || "", lastPlot.yName || "", lastPlot.y2Name || "",
    lastPlot.splitName || "", lastPlot.splitSignature || "", lastPlot.bins || 0, lastPlot.xBins || 0,
    lastPlot.yBins || 0, lastPlot.xMin, lastPlot.xMax, lastPlot.yMin, lastPlot.yMax
  ]);
}}

function markerAreaAt(lastPlot, px, py) {{
  const facets = (lastPlot.mode === "1d-facet" || lastPlot.mode === "2d-facet") ? lastPlot.facets : null;
  if (facets) {{
    for (let index = 0; index < facets.length; index++) {{
      const area = facets[index].area;
      const pw = area.width - area.left - area.right;
      const ph = area.height - area.top - area.bottom;
      if (px >= area.left && px <= area.left + pw && py >= area.top && py <= area.top + ph) {{
        return {{area, facetIndex: index}};
      }}
    }}
    return null;
  }}
  const area = lastPlot.area;
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  return px >= area.left && px <= area.left + pw && py >= area.top && py <= area.top + ph
    ? {{area, facetIndex: -1}}
    : null;
}}

function pinnedMarkerAt(key, px, py) {{
  const lastPlot = panels[key]?.lastPlot;
  if (!lastPlot) return null;
  const hitArea = markerAreaAt(lastPlot, px, py);
  if (!hitArea) return null;
  const {{area, facetIndex}} = hitArea;
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  const signature = plotMarkerSignature(lastPlot);
  if (lastPlot.mode === "1d" || lastPlot.mode === "1d-facet") {{
    const bin = clamp(Math.floor((px - area.left) / pw * lastPlot.bins), 0, lastPlot.bins - 1);
    return {{id: `${{signature}}:${{facetIndex}}:${{bin}}`, signature, facetIndex, bin}};
  }}
  const xi = clamp(Math.floor((px - area.left) / pw * lastPlot.xBins), 0, lastPlot.xBins - 1);
  const yi = clamp(Math.floor((area.top + ph - py) / ph * lastPlot.yBins), 0, lastPlot.yBins - 1);
  return {{id: `${{signature}}:${{facetIndex}}:${{xi}}:${{yi}}`, signature, facetIndex, xi, yi}};
}}

function resolvePinnedMarker(panel, marker) {{
  const lastPlot = panel?.lastPlot;
  if (!lastPlot || marker.signature !== plotMarkerSignature(lastPlot)) return null;
  const facet = marker.facetIndex >= 0 ? lastPlot.facets?.[marker.facetIndex] : null;
  const area = facet?.area || lastPlot.area;
  if (!area) return null;
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  const facetText = facet ? `${{facet.label}}; ` : "";
  if (lastPlot.mode === "1d" || lastPlot.mode === "1d-facet") {{
    if (marker.bin < 0 || marker.bin >= lastPlot.bins) return null;
    const counts = facet ? facet.counts : lastPlot.counts;
    const value = counts?.[marker.bin];
    if (!Number.isFinite(value)) return null;
    const x0 = lastPlot.xMin + marker.bin / lastPlot.bins * (lastPlot.xMax - lastPlot.xMin);
    const x1 = lastPlot.xMin + (marker.bin + 1) / lastPlot.bins * (lastPlot.xMax - lastPlot.xMin);
    const xValue = (x0 + x1) / 2;
    const xPixel = area.left + (marker.bin + 0.5) / lastPlot.bins * pw;
    const yPixel = area.top + ph - (lastPlot.yMax > 0 ? value / lastPlot.yMax * ph : 0);
    return {{
      area, xPixel, yPixel, xValue, yValue: value, yLabelText: fmt(value),
      summary: `${{facetText}}x=${{fmt(xValue)}}, ${{lastPlot.density ? "density" : "count"}}=${{fmt(value)}}`
    }};
  }}
  if (marker.xi < 0 || marker.xi >= lastPlot.xBins || marker.yi < 0 || marker.yi >= lastPlot.yBins) return null;
  const counts = facet ? facet.counts : lastPlot.counts;
  const count = counts?.[marker.yi * lastPlot.xBins + marker.xi];
  const x0 = lastPlot.xMin + marker.xi / lastPlot.xBins * (lastPlot.xMax - lastPlot.xMin);
  const x1 = lastPlot.xMin + (marker.xi + 1) / lastPlot.xBins * (lastPlot.xMax - lastPlot.xMin);
  const y0 = lastPlot.yMin + marker.yi / lastPlot.yBins * (lastPlot.yMax - lastPlot.yMin);
  const y1 = lastPlot.yMin + (marker.yi + 1) / lastPlot.yBins * (lastPlot.yMax - lastPlot.yMin);
  const xValue = (x0 + x1) / 2;
  const yValue = (y0 + y1) / 2;
  const xPixel = area.left + (marker.xi + 0.5) / lastPlot.xBins * pw;
  const yPixel = area.top + ph - (marker.yi + 0.5) / lastPlot.yBins * ph;
  return {{
    area, xPixel, yPixel, xValue, yValue, yLabelText: fmt(yValue),
    summary: `${{facetText}}x=${{fmt(xValue)}}, y=${{fmt(yValue)}}, ${{lastPlot.density ? "density" : "count"}}=${{fmt(count)}}`
  }};
}}

function drawPinnedMarkersOnOverlay(key, overlay) {{
  const panel = panels[key];
  if (!panel || !overlay) return;
  const signature = plotMarkerSignature(panel.lastPlot);
  panel.pinnedMarkers = (panel.pinnedMarkers || []).filter(marker => marker.signature === signature);
  panel.pinnedMarkers.forEach((marker, lane) => {{
    const resolved = resolvePinnedMarker(panel, marker);
    if (!resolved) return;
    drawCrosshairOnOverlay(
      overlay, resolved.area, resolved.xPixel, resolved.yPixel, resolved.xValue, resolved.yValue,
      resolved.yLabelText, pinnedMarkerColor(marker.colorIndex), pinnedMarkerLabel(marker.colorIndex), lane
    );
  }});
}}

function renderPinnedMarkers(key) {{
  const panel = panels[key];
  if (panel) {{
    const signature = plotMarkerSignature(panel.lastPlot);
    panel.pinnedMarkers = (panel.pinnedMarkers || []).filter(marker => marker.signature === signature);
  }}
  if (!panel?.pinnedMarkers?.length) {{
    clearHoverOverlay(key);
    return;
  }}
  const overlay = prepareHoverOverlay(key);
  drawPinnedMarkersOnOverlay(key, overlay);
}}

function pinnedMarkerSummary(key) {{
  const panel = panels[key];
  if (!panel?.pinnedMarkers?.length) return "";
  const summaries = [];
  for (const marker of panel.pinnedMarkers) {{
    const resolved = resolvePinnedMarker(panel, marker);
    if (resolved) summaries.push(`${{pinnedMarkerLabel(marker.colorIndex)}}: ${{resolved.summary}}`);
  }}
  return summaries.length ? `Pinned · ${{summaries.join(" | ")}}` : "";
}}

function togglePinnedMarker(event, key) {{
  const panel = panels[key];
  const rect = el("plot" + key).getBoundingClientRect();
  const marker = pinnedMarkerAt(key, event.clientX - rect.left, event.clientY - rect.top);
  if (!panel || !marker) return;
  const existing = (panel.pinnedMarkers || []).findIndex(item => item.id === marker.id);
  if (existing >= 0) {{
    panel.pinnedMarkers.splice(existing, 1);
  }} else {{
    marker.colorIndex = panel.nextPinnedMarkerColor++;
    panel.pinnedMarkers.push(marker);
  }}
  setHoverText(key, pinnedMarkerSummary(key));
  renderPinnedMarkers(key);
  hideColorScaleMarker(key);
}}

function drawHoverCrosshair(key, area, xPixel, yPixel, xValue, yValue, yLabelText = null) {{
  const overlay = prepareHoverOverlay(key);
  if (!overlay) return;
  drawPinnedMarkersOnOverlay(key, overlay);
  drawCrosshairOnOverlay(overlay, area, xPixel, yPixel, xValue, yValue, yLabelText);
}}

function drawCrosshairOnOverlay(overlay, area, xPixel, yPixel, xValue, yValue, yLabelText = null, color = null, markerLabel = "", lane = 0) {{
  const {{ctx, width, height}} = overlay;
  const c = colors();
  const strokeColor = color || c.fg;
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  const x = clamp(xPixel, area.left, area.left + pw);
  const y = clamp(yPixel, area.top, area.top + ph);
  ctx.save();
  ctx.beginPath();
  ctx.rect(area.left, area.top, pw, ph);
  ctx.clip();
  ctx.globalAlpha = 0.82;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(x, area.top);
  ctx.lineTo(x, area.top + ph);
  ctx.moveTo(area.left, y);
  ctx.lineTo(area.left + pw, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 0.95;
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, Math.PI * 2);
  ctx.fillStyle = strokeColor;
  ctx.fill();
  ctx.restore();
  const prefix = markerLabel ? `${{markerLabel}} · ` : "";
  drawHoverAxisLabel(ctx, prefix + fmt(xValue), x, area.top + ph + 7 + lane * 18, "x", width, height, strokeColor);
  drawHoverAxisLabel(ctx, prefix + (yLabelText || fmt(yValue)), area.left - 8 + lane * 8, y, "y", width, height, strokeColor);
}}

function drawHoverAxisLabel(ctx, text, x, y, placement, width, height, color = null) {{
  const c = colors();
  const label = String(text);
  ctx.save();
  ctx.font = "11px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.textBaseline = "middle";
  const padX = 5;
  const boxH = 18;
  const boxW = Math.max(28, ctx.measureText(label).width + padX * 2);
  let left;
  let top;
  if (placement === "x") {{
    left = clamp(x - boxW / 2, 2, width - boxW - 2);
    top = clamp(y, 2, height - boxH - 2);
  }} else {{
    left = clamp(x - boxW, 2, width - boxW - 2);
    top = clamp(y - boxH / 2, 2, height - boxH - 2);
  }}
  ctx.fillStyle = c.bg;
  ctx.globalAlpha = 0.96;
  ctx.fillRect(left, top, boxW, boxH);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = color || c.border;
  ctx.strokeRect(left, top, boxW, boxH);
  ctx.fillStyle = color || c.fg;
  ctx.textAlign = "center";
  ctx.fillText(label, left + boxW / 2, top + boxH / 2);
  ctx.restore();
}}

function hideColorScaleMarker(key) {{
  for (const suffix of ["Primary", "Overlay"]) {{
    const marker = el("colorScaleHover" + key + suffix);
    if (marker) marker.style.display = "none";
  }}
}}

function showColorScaleMarker(key, scaleInfo, value, colorFn, suffix) {{
  const marker = el("colorScaleHover" + key + suffix);
  if (!marker || !scaleInfo) return;
  const maxValue = Math.max(scaleInfo.maxValue || 0, 0);
  const fraction = maxValue > 0
    ? (scaleInfo.logScale ? Math.log1p(Math.max(0, value)) / Math.log1p(maxValue) : Math.max(0, value) / maxValue)
    : 0;
  const clamped = clamp(fraction, 0, 1);
  const canvasRect = el("plot" + key).getBoundingClientRect();
  const paneRect = el("plotPane" + key).getBoundingClientRect();
  const preferredLeft = canvasRect.left - paneRect.left + scaleInfo.x - 4;
  const markerTop = canvasRect.top - paneRect.top + scaleInfo.y + scaleInfo.height - clamped * scaleInfo.height;
  marker.style.setProperty("--marker-width", (scaleInfo.width + 10) + "px");
  marker.style.setProperty("--marker-color", colorFn(clamped));
  marker.querySelector(".scale-name").textContent = scaleInfo.label || "";
  marker.querySelector(".scale-value").textContent = fmt(value);
  marker.style.display = "flex";
  const markerLeft = constrainedOverlayLeft(preferredLeft, marker.offsetWidth, paneRect.width);
  marker.style.left = markerLeft + "px";
  marker.style.top = markerTop + "px";
}}

function constrainedOverlayLeft(preferredLeft, overlayWidth, containerWidth, padding = 4) {{
  return clamp(preferredLeft, padding, Math.max(padding, containerWidth - overlayWidth - padding));
}}

function showColorScaleMarkers(key, colorScale, value, overlayValue) {{
  hideColorScaleMarker(key);
  if (!colorScale) return;
  showColorScaleMarker(key, colorScale.primary, value, heatColor, "Primary");
  if (colorScale.overlay && Number.isFinite(overlayValue)) {{
    showColorScaleMarker(key, colorScale.overlay, overlayValue, overlayHeatColor, "Overlay");
  }}
}}

function showHoverInfo(event, key) {{
  const lastPlot = panels[key].lastPlot;
  if (!lastPlot) return;
  const rect = el("plot" + key).getBoundingClientRect();
  const px = event.clientX - rect.left;
  const py = event.clientY - rect.top;
  if (lastPlot.mode === "1d-facet" || lastPlot.mode === "2d-facet") {{
    showFacetHover(px, py, key);
    return;
  }}
  const area = lastPlot.area;
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  if (px < area.left || px > area.left + pw || py < area.top || py > area.top + ph) {{
    setHoverText(key, pinnedMarkerSummary(key));
    renderPinnedMarkers(key);
    hideColorScaleMarker(key);
    return;
  }}
  if (lastPlot.mode === "1d") {{
    const bin = clamp(Math.floor((px - area.left) / pw * lastPlot.bins), 0, lastPlot.bins - 1);
    const x0 = lastPlot.xMin + bin / lastPlot.bins * (lastPlot.xMax - lastPlot.xMin);
    const x1 = lastPlot.xMin + (bin + 1) / lastPlot.bins * (lastPlot.xMax - lastPlot.xMin);
    const value = lastPlot.counts[bin];
    const overlayValue = lastPlot.overlayCounts ? lastPlot.overlayCounts[bin] : NaN;
    const label = lastPlot.density ? "density" : "count";
    const overlayText = lastPlot.x2Name ? `; ${{byName[lastPlot.x2Name].label}} ${{label}}=${{fmt(overlayValue)}}; overlay selected=${{lastPlot.overlaySelected.toLocaleString()}}` : "";
    setHoverText(key, `Panel ${{key}}; ${{byName[lastPlot.xName].label}} [${{fmt(x0)}}, ${{fmt(x1)}}): ${{label}}=${{fmt(value)}}${{overlayText}}; bin=${{bin + 1}}/${{lastPlot.bins}}; selected=${{lastPlot.selected.toLocaleString()}}`);
    const xCenter = (x0 + x1) / 2;
    const xPixel = area.left + (bin + 0.5) / lastPlot.bins * pw;
    const yPixel = area.top + ph - (lastPlot.yMax > 0 ? value / lastPlot.yMax * ph : 0);
    drawHoverCrosshair(key, area, xPixel, yPixel, xCenter, value);
    hideColorScaleMarker(key);
    return;
  }}
  const xi = clamp(Math.floor((px - area.left) / pw * lastPlot.xBins), 0, lastPlot.xBins - 1);
  const yi = clamp(Math.floor((area.top + ph - py) / ph * lastPlot.yBins), 0, lastPlot.yBins - 1);
  const x0 = lastPlot.xMin + xi / lastPlot.xBins * (lastPlot.xMax - lastPlot.xMin);
  const x1 = lastPlot.xMin + (xi + 1) / lastPlot.xBins * (lastPlot.xMax - lastPlot.xMin);
  const y0 = lastPlot.yMin + yi / lastPlot.yBins * (lastPlot.yMax - lastPlot.yMin);
  const y1 = lastPlot.yMin + (yi + 1) / lastPlot.yBins * (lastPlot.yMax - lastPlot.yMin);
  const value = lastPlot.counts[yi * lastPlot.xBins + xi];
  const overlayValue = lastPlot.overlayCounts ? lastPlot.overlayCounts[yi * lastPlot.xBins + xi] : NaN;
  const label = lastPlot.density ? "density" : "count";
  const overlayText = lastPlot.overlayCounts ? `; ${{overlay2dLabel(lastPlot)}} ${{label}}=${{fmt(overlayValue)}}; overlay selected=${{lastPlot.overlaySelected.toLocaleString()}}` : "";
  setHoverText(key, `Panel ${{key}}; ${{byName[lastPlot.yName].label}} [${{fmt(y0)}}, ${{fmt(y1)}}), ${{byName[lastPlot.xName].label}} [${{fmt(x0)}}, ${{fmt(x1)}}): ${{label}}=${{fmt(value)}}${{overlayText}}; bin=(${{xi + 1}}, ${{yi + 1}}); selected=${{lastPlot.selected.toLocaleString()}}`);
  const xCenter = (x0 + x1) / 2;
  const yCenter = (y0 + y1) / 2;
  const xPixel = area.left + (xi + 0.5) / lastPlot.xBins * pw;
  const yPixel = area.top + ph - (yi + 0.5) / lastPlot.yBins * ph;
  drawHoverCrosshair(key, area, xPixel, yPixel, xCenter, yCenter);
  showColorScaleMarkers(key, lastPlot.colorScale, value, overlayValue);
}}

function showFacetHover(px, py, key) {{
  const lastPlot = panels[key].lastPlot;
  for (const facet of lastPlot.facets) {{
    const area = facet.area;
    const pw = area.width - area.left - area.right;
    const ph = area.height - area.top - area.bottom;
    if (px < area.left || px > area.left + pw || py < area.top || py > area.top + ph) continue;
    if (lastPlot.mode === "1d-facet") {{
      const bin = clamp(Math.floor((px - area.left) / pw * lastPlot.bins), 0, lastPlot.bins - 1);
      const x0 = lastPlot.xMin + bin / lastPlot.bins * (lastPlot.xMax - lastPlot.xMin);
      const x1 = lastPlot.xMin + (bin + 1) / lastPlot.bins * (lastPlot.xMax - lastPlot.xMin);
      const value = facet.counts[bin];
      const overlayValue = facet.overlayCounts ? facet.overlayCounts[bin] : NaN;
      const label = lastPlot.density ? "density" : "count";
      const overlayText = lastPlot.x2Name ? `; ${{byName[lastPlot.x2Name].label}} ${{label}}=${{fmt(overlayValue)}}; overlay split selected=${{facet.overlaySelected.toLocaleString()}}` : "";
      setHoverText(key, `Panel ${{key}}; ${{facet.label}}; ${{byName[lastPlot.xName].label}} [${{fmt(x0)}}, ${{fmt(x1)}}): ${{label}}=${{fmt(value)}}${{overlayText}}; bin=${{bin + 1}}/${{lastPlot.bins}}; split selected=${{facet.selected.toLocaleString()}}`);
      const xCenter = (x0 + x1) / 2;
      const xPixel = area.left + (bin + 0.5) / lastPlot.bins * pw;
      const yPixel = area.top + ph - (lastPlot.yMax > 0 ? value / lastPlot.yMax * ph : 0);
      drawHoverCrosshair(key, area, xPixel, yPixel, xCenter, value);
      hideColorScaleMarker(key);
      return;
    }}
    const xi = clamp(Math.floor((px - area.left) / pw * lastPlot.xBins), 0, lastPlot.xBins - 1);
    const yi = clamp(Math.floor((area.top + ph - py) / ph * lastPlot.yBins), 0, lastPlot.yBins - 1);
    const x0 = lastPlot.xMin + xi / lastPlot.xBins * (lastPlot.xMax - lastPlot.xMin);
    const x1 = lastPlot.xMin + (xi + 1) / lastPlot.xBins * (lastPlot.xMax - lastPlot.xMin);
    const y0 = lastPlot.yMin + yi / lastPlot.yBins * (lastPlot.yMax - lastPlot.yMin);
    const y1 = lastPlot.yMin + (yi + 1) / lastPlot.yBins * (lastPlot.yMax - lastPlot.yMin);
    const value = facet.counts[yi * lastPlot.xBins + xi];
    const overlayValue = facet.overlayCounts ? facet.overlayCounts[yi * lastPlot.xBins + xi] : NaN;
    const label = lastPlot.density ? "density" : "count";
    const overlayText = facet.overlayCounts ? `; ${{overlay2dLabel(lastPlot)}} ${{label}}=${{fmt(overlayValue)}}; overlay split selected=${{facet.overlaySelected.toLocaleString()}}` : "";
    setHoverText(key, `Panel ${{key}}; ${{facet.label}}; ${{byName[lastPlot.yName].label}} [${{fmt(y0)}}, ${{fmt(y1)}}), ${{byName[lastPlot.xName].label}} [${{fmt(x0)}}, ${{fmt(x1)}}): ${{label}}=${{fmt(value)}}${{overlayText}}; bin=(${{xi + 1}}, ${{yi + 1}}); split selected=${{facet.selected.toLocaleString()}}`);
    const xCenter = (x0 + x1) / 2;
    const yCenter = (y0 + y1) / 2;
    const xPixel = area.left + (xi + 0.5) / lastPlot.xBins * pw;
    const yPixel = area.top + ph - (yi + 0.5) / lastPlot.yBins * ph;
    drawHoverCrosshair(key, area, xPixel, yPixel, xCenter, yCenter);
    showColorScaleMarkers(key, facet.colorScale, value, overlayValue);
    return;
  }}
  setHoverText(key, pinnedMarkerSummary(key));
  renderPinnedMarkers(key);
  hideColorScaleMarker(key);
}}

function heatColor(t) {{
  const hue = 225 - 175 * t;
  const light = 92 - 45 * t;
  return `hsl(${{hue}} 78% ${{light}}%)`;
}}

function overlayHeatColor(t) {{
  const hue = 150 + 145 * t;
  const light = 91 - 43 * t;
  return `hsl(${{hue}} 82% ${{light}}%)`;
}}

function clamp(value, min, max) {{
  return Math.max(min, Math.min(max, value));
}}

function maxOf(values, fallback) {{
  let result = fallback;
  for (let i = 0; i < values.length; i++) {{
    if (values[i] > result) result = values[i];
  }}
  return result;
}}

function normalizeHistogram(values, total) {{
  if (!values || total <= 0) return;
  for (let i = 0; i < values.length; i++) values[i] /= total;
}}

function histogramMax(...arrays) {{
  let result = 0;
  for (const values of arrays) {{
    if (!values) continue;
    for (let i = 0; i < values.length; i++) {{
      const value = values[i];
      if (Number.isFinite(value) && value > result) result = value;
    }}
  }}
  return result > 0 ? result : 1;
}}

function setPanelStats(panel, selected, meanX, meanY) {{
  panel.stats = {{selected, meanX, meanY}};
  updateQuantityBanner(panel);
}}

function updateActiveStats() {{
  const stats = currentPanel().stats;
  el("selectedCount").textContent = stats.selected.toLocaleString();
  el("meanX").textContent = fmt(stats.meanX);
  el("meanY").textContent = fmt(stats.meanY);
  renderFitSummary(currentPanel());
}}

function renderPreview(mask) {{
  const panel = currentPanel();
  const names = [panel.xvar, panel.mode === "1d" ? panel.x2var : "", panel.mode === "2d" ? panel.yvar : "", panel.mode === "2d" ? panel.y2var : "", "Q2", "xB", "t", "t_pi0", "rec_minus_t", "rec_minus_t_pi0", "pDet", "passFiducial", "passSamplingFraction", "passExclusivity", "rec_passFiducial", "rec_passSamplingFraction", "rec_passExclusivity"]
    .filter((name, index, arr) => name && columns[name] && arr.indexOf(name) === index)
    .slice(0, 8);
  const table = el("preview");
  table.innerHTML = "";
  const head = document.createElement("tr");
  head.appendChild(document.createElement("th")).textContent = "row";
  for (const name of names) head.appendChild(document.createElement("th")).textContent = byName[name]?.label || name;
  table.appendChild(head);
  let shown = 0;
  for (let i = 0; i < rowCount && shown < 20; i++) {{
    if (!mask[i]) continue;
    const row = document.createElement("tr");
    row.appendChild(document.createElement("td")).textContent = i;
    for (const name of names) row.appendChild(document.createElement("td")).textContent = fmtColumn(name, columns[name][i]);
    table.appendChild(row);
    shown++;
  }}
}}

async function startVisualizer() {{
  await decodeInitialPayloadColumns();
  await init();
}}

startVisualizer().catch(error => {{
  console.error(error);
  setStartupProgress(100, `Could not load visualizer: ${{error.message || error}}`);
  document.body.removeAttribute("aria-busy");
}});
</script>
</body>
</html>
"""


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
