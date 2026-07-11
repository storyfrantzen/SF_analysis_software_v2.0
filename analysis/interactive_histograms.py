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


ROOT_VECTOR_BRANCHES = (
    "selectedRoles",
    "selectedIdx",
    "selectedPid",
    "selectedDet",
    "selectedSector",
    "selectedP",
    "selectedTheta",
    "selectedPhi",
)

ROOT_PREFERRED_BRANCHES = (
    "sourceFileId",
    "sourceEventIndex",
    "runNum",
    "eventNum",
    "helicity",
    "charge",
    "passTopology",
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
)


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
    "protonTheta": "theta_p",
    "protonTheta_deg": "theta_p deg",
    "protonP": "p_p",
    "protonIdx": "proton index",
    "protonSector": "proton sector",
    "gammaIdx": "gamma index",
    "gammaSector": "gamma sector",
    "gamma1Idx": "gamma 1 index",
    "gamma1Sector": "gamma 1 sector",
    "gamma2Idx": "gamma 2 index",
    "gamma2Sector": "gamma 2 sector",
    "electronTheta_deg": "theta_e deg",
    "electronIdx": "electron index",
    "pi0_theta_deg": "theta_pi0 deg",
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
    parser.add_argument("--tree", default="Events", help="ROOT tree name")
    parser.add_argument("--dictionary", type=Path, help="Optional ROOT dictionary shared library")
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
            "Maximum rows embedded in the HTML. ROOT inputs read at most this many rows "
            "by default; use 0 to read all rows."
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
        log(f"Reading NPZ input {args.input}")
        arrays, metadata = load_npz(args.input)
    else:
        arrays, metadata = load_root(
            args.input,
            args.tree,
            args.dictionary,
            args.columns,
            max_events=args.max_events,
        )

    log("Preparing embedded data")
    arrays = add_derived_quantities(arrays)
    arrays = normalize_visual_columns(arrays)
    arrays = rectangular_numeric_and_text(arrays)
    arrays, downsample = downsample_arrays(arrays, args.max_events, args.seed)
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
    tree_name: str,
    dictionary: Path | None,
    requested_columns: list[str] | None,
    max_events: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    log(f"Opening ROOT input {path}")
    loaded_dictionary = load_root_dictionary(ROOT, dictionary)

    root_path = str(path.resolve())
    root_file = ROOT.TFile.Open(root_path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")
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

    read_limit = entries if max_events <= 0 else min(entries, max_events)
    if read_limit < entries:
        log(f"Reading {len(columns)} ROOT columns from first {read_limit} of {entries} rows")
    else:
        log(f"Reading {len(columns)} ROOT columns from {entries} rows")
    raw = read_root_arrays(
        ROOT,
        root_path,
        tree_name,
        columns,
        aliases=object_branch_aliases(available),
        max_events=max_events,
        strict=bool(requested_columns),
    )
    arrays = {name: raw[name] for name in columns}
    arrays.update(extract_selected_particle_quantities(raw))
    metadata = {"format": "root", "tree": tree_name}
    if loaded_dictionary:
        metadata["dictionary"] = str(loaded_dictionary)
    if max_events > 0 and entries > max_events:
        metadata["root_rows_total"] = entries
        metadata["root_rows_read"] = max_events
    return arrays, metadata


def load_root_dictionary(ROOT: Any, dictionary: Path | None) -> Path | None:
    if dictionary is None:
        return None
    candidates = [dictionary]
    if dictionary.suffix == ".dylib":
        candidates.append(dictionary.with_suffix(".so"))
    elif dictionary.suffix == ".so":
        candidates.append(dictionary.with_suffix(".dylib"))

    for candidate in candidates:
        if not candidate.exists():
            continue
        if ROOT.gSystem.Load(str(candidate.resolve())) >= 0:
            if candidate != dictionary:
                print(f"Using ROOT dictionary {candidate} instead of {dictionary}", file=sys.stderr)
            return candidate
        print(f"Warning: could not load ROOT dictionary {candidate}; trying without it", file=sys.stderr)
        return None

    tried = ", ".join(str(candidate) for candidate in candidates)
    print(f"Warning: ROOT dictionary not found ({tried}); continuing without it", file=sys.stderr)
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
        aliases.update({f"event_{field}": f"event.{field}" for field in EVENT_OBJECT_FIELDS})
    if "rec" in available:
        aliases.update({f"rec_{field}": f"rec.{field}" for field in REC_OBJECT_FIELDS})
    if "gen" in available:
        aliases.update({f"gen_{field}": f"gen.{field}" for field in GEN_OBJECT_FIELDS})
    return aliases


def read_root_arrays(
    ROOT: Any,
    root_path: str,
    tree_name: str,
    columns: list[str],
    *,
    aliases: dict[str, str],
    max_events: int,
    strict: bool,
) -> dict[str, Any]:
    remaining = list(columns)
    while remaining:
        try:
            frame = ROOT.RDataFrame(tree_name, root_path)
            for name in remaining:
                if name in aliases:
                    frame = frame.Define(name, aliases[name])
            if max_events > 0:
                frame = frame.Range(max_events)
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
    raise RuntimeError(f"No readable scalar or vector branches found in {tree_name}")


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
    return derived


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
    return normalized


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


def downsample_arrays(
    arrays: dict[str, np.ndarray],
    max_events: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    row_count = next(iter(arrays.values())).shape[0]
    if max_events <= 0 or row_count <= max_events:
        return arrays, {"originalRows": int(row_count), "embeddedRows": int(row_count), "sampled": False}
    rng = random.Random(seed)
    indices = np.asarray(sorted(rng.sample(range(row_count), max_events)), dtype=np.int64)
    sampled = {name: value[indices] for name, value in arrays.items()}
    return sampled, {"originalRows": int(row_count), "embeddedRows": int(max_events), "sampled": True}


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
            display_min = 0.0 if is_wrapped_phi_degree_column(name) else float(np.min(finite))
            display_max = 360.0 if is_wrapped_phi_degree_column(name) else float(np.max(finite))
            variables.append(
                {
                    "name": name,
                    "label": label_for(name),
                    "min": display_min,
                    "max": display_max,
                    "mean": float(np.mean(finite)),
                    "finite": int(finite.size),
                    "integer": bool(is_integer_category(name)),
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
        if unique.size >= 2 and set(unique.tolist()).issubset({1, 2, 3, 4, 5, 6}):
            candidates.append({"name": name, "label": label_for(name)})
    return sorted(candidates, key=lambda item: sort_key(str(item["name"])))


def is_wrapped_phi_degree_column(name: str) -> bool:
    return name.endswith("_deg") and is_wrapped_phi_column_name(name[:-4])


def sort_key(name: str) -> tuple[int, str]:
    preferred = (
        "run",
        "runNum",
        "rec_runNum",
        "gen_runNum",
        "Q2",
        "rec_Q2",
        "gen_Q2",
        "xB",
        "rec_xB",
        "gen_xB",
        "t",
        "t_pi0",
        "minus_t",
        "rec_minus_t",
        "rec_minus_t_pi0",
        "gen_minus_t",
        "protonTheta_deg",
        "gen_protonTheta_deg",
        "electronTheta_deg",
        "gen_electronTheta_deg",
        "pi0_theta_deg",
        "gen_pi0Theta_deg",
        "gen_gamma1Theta_deg",
        "gen_gamma2Theta_deg",
        "trentoPhi_deg",
        "rec_trento_phi_deg",
        "gen_trento_phi_deg",
        "rec_theta_deg",
        "gen_theta_deg",
        "rec_phi_deg",
        "gen_phi_deg",
        "gen_electronP",
        "gen_protonP",
        "gen_gamma1P",
        "gen_gamma2P",
        "gen_pi0P",
        "electronSamplingFraction",
        "electronSamplingFractionECAL",
        "electronSamplingFractionPCAL",
        "electronSamplingFractionECIN",
        "electronSamplingFractionECOUT",
        "passSamplingFraction",
        "rec_electronSamplingFraction",
        "rec_electronSamplingFractionECAL",
        "rec_electronSamplingFractionPCAL",
        "rec_electronSamplingFractionECIN",
        "rec_electronSamplingFractionECOUT",
        "rec_passFiducial",
        "rec_passSamplingFraction",
        "rec_passExclusivity",
        "electronSector",
        "pSector",
        "g1Sector",
        "g2Sector",
        "eIdx",
        "pIdx",
        "g1Idx",
        "g2Idx",
        "rec_trento_phi",
        "gen_trento_phi",
        "m_gg",
        "rec_m_gg",
        "pT_miss",
        "rec_pT_miss",
    )
    return (preferred.index(name) if name in preferred else len(preferred), name)


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
    canonical = lowered.removeprefix("rec_").removeprefix("gen_").replace("_", "")
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
    canonical = name.lower().removeprefix("rec_").removeprefix("gen_").replace("_", "")
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
    return name.startswith("pass") or name.startswith("rec_pass")


def is_index_column(name: str) -> bool:
    return name.endswith("Idx") or name.endswith("Index")


def is_run_number_column(name: str) -> bool:
    return name in {"run", "runNum", "rec_runNum", "gen_runNum"} or name.lower().endswith("runnum")


def is_integer_category(name: str) -> bool:
    return is_run_number_column(name) or is_index_column(name) or "sector" in name.lower() or name.endswith("Det")


def label_for(name: str) -> str:
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    return name.replace("_", " ")


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
  --mark: color-mix(in srgb, Highlight 78%, CanvasText);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 14px/1.42 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{
  display: grid;
  grid-template-columns: minmax(250px, 320px) 1fr;
  min-height: 100vh;
}}
aside {{
  border-right: 1px solid var(--border);
  padding: 14px;
  background: var(--panel);
  overflow: auto;
}}
section {{
  padding: 14px;
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
.control-actions {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
}}
.control-actions button {{
  flex: 0 1 auto;
}}
h1 {{
  font-size: 17px;
  font-weight: 600;
  margin: 0 0 4px;
}}
h2 {{
  font-size: 13px;
  font-weight: 600;
  margin: 18px 0 8px;
}}
.subtle {{ color: var(--muted); font-size: 12px; }}
.row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
label {{ display: grid; gap: 4px; margin: 8px 0; }}
select, input, button {{
  font: inherit;
  color: inherit;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 7px 8px;
  min-width: 0;
}}
button {{ cursor: pointer; }}
button.active {{
  background: var(--accent);
  color: var(--accent-text);
  border-color: var(--accent);
}}
.segmented {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }}
.segmented button {{ flex: 1 1 72px; }}
.chips {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.chip {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 4px 8px;
  background: var(--bg);
}}
.chip input {{ margin: 0; }}
.quick-category {{
  display: grid;
  gap: 6px;
  margin: 8px 0 10px;
}}
.quick-category .chips {{
  max-height: 118px;
  overflow: auto;
  padding-right: 2px;
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
  width: 34px;
  min-width: 34px;
  padding: 7px 0;
  text-align: center;
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
  grid-template-columns: 1fr 82px 82px auto;
  gap: 6px;
  margin: 6px 0;
  align-items: center;
}}
.filter-row input {{ width: 100%; }}
.operation-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 6px;
}}
.operation-grid .row {{ grid-template-columns: 1fr 1fr; }}
.operation-grid button {{ width: 100%; }}
.stats {{
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: 10px;
}}
.stat {{
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 7px 9px;
  min-width: 92px;
}}
.stat strong {{ display: block; font-size: 16px; font-weight: 600; }}
canvas {{
  display: block;
  width: 100%;
  height: min(70vh, 700px);
  min-height: 420px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
}}
.plot-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
}}
.plot-grid.compare {{
  grid-template-columns: repeat(2, minmax(0, 1fr));
}}
.plot-pane.hidden {{ display: none; }}
.plot-head {{
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: baseline;
  margin-bottom: 5px;
}}
.plot-title {{
  font-weight: 600;
  font-size: 13px;
}}
.plot-summary {{
  color: var(--muted);
  font-size: 12px;
  text-align: right;
}}
.plot-grid.compare canvas {{
  height: min(64vh, 660px);
  min-height: 390px;
}}
.hover-info {{
  min-height: 24px;
  margin: -2px 0 8px;
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
@media (max-width: 820px) {{
  main {{ grid-template-columns: 1fr; }}
  aside {{ border-right: 0; border-bottom: 1px solid var(--border); }}
  .control-deck {{ grid-template-columns: 1fr; }}
  #categoryFilters {{ column-width: auto; }}
  .plot-grid.compare {{ grid-template-columns: 1fr; }}
  canvas {{ min-height: 340px; height: 58vh; }}
}}
</style>
</head>
<body>
<main id="app">
  <aside>
    <h1></h1>
    <div class="subtle" id="source"></div>
    <h2>Plot</h2>
    <div class="chips">
      <label class="chip"><input id="compareMode" type="checkbox"> two panels</label>
    </div>
    <div class="segmented" id="panelSelector">
      <button type="button" id="panelA">Panel A</button>
      <button type="button" id="panelB">Panel B</button>
    </div>
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
    <label id="splitLabel">Split by sector <select id="splitVar"></select></label>
    <div class="quick-category" id="quickCategoryBlock">
      <label>Filter topology <select id="quickCategoryFilter"></select></label>
      <div class="chips" id="quickCategoryChips"></div>
      <div class="segmented">
        <button type="button" id="quickCategoryAll">All</button>
        <button type="button" id="quickCategoryNone">None</button>
      </div>
      <div class="subtle" id="quickCategorySummary"></div>
    </div>
    <div class="row">
      <label>X bins <input id="xbins" type="number" min="5" max="400" value="80"></label>
      <label>Y bins <input id="ybins" type="number" min="5" max="300" value="80"></label>
    </div>
    <div class="row">
      <label><span>X min</span><input id="xmin" type="number" step="any"></label>
      <label><span>X max</span><input id="xmax" type="number" step="any"></label>
    </div>
    <div class="row" id="yrange">
      <label><span>Y min</span><input id="ymin" type="number" step="any"></label>
      <label><span>Y max</span><input id="ymax" type="number" step="any"></label>
    </div>
    <div class="row">
      <label><span>X ticks <span id="xtickValue"></span></span><input id="xticks" type="range" min="1" max="40" step="0.5" value="6"></label>
      <label><span>Y ticks <span id="ytickValue"></span></span><input id="yticks" type="range" min="1" max="40" step="0.5" value="6"></label>
    </div>
    <div class="chips">
      <label class="chip"><input id="logz" type="checkbox"> log color</label>
      <label class="chip"><input id="density" type="checkbox"> density</label>
    </div>
  </aside>
  <section>
    <div class="stats">
      <div class="stat"><span class="subtle">selected</span><strong id="selectedCount">0</strong></div>
      <div class="stat"><span class="subtle">embedded</span><strong id="embeddedCount">0</strong></div>
      <div class="stat"><span class="subtle">mean X</span><strong id="meanX">-</strong></div>
      <div class="stat"><span class="subtle">mean Y</span><strong id="meanY">-</strong></div>
      <div class="subtle" id="samplingNote"></div>
    </div>
    <div class="plot-grid" id="plotGrid">
      <div class="plot-pane" id="plotPaneA">
        <div class="plot-head">
          <div class="plot-title">Panel A</div>
          <div class="plot-summary" id="panelSummaryA"></div>
        </div>
        <div class="hover-info" id="hoverInfoA">Hover over a bin to inspect it.</div>
        <canvas id="plotA" width="1200" height="780"></canvas>
      </div>
      <div class="plot-pane hidden" id="plotPaneB">
        <div class="plot-head">
          <div class="plot-title">Panel B</div>
          <div class="plot-summary" id="panelSummaryB"></div>
        </div>
        <div class="hover-info" id="hoverInfoB">Hover over a bin to inspect it.</div>
        <canvas id="plotB" width="1200" height="780"></canvas>
      </div>
    </div>
    <div class="control-deck">
      <div class="control-panel">
        <h2>Range Filters</h2>
        <div class="filter-row">
          <select id="rangeVar"></select>
          <input id="rangeMin" type="number" step="any" placeholder="min">
          <input id="rangeMax" type="number" step="any" placeholder="max">
          <button type="button" id="addRange">Add</button>
        </div>
        <div id="rangeFilters"></div>
      </div>
      <div class="control-panel">
        <h2>Derived Operations</h2>
        <div class="operation-grid">
          <label>Left <select id="opLeft"></select></label>
          <div class="row">
            <label>Operation <select id="opKind">
              <option value="subtract">left - right</option>
              <option value="add">left + right</option>
              <option value="ratio">left / right</option>
              <option value="fractional">(left - right) / right</option>
            </select></label>
            <label>Right <select id="opRight"></select></label>
          </div>
          <button type="button" id="addDerived">Add variable</button>
          <div class="subtle" id="opStatus"></div>
        </div>
      </div>
      <div class="control-panel">
        <h2>Fit</h2>
        <label>Model <select id="fitModel">
          <option value="none">none</option>
          <option value="gaussian">Gaussian</option>
          <option value="linear">linear</option>
          <option value="quadratic">quadratic</option>
        </select></label>
        <div class="subtle" id="fitSummary">No fit</div>
        <h2>Actions</h2>
        <div class="control-actions">
          <button type="button" id="resetFilters">Reset filters</button>
          <button type="button" id="resetRanges">Reset axes</button>
          <button type="button" id="savePng">Save PNG</button>
        </div>
      </div>
      <div class="control-panel" id="textFilterPanel">
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
<script>
const payload = {payload_json};
const columns = {{}};
const textColumns = {{}};
const rowCount = payload.rowCount;
for (const [name, value] of Object.entries(payload.columns)) {{
  if (value && value.dtype === "float32") {{
    const binary = atob(value.data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    columns[name] = new Float32Array(bytes.buffer);
  }} else {{
    textColumns[name] = value;
  }}
}}
const variables = payload.variables;
const byName = Object.fromEntries(variables.map(v => [v.name, v]));
const integerVariables = new Set(variables.filter(v => v.integer).map(v => v.name));
const panelKeys = ["A", "B"];
let activePanel = "A";
let compareMode = false;
let activeRanges = [];
const categoryState = {{}};
const panels = {{
  A: makePanel("A", payload.defaultX, payload.defaultY),
  B: makePanel("B", comparisonDefaultX(), payload.defaultY)
}};

const el = id => document.getElementById(id);
const fmt = value => Number.isFinite(value) ? (Math.abs(value) >= 1000 || Math.abs(value) < 0.01 ? value.toExponential(3) : value.toPrecision(4)) : "-";
const fmtColumn = (name, value) => integerVariables.has(name) && Number.isFinite(value) ? String(Math.round(value)) : fmt(value);
const fmtTickTarget = value => Number.isInteger(value) ? String(value) : value.toFixed(1);

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
    splitVar: "",
    xbins: 80,
    ybins: 80,
    xticks: 6,
    yticks: 6,
    xmin: xInfo.min,
    xmax: xInfo.max,
    ymin: yInfo.min,
    ymax: yInfo.max,
    logz: false,
    density: false,
    fitModel: "none",
    fitSummary: "No fit",
    lastPlot: null,
    stats: {{selected: 0, meanX: NaN, meanY: NaN}}
  }};
}}

function currentPanel() {{
  return panels[activePanel];
}}

function fillSelect(select, selected) {{
  select.innerHTML = "";
  for (const variable of variables) {{
    const option = document.createElement("option");
    option.value = variable.name;
    option.textContent = variable.label;
    select.appendChild(option);
  }}
  select.value = selected;
}}

function fillOverlaySelect(select, selected) {{
  select.innerHTML = "";
  for (const variable of variables) {{
    const option = document.createElement("option");
    option.value = variable.name;
    option.textContent = variable.label;
    select.appendChild(option);
  }}
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
  const variable = {{name, label, min, max, mean: sum / finite, finite}};
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

function fillSectorSelect(selected) {{
  const label = el("splitLabel");
  const select = el("splitVar");
  select.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "none";
  select.appendChild(none);
  for (const split of payload.sectorSplits || []) {{
    const option = document.createElement("option");
    option.value = split.name;
    option.textContent = split.label;
    select.appendChild(option);
  }}
  select.value = selected || "";
  label.style.display = (payload.sectorSplits || []).length ? "" : "none";
}}

function init() {{
  document.querySelector("h1").textContent = payload.title;
  el("source").textContent = payload.source;
  el("embeddedCount").textContent = rowCount.toLocaleString();
  if (payload.downsample.sampled) {{
    el("samplingNote").textContent = `downsampled from ${{payload.downsample.originalRows.toLocaleString()}} rows`;
  }}
  fillSelect(el("rangeVar"), payload.defaultX);
  fillOperationSelects();
  initializeCategoryState();
  renderCategoryFilters();
  renderQuickCategoryOptions();
  renderQuickCategory();
  renderTextFilters();
  attachEvents();
  syncControlsFromPanel();
  update();
}}

function initializeCategoryState() {{
  for (const filter of payload.categoricalFilters) {{
    if (!categoryState[filter.name]) {{
      categoryState[filter.name] = new Set(filter.values.map(value => Number(value)));
    }}
  }}
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
  select.innerHTML = "";
  let currentGroup = "";
  let groupNode = null;
  for (const filter of payload.categoricalFilters) {{
    const group = filter.group || "Other";
    if (group !== currentGroup) {{
      currentGroup = group;
      groupNode = document.createElement("optgroup");
      groupNode.label = group;
      select.appendChild(groupNode);
    }}
    const option = document.createElement("option");
    option.value = filter.name;
    option.textContent = filter.label;
    groupNode.appendChild(option);
  }}
  select.value = payload.categoricalFilters.some(filter => filter.name === previous)
    ? previous
    : payload.categoricalFilters[0].name;
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
  target.innerHTML = "";
  panel.style.display = payload.textFilters.length ? "" : "none";
  for (const filter of payload.textFilters) {{
    const label = document.createElement("label");
    label.textContent = filter.label;
    const input = document.createElement("input");
    input.type = "search";
    input.placeholder = "contains...";
    input.dataset.textFilter = filter.name;
    label.appendChild(input);
    target.appendChild(label);
  }}
}}

function attachEvents() {{
  ["x2var","y2var","splitVar","xbins","ybins","xticks","yticks","xmin","xmax","ymin","ymax","logz","density","fitModel"].forEach(id => {{
    el(id).addEventListener("input", () => {{ readControlsToPanel(); update(); }});
  }});
  el("xvar").addEventListener("change", () => {{ setPanelVariable("x"); update(); }});
  el("yvar").addEventListener("change", () => {{ setPanelVariable("y"); update(); }});
  el("addDerived").addEventListener("click", addDerivedVariable);
  el("compareMode").addEventListener("input", () => {{ compareMode = el("compareMode").checked; update(); }});
  el("panelA").addEventListener("click", () => setActivePanel("A"));
  el("panelB").addEventListener("click", () => setActivePanel("B"));
  el("mode1d").addEventListener("click", () => setMode("1d"));
  el("mode2d").addEventListener("click", () => setMode("2d"));
  el("addXVar").addEventListener("click", () => addAdditionalVariable("x"));
  el("addYVar").addEventListener("click", () => addAdditionalVariable("y"));
  el("removeXVar").addEventListener("click", () => removeAdditionalVariable("x"));
  el("removeYVar").addEventListener("click", () => removeAdditionalVariable("y"));
  el("addRange").addEventListener("click", addRangeFilter);
  el("resetFilters").addEventListener("click", resetFilters);
  el("resetRanges").addEventListener("click", () => {{ resetAxisRanges(currentPanel()); syncControlsFromPanel(); update(); }});
  el("savePng").addEventListener("click", savePng);
  el("quickCategoryFilter").addEventListener("change", renderQuickCategory);
  el("quickCategoryAll").addEventListener("click", () => setCurrentCategoryValues(true));
  el("quickCategoryNone").addEventListener("click", () => setCurrentCategoryValues(false));
  document.querySelectorAll("input[data-text-filter]").forEach(input => input.addEventListener("input", update));
  for (const key of panelKeys) {{
    el("plot" + key).addEventListener("mousemove", event => showHoverInfo(event, key));
    el("plot" + key).addEventListener("mouseleave", () => {{
      hoverElement(key).textContent = "Hover over a bin to inspect it.";
    }});
  }}
  window.addEventListener("resize", update);
}}

function setActivePanel(key) {{
  activePanel = key;
  syncControlsFromPanel();
  update();
}}

function syncControlsFromPanel() {{
  const panel = currentPanel();
  fillSelect(el("xvar"), panel.xvar);
  fillOverlaySelect(el("x2var"), panel.x2var);
  fillSelect(el("yvar"), panel.yvar);
  fillOverlaySelect(el("y2var"), panel.y2var);
  fillSectorSelect(panel.splitVar);
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
  el("fitModel").value = panel.fitModel || "none";
  el("fitSummary").textContent = panel.fitSummary || "No fit";
  el("compareMode").checked = compareMode;
  el("panelA").classList.toggle("active", activePanel === "A");
  el("panelB").classList.toggle("active", activePanel === "B");
  el("mode1d").classList.toggle("active", panel.mode === "1d");
  el("mode2d").classList.toggle("active", panel.mode === "2d");
  el("yAxisControl").style.display = panel.mode === "2d" ? "" : "none";
  const showExtraX = Boolean(panel.x2var);
  const showExtraY = panel.mode === "2d" && Boolean(panel.y2var);
  el("extraXControls").style.display = showExtraX ? "" : "none";
  el("addXVar").style.display = !panel.x2var ? "" : "none";
  el("extraYControls").style.display = showExtraY ? "" : "none";
  el("addYVar").style.display = panel.mode === "2d" && !panel.y2var ? "" : "none";
  el("yrange").style.display = panel.mode === "2d" ? "" : "none";
  el("ybins").closest("label").style.display = panel.mode === "2d" ? "" : "none";
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
  panel.splitVar = el("splitVar").value;
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
  panel.fitModel = el("fitModel").value;
}}

function setMode(next) {{
  currentPanel().mode = next;
  syncControlsFromPanel();
  update();
}}

function setPanelVariable(axis) {{
  const panel = currentPanel();
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
  activeRanges.push({{name, min, max}});
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
    name.textContent = byName[filter.name]?.label || filter.name;
    const min = document.createElement("input");
    min.type = "number";
    min.step = "any";
    min.value = Number.isFinite(filter.min) ? filter.min : "";
    min.addEventListener("input", () => {{ filter.min = parseNumber(min.value); update(); }});
    const max = document.createElement("input");
    max.type = "number";
    max.step = "any";
    max.value = Number.isFinite(filter.max) ? filter.max : "";
    max.addEventListener("input", () => {{ filter.max = parseNumber(max.value); update(); }});
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "x";
    remove.addEventListener("click", () => {{ activeRanges.splice(index, 1); renderRangeFilters(); update(); }});
    row.append(name, min, max, remove);
    target.appendChild(row);
  }});
}}

function resetFilters() {{
  for (const filter of payload.categoricalFilters) {{
    categoryState[filter.name] = new Set(filter.values.map(value => Number(value)));
  }}
  renderQuickCategory();
  renderCategoryFilters();
  document.querySelectorAll("input[data-text-filter]").forEach(input => input.value = "");
  activeRanges = [];
  renderRangeFilters();
  update();
}}

function parseNumber(value) {{
  const number = Number(value);
  return Number.isFinite(number) ? number : NaN;
}}

function selectedMask() {{
  const mask = new Uint8Array(rowCount);
  mask.fill(1);
  for (const filter of payload.categoricalFilters) {{
    const allowed = categoryState[filter.name];
    const values = columns[filter.name];
    if (!allowed || !values) continue;
    for (let i = 0; i < rowCount; i++) {{
      if (mask[i] && !allowed.has(Math.round(values[i]))) mask[i] = 0;
    }}
  }}
  for (const filter of activeRanges) {{
    const values = columns[filter.name];
    if (!values) continue;
    for (let i = 0; i < rowCount; i++) {{
      const value = values[i];
      if (mask[i] && (!Number.isFinite(value) || (Number.isFinite(filter.min) && value < filter.min) || (Number.isFinite(filter.max) && value > filter.max))) mask[i] = 0;
    }}
  }}
  document.querySelectorAll("input[data-text-filter]").forEach(input => {{
    const needle = input.value.trim().toLowerCase();
    if (!needle) return;
    const values = textColumns[input.dataset.textFilter];
    if (!values) return;
    for (let i = 0; i < rowCount; i++) {{
      if (mask[i] && !String(values[i]).toLowerCase().includes(needle)) mask[i] = 0;
    }}
  }});
  return mask;
}}

function update() {{
  readControlsToPanel();
  const mask = selectedMask();
  updatePanelVisibility();
  for (const key of visiblePanelKeys()) {{
    const panel = panels[key];
    if (!columns[panel.xvar]) continue;
    if (panel.mode === "1d") draw1d(panel, mask);
    else draw2d(panel, mask);
  }}
  updateActiveStats();
  renderPreview(mask);
}}

function visiblePanelKeys() {{
  return compareMode ? panelKeys : [activePanel];
}}

function updatePanelVisibility() {{
  el("plotGrid").classList.toggle("compare", compareMode);
  for (const key of panelKeys) {{
    const visible = compareMode || key === activePanel;
    el("plotPane" + key).classList.toggle("hidden", !visible);
  }}
}}

function plotArea(canvas) {{
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(400, Math.floor(rect.width * dpr));
  canvas.height = Math.max(320, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  return {{ctx, width, height, left: 76, right: 22, top: 18, bottom: 62}};
}}

function colors() {{
  const style = getComputedStyle(document.documentElement);
  return {{
    fg: style.getPropertyValue("--fg").trim(),
    muted: style.getPropertyValue("--muted").trim(),
    border: style.getPropertyValue("--border").trim(),
    mark: style.getPropertyValue("--mark").trim(),
    bg: style.getPropertyValue("--bg").trim()
  }};
}}

function drawAxes(ctx, area, xMin, xMax, yMin, yMax, xLabel, yLabel, xTickCount, yTickCount) {{
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
    ctx.textAlign = "center";
    ctx.fillText(fmt(tick), x, area.top + ph + 20);
  }}
  for (const tick of niceTicks(yMin, yMax, yTickCount)) {{
    const y = area.top + ph - (tick - yMin) / (yMax - yMin) * ph;
    ctx.beginPath();
    ctx.moveTo(area.left - 5, y);
    ctx.lineTo(area.left + pw, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(fmt(tick), area.left - 8, y);
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
  ctx.fillText(xLabel, area.left + pw / 2, area.height - 14);
  ctx.save();
  ctx.translate(16, area.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}}

function draw1d(panel, mask) {{
  const splitName = panel.splitVar;
  if (splitName) {{
    draw1dFacets(panel, mask, splitName);
    return;
  }}
  const xName = panel.xvar;
  const x2Name = panel.x2var && panel.x2var !== xName && columns[panel.x2var] ? panel.x2var : "";
  const x = columns[xName];
  const x2 = x2Name ? columns[x2Name] : null;
  const bins = panel.xbins;
  const xMin = panel.xmin;
  const xMax = panel.xmax;
  const counts = new Float64Array(bins);
  const overlayCounts = x2 ? new Float64Array(bins) : null;
  let selected = 0, overlaySelected = 0, sumX = 0;
  for (let i = 0; i < rowCount; i++) {{
    const xv = x[i];
    if (!mask[i] || xMax <= xMin) continue;
    if (Number.isFinite(xv) && xv >= xMin && xv <= xMax) {{
      const bin = Math.min(bins - 1, Math.max(0, Math.floor((xv - xMin) / (xMax - xMin) * bins)));
      counts[bin] += 1;
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
  if (panel.density && selected > 0) {{
    for (let i = 0; i < bins; i++) counts[i] /= selected;
  }}
  if (panel.density && overlayCounts && overlaySelected > 0) {{
    for (let i = 0; i < bins; i++) overlayCounts[i] /= overlaySelected;
  }}
  const maxCount = Math.max(maxOf(counts, 1), overlayCounts ? maxOf(overlayCounts, 1) : 1);
  const canvas = el("plot" + panel.key);
  const area = plotArea(canvas);
  const {{ctx, width, height, left, right, top, bottom}} = area;
  const c = colors();
  ctx.clearRect(0, 0, width, height);
  const pw = width - left - right;
  const ph = height - top - bottom;
  ctx.fillStyle = c.mark;
  for (let i = 0; i < bins; i++) {{
    const barH = counts[i] / maxCount * ph;
    const x0 = left + i / bins * pw;
    const x1 = left + (i + 1) / bins * pw;
    ctx.fillRect(x0, top + ph - barH, Math.max(1, x1 - x0 - 1), barH);
  }}
  if (overlayCounts) {{
    ctx.save();
    ctx.globalAlpha = 0.64;
    ctx.fillStyle = overlayHeatColor(0.82);
    for (let i = 0; i < bins; i++) {{
      const barH = overlayCounts[i] / maxCount * ph;
      const binW = pw / bins;
      const x0 = left + i * binW + binW * 0.2;
      ctx.fillRect(x0, top + ph - barH, Math.max(1, binW * 0.6), barH);
    }}
    ctx.restore();
  }}
  drawAxes(ctx, area, xMin, xMax, 0, maxCount, byName[xName].label, panel.density ? "density" : "counts", panel.xticks, panel.yticks);
  if (x2Name) drawOverlayLegend(ctx, area, byName[xName].label, byName[x2Name].label);
  panel.fitSummary = draw1dFit(ctx, area, panel, counts, xMin, xMax, 0, maxCount);
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
  if (panel.density && selected > 0) {{
    for (let i = 0; i < counts.length; i++) counts[i] /= selected;
  }}
  if (panel.density && overlayCounts && overlaySelected > 0) {{
    for (let i = 0; i < overlayCounts.length; i++) overlayCounts[i] /= overlaySelected;
  }}
  const maxCount = maxOf(counts, 1);
  const overlayMaxCount = overlayCounts ? maxOf(overlayCounts, 1) : 1;
  const canvas = el("plot" + panel.key);
  const area = plotArea(canvas);
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
  drawAxes(ctx, area, xMin, xMax, yMin, yMax, xAxisLabel, yAxisLabel, panel.xticks, panel.yticks);
  if (x2Name || y2Name) drawOverlayLegend(ctx, area, `${{byName[yName].label}} vs ${{byName[xName].label}}`, overlay2dLabel({{xName, x2Name, yName, y2Name}}));
  panel.fitSummary = draw2dFit(ctx, area, panel, mask, x, y, xMin, xMax, yMin, yMax);
  panel.lastPlot = {{
    mode: "2d", area, xName, x2Name, yName, y2Name, xMin, xMax, yMin, yMax,
    xBins, yBins, counts, overlayCounts, selected, overlaySelected, density: panel.density
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
  let totalSelected = 0, totalOverlaySelected = 0, sumXAll = 0;
  for (let sector = 1; sector <= 6; sector++) {{
    const counts = new Float64Array(bins);
    const overlayCounts = x2 ? new Float64Array(bins) : null;
    let selected = 0, overlaySelected = 0, sumX = 0;
    for (let i = 0; i < rowCount; i++) {{
      const xv = x[i];
      if (!mask[i] || Math.round(split[i]) !== sector || xMax <= xMin) continue;
      if (Number.isFinite(xv) && xv >= xMin && xv <= xMax) {{
        const bin = Math.min(bins - 1, Math.max(0, Math.floor((xv - xMin) / (xMax - xMin) * bins)));
        counts[bin] += 1;
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
    facets.push({{sector, counts, overlayCounts, selected, overlaySelected}});
  }}
  if (panel.density) {{
    for (const facet of facets) {{
      if (facet.selected > 0) {{
        for (let i = 0; i < bins; i++) facet.counts[i] /= facet.selected;
      }}
      if (facet.overlayCounts && facet.overlaySelected > 0) {{
        for (let i = 0; i < bins; i++) facet.overlayCounts[i] /= facet.overlaySelected;
      }}
    }}
  }}
  const maxCount = Math.max(
    1,
    ...facets.map(f => maxOf(f.counts, 0)),
    ...facets.map(f => f.overlayCounts ? maxOf(f.overlayCounts, 0) : 0)
  );
  const canvas = el("plot" + panel.key);
  const area = plotArea(canvas);
  const {{ctx, width, height}} = area;
  const c = colors();
  ctx.clearRect(0, 0, width, height);
  const layout = facetLayout(area);
  for (let index = 0; index < facets.length; index++) {{
    const facet = facets[index];
    const facetAreaInfo = panelArea(area, layout, index);
    const pw = facetAreaInfo.width - facetAreaInfo.left - facetAreaInfo.right;
    const ph = facetAreaInfo.height - facetAreaInfo.top - facetAreaInfo.bottom;
    ctx.fillStyle = c.mark;
    for (let i = 0; i < bins; i++) {{
      const barH = facet.counts[i] / maxCount * ph;
      const x0 = facetAreaInfo.left + i / bins * pw;
      const x1 = facetAreaInfo.left + (i + 1) / bins * pw;
      ctx.fillRect(x0, facetAreaInfo.top + ph - barH, Math.max(1, x1 - x0 - 1), barH);
    }}
    if (facet.overlayCounts) {{
      ctx.save();
      ctx.globalAlpha = 0.64;
      ctx.fillStyle = overlayHeatColor(0.82);
      for (let i = 0; i < bins; i++) {{
        const barH = facet.overlayCounts[i] / maxCount * ph;
        const binW = pw / bins;
        const x0 = facetAreaInfo.left + i * binW + binW * 0.2;
        ctx.fillRect(x0, facetAreaInfo.top + ph - barH, Math.max(1, binW * 0.6), barH);
      }}
      ctx.restore();
    }}
    drawAxes(ctx, facetAreaInfo, xMin, xMax, 0, maxCount, byName[xName].label, panel.density ? "density" : "counts", panel.xticks, panel.yticks);
    if (x2Name && index === 0) drawOverlayLegend(ctx, facetAreaInfo, byName[xName].label, byName[x2Name].label);
    drawFacetTitle(ctx, facetAreaInfo, `Sector ${{facet.sector}} (${{facet.selected.toLocaleString()}})`);
    facet.area = facetAreaInfo;
  }}
  panel.lastPlot = {{
    mode: "1d-facet", area, facets, splitName, xName, x2Name, xMin, xMax, bins,
    selected: totalSelected, overlaySelected: totalOverlaySelected, density: panel.density
  }};
  panel.fitSummary = panel.fitModel === "none" ? "No fit" : "Fits are disabled while sector split is active";
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
  let totalSelected = 0, totalOverlaySelected = 0, sumXAll = 0, sumYAll = 0;
  for (let sector = 1; sector <= 6; sector++) {{
    const counts = new Float64Array(xBins * yBins);
    const overlayCounts = (x2 || y2) ? new Float64Array(xBins * yBins) : null;
    let selected = 0, overlaySelected = 0, sumX = 0, sumY = 0;
    for (let i = 0; i < rowCount; i++) {{
      const xv = x[i], yv = y[i];
      if (!mask[i] || Math.round(split[i]) !== sector || xMax <= xMin || yMax <= yMin) continue;
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
    totalSelected += selected;
    totalOverlaySelected += overlaySelected;
    sumXAll += sumX;
    sumYAll += sumY;
    facets.push({{sector, counts, overlayCounts, selected, overlaySelected}});
  }}
  if (panel.density) {{
    for (const facet of facets) {{
      if (facet.selected > 0) {{
        for (let i = 0; i < facet.counts.length; i++) facet.counts[i] /= facet.selected;
      }}
      if (facet.overlayCounts && facet.overlaySelected > 0) {{
        for (let i = 0; i < facet.overlayCounts.length; i++) facet.overlayCounts[i] /= facet.overlaySelected;
      }}
    }}
  }}
  const maxCount = Math.max(1, ...facets.map(f => maxOf(f.counts, 0)));
  const overlayMaxCount = Math.max(1, ...facets.map(f => f.overlayCounts ? maxOf(f.overlayCounts, 0) : 0));
  const canvas = el("plot" + panel.key);
  const area = plotArea(canvas);
  const {{ctx, width, height}} = area;
  ctx.clearRect(0, 0, width, height);
  const layout = facetLayout(area);
  for (let index = 0; index < facets.length; index++) {{
    const facet = facets[index];
    const facetAreaInfo = panelArea(area, layout, index);
    const pw = facetAreaInfo.width - facetAreaInfo.left - facetAreaInfo.right;
    const ph = facetAreaInfo.height - facetAreaInfo.top - facetAreaInfo.bottom;
    for (let yi = 0; yi < yBins; yi++) {{
      for (let xi = 0; xi < xBins; xi++) {{
        const count = facet.counts[yi * xBins + xi];
        if (count <= 0) continue;
        const fraction = panel.logz ? Math.log1p(count) / Math.log1p(maxCount) : count / maxCount;
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
          const fraction = panel.logz ? Math.log1p(count) / Math.log1p(overlayMaxCount) : count / overlayMaxCount;
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
    drawAxes(ctx, facetAreaInfo, xMin, xMax, yMin, yMax, xAxisLabel, yAxisLabel, panel.xticks, panel.yticks);
    if ((x2Name || y2Name) && index === 0) drawOverlayLegend(ctx, facetAreaInfo, `${{byName[yName].label}} vs ${{byName[xName].label}}`, overlay2dLabel({{xName, x2Name, yName, y2Name}}));
    drawFacetTitle(ctx, facetAreaInfo, `Sector ${{facet.sector}} (${{facet.selected.toLocaleString()}})`);
    facet.area = facetAreaInfo;
  }}
  panel.lastPlot = {{
    mode: "2d-facet", area, facets, splitName, xName, x2Name, yName, y2Name, xMin, xMax, yMin, yMax,
    xBins, yBins, selected: totalSelected, overlaySelected: totalOverlaySelected, density: panel.density
  }};
  panel.fitSummary = panel.fitModel === "none" ? "No fit" : "Fits are disabled while sector split is active";
  setPanelStats(panel, totalSelected, sumXAll / totalSelected, sumYAll / totalSelected);
}}

function facetLayout(area) {{
  const cols = area.width >= 900 ? 3 : 2;
  const rows = Math.ceil(6 / cols);
  return {{cols, rows, gapX: 16, gapY: 30, outerLeft: 8, outerRight: 8, outerTop: 4, outerBottom: 8}};
}}

function panelArea(area, layout, index) {{
  const cellW = (area.width - layout.outerLeft - layout.outerRight - (layout.cols - 1) * layout.gapX) / layout.cols;
  const cellH = (area.height - layout.outerTop - layout.outerBottom - (layout.rows - 1) * layout.gapY) / layout.rows;
  const col = index % layout.cols;
  const row = Math.floor(index / layout.cols);
  const cellLeft = layout.outerLeft + col * (cellW + layout.gapX);
  const cellTop = layout.outerTop + row * (cellH + layout.gapY);
  const miniLeft = 52, miniRight = 8, miniTop = 24, miniBottom = 40;
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

function overlay2dLabel(plot) {{
  const xName = plot.x2Name || plot.xName;
  const yName = plot.y2Name || plot.yName;
  return `${{byName[yName]?.label || yName}} vs ${{byName[xName]?.label || xName}}`;
}}

function draw1dFit(ctx, area, panel, counts, xMin, xMax, yMin, yMax) {{
  const model = panel.fitModel || "none";
  if (model === "none") return "No fit";
  const xs = [];
  const ys = [];
  const binWidth = (xMax - xMin) / counts.length;
  for (let i = 0; i < counts.length; i++) {{
    const y = counts[i];
    if (!Number.isFinite(y)) continue;
    xs.push(xMin + (i + 0.5) * binWidth);
    ys.push(y);
  }}
  if (xs.length < 3) return "Not enough bins for fit";
  let fit = null;
  if (model === "gaussian") fit = gaussianMomentFit(xs, ys);
  else fit = polynomialFit(xs, ys, model === "quadratic" ? 2 : 1);
  if (!fit) return "Fit failed";
  drawFitCurve(ctx, area, xMin, xMax, yMin, yMax, fit.predict);
  return fit.summary;
}}

function draw2dFit(ctx, area, panel, mask, xValues, yValues, xMin, xMax, yMin, yMax) {{
  const model = panel.fitModel || "none";
  if (model === "none") return "No fit";
  if (model === "gaussian") return "Gaussian fit is available for 1D histograms";
  const xs = [];
  const ys = [];
  for (let i = 0; i < rowCount; i++) {{
    const x = xValues[i];
    const y = yValues[i];
    if (!mask[i] || !Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < xMin || x > xMax || y < yMin || y > yMax) continue;
    xs.push(x);
    ys.push(y);
  }}
  if (xs.length < (model === "quadratic" ? 3 : 2)) return "Not enough selected points for fit";
  const fit = polynomialFit(xs, ys, model === "quadratic" ? 2 : 1);
  if (!fit) return "Fit failed";
  drawFitCurve(ctx, area, xMin, xMax, yMin, yMax, fit.predict);
  return `${{fit.summary}}; n=${{xs.length.toLocaleString()}}`;
}}

function drawFitCurve(ctx, area, xMin, xMax, yMin, yMax, predict) {{
  const c = colors();
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  ctx.save();
  ctx.strokeStyle = c.fg;
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 4]);
  ctx.beginPath();
  let started = false;
  for (let step = 0; step <= 160; step++) {{
    const x = xMin + (xMax - xMin) * step / 160;
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

function polynomialFit(xs, ys, degree) {{
  const n = degree + 1;
  const matrix = Array.from({{length: n}}, () => Array(n).fill(0));
  const rhs = Array(n).fill(0);
  for (let i = 0; i < xs.length; i++) {{
    const x = xs[i];
    const y = ys[i];
    const powers = [1];
    for (let p = 1; p <= degree * 2; p++) powers[p] = powers[p - 1] * x;
    for (let row = 0; row < n; row++) {{
      rhs[row] += y * powers[row];
      for (let col = 0; col < n; col++) matrix[row][col] += powers[row + col];
    }}
  }}
  const coeff = solveLinearSystem(matrix, rhs);
  if (!coeff) return null;
  const predict = x => coeff.reduce((sum, value, power) => sum + value * Math.pow(x, power), 0);
  const summary = degree === 1
    ? `linear: y=${{fmt(coeff[1])}}x + ${{fmt(coeff[0])}}`
    : `quadratic: y=${{fmt(coeff[2])}}x^2 + ${{fmt(coeff[1])}}x + ${{fmt(coeff[0])}}`;
  return {{predict, summary, coeff}};
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
  return {{
    predict,
    summary: `Gaussian: mu=${{fmt(mean)}}, sigma=${{fmt(sigma)}}, A=${{fmt(amplitude)}}`
  }};
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
  const plots = keys.map(key => ({{
    key,
    canvas: el("plot" + key),
    title: `Panel ${{key}}`,
    summary: el("panelSummary" + key).textContent || ""
  }}));
  if (!plots.length) return;
  const horizontal = compareMode && plots.length > 1;
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

function setHoverText(key, text) {{
  hoverElement(key).textContent = text;
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
    setHoverText(key, "Hover over a bin to inspect it.");
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
      const overlayText = lastPlot.x2Name ? `; ${{byName[lastPlot.x2Name].label}} ${{label}}=${{fmt(overlayValue)}}; overlay sector selected=${{facet.overlaySelected.toLocaleString()}}` : "";
      setHoverText(key, `Panel ${{key}}; sector ${{facet.sector}}; ${{byName[lastPlot.xName].label}} [${{fmt(x0)}}, ${{fmt(x1)}}): ${{label}}=${{fmt(value)}}${{overlayText}}; bin=${{bin + 1}}/${{lastPlot.bins}}; sector selected=${{facet.selected.toLocaleString()}}`);
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
    const overlayText = facet.overlayCounts ? `; ${{overlay2dLabel(lastPlot)}} ${{label}}=${{fmt(overlayValue)}}; overlay sector selected=${{facet.overlaySelected.toLocaleString()}}` : "";
    setHoverText(key, `Panel ${{key}}; sector ${{facet.sector}}; ${{byName[lastPlot.yName].label}} [${{fmt(y0)}}, ${{fmt(y1)}}), ${{byName[lastPlot.xName].label}} [${{fmt(x0)}}, ${{fmt(x1)}}): ${{label}}=${{fmt(value)}}${{overlayText}}; bin=(${{xi + 1}}, ${{yi + 1}}); sector selected=${{facet.selected.toLocaleString()}}`);
    return;
  }}
  setHoverText(key, "Hover over a bin to inspect it.");
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

function setPanelStats(panel, selected, meanX, meanY) {{
  panel.stats = {{selected, meanX, meanY}};
  const overlayLabel = panel.mode === "1d" && panel.x2var && panel.x2var !== panel.xvar && byName[panel.x2var]
      ? ` + ${{byName[panel.x2var].label}}`
      : "";
  const overlay2d = panel.mode === "2d" && ((panel.x2var && panel.x2var !== panel.xvar && byName[panel.x2var]) || (panel.y2var && panel.y2var !== panel.yvar && byName[panel.y2var]))
    ? ` + ${{overlay2dLabel({{xName: panel.xvar, x2Name: panel.x2var, yName: panel.yvar, y2Name: panel.y2var}})}}`
    : "";
  const yLabel = panel.mode === "2d" ? `${{byName[panel.yvar]?.label || panel.yvar}} vs ` : "";
  el("panelSummary" + panel.key).textContent = `${{yLabel}}${{byName[panel.xvar]?.label || panel.xvar}}${{panel.mode === "1d" ? overlayLabel : overlay2d}}; selected ${{selected.toLocaleString()}}`;
}}

function updateActiveStats() {{
  const stats = currentPanel().stats;
  el("selectedCount").textContent = stats.selected.toLocaleString();
  el("meanX").textContent = fmt(stats.meanX);
  el("meanY").textContent = fmt(stats.meanY);
  el("fitSummary").textContent = currentPanel().fitSummary || "No fit";
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

init();
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
