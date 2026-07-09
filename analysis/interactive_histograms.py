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


DISPLAY_NAMES = {
    "Q2": "Q2",
    "rec_Q2": "REC Q2",
    "gen_Q2": "GEN Q2",
    "xB": "xB",
    "rec_xB": "REC xB",
    "gen_xB": "GEN xB",
    "t": "-t",
    "minus_t": "-t",
    "rec_minus_t": "REC -t",
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
    "passSamplingFraction": "sampling fraction",
    "passExclusivity": "loose exclusivity",
    "protonTheta": "theta_p",
    "protonTheta_deg": "theta_p deg",
    "protonP": "p_p",
    "electronTheta_deg": "theta_e deg",
    "pi0_theta_deg": "theta_pi0 deg",
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
        help="Maximum rows embedded in the HTML after deterministic downsampling.",
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
        arrays, metadata = load_npz(args.input)
    else:
        arrays, metadata = load_root(args.input, args.tree, args.dictionary, args.columns)

    arrays = add_derived_quantities(arrays)
    arrays = rectangular_numeric_and_text(arrays)
    arrays, downsample = downsample_arrays(arrays, args.max_events, args.seed)
    payload = build_payload(args.input, arrays, metadata, downsample, args.title)
    html = render_html(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Rows embedded: {payload['rowCount']}")
    print(f"Variables: {len(payload['variables'])}")
    print(f"Wrote {args.output}")
    return 0


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
) -> tuple[dict[str, Any], dict[str, Any]]:
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    loaded_dictionary = load_root_dictionary(ROOT, dictionary)

    root_path = str(path.resolve())
    root_file = ROOT.TFile.Open(root_path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find tree {tree_name} in {root_path}")

    branches = {branch.GetName(): branch for branch in tree.GetListOfBranches()}
    available = set(branches)
    if requested_columns:
        columns = [name for name in requested_columns if name in available]
        missing = [name for name in requested_columns if name not in available]
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
    root_file.Close()

    raw = read_root_arrays(ROOT, root_path, tree_name, columns, strict=bool(requested_columns))
    arrays = {name: raw[name] for name in columns}
    arrays.update(extract_selected_particle_quantities(raw))
    metadata = {"format": "root", "tree": tree_name}
    if loaded_dictionary:
        metadata["dictionary"] = str(loaded_dictionary)
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


def read_root_arrays(
    ROOT: Any,
    root_path: str,
    tree_name: str,
    columns: list[str],
    *,
    strict: bool,
) -> dict[str, Any]:
    remaining = list(columns)
    while remaining:
        try:
            return ROOT.RDataFrame(tree_name, root_path).AsNumpy(remaining)
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
    for role in ("electron", "proton", "gamma"):
        output[f"{role}P"] = np.full(size, np.nan, dtype=float)
        output[f"{role}Theta"] = np.full(size, np.nan, dtype=float)
        output[f"{role}Phi"] = np.full(size, np.nan, dtype=float)
        output[f"{role}Det"] = np.full(size, -999, dtype=np.int64)
    selected_det = raw.get("selectedDet")
    for row in range(size):
        row_roles = vector_to_list(roles[row])
        row_p = vector_to_list(raw["selectedP"][row])
        row_theta = vector_to_list(raw["selectedTheta"][row])
        row_phi = vector_to_list(raw["selectedPhi"][row])
        row_det = vector_to_list(selected_det[row]) if selected_det is not None else []
        seen: set[str] = set()
        for index, role_value in enumerate(row_roles):
            role = str(role_value)
            if role not in ("electron", "proton", "gamma") or role in seen:
                continue
            seen.add(role)
            if index < len(row_p):
                output[f"{role}P"][row] = as_float(row_p[index])
            if index < len(row_theta):
                output[f"{role}Theta"][row] = as_float(row_theta[index])
            if index < len(row_phi):
                output[f"{role}Phi"][row] = as_float(row_phi[index])
            if index < len(row_det):
                output[f"{role}Det"][row] = int(as_float(row_det[index]))
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
                    derived[f"{source}_deg"] = np.asarray(values, dtype=float) * 180.0 / math.pi
    if "rec_proton_detector" in derived and "pDet" not in derived:
        derived["pDet"] = derived["rec_proton_detector"]
    if "rec_selected" in derived:
        derived["rec_not_selected"] = ~np.asarray(derived["rec_selected"], dtype=bool)
    return derived


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

    for name in sorted(arrays, key=sort_key):
        values = arrays[name]
        if values.dtype.kind in "biuf":
            numeric = numeric_array(values)
            finite = numeric[np.isfinite(numeric)]
            if finite.size == 0:
                continue
            variables.append(
                {
                    "name": name,
                    "label": label_for(name),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "finite": int(finite.size),
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

    preferred_x = first_present(
        variables,
        ("rec_minus_t", "t", "minus_t", "gen_minus_t", "Q2", "rec_Q2"),
    )
    preferred_y = first_present(
        variables,
        ("protonTheta_deg", "theta_p_deg", "protonTheta", "rec_Q2", "Q2", "xB"),
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
        "defaultX": preferred_x,
        "defaultY": preferred_y,
    }


def sort_key(name: str) -> tuple[int, str]:
    preferred = (
        "Q2",
        "rec_Q2",
        "gen_Q2",
        "xB",
        "rec_xB",
        "gen_xB",
        "t",
        "minus_t",
        "rec_minus_t",
        "gen_minus_t",
        "protonTheta_deg",
        "protonTheta",
        "trentoPhi",
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
    unique = np.unique(finite.astype(np.int64) if integers else finite)
    if 1 < unique.size <= 12 and (integers or name.startswith("pass") or name.endswith("Det")):
        labels = [category_label(name, item) for item in unique.tolist()]
        return {
            "name": name,
            "label": label_for(name),
            "values": [int(item) if float(item).is_integer() else float(item) for item in unique.tolist()],
            "labels": labels,
        }
    return None


def category_label(name: str, value: Any) -> str:
    if name in {"pDet", "rec_proton_detector", "protonDet"}:
        return {1: "FD", 2: "CD", 0: "FT", -999: "missing"}.get(int(value), str(value))
    if name.startswith("pass") or name in {"rec_selected", "rec_not_selected", "passTopology"}:
        return "pass" if int(value) == 1 else "fail"
    return str(value)


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
  grid-template-columns: minmax(260px, 340px) 1fr;
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
.filter-row {{
  display: grid;
  grid-template-columns: 1fr 82px 82px auto;
  gap: 6px;
  margin: 6px 0;
  align-items: center;
}}
.filter-row input {{ width: 100%; }}
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
  height: min(72vh, 720px);
  min-height: 420px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
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
    <div class="segmented">
      <button type="button" id="mode1d">1D</button>
      <button type="button" id="mode2d">2D</button>
    </div>
    <label>X <select id="xvar"></select></label>
    <label id="ylabel">Y <select id="yvar"></select></label>
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
    <div class="chips">
      <label class="chip"><input id="logz" type="checkbox"> log color</label>
      <label class="chip"><input id="density" type="checkbox"> density</label>
    </div>
    <h2>Category Filters</h2>
    <div id="categoryFilters"></div>
    <h2>Range Filters</h2>
    <div class="filter-row">
      <select id="rangeVar"></select>
      <input id="rangeMin" type="number" step="any" placeholder="min">
      <input id="rangeMax" type="number" step="any" placeholder="max">
      <button type="button" id="addRange">Add</button>
    </div>
    <div id="rangeFilters"></div>
    <h2>Text Filters</h2>
    <div id="textFilters"></div>
    <div class="segmented">
      <button type="button" id="resetFilters">Reset filters</button>
      <button type="button" id="resetRanges">Reset axes</button>
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
    <canvas id="plot" width="1200" height="780"></canvas>
    <div class="table-wrap"><table id="preview"></table></div>
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
let mode = "2d";
let activeRanges = [];

const el = id => document.getElementById(id);
const fmt = value => Number.isFinite(value) ? (Math.abs(value) >= 1000 || Math.abs(value) < 0.01 ? value.toExponential(3) : value.toPrecision(4)) : "-";

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

function init() {{
  document.querySelector("h1").textContent = payload.title;
  el("source").textContent = payload.source;
  el("embeddedCount").textContent = rowCount.toLocaleString();
  if (payload.downsample.sampled) {{
    el("samplingNote").textContent = `downsampled from ${{payload.downsample.originalRows.toLocaleString()}} rows`;
  }}
  fillSelect(el("xvar"), payload.defaultX);
  fillSelect(el("yvar"), payload.defaultY);
  fillSelect(el("rangeVar"), payload.defaultX);
  renderCategoryFilters();
  renderTextFilters();
  setMode("2d");
  resetAxisRanges();
  attachEvents();
  update();
}}

function renderCategoryFilters() {{
  const target = el("categoryFilters");
  target.innerHTML = "";
  for (const filter of payload.categoricalFilters) {{
    const block = document.createElement("div");
    const title = document.createElement("div");
    title.className = "subtle";
    title.textContent = filter.label;
    const chips = document.createElement("div");
    chips.className = "chips";
    filter.values.forEach((value, index) => {{
      const label = document.createElement("label");
      label.className = "chip";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = true;
      input.dataset.filter = filter.name;
      input.dataset.value = String(value);
      label.appendChild(input);
      label.appendChild(document.createTextNode(filter.labels[index]));
      chips.appendChild(label);
    }});
    block.appendChild(title);
    block.appendChild(chips);
    target.appendChild(block);
  }}
}}

function renderTextFilters() {{
  const target = el("textFilters");
  target.innerHTML = "";
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
  ["xvar","yvar","xbins","ybins","xmin","xmax","ymin","ymax","logz","density"].forEach(id => el(id).addEventListener("input", update));
  el("xvar").addEventListener("change", () => {{ setRangeInputs("x"); update(); }});
  el("yvar").addEventListener("change", () => {{ setRangeInputs("y"); update(); }});
  el("mode1d").addEventListener("click", () => setMode("1d"));
  el("mode2d").addEventListener("click", () => setMode("2d"));
  el("addRange").addEventListener("click", addRangeFilter);
  el("resetFilters").addEventListener("click", resetFilters);
  el("resetRanges").addEventListener("click", () => {{ resetAxisRanges(); update(); }});
  document.querySelectorAll("input[data-filter], input[data-text-filter]").forEach(input => input.addEventListener("input", update));
  window.addEventListener("resize", update);
}}

function setMode(next) {{
  mode = next;
  el("mode1d").classList.toggle("active", mode === "1d");
  el("mode2d").classList.toggle("active", mode === "2d");
  el("ylabel").style.display = mode === "2d" ? "" : "none";
  el("yrange").style.display = mode === "2d" ? "" : "none";
  el("ybins").closest("label").style.display = mode === "2d" ? "" : "none";
  update();
}}

function setRangeInputs(axis) {{
  const variable = byName[el(axis + "var").value];
  el(axis + "min").value = variable ? variable.min : "";
  el(axis + "max").value = variable ? variable.max : "";
}}

function resetAxisRanges() {{
  setRangeInputs("x");
  setRangeInputs("y");
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
  document.querySelectorAll("input[data-filter]").forEach(input => input.checked = true);
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
  const groups = {{}};
  document.querySelectorAll("input[data-filter]").forEach(input => {{
    if (!groups[input.dataset.filter]) groups[input.dataset.filter] = new Set();
    if (input.checked) groups[input.dataset.filter].add(Number(input.dataset.value));
  }});
  for (const [name, allowed] of Object.entries(groups)) {{
    const values = columns[name];
    if (!values) continue;
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
  if (!columns[el("xvar").value]) return;
  const mask = selectedMask();
  if (mode === "1d") draw1d(mask);
  else draw2d(mask);
  renderPreview(mask);
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
  return {{ctx, width, height, left: 58, right: 18, top: 18, bottom: 50}};
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

function drawAxes(ctx, area, xMin, xMax, yMin, yMax, xLabel, yLabel) {{
  const c = colors();
  const pw = area.width - area.left - area.right;
  const ph = area.height - area.top - area.bottom;
  ctx.strokeStyle = c.border;
  ctx.fillStyle = c.muted;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(area.left, area.top);
  ctx.lineTo(area.left, area.top + ph);
  ctx.lineTo(area.left + pw, area.top + ph);
  ctx.stroke();
  ctx.textAlign = "center";
  ctx.fillText(xLabel, area.left + pw / 2, area.height - 12);
  ctx.save();
  ctx.translate(14, area.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
  ctx.textAlign = "left";
  ctx.fillText(fmt(xMin), area.left, area.top + ph + 18);
  ctx.textAlign = "right";
  ctx.fillText(fmt(xMax), area.left + pw, area.top + ph + 18);
  ctx.textAlign = "right";
  ctx.fillText(fmt(yMin), area.left - 6, area.top + ph);
  ctx.fillText(fmt(yMax), area.left - 6, area.top + 8);
}}

function draw1d(mask) {{
  const xName = el("xvar").value;
  const x = columns[xName];
  const bins = clamp(Number(el("xbins").value) || 80, 5, 400);
  const xMin = parseNumber(el("xmin").value);
  const xMax = parseNumber(el("xmax").value);
  const counts = new Float64Array(bins);
  let selected = 0, sumX = 0;
  for (let i = 0; i < rowCount; i++) {{
    const xv = x[i];
    if (!mask[i] || !Number.isFinite(xv) || xv < xMin || xv > xMax || xMax <= xMin) continue;
    const bin = Math.min(bins - 1, Math.max(0, Math.floor((xv - xMin) / (xMax - xMin) * bins)));
    counts[bin] += 1;
    selected++;
    sumX += xv;
  }}
  if (el("density").checked && selected > 0) {{
    for (let i = 0; i < bins; i++) counts[i] /= selected;
  }}
  const maxCount = maxOf(counts, 1);
  const canvas = el("plot");
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
  drawAxes(ctx, area, xMin, xMax, 0, maxCount, byName[xName].label, el("density").checked ? "density" : "counts");
  updateStats(selected, sumX / selected, NaN);
}}

function draw2d(mask) {{
  const xName = el("xvar").value;
  const yName = el("yvar").value;
  const x = columns[xName];
  const y = columns[yName];
  const xBins = clamp(Number(el("xbins").value) || 80, 5, 400);
  const yBins = clamp(Number(el("ybins").value) || 80, 5, 300);
  const xMin = parseNumber(el("xmin").value);
  const xMax = parseNumber(el("xmax").value);
  const yMin = parseNumber(el("ymin").value);
  const yMax = parseNumber(el("ymax").value);
  const counts = new Float64Array(xBins * yBins);
  let selected = 0, sumX = 0, sumY = 0;
  for (let i = 0; i < rowCount; i++) {{
    const xv = x[i], yv = y[i];
    if (!mask[i] || !Number.isFinite(xv) || !Number.isFinite(yv) || xv < xMin || xv > xMax || yv < yMin || yv > yMax || xMax <= xMin || yMax <= yMin) continue;
    const xi = Math.min(xBins - 1, Math.max(0, Math.floor((xv - xMin) / (xMax - xMin) * xBins)));
    const yi = Math.min(yBins - 1, Math.max(0, Math.floor((yv - yMin) / (yMax - yMin) * yBins)));
    counts[yi * xBins + xi] += 1;
    selected++;
    sumX += xv;
    sumY += yv;
  }}
  if (el("density").checked && selected > 0) {{
    for (let i = 0; i < counts.length; i++) counts[i] /= selected;
  }}
  const maxCount = maxOf(counts, 1);
  const canvas = el("plot");
  const area = plotArea(canvas);
  const {{ctx, width, height, left, right, top, bottom}} = area;
  ctx.clearRect(0, 0, width, height);
  const pw = width - left - right;
  const ph = height - top - bottom;
  for (let yi = 0; yi < yBins; yi++) {{
    for (let xi = 0; xi < xBins; xi++) {{
      const count = counts[yi * xBins + xi];
      if (count <= 0) continue;
      const fraction = el("logz").checked ? Math.log1p(count) / Math.log1p(maxCount) : count / maxCount;
      ctx.fillStyle = heatColor(fraction);
      const x0 = left + xi / xBins * pw;
      const x1 = left + (xi + 1) / xBins * pw;
      const y0 = top + ph - (yi + 1) / yBins * ph;
      const y1 = top + ph - yi / yBins * ph;
      ctx.fillRect(x0, y0, Math.ceil(x1 - x0), Math.ceil(y1 - y0));
    }}
  }}
  drawAxes(ctx, area, xMin, xMax, yMin, yMax, byName[xName].label, byName[yName].label);
  updateStats(selected, sumX / selected, sumY / selected);
}}

function heatColor(t) {{
  const hue = 225 - 175 * t;
  const light = 92 - 45 * t;
  return `hsl(${{hue}} 78% ${{light}}%)`;
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

function updateStats(selected, meanX, meanY) {{
  el("selectedCount").textContent = selected.toLocaleString();
  el("meanX").textContent = fmt(meanX);
  el("meanY").textContent = fmt(meanY);
}}

function renderPreview(mask) {{
  const names = [el("xvar").value, el("yvar").value, "Q2", "xB", "t", "rec_minus_t", "pDet", "passFiducial", "passSamplingFraction", "passExclusivity"]
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
    for (const name of names) row.appendChild(document.createElement("td")).textContent = fmt(columns[name][i]);
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
