#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.root_trees import resolve  # noqa: E402
from eppi0.exclusivity_models import estimate_model  # noqa: E402
from particle_kinematics_diagnostics import (  # noqa: E402
    BANK_ORDER,
    banks_with_content,
    derive_event_arrays,
    discover_particle_prefixes,
    make_particle,
    render_angular_coverage_report,
    render_report,
    required_branches,
    summary_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a generic multipage PDF of particle kinematics and selected detector-bank "
            "coordinates from a ROOT tree. Particle roles and unavailable banks are discovered "
            "from scalar branch names."
        )
    )
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--tree", default="sEvents", help="ROOT tree (default: sEvents)")
    parser.add_argument(
        "--particle",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "Particle branch prefix, such as electron, proton, gamma1, or row for an "
            "unprefixed sParticles tree. May be repeated; default discovers every particle."
        ),
    )
    parser.add_argument(
        "--bank",
        action="append",
        choices=("all", "auto", *BANK_ORDER),
        default=[],
        help=(
            "Page group to include; may be repeated. Default auto includes event/DIS/kinematics "
            "and each detector bank with available coordinates."
        ),
    )
    parser.add_argument("--where", help="Optional ROOT RDataFrame filter expression")
    parser.add_argument("--selection-mask", type=Path, help="Boolean NPY mask aligned to unfiltered input rows")
    parser.add_argument("--max-rows", type=int, help="Read at most this many rows after --where")
    parser.add_argument(
        "--angular-only",
        action="store_true",
        help="Write one full-page theta-versus-phi coverage plot for each requested particle",
    )
    parser.add_argument(
        "--peak-window",
        metavar="BRANCH",
        help=(
            "Fit a signal-plus-background peak in this scalar branch and retain the fitted "
            "signal window. Requires --peak-search."
        ),
    )
    parser.add_argument(
        "--peak-search",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Broad interval used to locate and fit --peak-window",
    )
    parser.add_argument(
        "--peak-expected",
        type=float,
        help="Expected peak vicinity used only to seed and validate the floating fitted center",
    )
    parser.add_argument(
        "--peak-n-sigma",
        type=float,
        default=3.0,
        help="Gaussian-equivalent fitted signal containment in standard deviations (default: 3)",
    )
    parser.add_argument(
        "--peak-maximum-center-deviation",
        type=float,
        help="Reject a fit farther than this from --peak-expected",
    )
    parser.add_argument(
        "--peak-maximum-sigma",
        type=float,
        help="Reject a fitted core wider than this value",
    )
    parser.add_argument(
        "--peak-minimum-events",
        type=int,
        default=200,
        help="Minimum finite entries in the broad search interval (default: 200)",
    )
    parser.add_argument("--beam-energy", type=float, help="Beam energy in GeV, used to derive y when absent")
    parser.add_argument("--dictionary", type=Path, help="Optional ROOT dictionary shared library")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="JSON sidecar; defaults beside the PDF")
    parser.add_argument("--hash-input", action="store_true", help="SHA256 the ROOT input")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.selection_mask is not None and args.where:
        raise ValueError("--selection-mask cannot be combined with --where because filtered rows no longer align")
    validate_peak_arguments(args)

    tree_name, input_rows, available = inspect_tree(args.input_root, args.tree, args.dictionary)
    prefixes = selected_prefixes(args.particle, available)
    particles = [make_particle(prefix, available) for prefix in prefixes]
    banks = ["kinematics"] if args.angular_only else selected_banks(args.bank, particles, available)
    branches = required_branches(particles, banks, available)
    peak_value_source = None
    if args.peak_window:
        peak_inputs, peak_value_source = peak_input_branches(
            args.peak_window,
            available,
            args.beam_energy,
        )
        branches.extend(name for name in peak_inputs if name not in branches)
    if not branches:
        raise RuntimeError("No scalar branches are available for the requested particles and banks")
    arrays, rows_after_filter = read_arrays(
        args.input_root,
        tree_name,
        branches,
        args.where,
        args.max_rows,
    )
    selection = load_selection(args.selection_mask, input_rows, rows_after_filter, args.max_rows)
    arrays = {name: values[selection] for name, values in arrays.items()}
    rows_before_peak = array_rows(arrays)
    peak_window = None
    if args.peak_window:
        derived_arrays = derive_event_arrays(arrays, args.beam_energy)
        peak_selection, peak_window = fit_peak_window(
            derived_arrays[args.peak_window],
            branch=args.peak_window,
            value_source=peak_value_source,
            search=tuple(args.peak_search),
            expected_center=args.peak_expected,
            n_sigma=args.peak_n_sigma,
            maximum_center_deviation=args.peak_maximum_center_deviation,
            maximum_sigma=args.peak_maximum_sigma,
            minimum_events=args.peak_minimum_events,
        )
        arrays = {name: values[peak_selection] for name, values in arrays.items()}
    plotted_rows = array_rows(arrays)

    provenance = [
        f"Input ROOT: {args.input_root.resolve()}",
        f"Tree: {tree_name}; input rows: {input_rows:,}; plotted rows: {plotted_rows:,}",
        f"Filter: {args.where or 'none'}",
        f"Selection mask: {args.selection_mask.resolve() if args.selection_mask else 'none'}",
        peak_provenance(peak_window),
        f"Maximum rows: {args.max_rows if args.max_rows is not None else 'none'}",
    ]
    if args.angular_only:
        pages, page_records = render_angular_coverage_report(
            args.output,
            arrays,
            particles,
            label=args.label,
            provenance=provenance,
        )
    else:
        pages, page_records = render_report(
            args.output,
            arrays,
            particles,
            banks,
            label=args.label,
            provenance=provenance,
            beam_energy=args.beam_energy,
        )
    summary = summary_for(arrays, particles, banks, page_records)
    summary.update(
        {
            "schema_version": 2,
            "label": args.label,
            "input_root": file_record(args.input_root, args.hash_input),
            "tree": tree_name,
            "input_rows": input_rows,
            "rows_after_filter_and_limit": rows_after_filter,
            "rows_before_peak_window": rows_before_peak,
            "selection_mask": file_record(args.selection_mask, True) if args.selection_mask else None,
            "where": args.where,
            "peak_window": peak_window,
            "max_rows": args.max_rows,
            "report_mode": "angular-only" if args.angular_only else "full",
            "beam_energy_GeV": args.beam_energy,
            "available_scalar_branches": available,
            "requested_particle_prefixes": prefixes,
            "requested_branches": branches,
            "angle_convention": "particle and CVT theta/phi branches are radians and are plotted in degrees",
            "output_pdf": str(args.output.resolve()),
        }
    )
    summary_path = args.summary or args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)

    print(f"Tree: {tree_name}")
    print(f"Input rows: {input_rows}")
    print(f"Plotted rows: {summary['rows']}")
    if peak_window:
        print(
            f"Peak window {peak_window['branch']}: center={peak_window['center']:.8g}, "
            f"sigma={peak_window['sigma']:.8g}, "
            f"range=[{peak_window['lower']:.8g}, {peak_window['upper']:.8g}], "
            f"retained={peak_window['retained_rows']}/{peak_window['candidate_rows']}"
        )
    print("Particles: " + ", ".join(particle.name for particle in particles))
    for particle in particles:
        print(f"  {particle.name} banks: {', '.join(banks_with_content(particle))}")
    print("Requested groups: " + ", ".join(banks))
    print(f"PDF pages: {pages}")
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {summary_path.resolve()}")
    return 0


def selected_prefixes(requested: list[str], available: list[str]) -> list[str]:
    discovered = discover_particle_prefixes(available)
    if not requested:
        if not discovered:
            raise RuntimeError("No particle branch triplets (P/Theta/Phi or p/theta/phi) were found")
        return discovered
    prefixes = ["" if item in {"row", "unprefixed"} else item for item in requested]
    return list(dict.fromkeys(prefixes))


def selected_banks(requested: list[str], particles, available: list[str]) -> list[str]:
    if "all" in requested:
        return list(BANK_ORDER)
    if requested and "auto" not in requested:
        return list(dict.fromkeys(requested))
    names = set(available)
    banks = [name for name in ("event", "dis") if any(branch in names for branch in group_branches(name))]
    banks.append("kinematics")
    for bank in BANK_ORDER:
        if bank in {"event", "dis", "kinematics"}:
            continue
        if any(bank in banks_with_content(particle) for particle in particles):
            banks.append(bank)
    return banks


def group_branches(group: str) -> tuple[str, ...]:
    if group == "event":
        return "runNum", "helicity", "charge"
    if group == "dis":
        return "Q2", "xB", "nu", "W", "y", "t"
    return ()


def validate_peak_arguments(args: argparse.Namespace) -> None:
    peak_options_used = any(
        value is not None
        for value in (
            args.peak_search,
            args.peak_expected,
            args.peak_maximum_center_deviation,
            args.peak_maximum_sigma,
        )
    )
    if not args.peak_window:
        if peak_options_used or args.peak_n_sigma != 3.0 or args.peak_minimum_events != 200:
            raise ValueError("peak-fit options require --peak-window")
        return
    if args.peak_search is None:
        raise ValueError("--peak-window requires --peak-search MIN MAX")
    lower, upper = args.peak_search
    if not (np.isfinite(lower) and np.isfinite(upper) and upper > lower):
        raise ValueError("--peak-search must be a finite increasing interval")
    if not np.isfinite(args.peak_n_sigma) or args.peak_n_sigma <= 0.0:
        raise ValueError("--peak-n-sigma must be positive")
    if args.peak_minimum_events < 20:
        raise ValueError("--peak-minimum-events must be at least 20")
    if args.peak_expected is None and args.peak_maximum_center_deviation is not None:
        raise ValueError("--peak-maximum-center-deviation requires --peak-expected")
    for name, value in (
        ("--peak-maximum-center-deviation", args.peak_maximum_center_deviation),
        ("--peak-maximum-sigma", args.peak_maximum_sigma),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be positive")


def peak_input_branches(
    branch: str,
    available: list[str],
    beam_energy: float | None,
) -> tuple[list[str], str]:
    names = set(available)
    if branch in names:
        return [branch], "stored branch"
    if branch == "W" and {"Q2", "nu"}.issubset(names):
        return ["Q2", "nu"], "derived from Q2 and nu"
    if branch == "y" and "nu" in names and beam_energy is not None:
        return ["nu"], "derived from nu and beam energy"
    raise ValueError(
        f"peak-window quantity {branch!r} is neither stored nor derivable from available branches"
    )


def fit_peak_window(
    values: np.ndarray,
    *,
    branch: str,
    value_source: str = "stored branch",
    search: tuple[float, float],
    expected_center: float | None,
    n_sigma: float,
    maximum_center_deviation: float | None,
    maximum_sigma: float | None,
    minimum_events: int,
) -> tuple[np.ndarray, dict[str, object]]:
    raw = np.asarray(values, dtype=float)
    search_lower, search_upper = search
    in_search = np.isfinite(raw) & (raw >= search_lower) & (raw <= search_upper)
    fit_values = raw[in_search]
    estimate, reason = estimate_model(
        fit_values,
        "rec_m_eggX",
        n_sigma,
        minimum_events,
        5.0,
        100,
        1.0e-5,
        160,
        0.10,
        3.0,
        expected_center,
        search_lower,
        maximum_center_deviation,
        maximum_sigma,
        cut_component="core",
    )
    if estimate is None:
        raise RuntimeError(f"could not fit {branch} peak: {reason}")
    lower = max(float(estimate.lower), search_lower)
    upper = min(float(estimate.upper), search_upper)
    if not upper > lower:
        raise RuntimeError(f"fitted {branch} window does not overlap its search interval")
    selection = np.isfinite(raw) & (raw >= lower) & (raw <= upper)
    record: dict[str, object] = {
        "branch": branch,
        "value_source": value_source,
        "search_lower": search_lower,
        "search_upper": search_upper,
        "expected_center": expected_center,
        "n_sigma": n_sigma,
        "center": float(estimate.center),
        "sigma": float(estimate.sigma),
        "lower": lower,
        "upper": upper,
        "fit_lower": float(estimate.fit_lower),
        "fit_upper": float(estimate.fit_upper),
        "candidate_rows": int(raw.size),
        "search_rows": int(fit_values.size),
        "fit_rows": int(estimate.fit_entries),
        "retained_rows": int(np.count_nonzero(selection)),
        "fit_model": estimate.fit_model,
        "signal_fraction": float(estimate.signal_fraction),
        "background_fraction": float(estimate.background_fraction),
        "peak_significance": float(estimate.peak_significance),
        "maximum_center_deviation": maximum_center_deviation,
        "maximum_sigma": maximum_sigma,
    }
    return selection, record


def peak_provenance(peak_window: dict[str, object] | None) -> str:
    if not peak_window:
        return "Fitted peak window: none"
    return (
        f"Fitted {peak_window['branch']} peak: center={peak_window['center']:.6g}, "
        f"sigma={peak_window['sigma']:.6g}, "
        f"window=[{peak_window['lower']:.6g}, {peak_window['upper']:.6g}] "
        f"({peak_window['n_sigma']:.3g} sigma)"
    )


def array_rows(arrays: dict[str, np.ndarray]) -> int:
    return int(next(iter(arrays.values())).size) if arrays else 0


def inspect_tree(path: Path, requested_tree: str, dictionary: Path | None) -> tuple[str, int, list[str]]:
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    if dictionary and ROOT.gSystem.Load(str(dictionary.resolve())) < 0:
        raise RuntimeError(f"Could not load ROOT dictionary: {dictionary}")
    root_file = ROOT.TFile.Open(str(path.resolve()), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {path.resolve()}")
    try:
        tree_name = resolve(root_file, requested_tree)
        tree = root_file.Get(tree_name)
        if not tree:
            raise RuntimeError(f"Could not find tree {requested_tree} in {path.resolve()}")
        scalar = []
        for branch in tree.GetListOfBranches():
            class_name = str(branch.GetClassName())
            if class_name:
                continue
            leaf = branch.GetLeaf(branch.GetName())
            if leaf and int(leaf.GetLenStatic()) == 1:
                scalar.append(str(branch.GetName()))
        return tree_name, int(tree.GetEntries()), sorted(scalar)
    finally:
        root_file.Close()


def read_arrays(
    path: Path,
    tree: str,
    branches: list[str],
    where: str | None,
    max_rows: int | None,
) -> tuple[dict[str, np.ndarray], int]:
    import ROOT  # type: ignore

    frame = ROOT.RDataFrame(tree, str(path.resolve()))
    if where:
        frame = frame.Filter(where)
    if max_rows is not None:
        frame = frame.Range(max_rows)
    result = {name: np.asarray(values) for name, values in frame.AsNumpy(branches).items()}
    rows = int(next(iter(result.values())).size) if result else 0
    return result, rows


def load_selection(
    path: Path | None,
    input_rows: int,
    rows_after_filter: int,
    max_rows: int | None,
) -> np.ndarray:
    if path is None:
        return np.ones(rows_after_filter, dtype=bool)
    selection = np.asarray(np.load(path, allow_pickle=False), dtype=bool)
    if selection.ndim != 1 or selection.size != input_rows:
        raise ValueError(f"selection mask has shape {selection.shape}; expected ({input_rows},)")
    if max_rows is not None:
        selection = selection[:max_rows]
    if selection.size != rows_after_filter:
        raise ValueError("selection mask does not align with rows read from the ROOT tree")
    return selection


def file_record(path: Path, include_hash: bool) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    record: dict[str, object] = {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }
    if include_hash:
        digest = hashlib.sha256()
        with resolved.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        record["sha256"] = digest.hexdigest()
    return record


if __name__ == "__main__":
    raise SystemExit(main())
