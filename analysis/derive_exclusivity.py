#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from eppi0.binning import from_config
from eppi0.exclusivity import (
    DEFAULT_VARIABLES,
    ExclusivityCuts,
    apply_cuts,
    derive_cuts,
    load_cuts,
    save_cuts,
    topology_ids_from_groups,
)
from eppi0.topology import ft_photon_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive one exclusivity definition and apply it to an event sample."
    )
    parser.add_argument("sample", type=Path, help="Reference sample used to derive cuts")
    parser.add_argument(
        "--apply-to", type=Path, help="Optional second sample; defaults to reference"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cuts", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("npz", "selected-root"),
        default="npz",
        help="Read either a dense event NPZ or the selected-candidate ROOT tree directly",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        help="ROOT dictionary shared library for selected-root input",
    )
    parser.add_argument(
        "--tree", default="sEvents", help="ROOT tree name for selected-root input"
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="Optional multipage PDF containing the stored fits and cut-flow audit",
    )
    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument(
        "--global-cuts",
        dest="global_cuts",
        action="store_true",
        help="Pool kinematic bins within each detector topology (default)",
    )
    grouping.add_argument(
        "--per-bin-cuts",
        dest="global_cuts",
        action="store_false",
        help="Derive local Q2/xB/-t windows with same-topology fallbacks",
    )
    parser.set_defaults(global_cuts=None)
    parser.add_argument(
        "--n-sigma",
        type=float,
        default=None,
        help=(
            "Gaussian-equivalent signal containment; 3 gives 99.73%% "
            "containment for every fitted signal model"
        ),
    )
    parser.add_argument("--minimum-events", type=int)
    parser.add_argument("--fit-window-sigma", type=float)
    parser.add_argument("--fit-max-iterations", type=int)
    parser.add_argument("--fit-convergence", type=float)
    parser.add_argument("--fit-histogram-bins", type=int)
    parser.add_argument("--minimum-signal-fraction", type=float)
    parser.add_argument("--minimum-peak-significance", type=float)
    parser.add_argument("--maximum-local-sigma-ratio", type=float)
    parser.add_argument("--maximum-local-center-shift-sigma", type=float)
    parser.add_argument("--refinement-max-iterations", type=int)
    parser.add_argument("--refinement-min-iterations", type=int)
    parser.add_argument("--refinement-boundary-tolerance", type=float)
    parser.add_argument(
        "--no-continuous-refinement",
        dest="continuous_refinement",
        action="store_false",
        default=None,
        help="Disable continuous parameter refinement after the coarse model scan",
    )
    parser.add_argument("--reuse-cuts", action="store_true")
    return parser.parse_args()


def arrays(sample) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...]]:
    values = {name: sample[name] for name in DEFAULT_VARIABLES}
    if "rec_ft_photon_count" in sample:
        ft_photons = sample["rec_ft_photon_count"]
    else:
        gamma1_name = _first_available(
            sample, "rec_gamma1_detector", "rec_g1Det", "rec_gamma1Det"
        )
        gamma2_name = _first_available(
            sample, "rec_gamma2_detector", "rec_g2Det", "rec_gamma2Det"
        )
        ft_photons = ft_photon_count(sample[gamma1_name], sample[gamma2_name])
    return values, (
        sample["rec_proton_detector"],
        ft_photons,
        sample["rec_Q2"],
        sample["rec_xB"],
        sample["rec_minus_t"],
    )


def _first_available(sample, *names: str) -> str:
    for name in names:
        if name in sample:
            return name
    raise KeyError(
        "event sample lacks selected-photon detector topology; rebuild or re-export it"
    )


def selected_root_arrays(path: Path, dictionary: Path | None, tree_name: str):
    import ROOT  # type: ignore
    from eppi0.root_trees import resolve

    ROOT.gROOT.SetBatch(True)
    if dictionary:
        status = ROOT.gSystem.Load(str(dictionary.resolve()))
        if status < 0:
            raise RuntimeError(f"Could not load ROOT dictionary: {dictionary}")

    root_path = str(path.resolve())
    root_file = ROOT.TFile.Open(root_path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open selected ROOT file: {root_path}")
    resolved_tree_name = resolve(root_file, tree_name)
    tree = root_file.Get(resolved_tree_name)
    if tree and resolved_tree_name != tree_name:
        print(f"Warning: using compatible tree {resolved_tree_name}", file=sys.stderr)
        tree_name = resolved_tree_name
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Could not find tree {tree_name} in {root_path}")
    columns = [
        "pDet", "g1Det", "g2Det", "Q2", "xB", "t", "m_gg", "pT_miss",
        "m2_epX", "m_eggX", "E_miss", "m2_miss",
    ]
    missing = [name for name in columns if not tree.GetBranch(name)]
    root_file.Close()
    if missing:
        raise RuntimeError(f"Tree {tree_name} in {root_path} is missing branches: {missing}")

    raw = ROOT.RDataFrame(tree_name, root_path).AsNumpy(columns)
    values = {
        "rec_m_gg": raw["m_gg"],
        "rec_pT_miss": raw["pT_miss"],
        "rec_m2_epX": raw["m2_epX"],
        "rec_m_eggX": raw["m_eggX"],
        "rec_E_miss": raw["E_miss"],
        "rec_m2_miss": raw["m2_miss"],
    }
    return values, (
        raw["pDet"],
        ft_photon_count(raw["g1Det"], raw["g2Det"]),
        raw["Q2"],
        raw["xB"],
        raw["t"],
    )


def load_arrays(path: Path, input_format: str, dictionary: Path | None, tree_name: str):
    if input_format == "selected-root":
        return selected_root_arrays(path, dictionary, tree_name)
    sample = np.load(path, allow_pickle=False)
    return arrays(sample)


def print_group_diagnostics(cuts: ExclusivityCuts) -> None:
    print(
        f"Cut groups: populated={cuts.populated_group_ids.size}, "
        f"retained={cuts.group_ids.size}, dropped={cuts.dropped_group_ids.size}"
    )
    if not cuts.dropped_group_ids.size:
        return
    print("Dropped cut groups:")
    topologies = topology_ids_from_groups(
        cuts.dropped_group_ids, cuts.global_mode
    )
    for group_id, topology, variable, reason in zip(
        cuts.dropped_group_ids,
        topologies,
        cuts.dropped_variables,
        cuts.dropped_reasons,
        strict=True,
    ):
        print(
            f"  group_id={int(group_id)}, pDet={int(topology) // 4}, "
            f"FT photons={int(topology) % 4}, failed_at={variable}: {reason}"
        )


def derivation_settings(args: argparse.Namespace) -> dict[str, object]:
    document = json.loads(args.config.read_text())
    configured = document.get("exclusivity", {})
    if not isinstance(configured, dict):
        raise ValueError("analysis config exclusivity section must be an object")

    defaults = {
        "n_sigma": 3.0,
        "minimum_events": 50,
        "fit_window_sigma": 5.0,
        "fit_max_iterations": 100,
        "fit_convergence": 1.0e-5,
        "fit_histogram_bins": 160,
        "minimum_signal_fraction": 0.1,
        "minimum_peak_significance": 3.0,
        "maximum_local_sigma_ratio": 2.0,
        "maximum_local_center_shift_sigma": 2.5,
        "continuous_refinement": True,
    }
    settings: dict[str, object] = {}
    settings["global_mode"] = (
        args.global_cuts
        if args.global_cuts is not None
        else configured.get("global_cuts", True)
    )
    for name, default in defaults.items():
        command_line = getattr(args, name)
        settings[name] = (
            command_line
            if command_line is not None
            else configured.get(name, default)
        )

    refinement = configured.get("refinement", {})
    if not isinstance(refinement, dict):
        raise ValueError("exclusivity refinement section must be an object")
    refinement_options = {
        "refinement_max_iterations": ("maximum_iterations", 8),
        "refinement_min_iterations": ("minimum_iterations", 3),
        "refinement_boundary_tolerance": ("boundary_relative_tolerance", 0.02),
    }
    for argument, (key, default) in refinement_options.items():
        command_line = getattr(args, argument)
        settings[argument] = (
            command_line
            if command_line is not None
            else refinement.get(key, default)
        )

    policies = configured.get("variables", {})
    if not isinstance(policies, dict):
        raise ValueError("exclusivity variables section must be an object")
    unknown = sorted(set(policies) - set(DEFAULT_VARIABLES))
    if unknown:
        raise ValueError(f"unknown exclusivity variables in config: {unknown}")
    containments: dict[str, float] = {}
    components: dict[str, str] = {}
    for name, policy in policies.items():
        if not isinstance(policy, dict):
            raise ValueError(f"exclusivity policy for {name} must be an object")
        if "containment" in policy:
            containments[name] = float(policy["containment"])
        if "cut_component" in policy:
            components[name] = str(policy["cut_component"])
    settings["cut_containments"] = containments
    settings["cut_components"] = components
    return settings


def main() -> int:
    args = parse_args()
    binning = from_config(args.config)
    settings = derivation_settings(args)
    reference_values, (detector, ft_photons, q2, xb, minus_t) = load_arrays(
        args.sample, args.format, args.dictionary, args.tree
    )
    iq2, ixb, it, _ = binning.indices(q2, xb, minus_t, np.zeros_like(q2))
    if args.reuse_cuts:
        cuts = load_cuts(str(args.cuts))
    else:
        cuts = derive_cuts(
            reference_values,
            detector,
            ft_photons,
            iq2,
            ixb,
            it,
            **settings,
        )
        args.cuts.parent.mkdir(parents=True, exist_ok=True)
        save_cuts(str(args.cuts), cuts)
    if args.diagnostics:
        from eppi0.exclusivity_diagnostics import render_diagnostics

        args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
        rendered = render_diagnostics(cuts, args.diagnostics)
        print(
            f"Diagnostic PDF: rendered {len(rendered)} groups to "
            f"{args.diagnostics}"
        )
    print_group_diagnostics(cuts)
    if cuts.group_ids.size == 0:
        raise RuntimeError(
            f"No complete exclusivity cut groups; diagnostics saved to {args.cuts}"
        )

    target_values, (detector, ft_photons, q2, xb, minus_t) = load_arrays(
        args.apply_to or args.sample, args.format, args.dictionary, args.tree
    )
    iq2, ixb, it, _ = binning.indices(q2, xb, minus_t, np.zeros_like(q2))
    mask = apply_cuts(cuts, target_values, detector, ft_photons, iq2, ixb, it)
    args.mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.mask, mask)
    global_windows = int(np.count_nonzero(cuts.window_source == "global"))
    local = int(np.count_nonzero(cuts.window_source == "local"))
    fallback = int(np.count_nonzero(cuts.window_source == "topology_fallback"))
    consistency = int(
        np.count_nonzero(cuts.window_source == "topology_consistency_fallback")
    )
    print(
        f"Windows: global={global_windows}, local={local}, "
        f"topology fallback={fallback}, "
        f"topology consistency fallback={consistency}"
    )
    print(
        f"Signal containment: {100.0 * cuts.signal_containment:.5g}% "
        f"(Gaussian-equivalent n={cuts.n_sigma:g})"
    )
    for index, name in enumerate(cuts.variables):
        models, model_counts = np.unique(cuts.fit_model[:, index], return_counts=True)
        model_summary = ", ".join(
            f"{model}={count}" for model, count in zip(models, model_counts, strict=True)
        )
        print(
            f"  {name}: center median={np.median(cuts.centers[:, index]):.7g}, "
            f"characteristic-scale median={np.median(cuts.sigmas[:, index]):.7g}, "
            f"lower median={np.median(cuts.lower[:, index]):.7g}, "
            f"upper median={np.median(cuts.upper[:, index]):.7g}, "
            f"signal fraction median={np.median(cuts.signal_fractions[:, index]):.4g}, "
            f"cut component={cuts.cut_components[index]}, "
            f"cut containment={100.0 * cuts.cut_containments[index]:.5g}%, "
            f"significance median={np.median(cuts.peak_significance[:, index]):.4g}, "
            f"chi2/ndof median="
            f"{np.median(cuts.pearson_chi2[:, index] / cuts.fit_ndof[:, index]):.4g}, "
            f"models: {model_summary}"
        )
    print(
        f"N-1 refinement: iterations={cuts.refinement_iterations}, "
        f"converged={cuts.refinement_converged}, maximum relative boundary "
        f"change={cuts.maximum_boundary_change:.5g}"
    )
    print(
        "Boundary-change history: "
        + ", ".join(f"{value:.5g}" for value in cuts.boundary_change_history)
    )
    print(f"Passing events: {mask.sum()}/{mask.size}")
    if args.format == "selected-root":
        print(
            "Mask rows correspond to selected ROOT candidates; pass this mask "
            "to response-root."
        )
    print(f"Wrote {args.cuts}")
    print(f"Wrote {args.mask}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
