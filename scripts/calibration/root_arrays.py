from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def project_dictionary_candidates() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    override = os.environ.get("ROOTBRANCHES_DICT")
    if override:
        candidates.append(Path(override).expanduser())

    build_dirs = [project_root / "build", project_root / "work-build"]
    build_dirs.extend(sorted(project_root.glob("cmake-build-*")))
    candidates.extend(build_dir / "libROOTBranchesDict.so" for build_dir in build_dirs)
    candidates.extend(build_dir / "libROOTBranchesDict.dylib" for build_dir in build_dirs)
    return candidates


def load_project_dictionary(ROOT: Any) -> Path | None:
    if ROOT.TClass.GetClass("RecBranches"):
        return None

    for candidate in project_dictionary_candidates():
        if not candidate.is_file():
            continue
        status = int(ROOT.gSystem.Load(str(candidate.resolve())))
        if status < 0:
            raise RuntimeError(f"ROOT could not load project dictionary: {candidate}")
        return candidate
    return None


def import_root() -> Any:
    import ROOT  # type: ignore

    load_project_dictionary(ROOT)
    return ROOT


def load_dataframe(input_file: str | Path, tree: str):
    ROOT = import_root()
    path = str(input_file)
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {path}")
    aliases = {
        "rParticles": ("rParticles", "ReconstructedParticles", "Events"),
        "ReconstructedParticles": ("ReconstructedParticles", "rParticles", "Events"),
    }
    resolved = next(
        (name for name in aliases.get(tree, (tree,)) if root_file.Get(name)),
        tree,
    )
    if resolved != tree:
        print(f"Warning: using compatible particle tree {resolved}")
    root_file.Close()
    return ROOT.RDataFrame(resolved, path)


def has_column(df, column: str) -> bool:
    return column in {str(name) for name in df.GetColumnNames()}


def arrays_from_dataframe(df, columns: list[str], max_rows: int | None = None) -> dict[str, np.ndarray]:
    if max_rows is not None:
        df = df.Range(max_rows)
    arrays = df.AsNumpy(columns)
    return {name: np.asarray(values) for name, values in arrays.items()}


def define_common_proton_residuals(df):
    rad_to_deg = 180.0 / np.pi
    return (
        df.Filter("rec.pid == 2212 && rec.charge == 1")
        .Filter("gen.pid == rec.pid && rec.matchedGenIdx >= 0")
        .Define("theta_deg", f"rec.theta * {rad_to_deg}")
        .Define("delta_p_fit", "gen.p - rec.p")
        .Define("delta_theta_fit", f"(gen.theta - rec.theta) * {rad_to_deg}")
        .Define(
            "delta_phi_fit",
            f"TMath::ATan2(TMath::Sin(gen.phi - rec.phi), "
            f"TMath::Cos(gen.phi - rec.phi)) * {rad_to_deg}",
        )
    )


def define_common_electron_sf(df):
    if has_column(df, "electronP"):
        return (
            df.Define("sf_p", "electronP")
            .Define("sf_sector", "electronSector")
            .Define("sf_epcal", "electronEPCAL")
            .Define("sf_ecin", "electronEECIN")
            .Define("sampling_fraction", "(electronEPCAL + electronEECIN + electronEECOUT) / electronP")
            .Filter("electronP > 0 && sampling_fraction > 0 && electronSector >= 1 && electronSector <= 6")
        )

    return (
        df.Filter("rec.pid == 11")
        .Define("sf_p", "rec.p")
        .Define("sf_sector", "rec.sector")
        .Define("sf_epcal", "rec.E_PCAL")
        .Define("sf_ecin", "rec.E_ECIN")
        .Define("sampling_fraction", "(rec.E_PCAL + rec.E_ECIN + rec.E_ECOUT) / rec.p")
        .Filter("rec.p > 0 && sampling_fraction > 0 && rec.sector >= 1 && rec.sector <= 6")
    )
