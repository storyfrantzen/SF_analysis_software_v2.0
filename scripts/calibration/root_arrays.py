from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def import_root() -> Any:
    import ROOT  # type: ignore

    return ROOT


def load_dataframe(input_file: str | Path, tree: str):
    ROOT = import_root()
    return ROOT.RDataFrame(tree, str(input_file))


def arrays_from_dataframe(df, columns: list[str], max_rows: int | None = None) -> dict[str, np.ndarray]:
    if max_rows is not None:
        df = df.Range(max_rows)
    arrays = df.AsNumpy(columns)
    return {name: np.asarray(values) for name, values in arrays.items()}


def define_common_proton_residuals(df):
    rad_to_deg = 180.0 / np.pi
    return (
        df.Filter("rec.pid == 2212 && rec.charge == 1")
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
    return (
        df.Filter("rec.pid == 11")
        .Define("sampling_fraction", "(rec.E_PCAL + rec.E_ECIN + rec.E_ECOUT) / rec.p")
        .Filter("rec.p > 0 && sampling_fraction > 0 && rec.sector >= 1 && rec.sector <= 6")
    )
