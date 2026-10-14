from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray
PROTON_MASS_GEV = 0.9382720813
ELECTRON_MASS_GEV = 0.00051099895


@dataclass(frozen=True)
class AnalysisPhaseSpace:
    """Generated-level phase-space cuts defining the selected analysis region."""

    q2_min: float | None = None
    w_min: float | None = None
    electron_p_min: float | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AnalysisPhaseSpace":
        raw = config.get("phase_space", {})
        if raw is None:
            raw = {}
        if "y_max" in raw:
            raise ValueError(
                "phase_space.y_max is no longer supported; use electron_p_min"
            )
        electron_p_min = _optional_float(raw, "electron_p_min")
        if electron_p_min is not None and electron_p_min < 0.0:
            raise ValueError("phase_space.electron_p_min must be nonnegative")
        return cls(
            q2_min=_optional_float(raw, "Q2_min"),
            w_min=_optional_float(raw, "W_min"),
            electron_p_min=electron_p_min,
        )

    @property
    def enabled(self) -> bool:
        return (
            self.q2_min is not None
            or self.w_min is not None
            or self.electron_p_min is not None
        )

    def mask(self, q2: Array, xb: Array, beam_energy: float) -> Array:
        q2, xb = np.broadcast_arrays(np.asarray(q2, float), np.asarray(xb, float))
        valid = np.isfinite(q2) & np.isfinite(xb) & (xb > 0.0) & (q2 > 0.0)
        if self.q2_min is not None:
            valid &= q2 >= float(self.q2_min)
        if self.w_min is not None:
            w2 = PROTON_MASS_GEV**2 + q2 * (1.0 / xb - 1.0)
            valid &= w2 >= float(self.w_min) ** 2
        if self.electron_p_min is not None:
            valid &= scattered_electron_momentum(q2, xb, beam_energy) >= float(
                self.electron_p_min
            )
        return valid

    def as_npz_fields(self, prefix: str = "phase_space") -> dict[str, float]:
        return {
            f"{prefix}_Q2_min": _nan_if_none(self.q2_min),
            f"{prefix}_W_min": _nan_if_none(self.w_min),
            f"{prefix}_electron_p_min": _nan_if_none(self.electron_p_min),
        }

    def description(self) -> str:
        parts = []
        if self.q2_min is not None:
            parts.append(f"Q2 >= {self.q2_min:g}")
        if self.w_min is not None:
            parts.append(f"W >= {self.w_min:g}")
        if self.electron_p_min is not None:
            parts.append(f"electron p >= {self.electron_p_min:g} GeV")
        return "none" if not parts else " and ".join(parts)


def scattered_electron_momentum(
    q2: Array,
    xb: Array,
    beam_energy: float,
    proton_mass: float = PROTON_MASS_GEV,
    electron_mass: float = ELECTRON_MASS_GEV,
) -> Array:
    """Reconstruct scattered-electron momentum from DIS coordinates."""
    q2, xb = np.broadcast_arrays(np.asarray(q2, float), np.asarray(xb, float))
    nu = np.divide(
        q2,
        2.0 * float(proton_mass) * xb,
        out=np.full_like(q2, np.nan),
        where=xb > 0.0,
    )
    electron_energy = float(beam_energy) - nu
    momentum_squared = electron_energy**2 - float(electron_mass) ** 2
    momentum = np.full_like(momentum_squared, np.nan)
    valid = (
        np.isfinite(electron_energy)
        & (electron_energy >= float(electron_mass))
        & np.isfinite(momentum_squared)
        & (momentum_squared >= 0.0)
    )
    momentum[valid] = np.sqrt(momentum_squared[valid])
    return momentum


def _optional_float(raw: dict[str, Any], key: str) -> float | None:
    if key not in raw or raw[key] is None:
        return None
    return float(raw[key])


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)
