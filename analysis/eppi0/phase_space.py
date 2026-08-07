from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray
PROTON_MASS_GEV = 0.9382720813


@dataclass(frozen=True)
class AnalysisPhaseSpace:
    """Generated-level phase-space cuts defining the selected analysis region."""

    q2_min: float | None = None
    w_min: float | None = None
    y_max: float | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AnalysisPhaseSpace":
        raw = config.get("phase_space", {})
        if raw is None:
            raw = {}
        return cls(
            q2_min=_optional_float(raw, "Q2_min"),
            w_min=_optional_float(raw, "W_min"),
            y_max=_optional_float(raw, "y_max"),
        )

    @property
    def enabled(self) -> bool:
        return self.q2_min is not None or self.w_min is not None or self.y_max is not None

    def mask(self, q2: Array, xb: Array, beam_energy: float) -> Array:
        q2, xb = np.broadcast_arrays(np.asarray(q2, float), np.asarray(xb, float))
        valid = np.isfinite(q2) & np.isfinite(xb) & (xb > 0.0) & (q2 > 0.0)
        if self.q2_min is not None:
            valid &= q2 >= float(self.q2_min)
        if self.w_min is not None:
            w2 = PROTON_MASS_GEV**2 + q2 * (1.0 / xb - 1.0)
            valid &= w2 >= float(self.w_min) ** 2
        if self.y_max is not None:
            y = np.divide(
                q2,
                2.0 * PROTON_MASS_GEV * float(beam_energy) * xb,
                out=np.full_like(q2, np.nan),
                where=xb > 0.0,
            )
            valid &= y <= float(self.y_max)
        return valid

    def as_npz_fields(self, prefix: str = "phase_space") -> dict[str, float]:
        return {
            f"{prefix}_Q2_min": _nan_if_none(self.q2_min),
            f"{prefix}_W_min": _nan_if_none(self.w_min),
            f"{prefix}_y_max": _nan_if_none(self.y_max),
        }

    def description(self) -> str:
        parts = []
        if self.q2_min is not None:
            parts.append(f"Q2 >= {self.q2_min:g}")
        if self.w_min is not None:
            parts.append(f"W >= {self.w_min:g}")
        if self.y_max is not None:
            parts.append(f"y <= {self.y_max:g}")
        return "none" if not parts else " and ".join(parts)


def _optional_float(raw: dict[str, Any], key: str) -> float | None:
    if key not in raw or raw[key] is None:
        return None
    return float(raw[key])


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)
