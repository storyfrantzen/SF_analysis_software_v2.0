from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np

from .exclusive_kinematics import PROTON_MASS_GEV


Array = np.ndarray

EXCLUGEN_CROSS_SECTION_QUANTITY = "d2sigma/(dt dphi)"
EXCLUGEN_CROSS_SECTION_UNITS = "nb/(GeV^2 rad)"
EXCLUGEN_T_CONVENTION = "signed nucleon t in GeV^2 (negative spacelike)"
EXCLUGEN_DEFAULT_MODEL = "pi0.amp2614"
EXCLUGEN_DEFAULT_CHANNEL = "pi0p"


def _load_exclugen_models(source_root: Path) -> ModuleType:
    """Import ``exclurad_py.models`` from one explicit source tree.

    The model layer is NumPy-only.  Importing the package also initializes the
    optional radiative kernel, so disable that unused backend while importing
    to keep model evaluation independent of the generator's Numba runtime.
    """

    source_root = source_root.resolve()
    package = source_root / "exclurad_py"
    if not (package / "models").is_dir():
        raise FileNotFoundError(
            f"EXCLUGEN source tree does not contain exclurad_py/models: {source_root}"
        )

    loaded = sys.modules.get("exclurad_py")
    if loaded is not None:
        loaded_name = getattr(loaded, "__file__", None)
        if not loaded_name:
            raise RuntimeError("loaded exclurad_py package has no source path")
        loaded_file = Path(loaded_name).resolve()
        if package not in loaded_file.parents:
            raise RuntimeError(
                "exclurad_py is already imported from a different source tree: "
                f"{loaded_file}"
            )

    inserted = str(source_root) not in sys.path
    if inserted:
        sys.path.insert(0, str(source_root))
    previous_backend = os.environ.get("RC_NUMBA")
    os.environ["RC_NUMBA"] = "0"
    try:
        return importlib.import_module("exclurad_py.models")
    finally:
        if previous_backend is None:
            os.environ.pop("RC_NUMBA", None)
        else:
            os.environ["RC_NUMBA"] = previous_backend
        if inserted:
            sys.path.remove(str(source_root))


def _deployed_revision(source_root: Path) -> str | None:
    deployed = source_root / "exclurad_py" / "DEPLOYED_VERSION"
    if not deployed.is_file():
        return None
    for line in deployed.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "commit" and value.strip():
            return value.strip()
    return None


def _source_identity(source_root: Path) -> tuple[str, str]:
    """Return the source revision and clean/dirty state.

    A deployed tree may intentionally omit ``.git``; in that case use its
    ``DEPLOYED_VERSION`` stamp and report the dirty state as unavailable.
    """

    try:
        revision_result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        revision_result = None
    if revision_result is None or revision_result.returncode != 0:
        deployed = _deployed_revision(source_root)
        if deployed is None:
            raise ValueError(
                "could not determine the EXCLUGEN source revision; provide a Git "
                "checkout or an exclurad_py/DEPLOYED_VERSION file"
            )
        return deployed, "unavailable"

    revision = revision_result.stdout.strip()
    if not revision:
        raise ValueError(f"empty EXCLUGEN revision reported for {source_root}")
    dirty_result = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if dirty_result.returncode != 0:
        dirty_state = "unavailable"
    else:
        dirty_state = "dirty" if dirty_result.stdout.strip() else "clean"
    return revision, dirty_state


class ExclugenModelEvaluator:
    """Evaluate an EXCLUGEN amplitude model as reduced virtual-photon cross section.

    Input points have columns ``(xB, Q2, signed_t, phi_deg)``.  The returned
    quantity is ``d2sigma/(dt dphi)`` in ``nb/(GeV^2 rad)``.  This adapter uses
    the model's raw Hand-convention structure functions.  It deliberately does
    not call ``StructureFunctions.__call__`` (which converts them for the
    covariant RC tensor) or ``born_xsec`` (which adds electron-level flux and
    the extra Q2/W2 differentials).
    """

    def __init__(
        self,
        source_root: str | Path,
        beam_energy: float,
        *,
        model_name: str = EXCLUGEN_DEFAULT_MODEL,
        expected_revision: str | None = None,
        channel: str = EXCLUGEN_DEFAULT_CHANNEL,
        beam_helicity: int = 0,
        allow_dirty: bool = False,
    ) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self.beam_energy = float(beam_energy)
        self.model_name = str(model_name)
        self.channel = str(channel)
        self.beam_helicity = int(beam_helicity)
        if not np.isfinite(self.beam_energy) or self.beam_energy <= 0.0:
            raise ValueError("beam energy must be finite and positive")
        if self.channel != EXCLUGEN_DEFAULT_CHANNEL:
            raise ValueError(
                "the supported analysis channel is pi0p; neutron and eta model "
                "generalization is outside the current production scope"
            )
        if self.beam_helicity not in (-1, 0, 1):
            raise ValueError("beam helicity must be -1, 0, or +1")

        self.revision, self.dirty_state = _source_identity(self.source_root)
        if expected_revision and not self.revision.startswith(str(expected_revision)):
            raise ValueError(
                f"EXCLUGEN revision {self.revision} does not match expected "
                f"revision {expected_revision}"
            )
        if self.dirty_state == "dirty" and not allow_dirty:
            raise ValueError(
                "EXCLUGEN source tree is dirty; commit/stash its changes or pass "
                "--allow-dirty-model for an explicitly non-production evaluation"
            )

        models = _load_exclugen_models(self.source_root)
        self._model = models.get(
            self.model_name,
            t_nucl=-0.1,
            Ebeam=self.beam_energy,
        )
        if getattr(self._model, "channel", None) != "pi0":
            raise ValueError(
                f"model {self.model_name!r} is not a proton pi0 model"
            )
        if not callable(getattr(self._model, "_sfs_tn", None)):
            raise TypeError(
                f"model {self.model_name!r} does not expose raw nucleon-t "
                "structure functions"
            )
        if not callable(getattr(self._model, "_epsilon", None)):
            raise TypeError(
                f"model {self.model_name!r} does not expose its virtual-photon "
                "polarization convention"
            )

        self.n_calls = 0
        self.n_points = 0
        self.n_out_of_range = 0

    def __enter__(self) -> "ExclugenModelEvaluator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    @property
    def helicity_mode(self) -> str:
        if self.beam_helicity == 0:
            return "beam-helicity averaged"
        return f"fixed beam helicity {self.beam_helicity:+d}"

    def _validity_mask(self, xb: Array, q2: Array, signed_t: Array) -> Array:
        w2 = PROTON_MASS_GEV**2 + q2 * (1.0 / xb - 1.0)
        values = {"W2": w2, "Q2": q2, "mt": -signed_t}
        in_range = np.ones(xb.shape, dtype=bool)
        for name, limits in (getattr(self._model, "validity", {}) or {}).items():
            if name not in values:
                continue
            low, high = map(float, limits)
            in_range &= (values[name] >= low) & (values[name] <= high)
        return in_range

    def __call__(self, points: Array) -> Array:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(
                "EXCLUGEN model points must have columns (xB, Q2, signed_t, phi_deg)"
            )
        result = np.full(points.shape[0], np.nan, dtype=float)
        if not points.size:
            return result

        xb, q2, signed_t, phi_deg = points.T
        valid = np.all(np.isfinite(points), axis=1)
        valid &= (xb > 0.0) & (xb < 1.0) & (q2 > 0.0) & (signed_t <= 0.0)
        self.n_calls += 1
        self.n_points += points.shape[0]
        if not np.any(valid):
            return result

        xb_valid = xb[valid]
        q2_valid = q2[valid]
        t_valid = signed_t[valid]
        raw = self._model._sfs_tn(t_valid, xb_valid, q2_valid)
        if len(raw) not in (4, 5, 9):
            raise ValueError(
                f"model {self.model_name!r} returned {len(raw)} structure functions; "
                "expected 4, 5, or 9"
            )
        structure = [
            np.broadcast_to(np.asarray(value, dtype=float), xb_valid.shape)
            for value in raw
        ]
        sigma_t, sigma_l, sigma_tt, sigma_lt = structure[:4]
        sigma_ltp = (
            structure[4] if len(structure) >= 5 else np.zeros_like(sigma_t)
        )
        epsilon = np.broadcast_to(
            np.asarray(self._model._epsilon(xb_valid, q2_valid), dtype=float),
            xb_valid.shape,
        )
        phi = np.deg2rad(phi_deg[valid])
        reduced = (
            sigma_t
            + epsilon * sigma_l
            + epsilon * np.cos(2.0 * phi) * sigma_tt
            + np.sqrt(2.0 * epsilon * (1.0 + epsilon))
            * np.cos(phi)
            * sigma_lt
        )
        if self.beam_helicity:
            reduced = reduced + (
                self.beam_helicity
                * np.sqrt(2.0 * epsilon * (1.0 - epsilon))
                * np.sin(phi)
                * sigma_ltp
            )
        result[valid] = reduced / (2.0 * np.pi)
        self.n_out_of_range += int(
            np.count_nonzero(~self._validity_mask(xb_valid, q2_valid, t_valid))
        )
        return result

    def metadata(self) -> dict[str, object]:
        provenance = str(getattr(self._model, "provenance", ""))
        validity = getattr(self._model, "validity", {}) or {}
        return {
            "model_provider": "exclugen",
            "model_name": self.model_name,
            "model_revision": self.revision,
            "channel": self.channel,
            "t_nucl": EXCLUGEN_T_CONVENTION,
            "cross_section_quantity": EXCLUGEN_CROSS_SECTION_QUANTITY,
            "units": EXCLUGEN_CROSS_SECTION_UNITS,
            "beam_energy": self.beam_energy,
            "helicity_mode": self.helicity_mode,
            "model_source_kind": "exclugen_source_tree",
            "model_source_path": str(self.source_root),
            "model_dirty_state": self.dirty_state,
            "model_out_of_range_policy": str(
                getattr(self._model, "out_of_range", "unknown")
            ),
            "model_validity_json": json.dumps(validity, sort_keys=True),
            "model_provenance": provenance,
        }
