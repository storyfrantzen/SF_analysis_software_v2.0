from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.exclugen_model import (
    EXCLUGEN_CROSS_SECTION_QUANTITY,
    EXCLUGEN_CROSS_SECTION_UNITS,
    ExclugenModelEvaluator,
)
from run_analysis import command_bin_centering, parser


AMP2614_REFERENCE_STRUCTURE_FUNCTIONS = (
    811.4259825767153,
    112.16184777543744,
    -346.30477852995,
    -22.593900582274053,
    99.87099082471912,
    157.38746451251305,
    -22.310262656125502,
    600.1144075085095,
    158.71006422683558,
)
AMP2614_REFERENCE_UNPOLARIZED = np.asarray(
    [
        86.76774041508828,
        198.54915669464216,
        100.62394914906324,
        198.54915669464216,
    ]
)


class _ReferenceModel:
    channel = "pi0"
    provenance = "amp2614 fixed reference"
    validity = {"W2": (2.2, 16.0), "Q2": (1.15, 8.31), "mt": (0.0, 1.8)}
    out_of_range = "report"
    Ebeam = 10.604

    def _epsilon(self, xb, q2):
        proton_mass = 0.93827208816
        xb = np.asarray(xb, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        nu = q2 / (2.0 * proton_mass * xb)
        y = nu / self.Ebeam
        gamma2 = (2.0 * proton_mass * xb) ** 2 / q2
        return (1.0 - y - 0.25 * gamma2 * y**2) / (
            1.0 - y + 0.5 * y**2 + 0.25 * gamma2 * y**2
        )

    def _sfs_tn(self, signed_t, xb, q2):
        shape = np.asarray(xb).shape
        return tuple(np.full(shape, value) for value in AMP2614_REFERENCE_STRUCTURE_FUNCTIONS)


class _ReferenceModels:
    @staticmethod
    def get(name, **kwargs):
        if name != "pi0.amp2614":
            raise KeyError(name)
        model = _ReferenceModel()
        model.Ebeam = float(kwargs["Ebeam"])
        return model


class ExclugenModelEvaluatorTests(unittest.TestCase):
    def _evaluator(
        self, beam_energy: float = 10.604, **kwargs
    ) -> ExclugenModelEvaluator:
        identity = patch(
            "eppi0.exclugen_model._source_identity",
            return_value=("50a5074da87718d5588d429b57468489aca75e6c", "clean"),
        )
        loader = patch(
            "eppi0.exclugen_model._load_exclugen_models",
            return_value=_ReferenceModels(),
        )
        with identity, loader:
            return ExclugenModelEvaluator(
                "/unused/exclugen",
                beam_energy,
                expected_revision="50a5074",
                **kwargs,
            )

    def test_amp2614_reference_uses_raw_hand_structure_functions(self) -> None:
        evaluator = self._evaluator()
        points = np.column_stack(
            (
                np.full(4, 0.3),
                np.full(4, 1.5),
                np.full(4, -0.3),
                np.asarray([0.0, 90.0, 180.0, 270.0]),
            )
        )
        np.testing.assert_allclose(
            evaluator(points),
            AMP2614_REFERENCE_UNPOLARIZED,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_metadata_records_generic_model_contract(self) -> None:
        metadata = self._evaluator().metadata()
        self.assertEqual(metadata["model_provider"], "exclugen")
        self.assertEqual(metadata["model_name"], "pi0.amp2614")
        self.assertEqual(metadata["model_revision"], "50a5074da87718d5588d429b57468489aca75e6c")
        self.assertEqual(metadata["channel"], "pi0p")
        self.assertEqual(metadata["cross_section_quantity"], EXCLUGEN_CROSS_SECTION_QUANTITY)
        self.assertEqual(metadata["units"], EXCLUGEN_CROSS_SECTION_UNITS)
        self.assertEqual(metadata["beam_energy"], 10.604)
        self.assertEqual(metadata["helicity_mode"], "beam-helicity averaged")

    def test_dirty_source_is_rejected_by_default(self) -> None:
        with patch(
            "eppi0.exclugen_model._source_identity",
            return_value=("50a5074da87718d5588d429b57468489aca75e6c", "dirty"),
        ):
            with self.assertRaisesRegex(ValueError, "source tree is dirty"):
                ExclugenModelEvaluator("/unused/exclugen", 10.604)

    def test_cli_defaults_to_exclugen_and_keeps_alternatives_explicit(self) -> None:
        bin_centering = parser().parse_args(
            ["bin-centering", "--config", "config.json", "--output", "C_BC.npz"]
        )
        self.assertEqual(bin_centering.model_provider, "exclugen")
        self.assertIsNone(bin_centering.exe)

        model = parser().parse_args(
            [
                "model-prediction",
                "--config",
                "config.json",
                "--cross-section",
                "cross_section.npz",
                "--output",
                "model.npz",
            ]
        )
        self.assertEqual(model.model_provider, "exclugen")
        self.assertIsNone(model.aao_exe)
        self.assertIsNone(model.table)

    def test_bin_centering_artifact_carries_exclugen_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            output = root / "C_BC.npz"
            config.write_text(
                json.dumps(
                    {
                        "beam_energy": 6.535,
                        "binning": {
                            "Q2": [1.9, 2.1],
                            "xB": [0.29, 0.31],
                            "minus_t": [0.15, 0.25],
                            "phi_deg": [0.0, 180.0, 360.0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = parser().parse_args(
                [
                    "bin-centering",
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                    "--N",
                    "1",
                ]
            )
            evaluator = self._evaluator(beam_energy=6.535)
            with patch(
                "run_analysis._exclugen_model_evaluator", return_value=evaluator
            ):
                command_bin_centering(args)
            with np.load(output, allow_pickle=False) as artifact:
                self.assertEqual(str(artifact["model_provider"]), "exclugen")
                self.assertEqual(str(artifact["model_name"]), "pi0.amp2614")
                self.assertEqual(str(artifact["channel"]), "pi0p")
                self.assertEqual(
                    str(artifact["cross_section_quantity"]),
                    EXCLUGEN_CROSS_SECTION_QUANTITY,
                )
                self.assertEqual(str(artifact["units"]), EXCLUGEN_CROSS_SECTION_UNITS)
                self.assertEqual(float(artifact["beam_energy"]), 6.535)


@unittest.skipUnless(
    os.environ.get("EXCLUGEN_ROOT"),
    "set EXCLUGEN_ROOT to run the pinned amp2614 source-tree reference",
)
class PinnedAmp2614IntegrationTests(unittest.TestCase):
    def test_pinned_source_tree_matches_reference_point(self) -> None:
        evaluator = ExclugenModelEvaluator(
            os.environ["EXCLUGEN_ROOT"],
            10.604,
            expected_revision=os.environ.get("EXCLUGEN_REVISION", "50a5074"),
        )
        points = np.column_stack(
            (
                np.full(4, 0.3),
                np.full(4, 1.5),
                np.full(4, -0.3),
                np.asarray([0.0, 90.0, 180.0, 270.0]),
            )
        )
        np.testing.assert_allclose(
            evaluator(points),
            AMP2614_REFERENCE_UNPOLARIZED,
            rtol=1.0e-12,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
