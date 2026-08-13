from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np


from analysis.eppi0.phase_space import (
    AnalysisPhaseSpace,
    scattered_electron_momentum,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def electron_roles(value):
    if isinstance(value, dict):
        if value.get("role") == "electron":
            yield value
        for child in value.values():
            yield from electron_roles(child)
    elif isinstance(value, list):
        for child in value:
            yield from electron_roles(child)


class ConfigPhaseSpaceConsistencyTests(unittest.TestCase):
    CASES = {
        "rga": {
            "analysis_momentum_minimum": 2.0,
            "processing_momentum_minimum": 1.5,
            "analysis": REPOSITORY / "configs/analysis/rga/10.604.json",
            "processing": REPOSITORY / "configs/processing/rga/10.604",
            "post": REPOSITORY / "configs/post/rga/10.604",
        },
        "rgk": {
            "analysis_momentum_minimum": 1.0,
            "processing_momentum_minimum": 0.3,
            "analysis": REPOSITORY / "configs/analysis/rgk/6.535.json",
            "processing": REPOSITORY / "configs/processing/rgk/6.535",
            "post": REPOSITORY / "configs/post/rgk/6.535",
        },
    }

    def test_processing_thresholds_are_padded_below_analysis(self) -> None:
        for run_group, case in self.CASES.items():
            analysis_minimum = case["analysis_momentum_minimum"]
            processing_minimum = case["processing_momentum_minimum"]
            self.assertLess(processing_minimum, analysis_minimum)
            with self.subTest(run_group=run_group, layer="analysis"):
                analysis = json.loads(case["analysis"].read_text(encoding="utf-8"))
                self.assertNotIn("y_max", analysis["phase_space"])
                self.assertEqual(
                    float(analysis["phase_space"]["electron_p_min"]),
                    analysis_minimum,
                )

            skim_configs = 0
            for path in sorted(case["processing"].rglob("*.json")):
                config = json.loads(path.read_text(encoding="utf-8"))
                if not config.get("enableSkim", False):
                    continue
                skim_configs += 1
                with self.subTest(run_group=run_group, layer="processing", path=path):
                    self.assertNotIn("y_max", config)
                    self.assertEqual(
                        float(config["electron_p_min"]), processing_minimum
                    )
            self.assertGreater(skim_configs, 0)

    def test_declared_post_electron_momentum_thresholds_are_consistent(self) -> None:
        for run_group, case in self.CASES.items():
            role_count = 0
            for path in sorted(case["post"].rglob("*.json")):
                config = json.loads(path.read_text(encoding="utf-8"))
                for role in electron_roles(config):
                    role_count += 1
                    min_p_cuts = [
                        cut for cut in role.get("cuts", []) if cut.get("op") == "minP"
                    ]
                    with self.subTest(run_group=run_group, layer="post", path=path):
                        self.assertEqual(len(min_p_cuts), 1)
                        self.assertEqual(
                            float(min_p_cuts[0]["min"]),
                            case["analysis_momentum_minimum"],
                        )
            self.assertGreater(role_count, 0)

    def test_rga_post_electrons_are_fd_only(self) -> None:
        role_count = 0
        for path in sorted(self.CASES["rga"]["post"].rglob("*.json")):
            config = json.loads(path.read_text(encoding="utf-8"))
            for role in electron_roles(config):
                role_count += 1
                with self.subTest(path=path):
                    self.assertEqual(role.get("detectors"), [1])
        self.assertGreater(role_count, 0)

    def test_generated_phase_space_uses_scattered_electron_momentum(self) -> None:
        q2 = np.array([1.2, 1.2])
        xb = np.array([0.2, 0.8])
        momentum = scattered_electron_momentum(q2, xb, 6.535)
        self.assertLess(momentum[0], 5.0)
        self.assertGreater(momentum[1], 5.0)
        mask = AnalysisPhaseSpace(electron_p_min=5.0).mask(q2, xb, 6.535)
        np.testing.assert_array_equal(mask, [False, True])

    def test_legacy_y_cut_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "electron_p_min"):
            AnalysisPhaseSpace.from_config({"phase_space": {"y_max": 0.8}})


if __name__ == "__main__":
    unittest.main()
