from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "compare_bsa_to_andrey.py"
SPEC = importlib.util.spec_from_file_location("compare_bsa_to_andrey", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompareBsaToAndreyTest(unittest.TestCase):
    def test_reference_table_has_expected_rows_and_polarities(self) -> None:
        rows = MODULE.load_reference(MODULE.REFERENCE_DEFAULT)
        self.assertEqual(len(rows), 30)
        self.assertEqual(sum(row["polarity"] == "out" for row in rows), 15)
        self.assertEqual(sum(row["polarity"] == "in" for row in rows), 15)

    def test_nearest_production_bin_is_selected_by_polarity(self) -> None:
        artifact = {
            "q2_edges": np.array([2.0, 3.0, 4.0]),
            "xb_edges": np.array([0.2, 0.3, 0.4]),
            "t_edges": np.array([0.2, 0.6, 1.0]),
            "sin_phi_amplitude": np.full((2, 2, 2), 0.05),
            "sin_phi_amplitude_uncertainty": np.full((2, 2, 2), 0.02),
            "sin_phi_amplitude_polarization_uncertainty": np.full((2, 2, 2), 0.003),
            "sin_phi_fit_quality": np.ones((2, 2, 2), dtype=bool),
        }
        artifact["sin_phi_fit_quality"][0, 0, 0] = False
        artifact["sin_phi_amplitude"][0, 0, 1] = 0.08
        reference = [{
            "region": 0, "polarity": "out", "Q2": 2.5, "xB": 0.25,
            "minus_t": 0.35, "reference_amplitude": 0.06,
            "reference_statistical_uncertainty": 0.01,
        }]
        matched = MODULE.match_reference_rows(
            reference, {"out": artifact}, q2_scale=0.5, xb_scale=0.05, t_scale=0.25
        )
        self.assertEqual((matched[0]["iq2"], matched[0]["ixb"], matched[0]["it"]), (0, 0, 1))
        self.assertAlmostEqual(matched[0]["campaign_amplitude"], 0.08)
        self.assertAlmostEqual(matched[0]["difference"], 0.02)


if __name__ == "__main__":
    unittest.main()
