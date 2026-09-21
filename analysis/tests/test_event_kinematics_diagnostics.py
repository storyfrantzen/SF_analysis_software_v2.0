from __future__ import annotations

import unittest

import numpy as np

from eppi0.event_kinematics_diagnostics import (
    TOPOLOGY_LABELS,
    available_variables,
    observed_topologies,
    reconstructed_topology,
    report_summary,
    robust_range,
    trento_degrees,
)


class EventKinematicsDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arrays = {
            "Q2": np.array([1.2, 2.3, 3.4, 4.5]),
            "xB": np.array([0.15, 0.25, 0.35, 0.45]),
            "t": np.array([0.2, 0.3, 0.4, 0.5]),
            "trentoPhi": np.array([20.0, 100.0, 200.0, 300.0]),
            "pDet": np.array([1, 2, 2, 1]),
            "g1Det": np.array([1, 1, 0, 0]),
            "g2Det": np.array([1, 1, 1, 0]),
        }

    def test_reconstructed_topology_matches_campaign_encoding(self) -> None:
        np.testing.assert_array_equal(
            reconstructed_topology(self.arrays), np.array([4, 8, 9, 6])
        )

    def test_summary_applies_final_selection_before_topology_counts(self) -> None:
        mask = np.array([True, False, True, True])
        topology = reconstructed_topology(self.arrays)
        summary = report_summary(self.arrays, mask, topology)
        self.assertEqual(summary["input_rows"], 4)
        self.assertEqual(summary["selected_rows"], 3)
        self.assertEqual(summary["topology_counts"], {"4": 1, "6": 1, "9": 1})
        self.assertEqual(observed_topologies(topology, mask), [4, 6, 9])
        self.assertTrue(all(group in TOPOLOGY_LABELS for group in (4, 6, 9)))

    def test_robust_range_is_not_set_by_large_outlier(self) -> None:
        ordinary = np.linspace(0.0, 1.0, 10_000)
        values = np.concatenate([ordinary, [1.0e9]])
        low, high = robust_range(values)
        self.assertLess(low, 0.01)
        self.assertLess(high, 2.0)

    def test_available_variables_omits_missing_optional_branches(self) -> None:
        names = {variable.branch for variable in available_variables(self.arrays)}
        self.assertEqual(names, {"Q2", "xB", "t", "trentoPhi"})

    def test_trento_phi_is_wrapped_and_converted_to_degrees(self) -> None:
        values = trento_degrees(np.array([-np.pi / 2.0, 0.0, 2.0 * np.pi]))
        np.testing.assert_allclose(values, [270.0, 0.0, 0.0], atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
