from __future__ import annotations

import unittest

import numpy as np

from eppi0.event_kinematics_diagnostics import (
    CORRELATIONS,
    DETECTOR_MAPS,
    MIN_DETAILED_TOPOLOGY_EVENTS,
    TOPOLOGY_LABELS,
    VARIABLES,
    _balanced_batches,
    _grouped_batches,
    _panel_dimensions,
    available_detector_maps,
    available_variables,
    detailed_topologies,
    detector_map_values,
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
        self.assertEqual(summary["detailed_topologies"], [])
        self.assertEqual(
            summary["topology_detail_minimum_candidates"],
            MIN_DETAILED_TOPOLOGY_EVENTS,
        )
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

    def test_detector_maps_require_all_coordinate_branches(self) -> None:
        arrays = dict(self.arrays)
        arrays["electronXDC1"] = np.arange(4.0)
        self.assertEqual(available_detector_maps(arrays), ())
        arrays["electronYDC1"] = np.arange(4.0)
        self.assertEqual(
            [item.key for item in available_detector_maps(arrays)],
            ["electron_dc_r1"],
        )

    def test_combined_photon_map_concatenates_finite_hits(self) -> None:
        arrays = dict(self.arrays)
        arrays.update(
            {
                "gamma1XFT": np.array([1.0, np.nan, 3.0, np.nan]),
                "gamma1YFT": np.array([2.0, np.nan, 4.0, np.nan]),
                "gamma2XFT": np.array([np.nan, 5.0, np.nan, 7.0]),
                "gamma2YFT": np.array([np.nan, 6.0, np.nan, 8.0]),
            }
        )
        detector_map = next(
            item
            for item in available_detector_maps(arrays)
            if item.key == "photon_ftcal_xy"
        )
        x, y = detector_map_values(detector_map, arrays, np.ones(4, dtype=bool))
        np.testing.assert_array_equal(x, [1.0, 3.0, 5.0, 7.0])
        np.testing.assert_array_equal(y, [2.0, 4.0, 6.0, 8.0])

    def test_balanced_pagination_avoids_sparse_final_pages(self) -> None:
        self.assertEqual(
            [len(batch) for batch in _balanced_batches(list(range(15)), 6)],
            [5, 5, 5],
        )
        self.assertEqual(
            [len(batch) for batch in _balanced_batches(list(range(35)), 6)],
            [6, 6, 6, 6, 6, 5],
        )
        self.assertEqual(
            [len(batch) for batch in _balanced_batches(list(range(15)), 4)],
            [4, 4, 4, 3],
        )

    def test_balanced_pagination_rejects_invalid_page_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            _balanced_batches([1], 0)

    def test_dense_pages_balance_to_compact_nine_panel_grids(self) -> None:
        self.assertEqual(
            [len(batch) for batch in _balanced_batches(list(range(13)), 9)],
            [7, 6],
        )
        self.assertEqual(
            [len(batch) for batch in _balanced_batches(list(range(11)), 9)],
            [6, 5],
        )

    def test_panel_dimensions_avoid_full_page_sparse_plots(self) -> None:
        self.assertEqual(_panel_dimensions(1), (2, 2))
        self.assertEqual(_panel_dimensions(3), (2, 3))
        self.assertEqual(_panel_dimensions(4), (2, 2))
        self.assertEqual(_panel_dimensions(6), (2, 3))
        self.assertEqual(_panel_dimensions(9), (3, 3))

    def test_sparse_topologies_remain_counts_only(self) -> None:
        topology = np.concatenate(
            (
                np.full(MIN_DETAILED_TOPOLOGY_EVENTS, 4),
                np.full(MIN_DETAILED_TOPOLOGY_EVENTS - 1, 5),
            )
        )
        selected = np.ones(topology.size, dtype=bool)
        self.assertEqual(detailed_topologies(topology, selected), [4])

    def test_kinematic_pages_never_mix_physics_sections(self) -> None:
        pages = _grouped_batches(VARIABLES, 6)
        self.assertTrue(pages)
        for section, page_index, page_count, variables in pages:
            self.assertGreaterEqual(page_index, 1)
            self.assertLessEqual(page_index, page_count)
            self.assertTrue(all(variable.section == section for variable in variables))
        self.assertEqual(
            [section for section, page, _count, _items in pages if page == 1],
            [
                "DIS",
                "channel kinematics",
                "exclusivity",
                "electron",
                "proton",
                "photons",
            ],
        )

    def test_correlations_and_detector_maps_have_coherent_page_sections(self) -> None:
        for values, maximum_size in ((CORRELATIONS, 6), (DETECTOR_MAPS, 4)):
            for section, _page, _count, items in _grouped_batches(
                values, maximum_size
            ):
                self.assertTrue(all(item.section == section for item in items))


if __name__ == "__main__":
    unittest.main()
