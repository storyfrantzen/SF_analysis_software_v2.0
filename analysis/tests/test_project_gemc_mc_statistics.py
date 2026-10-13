from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_gemc_mc_statistics import (  # noqa: E402
    coverage_at_events,
    events_for_cell_coverage,
    required_events_by_cell,
)


class GemcMcStatisticsProjectionTests(unittest.TestCase):
    def test_required_events_uses_requested_phi_coverage_order_statistic(self) -> None:
        accepted = np.asarray([[[[25.0, 100.0, 4.0, 0.0]]]])
        supported = accepted > 0.0
        required = required_events_by_cell(
            np.ones_like(accepted),
            accepted,
            supported,
            raw_events=1_000_000,
            precision=0.10,
            coverage_fraction=0.50,
        )
        # Two of four phi bins are required.  Their requirements are 1M and 4M.
        self.assertAlmostEqual(required.item(), 4_000_000.0)

    def test_insufficient_supported_phi_bins_is_infinite(self) -> None:
        accepted = np.asarray([[[[100.0, 0.0, 0.0, 0.0]]]])
        required = required_events_by_cell(
            np.ones_like(accepted),
            accepted,
            accepted > 0.0,
            raw_events=1_000_000,
            precision=0.10,
            coverage_fraction=0.75,
        )
        self.assertTrue(np.isinf(required.item()))

    def test_coverage_curve_counts_infinite_requirement_as_failure(self) -> None:
        required = np.asarray([1.0e6, 5.0e6, np.inf, 10.0e6, np.nan])
        projected = np.asarray([1.0e6, 5.0e6, 10.0e6])
        np.testing.assert_allclose(
            coverage_at_events(required, projected),
            [0.25, 0.50, 0.75],
        )

    def test_event_requirement_can_report_unreachable_coverage(self) -> None:
        required = np.asarray([1.0e6, 5.0e6, np.inf, 10.0e6, np.nan])
        self.assertEqual(events_for_cell_coverage(required, 0.50), 5.0e6)
        self.assertEqual(events_for_cell_coverage(required, 0.75), 10.0e6)
        self.assertTrue(np.isinf(events_for_cell_coverage(required, 0.90)))


if __name__ == "__main__":
    unittest.main()
