from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.beam_spin_binning import (
    BinningThresholds,
    PreparedBinningSample,
    automatic_bin_counts,
    candidate_binnings,
    circular_maximum_gap,
    coarsen_edges,
    evaluate_sample,
    pareto_frontier,
)
from eppi0.binning import AnalysisBinning


class BeamSpinBinningTest(unittest.TestCase):
    def test_coarsening_preserves_source_boundaries(self) -> None:
        source = np.asarray([1.0, 1.5, 2.0, 3.0, 4.0, 7.0])
        result = coarsen_edges(source, 3)
        self.assertEqual(result[0], source[0])
        self.assertEqual(result[-1], source[-1])
        self.assertTrue(set(result).issubset(set(source)))
        self.assertEqual(automatic_bin_counts(8), [4, 6, 8])

    def test_candidate_scan_uses_uniform_phi_edges(self) -> None:
        source = AnalysisBinning(
            [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [0.1, 0.2, 0.4],
            np.linspace(0.0, 360.0, 21),
        )
        candidates = candidate_binnings(source, [1, 2], [1], [2], [8, 12])
        self.assertEqual(len(candidates), 4)
        twelve = dict(candidates)["q1_x1_t2_p12"]
        np.testing.assert_allclose(np.diff(twelve.phi_edges), 30.0)

    def test_projected_precision_and_coverage_select_populated_cell(self) -> None:
        phi_edges = np.linspace(0.0, 360.0, 13)
        phi_centers = np.deg2rad(0.5 * (phi_edges[:-1] + phi_edges[1:]))
        repetitions = 100
        phi = np.repeat(phi_centers, 2 * repetitions)
        helicity = np.tile(
            np.r_[np.ones(repetitions), -np.ones(repetitions)], phi_centers.size
        )
        size = phi.size
        sample = PreparedBinningSample(
            label="sample",
            q2=np.full(size, 1.5),
            xb=np.full(size, 0.2),
            minus_t=np.full(size, 0.3),
            phi_rad=phi,
            denominator_weight=np.ones(size),
            denominator_variance_weight=np.ones(size),
            numerator_variance_weight=np.full(size, 1.0 / 0.8**2),
            signal_region_weight=np.ones(size),
            background_weight=np.zeros(size),
            plus_signal=helicity > 0.0,
            minus_signal=helicity < 0.0,
        )
        binning = AnalysisBinning(
            [1.0, 2.0], [0.1, 0.3], [0.1, 0.5], phi_edges
        )
        metrics = evaluate_sample(
            sample,
            binning,
            BinningThresholds(
                minimum_effective_signal_events=400.0,
                maximum_projected_uncertainty=0.10,
            ),
        )
        self.assertTrue(metrics.good.item())
        self.assertEqual(metrics.populated_phi_bins.item(), 12)
        self.assertEqual(metrics.maximum_phi_gap_deg.item(), 0.0)
        self.assertLess(metrics.projected_uncertainty.item(), 0.05)
        self.assertAlmostEqual(metrics.effective_signal_events.item(), size)

    def test_circular_gap_includes_wraparound(self) -> None:
        valid = np.array([[True, False, False, True, True, False]])
        edges = np.linspace(0.0, 360.0, 7)
        np.testing.assert_allclose(circular_maximum_gap(valid, edges), [120.0])

    def test_pareto_frontier_rejects_dominated_candidate(self) -> None:
        frontier = pareto_frontier(
            np.array([10, 10, 8]),
            np.array([0.08, 0.09, 0.10]),
            np.array([100, 120, 80]),
        )
        np.testing.assert_array_equal(frontier, [True, False, True])


if __name__ == "__main__":
    unittest.main()
