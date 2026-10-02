from __future__ import annotations

import unittest

import numpy as np

from scripts.calibration.eppi0_proton_root_audit import summarize_root_region


class ProtonRootAuditTests(unittest.TestCase):
    def test_branch_flip_metrics_track_boundary_crossing(self) -> None:
        roots = np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        valid = np.ones(roots.shape, dtype=bool)
        measured = np.asarray([1.2, 1.49, 1.8])
        summary = summarize_root_region(
            roots, valid, measured, np.ones(3, dtype=bool),
            perturbation_fraction=0.05,
            corrected_momentum=np.asarray([1.2, 1.6, 1.8]),
            correction_support=np.ones(3, dtype=bool),
            correction_fraction=np.asarray([0.0, 1.6 / 1.49 - 1.0, 0.0]),
        )
        self.assertEqual(summary["entries"], 3)
        self.assertEqual(summary["ambiguousRootFraction"], 1.0)
        self.assertAlmostEqual(summary["rootSeparationGeV"]["median"], 1.0)
        self.assertGreater(
            summary["branchFlipFractionOfAmbiguous"]["plusPerturbation"], 0.0
        )
        self.assertAlmostEqual(
            summary["provisionalCorrection"][
                "branchFlipFractionOfAmbiguousSupported"
            ],
            1.0 / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
