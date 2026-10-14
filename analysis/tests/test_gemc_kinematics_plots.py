from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_gen_rec_kinematics import overlay_histograms, resolve_beam_energy


class GemcKinematicsPlotTests(unittest.TestCase):
    def test_beam_energy_comes_from_metadata_or_explicit_override(self) -> None:
        self.assertEqual(resolve_beam_energy({"beam_energy": 10.604}), 10.604)
        self.assertEqual(resolve_beam_energy({"beam_energy": 10.604}, 6.535), 6.535)
        with self.assertRaisesRegex(ValueError, "absent"):
            resolve_beam_energy({})
        with self.assertRaisesRegex(ValueError, "positive"):
            resolve_beam_energy({"beam_energy": 0.0})

    def test_all_generated_curve_is_scaled_to_same_event_integral(self) -> None:
        generated = np.array([0.2, 0.4, 0.6, 0.8])
        reconstructed = np.array([np.nan, 0.45, np.nan, 0.75])
        selected = np.array([False, True, False, True])
        edges = np.array([0.0, 0.5, 1.0])

        rec, same, all_scaled, counts = overlay_histograms(
            generated, reconstructed, selected, edges
        )

        np.testing.assert_array_equal(rec, [1, 1])
        np.testing.assert_array_equal(same, [1, 1])
        np.testing.assert_allclose(all_scaled, [1.0, 1.0])
        self.assertEqual(counts, {"rec": 2, "gen_same": 2, "gen_all": 4})
        self.assertAlmostEqual(float(all_scaled.sum()), float(same.sum()))

    def test_overlay_inputs_must_be_event_aligned(self) -> None:
        with self.assertRaisesRegex(ValueError, "must align"):
            overlay_histograms(
                np.ones(3), np.ones(2), np.ones(3, dtype=bool), np.array([0.0, 1.0])
            )


if __name__ == "__main__":
    unittest.main()
