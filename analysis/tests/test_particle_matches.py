from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.particle_matches import (  # noqa: E402
    join_selected_particle_matches,
    particle_match_summary,
)


def selected_table() -> dict[str, np.ndarray]:
    return {
        "source_file_id": np.array([7, 7], dtype=np.uint64),
        "source_event_index": np.array([10, 10], dtype=np.uint64),
        "run": np.array([11, 11]),
        "event": np.array([42, 42]),
        "role": np.array(["electron", "gamma"]),
        "occurrence": np.array([1, 1]),
        "particle_index": np.array([0, 3]),
        "pid": np.array([11, 22]),
        "detector": np.array([1, 0]),
        "sector": np.array([2, 0]),
        "rec_p": np.array([4.0, 0.8]),
        "rec_theta": np.array([0.2, 0.1]),
        "rec_phi": np.array([1.0, -0.4]),
    }


def reconstructed_chunk() -> dict[str, np.ndarray]:
    return {
        "source_file_id": np.array([8, 7, 7], dtype=np.uint64),
        "source_event_index": np.array([1, 10, 10], dtype=np.uint64),
        "particle_index": np.array([0, 3, 0]),
        "pid": np.array([11, 22, 11]),
        "detector": np.array([1, 0, 1]),
        "sector": np.array([1, 0, 2]),
        "rec_p": np.array([3.0, 0.8, 4.0]),
        "rec_theta": np.array([0.3, 0.1, 0.2]),
        "rec_phi": np.array([0.5, -0.4, 1.0]),
        "matched_gen_index": np.array([1, -999, 4]),
        "match_angle_deg": np.array([0.2, np.nan, 2.7]),
        "gen_pid": np.array([11, -999, 11]),
        "gen_p": np.array([2.9, np.nan, 4.1]),
        "gen_theta": np.array([0.31, np.nan, 0.21]),
        "gen_phi": np.array([0.51, np.nan, 1.01]),
    }


class ParticleMatchTests(unittest.TestCase):
    def test_join_preserves_selected_order_and_unmatched_rows(self) -> None:
        matches, stats = join_selected_particle_matches(
            selected_table(), [reconstructed_chunk()]
        )
        np.testing.assert_array_equal(matches["role"], ["electron", "gamma"])
        np.testing.assert_array_equal(matches["converter_row_found"], [True, True])
        np.testing.assert_array_equal(matches["gen_matched"], [True, False])
        np.testing.assert_allclose(matches["gen_p"][:1], [4.1])
        self.assertTrue(np.isnan(matches["gen_p"][1]))
        self.assertEqual(stats.selected_rows, 2)
        self.assertEqual(stats.converter_rows_scanned, 3)
        self.assertEqual(stats.converter_rows_found, 2)
        self.assertEqual(stats.generated_matches, 1)
        self.assertEqual(stats.generated_unmatched, 1)

    def test_summary_splits_role_and_detector(self) -> None:
        matches, stats = join_selected_particle_matches(
            selected_table(), [reconstructed_chunk()]
        )
        summary = particle_match_summary(matches, stats, warning_angle_deg=2.5)
        self.assertEqual(summary["generated_match_fraction"], 0.5)
        groups = {
            (item["role"], item["detector"]): item for item in summary["groups"]
        }
        self.assertEqual(groups[("electron", 1)]["matched_rows"], 1)
        self.assertEqual(groups[("gamma", 0)]["matched_rows"], 0)
        self.assertEqual(groups[("electron", 1)]["match_angle_warning_fraction"], 1.0)

    def test_duplicate_selected_key_is_rejected(self) -> None:
        selected = selected_table()
        selected["particle_index"] = np.array([0, 0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            join_selected_particle_matches(selected, [])

    def test_missing_converter_row_is_rejected(self) -> None:
        chunk = reconstructed_chunk()
        reduced = {name: values[:2] for name, values in chunk.items()}
        with self.assertRaisesRegex(ValueError, "missing 1 selected rows"):
            join_selected_particle_matches(selected_table(), [reduced])

    def test_selected_identity_mismatch_is_rejected(self) -> None:
        chunk = reconstructed_chunk()
        chunk["detector"] = chunk["detector"].copy()
        chunk["detector"][1] = 1
        with self.assertRaisesRegex(ValueError, "detector differ"):
            join_selected_particle_matches(selected_table(), [chunk])

    def test_claimed_match_requires_same_pid_and_finite_truth(self) -> None:
        chunk = reconstructed_chunk()
        chunk["gen_pid"] = chunk["gen_pid"].copy()
        chunk["gen_pid"][2] = 22
        with self.assertRaisesRegex(ValueError, "mismatched REC/GEN PIDs"):
            join_selected_particle_matches(selected_table(), [chunk])


if __name__ == "__main__":
    unittest.main()
