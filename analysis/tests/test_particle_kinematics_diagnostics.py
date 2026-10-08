from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from particle_kinematics_diagnostics import (
    banks_with_content,
    derive_event_arrays,
    discover_particle_prefixes,
    make_particle,
    render_report,
    required_branches,
    summary_for,
)
from plot_particle_kinematics import fit_peak_window, peak_input_branches


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.branches = [
            "Q2", "nu", "xB", "runNum", "electronP", "electronTheta", "electronPhi",
            "electronSector", "electronDet", "electronXPCAL", "electronYPCAL",
            "electronUPCAL", "electronVPCAL", "electronWPCAL", "electronEPCAL",
            "electronXDC1", "electronYDC1", "selectedP", "selectedTheta", "selectedPhi",
            "protonP", "protonTheta", "protonPhi",
        ]

    def test_discovers_role_prefixes_but_not_vector_selected_columns(self) -> None:
        self.assertEqual(discover_particle_prefixes(self.branches), ["electron", "proton"])

    def test_particle_maps_available_detector_branches(self) -> None:
        particle = make_particle("electron", self.branches)
        self.assertEqual(particle.branches["p"], "electronP")
        self.assertEqual(particle.branches["pcal_x"], "electronXPCAL")
        self.assertIn("pcal", banks_with_content(particle))
        self.assertIn("dc", banks_with_content(particle))
        self.assertNotIn("ft", banks_with_content(particle))

    def test_unprefixed_particle_tree_is_supported(self) -> None:
        branches = ["p", "theta", "phi", "pid", "xFT", "yFT", "E_FTCAL"]
        self.assertEqual(discover_particle_prefixes(branches), [""])
        particle = make_particle("", branches)
        self.assertEqual(particle.name, "particle")
        self.assertEqual(particle.branches["ft_e"], "E_FTCAL")

    def test_required_branches_only_requests_selected_banks(self) -> None:
        particle = make_particle("electron", self.branches)
        requested = required_branches([particle], ["dis", "kinematics"], self.branches)
        self.assertIn("Q2", requested)
        self.assertIn("electronP", requested)
        self.assertNotIn("electronXPCAL", requested)


class DerivedKinematicsTests(unittest.TestCase):
    def test_w_and_y_are_derived_when_absent(self) -> None:
        arrays = {
            "Q2": np.array([1.0]),
            "nu": np.array([1.5]),
        }
        result = derive_event_arrays(arrays, beam_energy=6.0)
        expected_w2 = 0.9382720813**2 + 2.0 * 0.9382720813 * 1.5 - 1.0
        self.assertAlmostEqual(float(result["W"][0]), expected_w2**0.5)
        self.assertAlmostEqual(float(result["y"][0]), 0.25)

    def test_fitted_peak_window_tracks_shifted_signal(self) -> None:
        rng = np.random.default_rng(23)
        signal = rng.normal(0.976, 0.024, 12000)
        background = rng.uniform(0.72, 1.20, 5000)
        values = np.concatenate((signal, background))
        selection, result = fit_peak_window(
            values,
            branch="W",
            search=(0.72, 1.20),
            expected_center=0.9382720813,
            n_sigma=3.0,
            maximum_center_deviation=0.15,
            maximum_sigma=0.12,
            minimum_events=200,
        )
        self.assertAlmostEqual(result["center"], 0.976, delta=0.01)
        self.assertAlmostEqual(result["sigma"], 0.024, delta=0.01)
        self.assertGreater(int(np.count_nonzero(selection)), 10000)
        self.assertLess(result["lower"], result["center"])
        self.assertGreater(result["upper"], result["center"])

    def test_peak_window_accepts_derived_w(self) -> None:
        branches, source = peak_input_branches(
            "W",
            ["Q2", "nu", "electronP"],
            6.395,
        )
        self.assertEqual(branches, ["Q2", "nu"])
        self.assertEqual(source, "derived from Q2 and nu")


class RenderingTests(unittest.TestCase):
    def test_minimal_report_and_summary_are_nonempty(self) -> None:
        count = 800
        rng = np.random.default_rng(14)
        arrays = {
            "Q2": rng.uniform(1.0, 4.0, count),
            "nu": rng.uniform(1.0, 3.0, count),
            "xB": rng.uniform(0.1, 0.6, count),
            "electronP": rng.uniform(1.0, 6.0, count),
            "electronTheta": rng.uniform(0.08, 0.5, count),
            "electronPhi": rng.uniform(-np.pi, np.pi, count),
            "electronSector": rng.integers(1, 7, count),
            "electronXPCAL": rng.normal(0.0, 120.0, count),
            "electronYPCAL": rng.normal(0.0, 120.0, count),
        }
        particle = make_particle("electron", arrays)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "diagnostics.pdf"
            pages, records = render_report(
                output,
                arrays,
                [particle],
                ["dis", "kinematics", "pcal"],
                label="synthetic",
                provenance=["unit test"],
                beam_energy=6.4,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertGreaterEqual(pages, 4)
            summary = summary_for(arrays, [particle], ["dis", "kinematics", "pcal"], records)
            self.assertEqual(summary["rows"], count)
            self.assertIn("pcal", summary["particles"]["electron"]["rendered_sections"])


if __name__ == "__main__":
    unittest.main()
