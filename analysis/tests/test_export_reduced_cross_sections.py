from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "export_reduced_cross_sections.py"
SPEC = importlib.util.spec_from_file_location("export_reduced_cross_sections", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExportReducedCrossSectionsTest(unittest.TestCase):
    def artifact(self) -> dict[str, np.ndarray]:
        shape = (2, 1, 1, 2)
        uncertainty = np.full(shape, 0.2)
        covariance = np.zeros((2, 1, 1, 2, 2))
        covariance[..., 0, 0] = 0.04
        covariance[..., 1, 1] = 0.04
        valid = np.ones(shape, dtype=bool)
        valid[1, 0, 0, 1] = False
        return {
            "q2_edges": np.array([1.0, 2.0, 3.0]),
            "xb_edges": np.array([0.1, 0.2]),
            "t_edges": np.array([0.2, 0.4]),
            "phi_edges": np.array([0.0, 180.0, 360.0]),
            "reduced_cross_section": np.full(shape, 1.5),
            "uncertainty": uncertainty,
            "final_validity_mask": valid,
            "flux_q2_coordinate": np.full(np.prod(shape), 1.7),
            "flux_xb_coordinate": np.full(shape, 0.15),
            "covariance_phi": covariance,
            "covariance_phi_bootstrap_experiments": np.asarray(300),
            "reduced_cross_section_units": np.asarray("nb/(GeV^2 rad)"),
        }

    def test_rows_include_only_final_valid_bins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cross_section.npz"
            np.savez(path, **self.artifact())
            rows, metadata = MODULE.rows_for_sample(
                "torus_plus1", "torus+1", MODULE.load_artifact(path), path,
                beam_energy=10.604,
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(metadata["valid_bins"], 3)
            self.assertEqual(metadata["covariance_bootstrap_experiments"], 300)
            self.assertAlmostEqual(
                rows[0]["propagated_statistical_and_finite_MC_uncertainty_nb_per_GeV2_rad"],
                0.2,
            )
            self.assertGreater(rows[0]["virtual_photon_epsilon"], 0.0)
            self.assertLess(rows[0]["virtual_photon_epsilon"], 1.0)
            self.assertEqual(metadata["beam_energy_GeV"], 10.604)
            output = Path(directory) / "out.csv"
            MODULE.write_csv(output, rows)
            with output.open(newline="", encoding="utf-8") as source:
                written = list(csv.DictReader(source))
            self.assertEqual(len(written), 3)
            self.assertEqual(written[0]["campaign_key"], "torus_plus1")

    def test_covariance_diagonal_is_validated(self) -> None:
        artifact = self.artifact()
        artifact["covariance_phi"][..., 0, 0] = 0.09
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, **artifact)
            with self.assertRaisesRegex(ValueError, "covariance diagonal"):
                MODULE.validate_artifact(MODULE.load_artifact(path), path)

    def test_legacy_flat_coordinate_order_is_unflattened(self) -> None:
        binning = MODULE.AnalysisBinning(
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            [0.2, 0.4, 0.6],
            [0.0, 180.0, 360.0],
        )
        expected = np.arange(binning.size, dtype=float).reshape(binning.shape)
        legacy_flat = binning.flatten_values(expected)
        self.assertFalse(np.array_equal(legacy_flat.reshape(binning.shape), expected))
        actual = MODULE.value_grid(
            {"flux_q2_coordinate": legacy_flat},
            "flux_q2_coordinate",
            binning,
        )
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
