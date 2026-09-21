from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "compare_beam_spin_asymmetries.py"
SPEC = importlib.util.spec_from_file_location("compare_beam_spin_asymmetries", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompareBeamSpinTest(unittest.TestCase):
    def test_extended_fit_recovers_signal_and_null_harmonics(self) -> None:
        phi = np.arange(9.0, 360.0, 18.0)
        radians = np.deg2rad(phi)
        expected = np.array([0.03, 0.24, -0.02, 0.01])
        values = (
            expected[0]
            + expected[1] * np.sin(radians)
            + expected[2] * np.cos(radians)
            + expected[3] * np.sin(2.0 * radians)
        )[None, None, None, :]
        errors = np.full(values.shape, 0.02)
        coefficients, uncertainties, chi2, ndof = MODULE.fit_extended_harmonics(
            values, np.ones(values.shape) * 0.02,
            np.ones(values.shape, dtype=bool), phi,
        )
        np.testing.assert_allclose(coefficients.reshape(-1, 4)[0], expected, atol=1e-12)
        self.assertTrue(np.all(uncertainties > 0.0))
        self.assertAlmostEqual(float(chi2.item()), 0.0, places=12)
        self.assertEqual(int(ndof.item()), 16)


if __name__ == "__main__":
    unittest.main()
