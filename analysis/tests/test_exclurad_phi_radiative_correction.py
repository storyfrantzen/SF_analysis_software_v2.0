from __future__ import annotations

import unittest

import numpy as np

from analysis.exclurad_phi_radiative_correction import (
    inelasticity_v,
    ratio_and_error,
    sigma_nb,
    trento_phi,
)
from analysis.exclurad_4d_radiative_correction import (
    dis_kinematics,
    minus_t,
    weighted_effective_count,
)


class ExcluradPhiRadiativeCorrectionTests(unittest.TestCase):
    def test_nested_sigma_nb_is_found(self) -> None:
        self.assertEqual(sigma_nb({"result": {"sigma_nb": 1.25}}), 1.25)

    def test_inconsistent_nested_sigma_nb_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            sigma_nb({"a": {"sigma_nb": 1.0}, "b": {"sigma_nb": 2.0}})

    def test_ratio_error_propagation(self) -> None:
        ratio, error = ratio_and_error(
            np.asarray([0.9]),
            np.asarray([0.03]),
            np.asarray([1.0]),
            np.asarray([0.04]),
        )
        np.testing.assert_allclose(ratio, [0.9])
        np.testing.assert_allclose(error, [0.9 * np.hypot(0.03 / 0.9, 0.04)])

    def test_born_exclusive_event_has_zero_inelasticity(self) -> None:
        beam_energy = 10.6
        electron = np.asarray([[0.5, 0.0, 7.0, np.hypot(7.0, 0.5)]])
        # Construct the recoil so the missing four-vector is an on-shell pi0
        # at rest.  This is not a physical scattering angle test; it isolates
        # the invariant used by the offline v cut.
        proton_energy = beam_energy + 0.93827208816 - electron[0, 3] - 0.1349768
        proton = np.asarray([[-electron[0, 0], 0.0, beam_energy - electron[0, 2], proton_energy]])
        np.testing.assert_allclose(
            inelasticity_v(electron, proton, beam_energy),
            [0.0],
            atol=3e-6,
        )

    def test_trento_phi_is_wrapped(self) -> None:
        electron = np.asarray([[1.0, 0.0, 8.0, np.sqrt(65.0)]])
        proton = np.asarray([[0.0, 1.0, 0.5, 1.5]])
        phi = trento_phi(electron, proton, 10.6)
        self.assertTrue(np.isfinite(phi[0]))
        self.assertGreaterEqual(phi[0], 0.0)
        self.assertLess(phi[0], 2.0 * np.pi)

    def test_weighted_effective_count(self) -> None:
        # Two entries with weights 1 and 2 have Kish support 9/5.
        np.testing.assert_allclose(
            weighted_effective_count(np.asarray([3.0]), np.asarray([5.0])),
            [1.8],
        )

    def test_dis_and_minus_t_are_finite(self) -> None:
        electron = np.asarray([[0.5, 0.1, 7.0, np.sqrt(49.26)]])
        proton = np.asarray([[0.2, -0.1, 0.6, 1.15]])
        q2, xb = dis_kinematics(electron, 10.6)
        mt = minus_t(proton)
        self.assertTrue(np.all(np.isfinite(q2)))
        self.assertTrue(np.all(np.isfinite(xb)))
        self.assertTrue(np.all(np.isfinite(mt)))


if __name__ == "__main__":
    unittest.main()
