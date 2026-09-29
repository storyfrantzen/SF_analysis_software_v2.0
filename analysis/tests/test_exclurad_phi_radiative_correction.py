from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.eppi0.binning import from_config
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
from analysis.run_analysis import _plot_radiative_correction_diagnostics


class ExcluradPhiRadiativeCorrectionTests(unittest.TestCase):
    def test_correction_overlay_rejects_different_phi_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            primary = directory / "primary.npz"
            overlay = directory / "overlay.npz"
            shape = (1, 1, 1, 2)
            common = {
                "C_rad": np.ones(shape),
                "delta_C": np.full(shape, 0.1),
                "reliable": np.ones(shape, dtype=bool),
                "support_overlap": np.ones(shape, dtype=bool),
                "support_status": np.zeros(shape, dtype=np.uint8),
                "H_born": np.full(shape, 10.0),
                "H_rad": np.full(shape, 10.0),
                "H_born_effective": np.full(shape, 10.0),
                "H_rad_effective": np.full(shape, 10.0),
                "q2_edges": np.asarray([1.0, 2.0]),
                "xb_edges": np.asarray([0.1, 0.2]),
                "t_edges": np.asarray([0.1, 0.2]),
                "min_counts": 5,
            }
            np.savez(primary, **common, phi_edges=np.asarray([0.0, 180.0, 360.0]))
            np.savez(overlay, **common, phi_edges=np.asarray([0.0, 90.0, 360.0]))
            with self.assertRaisesRegex(ValueError, "overlay phi_edges"):
                _plot_radiative_correction_diagnostics(
                    primary,
                    directory / "comparison.pdf",
                    overlay_correction_path=overlay,
                )

    def test_coarse_quilt_config_has_requested_shape(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        config = repository / (
            "configs/diagnostics/rga/10.604/"
            "exclurad_radiative_correction_coarse_5x5x4x12.json"
        )
        binning = from_config(config)
        self.assertEqual(binning.shape, (5, 5, 4, 12))
        np.testing.assert_allclose(binning.phi_edges, np.linspace(0.0, 360.0, 13))

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
