from __future__ import annotations

import unittest

import numpy as np

from scripts.calibration.eppi0_momentum_validation import (
    ELECTRON_MASS_GEV,
    PROTON_MASS_GEV,
    _four_vector,
    _unit_vectors,
)
from scripts.calibration.eppi0_proton_root_audit import (
    photon_branch_metrics,
    summarize_root_region,
)
from scripts.calibration.exclusive_particle_momentum import (
    proton_momentum_roots_eppi0,
)


class ProtonRootAuditTests(unittest.TestCase):
    def test_photon_direction_identifies_constructed_root(self) -> None:
        beam_energy = 6.535
        electron_p = np.asarray([4.2, 3.6, 4.8])
        electron_theta = np.deg2rad([18.0, 22.0, 14.0])
        electron_phi = np.deg2rad([10.0, -25.0, 35.0])
        proton_theta = np.deg2rad([32.0, 38.0, 27.0])
        proton_phi = electron_phi + np.deg2rad([165.0, 175.0, 170.0])
        roots, valid = proton_momentum_roots_eppi0(
            electron_p, electron_theta, electron_phi,
            proton_theta, proton_phi, beam_energy,
        )
        event = int(np.flatnonzero(np.sum(valid, axis=1) == 2)[0])
        electron = _four_vector(
            electron_p[event:event + 1],
            electron_theta[event:event + 1],
            electron_phi[event:event + 1],
            ELECTRON_MASS_GEV,
        )[0]
        hadronic = np.asarray([beam_energy + PROTON_MASS_GEV, 0.0, 0.0,
                               beam_energy]) - electron
        direction = _unit_vectors(
            proton_theta[event:event + 1], proton_phi[event:event + 1]
        )[0]
        chosen_root = roots[event, 0]
        proton = np.concatenate((
            [np.sqrt(chosen_root**2 + PROTON_MASS_GEV**2)],
            chosen_root * direction,
        ))
        pi0_direction = hadronic[1:] - proton[1:]
        pi0_direction /= np.linalg.norm(pi0_direction)
        gamma_theta = np.arccos(pi0_direction[2])
        gamma_phi = np.arctan2(pi0_direction[1], pi0_direction[0])
        arrays = {
            "electronTheta": electron_theta[event:event + 1],
            "electronPhi": electron_phi[event:event + 1],
            "protonTheta": proton_theta[event:event + 1],
            "protonPhi": proton_phi[event:event + 1],
            "gamma1P": np.asarray([0.5]),
            "gamma1Theta": np.asarray([gamma_theta]),
            "gamma1Phi": np.asarray([gamma_phi]),
            "gamma2P": np.asarray([0.5]),
            "gamma2Theta": np.asarray([gamma_theta]),
            "gamma2Phi": np.asarray([gamma_phi]),
        }
        angles, _ = photon_branch_metrics(
            arrays, electron_p[event:event + 1],
            roots[event:event + 1], valid[event:event + 1], beam_energy,
        )
        self.assertAlmostEqual(angles[0, 0], 0.0, places=6)
        self.assertGreater(angles[0, 1], angles[0, 0])

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
            photon_direction_angle_deg=np.asarray([
                [1.0, 10.0], [10.0, 1.0], [10.0, 1.0]
            ]),
            photon_momentum_residual_gev=np.asarray([
                [0.1, 1.0], [1.0, 0.1], [1.0, 0.1]
            ]),
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
        self.assertAlmostEqual(
            summary["photonDirectionCrossCheck"]["choiceAgreementFraction"],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            summary["photonMomentumCrossCheck"]["choiceAgreementFraction"],
            2.0 / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
