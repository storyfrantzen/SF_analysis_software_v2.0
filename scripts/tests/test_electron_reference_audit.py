from __future__ import annotations

import unittest

import numpy as np

from scripts.calibration.electron_reference_audit import (
    METHOD_LABELS,
    _profile_rows,
    electron_w,
    sector_mapping_candidates,
)
from scripts.calibration.elastic_momentum import (
    PROTON_MASS_GEV,
    ElasticFitConfig,
    elastic_electron_momentum,
)


class ElectronReferenceAuditTests(unittest.TestCase):
    def test_sector_mapping_scan_covers_detector_symmetries(self) -> None:
        candidates = sector_mapping_candidates()
        mappings = {
            tuple(item["coefficientSectorByObservedSector"])
            for item in candidates
        }
        self.assertEqual(len(candidates), 12)
        self.assertEqual(len(mappings), 12)
        self.assertEqual(candidates[0]["name"], "identity")
        self.assertEqual(
            candidates[0]["coefficientSectorByObservedSector"],
            [1, 2, 3, 4, 5, 6],
        )

    def test_elastic_momentum_reconstructs_proton_mass(self) -> None:
        theta = np.deg2rad(np.asarray([6.0, 10.0, 15.0, 25.0]))
        momentum = elastic_electron_momentum(theta, 6.535)
        np.testing.assert_allclose(
            electron_w(momentum, theta, 6.535),
            PROTON_MASS_GEV,
            atol=2.0e-7,
            rtol=0.0,
        )

    def test_profile_cells_share_identical_event_groups_across_methods(self) -> None:
        rng = np.random.default_rng(17)
        entries = 400
        theta = np.r_[np.full(200, 6.5), np.full(200, 8.5)]
        phi = np.r_[np.full(200, 2.5), np.full(200, 62.5)]
        sectors = np.r_[np.ones(200, dtype=int), np.full(200, 2, dtype=int)]
        runs = np.r_[np.full(200, 5900), np.full(200, 5980)]
        base_w = PROTON_MASS_GEV + rng.normal(0.002, 0.003, entries)
        w_by_method = {
            method: base_w + index * 1.0e-4
            for index, method in enumerate(METHOD_LABELS)
        }
        rows = _profile_rows(
            sample="synthetic",
            theta_deg=theta,
            global_phi_deg=phi,
            sectors=sectors,
            run_numbers=runs,
            run_class_by_run={5900: "P3", 5980: "P4"},
            w_by_method=w_by_method,
            support=np.ones(entries, dtype=bool),
            cfg=ElasticFitConfig(
                beam_energy=6.535,
                min_bin_entries=20,
                peak_seed_half_width=0.02,
            ),
            phi_cell_width_deg=5.0,
        )
        all_selected = [
            row for row in rows
            if row["runClass"] == "all" and row["cohort"] == "allSelected"
        ]
        self.assertEqual(len(all_selected), 2 * len(METHOD_LABELS))
        self.assertEqual({row["rawEntries"] for row in all_selected}, {200})
        self.assertEqual({row["method"] for row in all_selected}, set(METHOD_LABELS))

    def test_phi_cell_width_must_tile_full_azimuth(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide 360"):
            _profile_rows(
                sample="synthetic",
                theta_deg=np.asarray([7.5, 7.6]),
                global_phi_deg=np.asarray([0.0, 0.0]),
                sectors=np.asarray([1, 1]),
                run_numbers=np.asarray([5900, 5900]),
                run_class_by_run=None,
                w_by_method={"none": np.asarray([0.94, 0.94])},
                support=np.ones(2, dtype=bool),
                cfg=ElasticFitConfig(beam_energy=6.535, min_bin_entries=2),
                phi_cell_width_deg=7.0,
            )


if __name__ == "__main__":
    unittest.main()
