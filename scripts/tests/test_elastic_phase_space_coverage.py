from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.calibration.elastic_phase_space_coverage import (
    analyze_phase_space_coverage,
    run_coverage_diagnostic,
)


def _parameters() -> dict[str, object]:
    regions = []
    assignments = []
    for sector in range(1, 7):
        regions.append({
            "pid": 11,
            "detector": 1,
            "sector": sector,
            "thetaRangeDeg": [6.0, 8.0],
            "phiRangeDeg": [-20.0, 20.0],
            "thetaCenterDeg": 7.0,
            "thetaScaleDeg": 1.0,
            "phiCenterDeg": 0.0,
            "phiScaleDeg": 20.0,
            "basis": "polynomial",
            "terms": [{
                "thetaPower": 0,
                "phiPower": 0,
                "coefficient": 0.004,
            }],
            "supportCells": [
                {
                    "thetaRangeDeg": [6.0, 7.0],
                    "phiRangeDeg": [-20.0, 0.0],
                },
                {
                    "thetaRangeDeg": [7.0, 8.0],
                    "phiRangeDeg": [0.0, 20.0],
                },
            ],
        })
        assignments.append({
            "pid": 11,
            "detector": 1,
            "sector": sector,
            "model": "constant",
            "domainSensitive": sector in (2, 5),
        })
    return {
        "datasetTag": "synthetic-elastic",
        "beamEnergyGeV": 10.604,
        "calibrationRole": "test",
        "modelSelection": {"assignments": assignments},
        "regions": regions,
    }


def _arrays() -> dict[str, np.ndarray]:
    theta_pattern = np.asarray([6.5, 7.5, 5.5, 8.5, 6.5, 6.5])
    local_phi_pattern = np.asarray([-10.0, 10.0, -10.0, 10.0, 10.0, 25.0])
    sectors = np.repeat(np.arange(1, 7), theta_pattern.size)
    theta = np.tile(theta_pattern, 6)
    local_phi = np.tile(local_phi_pattern, 6)
    global_phi = local_phi + 60.0 * (sectors - 1)
    global_phi = (global_phi + 180.0) % 360.0 - 180.0
    entries = sectors.size
    return {
        "electronP": np.full(entries, 3.0),
        "electronTheta": np.deg2rad(theta),
        "electronPhi": np.deg2rad(global_phi),
        "electronDet": np.ones(entries, dtype=int),
        "electronSector": sectors,
        "Q2": np.full(entries, 2.0),
        "W": np.full(entries, 2.5),
    }


class ElasticPhaseSpaceCoverageTests(unittest.TestCase):
    def test_exact_support_categories_partition_each_sector(self) -> None:
        report, _ = analyze_phase_space_coverage(_arrays(), _parameters())
        self.assertEqual(report["overall"]["entries"], 36)
        self.assertEqual(report["overall"]["supportedEntries"], 12)
        self.assertAlmostEqual(report["overall"]["supportFraction"], 1.0 / 3.0)
        for region in report["regions"]:
            self.assertEqual(region["entries"], 6)
            self.assertEqual(region["supportedEntries"], 2)
            categories = region["coverageCategories"]
            self.assertEqual(categories["belowThetaRange"]["entries"], 1)
            self.assertEqual(categories["aboveThetaRange"]["entries"], 1)
            self.assertEqual(categories["outsidePhiRange"]["entries"], 1)
            self.assertEqual(
                categories["insideRangeOutsideSupportCells"]["entries"], 1
            )
            self.assertAlmostEqual(region["thetaBelowSplitFraction"], 1.0 / 6.0)

    def test_analysis_thresholds_are_applied_before_coverage(self) -> None:
        arrays = _arrays()
        arrays["Q2"][:6] = 0.5
        report, _ = analyze_phase_space_coverage(arrays, _parameters())
        self.assertEqual(report["selection"]["selectedEntries"], 30)
        self.assertEqual(report["regions"][0]["entries"], 0)
        self.assertEqual(report["regions"][1]["entries"], 6)

    def test_diagnostic_writes_machine_readable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            report = run_coverage_diagnostic(
                _arrays(),
                _parameters(),
                input_files=[Path("synthetic.root")],
                parameter_file=Path("parameters.json"),
                tree="sEvents",
                output_dir=output_dir,
                dataset_tag="synthetic",
                make_plot=False,
            )
            self.assertEqual(
                report["schema"], "elastic_momentum_phase_space_coverage/v1"
            )
            self.assertTrue((output_dir / "elastic_support_coverage.json").is_file())
            self.assertTrue((output_dir / "elastic_support_coverage.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
