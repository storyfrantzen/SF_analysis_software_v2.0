from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.calibration.elastic_momentum import ElasticFitConfig
from scripts.calibration.elastic_systematic_scan import (
    compare_parameter_surfaces,
    make_one_at_a_time_variations,
    rebuild_existing_systematic_scan,
    run_systematic_scan,
)


def _parameters(coefficient: float) -> dict[str, object]:
    cells = [
        {
            "thetaMeanDeg": theta,
            "phiMeanDeg": phi,
        }
        for theta, phi in ((6.5, -10.0), (7.5, 0.0), (8.5, 10.0))
    ]
    return {
        "regions": [{
            "pid": 11,
            "detector": 1,
            "sector": 1,
            "thetaRangeDeg": [6.0, 9.0],
            "phiRangeDeg": [-30.0, 30.0],
            "thetaCenterDeg": 7.5,
            "thetaScaleDeg": 1.5,
            "phiCenterDeg": 0.0,
            "phiScaleDeg": 30.0,
            "basis": "polynomial",
            "terms": [{
                "thetaPower": 0,
                "phiPower": 0,
                "coefficient": coefficient,
            }],
            "fit": {"acceptedProfileCells": cells},
        }],
    }


class ElasticSystematicScanTests(unittest.TestCase):
    def test_one_at_a_time_variations_change_only_one_setting(self) -> None:
        cfg = ElasticFitConfig(
            beam_energy=10.604,
            missing_energy_max_gev=0.75,
            theta_min_deg=6.1,
            target_cell_entries=2_000,
        )
        variations = make_one_at_a_time_variations(
            cfg,
            missing_energy_values=[0.50, 0.75, 1.00],
            theta_min_values=[6.10, 6.50],
            target_cell_entry_values=[1_500, 2_000, 3_000],
        )
        self.assertEqual(
            [variation.name for variation in variations],
            [
                "nominal",
                "missing_energy_0p50",
                "missing_energy_1p00",
                "theta_min_6p50",
                "target_cell_entries_1500",
                "target_cell_entries_3000",
            ],
        )
        self.assertTrue(all(len(variation.overrides) <= 1 for variation in variations))

    def test_surface_comparison_uses_common_nominal_cells(self) -> None:
        comparisons = compare_parameter_surfaces(
            _parameters(0.004), _parameters(0.005)
        )
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["commonCells"], 3)
        self.assertEqual(comparisons[0]["commonCellFraction"], 1.0)
        self.assertAlmostEqual(comparisons[0]["meanDifference"], 0.001)
        self.assertAlmostEqual(comparisons[0]["rmsDifference"], 0.001)

    def test_scan_reuses_nominal_run_partition_and_writes_summary(self) -> None:
        cfg = ElasticFitConfig(
            beam_energy=10.604,
            torus=1,
            missing_energy_max_gev=0.75,
            theta_min_deg=6.1,
            target_cell_entries=2_000,
        )
        variations = make_one_at_a_time_variations(
            cfg,
            missing_energy_values=[0.50, 0.75],
            theta_min_values=[6.10],
            target_cell_entry_values=[2_000],
        )
        observed_run_groups: list[object] = []

        def fake_validation(*args: object, **kwargs: object) -> dict[str, object]:
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            run_groups = kwargs.get("run_groups")
            observed_run_groups.append(run_groups)
            coefficient = 0.004 + 0.001 * (
                float(args[1].missing_energy_max_gev) - 0.75
            )
            parameters = _parameters(coefficient)
            (output_dir / "recommended_mixed_parameters.json").write_text(
                json.dumps(parameters)
            )
            (output_dir / "constant").mkdir()
            (output_dir / "constant" / "pooled_parameters.json").write_text(
                json.dumps(parameters)
            )
            groups = run_groups or ((5423,), (5424,))
            region_summary = {
                "pid": 11,
                "detector": 1,
                "sector": 1,
                "region": "pid11_det1_sector1",
                "medianCellCenterRmsAfter": 0.001,
                "completeHeldOutBlockCoverage": True,
                "heldoutFolds": {
                    "A": {"medianCellCenterRmsAfter": 0.001},
                    "B": {"medianCellCenterRmsAfter": 0.001},
                },
                "foldSurfaceAgreement": {"rms": 0.0001},
            }
            result = {
                "selection": {"selectedCandidates": 10_000},
                "runBlockDefinition": (
                    "fixed-from-nominal-systematic-scan"
                    if run_groups is not None else "selected-candidate-target"
                ),
                "runBlocks": [
                    {
                        "index": index,
                        "fold": "A" if index % 2 == 0 else "B",
                        "runs": list(group),
                        "runMin": min(group),
                        "runMax": max(group),
                        "runCenter": float(np.mean(group)),
                        "selectedCandidates": 5_000,
                    }
                    for index, group in enumerate(groups)
                ],
                "models": {
                    "constant": {
                        "parameterFile": "constant/pooled_parameters.json",
                    },
                },
                "regionModelComparison": {"constant": [region_summary]},
                "recommendation": {
                    "model": "constant",
                    "status": "test",
                    "parameterFile": "recommended_mixed_parameters.json",
                    "regions": [{
                        "pid": 11,
                        "detector": 1,
                        "sector": 1,
                        "region": "pid11_det1_sector1",
                        "model": "constant",
                        "eligibleModels": ["constant"],
                        "medianHeldOutCellRms": 0.001,
                        "heldoutFoldCellRms": {"A": 0.001, "B": 0.001},
                        "foldSurfaceAgreement": {"rms": 0.0001},
                    }],
                },
            }
            (output_dir / "run_validation_report.json").write_text(
                json.dumps(result)
            )
            return result

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with patch(
                "scripts.calibration.elastic_systematic_scan.run_validation",
                side_effect=fake_validation,
            ):
                report = run_systematic_scan(
                    {"runNum": np.asarray([5423, 5424])},
                    cfg,
                    particle="electron",
                    models=["constant"],
                    block_target=1_000,
                    output_dir=output_dir,
                    dataset_tag="synthetic",
                    variations=variations,
                    make_summary_plots=False,
                )
            self.assertIsNone(observed_run_groups[0])
            self.assertEqual(observed_run_groups[1], ((5423,), (5424,)))
            self.assertTrue(report["fixedRunPartitionAcrossVariations"])
            self.assertTrue(report["allRegionModelAssignmentsStable"])
            self.assertEqual(
                report["conservativeRecommendation"]["regions"][0]["model"],
                "constant",
            )
            self.assertTrue(
                report["fixedModelStability"][0]["models"][0][
                    "eligibleEveryVariation"
                ]
            )
            self.assertTrue((output_dir / "systematic_scan_report.json").is_file())
            self.assertTrue((output_dir / "systematic_scan_summary.tsv").is_file())
            self.assertTrue((output_dir / "fixed_model_systematics.tsv").is_file())
            self.assertTrue((output_dir / "recommended_robust_parameters.json").is_file())

            rebuilt = rebuild_existing_systematic_scan(
                output_dir=output_dir,
                variations=variations,
                particle="electron",
                dataset_tag="synthetic",
                beam_energy=10.604,
                torus=1,
                models=["constant"],
                make_summary_plots=False,
            )
            self.assertEqual(rebuilt["schema"], "elastic_momentum_systematic_scan/v2")
            self.assertEqual(
                rebuilt["conservativeRecommendation"]["parameterFile"],
                "recommended_robust_parameters.json",
            )


if __name__ == "__main__":
    unittest.main()
