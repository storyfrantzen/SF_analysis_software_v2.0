from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eppi0.binning import AnalysisBinning
from eppi0.bin_centering import (
    AaoExecutableEvaluator,
    aao_sigma0_to_reduced_cross_section,
)
from eppi0.model_comparison import (
    TabulatedModelEvaluator,
    average_model_over_bins,
)
from eppi0.structure_functions import (
    epsilon_from_xb_q2,
    harmonic_to_structure_functions,
)
from run_analysis import (
    _model_fit_payload,
    command_model_comparison_plots,
    command_model_prediction,
    command_model_prediction_merge,
    command_structure_functions,
)


class StructureFunctionTests(unittest.TestCase):
    def test_full_covariance_is_scaled_by_jacobian(self) -> None:
        parameters = np.asarray([[[[2.0, 3.0, -4.0]]]])
        covariance = np.asarray(
            [[[[[4.0, 0.5, -0.2], [0.5, 9.0, 0.3], [-0.2, 0.3, 16.0]]]]]
        )
        epsilon = np.asarray([[[0.8]]])
        result = harmonic_to_structure_functions(parameters, covariance, epsilon)
        scales = np.asarray(
            [
                2.0 * np.pi,
                2.0 * np.pi / np.sqrt(2.0 * 0.8 * 1.8),
                2.0 * np.pi / 0.8,
            ]
        )
        np.testing.assert_allclose(result.values[0, 0, 0], parameters[0, 0, 0] * scales)
        np.testing.assert_allclose(
            result.covariance[0, 0, 0],
            covariance[0, 0, 0] * scales[:, None] * scales[None, :],
        )
        self.assertTrue(result.valid[0, 0, 0])

    def test_invalid_epsilon_masks_output(self) -> None:
        parameters = np.ones((2, 3))
        covariance = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
        result = harmonic_to_structure_functions(parameters, covariance, [0.0, 1.2])
        self.assertFalse(np.any(result.valid))
        self.assertTrue(np.all(np.isnan(result.values)))

    def test_negative_covariance_diagonal_is_invalid(self) -> None:
        covariance = np.eye(3)[None, ...]
        covariance[0, 1, 1] = -1.0
        result = harmonic_to_structure_functions(
            np.ones((1, 3)), covariance, np.asarray([0.8])
        )
        self.assertFalse(result.valid[0])
        self.assertTrue(np.all(np.isnan(result.uncertainties[0])))


class AaoCrossSectionConventionTests(unittest.TestCase):
    def test_sigma0_is_converted_from_angular_microbarns_to_reduced_nb(self) -> None:
        points = np.asarray([[0.3, 2.0, -0.2, 37.0]])
        converted = aao_sigma0_to_reduced_cross_section(
            points, np.asarray([1.0]), channel=1
        )
        np.testing.assert_allclose(converted, [333.1022415446145], rtol=1.0e-12)

    def test_evaluator_converts_raw_executable_output(self) -> None:
        points = np.asarray([[0.3, 2.0, -0.2, 37.0]])
        evaluator = AaoExecutableEvaluator(
            Path("/unused/aao_xsec"), 6.535, channel=1, workers=1
        )
        with patch("eppi0.bin_centering._call_aao_xsec_job", return_value=0.003):
            converted = evaluator(points)
        np.testing.assert_allclose(
            converted,
            aao_sigma0_to_reduced_cross_section(points, np.asarray([0.003])),
        )

    def test_unsupported_channel_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported AAO channel"):
            aao_sigma0_to_reduced_cross_section(
                np.asarray([[0.3, 2.0, -0.2, 0.0]]),
                np.asarray([1.0]),
                channel=2,
            )


class TabulatedModelTests(unittest.TestCase):
    def test_structure_function_table_builds_reduced_phi_cross_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "model.csv"
            table.write_text(
                "xB,Q2,minus_t,sigma_T,sigma_L,sigma_LT,sigma_TT\n"
                "0.3,2.0,0.2,10.0,2.0,3.0,-4.0\n",
                encoding="utf-8",
            )
            evaluator = TabulatedModelEvaluator(table, 6.535, interpolation="nearest")
            points = np.asarray(
                [[0.3, 2.0, -0.2, 0.0], [0.3, 2.0, -0.2, 90.0]]
            )
            values = evaluator(points)
            epsilon = float(epsilon_from_xb_q2(2.0, 0.3, 6.535))
            sigma_u = 10.0 + epsilon * 2.0
            expected = np.asarray(
                [
                    sigma_u + np.sqrt(2.0 * epsilon * (1.0 + epsilon)) * 3.0 + epsilon * -4.0,
                    sigma_u - epsilon * -4.0,
                ]
            ) / (2.0 * np.pi)
            np.testing.assert_allclose(values, expected)

    def test_reduced_table_is_periodic_in_phi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "model.csv"
            table.write_text(
                "xB,Q2,t,phi_deg,reduced_cross_section\n"
                "0.3,2.0,-0.2,0,5\n"
                "0.3,2.0,-0.2,90,3\n"
                "0.3,2.0,-0.2,180,1\n"
                "0.3,2.0,-0.2,270,3\n",
                encoding="utf-8",
            )
            evaluator = TabulatedModelEvaluator(table, 6.535, interpolation="linear")
            values = evaluator(
                np.asarray(
                    [[0.3, 2.0, -0.2, 0.0], [0.3, 2.0, -0.2, 360.0]]
                )
            )
            np.testing.assert_allclose(values, [5.0, 5.0])


class ModelAveragingTests(unittest.TestCase):
    def test_forward_average_applies_extraction_flux_and_transform(self) -> None:
        binning = AnalysisBinning(
            [1.9, 2.1],
            [0.29, 0.31],
            [0.15, 0.25],
            [0.0, 90.0, 180.0, 270.0, 360.0],
        )
        result = average_model_over_bins(
            binning,
            6.535,
            lambda points: np.full(points.shape[0], 5.0),
            samples_per_dimension=1,
            transform=np.full(binning.shape, 2.0),
        )
        self.assertTrue(np.all(result.reliable))
        np.testing.assert_allclose(result.extracted_bin_average, 5.0)
        np.testing.assert_allclose(result.reduced_cross_section, 2.5)

    def test_model_harmonics_use_the_data_harmonic_reference(self) -> None:
        phi_edges = np.arange(0.0, 360.0 + 45.0, 45.0)
        phi = np.deg2rad(0.5 * (phi_edges[:-1] + phi_edges[1:]))
        values = (
            2.0
            + 0.3 * np.cos(phi)
            - 0.2 * np.cos(2.0 * phi)
            + 0.02 * np.cos(3.0 * phi)
        )[None, None, None]
        result = SimpleNamespace(
            reduced_cross_section=values,
            reliable=np.ones(values.shape, dtype=bool),
            q2_reference=np.broadcast_to(
                np.linspace(1.8, 2.2, values.shape[-1]), values.shape
            ),
            xb_reference=np.broadcast_to(
                np.linspace(0.28, 0.32, values.shape[-1]), values.shape
            ),
        )
        q2_reference = np.asarray([[[2.15]]])
        xb_reference = np.asarray([[[0.31]]])
        payload = _model_fit_payload(
            result,
            phi_edges,
            6.535,
            q2_harmonic_reference=q2_reference,
            xb_harmonic_reference=xb_reference,
        )
        expected = epsilon_from_xb_q2(q2_reference, xb_reference, 6.535)
        np.testing.assert_allclose(payload["epsilon"], expected)
        np.testing.assert_allclose(payload["q2_harmonic_reference"], q2_reference)
        np.testing.assert_allclose(payload["xb_harmonic_reference"], xb_reference)
        self.assertTrue(payload["fit_success"][0, 0, 0])
        self.assertTrue(payload["quality_mask"][0, 0, 0])
        self.assertEqual(payload["quality_status"][0, 0, 0], 0)
        self.assertLess(payload["chi2_ndf"][0, 0, 0], 1.0e-3)


class ModelCommandIntegrationTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        config = root / "config.json"
        cross_section = root / "cross_section.npz"
        table = root / "model.csv"
        config.write_text(
            json.dumps(
                {
                    "beam_energy": 6.535,
                    "phase_space": {"Q2_min": 1.0, "W_min": 2.0},
                    "binning": {
                        "Q2": [1.9, 2.1],
                        "xB": [0.29, 0.31],
                        "minus_t": [0.15, 0.25, 0.35],
                        "phi_deg": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0],
                    },
                }
            ),
            encoding="utf-8",
        )
        shape = (1, 1, 2, 8)
        q2 = np.full(shape, 2.0)
        xb = np.full(shape, 0.3)
        epsilon = float(epsilon_from_xb_q2(2.0, 0.3, 6.535))
        phi = np.deg2rad(np.arange(22.5, 360.0, 45.0))
        sigma_u = 10.0 + epsilon * 2.0
        curve = (
            sigma_u
            + np.sqrt(2.0 * epsilon * (1.0 + epsilon)) * 3.0 * np.cos(phi)
            + epsilon * -4.0 * np.cos(2.0 * phi)
        ) / (2.0 * np.pi)
        values = np.broadcast_to(curve, shape).copy()
        np.savez_compressed(
            cross_section,
            reduced_cross_section=values,
            uncertainty=np.full(shape, 0.1),
            final_validity_mask=np.ones(shape, dtype=bool),
            flux_q2_coordinate=q2,
            flux_xb_coordinate=xb,
            bin_centering_C_BC=np.full(shape, 2.0),
            q2_edges=np.asarray([1.9, 2.1]),
            xb_edges=np.asarray([0.29, 0.31]),
            t_edges=np.asarray([0.15, 0.25, 0.35]),
            phi_edges=np.arange(0.0, 360.0 + 45.0, 45.0),
        )
        table.write_text(
            "xB,Q2,minus_t,sigma_T,sigma_L,sigma_LT,sigma_TT\n"
            "0.3,2.0,0.2,10,2,3,-4\n"
            "0.3,2.0,0.3,10,2,3,-4\n",
            encoding="utf-8",
        )
        return config, cross_section, table

    def _prediction_args(
        self,
        config: Path,
        cross_section: Path,
        table: Path,
        output: Path,
        chunk: int,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            config=config,
            cross_section=cross_section,
            output=output,
            model_name="synthetic",
            aao_exe=None,
            table=table,
            interpolation="nearest",
            N=1,
            workers=None,
            chunk_size=1,
            progress_chunks=0,
            bin_start=0,
            bin_stop=None,
            bin_chunks=2,
            bin_chunk_index=chunk,
            theory=5,
            channel=1,
            resonance=0,
            max_failure_fraction=0.0,
            verbose_failures=False,
            all_bins=False,
        )

    def test_partial_prediction_merge_structure_export_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cross_section, table = self._write_inputs(root)
            partials = [root / "part0.npz", root / "part1.npz"]
            for chunk, output in enumerate(partials):
                command_model_prediction(
                    self._prediction_args(
                        config, cross_section, table, output, chunk
                    )
                )
            merged = root / "model.npz"
            command_model_prediction_merge(
                SimpleNamespace(partials=partials, output=merged)
            )
            model = np.load(merged, allow_pickle=False)
            self.assertTrue(np.all(model["reliable"]))
            self.assertTrue(np.all(model["fit_success"]))
            self.assertTrue(np.all(model["quality_mask"]))
            self.assertEqual(str(model["model_name"]), "synthetic")
            epsilon = float(epsilon_from_xb_q2(2.0, 0.3, 6.535))
            expected_u_after_transform = (10.0 + epsilon * 2.0) / 2.0
            np.testing.assert_allclose(
                model["structure_functions"][..., 0],
                expected_u_after_transform,
                rtol=1.0e-12,
                atol=1.0e-12,
            )

            harmonics = root / "harmonics.npz"
            np.savez_compressed(
                harmonics,
                parameters=model["parameters"],
                covariance=np.broadcast_to(np.eye(3) * 0.01, (1, 1, 2, 3, 3)),
                fit_success=np.ones((1, 1, 2), dtype=bool),
                quality_mask=np.ones((1, 1, 2), dtype=bool),
                q2_edges=model["q2_edges"],
                xb_edges=model["xb_edges"],
                t_edges=model["t_edges"],
                phi_edges=model["phi_edges"],
            )
            structure_output = root / "structure.npz"
            command_structure_functions(
                SimpleNamespace(
                    harmonics=harmonics,
                    cross_section=cross_section,
                    config=config,
                    output=structure_output,
                    include_quality_rejected=False,
                )
            )
            structure = np.load(structure_output, allow_pickle=False)
            self.assertTrue(np.all(structure["valid"]))
            self.assertEqual(structure["covariance"].shape, (1, 1, 2, 3, 3))

            plot_dir = root / "plots"
            command_model_comparison_plots(
                SimpleNamespace(
                    cross_section=cross_section,
                    harmonics=harmonics,
                    models=[merged],
                    config=config,
                    output_dir=plot_dir,
                    include_quality_rejected=False,
                )
            )
            self.assertTrue((plot_dir / "model_comparison_vs_phi.pdf").is_file())
            self.assertTrue(
                (plot_dir / "model_comparison_structure_functions.pdf").is_file()
            )
            self.assertTrue((plot_dir / "model_comparison_summary.csv").is_file())

    def test_plots_reject_legacy_unconverted_aao_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cross_section, table = self._write_inputs(root)
            model_path = root / "model.npz"
            command_model_prediction(
                self._prediction_args(config, cross_section, table, model_path, 0)
            )
            with np.load(model_path, allow_pickle=False) as source:
                payload = {name: np.asarray(source[name]) for name in source.files}
            payload["model_source_kind"] = "aao_executable"
            payload.pop("aao_cross_section_conversion", None)
            np.savez_compressed(model_path, **payload)
            harmonics = root / "harmonics.npz"
            model = np.load(model_path, allow_pickle=False)
            np.savez_compressed(
                harmonics,
                parameters=model["parameters"],
                covariance=np.broadcast_to(np.eye(3) * 0.01, (1, 1, 2, 3, 3)),
                fit_success=np.ones((1, 1, 2), dtype=bool),
                quality_mask=np.ones((1, 1, 2), dtype=bool),
                q2_edges=model["q2_edges"],
                xb_edges=model["xb_edges"],
                t_edges=model["t_edges"],
                phi_edges=model["phi_edges"],
            )
            with self.assertRaisesRegex(ValueError, "regenerate"):
                command_model_comparison_plots(
                    SimpleNamespace(
                        cross_section=cross_section,
                        harmonics=harmonics,
                        models=[model_path],
                        config=config,
                        output_dir=root / "plots",
                        include_quality_rejected=False,
                    )
                )


if __name__ == "__main__":
    unittest.main()
