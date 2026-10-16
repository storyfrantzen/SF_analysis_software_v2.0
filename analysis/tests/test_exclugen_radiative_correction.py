from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis.eppi0.binning import AnalysisBinning
from analysis.eppi0.event_sample import PI0_MASS, PROTON_MASS
from analysis.eppi0.generators import GeneratorCampaign, GeneratorRunRecord
from analysis.eppi0.phase_space import AnalysisPhaseSpace
from analysis.eppi0.radiative_correction import (
    DirectInternalRadiativeResult,
    ExclugenModeEstimate,
    EXCLUGEN_NORMALIZATION_METHOD,
    EXCLUGEN_UNCERTAINTY_MODEL,
    combined_correction_closure,
    compute_direct_internal_radiative_correction,
    compute_exclugen_combined_correction,
    compute_exclugen_radiative_closure,
    compute_exclugen_radiative_correction,
    estimate_exclugen_mode,
    exclugen_inelasticity_v,
    read_exclugen_lund,
)
from analysis.run_analysis import (
    _write_exclugen_radiative_closure,
    _write_exclugen_radiative_correction,
    parser,
)


BEAM_ENERGY = 10.604


def _vectors(*, high_v: bool = False) -> tuple[np.ndarray, np.ndarray]:
    electron = np.asarray(
        [0.5, 0.1, 7.0, math.sqrt(7.0**2 + 0.5**2 + 0.1**2)]
    )
    proton_momentum = np.asarray([-0.2, 0.3, 0.6])
    q_momentum = np.asarray([-electron[0], -electron[1], BEAM_ENERGY - electron[2]])
    missing_momentum = q_momentum - proton_momentum
    missing_energy = math.sqrt(PI0_MASS**2 + np.dot(missing_momentum, missing_momentum))
    proton_energy = BEAM_ENERGY + PROTON_MASS - electron[3] - missing_energy
    if high_v:
        proton_energy -= 0.5
    proton = np.asarray([*proton_momentum, proton_energy])
    return electron, proton


def _lund_event(*, high_v: bool = False, particle_count: int = 5) -> str:
    electron, proton = _vectors(high_v=high_v)
    pids = [11, 2212, 111, 22, 22, 22][:particle_count]
    vectors = {
        11: electron,
        2212: proton,
        111: np.asarray([0.0, 0.0, 0.0, PI0_MASS]),
        22: np.asarray([0.0, 0.0, 0.1, 0.1]),
    }
    rows = [f"{particle_count} 1 1 0 1 11 {BEAM_ENERGY} 2212 0 1"]
    for index, pid in enumerate(pids, start=1):
        px, py, pz, energy = vectors[pid]
        rows.append(
            f"{index} 0 1 {pid} 0 0 {px:.16g} {py:.16g} "
            f"{pz:.16g} {energy:.16g} 0 0 0 0"
        )
    return "\n".join(rows) + "\n"


def _record(
    directory: Path,
    mode: str,
    index: int,
    sigma: float,
    events: list[str],
    *,
    itarg: int = 1,
) -> GeneratorRunRecord:
    path = directory / f"{mode}_{index:03d}.lund"
    path.write_text("".join(events))
    return GeneratorRunRecord(
        provider="exclugen",
        mode=mode,
        channel="pi0p",
        model="pi0.amp2614",
        generator_revision="50a5074",
        generator_branch="main",
        dirty_state="clean",
        run_index=index,
        seed=810000000 + index,
        lund_path=path,
        config_path=directory / f"{mode}_{index:03d}.config.json",
        requested_events=len(events),
        observed_events=len(events),
        sigma_value=sigma,
        sigma_units="nb",
        card_signature="shared-card",
        config_sha256=f"sha256-{mode}-{index}",
        execution_path="farm",
        beam_energy_GeV=BEAM_ENERGY,
        generation_vcut_GeV2=4.0,
        t_nucl_GeV2=-0.3,
        physics_card={
            "model": "pi0.amp2614",
            "ebeam": BEAM_ENERGY,
            "W2": [4.0, 12.0],
            "Q2": [1.0, 5.0],
            "t": [-2.0, -0.1],
            "vcut": 4.0,
            "t_nucl": -0.3,
            "ihel": 0,
            "idecay": 1,
            "y": [0.0, 1.0],
            "xB": [0.0, 1.0],
            "iacc": 0,
            "vmin": 1.0e-4,
            "itarg": itarg,
            "vz": [-5.5, -0.5],
            "ptarg": 0.0,
            "fermi": "none",
            "theta_e": [0.0, 180.0],
        },
    )


def _binning() -> AnalysisBinning:
    return AnalysisBinning([0.0, 1.0], [0.0, 0.2], [0.0, 1.0], [0.0, 360.0])


class ExclugenRadiativeCorrectionTests(unittest.TestCase):
    def test_combined_correction_uses_only_r1_and_one_b0_center_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            radiative_record = _record(
                directory,
                "rad",
                0,
                2.0,
                [_lund_event(particle_count=6)],
                itarg=1,
            )
            campaign = GeneratorCampaign(directory, (radiative_record,))
            binning = AnalysisBinning(
                [1.4, 1.6],
                [0.28, 0.32],
                [0.2, 0.3],
                [0.0, 180.0],
            )
            estimate = ExclugenModeEstimate(
                cross_section=np.asarray([0.25]),
                standard_error=np.asarray([0.025]),
                counts=np.asarray([100.0]),
                effective_counts=np.asarray([100.0]),
                run_estimates=np.asarray([[0.25], [0.25]]),
                records=(radiative_record,),
                generated_events=1000,
                selected_events=900,
                in_range_events=100,
            )
            calls = []

            def evaluator(points: np.ndarray) -> np.ndarray:
                calls.append(points.copy())
                return np.full(points.shape[0], 5.0)

            with patch(
                "analysis.eppi0.radiative_correction.estimate_exclugen_mode",
                return_value=estimate,
            ):
                result = compute_exclugen_combined_correction(
                    campaign,
                    binning,
                    BEAM_ENERGY,
                    evaluator,
                    min_counts=1,
                    reference_samples_per_dimension=2,
                    phase_space=AnalysisPhaseSpace(q2_min=1.0, w_min=2.0),
                )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].shape, (1, 4))
            expected = 0.25 / (
                result.physical_volume.item()
                * result.born_center_electron.item()
            )
            self.assertAlmostEqual(result.correction_raw.item(), expected)
            self.assertTrue(result.reliable.item())

    def test_combined_factorization_cancels_b0_bin_integral(self) -> None:
        closure = combined_correction_closure(
            np.asarray([12.0, 6.0]),
            np.asarray([8.0, 4.0]),
            np.asarray([2.0, 2.0]),
            np.asarray([3.0, 1.0]),
        )
        np.testing.assert_allclose(closure.direct, [2.0, 3.0])
        np.testing.assert_allclose(
            closure.radiative_over_born_average, [1.5, 1.5]
        )
        np.testing.assert_allclose(
            closure.born_average_over_center, [4.0 / 3.0, 2.0]
        )
        np.testing.assert_allclose(closure.product, closure.direct)
        np.testing.assert_allclose(closure.residual, 0.0, atol=1.0e-15)

    def test_lund_reader_accepts_five_and_six_particle_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mixed.lund"
            path.write_text(
                _lund_event(particle_count=5) + _lund_event(particle_count=6)
            )
            electron, proton, events = read_exclugen_lund(path)
            self.assertEqual(events, 2)
            self.assertEqual(electron.shape, (2, 4))
            self.assertEqual(proton.shape, (2, 4))

    def test_per_run_mean_sem_ratio_and_kish_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runs = (
                _record(directory, "born", 0, 2.0, [_lund_event()]),
                _record(directory, "born", 1, 4.0, [_lund_event()]),
                _record(directory, "rad", 0, 1.0, [_lund_event(particle_count=6)]),
                _record(directory, "rad", 1, 5.0, [_lund_event(particle_count=6)]),
            )
            campaign = GeneratorCampaign(directory, runs)
            result = compute_exclugen_radiative_correction(
                campaign,
                _binning(),
                min_counts=1,
                phase_space=AnalysisPhaseSpace(),
            )
            np.testing.assert_allclose(result.born.cross_section, [3.0])
            np.testing.assert_allclose(result.born.standard_error, [1.0])
            np.testing.assert_allclose(result.radiative.cross_section, [3.0])
            np.testing.assert_allclose(result.radiative.standard_error, [2.0])
            np.testing.assert_allclose(result.c_rad_raw, [[[[1.0]]]])
            np.testing.assert_allclose(result.delta_c_raw, [[[[math.sqrt(5.0) / 3.0]]]])
            np.testing.assert_allclose(result.born.effective_counts, [36.0 / 20.0])
            np.testing.assert_allclose(result.radiative.effective_counts, [36.0 / 26.0])
            self.assertTrue(result.reliable.item())

    def test_offline_v_cut_is_radiative_only_and_not_renormalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            events = [_lund_event(), _lund_event(high_v=True)]
            born = _record(directory, "born", 0, 4.0, events)
            rad = _record(directory, "rad", 0, 4.0, events)
            campaign = GeneratorCampaign(directory, (born, rad))
            born_estimate = estimate_exclugen_mode(
                campaign,
                "born",
                _binning(),
                v_max=0.2,
                phase_space=AnalysisPhaseSpace(),
            )
            rad_estimate = estimate_exclugen_mode(
                campaign,
                "rad",
                _binning(),
                v_max=0.2,
                phase_space=AnalysisPhaseSpace(),
            )
            np.testing.assert_allclose(born_estimate.cross_section, [4.0])
            np.testing.assert_allclose(rad_estimate.cross_section, [2.0])
            self.assertEqual(born_estimate.selected_events, 2)
            self.assertEqual(rad_estimate.selected_events, 1)

    def test_four_sample_closure_promotes_rad1_over_born0(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root0 = directory / "itarg0"
            root1 = directory / "itarg1"
            root0.mkdir()
            root1.mkdir()
            campaign0 = GeneratorCampaign(
                root0,
                (
                    _record(root0, "born", 0, 2.0, [_lund_event()], itarg=0),
                    _record(root0, "born", 1, 2.0, [_lund_event()], itarg=0),
                    _record(
                        root0,
                        "rad",
                        0,
                        4.0,
                        [_lund_event(particle_count=6)],
                        itarg=0,
                    ),
                    _record(
                        root0,
                        "rad",
                        1,
                        4.0,
                        [_lund_event(particle_count=6)],
                        itarg=0,
                    ),
                ),
            )
            campaign1 = GeneratorCampaign(
                root1,
                (
                    _record(root1, "born", 0, 3.0, [_lund_event()], itarg=1),
                    _record(root1, "born", 1, 3.0, [_lund_event()], itarg=1),
                    _record(
                        root1,
                        "rad",
                        0,
                        6.0,
                        [_lund_event(particle_count=6)],
                        itarg=1,
                    ),
                    _record(
                        root1,
                        "rad",
                        1,
                        6.0,
                        [_lund_event(particle_count=6)],
                        itarg=1,
                    ),
                ),
            )
            result = compute_exclugen_radiative_closure(
                campaign0,
                campaign1,
                _binning(),
                min_counts=1,
                phase_space=AnalysisPhaseSpace(),
            )
            self.assertEqual(
                result.ratios["total"].numerator, "rad_itarg1"
            )
            self.assertEqual(
                result.ratios["total"].denominator, "born_itarg0"
            )
            np.testing.assert_allclose(
                result.ratios["total"].correction_raw, 3.0
            )
            np.testing.assert_allclose(
                result.ratios["internal_external_folded"].correction_raw, 2.0
            )
            np.testing.assert_allclose(
                result.ratios["external_born"].correction_raw, 1.5
            )
            np.testing.assert_allclose(
                result.ratios["internal_no_external"].correction_raw, 2.0
            )
            np.testing.assert_allclose(
                result.ratios["external_radiative"].correction_raw, 1.5
            )
            product = (
                result.ratios["internal_external_folded"].correction_raw
                * result.ratios["external_born"].correction_raw
            )
            np.testing.assert_allclose(result.ratios["total"].correction_raw, product)

    def test_four_sample_closure_rejects_wrong_itarg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root0 = directory / "itarg0"
            root1 = directory / "itarg1"
            root0.mkdir()
            root1.mkdir()
            campaign0 = GeneratorCampaign(
                root0,
                (
                    _record(root0, "born", 0, 1.0, [_lund_event()], itarg=0),
                    _record(
                        root0,
                        "rad",
                        0,
                        1.0,
                        [_lund_event(particle_count=6)],
                        itarg=0,
                    ),
                ),
            )
            campaign1 = GeneratorCampaign(
                root1,
                (
                    _record(root1, "born", 0, 1.0, [_lund_event()], itarg=0),
                    _record(
                        root1,
                        "rad",
                        0,
                        1.0,
                        [_lund_event(particle_count=6)],
                        itarg=0,
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "itarg=1 closure campaign"):
                compute_exclugen_radiative_closure(
                    campaign0,
                    campaign1,
                    _binning(),
                    min_counts=1,
                    phase_space=AnalysisPhaseSpace(),
                )

    def test_direct_internal_integration_uses_ratio_of_bin_integrals(self) -> None:
        binning = AnalysisBinning(
            [1.4, 1.6],
            [0.28, 0.32],
            [0.2, 0.3],
            [0.0, 360.0],
        )

        def evaluator(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            born = 1.0 + 4.0 * points[:, 0] + 0.1 * points[:, 3]
            eta = np.where(points[:, 3] < 180.0, 0.8, 1.2)
            return born, born * eta

        result = compute_direct_internal_radiative_correction(
            binning,
            BEAM_ENERGY,
            evaluator,
            samples_per_dimension=2,
            phase_space=AnalysisPhaseSpace(),
        )
        self.assertTrue(np.all(result.computed))
        self.assertTrue(np.all(result.reliable))
        self.assertGreater(float(result.correction_raw.item()), 1.0)
        self.assertLess(float(result.correction_raw.item()), 1.2)

    def test_inelasticity_fixture_separates_soft_and_hard_tail(self) -> None:
        electron, proton = _vectors()
        _, high_v_proton = _vectors(high_v=True)
        soft = exclugen_inelasticity_v(electron[None, :], proton[None, :], BEAM_ENERGY)
        hard = exclugen_inelasticity_v(
            electron[None, :], high_v_proton[None, :], BEAM_ENERGY
        )
        self.assertLess(abs(soft.item()), 1.0e-6)
        self.assertGreater(hard.item(), 0.2)

    def test_cli_defaults_to_exclugen_and_requires_explicit_aao(self) -> None:
        default = parser().parse_args(
            [
                "radiative-correction",
                "--config",
                "analysis.json",
                "--output",
                "correction.npz",
            ]
        )
        self.assertEqual(default.generator, "exclugen")
        self.assertIsNone(default.born)
        explicit = parser().parse_args(
            [
                "radiative-correction",
                "born",
                "rad",
                "--generator",
                "aao",
                "--config",
                "analysis.json",
                "--output",
                "correction.npz",
            ]
        )
        self.assertEqual(explicit.generator, "aao")

        closure = parser().parse_args(
            [
                "radiative-correction-closure",
                "--production-root-itarg0",
                "itarg0",
                "--production-root-itarg1",
                "itarg1",
                "--config",
                "analysis.json",
                "--output",
                "closure.npz",
            ]
        )
        self.assertEqual(closure.expected_runs, 200)
        self.assertEqual(closure.v_max, 0.2)
        self.assertEqual(closure.direct_samples_per_dimension, 0)

    def test_artifact_preserves_normalization_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / "analysis.json"
            config = {
                "beam_energy": BEAM_ENERGY,
                "binning": {
                    "Q2": [0.0, 1.0],
                    "xB": [0.0, 0.2],
                    "minus_t": [0.0, 1.0],
                    "phi_deg": [0.0, 360.0],
                },
            }
            config_path.write_text(json.dumps(config))
            runs = (
                _record(directory, "born", 0, 2.0, [_lund_event()]),
                _record(directory, "born", 1, 4.0, [_lund_event()]),
                _record(directory, "rad", 0, 1.0, [_lund_event(particle_count=6)]),
                _record(directory, "rad", 1, 5.0, [_lund_event(particle_count=6)]),
            )
            campaign = GeneratorCampaign(directory, runs)
            binning = _binning()
            phase_space = AnalysisPhaseSpace()
            result = compute_exclugen_radiative_correction(
                campaign,
                binning,
                min_counts=1,
                phase_space=phase_space,
            )
            output = directory / "correction.npz"
            _write_exclugen_radiative_correction(
                output,
                config_path,
                config,
                binning,
                phase_space,
                result,
                min_counts=1,
                max_files=None,
            )
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(str(data["generator_provider"]), "exclugen")
                self.assertEqual(str(data["normalization_units"]), "nb")
                self.assertEqual(
                    str(data["normalization_method"]), EXCLUGEN_NORMALIZATION_METHOD
                )
                self.assertEqual(
                    str(data["uncertainty_model"]), EXCLUGEN_UNCERTAINTY_MODEL
                )
                self.assertFalse(
                    bool(data["radiative_histogram_renormalized_after_v_cut"])
                )
                np.testing.assert_array_equal(data["born_run_indices"], [0, 1])
                np.testing.assert_allclose(data["radiative_sigma_nb"], [1.0, 5.0])
                manifest = json.loads(str(data["generator_manifest_json"]))
                self.assertEqual(manifest["generator"]["provider"], "exclugen")

    def test_closure_artifact_keeps_candidate_and_all_systematic_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root0 = directory / "itarg0"
            root1 = directory / "itarg1"
            root0.mkdir()
            root1.mkdir()
            config_path = directory / "analysis.json"
            config = {
                "beam_energy": BEAM_ENERGY,
                "binning": {
                    "Q2": [0.0, 1.0],
                    "xB": [0.0, 0.2],
                    "minus_t": [0.0, 1.0],
                    "phi_deg": [0.0, 360.0],
                },
            }
            config_path.write_text(json.dumps(config))
            campaign0 = GeneratorCampaign(
                root0,
                (
                    _record(root0, "born", 0, 2.0, [_lund_event()], itarg=0),
                    _record(root0, "born", 1, 2.0, [_lund_event()], itarg=0),
                    _record(
                        root0,
                        "rad",
                        0,
                        4.0,
                        [_lund_event(particle_count=6)],
                        itarg=0,
                    ),
                    _record(
                        root0,
                        "rad",
                        1,
                        4.0,
                        [_lund_event(particle_count=6)],
                        itarg=0,
                    ),
                ),
            )
            campaign1 = GeneratorCampaign(
                root1,
                (
                    _record(root1, "born", 0, 3.0, [_lund_event()], itarg=1),
                    _record(root1, "born", 1, 3.0, [_lund_event()], itarg=1),
                    _record(
                        root1,
                        "rad",
                        0,
                        6.0,
                        [_lund_event(particle_count=6)],
                        itarg=1,
                    ),
                    _record(
                        root1,
                        "rad",
                        1,
                        6.0,
                        [_lund_event(particle_count=6)],
                        itarg=1,
                    ),
                ),
            )
            result = compute_exclugen_radiative_closure(
                campaign0,
                campaign1,
                _binning(),
                min_counts=0,
                phase_space=AnalysisPhaseSpace(),
            )
            shape = _binning().shape
            direct = DirectInternalRadiativeResult(
                correction=np.full(shape, 2.1),
                correction_raw=np.full(shape, 2.1),
                reliable=np.ones(shape, dtype=bool),
                computed=np.ones(shape, dtype=bool),
                born_integral_nb=np.full(shape, 10.0),
                observed_integral_nb=np.full(shape, 21.0),
                n_physical=np.full(shape, 16, dtype=np.int64),
                n_valid=np.full(shape, 16, dtype=np.int64),
                n_failed=np.zeros(shape, dtype=np.int64),
                failure_fraction=np.zeros(shape),
                samples_per_dimension=2,
                bin_start=0,
                bin_stop=1,
            )
            output = directory / "closure.npz"
            _write_exclugen_radiative_closure(
                output,
                config_path,
                config,
                _binning(),
                AnalysisPhaseSpace(),
                result,
                min_counts=0,
                max_files=None,
                direct_result=direct,
                direct_metadata={"model_revision": "50a5074"},
            )
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(str(data["schema"]), "sf-exclugen-radiative-closure-v1")
                self.assertEqual(str(data["candidate_definition"]), "rad_itarg1 / born_itarg0")
                self.assertEqual(
                    str(data["candidate_status"]),
                    "validation_only_not_production_default",
                )
                np.testing.assert_allclose(data["C_rad_candidate"], 3.0)
                np.testing.assert_allclose(data["C_internal_external_folded"], 2.0)
                np.testing.assert_allclose(data["C_external_born"], 1.5)
                np.testing.assert_allclose(data["factorization_residual_raw"], 0.0)
                np.testing.assert_allclose(
                    data["C_internal_no_external_direct_raw"], 2.1
                )
                np.testing.assert_allclose(
                    data["direct_over_generated_internal_no_external_raw"], 1.05
                )
                self.assertEqual(
                    json.loads(str(data["direct_internal_metadata_json"]))[
                        "model_revision"
                    ],
                    "50a5074",
                )
                manifest0 = json.loads(str(data["campaign_itarg0_manifest_json"]))
                manifest1 = json.loads(str(data["campaign_itarg1_manifest_json"]))
                self.assertEqual(manifest0["generator"]["provider"], "exclugen")
                self.assertEqual(manifest1["generator"]["provider"], "exclugen")


if __name__ == "__main__":
    unittest.main()
