from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.eppi0.binning import AnalysisBinning
from analysis.eppi0.event_sample import PI0_MASS, PROTON_MASS
from analysis.eppi0.generators import GeneratorCampaign, GeneratorRunRecord
from analysis.eppi0.phase_space import AnalysisPhaseSpace
from analysis.eppi0.radiative_correction import (
    EXCLUGEN_NORMALIZATION_METHOD,
    EXCLUGEN_UNCERTAINTY_MODEL,
    compute_exclugen_radiative_correction,
    estimate_exclugen_mode,
    exclugen_inelasticity_v,
    read_exclugen_lund,
)
from analysis.run_analysis import _write_exclugen_radiative_correction, parser


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
        physics_card={},
    )


def _binning() -> AnalysisBinning:
    return AnalysisBinning([0.0, 1.0], [0.0, 0.2], [0.0, 1.0], [0.0, 360.0])


class ExclugenRadiativeCorrectionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
