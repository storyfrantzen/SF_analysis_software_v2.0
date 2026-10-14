from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from analysis.eppi0.generators import GeneratorCampaign, GeneratorProvenanceError
from analysis.eppi0.generators.exclugen import (
    count_lund_events,
    discover_exclugen_campaign,
    discover_exclugen_mode,
    sigma_nb,
)


def _config(mode: str, index: int, *, revision: str = "ec7367f") -> dict:
    card = {
        "y": [0.0, 0.9],
        "xB": [0.0, 1.0],
        "iacc": "none",
        "vmin": 0.0001,
        "itarg": 1,
        "vz": [-5.5, -0.5],
        "ptarg": 0.0,
        "fermi": "none",
        "theta_e": [5.0, 180.0],
        "model": "pi0.amp2614",
        "ebeam": 10.604,
        "nev": 1,
        "iborn": 1 if mode == "born" else 0,
        "W2": [3.24, 16.0],
        "Q2": [0.8, 10.0],
        "t": [-2.5, -0.001],
        "vcut": 0.0,
        "t_nucl": -0.3,
        "ihel": 0,
        "seed": (810000000 if mode == "born" else 820000000) + index,
        "idecay": 1,
        "outroot": f"/temporary/{mode}/{index:03d}",
    }
    return {
        "card": card,
        "stamp": {
            "code": "exclurad_py",
            "git_commit": revision,
            "git_branch": "HEAD",
            "sf": {
                "model": card["model"],
                "channel": "pi0",
                "target": "p",
                "detected": "p",
                "missing": "pi0",
                "Ebeam": card["ebeam"],
            },
            "run": {
                "mode": mode,
                "box": {
                    "Q2": card["Q2"],
                    "W2": card["W2"],
                    "t": card["t"],
                    "phi": [0.0, 2.0 * math.pi],
                },
                "vcut": card["vcut"],
                "Ebeam": card["ebeam"],
                "t_nucl_card": card["t_nucl"],
                "n_events": card["nev"],
                "seed": card["seed"],
            },
        },
        "report": {"sigma_nb": 1.0 + index / 100.0},
    }


def _event(particle_count: int) -> str:
    if particle_count < 2:
        raise ValueError("fixture needs an electron and proton")
    pids = [11, 2212, 111, 22, 22, 22][:particle_count]
    rows = [
        f"{particle_count} 1 1 0.0 1.0 11 10.604 2212 0 1.0",
    ]
    for particle_index, pid in enumerate(pids, start=1):
        rows.append(
            f"{particle_index} 0.0 1 {pid} 0 0 0.1 0.2 1.0 1.1 0.0 0 0 0"
        )
    return "\n".join(rows) + "\n"


def _write_run(
    root: Path,
    mode: str,
    index: int,
    *,
    revision: str = "ec7367f",
    particle_count: int | None = None,
    lund_events: int = 1,
    config: dict | None = None,
) -> tuple[Path, Path]:
    lund_dir = root / mode / "lund_osg"
    config_dir = root / mode / "prov"
    lund_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    lund_path = lund_dir / f"exclugen_pi0_{mode}_{index:03d}.lund"
    config_path = config_dir / f"{mode}_{index:03d}.config.json"
    topology_size = particle_count or (5 if mode == "born" else 6)
    lund_path.write_text(_event(topology_size) * lund_events)
    value = copy.deepcopy(config) if config is not None else _config(mode, index, revision=revision)
    config_path.write_text(json.dumps(value, indent=2) + "\n")
    return lund_path, config_path


class ExclugenGeneratorProviderTests(unittest.TestCase):
    def test_nested_sigma_nb_is_found(self) -> None:
        self.assertEqual(sigma_nb({"result": {"sigma_nb": 1.25}}), 1.25)

    def test_invalid_sigma_nb_is_rejected(self) -> None:
        invalid = (
            ({}, "no sigma_nb"),
            ({"sigma_nb": float("nan")}, "invalid"),
            ({"sigma_nb": float("inf")}, "invalid"),
            ({"sigma_nb": 0.0}, "invalid"),
            ({"sigma_nb": -1.0}, "invalid"),
            (
                {"report": {"sigma_nb": 1.0}, "summary": {"sigma_nb": 1.1}},
                "inconsistent",
            ),
        )
        for config, message in invalid:
            with self.subTest(config=config):
                with self.assertRaisesRegex(GeneratorProvenanceError, message):
                    sigma_nb(config)

    def test_five_and_six_particle_lund_events_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "topologies.lund"
            path.write_text(_event(5) + _event(6))
            self.assertEqual(count_lund_events(path), 2)

    def test_lund_config_run_index_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config_path = _write_run(root, "born", 1)
            config_path.rename(config_path.with_name("born_002.config.json"))
            with self.assertRaisesRegex(
                GeneratorProvenanceError, "LUND/config run-index mismatch"
            ):
                discover_exclugen_mode(root, "born", execution_path="farm")

    def test_wrong_event_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config("born", 1)
            config["card"]["nev"] = 2
            config["stamp"]["run"]["n_events"] = 2
            _write_run(root, "born", 1, config=config, lund_events=1)
            with self.assertRaisesRegex(GeneratorProvenanceError, "observed 1 events"):
                discover_exclugen_mode(root, "born", execution_path="farm")

    def test_mixed_model_and_card_settings_are_rejected(self) -> None:
        for variation, message in (("model", "model"), ("Q2", "card_signature")):
            with self.subTest(variation=variation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _write_run(root, "born", 1)
                config = _config("born", 2)
                if variation == "model":
                    config["card"]["model"] = "pi0.amp9999"
                    config["stamp"]["sf"]["model"] = "pi0.amp9999"
                else:
                    config["card"]["Q2"] = [0.9, 10.0]
                    config["stamp"]["run"]["box"]["Q2"] = [0.9, 10.0]
                _write_run(root, "born", 2, config=config)
                with self.assertRaisesRegex(GeneratorProvenanceError, message):
                    discover_exclugen_mode(root, "born", execution_path="farm")

    def test_born_radiative_revision_mismatch_requires_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, "born", 1, revision="ec7367f")
            _write_run(root, "rad", 1, revision="deadbee")
            with self.assertRaisesRegex(GeneratorProvenanceError, "revision mismatch"):
                discover_exclugen_campaign(root, execution_path="farm")
            campaign = discover_exclugen_campaign(
                root,
                execution_path="farm",
                allow_revision_mismatch=True,
            )
            self.assertEqual(len(campaign.runs), 2)

    def test_historical_dirty_revision_is_recorded_not_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, "born", 1, revision="ac3e572-dirty")
            _write_run(root, "rad", 1, revision="ac3e572-dirty")
            campaign = discover_exclugen_campaign(
                root,
                execution_path="farm",
                expected_revision="ac3e572",
            )
            self.assertEqual(
                {run.generator_revision for run in campaign.runs}, {"ac3e572"}
            )
            self.assertEqual({run.dirty_state for run in campaign.runs}, {"dirty"})

    def test_mixed_dirty_state_requires_narrow_historical_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root, "born", 1, revision="50a5074")
            _write_run(root, "born", 2, revision="50a5074")
            _write_run(root, "rad", 1, revision="50a5074")
            _write_run(root, "rad", 2, revision="50a5074-dirty")
            with self.assertRaisesRegex(GeneratorProvenanceError, "dirty_state"):
                discover_exclugen_campaign(root, execution_path="farm")
            campaign = discover_exclugen_campaign(
                root,
                execution_path="farm",
                allow_mixed_dirty_state=True,
            )
            self.assertEqual(
                {run.generator_revision for run in campaign.runs}, {"50a5074"}
            )
            self.assertEqual(
                {run.dirty_state for run in campaign.runs}, {"clean", "dirty"}
            )
            generator = campaign.to_manifest()["generator"]
            self.assertEqual(generator["revisions"], ["50a5074"])
            self.assertEqual(generator["dirty_state_counts"], {"clean": 3, "dirty": 1})

    def test_campaign_records_units_hashes_and_execution_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "production"
            _write_run(root, "born", 1)
            _write_run(root, "rad", 1)
            campaign = discover_exclugen_campaign(
                root,
                execution_path="farm",
                expected_runs=1,
                expected_model="pi0.amp2614",
                expected_revision="ec7367f",
            )
            self.assertEqual(campaign.provider, "exclugen")
            self.assertEqual(campaign.channel, "pi0p")
            self.assertTrue(all(run.sigma_units == "nb" for run in campaign.runs))
            self.assertTrue(all(run.execution_path == "farm" for run in campaign.runs))
            self.assertTrue(all(len(run.config_sha256) == 64 for run in campaign.runs))
            self.assertTrue(all(len(run.card_signature) == 64 for run in campaign.runs))

            manifest_path = directory / "campaign.json"
            campaign.write_manifest(manifest_path)
            loaded = GeneratorCampaign.from_manifest(manifest_path)
            self.assertEqual(loaded.to_manifest(), campaign.to_manifest())

    def test_manifest_conflicting_summary_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "production"
            _write_run(root, "born", 1)
            _write_run(root, "rad", 1)
            campaign = discover_exclugen_campaign(root, execution_path="farm")
            manifest = campaign.to_manifest()
            manifest["generator"]["model"] = "pi0.wrong"
            path = directory / "campaign.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(GeneratorProvenanceError, "conflicts"):
                GeneratorCampaign.from_manifest(path)

    def test_unknown_manifest_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign.json"
            path.write_text(json.dumps({"schema": "unknown"}))
            with self.assertRaisesRegex(GeneratorProvenanceError, "unknown"):
                GeneratorCampaign.from_manifest(path)


if __name__ == "__main__":
    unittest.main()
