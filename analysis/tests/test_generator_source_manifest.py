from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from analysis.eppi0.generators import (
    GeneratorCampaign,
    GeneratorProvenanceError,
    GeneratorRunRecord,
    GeneratorSourceManifest,
    build_generator_source_manifest,
)


def _run(root: Path, index: int, *, model: str = "pi0.amp2614") -> GeneratorRunRecord:
    lund = root / "born" / "lund_osg" / f"excluradpy_pi0_fa18_born_{index:03d}.lund"
    config = root / "born" / "prov" / f"born_{index:03d}.config.json"
    lund.parent.mkdir(parents=True, exist_ok=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    lund.write_text("4 0 0 0 0 0 0 0 0 0\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    return GeneratorRunRecord(
        provider="exclugen",
        mode="born",
        channel="pi0p",
        model=model,
        generator_revision="50a5074da87718d5588d429b57468489aca75e6c",
        generator_branch="main",
        dirty_state="clean",
        run_index=index,
        seed=1000 + index,
        lund_path=lund.resolve(),
        config_path=config.resolve(),
        requested_events=5000,
        observed_events=5000,
        sigma_value=0.9,
        sigma_units="nb",
        card_signature="a" * 64,
        config_sha256=f"{index:064x}",
        execution_path="farm",
        beam_energy_GeV=10.604,
        generation_vcut_GeV2=0.0,
        t_nucl_GeV2=-0.3,
        physics_card={"ebeam": 10.604},
    )


class GeneratorSourceManifestTests(unittest.TestCase):
    def test_builds_complete_exclugen_mapping_with_unit_response_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            production = root / "production"
            campaign = GeneratorCampaign(
                production_root=production,
                runs=(_run(production, 1), _run(production, 2)),
            )
            campaign_path = root / "campaign.json"
            campaign.write_manifest(campaign_path)
            hipo_dir = root / "dst_inb"
            hipo_dir.mkdir()
            hipos = [
                hipo_dir / "excluradpy_pi0_born_fa18_inb_bg50nA_001.hipo",
                hipo_dir / "excluradpy_pi0_born_fa18_inb_bg50nA_002.hipo",
            ]
            for path in hipos:
                path.touch()

            manifest = build_generator_source_manifest(
                campaign_path, reversed(hipos), mode="born"
            )
            self.assertEqual(
                [source.run.run_index for source in manifest.sources], [1, 2]
            )
            self.assertEqual(manifest.modes, ("born",))
            payload = manifest.to_manifest()
            self.assertNotIn("generator_weight", payload["sources"][0])
            self.assertEqual(payload["sources"][0]["sigma_units"], "nb")

            output = root / "sources.json"
            manifest.write_manifest(output)
            loaded = GeneratorSourceManifest.from_manifest(output)
            self.assertEqual(loaded.to_manifest(), manifest.to_manifest())
            self.assertEqual(
                loaded.source_for_hipo(hipos[1]).run.run_index, 2
            )

    def test_missing_final_hipo_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            production = root / "production"
            campaign = GeneratorCampaign(
                production_root=production,
                runs=(_run(production, 1), _run(production, 2)),
            )
            campaign_path = root / "campaign.json"
            campaign.write_manifest(campaign_path)
            hipo = root / "excluradpy_pi0_born_001.hipo"
            hipo.touch()
            with self.assertRaisesRegex(
                GeneratorProvenanceError, "no final HIPO mapping"
            ):
                build_generator_source_manifest(
                    campaign_path, [hipo], mode="born"
                )

    def test_unknown_source_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sources.json"
            path.write_text(json.dumps({"schema": "unknown", "sources": []}))
            with self.assertRaisesRegex(GeneratorProvenanceError, "unknown"):
                GeneratorSourceManifest.from_manifest(path)

    def test_conflicting_run_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _run(root, 1)
            second = _run(root, 1, model="pi0.wrong")
            first_value = first.to_dict()
            second_value = second.to_dict()
            first_value["hipo_path"] = str(root / "first_born_001.hipo")
            second_value["hipo_path"] = str(root / "second_born_001.hipo")
            path = root / "sources.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "sf-generator-source-manifest-v1",
                        "modes": ["born"],
                        "sources": [first_value, second_value],
                    }
                )
            )
            with self.assertRaisesRegex(GeneratorProvenanceError, "conflicting"):
                GeneratorSourceManifest.from_manifest(path)

    def test_unmapped_hipo_lookup_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _run(root, 1)
            mapped = root / "sample_born_001.hipo"
            mapped.touch()
            campaign = GeneratorCampaign(production_root=root, runs=(run,))
            campaign_path = root / "campaign.json"
            campaign.write_manifest(campaign_path)
            manifest = build_generator_source_manifest(
                campaign_path, [mapped], mode="born"
            )
            with self.assertRaisesRegex(GeneratorProvenanceError, "absent"):
                manifest.source_for_hipo(root / "missing_born_002.hipo")


if __name__ == "__main__":
    unittest.main()
