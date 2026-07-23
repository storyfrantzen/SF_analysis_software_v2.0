from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from scripts.repack_lund_for_osg import (
    repack_fixed_line_lund_files,
    repack_lund_files,
    repack_stratified_lund_files,
    write_chunk_provenance,
    write_manifests,
)


def event(particles: int, label: str) -> str:
    rows = [f"{particles} 1 1 0 0 0 0 0 0 0 # {label}\n"]
    rows.extend(f"{index + 1} 0 1 11 0 0 0 0 1 1 0 0 0 0\n" for index in range(particles))
    return "".join(rows)


class RepackLundForOsgTests(unittest.TestCase):
    def test_stratified_repack_never_mixes_strata_and_encodes_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            first = input_dir / "s00001__g0001.lund"
            second = input_dir / "s00002__g0001.lund"
            first.write_text(
                event(1, "s1-a") + event(1, "s1-b") + event(1, "s1-c"),
                encoding="utf-8",
            )
            second.write_text(event(1, "s2-a") + event(1, "s2-b"), encoding="utf-8")
            campaign = root / "campaign_weights.json"
            campaign.write_text(
                json.dumps(
                    {
                        "schema": "aao-born-bin-conditional-weights-v1",
                        "strata": [
                            {
                                "stratum_id": "s00001",
                                "flat_index": 1,
                                "pooled_event_weight_microbarn": 0.1,
                                "combined_sig_sum_microbarn": 0.3,
                                "generations": [
                                    {
                                        "generation_id": "g0001",
                                        "replica_index": 1,
                                        "events": 3,
                                        "lund_file": str(first),
                                    }
                                ],
                            },
                            {
                                "stratum_id": "s00002",
                                "flat_index": 2,
                                "pooled_event_weight_microbarn": 0.2,
                                "combined_sig_sum_microbarn": 0.4,
                                "generations": [
                                    {
                                        "generation_id": "g0001",
                                        "replica_index": 1,
                                        "events": 2,
                                        "lund_file": str(second),
                                    }
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            stats, provenance = repack_stratified_lund_files(
                [first, second],
                output_dir,
                events_per_file=2,
                prefix="aao_born",
                dry_run=False,
                campaign_weights=campaign,
            )
            write_chunk_provenance(
                output_dir, provenance, campaign_weights=campaign
            )

            self.assertEqual(stats.events, 5)
            self.assertEqual(stats.output_files, 3)
            chunks = sorted(output_dir.glob("*.lund"))
            self.assertEqual(
                [path.name for path in chunks],
                [
                    "aao_born__s00001__g0001__p000001.lund",
                    "aao_born__s00001__g0001__p000002.lund",
                    "aao_born__s00002__g0001__p000001.lund",
                ],
            )
            self.assertNotIn("s2-", chunks[0].read_text(encoding="utf-8"))
            self.assertNotIn("s2-", chunks[1].read_text(encoding="utf-8"))
            self.assertNotIn("s1-", chunks[2].read_text(encoding="utf-8"))
            provenance_json = json.loads(
                (output_dir / "chunk_provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["stratum_id"] for item in provenance_json["chunks"]],
                ["s00001", "s00001", "s00002"],
            )
            self.assertEqual(
                provenance_json["chunks"][0]["pooled_event_weight_microbarn"], 0.1
            )

    def test_repack_preserves_event_boundaries_and_writes_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            source = input_dir / "source.lund"
            source.write_text(
                event(1, "a") + event(2, "b") + event(1, "c") + event(3, "d") + event(1, "e"),
                encoding="utf-8",
            )

            stats = repack_lund_files(
                [source],
                output_dir,
                events_per_file=2,
                prefix="chunk",
                dry_run=False,
            )
            write_manifests(output_dir, prefix="chunk", jobs_per_submission=2, relative_paths=True)

            self.assertEqual(stats.events, 5)
            self.assertEqual(stats.output_files, 3)
            chunks = sorted(output_dir.glob("chunk_*.lund"))
            self.assertEqual([path.name for path in chunks], ["chunk_000001.lund", "chunk_000002.lund", "chunk_000003.lund"])
            self.assertIn("# a", chunks[0].read_text(encoding="utf-8"))
            self.assertIn("# b", chunks[0].read_text(encoding="utf-8"))
            self.assertIn("# c", chunks[1].read_text(encoding="utf-8"))
            self.assertIn("# d", chunks[1].read_text(encoding="utf-8"))
            self.assertIn("# e", chunks[2].read_text(encoding="utf-8"))

            manifests = sorted((output_dir / "manifests").glob("*.list"))
            self.assertEqual(len(manifests), 2)
            self.assertEqual(
                manifests[0].read_text(encoding="utf-8").splitlines(),
                ["chunk_000001.lund", "chunk_000002.lund"],
            )
            self.assertEqual(
                manifests[1].read_text(encoding="utf-8").splitlines(),
                ["chunk_000003.lund"],
            )

    def test_fixed_line_repack_coalesces_remainders_across_input_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "a.lund").write_text(event(1, "a") + event(1, "b") + event(1, "c"), encoding="utf-8")
            (input_dir / "b.lund").write_text(event(1, "d") + event(1, "e"), encoding="utf-8")

            stats = repack_fixed_line_lund_files(
                sorted(input_dir.glob("*.lund")),
                output_dir,
                events_per_file=4,
                lines_per_event=2,
                prefix="fixed",
                dry_run=False,
            )

            self.assertEqual(stats.events, 5)
            self.assertEqual(stats.output_files, 2)
            chunks = sorted(output_dir.glob("fixed_*.lund"))
            self.assertEqual(chunks[0].read_text(encoding="utf-8").count("#"), 4)
            self.assertEqual(chunks[1].read_text(encoding="utf-8").count("#"), 1)

    def test_fixed_line_dry_run_counts_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            source = input_dir / "source.lund"
            source.write_text(event(1, "a") + event(1, "b") + event(1, "c"), encoding="utf-8")

            stats = repack_fixed_line_lund_files(
                [source],
                output_dir,
                events_per_file=2,
                lines_per_event=2,
                prefix="fixed",
                dry_run=True,
            )

            self.assertEqual(stats.events, 3)
            self.assertEqual(stats.output_files, 2)
            self.assertFalse(output_dir.exists())

    def test_fixed_line_repack_does_not_write_empty_trailing_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            source = input_dir / "source.lund"
            source.write_text(event(1, "a") + event(1, "b") + event(1, "c") + event(1, "d"), encoding="utf-8")

            stats = repack_fixed_line_lund_files(
                [source],
                output_dir,
                events_per_file=2,
                lines_per_event=2,
                prefix="fixed",
                dry_run=False,
            )

            chunks = sorted(output_dir.glob("fixed_*.lund"))
            self.assertEqual(stats.events, 4)
            self.assertEqual(stats.output_files, 2)
            self.assertEqual(len(chunks), 2)
            self.assertTrue(all(path.stat().st_size > 0 for path in chunks))


if __name__ == "__main__":
    unittest.main()
