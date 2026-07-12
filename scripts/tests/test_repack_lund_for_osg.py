from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.repack_lund_for_osg import repack_fixed_line_lund_files, repack_lund_files, write_manifests


def event(particles: int, label: str) -> str:
    rows = [f"{particles} 1 1 0 0 0 0 0 0 0 # {label}\n"]
    rows.extend(f"{index + 1} 0 1 11 0 0 0 0 1 1 0 0 0 0\n" for index in range(particles))
    return "".join(rows)


class RepackLundForOsgTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
