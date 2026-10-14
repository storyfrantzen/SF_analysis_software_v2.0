from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from analysis.stage_aao_normalizations import selected_sidecars, source_lookup, stage


class StageAaoNormalizationsTests(unittest.TestCase):
    def test_direct_sources_stage_matching_norms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lund = root / "born_1.lund"
            norm = root / "born_1.norm"
            lund.write_text("lund\n")
            norm.write_text("sig_sum=1.0\n")
            manifest = root / "manifest.tsv"
            manifest.write_text(
                f"index\tevents\tsource\n1\t5000\t{lund}\n",
                encoding="utf-8",
            )
            selected = selected_sidecars(manifest, None)
            output = root / "staged"
            stage(selected, output)
            link = output / "00001_born_1.norm"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), norm.resolve())

    def test_original_source_list_remaps_frozen_lund(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original" / "rad_1.lund"
            original.parent.mkdir()
            original.write_text("lund\n")
            original.with_suffix(".norm").write_text("sig_sum=2.0\n")
            frozen = root / "frozen" / original.name
            manifest = root / "manifest.tsv"
            manifest.write_text(
                f"index\tevents\tsource\n1\t5001\t{frozen}\n",
                encoding="utf-8",
            )
            sources = root / "original.list"
            sources.write_text(f"{original}\n", encoding="utf-8")
            selected = selected_sidecars(manifest, source_lookup(sources))
            self.assertEqual(selected[0][3], original.with_suffix(".norm"))

    def test_missing_norm_is_rejected_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lund = root / "missing.lund"
            lund.write_text("lund\n")
            manifest = root / "manifest.tsv"
            manifest.write_text(
                f"index\tevents\tsource\n1\t5000\t{lund}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "missing nonempty"):
                selected_sidecars(manifest, None)


if __name__ == "__main__":
    unittest.main()
