from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.calibration import plot_utils


class PlotRendererIsolationTests(unittest.TestCase):
    def test_png_is_written_by_isolated_renderer(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "diagnostic.png"
            fig, ax = plt.subplots()
            ax.plot([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
            ax.set_xlabel("momentum [GeV]")
            plot_utils.save_plot(fig, output, "Renderer isolation test")
            plt.close(fig)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 100)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_glyph_overflow_in_isolated_renderer_falls_back_to_svg(self) -> None:
        raster_error = plot_utils.IsolatedRenderError(
            "FT_Render_Glyph failed with error 0x62: raster overflow"
        )
        figure = MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "diagnostic.png"
            with patch.object(
                plot_utils, "_save_png_isolated", side_effect=raster_error
            ):
                plot_utils.save_plot(
                    figure,
                    output,
                    "Fallback diagnostic",
                    tight_layout=False,
                )

        self.assertEqual(figure.savefig.call_count, 1)
        self.assertEqual(
            figure.savefig.call_args.args[0], output.with_suffix(".svg")
        )

    def test_unrelated_isolated_renderer_error_is_not_hidden(self) -> None:
        figure = MagicMock()
        output = Path("diagnostic.png")
        with patch.object(
            plot_utils,
            "_save_png_isolated",
            side_effect=plot_utils.IsolatedRenderError("pickle failed"),
        ):
            with self.assertRaisesRegex(plot_utils.IsolatedRenderError, "pickle failed"):
                plot_utils.save_plot(figure, output, "Broken diagnostic")


if __name__ == "__main__":
    unittest.main()
