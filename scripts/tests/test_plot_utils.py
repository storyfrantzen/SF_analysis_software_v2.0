from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from scripts.calibration import plot_utils


class PlotRendererFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        plot_utils._force_svg_output = False

    def tearDown(self) -> None:
        plot_utils._force_svg_output = False

    def test_raster_overflow_switches_current_and_later_plots_to_svg(self) -> None:
        raster_error = RuntimeError(
            "FT_Render_Glyph failed with error 0x62: raster overflow"
        )
        first_figure = MagicMock()
        first_figure.savefig.side_effect = [raster_error, None]
        second_figure = MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "first.png"
            second_path = Path(tmp) / "second.png"

            plot_utils.save_plot(
                first_figure,
                first_path,
                "First diagnostic",
                tight_layout=False,
            )
            plot_utils.save_plot(
                second_figure,
                second_path,
                "Second diagnostic",
                tight_layout=False,
            )

        self.assertEqual(first_figure.savefig.call_count, 2)
        self.assertEqual(first_figure.savefig.call_args_list[0].args[0], first_path)
        self.assertEqual(
            first_figure.savefig.call_args_list[1].args[0], first_path.with_suffix(".svg")
        )
        self.assertEqual(second_figure.savefig.call_count, 1)
        self.assertEqual(
            second_figure.savefig.call_args.args[0], second_path.with_suffix(".svg")
        )


if __name__ == "__main__":
    unittest.main()
