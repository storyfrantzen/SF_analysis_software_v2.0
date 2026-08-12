from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from pathlib import Path


class IsolatedRenderError(RuntimeError):
    """Raised when the ROOT-free Matplotlib renderer cannot save a figure."""


def plot_context(dataset_tag: str = "", beam_energy: float | None = None) -> str:
    parts = []
    if dataset_tag:
        parts.append(f"Dataset: {dataset_tag}")
    if beam_energy is not None:
        parts.append(f"E_beam: {beam_energy:g} GeV")
    return " | ".join(parts)


def _apply_layout(fig, tight_layout: bool, context: str) -> None:
    if tight_layout:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93 if context else 0.95))
    else:
        fig.subplots_adjust(
            left=0.07,
            right=0.96,
            bottom=0.14,
            top=0.78 if context else 0.84,
            wspace=0.42,
        )


def _save_png_isolated(fig,
                       output_path: Path,
                       dpi: int,
                       metadata: dict[str, str],
                       tight_layout: bool,
                       context: str) -> None:
    """Render a pickled Matplotlib figure without importing PyROOT.

    ROOT and Matplotlib both load FreeType.  Some CLAS12 module combinations
    make Matplotlib's Agg renderer call the incompatible copy already loaded by
    PyROOT, producing ``FT_Render_Glyph ... raster overflow``.  A fresh Python
    process sees the same environment but never imports ROOT, matching the
    standalone Matplotlib smoke test that succeeds on the farm.
    """

    renderer = Path(__file__).with_name("render_plot.py")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".plot-render-", dir=output_path.parent
    ) as tmp:
        payload_path = Path(tmp) / "figure.pickle"
        rendered_path = Path(tmp) / output_path.name
        payload = {
            "figure": fig,
            "outputPath": str(rendered_path),
            "dpi": int(dpi),
            "metadata": metadata,
            "tightLayout": bool(tight_layout),
            "context": context,
        }
        try:
            with payload_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            raise IsolatedRenderError(f"Could not serialize calibration figure: {error}") from error

        result = subprocess.run(
            [sys.executable, str(renderer), str(payload_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise IsolatedRenderError(
                f"Isolated Matplotlib renderer exited with status {result.returncode}:\n{details}"
            )
        if not rendered_path.is_file():
            raise IsolatedRenderError(
                "Isolated Matplotlib renderer succeeded without writing its PNG"
            )
        rendered_path.replace(output_path)


def save_plot(fig,
              output_path: Path,
              title: str,
              dataset_tag: str = "",
              beam_energy: float | None = None,
              dpi: int = 180,
              tight_layout: bool = True) -> None:
    context = plot_context(dataset_tag, beam_energy)
    figure_title = title if not context else f"{title}\n{context}"
    fig.suptitle(figure_title)

    raster_metadata = {"Title": title}
    if dataset_tag:
        raster_metadata["DatasetTag"] = dataset_tag
    if beam_energy is not None:
        raster_metadata["BeamEnergy"] = f"{beam_energy:g}"

    vector_metadata = {"Title": title}
    if context:
        vector_metadata["Description"] = context

    def save_svg(path: Path) -> None:
        import matplotlib as mpl

        _apply_layout(fig, tight_layout, context)
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig.savefig(path, format="svg", metadata=vector_metadata)

    if output_path.suffix.lower() == ".svg":
        save_svg(output_path)
        return

    if output_path.suffix.lower() != ".png":
        _apply_layout(fig, tight_layout, context)
        fig.savefig(output_path, dpi=dpi, metadata=raster_metadata)
        return

    try:
        _save_png_isolated(
            fig,
            output_path,
            dpi,
            raster_metadata,
            tight_layout,
            context,
        )
        output_path.with_suffix(".svg").unlink(missing_ok=True)
    except IsolatedRenderError as error:
        message = str(error)
        if "FT_Render_Glyph" not in message or "raster overflow" not in message:
            raise

        output_path.unlink(missing_ok=True)
        vector_path = output_path.with_suffix(".svg")
        print(
            "[WARN] ROOT-free PNG renderer still overflowed; "
            f"saving SVG diagnostics instead: {vector_path}"
        )
        save_svg(vector_path)
