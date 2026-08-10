from __future__ import annotations

from pathlib import Path


_force_svg_output = False


def plot_context(dataset_tag: str = "", beam_energy: float | None = None) -> str:
    parts = []
    if dataset_tag:
        parts.append(f"Dataset: {dataset_tag}")
    if beam_energy is not None:
        parts.append(f"E_beam: {beam_energy:g} GeV")
    return " | ".join(parts)


def save_plot(fig,
              output_path: Path,
              title: str,
              dataset_tag: str = "",
              beam_energy: float | None = None,
              dpi: int = 180,
              tight_layout: bool = True) -> None:
    global _force_svg_output

    context = plot_context(dataset_tag, beam_energy)
    figure_title = title if not context else f"{title}\n{context}"
    fig.suptitle(figure_title)
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

        with mpl.rc_context({"svg.fonttype": "none"}):
            fig.savefig(path, format="svg", metadata=vector_metadata)

    if output_path.suffix.lower() == ".svg":
        save_svg(output_path)
        return

    if _force_svg_output and output_path.suffix.lower() == ".png":
        vector_path = output_path.with_suffix(".svg")
        print(f"[WARN] Saving vector diagnostic after PNG renderer failure: {vector_path}")
        save_svg(vector_path)
        return

    try:
        fig.savefig(output_path, dpi=dpi, metadata=raster_metadata)
    except RuntimeError as error:
        message = str(error)
        is_glyph_raster_overflow = (
            output_path.suffix.lower() == ".png"
            and "FT_Render_Glyph" in message
            and "raster overflow" in message
        )
        if not is_glyph_raster_overflow:
            raise

        _force_svg_output = True
        output_path.unlink(missing_ok=True)
        vector_path = output_path.with_suffix(".svg")
        print(
            "[WARN] PNG text rendering overflowed; "
            f"saving SVG diagnostics instead: {vector_path}"
        )
        save_svg(vector_path)
