from __future__ import annotations

from pathlib import Path


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
              dpi: int = 180) -> None:
    context = plot_context(dataset_tag, beam_energy)
    figure_title = title if not context else f"{title}\n{context}"
    fig.suptitle(figure_title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93 if context else 0.95))

    metadata = {"Title": title}
    if dataset_tag:
        metadata["DatasetTag"] = dataset_tag
    if beam_energy is not None:
        metadata["BeamEnergy"] = f"{beam_energy:g}"
    fig.savefig(output_path, dpi=dpi, metadata=metadata)
