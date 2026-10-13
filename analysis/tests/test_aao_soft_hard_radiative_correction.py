from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.aao_soft_hard_radiative_correction import (
    integrated_cross_sections,
    lund_clusters,
    manifest_paths,
    ratio_and_error,
)


def test_soft_and_hard_contributions_add_to_total() -> None:
    born = np.array([100.0, 200.0])
    born_sumw2 = np.array([100.0, 200.0])
    soft = np.array([30.0, 80.0])
    hard = np.array([20.0, 20.0])
    soft_sumw2 = np.array([40.0, 100.0])
    hard_sumw2 = np.array([30.0, 25.0])
    normalization = 2.5
    c_soft, _ = ratio_and_error(soft, soft_sumw2, born, born_sumw2, normalization)
    c_hard, _ = ratio_and_error(hard, hard_sumw2, born, born_sumw2, normalization)
    c_total, _ = ratio_and_error(
        soft + hard,
        soft_sumw2 + hard_sumw2,
        born,
        born_sumw2,
        normalization,
    )
    np.testing.assert_allclose(c_total, c_soft + c_hard)


def test_manifest_rejects_duplicate_files(tmp_path: Path) -> None:
    source = tmp_path / "one.lund"
    source.write_text("1 1 1\n", encoding="utf-8")
    manifest = tmp_path / "files.list"
    manifest.write_text(f"{source}\n{source}\n", encoding="utf-8")
    try:
        manifest_paths(manifest)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate manifest paths were accepted")


def test_integrated_cross_sections(tmp_path: Path) -> None:
    artifact = tmp_path / "normalization.npz"
    np.savez(
        artifact,
        born_integrated_cross_section=0.4,
        radiative_integrated_cross_section=1.2,
    )
    assert integrated_cross_sections(artifact) == (0.4, 1.2)


def test_lund_clusters_preserve_multiplicity_and_photon_energy(tmp_path: Path) -> None:
    def event(photon_energy: float) -> str:
        return "".join(
            [
                "4 1 1 0 0 0 0 0 0 1\n",
                "1 0 1 11 0 0 1 0 7 7.0710678 0 0 0\n",
                "2 1 1 2212 0 0 .2 .3 .4 1.083 0 0 0\n",
                "3 0 1 111 0 0 -.2 -.3 .5 .65 0 0 0\n",
                f"4 0 1 22 0 0 0 0 {photon_energy} {photon_energy} 0 0 0\n",
            ]
        )

    source = tmp_path / "radiative.lund"
    soft = event(1.0e-5)
    source.write_text(soft + soft + event(0.02), encoding="utf-8")
    clusters = list(lund_clusters([source], radiative=True, progress_files=0))
    assert [cluster[3] for cluster in clusters] == [2, 1]
    assert [cluster[2] for cluster in clusters] == [1.0e-5, 0.02]
