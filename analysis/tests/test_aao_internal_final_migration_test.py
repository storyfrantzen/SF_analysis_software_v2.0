from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from analysis.aao_internal_final_migration_test import (
    CATEGORY_LABELS,
    PhaseSpace,
    accumulate_surveys,
    parse_cell,
    summaries,
    wrap_degrees,
    write_artifacts,
)


def _write_survey(directory: Path) -> None:
    directory.mkdir()
    (directory / "aao_rad.norm").write_text(
        "ntries=100\nsurvey_ntrials_requested=100\n", encoding="utf-8"
    )
    fields = [
        "final_valid", "candidate_status", "intreg", "energy_e_lund",
        "energy_gamma", "q2_hard", "xb_hard", "minus_t_hard",
        "phi_cm_deg", "q2_observed", "xb_observed", "minus_t_observed",
        "phi_observed_deg", "w_observed", "trial_xsec_observed_microbarn",
    ]
    rows = [
        dict(zip(fields, (1, 0, 6, 7.0, 0.001, 1.5, 0.30, 0.30, 15,
                          1.5, 0.30, 0.30, 180, 2.1, 2.0), strict=True)),
        dict(zip(fields, (1, 0, 1, 7.0, 0.2, 2.2, 0.45, 0.60, 180,
                          1.5, 0.30, 0.30, 180, 2.1, 1.0), strict=True)),
        dict(zip(fields, (1, 0, 1, 7.0, 0.2, 1.5, 0.30, 0.30, 15,
                          1.5, 0.30, 0.30, 15, 2.1, 1.0), strict=True)),
    ]
    with (directory / "aao_rad.survey.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        stream.write("# schema=aao-rad-survey-v2\n")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_wrap_degrees() -> None:
    np.testing.assert_allclose(
        wrap_degrees(np.array((0.0, 180.0, 190.0, 350.0))),
        np.array((0.0, -180.0, -170.0, -10.0)),
    )


def test_central_sources_are_split_by_internal_origin(tmp_path: Path) -> None:
    survey = tmp_path / "survey"
    _write_survey(survey)
    cell = parse_cell("cell:1:2:0.25:0.38:0.2:0.4")
    phi_edges = np.arange(0.0, 361.0, 30.0)
    delta_edges = np.linspace(-180.0, 180.0, 73)
    phase_space = PhaseSpace(1.0, 2.0, 2.0, None, 10.604)
    result, metadata = accumulate_surveys(
        [survey],
        [cell],
        phi_edges,
        delta_edges,
        phase_space,
    )
    assert metadata["total_proposals"] == 100
    assert result.rows[0, 0] == 3
    assert result.sumw[0, 0] == 4.0
    assert result.sumw[0, CATEGORY_LABELS.index("soft_intreg6")] == 2.0
    assert result.sumw[0, CATEGORY_LABELS.index("hard_intreg1to5")] == 2.0
    summary = summaries(result, [cell], 100)[0]["categories"][0]
    assert summary["final_central_fraction"] == 0.75
    assert summary["central_from_hard_central_fraction"] == 1.0 / 3.0
    assert summary["central_from_same_hard_cell_fraction"] == 2.0 / 3.0
    assert summary["central_from_same_phi_bin_fraction"] == 1.0 / 3.0
    output = tmp_path / "result"
    payload = write_artifacts(
        output,
        result,
        [cell],
        phi_edges,
        delta_edges,
        metadata,
        phase_space,
    )
    assert payload["schema"] == "aao-internal-final-migration-test-v1"
    for suffix in (".npz", ".csv", ".json", ".pdf", ".png"):
        assert output.with_suffix(suffix).stat().st_size > 0
