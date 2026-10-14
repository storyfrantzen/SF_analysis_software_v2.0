from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from analysis.aao_calibration_phi_convention import (
    CalibrationSpec,
    correction_and_error,
    load_calibration,
    main,
    parse_calibration,
    wrap_degrees,
)


def test_wrap_degrees_uses_half_open_interval() -> None:
    values = np.asarray([-540.0, -180.0, -1.0, 0.0, 179.0, 180.0, 540.0])
    np.testing.assert_allclose(
        wrap_degrees(values),
        np.asarray([-180.0, -180.0, -1.0, 0.0, 179.0, -180.0, -180.0]),
    )


def test_parse_calibration() -> None:
    result = parse_calibration("mm020=/tmp/envelope.json")
    assert result.label == "mm020"
    assert result.report == Path("/tmp/envelope.json")


def test_parse_calibration_rejects_missing_label() -> None:
    try:
        parse_calibration("/tmp/envelope.json")
    except ValueError as error:
        assert "LABEL=REPORT" in str(error)
    else:
        raise AssertionError("malformed calibration specification was accepted")


def test_correction_and_error() -> None:
    correction, uncertainty = correction_and_error(
        np.asarray([2.0, 4.0]),
        np.asarray([0.2, 0.4]),
        np.asarray([1.0, 2.0]),
        np.asarray([0.1, 0.2]),
    )
    np.testing.assert_allclose(correction, [2.0, 2.0])
    np.testing.assert_allclose(uncertainty, [2.0**0.5 * 0.2, 2.0**0.5 * 0.2])


def test_load_calibration_preserves_integral_and_maps_angles(tmp_path: Path) -> None:
    run_json = tmp_path / "replica001.json"
    run_json.write_text("{}\n", encoding="utf-8")
    csv_path = run_json.with_suffix(".calibration.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        stream.write("# schema=aao-rad-mode3-calibration-v1\n")
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "integrand_corrected",
                "phi_cm_deg",
                "phi_observed_deg",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "integrand_corrected": 2.0,
                "phi_cm_deg": 10.0,
                "phi_observed_deg": 190.0,
            }
        )
        writer.writerow(
            {
                "integrand_corrected": 4.0,
                "phi_cm_deg": 200.0,
                "phi_observed_deg": 20.0,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"runs": [{"run_json": run_json.name}]}) + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "envelope_calibration.json"
    report.write_text(
        json.dumps(
            {
                "schema": "aao-rad-mode3-envelope-v1",
                "source_manifest": manifest.name,
                "proposals": 10,
                "integrated_cross_section_microbarn": 0.6,
                "integrated_cross_section_sem_microbarn": 0.1,
                "runs": [{"phase_volume": 1.0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    phi_edges = np.linspace(0.0, 360.0, 13)
    result = load_calibration(CalibrationSpec("synthetic", report), phi_edges)
    expected_observed = np.zeros(12)
    expected_observed[[0, 6]] = [4.0, 2.0]
    expected_internal = np.zeros(12)
    expected_internal[[0, 6]] = [2.0, 4.0]
    np.testing.assert_allclose(result.observed_sumw, expected_observed)
    np.testing.assert_allclose(result.internal_sumw, expected_internal)
    np.testing.assert_allclose(result.shifted_sumw, expected_internal)
    assert result.calibration_rows == 2
    assert result.delta_near_zero_fraction == 0.0
    assert result.delta_near_180_fraction == 1.0

    born_reference = tmp_path / "born_reference.npz"
    np.savez_compressed(
        born_reference,
        phi_edges_deg=phi_edges,
        H_born=np.ones(12),
        H_born_cluster_sumw2=np.ones(12),
        born_topology_events=12,
        born_integrated_cross_section=1.0,
    )
    output = tmp_path / "phi_test"
    assert (
        main(
            [
                "--calibration",
                f"synthetic={report}",
                "--born-reference",
                str(born_reference),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    for suffix in (".npz", ".csv", ".pdf", ".png", ".json"):
        assert output.with_suffix(suffix).stat().st_size > 0
