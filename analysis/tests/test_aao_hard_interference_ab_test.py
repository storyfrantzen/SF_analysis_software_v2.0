from __future__ import annotations

import json

import numpy as np

from analysis.aao_hard_interference_ab_test import (
    NEW_HARD_DENOMINATOR,
    NEW_LT_NORMALIZATION,
    OLD_HARD_DENOMINATOR,
    OLD_LT_NORMALIZATION,
    _correction,
    calibration_csv_by_replica,
    override_input_records,
    patch_hard_denominator,
    patch_lt_normalization,
    patch_variant,
)


def test_patch_changes_only_the_selected_fixed_form_block() -> None:
    source = f"before\n{OLD_HARD_DENOMINATOR}\nafter\n"
    patched = patch_hard_denominator(source)
    assert OLD_HARD_DENOMINATOR not in patched
    assert patched.count(NEW_HARD_DENOMINATOR) == 1
    assert patched == f"before\n{NEW_HARD_DENOMINATOR}\nafter\n"


def test_patch_rejects_ambiguous_source() -> None:
    try:
        patch_hard_denominator("no matching Fortran block")
    except ValueError as error:
        assert "found 0" in str(error)
    else:
        raise AssertionError("missing source block was accepted")


def test_lt_patch_changes_exactly_soft_and_hard_coefficients() -> None:
    source = f"soft {OLD_LT_NORMALIZATION}\nhard {OLD_LT_NORMALIZATION}\n"
    patched = patch_lt_normalization(source)
    assert OLD_LT_NORMALIZATION not in patched
    assert patched.count(NEW_LT_NORMALIZATION) == 2


def test_lt_patch_rejects_an_unexpected_source_revision() -> None:
    try:
        patch_lt_normalization(f"only {OLD_LT_NORMALIZATION}\n")
    except ValueError as error:
        assert "found 1" in str(error)
    else:
        raise AssertionError("source with one LT factor was accepted")


def test_correction_propagates_errors() -> None:
    value, error = _correction(
        np.asarray([2.0]),
        np.asarray([0.2]),
        np.asarray([1.0]),
        np.asarray([0.1]),
    )
    np.testing.assert_allclose(value, [2.0])
    np.testing.assert_allclose(error, [2.0**0.5 * 0.2])


def test_calibration_csvs_are_resolved_through_manifest(tmp_path) -> None:
    run_json = tmp_path / "runs" / "replica000.json"
    run_json.parent.mkdir()
    csv_path = run_json.with_suffix(".calibration.csv")
    csv_path.write_text("nonempty\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "replica_index": 0,
                        "run_json": "runs/replica000.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "envelope_calibration.json"
    report = {
        "source_manifest": "manifest.json",
        "runs": [{"replica_index": 0, "ntries": 100}],
    }
    assert calibration_csv_by_replica(report, report_path) == {
        0: csv_path.resolve()
    }


def test_input_overrides_replace_physics_records() -> None:
    source = "\n".join(f"{index} ! record {index}" for index in range(17)) + "\n"
    patched, values = override_input_records(
        source,
        electron_helicity_flag=0,
        target_length_cm=0.25,
        minimum_photon_energy=0.02,
    )
    records = [line.split("!", 1)[0].strip() for line in patched.splitlines()]
    assert records[1] == "0"
    assert records[6] == "0.25"
    assert records[14] == "0.02"
    assert values == {
        "electron_helicity_flag": 0,
        "target_length_cm": 0.25,
        "minimum_photon_energy_GeV": 0.02,
    }


def test_source_variants_apply_only_requested_diagnostics() -> None:
    from analysis.aao_hard_interference_ab_test import (
        INCOMING_EXTERNAL_HOOK,
        NOMINAL_HARD_MODULATION,
        OUTGOING_EXTERNAL_HOOK,
    )

    source = "\n".join(
        (INCOMING_EXTERNAL_HOOK, OUTGOING_EXTERNAL_HOOK, NOMINAL_HARD_MODULATION)
    )
    control, changes = patch_variant(source, "control")
    assert control == source
    assert changes == ["none: unmodified-source rebuild control"]

    external, changes = patch_variant(source, "external-off")
    assert "eloss=0." in external
    assert NOMINAL_HARD_MODULATION in external
    assert len(changes) == 1

    no_interference, changes = patch_variant(source, "interference-off")
    assert "123   sigf=sigf" in no_interference
    assert "eloss=0." not in no_interference
    assert len(changes) == 1

    lt_source = f"{OLD_LT_NORMALIZATION}\n{OLD_LT_NORMALIZATION}\n"
    lt_fixed, changes = patch_variant(lt_source, "lt-normalization")
    assert lt_fixed.count(NEW_LT_NORMALIZATION) == 2
    assert len(changes) == 1

    combined_source = f"{OLD_HARD_DENOMINATOR}\n{lt_source}"
    combined, changes = patch_variant(
        combined_source, "denominator-lt-normalization"
    )
    assert combined.count(NEW_HARD_DENOMINATOR) == 1
    assert combined.count(NEW_LT_NORMALIZATION) == 2
    assert len(changes) == 2
