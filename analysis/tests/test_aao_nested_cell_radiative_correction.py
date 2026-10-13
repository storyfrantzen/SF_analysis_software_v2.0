from __future__ import annotations

import numpy as np

from analysis.aao_nested_cell_radiative_correction import (
    central_summary,
    parse_cell,
    validate_nested_cells,
    wrapped_phi_bin_indices,
)


def test_parse_and_validate_nested_cells() -> None:
    cells = (
        parse_cell("broad:1:2:0.25:0.38:0.2:0.4"),
        parse_cell("narrow:1.4:1.6:0.285:0.315:0.275:0.325"),
    )
    validate_nested_cells(cells, (1.5, 0.30, 0.30))
    assert cells[0].label == "broad"
    assert np.isclose(cells[0].volume, 0.026)
    assert cells[1].volume < cells[0].volume


def test_rejects_non_nested_or_off_point_cell() -> None:
    broad = parse_cell("broad:1:2:0.25:0.38:0.2:0.4")
    outside = parse_cell("outside:2:3:0.25:0.38:0.2:0.4")
    try:
        validate_nested_cells((broad, outside), (1.5, 0.30, 0.30))
    except ValueError as error:
        assert "does not contain" in str(error)
    else:
        raise AssertionError("off-point cell was accepted")


def test_central_summary_uses_two_central_and_edge_bins() -> None:
    edges = np.arange(0.0, 361.0, 30.0)
    values = np.ones(12)
    values[5:7] = 2.0
    errors = np.full(12, 0.2)
    result = central_summary(edges, values, errors)
    assert result["central_mean"] == 2.0
    assert result["edge_mean"] == 1.0
    assert result["central_over_edge"] == 2.0
    np.testing.assert_allclose(result["central_error"], 0.2 / np.sqrt(2.0))


def test_signed_radian_phi_is_wrapped_and_binned_in_degrees() -> None:
    edges = np.arange(0.0, 361.0, 30.0)
    angles = np.radians([-170.0, -10.0, 10.0, 170.0])
    np.testing.assert_array_equal(
        wrapped_phi_bin_indices(angles, edges),
        np.array([6, 11, 0, 5]),
    )
