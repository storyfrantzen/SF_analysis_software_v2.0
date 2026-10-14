from __future__ import annotations

import numpy as np

from analysis.klimenko_photon_map_sensitivity import (
    map_values,
    topology_name,
    wrapped_phi_degrees,
)


def test_wrapped_phi_uses_published_half_open_domain() -> None:
    radians = np.deg2rad(np.asarray([-31.0, -30.0, 0.0, 329.0, 330.0]))
    np.testing.assert_allclose(
        wrapped_phi_degrees(radians),
        np.asarray([329.0, -30.0, 0.0, 329.0, -30.0]),
    )


def test_map_values_use_displayed_bins_and_reject_outside_domain() -> None:
    values = map_values(
        np.asarray([0.6, 1.2, 0.4, 2.0]),
        np.asarray([10.0, 21.0, 10.0, 40.0]),
        np.asarray([0.0, 60.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(values[:2], np.asarray([0.71, 1.00]))
    assert np.isnan(values[2])
    assert np.isnan(values[3])


def test_topology_name_classifies_fd_and_ft_pairs_symmetrically() -> None:
    names = topology_name(
        np.asarray([1, 1, 0, 0, 2]),
        np.asarray([1, 0, 1, 0, 1]),
    )
    assert names.tolist() == ["FD/FD", "FD/FT", "FD/FT", "FT/FT", "other"]
