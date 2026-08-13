from __future__ import annotations

import numpy as np


Array = np.ndarray
INVALID_TOPOLOGY = -999
FT_DETECTOR = 0
FD_DETECTOR = 1


def ft_photon_count(gamma1_detector: Array, gamma2_detector: Array) -> Array:
    """Return the order-independent number of selected photons in the FT."""
    gamma1 = np.asarray(gamma1_detector, dtype=np.int64)
    gamma2 = np.asarray(gamma2_detector, dtype=np.int64)
    if gamma1.shape != gamma2.shape:
        raise ValueError("selected-photon detector arrays must have equal shapes")

    output = np.full(gamma1.shape, INVALID_TOPOLOGY, dtype=np.int64)
    supported = np.isin(gamma1, (FT_DETECTOR, FD_DETECTOR))
    supported &= np.isin(gamma2, (FT_DETECTOR, FD_DETECTOR))
    output[supported] = (
        (gamma1[supported] == FT_DETECTOR).astype(np.int64)
        + (gamma2[supported] == FT_DETECTOR).astype(np.int64)
    )
    return output
