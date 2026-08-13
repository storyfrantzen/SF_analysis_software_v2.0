from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, normaltest


Array = np.ndarray

GROUPING = "proton-detector+ft-photon-count-v1"
_TOPOLOGY_RADIX = 4
_BIN_RADIX = 128

DEFAULT_VARIABLES = (
    "rec_m_gg",
    "rec_pT_miss",
    "rec_m2_epX",
    "rec_m_eggX",
    "rec_E_miss",
    "rec_m2_miss",
)


@dataclass(frozen=True)
class ExclusivityCuts:
    variables: tuple[str, ...]
    group_ids: Array
    lower: Array
    upper: Array
    global_mode: bool
    n_sigma: float
    minimum_events: int
    grouping: str


def derive_cuts(
    values: dict[str, Array],
    proton_detector: Array,
    ft_photons: Array,
    iq2: Array,
    ixb: Array,
    it: Array,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    topologies: tuple[int, ...] = (1, 2),
    photon_topologies: tuple[int, ...] = (0, 1, 2),
    n_sigma: float = 3.0,
    minimum_events: int = 50,
    global_mode: bool = False,
) -> ExclusivityCuts:
    """Derive sequential windows by proton and order-independent photon topology."""
    detector = np.asarray(proton_detector, dtype=np.int64)
    ft_photons = np.asarray(ft_photons, dtype=np.int64)
    iq2, ixb, it = [np.asarray(item, dtype=np.int64) for item in (iq2, ixb, it)]
    event_count = detector.size
    if not (ft_photons.size == iq2.size == ixb.size == it.size == event_count):
        raise ValueError("topology and bin-index arrays must have equal lengths")
    arrays = {name: np.asarray(values[name], dtype=float) for name in variables}
    if any(array.size != event_count for array in arrays.values()):
        raise ValueError("exclusivity arrays must match event count")

    in_range = (iq2 >= 0) & (ixb >= 0) & (it >= 0)
    in_range &= np.isin(detector, topologies)
    in_range &= np.isin(ft_photons, photon_topologies)
    groups = _group_ids(detector, ft_photons, iq2, ixb, it, global_mode)
    populated = np.unique(groups[in_range])
    lower = np.full((populated.size, len(variables)), np.nan)
    upper = np.full_like(lower, np.nan)

    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    left = np.searchsorted(sorted_groups, populated, side="left")
    right = np.searchsorted(sorted_groups, populated, side="right")
    retained_groups: list[int] = []
    retained_lower: list[Array] = []
    retained_upper: list[Array] = []
    for group_index, (start, stop) in enumerate(zip(left, right, strict=True)):
        rows = order[start:stop]
        rows = rows[in_range[rows]]
        if rows.size < minimum_events:
            continue
        running = np.ones(rows.size, dtype=bool)
        for variable_index, name in enumerate(variables):
            bounds = estimate_window(arrays[name][rows][running], n_sigma, minimum_events)
            if bounds is None:
                continue
            lo, hi = bounds
            lower[group_index, variable_index] = lo
            upper[group_index, variable_index] = hi
            running &= (arrays[name][rows] >= lo) & (arrays[name][rows] <= hi)
        retained_groups.append(int(populated[group_index]))
        retained_lower.append(lower[group_index])
        retained_upper.append(upper[group_index])
    return ExclusivityCuts(
        variables=variables,
        group_ids=np.asarray(retained_groups, dtype=np.int64),
        lower=np.asarray(retained_lower, dtype=float).reshape(-1, len(variables)),
        upper=np.asarray(retained_upper, dtype=float).reshape(-1, len(variables)),
        global_mode=global_mode,
        n_sigma=n_sigma,
        minimum_events=minimum_events,
        grouping=GROUPING,
    )


def apply_cuts(
    cuts: ExclusivityCuts,
    values: dict[str, Array],
    proton_detector: Array,
    ft_photons: Array,
    iq2: Array,
    ixb: Array,
    it: Array,
) -> Array:
    if cuts.grouping != GROUPING:
        raise ValueError(f"unsupported exclusivity grouping: {cuts.grouping}")
    detector = np.asarray(proton_detector, dtype=np.int64)
    ft_photons = np.asarray(ft_photons, dtype=np.int64)
    groups = _group_ids(detector, ft_photons, iq2, ixb, it, cuts.global_mode)
    mask = np.zeros(detector.size, dtype=bool)
    if cuts.group_ids.size == 0:
        return mask
    positions = np.searchsorted(cuts.group_ids, groups)
    bounded = positions < cuts.group_ids.size
    known = np.zeros(detector.size, dtype=bool)
    known[bounded] = cuts.group_ids[positions[bounded]] == groups[bounded]
    rows = np.flatnonzero(known)
    mask[rows] = True
    for variable_index, name in enumerate(cuts.variables):
        raw = np.asarray(values[name], dtype=float)
        lo = cuts.lower[positions[rows], variable_index]
        hi = cuts.upper[positions[rows], variable_index]
        active = np.isfinite(lo) & np.isfinite(hi)
        passed = np.ones(rows.size, dtype=bool)
        passed[active] = (raw[rows][active] >= lo[active]) & (raw[rows][active] <= hi[active])
        mask[rows] &= passed
    return mask


def estimate_window(values: Array, n_sigma: float, minimum_events: int) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < minimum_events:
        return None
    p1, p99 = np.percentile(values, [1.0, 99.0])
    trimmed = values[(values >= p1) & (values <= p99)]
    if trimmed.size < max(minimum_events, 8):
        return None
    _, probability = normaltest(trimmed)
    if probability > 0.05:
        mean, sigma = norm.fit(trimmed)
        return float(mean - n_sigma * sigma), float(mean + n_sigma * sigma)
    quantiles = 100.0 * np.array([norm.cdf(-n_sigma), norm.cdf(n_sigma)])
    lo, hi = np.percentile(trimmed, quantiles)
    return float(lo), float(hi)


def save_cuts(path: str, cuts: ExclusivityCuts) -> None:
    np.savez_compressed(
        path,
        variables=np.asarray(cuts.variables),
        group_ids=cuts.group_ids,
        lower=cuts.lower,
        upper=cuts.upper,
        global_mode=cuts.global_mode,
        n_sigma=cuts.n_sigma,
        minimum_events=cuts.minimum_events,
        grouping=cuts.grouping,
    )


def load_cuts(path: str) -> ExclusivityCuts:
    saved = np.load(path, allow_pickle=False)
    if "grouping" not in saved.files:
        raise ValueError(
            "exclusivity cut table predates FT-photon topology grouping; re-derive it"
        )
    grouping = str(np.asarray(saved["grouping"]).item())
    if grouping != GROUPING:
        raise ValueError(f"unsupported exclusivity grouping: {grouping}")
    return ExclusivityCuts(
        variables=tuple(str(item) for item in saved["variables"]),
        group_ids=saved["group_ids"],
        lower=saved["lower"],
        upper=saved["upper"],
        global_mode=bool(saved["global_mode"]),
        n_sigma=float(saved["n_sigma"]),
        minimum_events=int(saved["minimum_events"]),
        grouping=grouping,
    )


def _group_ids(
    detector: Array,
    ft_photons: Array,
    iq2: Array,
    ixb: Array,
    it: Array,
    global_mode: bool,
) -> Array:
    detector = np.asarray(detector, dtype=np.int64)
    ft_photons = np.asarray(ft_photons, dtype=np.int64)
    topology = detector * _TOPOLOGY_RADIX + ft_photons
    if global_mode:
        return topology
    iq2, ixb, it = [np.asarray(item, dtype=np.int64) for item in (iq2, ixb, it)]
    # Pairing is collision-free for the supported nonnegative bin indices.
    return (((topology * _BIN_RADIX) + iq2) * _BIN_RADIX + ixb) * _BIN_RADIX + it
