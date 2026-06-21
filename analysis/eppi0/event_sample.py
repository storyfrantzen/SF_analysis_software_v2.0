from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray
ELECTRON_MASS = 0.00051099895
PROTON_MASS = 0.9382720813
PI0_MASS = 0.1349768


@dataclass(frozen=True)
class GeneratedSample:
    run: Array
    event: Array
    q2: Array
    xb: Array
    minus_t: Array
    trento_phi: Array
    radiative: Array


def build_generated_sample(
    run: Array,
    event: Array,
    pid: Array,
    momentum: Array,
    theta: Array,
    phi: Array,
    beam_energy: float,
) -> GeneratedSample:
    """Collapse per-particle GEN rows into one generated EPPI0 row per event."""
    run = np.asarray(run, dtype=np.int64)
    event = np.asarray(event, dtype=np.int64)
    pid = np.asarray(pid, dtype=np.int64)
    momentum = np.asarray(momentum, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if not (run.shape == event.shape == pid.shape == momentum.shape == theta.shape == phi.shape):
        raise ValueError("all generated-particle arrays must have equal shapes")

    valid = (pid != -999) & np.isfinite(momentum) & np.isfinite(theta) & np.isfinite(phi)
    run, event, pid = run[valid], event[valid], pid[valid]
    momentum, theta, phi = momentum[valid], theta[valid], phi[valid]
    keys = _keys(run, event)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    number_of_events = unique_keys.size

    electron = _first_particle(inverse, pid, momentum, theta, phi, 11, number_of_events)
    proton = _first_particle(inverse, pid, momentum, theta, phi, 2212, number_of_events)
    pi0 = _first_particle(inverse, pid, momentum, theta, phi, 111, number_of_events)
    photon_one, photon_two = _first_two_particles(
        inverse, pid, momentum, theta, phi, 22, number_of_events
    )

    radiative = pi0["present"]
    pi0_vector = _four_vector(pi0, PI0_MASS)
    nonradiative_pi0 = _four_vector(photon_one, 0.0) + _four_vector(photon_two, 0.0)
    pi0_vector = np.where(radiative[:, None], pi0_vector, nonradiative_pi0)
    topology = (
        electron["present"]
        & proton["present"]
        & (radiative | (photon_one["present"] & photon_two["present"]))
    )

    electron_vector = _four_vector(electron, ELECTRON_MASS)
    proton_vector = _four_vector(proton, PROTON_MASS)
    q2, xb = _dis(electron_vector, beam_energy)
    minus_t = _minus_t(proton_vector)
    trento = _trento_phi(electron_vector, proton_vector, beam_energy)

    return GeneratedSample(
        run=unique_keys["run"][topology],
        event=unique_keys["event"][topology],
        q2=q2[topology],
        xb=xb[topology],
        minus_t=minus_t[topology],
        trento_phi=trento[topology],
        radiative=radiative[topology],
    )


def join_reconstructed(
    generated: GeneratedSample,
    rec_run: Array,
    rec_event: Array,
    rec_columns: dict[str, Array],
) -> dict[str, Array]:
    """Left-join selected REC candidates onto all generated events."""
    rec_run = np.asarray(rec_run, dtype=np.int64)
    rec_event = np.asarray(rec_event, dtype=np.int64)
    rec_keys = _keys(rec_run, rec_event)
    if np.unique(rec_keys).size != rec_keys.size:
        raise ValueError("selected REC sample must contain at most one candidate per event")
    order = np.argsort(rec_keys, order=("run", "event"))
    sorted_keys = rec_keys[order]
    gen_keys = _keys(generated.run, generated.event)
    positions = np.searchsorted(sorted_keys, gen_keys)
    bounded = positions < sorted_keys.size
    matched = np.zeros(gen_keys.size, dtype=bool)
    matched[bounded] = sorted_keys[positions[bounded]] == gen_keys[bounded]

    output: dict[str, Array] = {
        "run": generated.run,
        "event": generated.event,
        "gen_Q2": generated.q2,
        "gen_xB": generated.xb,
        "gen_minus_t": generated.minus_t,
        "gen_trento_phi": generated.trento_phi,
        "gen_radiative": generated.radiative,
        "rec_selected": matched,
    }
    for name, raw in rec_columns.items():
        raw = np.asarray(raw)
        if raw.shape != rec_run.shape:
            raise ValueError(f"REC column {name} does not match event keys")
        dtype = float if raw.dtype.kind not in "iu" else raw.dtype
        fill = np.nan if np.dtype(dtype).kind == "f" else -999
        joined = np.full(gen_keys.size, fill, dtype=dtype)
        joined[matched] = raw[order[positions[matched]]]
        output[name] = joined
    return output


def _keys(run: Array, event: Array) -> Array:
    keys = np.empty(np.asarray(run).size, dtype=[("run", "<i8"), ("event", "<i8")])
    keys["run"] = run
    keys["event"] = event
    return keys


def _empty_particles(number_of_events: int) -> dict[str, Array]:
    return {
        "present": np.zeros(number_of_events, dtype=bool),
        "p": np.full(number_of_events, np.nan),
        "theta": np.full(number_of_events, np.nan),
        "phi": np.full(number_of_events, np.nan),
    }


def _first_particle(
    inverse: Array,
    pid: Array,
    momentum: Array,
    theta: Array,
    phi: Array,
    wanted_pid: int,
    number_of_events: int,
) -> dict[str, Array]:
    result = _empty_particles(number_of_events)
    rows = np.flatnonzero(pid == wanted_pid)
    if rows.size == 0:
        return result
    _, first_positions = np.unique(inverse[rows], return_index=True)
    rows = rows[first_positions]
    groups = inverse[rows]
    result["present"][groups] = True
    result["p"][groups] = momentum[rows]
    result["theta"][groups] = theta[rows]
    result["phi"][groups] = phi[rows]
    return result


def _first_two_particles(
    inverse: Array,
    pid: Array,
    momentum: Array,
    theta: Array,
    phi: Array,
    wanted_pid: int,
    number_of_events: int,
) -> tuple[dict[str, Array], dict[str, Array]]:
    first = _empty_particles(number_of_events)
    second = _empty_particles(number_of_events)
    rows = np.flatnonzero(pid == wanted_pid)
    if rows.size == 0:
        return first, second
    rows = rows[np.argsort(inverse[rows], kind="stable")]
    groups = inverse[rows]
    starts = np.r_[True, groups[1:] != groups[:-1]]
    rank = np.arange(rows.size) - np.maximum.accumulate(np.where(starts, np.arange(rows.size), 0))
    for target, wanted_rank in ((first, 0), (second, 1)):
        selected_rows = rows[rank == wanted_rank]
        selected_groups = inverse[selected_rows]
        target["present"][selected_groups] = True
        target["p"][selected_groups] = momentum[selected_rows]
        target["theta"][selected_groups] = theta[selected_rows]
        target["phi"][selected_groups] = phi[selected_rows]
    return first, second


def _four_vector(particle: dict[str, Array], mass: float) -> Array:
    p = particle["p"]
    sin_theta = np.sin(particle["theta"])
    px = p * sin_theta * np.cos(particle["phi"])
    py = p * sin_theta * np.sin(particle["phi"])
    pz = p * np.cos(particle["theta"])
    energy = np.sqrt(p * p + mass * mass)
    return np.column_stack((px, py, pz, energy))


def _dis(electron: Array, beam_energy: float) -> tuple[Array, Array]:
    q_energy = beam_energy - electron[:, 3]
    qx, qy, qz = -electron[:, 0], -electron[:, 1], beam_energy - electron[:, 2]
    q2 = qx * qx + qy * qy + qz * qz - q_energy * q_energy
    xb = np.divide(q2, 2.0 * PROTON_MASS * q_energy, out=np.full_like(q2, np.nan), where=q_energy != 0)
    return q2, xb


def _minus_t(proton: Array) -> Array:
    delta_energy = PROTON_MASS - proton[:, 3]
    t_invariant = delta_energy**2 - np.sum(proton[:, :3] ** 2, axis=1)
    return -t_invariant


def _trento_phi(electron: Array, proton: Array, beam_energy: float) -> Array:
    beam = np.zeros_like(electron[:, :3])
    beam[:, 2] = beam_energy
    q = beam - electron[:, :3]
    lepton_normal = np.cross(beam, electron[:, :3])
    hadron_normal = np.cross(proton[:, :3], q)
    lepton_normal = _unit(lepton_normal)
    hadron_normal = _unit(hadron_normal)
    q_hat = _unit(q)
    cosine = np.sum(lepton_normal * hadron_normal, axis=1)
    sine = np.sum(q_hat * np.cross(lepton_normal, hadron_normal), axis=1)
    return np.arctan2(sine, cosine)


def _unit(vectors: Array) -> Array:
    magnitude = np.linalg.norm(vectors, axis=1)
    return np.divide(vectors, magnitude[:, None], out=np.full_like(vectors, np.nan), where=magnitude[:, None] > 0)
