#!/usr/bin/env python3
"""Plot unweighted AAO LUND generator kinematics.

AAO Born writes e', p, gamma, gamma, while AAO radiative writes e', p,
pi0, gamma(rad).  This reader accepts either representation, ignores the
non-weight payload in the LUND header, and uses the explicit pi0 when present.

Examples:
  python3 scripts/aao_lund_kinematics.py /path/to/lund_dir out.png \
      --label "AAO Born" --ebeam 10.604
  python3 scripts/aao_lund_kinematics.py '/path/*.lund' out.png \
      --label "AAO radiative" --ebeam 10.604 --max-events 1000000
"""

import argparse
import glob
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MP = 0.93827208816
MPI0 = 0.1349768


def files_from(spec):
    path = Path(spec)
    if path.is_dir():
        files = sorted(path.rglob("*.lund"))
    elif path.is_file():
        files = [path]
    else:
        files = [Path(p) for p in sorted(glob.glob(spec, recursive=True))]
    if not files:
        raise SystemExit(f"no LUND files match: {spec}")
    return files


def mass2(v):
    return v[..., 0] ** 2 - np.sum(v[..., 1:] ** 2, axis=-1)


def photon_pair(photons):
    if len(photons) < 2:
        return None
    if len(photons) == 2:
        return photons
    return min(combinations(photons, 2),
               key=lambda pair: abs(np.sqrt(max(mass2(pair[0] + pair[1]), 0.0)) - MPI0))


def read_file(path, remaining=None):
    rows = []
    sources = {"pid111": 0, "gamma_gamma": 0}
    with open(path) as stream:
        while remaining is None or len(rows) < remaining:
            line = stream.readline()
            if not line:
                break
            fields = line.split()
            if line.startswith("#") or not fields:
                continue
            if len(fields) != 10:
                continue
            try:
                npart = int(fields[0])
            except ValueError:
                continue

            particles = []
            for _ in range(npart):
                particle = stream.readline().split()
                if len(particle) < 10:
                    continue
                try:
                    status, pid = int(particle[2]), int(particle[3])
                    p4 = np.array([float(particle[9]), float(particle[6]),
                                   float(particle[7]), float(particle[8])])
                except ValueError:
                    continue
                particles.append((status, pid, p4))

            final = [(pid, p4) for status, pid, p4 in particles if status == 1]
            electron = next((p4 for pid, p4 in final if pid == 11), None)
            proton = next((p4 for pid, p4 in final if pid == 2212), None)
            pi0 = next((p4 for pid, p4 in final if pid == 111), None)
            source = "pid111"
            if pi0 is None:
                pair = photon_pair([p4 for pid, p4 in final if pid == 22])
                if pair is not None:
                    pi0 = pair[0] + pair[1]
                    source = "gamma_gamma"
            if electron is None or proton is None or pi0 is None:
                continue
            rows.append(np.concatenate([electron, proton, pi0]))
            sources[source] += 1
    return np.asarray(rows, dtype=np.float64), sources


def read_sample(spec, max_events=None):
    chunks = []
    total = 0
    sources = {"pid111": 0, "gamma_gamma": 0}
    files = files_from(spec)
    for index, path in enumerate(files, 1):
        remaining = None if max_events is None else max_events - total
        if remaining is not None and remaining <= 0:
            break
        chunk, file_sources = read_file(path, remaining)
        if len(chunk):
            chunks.append(chunk)
            total += len(chunk)
            for source, count in file_sources.items():
                sources[source] += count
        if index % 25 == 0:
            print(f"read {index}/{len(files)} files, {total} events", flush=True)
    if not chunks:
        raise SystemExit("no complete e' p gamma gamma events were found")
    return np.concatenate(chunks), len(files), sources


def momentum(v):
    return np.linalg.norm(v[:, 1:], axis=1)


def theta(v):
    p = momentum(v)
    return np.degrees(np.arccos(np.clip(v[:, 3] / np.maximum(p, 1e-30), -1, 1)))


def hist(ax, values, label, bins=60, limits=None, log=False):
    values = np.asarray(values)
    good = np.isfinite(values)
    if limits is None:
        limits = tuple(np.percentile(values[good], [0.2, 99.8]))
        if limits[1] <= limits[0]:
            limits = (limits[0] - 0.5, limits[0] + 0.5)
    ax.hist(values[good], bins=bins, range=limits, histtype="stepfilled",
            facecolor="#9ecae1", edgecolor="#0072B2", linewidth=1.0)
    ax.set_xlabel(label)
    if log:
        ax.set_yscale("log")


def hist2d(ax, x, y, xlabel, ylabel, ylim=None):
    good = np.isfinite(x) & np.isfinite(y)
    yrange = list(ylim) if ylim else [float(np.min(y[good])), float(np.max(y[good]))]
    ax.hist2d(x[good], y[good], bins=60,
              range=[[float(np.min(x[good])), float(np.max(x[good]))], yrange],
              cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot(data, output, label, ebeam, sources):
    electron, proton, pi0 = (data[:, 4 * i:4 * (i + 1)] for i in range(3))

    beam = np.array([ebeam, 0.0, 0.0, ebeam])
    target = np.array([MP, 0.0, 0.0, 0.0])
    q = beam - electron
    q2 = -mass2(q)
    nu = q[:, 0]
    w = np.sqrt(np.maximum(mass2(q + target), 0.0))
    xb = q2 / np.maximum(2.0 * MP * nu, 1e-30)
    mt = -mass2(target - proton)
    missing = beam + target - electron - proton
    mm2 = mass2(missing)
    v = mm2 - MPI0 ** 2
    mpi0 = np.sqrt(np.maximum(mass2(pi0), 0.0))

    fig, axes = plt.subplots(5, 3, figsize=(13, 16))
    parts = [(electron, r"$e'$", (0.0, 30.0)),
             (proton, r"$p$", None),
             (pi0, r"$\pi^0$", None)]
    for row, (four, symbol, theta_limits) in enumerate(parts):
        p, th = momentum(four), theta(four)
        hist(axes[row, 0], p, rf"{symbol} $|p|$ [GeV]")
        hist(axes[row, 1], th, rf"{symbol} $\theta$ [deg]", limits=theta_limits)
        hist2d(axes[row, 2], p, th, rf"{symbol} $|p|$ [GeV]",
               rf"{symbol} $\theta$ [deg]", ylim=theta_limits)

    hist(axes[3, 0], q2, r"$Q^2$ [GeV$^2$]")
    hist(axes[3, 1], xb, r"$x_B$")
    hist(axes[3, 2], mt, r"$-t$ [GeV$^2$]")
    hist(axes[4, 0], w, r"$W$ [GeV]")
    hist(axes[4, 1], mm2, r"$M_X^2(e'p)$ [GeV$^2$]  (missing $\pi^0$)", log=True)
    axes[4, 1].axvline(MPI0 ** 2, color="grey", linestyle=":", linewidth=1)
    hist(axes[4, 2], v, r"$v=M_X^2(e'p)-m_{\pi^0}^2$ [GeV$^2$]", log=True)
    axes[4, 2].axvline(0.20, color="#D55E00", linestyle="--", linewidth=1.1,
                       label=r"$v=0.20$ reference")
    axes[4, 2].legend(fontsize=8)

    fig.suptitle(f"{label} — AAO LUND, {len(data):,} unweighted events, "
                 f"E={ebeam:.3f} GeV", fontsize=13)
    source_text = (f"pi0 source: PID 111={sources['pid111']:,}, "
                   f"gamma gamma={sources['gamma_gamma']:,}")
    fig.text(0.5, 0.005, f"{source_text}; median M(pi0) = "
             f"{np.median(mpi0):.5f} GeV", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.015, 1, 0.98])
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"wrote {output}")
    print(f"events={len(data)}  {source_text}  "
          f"median Mpi0={np.median(mpi0):.6f} GeV  "
          f"v<0.17={100.0 * np.mean(v < 0.17):.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="LUND file, directory, or quoted glob")
    parser.add_argument("output", help="output PNG")
    parser.add_argument("--label", default="AAO")
    parser.add_argument("--ebeam", type=float, default=10.604)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()

    data, nfiles, sources = read_sample(args.input, args.max_events)
    print(f"accepted {len(data)} events from up to {nfiles} input files; "
          "AAO header column 10 ignored")
    plot(data, args.output, args.label, args.ebeam, sources)


if __name__ == "__main__":
    main()
