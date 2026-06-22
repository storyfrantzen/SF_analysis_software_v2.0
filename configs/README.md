# Configuration layout

Production configurations are grouped first by run group and then by beam
energy:

```text
configs/processing/<run-group>/<energy>/
configs/post/<run-group>/<energy>/
analysis/configs/<run-group>/<energy>.json
```

Processing configurations control HIPO input reduction, QADB, reconstructed
skims, MC matching, generated-event storage, and kinematic corrections. Post
configurations define candidate construction and tuneable detector/physics
cuts. Analysis configurations contain binning, target, and normalization
settings.

The complete RGA 10.604 GeV EPPI0 pair is:

- `processing/rga/10.604/eppi0_data.json`;
- `processing/rga/10.604/eppi0_mc_nonradiative.json`;
- `post/rga/10.604/eppi0_data.json`;
- `post/rga/10.604/eppi0_mc_nonradiative.json`;
- `../analysis/configs/rga/10.604.json`.

Both post-processing files deliberately carry the same particle and detector
selection. They differ only in output name so data and MC cannot overwrite one
another. The flat files directly under `processing/` and `post/` remain as
generic examples and backward-compatible calibration configurations.

Calibration coefficients live under `parameters/`, not in this directory.
Relative paths in a configuration are resolved from the directory containing
that configuration.

## RGA 10.604 GeV EPPI0 smoke test

From the repository root:

```bash
./build/hipo2root configs/processing/rga/10.604/eppi0_data.json \
  /path/to/rga/eppi0/data

./build/hipo2root configs/processing/rga/10.604/eppi0_mc_nonradiative.json \
  /path/to/rga/nonradiative/gemc

./build/apply_cuts configs/post/rga/10.604/eppi0_data.json \
  10.604_rga_eppi0_data.root

./build/apply_cuts configs/post/rga/10.604/eppi0_mc_nonradiative.json \
  10.604_rga_eppi0_mc_nonradiative.root
```

The processing stage corrects reconstructed proton momentum and angles before
they are stored. Raw proton values and correction deltas remain in the ROOT
branches. The post-processing stage accepts proton detector IDs 1 (FD) and 2
(CD), applies PID-appropriate RGA fiducials, and applies the three electron
sampling-fraction components to FD electrons. FT electrons bypass the FD-only
sampling-fraction requirements.
