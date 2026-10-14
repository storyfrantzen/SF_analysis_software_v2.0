# Calibration parameters

Generated calibration coefficients live here, separately from executable analysis
configuration:

- `sampling_fraction/` contains electron sampling-fraction fit parameters.
- `proton_energy_loss/` contains proton kinematic-correction parameters.
- `momentum/` is the target directory for elastic, data-derived electron and
  proton momentum-scale parameters. Keep run groups and torus polarities in
  separate files.

Paths in processing and post-processing configs are resolved relative to the config
file that references them.

Physics-analysis settings such as beam energy, target properties, bin edges,
minimum acceptance, and branching ratios are not calibration coefficients.
They live under `configs/analysis/` instead.
