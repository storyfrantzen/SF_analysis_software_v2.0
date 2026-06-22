# Calibration parameters

Generated calibration coefficients live here, separately from executable analysis
configuration:

- `sampling_fraction/` contains electron sampling-fraction fit parameters.
- `proton_energy_loss/` contains proton kinematic-correction parameters.

Paths in processing and post-processing configs are resolved relative to the config
file that references them.

Physics-analysis settings such as beam energy, target properties, bin edges,
minimum acceptance, and branching ratios are not calibration coefficients.
They live under `analysis/configs/` instead.
