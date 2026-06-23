# Configuration layout

Production configurations are grouped first by run group and then by beam
energy:

```text
configs/processing/<run-group>/<energy>/
configs/post/<run-group>/<energy>/
analysis/configs/<run-group>/<energy>.json
```

Calibration-only workflows live one level deeper in a `calibration/`
subdirectory. No uncategorized executable configs are kept at the roots of
`processing/`, `post/`, or `analysis/configs/`.

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
another.

The active RGK 6.535 GeV files are:

- `processing/rgk/6.535/eppi0_mc_acceptance.json`;
- `processing/rgk/6.535/calibration/sidis_electrons_data.json`;
- `processing/rgk/6.535/calibration/sidis_electrons_mc.json`;
- `processing/rgk/6.535/calibration/proton_energy_loss_mc.json`;
- `post/rgk/6.535/calibration/electron_sf_candidates.json`;
- `post/rgk/6.535/calibration/electron_sf_candidates_mc.json`;
- `post/rgk/6.535/calibration/electron_sf_selected.json`;
- `../analysis/configs/rgk/6.535.json`.

RGA calibration inputs are under `processing/rga/10.604/calibration/` and
`post/rga/10.604/calibration/`.

The RGA sampling-fraction calibration has distinct data and GEMC processing
and candidate configs. This prevents the small validation outputs from
overwriting one another or the established full-statistics parameter file.

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

## RGA sampling-fraction rederivation test

The converter accepts either a directory or one explicit `.hipo` file. Run the
following from the repository root after rebuilding.

First process five lexicographically sorted Fall 2018 torus +1 data files:

```bash
./build/hipo2root \
  configs/processing/rga/10.604/calibration/sidis_electrons_data.json \
  /cache/clas12/rg-a/production/recon/fall2018/torus+1/pass2/train/nSidis/ \
  5 100000

./build/apply_cuts \
  configs/post/rga/10.604/calibration/electron_sf_candidates.json \
  10.604_rga_sidis_electrons.root

python3 scripts/derive_sampling_fraction.py \
  10.604_rga_electron_sf_candidates.root \
  --output parameters/sampling_fraction/SF_sigma_cut_params_10.604RGA_FA18_t+1_nSIDIS_5files.json \
  --plot-dir calibration_plots/sampling_fraction/rga_10.604_data_5files \
  --dataset-tag 10.604RGA_FA18_t+1_nSIDIS_5files \
  --beam-energy 10.604 --run-group RGA --skim nSIDIS --torus 1
```

Then process the exact GEMC file and derive one sector-independent resolution
fit copied to all six sectors:

```bash
./build/hipo2root \
  configs/processing/rga/10.604/calibration/sidis_electrons_mc.json \
  /cache/clas12/rg-a/production/montecarlo/clasdis_pass2/fa18_out/clasdis_rga_fa18_out_50nA_10604MeV-0000.hipo \
  1 100000

./build/apply_cuts \
  configs/post/rga/10.604/calibration/electron_sf_candidates_mc.json \
  10.604_rga_gemc_sidis_electrons.root

python3 scripts/derive_sampling_fraction.py \
  10.604_rga_gemc_electron_sf_candidates.root --gemc \
  --output parameters/sampling_fraction/SF_sigma_cut_params_10.604RGA_FA18_t+1_clasdisGEMC.json \
  --plot-dir calibration_plots/sampling_fraction/rga_10.604_gemc \
  --dataset-tag 10.604RGA_FA18_t+1_clasdisGEMC \
  --beam-energy 10.604 --run-group RGA --skim clasdis_pass2 --torus 1
```

The two output JSON files and plot directories are intentionally separate.
Compare the fitted mean and sigma curves and the retained fractions before
choosing which parameters belong in a physics-selection configuration.

## RGK 6.535 GeV GEMC calibration inputs

The RGK 6.535 GeV data-side sampling-fraction parameters already have a
calibration workflow. If a matching nonradiative `clasdis_pass2` GEMC sample is
produced locally, use the GEMC calibration configs to derive diagnostic GEMC
sampling-fraction parameters and proton energy-loss corrections.

For GEMC sampling fraction:

```bash
./build/hipo2root \
  configs/processing/rgk/6.535/calibration/sidis_electrons_mc.json \
  /path/to/reconstructed/6.535/clasdis_pass2.hipo \
  1 100000

./build/apply_cuts \
  configs/post/rgk/6.535/calibration/electron_sf_candidates_mc.json \
  6.535_rgk_gemc_sidis_electrons.root

python3 scripts/derive_sampling_fraction.py \
  6.535_rgk_gemc_electron_sf_candidates.root --gemc \
  --output parameters/sampling_fraction/SF_sigma_cut_params_6.535RGK_clasdisGEMC.json \
  --plot-dir calibration_plots/sampling_fraction/rgk_6.535_gemc \
  --dataset-tag 6.535RGK_clasdisGEMC \
  --beam-energy 6.535 --run-group RGK --skim clasdis_pass2 --torus 1
```

For proton energy loss:

```bash
./build/hipo2root \
  configs/processing/rgk/6.535/calibration/proton_energy_loss_mc.json \
  /path/to/reconstructed/6.535/clasdis_pass2.hipo \
  1 100000

python3 scripts/derive_proton_energy_loss.py \
  6.535_rgk_proton_energy_loss_mc.root \
  --output parameters/proton_energy_loss/6.535RGK_clasdisP2.json \
  --plot-dir calibration_plots/proton_energy_loss/rgk_6.535_clasdisP2 \
  --dataset-tag 6.535RGK_clasdisP2 \
  --beam-energy 6.535
```
