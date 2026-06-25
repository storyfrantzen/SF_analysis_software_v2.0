# Configuration layout

Production configurations are grouped first by run group and then by beam
energy:

```text
configs/processing/<run-group>/<energy>/
configs/post/<run-group>/<energy>/
configs/analysis/<run-group>/<energy>.json
```

Calibration-only workflows live one level deeper in a `calibration/`
subdirectory. No uncategorized executable configs are kept at the roots of
`processing/`, `post/`, or `analysis/`.

Processing configurations control HIPO input reduction, QADB, reconstructed
skims, MC matching, generated-event storage, row-level PID storage filters, and
kinematic corrections. Post configurations define candidate construction and
tuneable detector/physics cuts. Analysis configurations contain binning,
target, and normalization settings.

The complete RGA 10.604 GeV EPPI0 pair is:

- `processing/rga/10.604/eppi0_data.json`;
- `processing/rga/10.604/eppi0_mc_nonradiative.json`;
- `post/rga/10.604/eppi0_data.json`;
- `post/rga/10.604/eppi0_mc_nonradiative.json`;
- `analysis/rga/10.604.json`.

Both post-processing files extend `post/rga/10.604/eppi0_base.json` so the
shared particle, detector, fiducial, and loose-exclusivity selections live in
one place. The data and MC child configs override only the output name and the
sampling-fraction parameter file.

The active RGK 6.535 GeV files are:

- `processing/rgk/6.535/aao_rad_q2_0.7_ep_1.00.json`;
- `processing/rgk/6.535/aao_rad_q2_0.9_ep_1.15.json`;
- `post/rgk/6.535/eppi0_base.json`;
- `post/rgk/6.535/aao_rad_eppi0_loose.json`;
- `processing/rgk/6.535/eppi0_mc_acceptance.json`;
- `processing/rgk/6.535/calibration/sidis_electrons_data.json`;
- `processing/rgk/6.535/calibration/sidis_electrons_mc.json`;
- `processing/rgk/6.535/calibration/proton_energy_loss_mc.json`;
- `post/rgk/6.535/calibration/electron_sf_candidates.json`;
- `post/rgk/6.535/calibration/electron_sf_candidates_mc.json`;
- `post/rgk/6.535/calibration/electron_sf_selected.json`;
- `analysis/rgk/6.535.json`.

RGA calibration inputs are under `processing/rga/10.604/calibration/` and
`post/rga/10.604/calibration/`.

The RGA sampling-fraction calibration has distinct data and GEMC processing
and candidate configs. This prevents the small validation outputs from
overwriting one another or the established full-statistics parameter file.

Calibration coefficients live under `parameters/`, not in this directory.
Relative paths in a configuration are resolved from the directory containing
that configuration.

## Post-Config Inheritance

Post-processing configs may define an `extends` key that points to another JSON
file, resolved relative to the child config. The loader expands the parent
before parsing the normal post-processing fields. Object values merge
recursively, while arrays and scalar values replace the parent value.

Use this for data/MC pairs that intentionally share a topology and detector
selection. For example, `configs/post/rga/10.604/eppi0_data.json` extends the
RGA EPPI0 base config and overrides only `outputFile` plus
`samplingFraction.sigma.paramsFile`.

## RGA 10.604 GeV EPPI0 smoke test

From the repository root:

```bash
./build/hipo2root configs/processing/rga/10.604/eppi0_data.json \
  /path/to/rga/eppi0/data

./build/hipo2root configs/processing/rga/10.604/eppi0_mc_nonradiative.json \
  /path/to/rga/nonradiative/gemc

./build/post_process configs/post/rga/10.604/eppi0_data.json \
  10.604_rga_eppi0_data.root

./build/post_process configs/post/rga/10.604/eppi0_mc_nonradiative.json \
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

./build/post_process \
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

./build/post_process \
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

## RGK 6.535 GeV aaoRad reconstructed-distribution comparison

The RGK EPPI0 base config mirrors the RGA EPPI0 topology with RGK-specific
detector acceptance: no detector 0 FT candidates, RGK ECAL fiducials, RGK ECAL
edge cuts, and the same FD/CD proton treatment.

The aaoRad comparison configs are intended for quick shape checks across the
new `/volatile/clas12/osg/storyf/11221`, `11222`, `11223`, `11224`, `11225`,
and `11238` productions. The two processing configs match the generated
phase-space families in the filenames: `Q2 >= 0.7, electron p >= 1.00` for
`11221` through `11223`, and `Q2 >= 0.9, electron p >= 1.15` for `11224`,
`11225`, and `11238`. The post config extends the RGK EPPI0 base but keeps a
loose photon selection with a very low reconstructed photon momentum threshold,
so the comparison is sensitive to the generated `EG` threshold scan.

Example smoke-test commands on ifarm:

```bash
for run in 11221 11222 11223; do
  ./build/hipo2root configs/processing/rgk/6.535/aao_rad_q2_0.7_ep_1.00.json \
    /volatile/clas12/osg/storyf/${run} 5 100000
  mv 6.535_rgk_aao_rad_q2_0.7_ep_1.00.root aao_rad_${run}.root
  ./build/post_process configs/post/rgk/6.535/aao_rad_eppi0_loose.json \
    aao_rad_${run}.root 100000
  mv 6.535_rgk_aao_rad_eppi0_loose_selected.root aao_rad_${run}_selected.root
done

python3 analysis/compare_root_distributions.py --density \
  --sample EG0.005=aao_rad_11221_selected.root \
  --sample EG0.010=aao_rad_11222_selected.root \
  --sample EG0.015=aao_rad_11223_selected.root \
  --output-dir results/aao_rad/q2_0.7_ep_1.00
```

Repeat with the tighter generated phase-space family:

```bash
for run in 11224 11225 11238; do
  ./build/hipo2root configs/processing/rgk/6.535/aao_rad_q2_0.9_ep_1.15.json \
    /volatile/clas12/osg/storyf/${run} 5 100000
  mv 6.535_rgk_aao_rad_q2_0.9_ep_1.15.root aao_rad_${run}.root
  ./build/post_process configs/post/rgk/6.535/aao_rad_eppi0_loose.json \
    aao_rad_${run}.root 100000
  mv 6.535_rgk_aao_rad_eppi0_loose_selected.root aao_rad_${run}_selected.root
done

python3 analysis/compare_root_distributions.py --density \
  --sample EG0.005=aao_rad_11224_selected.root \
  --sample EG0.010=aao_rad_11225_selected.root \
  --sample EG0.015=aao_rad_11238_selected.root \
  --output-dir results/aao_rad/q2_0.9_ep_1.15
```

To compare converter-level particle distributions before EPPI0 candidate selection, pass
converter ROOT files to `analysis/compare_root_distributions.py`, choose
particle branches such as `p theta phi`, filter by PID, and load the ROOT
dictionary if object branches are not already discoverable:

```bash
python3 analysis/compare_root_distributions.py --density \
  --tree Events --columns p theta phi \
  --dictionary build/libROOTBranchesDict.dylib \
  --where 'rec.pid == 22' \
  --sample EG0.005=aao_rad_11221.root \
  --sample EG0.010=aao_rad_11222.root \
  --sample EG0.015=aao_rad_11223.root \
  --output-dir results/aao_rad/photons_q2_0.7_ep_1.00
```

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

./build/post_process \
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

The proton energy-loss processing configs use `outputPids: [2212]`. The
`finalState` requirement is still event-level, so events are kept when they
contain at least one reconstructed proton, but the output ROOT tree stores only
proton rows. This keeps the matched REC/GEN proton calibration sample broad
without writing every other reconstructed particle in those events.
