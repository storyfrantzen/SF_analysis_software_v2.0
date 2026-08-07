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
- `processing/rga/10.604/eppi0_mc_acceptance.json`;
- `post/rga/10.604/eppi0_data.json`;
- `post/rga/10.604/eppi0_mc_acceptance.json`;
- `analysis/rga/10.604.json`.

Both post-processing files extend `post/rga/10.604/eppi0_base.json` so the
shared particle, detector, fiducial, and loose-exclusivity selections live in
one place. The data and MC child configs override only the output name and the
sampling-fraction parameter file.

The active RGK 6.535 GeV files are:

- `processing/rgk/6.535/aao_rad_q2_0.7_ep_1.00.json`;
- `processing/rgk/6.535/aao_rad_q2_0.9_ep_1.15.json`;
- `processing/rgk/6.535/eppi0_data.json`;
- `processing/rgk/6.535/eppi0_data_full_dst.json`;
- `post/rgk/6.535/eppi0_data.json`;
- `post/rgk/6.535/eppi0_base.json`;
- `post/rgk/6.535/aao_rad_eppi0_loose.json`;
- `processing/rgk/6.535/eppi0_mc_acceptance.json`;
- `post/rgk/6.535/eppi0_mc_acceptance.json`;
- `processing/rgk/6.535/calibration/sidis_electrons_data.json`;
- `processing/rgk/6.535/calibration/sidis_electrons_mc.json`;
- `processing/rgk/6.535/calibration/proton_energy_loss_mc.json`;
- `post/rgk/6.535/calibration/electron_sf_candidates.json`;
- `post/rgk/6.535/calibration/electron_sf_candidates_mc.json`;
- `post/rgk/6.535/calibration/proton_energy_loss_fiducial.json`;
- `post/rgk/6.535/calibration/electron_sf_selected.json`;
- `efficiency/rgk/6.535/run_currents.json`;
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

Primitive particle cuts and pair-mass composites accept an optional `mode`:

- `"mode": "require"` is the default and rejects a particle combination when
  the cut fails;
- `"mode": "tag"` keeps the combination and records the result in
  `evaluatedCuts` and `failedCuts`.

Use required cuts for stable topology and candidate identity, and tagged cuts
for selections whose effects should be studied after candidate construction.
When `saveFailedCandidates` is true, loose-exclusivity failures are retained as
well. The visualizer expands these two CSV branches into filterable quantities
named `passCut_<cut_name>`.

`post/rga/10.604/eppi0_cut_diagnostics.json` is a ready-to-run RGA data
configuration. It requires the trigger electron and loose particle-quality
preselection, while tagging electron/proton/photon fiducials, electron
sampling-fraction cuts, the three CVT phi vetoes, and the pi0 mass window. It
also retains failed loose-exclusivity candidates.

## Candidate selection

Candidate construction exhaustively visits every particle combination that
passes the configured role cuts. The optional `candidateSelection` block
chooses which complete combination is retained. Its backward-compatible
default is `compositeDistance`, which minimizes the unscaled pair-mass
distance.

RGK EPPI0 uses an explicit two-stage prescription:

```json
"candidateSelection": {
    "method": "pi0MassThenMissingPt"
}
```

The algorithm first chooses the photon pair whose invariant mass is closest to
the configured pi0 mass. For that photon pair, it chooses the proton producing
the smallest missing transverse momentum. Exact remaining ties are resolved
lexicographically by the selected particles' input indices. This retains the
legacy photon-pair preference, adds a physically motivated proton choice, and
does not require guessed detector resolutions.

Every complete combination that passes required particle and composite
preselection participates. Loose exclusivity is applied once to the globally
selected winner; the event is not allowed to choose a different combination
solely because that alternative passes the downstream exclusivity gate. Data,
Born MC, radiative MC, and acceptance MC must use the same candidate-selection
definition when they enter the same correction workflow.

## RGA 10.604 GeV EPPI0 smoke test

From the repository root:

```bash
./build/hipo2root configs/processing/rga/10.604/eppi0_data.json \
  /path/to/rga/eppi0/data

./build/hipo2root configs/processing/rga/10.604/eppi0_mc_acceptance.json \
  /path/to/rga/gemc

./build/post_process configs/post/rga/10.604/eppi0_data.json \
  10.604_rga_eppi0_data.root

./build/post_process configs/post/rga/10.604/eppi0_mc_acceptance.json \
  10.604_rga_eppi0_mc_acceptance.root

./build/post_process configs/post/rga/10.604/eppi0_cut_diagnostics.json \
  10.604_rga_eppi0_data.root

python3 -m visualizer 10.604_rga_eppi0_cut_diagnostics.root \
  --tree sEvents --output 10.604_rga_eppi0_cut_diagnostics.html
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

The aaoRad comparison configs are intended for quick shape checks across any
compatible production. Choose the processing config whose phase-space family
matches the generator inputs: `Q2 >= 0.7, electron p >= 1.00` or
`Q2 >= 0.9, electron p >= 1.15`. The post config extends the RGK EPPI0 base but
keeps a loose photon selection with a very low reconstructed photon momentum
threshold, so comparisons remain sensitive to generated photon-threshold
changes without depending on a particular production identifier.

For RGK 6.535 GeV data processing, use the data config pair. The processing
config enables QADB, applies the `6.535RGK_INCLUSIVE_GEMC_100M` proton
energy-loss corrections, and records accumulated beam charge for cross-section
normalization. New converter outputs preserve both the file-level total and a
run-indexed `RunCharge` tree, so a mixed-run input can be separated after
conversion. The post config extends the nominal RGK EPPI0 base selection and
loads the data-side `6.535RGKSKIM1` sampling-fraction parameters:

```bash
./build/hipo2root configs/processing/rgk/6.535/eppi0_data.json \
  /path/to/rgk/6.535/data 0 1000000

./build/post_process configs/post/rgk/6.535/eppi0_data.json \
  6.535_rgk_eppi0_data.root 1000000

python3 analysis/export_selected_data.py \
  6.535_rgk_eppi0_data_selected.root \
  6.535_rgk_eppi0_data.root \
  results/data/rgk_6.535_data_events.npz \
  --dictionary build/libROOTBranchesDict.so

python3 analysis/study_data_efficiency.py \
  results/data/rgk_6.535_data_events.npz \
  --output-dir results/data_efficiency/rgk_6.535_preliminary
```

The efficiency-study command defaults to unflagged P3/P4 runs and counts all
selected candidates. Supply a fixed event mask with `--selection-mask` for the
signal-yield study. Add `--include-classes P3 P4 L5` only after confirming L5
trigger and prescale compatibility. The run-current manifest deliberately
retains suspect, mixed-trigger, empty-target, and half-torus entries so each
nominal exclusion remains auditable.

For conversion of the complete reconstructed-DST holdings, including the
luminosity-scan runs, use the more compact converter config:

```bash
./build/hipo2root configs/processing/rgk/6.535/eppi0_data_full_dst.json \
  /path/to/rgk/6.535/reconstructed/dsts 0 1000000
```

This config retains the nominal QADB policy and DIS/topology skim, writes only
electron, proton, and photon particle rows, and requires at least one photon
pair with `0 <= m_gg <= 0.30 GeV`. The pair condition is existential: in an
event with more than two photons, any pair may satisfy it. Its upper boundary
is deliberately looser than the nominal post-processing pi0 window, whose
upper edge is approximately `0.285 GeV`; candidate choice and all final
selection cuts remain downstream.

For calibrated RGK 6.535 GeV acceptance studies, use the compact acceptance
processing config together with the acceptance post config. The processing
config applies the `6.535RGK_INCLUSIVE_GEMC_100M` proton energy-loss
corrections; the post config extends the nominal RGK EPPI0 base selection and
loads the `6.535RGK_INCLUSIVE_GEMC_100M` sampling-fraction parameters:

```bash
./build/hipo2root configs/processing/rgk/6.535/eppi0_mc_acceptance.json \
  /path/to/rgk/6.535/gemc 0 1000000

./build/post_process configs/post/rgk/6.535/eppi0_mc_acceptance.json \
  6.535_rgk_eppi0_mc_acceptance.root 1000000
```

Example processing commands on ifarm for the lower-threshold phase-space family:

```bash
./build/hipo2root configs/processing/rgk/6.535/aao_rad_q2_0.7_ep_1.00.json \
  /path/to/compatible/production 5 100000

./build/post_process configs/post/rgk/6.535/aao_rad_eppi0_loose.json \
  6.535_rgk_aao_rad_q2_0.7_ep_1.00.root 100000
```

Use the other processing config for the tighter generated phase-space family:

```bash
./build/hipo2root configs/processing/rgk/6.535/aao_rad_q2_0.9_ep_1.15.json \
  /path/to/compatible/production 5 100000

./build/post_process configs/post/rgk/6.535/aao_rad_eppi0_loose.json \
  6.535_rgk_aao_rad_q2_0.9_ep_1.15.root 100000
```

Compare any set of labeled selected outputs with the generic comparison tool.
The optional processing-root pairs add converter counters, charge, and selected
row fractions to the output summary:

```bash
python3 analysis/compare_root_distributions.py --density \
  --sample reference=/path/to/reference_selected.root \
  --sample candidate=/path/to/candidate_selected.root \
  --processing-root reference=/path/to/reference_converter.root \
  --processing-root candidate=/path/to/candidate_converter.root \
  --reference reference \
  --output-dir results/aao_rad/comparison
```

To compare converter-level particle distributions before EPPI0 candidate
selection, set `--tree rParticles`, choose branches such as `p theta phi`, add a
PID filter such as `--where 'rec.pid == 22'`, and load the ROOT dictionary when
object branches are not already discoverable. Quantitative metrics and output
formats are documented in `analysis/README.md`.

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
  --output parameters/proton_energy_loss/6.535RGK_INCLUSIVE_GEMC_100M.json \
  --plot-dir calibration_plots/proton_energy_loss/6.535RGK_INCLUSIVE_GEMC_100M \
  --dataset-tag 6.535RGK_INCLUSIVE_GEMC_100M \
  --beam-energy 6.535
```

To compare against a fiducial-volume derivation, first filter the matched
proton rows through post-processing while preserving the `event`, `rec`, and
`gen` branches, then write comparison products to a scratch output path:

```bash
./build/post_process \
  configs/post/rgk/6.535/calibration/proton_energy_loss_fiducial.json \
  6.535_rgk_proton_energy_loss_mc.root

python3 scripts/derive_proton_energy_loss.py \
  6.535_rgk_proton_energy_loss_mc_fiducial.root \
  --output /path/to/scratch/6.535RGK_fiducial_test.json \
  --plot-dir /path/to/scratch/rgk_6.535_fiducial_test \
  --dataset-tag 6.535RGK_fiducial_test \
  --beam-energy 6.535

python3 scripts/compare_proton_energy_loss.py \
  6.535_rgk_proton_energy_loss_mc_fiducial.root \
  parameters/proton_energy_loss/6.535RGK_INCLUSIVE_GEMC_100M.json \
  /path/to/scratch/6.535RGK_fiducial_test.json \
  --baseline-label inclusive \
  --updated-label fiducial \
  --output /path/to/scratch/rgk_6.535_fiducial_compare/residual_summary.csv \
  --binned-output /path/to/scratch/rgk_6.535_fiducial_compare/residual_summary_binned.csv \
  --plot-dir /path/to/scratch/rgk_6.535_fiducial_compare \
  --dataset-tag 6.535RGK_fiducial_compare \
  --beam-energy 6.535
```

The proton energy-loss processing configs use `outputPids: [2212]`. The
`finalState` requirement is still event-level, so events are kept when they
contain at least one reconstructed proton, but the output ROOT tree stores only
proton rows. This keeps the matched REC/GEN proton calibration sample broad
without writing every other reconstructed particle in those events.
The companion `rEvents` tree nevertheless records multiplicities
from the complete reconstructed particle list before `outputPids` is applied.

For the 100M inclusive GEMC sample split across
`/volatile/clas12/osg/storyf/11262` and `/volatile/clas12/osg/storyf/11263`,
use the same RGK 6.535 GeV calibration configs and pass both directories to
`hipo2root` in one invocation:

```bash
./build/hipo2root \
  configs/processing/rgk/6.535/calibration/proton_energy_loss_mc.json \
  /volatile/clas12/osg/storyf/11262 \
  /volatile/clas12/osg/storyf/11263

./build/hipo2root \
  configs/processing/rgk/6.535/calibration/sidis_electrons_mc.json \
  /volatile/clas12/osg/storyf/11262 \
  /volatile/clas12/osg/storyf/11263

./build/post_process \
  configs/post/rgk/6.535/calibration/electron_sf_candidates_mc.json \
  6.535_rgk_gemc_sidis_electrons.root
```

Then derive the parameter files from those ROOT files:

```bash
python3 scripts/derive_proton_energy_loss.py \
  6.535_rgk_proton_energy_loss_mc.root \
  --detector both \
  --output parameters/proton_energy_loss/6.535RGK_INCLUSIVE_GEMC_100M.json \
  --plot-dir calibration_plots/proton_energy_loss/6.535RGK_INCLUSIVE_GEMC_100M \
  --dataset-tag 6.535RGK_INCLUSIVE_GEMC_100M \
  --beam-energy 6.535

python3 scripts/derive_sampling_fraction.py \
  6.535_rgk_gemc_electron_sf_candidates.root \
  --gemc \
  --output parameters/sampling_fraction/SF_sigma_cut_params_6.535RGK_INCLUSIVE_GEMC_100M.json \
  --plot-dir calibration_plots/sampling_fraction/6.535RGK_INCLUSIVE_GEMC_100M \
  --dataset-tag 6.535RGK_INCLUSIVE_GEMC_100M \
  --beam-energy 6.535 \
  --run-group RGK \
  --skim INCLUSIVE_GEMC_100M \
  --torus 1
```
