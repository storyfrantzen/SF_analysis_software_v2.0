# Standard EPPI0 campaign template

This is the maintained starting point for an EPPI0 campaign. It records the
workflow that has emerged from the RGK and RGA analyses without tying the
procedure to one energy, torus polarity, or farm directory.

The repository contains code, configurations, tests, and small reference
tables. Large ROOT files, campaign NPZ files, PDFs, logs, and collaborator
exports belong in a campaign work directory outside the clone.

## Finding your way back in

When returning after a break, start with these files:

1. this document for the campaign sequence and artifact layout;
2. `README.md` for the executable and repository overview;
3. `configs/README.md` for configuration inheritance and calibration inputs;
4. `analysis/README.md` for detailed command options and artifact schemas;
5. `docs/analysis_pipeline.md` for the detector-correction and selection
   design.

Then inspect the exact code state before looking at old shell history:

```tcsh
cd /path/to/SF_analysis_software_v2.0
git status --short --branch
git log -8 --oneline
python3 analysis/run_analysis.py --help
```

Old logs explain what happened. The committed config, input manifest, output
metadata, and code commit define what the result means.

## Maintained repository surfaces

The normal campaign uses a small set of entry points:

| Purpose | Maintained entry point |
| --- | --- |
| HIPO conversion | `build/hipo2root` |
| Candidate construction | `build/post_process` |
| Data event export | `analysis/export_selected_data.py` |
| Data and GEMC exclusivity | `analysis/derive_exclusivity.py` |
| Response through structure functions | `analysis/run_analysis.py` |
| Response closure and iteration choice | `analysis/validate_response_closure.py` |
| Event and detector diagnostics | `analysis/plot_event_kinematics.py` |
| Data/GEMC shape comparison | `analysis/compare_event_kinematics.py` |
| Current-dependence study | `analysis/study_data_efficiency.py` |
| Polarity combination | `analysis/combine_polarity_cross_sections.py` |
| Beam-spin asymmetry | `analysis/beam_spin_asymmetry.py` |
| Collaborator CSV export | `analysis/export_reduced_cross_sections.py` |

Reference comparisons, model studies, calibration derivations, and special
diagnostics are optional layers. They should consume frozen campaign artifacts
rather than define a second extraction path.

## Campaign identity

Assign these values before processing anything:

- run group and data-taking period;
- beam energy;
- torus polarity;
- data source and run range;
- GEMC source, background level, software versions, and generator sample;
- processing, post-processing, and analysis configs;
- code commit;
- result status: `baseline`, `validation`, or `production`.

Do not label a new result `nominal` until the response closure, correction
audits, and downstream fit checks have passed. Use descriptive variant names,
for example:

```text
born_0nA_response_uncorrected_v1
iterations_1_count_bootstrap_v1
unit_current_weights_v1
selection_window_scan_v2
```

A variant directory is immutable once it has been used in a comparison or
shared. Create a new variant for a changed input, config, correction, or
uncertainty model.

## Standard host directory

Use one host directory for one energy or one logically combined campaign:

```text
CAMPAIGN/
  01_provenance/
    data/
    gemc/
    configs/
    environment/
  02_processing/
    data/
    gemc/
  03_selection/
    data/
    gemc/
  04_response/
    POLARITY/VARIANT/
  05_corrections/
    current_efficiency/
    radiative/
    bin_centering/
  06_unfolding/
    POLARITY/VARIANT/
  07_xsec/
    POLARITY/VARIANT/
  08_harmonics/
    POLARITY/VARIANT/
  09_systematics/
    POLARITY/
  10_logs/
    processing/
    selection/
    response/
    unfolding/
    diagnostics/
```

For a single-polarity campaign, `POLARITY` may be omitted consistently. For a
multi-polarity campaign, keep the two polarities separate through the
cross-section stage and combine them only with the dedicated combination
utility.

The numbered directories describe dependencies, not a requirement that every
optional correction already exist. A baseline extraction may deliberately use
unit corrections, but its variant name and provenance must say so.

## 1. Freeze inputs and environment

Build manifests without opening every HIPO file. Store absolute paths, sizes,
and checksums of the manifests. Record software versions separately.

```tcsh
set repo = /path/to/SF_analysis_software_v2.0
set work = /work/clas12/$USER/RUN_GROUP/CAMPAIGN_NAME
set stamp = `date -u +%Y%m%dT%H%M%SZ`

mkdir -p "$work/01_provenance/data" \
         "$work/01_provenance/gemc" \
         "$work/01_provenance/configs" \
         "$work/01_provenance/environment"

find -L /path/to/data -type f -name '*.hipo' -print | sort \
  >! "$work/01_provenance/data/hipo_manifest_${stamp}.txt"
find -L /path/to/gemc -type f -name '*.hipo' -print | sort \
  >! "$work/01_provenance/gemc/hipo_manifest_${stamp}.txt"

sha256sum "$work/01_provenance/data/hipo_manifest_${stamp}.txt" \
          "$work/01_provenance/gemc/hipo_manifest_${stamp}.txt" \
  >! "$work/01_provenance/input_manifest_sha256_${stamp}.txt"

git -C "$repo" rev-parse HEAD \
  >! "$work/01_provenance/environment/git_commit.txt"
git -C "$repo" status --short \
  >! "$work/01_provenance/environment/git_status.txt"
module list >&! "$work/01_provenance/environment/modules_${stamp}.txt"
```

Also preserve the exact configs used. A symlink to a changing clone is not
enough for final provenance.

### Input gate

Proceed only after checking:

- file counts match the intended production;
- files are nonempty and readable;
- duplicate basenames are absent or explicitly handled;
- data runs match the intended run period and polarity;
- GEMC generator statistics and missing jobs are documented;
- GEMC and reconstruction software versions are known.

## 2. Prepare the three configuration layers

Every campaign needs three distinct configuration decisions:

1. `configs/processing/...`: HIPO reading, QADB, final-state filtering, stable
   corrections, and ROOT branches;
2. `configs/post/...`: candidate construction, particle cuts, topology policy,
   and loose exclusivity;
3. `configs/analysis/...`: beam energy, phase space, four-dimensional binning,
   target constants, minimum acceptance, and final exclusivity settings.

Data and GEMC usually have separate processing and post-processing configs.
Use shared base configs through inheritance where their definitions truly
match. Do not force data and GEMC to use the same fitted exclusivity windows.

### Configuration gate

Before a full run:

- validate JSON syntax;
- confirm the output filenames encoded by the processing configs;
- confirm the beam energy and polarity;
- confirm QADB and run-selection behavior for data;
- confirm enabled detector corrections and fiducials;
- confirm generated-event storage for GEMC;
- label provisional binning as provisional.

## 3. Build once for the campaign

Load the supported JLab environment and build in a fresh directory. The exact
module versions belong in provenance.

```tcsh
cd "$repo"
module purge
source docs/jlab-module-setup.csh
cmake -S . -B build
cmake --build build -j 8
```

If a campaign requires a different QADB version, load it before configuring
CMake and verify `QADB_INCLUDE_DIR` in `build/CMakeCache.txt`.

## 4. Convert and post-process data and GEMC

Run the converter in its intended output directory because filenames are
config-driven. Capture every command in a log.

```tcsh
mkdir -p "$work/02_processing/data/VARIANT" \
         "$work/02_processing/gemc/VARIANT" \
         "$work/03_selection/data/VARIANT" \
         "$work/03_selection/gemc/VARIANT" \
         "$work/10_logs/processing" \
         "$work/10_logs/selection"

cd "$work/02_processing/data/VARIANT"
"$repo/build/hipo2root" "$data_processing_config" $data_hipo_paths \
  >&! "$work/10_logs/processing/data_${stamp}.log"

cd "$work/02_processing/gemc/VARIANT"
"$repo/build/hipo2root" "$gemc_processing_config" $gemc_hipo_paths \
  >&! "$work/10_logs/processing/gemc_${stamp}.log"

cd "$work/03_selection/data/VARIANT"
"$repo/build/post_process" "$data_post_config" "$data_processing_root" \
  >&! "$work/10_logs/selection/data_${stamp}.log"

cd "$work/03_selection/gemc/VARIANT"
"$repo/build/post_process" "$gemc_post_config" "$gemc_processing_root" \
  >&! "$work/10_logs/selection/gemc_${stamp}.log"
```

### Processing gate

Record and inspect:

- input files opened and skipped;
- total, accepted, and selected event counts;
- accumulated data charge and excluded runs;
- generated and reconstructed GEMC counts;
- topology populations;
- candidate cut-flow counts.

Unexpected detector topologies, skipped files, or missing charge accounting
must be understood before building a response.

## 5. Derive final selections independently

Export a compact data sample and derive final data cuts from data. Derive GEMC
cuts from GEMC. This keeps resolution differences from being disguised by a
shared fitted window.

```tcsh
python3 "$repo/analysis/export_selected_data.py" \
  "$data_selected_root" "$data_processing_root" "$data_events" \
  --dictionary "$repo/build/libROOTBranchesDict.so"

python3 "$repo/analysis/derive_exclusivity.py" "$data_events" \
  --config "$analysis_config" \
  --cuts "$data_exclusivity_npz" \
  --mask "$data_exclusivity_mask" \
  --diagnostics "$data_exclusivity_pdf"

python3 "$repo/analysis/derive_exclusivity.py" "$gemc_selected_root" \
  --format selected-root \
  --dictionary "$repo/build/libROOTBranchesDict.so" \
  --config "$analysis_config" \
  --cuts "$gemc_exclusivity_npz" \
  --mask "$gemc_exclusivity_mask" \
  --diagnostics "$gemc_exclusivity_pdf"
```

Inspect the fitted centers and widths, cut stability, signal fraction,
topology populations, and N-minus-one diagnostics. A fit that merely exits
successfully is not automatically a valid selection model.

## 6. Produce event and detector diagnostics

Generate the same candidate-level report for data and GEMC. Compare shapes
after the final masks, while remembering that absolute normalization and
current-dependent weights may be separate questions.

```tcsh
python3 "$repo/analysis/plot_event_kinematics.py" "$data_selected_root" \
  --selection-mask "$data_exclusivity_mask" \
  --config "$analysis_config" \
  --dictionary "$repo/build/libROOTBranchesDict.so" \
  --label "$campaign_label data" \
  --output "$data_kinematics_pdf"

python3 "$repo/analysis/plot_event_kinematics.py" "$gemc_selected_root" \
  --selection-mask "$gemc_exclusivity_mask" \
  --config "$analysis_config" \
  --dictionary "$repo/build/libROOTBranchesDict.so" \
  --label "$campaign_label GEMC" \
  --output "$gemc_kinematics_pdf"
```

Check DIS variables, reconstructed momenta and angles, topology fractions,
calorimeter and tracking occupancy, exclusivity variables, and localized
data/GEMC discrepancies. Shape mismodeling is a detector/systematic question;
response closure alone cannot certify it.

## 7. Build and inspect the response

Build the response from the converter ROOT, selected GEMC ROOT, independently
derived GEMC mask, and analysis config.

```tcsh
python3 "$repo/analysis/run_analysis.py" response-root \
  "$gemc_processing_root" "$gemc_selected_root" \
  --selection-mask "$gemc_exclusivity_mask" \
  --config "$analysis_config" \
  --dictionary "$repo/build/libROOTBranchesDict.so" \
  --output-dir "$response_dir"

python3 "$repo/analysis/run_analysis.py" acceptance-plots \
  "$response_dir/response_meta.npz" \
  --response-matrix "$response_dir/response_matrix.npz" \
  --quilt \
  --output-dir "$response_dir/diagnostics"

python3 "$repo/analysis/run_analysis.py" response-plots \
  "$response_dir/response_matrix.npz" \
  "$response_dir/response_meta.npz" \
  --output "$response_dir/diagnostics/response_diagnostics.pdf"
```

Record generated support, selected REC yield, acceptance, feed-in, topology
composition, purity/migration, and zero-acceptance bins.

## 8. Certify the unfolding estimator

Choose the IBU iteration from closure, not from historical convention or the
smoothness of data. The maintained production uncertainty path uses integer
response counts and a joint data-and-response count bootstrap.

```tcsh
python3 "$repo/analysis/validate_response_closure.py" \
  "$gemc_processing_root" "$gemc_selected_root" \
  --selection-mask "$gemc_exclusivity_mask" \
  --dictionary "$repo/build/libROOTBranchesDict.so" \
  --config "$analysis_config" \
  --folds 5 \
  --iterations 0 1 2 4 8 \
  --bootstrap 300 \
  --response-uncertainty count-bootstrap \
  --output-dir "$closure_dir" \
  --label "$campaign_label"
```

The selected iteration must be coverage-qualified for that response. Preserve
the closure summary, fixed-truth metrics, harmonic-cell metrics, and diagnostic
PDF. Closure tests the response/unfolding estimator under simulated detector
behavior. It does not validate data background subtraction, luminosity,
current corrections, radiative corrections, bin centering, or data/GEMC
detector mismodeling.

## 9. Freeze correction artifacts

Treat corrections as inputs with their own provenance and validation:

- topology-aware current/tracking efficiency;
- diphoton sideband background model;
- radiative correction and reliability mask;
- bin-centering correction and convergence study;
- global normalization constants.

For each correction record source samples, code commit, config, reference
current or model settings, uncertainty definition, validity mask, and checksum.
Use unit-correction artifacts only for named systematic variants.

## 10. Run the extraction

Use the closure-qualified iteration and uncertainty method. Keep the unfolding,
cross-section, harmonics, and structure-function artifacts in parallel variant
directories.

```tcsh
python3 "$repo/analysis/run_analysis.py" unfold \
  "$data_events" \
  "$response_dir/response_matrix.npz" \
  "$response_dir/response_meta.npz" \
  --selection-mask "$data_exclusivity_mask" \
  --background-cuts "$data_exclusivity_npz" \
  --background-negative-policy "$background_negative_policy" \
  --current-efficiency-correction "$current_efficiency_json" \
  --radiative-correction "$radiative_correction_npz" \
  --config "$analysis_config" \
  --iterations "$certified_iterations" \
  --bootstrap 300 \
  --response-uncertainty count-bootstrap \
  --output "$unfolding_npz"

python3 "$repo/analysis/run_analysis.py" cross-section \
  "$unfolding_npz" \
  --config "$analysis_config" \
  --bin-centering "$bin_centering_npz" \
  --output "$cross_section_npz"

python3 "$repo/analysis/run_analysis.py" fit-harmonics \
  "$cross_section_npz" \
  --output "$harmonics_npz"

python3 "$repo/analysis/run_analysis.py" structure-functions \
  "$harmonics_npz" \
  --cross-section "$cross_section_npz" \
  --config "$analysis_config" \
  --output "$structure_functions_npz"
```

Keep the background-negative policy at its default `error` for the first run.
If small statistical deficits are inspected and clipping is adopted, set the
variable to `clip` and report the number of affected bins and total clipped
deficit.

## 11. Inspect final quality before comparison

Produce the numerical audit and plots from the exact final variant:

```tcsh
python3 "$repo/analysis/run_analysis.py" campaign-diagnostics \
  "$unfolding_npz" "$cross_section_npz" "$harmonics_npz" \
  --response-meta "$response_dir/response_meta.npz" \
  --title "$campaign_label" \
  --output "$harmonics_dir/campaign_diagnostics.md"

python3 "$repo/analysis/run_analysis.py" cross-section-plots \
  "$cross_section_npz" "$harmonics_npz" \
  --output-dir "$harmonics_dir/diagnostics"

python3 "$repo/analysis/run_analysis.py" harmonic-plots \
  "$harmonics_npz" \
  --output-dir "$harmonics_dir/diagnostics" \
  --quilt
```

Review valid-bin counts, covariance source, bootstrap count, numerical and
production-quality harmonic fits, rejection reasons, and structure-function
coverage. A change that increases uncertainties can improve coverage while
also changing which fits pass quality cuts; both effects belong in the audit.

## 12. Add optional layers after the single campaign is sound

### Multiple torus polarities

Extract each polarity independently through cross sections. Compare their
overlap, then combine with `combine_polarity_cross_sections.py`. The combined
artifact must state which uncertainties are shared and which are treated as
independent.

### Beam-spin asymmetry

BSA is a parallel data product with helicity, charge, and polarization
provenance. Audit physical helicity signs and charge balance before fitting.
Do not infer BSA validity from the unpolarized extraction.

### External references and theory

Compare only matched observables and document coordinate offsets, beam-energy
differences, epsilon differences, and missing systematic covariance. A model
or reference comparison is a validation layer, not a correction to the data.

## 13. Freeze and share a result

A share directory should contain only the intended public-facing files:

- reduced cross-section CSV with coordinates, epsilon, values, and statistical
  uncertainties;
- covariance information or a precise statement of what is omitted;
- README describing units, masks, corrections, iteration, uncertainty method,
  and provisional status;
- export summary with source paths and checksums;
- optional comparison PDFs.

Run automated export tests, count rows, check finite coordinates, and inspect a
few records manually. Never reconstruct missing coordinates from bin indices in
the collaborator's code when the campaign artifact can export them directly.

## Minimum completion checklist

A campaign is ready for physics comparison when all applicable items are true:

- [ ] input manifests and environment provenance are frozen;
- [ ] data charge, run exclusions, and QADB behavior are audited;
- [ ] data and GEMC selections were derived and inspected independently;
- [ ] candidate kinematics and detector occupancy were reviewed;
- [ ] response support, migration, feed-in, and topology content were reviewed;
- [ ] the chosen unfolding iteration has fixed-truth coverage certification;
- [ ] response statistics are propagated with the certified method;
- [ ] background, current, radiative, bin-centering, and normalization inputs
      are identified and checksummed;
- [ ] cross-section covariance reaches the harmonic fit;
- [ ] fit failures and quality rejections are understood;
- [ ] systematic variants use separate directories;
- [ ] final diagnostics and collaborator exports point to the same variant.

## Repository hygiene

Keep the clone understandable by following four rules:

1. Commit reusable code, configs, tests, small reference tables, and durable
   documentation.
2. Keep campaign products, PDFs, ROOT files, logs, and temporary render output
   outside the clone or in ignored local directories.
3. Pair a new utility with a test and documentation; otherwise keep it in a
   separate research workspace until its interface stabilizes.
4. Remove compatibility wrappers only after all documented commands and farm
   scripts have migrated to their replacement.

The detailed command catalog remains in `analysis/README.md`. This template is
the stable workflow; the catalog explains individual tools.
