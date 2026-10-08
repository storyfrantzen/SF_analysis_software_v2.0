# Generic particle and detector diagnostics

`analysis/plot_particle_kinematics.py` creates a balanced multipage PDF from
the scalar branches of any selected-event ROOT tree. It is independent of the
exclusive `ep -> e'p'pi0` topology required by `plot_event_kinematics.py`.

The utility discovers configured post-processing roles from branch triplets
such as `electronP`, `electronTheta`, and `electronPhi`. It also supports the
normalized `sParticles` layout, where the corresponding branches are `p`,
`theta`, and `phi`. Missing detector coordinates are omitted and recorded in
the JSON sidecar instead of causing the report to fail.

Available page groups are:

- `event`: run, helicity, and event-charge distributions;
- `dis`: `Q2`, `xB`, `nu`, `W`, `y`, and their coverage correlations;
- `kinematics`: momentum, polar angle, azimuth, angular coverage, and sector
  comparisons;
- `pid`: particle/detector identifiers, track quality, calorimeter energies,
  and sampling fraction when available;
- `pcal`, `ecin`, and `ecout`: calorimeter coordinate and energy maps;
- `dc`: region 1/2/3 trajectory maps and edge distances;
- `ft`: FTCAL occupancy and energy;
- `cvt`: layer-1 angular occupancy and CVT edge distances.

With no `--particle` or `--bank` arguments, the command discovers every
particle prefix and plots event/DIS/particle kinematics plus every populated
detector group:

```bash
python3 analysis/plot_particle_kinematics.py selected.root \
  --tree sEvents \
  --beam-energy 6.395 \
  --label "RGK Spring 2024, 6.395 GeV elastic candidates" \
  --output event_kinematics.pdf
```

Restrict the report by repeating `--particle` and `--bank`:

```bash
python3 analysis/plot_particle_kinematics.py selected.root \
  --particle electron \
  --bank dis \
  --bank kinematics \
  --bank pcal \
  --bank dc \
  --label "Electron coverage" \
  --output electron_coverage.pdf
```

For an `sParticles` tree, use `--particle row`. Arbitrary role names work as
long as the post-processing output contains the corresponding prefix, for
example `--particle piPlus` for `piPlusP`, `piPlusTheta`, and `piPlusPhi`.

Use `--where` for a ROOT expression such as `electronSector == 3`, or provide a
boolean `--selection-mask` aligned to the unfiltered input tree. These options
cannot be combined because a filtered RDataFrame no longer has the row
alignment of the external mask. `--max-rows` provides a deterministic cap for
large exploratory samples.

For a reconstructed peak whose center may move because of detector alignment
or calibration, use a fitted peak window. The broad `--peak-search` interval
locates the signal but does not fix its center. The utility fits a signal plus
background model, lets the signal center and width float, and retains the
requested Gaussian-equivalent containment. Fit diagnostics and the numerical
window are saved in the JSON sidecar and printed in the log:

```bash
python3 analysis/plot_particle_kinematics.py elastic_ep.root \
  --particle electron --particle proton \
  --peak-window W --peak-search 0.72 1.20 \
  --peak-expected 0.9382720813 \
  --peak-maximum-center-deviation 0.15 \
  --peak-maximum-sigma 0.12 --peak-n-sigma 3 \
  --label "Peak-selected elastic ep candidates" \
  --output elastic_ep_coverage.pdf
```

`--peak-expected` seeds and validates the floating peak; it is not the center
of the retained interval. The report should not be used when the fit fails its
center, width, signal-fraction, or significance checks. A requested `W` need
not be stored in the tree: when `Q2` and `nu` are available, the utility derives
`W` before fitting and records that source in the JSON sidecar.

The sidecar `<output-stem>_summary.json` records the input, tree, row counts,
particle prefixes, requested branches and groups, rendered pages, missing
groups, angle conventions, and optional input hash.
