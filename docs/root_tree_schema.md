# ROOT tree implementation notes

The converter separates event-level bookkeeping from particle-level rows. New
converter files use the following contracts.

## `ReconstructedEvents`

This is the canonical reconstructed event table. It contains one row for every
event that passes converter QADB, reconstructed final-state, and DIS filtering,
including events whose particle rows are restricted by `outputPids`.

Scalar event fields are `sourceFileId`, `sourceEventIndex`, `runNum`,
`eventNum`, `helicity`, and `charge`. Row-count fields distinguish the complete
reconstructed particle list from the particle table:

- `nReconstructedParticles`: all entries in `c12.getDetParticles()`;
- `nWrittenReconstructedParticles`: reconstructed rows retained after
  `outputPids`;
- `nParticleTreeRows`: all rows written to the particle tree, including any
  unmatched generated-only rows in legacy matched-MC output.

Topology multiplicities are evaluated before `outputPids` filtering. The
parallel vectors `topologyPids`, `topologyRequiredCounts`, `topologyExact`, and
`topologyPidCounts` retain the configured requirements and observed counts.
`topologyPidCountsFT`, `topologyPidCountsFD`, `topologyPidCountsCD`, and
`topologyPidCountsOther` split each observed count by the detector implied by
the reconstructed status.

Each configured PID also receives convenient scalar branches. For EPPI0 these
include `nPid11`, `nPid2212`, and `nPid22`, together with detector suffixes such
as `nPid2212FD` and `nPid2212CD`. A negative PID uses `Minus` in the branch name,
for example `nPidMinus211`.

The processing `reconstructedEventTree` block can rename or disable this tree;
it defaults to enabled with the name `ReconstructedEvents`.

## Converter particle tree

The processing config's `treeName` remains the reconstructed particle table
and defaults to `Events`. It contains one `rec` object per retained
reconstructed particle and an optional matched `gen` object. `rec.particleIdx`
is the position in the complete `c12.getDetParticles()` list, so PID filtering
can leave gaps.

For backward compatibility, this tree temporarily retains the repeated
`event` object used by existing post-processing, calibration scripts, and old
ROOT readers. New event-level diagnostics must use `ReconstructedEvents`; they
must not count repeated particle rows as events. The repeated object is a
compatibility foreign-key snapshot, not the canonical event table.

## `GeneratedEvents`

For configured MC conversion this remains the one-row-per-input-event truth
denominator. It is filled before QADB and reconstructed topology/DIS filtering,
so it is intentionally not row-aligned with `ReconstructedEvents`.
The source-aware key is the join contract.

## Selected output

The configured selected `Events` tree contains one row per retained candidate.
It propagates the converter topology vectors and exposes scalar PID counts such
as `nPid2212`, `nPid2212FD`, and `nPid2212CD`. When the companion
`ReconstructedEvents` tree is present, post-processing streams and joins those
counts by source-aware event key. Legacy inputs fall back to counting the
available particle rows.

`SelectedParticles` is the normalized selected-particle table. It contains one
row per selected role occurrence with the event key, role, occurrence,
`particleIdx`, PID, detector, sector, and selected kinematics. The legacy
selected-role vectors and scalar role branches remain in the candidate tree for
existing analysis and visualization consumers.

The post config keys `inputEventTree` and `outputParticleTree` default to
`ReconstructedEvents` and `SelectedParticles`, respectively.

## Proton-multiplicity diagnostic

The immediate cross-detector population is selected from either event-level
tree with

```text
nPid2212 >= 2 && nPid2212FD >= 1 && nPid2212CD >= 1
```

This identifies events with distinct FD and CD reconstructed proton rows. It
does not establish that the rows came from one physical proton; kinematic and
vertex residuals are still needed for that conclusion.
