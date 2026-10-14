"""Generator-neutral provenance records and campaign manifests.

The detector response continues to use unweighted generated events.  The
cross-section normalization stored here is generator provenance for estimators
such as the EXCLUGEN radiative correction; it is not an instruction to weight
response-matrix entries.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol


CAMPAIGN_MANIFEST_SCHEMA = "sf-generator-campaign-v1"
GENERATOR_MODES = frozenset(("born", "rad"))


class GeneratorProvenanceError(ValueError):
    """Raised when generator provenance is missing or internally inconsistent."""


class GeneratorProvider(Protocol):
    """Small interface implemented by generator-specific discovery backends."""

    provider_name: str

    def discover_campaign(self, production_root: str | Path, **kwargs: Any) -> "GeneratorCampaign":
        """Discover and validate one generator production campaign."""


@dataclass(frozen=True)
class GeneratorRunRecord:
    """Normalized provenance for one independently generated LUND run."""

    provider: str
    mode: str
    channel: str
    model: str
    generator_revision: str
    generator_branch: str
    dirty_state: str
    run_index: int
    seed: int
    lund_path: Path
    config_path: Path
    requested_events: int
    observed_events: int
    sigma_value: float
    sigma_units: str
    card_signature: str
    config_sha256: str
    execution_path: str
    beam_energy_GeV: float
    generation_vcut_GeV2: float
    t_nucl_GeV2: float
    physics_card: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.provider:
            raise GeneratorProvenanceError("generator provider must not be empty")
        if self.mode not in GENERATOR_MODES:
            raise GeneratorProvenanceError(
                f"generator mode must be one of {sorted(GENERATOR_MODES)}, got {self.mode!r}"
            )
        if not self.channel or not self.model:
            raise GeneratorProvenanceError("generator channel and model must not be empty")
        if not self.generator_revision or not self.generator_branch:
            raise GeneratorProvenanceError(
                "generator revision and branch must not be empty"
            )
        if self.dirty_state not in {"clean", "dirty"}:
            raise GeneratorProvenanceError(
                f"dirty_state must be 'clean' or 'dirty', got {self.dirty_state!r}"
            )
        if self.run_index < 0 or self.seed < 0:
            raise GeneratorProvenanceError("run index and seed must be non-negative")
        if self.requested_events <= 0 or self.observed_events < 0:
            raise GeneratorProvenanceError(
                "requested events must be positive and observed events non-negative"
            )
        if not math.isfinite(self.sigma_value) or self.sigma_value <= 0.0:
            raise GeneratorProvenanceError("generator cross section must be positive and finite")
        if self.sigma_units != "nb":
            raise GeneratorProvenanceError(
                f"generator cross-section units must be 'nb', got {self.sigma_units!r}"
            )
        if not self.card_signature or not self.config_sha256:
            raise GeneratorProvenanceError(
                "card signature and config SHA-256 must not be empty"
            )
        if not self.execution_path:
            raise GeneratorProvenanceError("execution path must not be empty")
        for value, label in (
            (self.beam_energy_GeV, "beam energy"),
            (self.generation_vcut_GeV2, "generation vcut"),
            (self.t_nucl_GeV2, "t_nucl"),
        ):
            if not math.isfinite(value):
                raise GeneratorProvenanceError(f"{label} must be finite")

    @property
    def revision_with_state(self) -> str:
        suffix = "-dirty" if self.dirty_state == "dirty" else ""
        return f"{self.generator_revision}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "mode": self.mode,
            "channel": self.channel,
            "model": self.model,
            "generator_revision": self.generator_revision,
            "generator_branch": self.generator_branch,
            "dirty_state": self.dirty_state,
            "run_index": self.run_index,
            "seed": self.seed,
            "lund_path": str(self.lund_path),
            "config_path": str(self.config_path),
            "requested_events": self.requested_events,
            "observed_events": self.observed_events,
            "sigma_value": self.sigma_value,
            "sigma_units": self.sigma_units,
            "card_signature": self.card_signature,
            "config_sha256": self.config_sha256,
            "execution_path": self.execution_path,
            "beam_energy_GeV": self.beam_energy_GeV,
            "generation_vcut_GeV2": self.generation_vcut_GeV2,
            "t_nucl_GeV2": self.t_nucl_GeV2,
            "physics_card": dict(self.physics_card),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratorRunRecord":
        try:
            return cls(
                provider=str(value["provider"]),
                mode=str(value["mode"]),
                channel=str(value["channel"]),
                model=str(value["model"]),
                generator_revision=str(value["generator_revision"]),
                generator_branch=str(value["generator_branch"]),
                dirty_state=str(value["dirty_state"]),
                run_index=int(value["run_index"]),
                seed=int(value["seed"]),
                lund_path=Path(str(value["lund_path"])),
                config_path=Path(str(value["config_path"])),
                requested_events=int(value["requested_events"]),
                observed_events=int(value["observed_events"]),
                sigma_value=float(value["sigma_value"]),
                sigma_units=str(value["sigma_units"]),
                card_signature=str(value["card_signature"]),
                config_sha256=str(value["config_sha256"]),
                execution_path=str(value["execution_path"]),
                beam_energy_GeV=float(value["beam_energy_GeV"]),
                generation_vcut_GeV2=float(value["generation_vcut_GeV2"]),
                t_nucl_GeV2=float(value["t_nucl_GeV2"]),
                physics_card=dict(value["physics_card"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GeneratorProvenanceError(
                f"invalid generator run record: {error}"
            ) from error


@dataclass(frozen=True)
class GeneratorCampaign:
    """Validated generator campaign containing normalized per-run provenance."""

    production_root: Path
    runs: tuple[GeneratorRunRecord, ...]
    analysis_config: str | None = None
    radiative_correction: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.runs:
            raise GeneratorProvenanceError("generator campaign contains no runs")
        identities = [(run.mode, run.run_index) for run in self.runs]
        if len(identities) != len(set(identities)):
            raise GeneratorProvenanceError(
                "generator campaign contains duplicate mode/run-index records"
            )
        for field_name in ("provider", "channel", "model"):
            values = {getattr(run, field_name) for run in self.runs}
            if len(values) != 1:
                raise GeneratorProvenanceError(
                    f"generator campaign mixes {field_name} values: {sorted(values)}"
                )
        if self.provider == "exclugen":
            mismatches = [
                (run.mode, run.run_index, run.requested_events, run.observed_events)
                for run in self.runs
                if run.requested_events != run.observed_events
            ]
            if mismatches:
                raise GeneratorProvenanceError(
                    f"EXCLUGEN campaign contains event-count mismatches: {mismatches}"
                )

    @property
    def provider(self) -> str:
        return self.runs[0].provider

    @property
    def channel(self) -> str:
        return self.runs[0].channel

    @property
    def model(self) -> str:
        return self.runs[0].model

    def runs_for_mode(self, mode: str) -> tuple[GeneratorRunRecord, ...]:
        if mode not in GENERATOR_MODES:
            raise GeneratorProvenanceError(f"unknown generator mode {mode!r}")
        return tuple(run for run in self.runs if run.mode == mode)

    def to_manifest(self) -> dict[str, Any]:
        revisions = sorted({run.generator_revision for run in self.runs})
        dirty_states = sorted({run.dirty_state for run in self.runs})
        branches = sorted({run.generator_branch for run in self.runs})
        execution_paths = sorted({run.execution_path for run in self.runs})
        signatures = sorted({run.card_signature for run in self.runs})
        beam_energies = sorted({run.beam_energy_GeV for run in self.runs})
        generation_vcuts = sorted({run.generation_vcut_GeV2 for run in self.runs})
        t_nucl_values = sorted({run.t_nucl_GeV2 for run in self.runs})
        generator: dict[str, Any] = {
            "provider": self.provider,
            "production_root": str(self.production_root),
            "model": self.model,
            "channel": self.channel,
            "revisions": revisions,
            "dirty_states": dirty_states,
            "branches": branches,
            "execution_paths": execution_paths,
            "card_signatures": signatures,
            "beam_energies_GeV": beam_energies,
            "generation_vcuts_GeV2": generation_vcuts,
            "t_nucl_values_GeV2": t_nucl_values,
            "normalization_quantity": "integrated_generated_cross_section",
            "normalization_units": "nb",
            "run_counts": {
                mode: len(self.runs_for_mode(mode)) for mode in sorted(GENERATOR_MODES)
            },
            "dirty_state_counts": {
                state: sum(run.dirty_state == state for run in self.runs)
                for state in dirty_states
            },
        }
        if len(revisions) == 1:
            generator["revision"] = revisions[0]
        if len(dirty_states) == 1:
            generator["dirty_state"] = dirty_states[0]
        if len(branches) == 1:
            generator["branch"] = branches[0]
        if len(execution_paths) == 1:
            generator["execution_path"] = execution_paths[0]
        if len(signatures) == 1:
            generator["card_signature"] = signatures[0]
        if len(beam_energies) == 1:
            generator["beam_energy_GeV"] = beam_energies[0]
        if len(generation_vcuts) == 1:
            generator["generation_vcut_GeV2"] = generation_vcuts[0]
        if len(t_nucl_values) == 1:
            generator["t_nucl_GeV2"] = t_nucl_values[0]

        manifest: dict[str, Any] = {
            "schema": CAMPAIGN_MANIFEST_SCHEMA,
            "generator": generator,
            "runs": [run.to_dict() for run in self.runs],
        }
        if self.analysis_config is not None:
            manifest["analysis_config"] = self.analysis_config
        if self.radiative_correction is not None:
            manifest["radiative_correction"] = dict(self.radiative_correction)
        return manifest

    def write_manifest(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n"
        )

    @classmethod
    def from_manifest(cls, path: str | Path) -> "GeneratorCampaign":
        source = Path(path)
        try:
            value = json.loads(source.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise GeneratorProvenanceError(
                f"could not read generator campaign manifest {source}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise GeneratorProvenanceError("generator campaign manifest must be an object")
        if value.get("schema") != CAMPAIGN_MANIFEST_SCHEMA:
            raise GeneratorProvenanceError(
                f"unknown generator campaign schema {value.get('schema')!r}"
            )
        generator = value.get("generator")
        raw_runs = value.get("runs")
        if not isinstance(generator, dict) or not isinstance(raw_runs, list):
            raise GeneratorProvenanceError(
                "generator campaign manifest requires generator and runs fields"
            )
        production_root = str(generator.get("production_root", "")).strip()
        if not production_root:
            raise GeneratorProvenanceError(
                "manifest generator.production_root must not be empty"
            )
        campaign = cls(
            production_root=Path(production_root),
            runs=tuple(GeneratorRunRecord.from_dict(run) for run in raw_runs),
            analysis_config=(
                None
                if value.get("analysis_config") is None
                else str(value["analysis_config"])
            ),
            radiative_correction=(
                None
                if value.get("radiative_correction") is None
                else dict(value["radiative_correction"])
            ),
        )
        for key, actual in (
            ("provider", campaign.provider),
            ("channel", campaign.channel),
            ("model", campaign.model),
        ):
            if generator.get(key) != actual:
                raise GeneratorProvenanceError(
                    f"manifest generator.{key} conflicts with run records"
                )
        summaries = (
            (
                "revisions",
                sorted({run.generator_revision for run in campaign.runs}),
            ),
            ("dirty_states", sorted({run.dirty_state for run in campaign.runs})),
            ("branches", sorted({run.generator_branch for run in campaign.runs})),
            ("execution_paths", sorted({run.execution_path for run in campaign.runs})),
            ("card_signatures", sorted({run.card_signature for run in campaign.runs})),
            ("beam_energies_GeV", sorted({run.beam_energy_GeV for run in campaign.runs})),
            (
                "generation_vcuts_GeV2",
                sorted({run.generation_vcut_GeV2 for run in campaign.runs}),
            ),
            ("t_nucl_values_GeV2", sorted({run.t_nucl_GeV2 for run in campaign.runs})),
        )
        for key, actual in summaries:
            recorded = generator.get(key)
            if recorded is not None:
                if not isinstance(recorded, list) or recorded != actual:
                    raise GeneratorProvenanceError(
                        f"manifest generator.{key} conflicts with run records"
                    )
        singular_summaries = (
            ("revision", summaries[0][1]),
            ("dirty_state", summaries[1][1]),
            ("branch", summaries[2][1]),
            ("execution_path", summaries[3][1]),
            ("card_signature", summaries[4][1]),
            ("beam_energy_GeV", summaries[5][1]),
            ("generation_vcut_GeV2", summaries[6][1]),
            ("t_nucl_GeV2", summaries[7][1]),
        )
        for key, actual in singular_summaries:
            if key in generator and (len(actual) != 1 or generator[key] != actual[0]):
                raise GeneratorProvenanceError(
                    f"manifest generator.{key} conflicts with run records"
                )
        if generator.get("normalization_units") != "nb":
            raise GeneratorProvenanceError(
                "manifest generator.normalization_units must be 'nb'"
            )
        if generator.get("normalization_quantity") != "integrated_generated_cross_section":
            raise GeneratorProvenanceError(
                "manifest generator.normalization_quantity is invalid"
            )
        run_counts = generator.get("run_counts")
        actual_counts = {
            mode: len(campaign.runs_for_mode(mode)) for mode in sorted(GENERATOR_MODES)
        }
        if run_counts is not None and run_counts != actual_counts:
            raise GeneratorProvenanceError(
                "manifest generator.run_counts conflicts with run records"
            )
        dirty_state_counts = generator.get("dirty_state_counts")
        actual_dirty_state_counts = {
            state: sum(run.dirty_state == state for run in campaign.runs)
            for state in sorted({run.dirty_state for run in campaign.runs})
        }
        if (
            dirty_state_counts is not None
            and dirty_state_counts != actual_dirty_state_counts
        ):
            raise GeneratorProvenanceError(
                "manifest generator.dirty_state_counts conflicts with run records"
            )
        return campaign


def merge_campaign_metadata(
    campaign: GeneratorCampaign,
    *,
    analysis_config: str | None = None,
    radiative_correction: Mapping[str, Any] | None = None,
) -> GeneratorCampaign:
    """Return ``campaign`` with artifact-level context attached."""

    return GeneratorCampaign(
        production_root=campaign.production_root,
        runs=campaign.runs,
        analysis_config=analysis_config,
        radiative_correction=radiative_correction,
    )
