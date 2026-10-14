"""Discovery and strict provenance validation for EXCLUGEN productions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from .base import GeneratorCampaign, GeneratorProvenanceError, GeneratorRunRecord


PROVIDER_NAME = "exclugen"
SIGMA_UNITS = "nb"
MODES = ("born", "rad")
LUND_RUN_INDEX = re.compile(r"_(\d+)\.lund$")
CONFIG_RUN_INDEX = re.compile(r"^(born|rad)_(\d+)\.config\.json$")
PHYSICS_CARD_FIELDS = (
    "model",
    "ebeam",
    "W2",
    "Q2",
    "t",
    "vcut",
    "t_nucl",
    "ihel",
    "idecay",
    "y",
    "xB",
    "iacc",
    "vmin",
    "itarg",
    "vz",
    "ptarg",
    "fermi",
    "theta_e",
)


def nested_values(value: Any, key: str) -> list[Any]:
    """Return all values named ``key`` in a nested JSON-compatible object."""

    found: list[Any] = []
    if isinstance(value, dict):
        if key in value:
            found.append(value[key])
        for child in value.values():
            found.extend(nested_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(nested_values(child, key))
    return found


def sigma_nb(config: Mapping[str, Any]) -> float:
    """Read one consistent, positive ``sigma_nb`` from an EXCLUGEN config."""

    raw_values = nested_values(config, "sigma_nb")
    if not raw_values:
        raise GeneratorProvenanceError("config contains no sigma_nb")
    try:
        values = [float(value) for value in raw_values]
    except (TypeError, ValueError) as error:
        raise GeneratorProvenanceError(
            f"invalid sigma_nb value(s): {raw_values}"
        ) from error
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise GeneratorProvenanceError(f"invalid sigma_nb value(s): {values}")
    if not all(
        math.isclose(value, values[0], rel_tol=1.0e-12, abs_tol=0.0)
        for value in values[1:]
    ):
        raise GeneratorProvenanceError(
            f"config contains inconsistent sigma_nb values: {values}"
        )
    return values[0]


def lund_run_index(path: str | Path) -> int:
    match = LUND_RUN_INDEX.search(Path(path).name)
    if match is None:
        raise GeneratorProvenanceError(
            f"cannot extract EXCLUGEN run index from {Path(path).name}"
        )
    return int(match.group(1))


def config_run_identity(path: str | Path) -> tuple[str, int]:
    match = CONFIG_RUN_INDEX.match(Path(path).name)
    if match is None:
        raise GeneratorProvenanceError(
            f"cannot extract EXCLUGEN mode/run index from {Path(path).name}"
        )
    return match.group(1), int(match.group(2))


def count_lund_events(path: str | Path) -> int:
    """Count complete LUND events without loading their momenta into memory."""

    source_path = Path(path)
    events = 0
    try:
        with source_path.open("r", errors="replace") as source:
            for line_number, header in enumerate(source, start=1):
                stripped = header.strip()
                if not stripped or stripped.startswith(("#", "!")):
                    continue
                fields = stripped.split()
                try:
                    particle_count = int(fields[0])
                except (IndexError, ValueError) as error:
                    raise GeneratorProvenanceError(
                        f"{source_path}:{line_number}: invalid LUND event header"
                    ) from error
                if particle_count <= 0:
                    raise GeneratorProvenanceError(
                        f"{source_path}:{line_number}: non-positive particle count"
                    )
                for particle_number in range(1, particle_count + 1):
                    particle = source.readline()
                    line_number += 1
                    if not particle:
                        raise GeneratorProvenanceError(
                            f"{source_path}: truncated event {events + 1} at "
                            f"particle {particle_number}/{particle_count}"
                        )
                    fields = particle.split()
                    if len(fields) < 4:
                        raise GeneratorProvenanceError(
                            f"{source_path}:{line_number}: malformed LUND particle row"
                        )
                    try:
                        int(fields[3])
                    except ValueError as error:
                        raise GeneratorProvenanceError(
                            f"{source_path}:{line_number}: invalid LUND particle PID"
                        ) from error
                events += 1
    except OSError as error:
        raise GeneratorProvenanceError(
            f"could not read EXCLUGEN LUND file {source_path}: {error}"
        ) from error
    return events


def physics_card(card: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in PHYSICS_CARD_FIELDS if key not in card]
    if missing:
        raise GeneratorProvenanceError(
            f"EXCLUGEN card is missing required physics fields: {missing}"
        )
    return {key: card[key] for key in PHYSICS_CARD_FIELDS}


def card_signature(card: Mapping[str, Any]) -> str:
    payload = json.dumps(
        physics_card(card), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GeneratorProvenanceError(f"EXCLUGEN config {label} must be an object")
    return value


def _require_positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise GeneratorProvenanceError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise GeneratorProvenanceError(f"{label} must be a positive integer") from error
    try:
        exact = float(value) == float(result)
    except (TypeError, ValueError):
        exact = False
    if result <= 0 or not exact:
        raise GeneratorProvenanceError(f"{label} must be a positive integer")
    return result


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise GeneratorProvenanceError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
        exact = float(value) == float(result)
    except (TypeError, ValueError) as error:
        raise GeneratorProvenanceError(
            f"{label} must be a non-negative integer"
        ) from error
    if result < 0 or not exact:
        raise GeneratorProvenanceError(f"{label} must be a non-negative integer")
    return result


def _same_json_value(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _require_matching(left: Any, right: Any, label: str) -> None:
    if not _same_json_value(left, right):
        raise GeneratorProvenanceError(
            f"EXCLUGEN config contains conflicting {label}: {left!r} versus {right!r}"
        )


def _revision_and_dirty_state(stamp: Mapping[str, Any]) -> tuple[str, str]:
    raw_revision = str(stamp.get("git_commit", "")).strip()
    if not raw_revision:
        raise GeneratorProvenanceError("EXCLUGEN config has no git_commit")
    suffix_dirty = raw_revision.endswith("-dirty")
    revision = raw_revision[:-6] if suffix_dirty else raw_revision
    explicit_dirty = stamp.get("git_dirty")
    if explicit_dirty is not None:
        if not isinstance(explicit_dirty, bool):
            raise GeneratorProvenanceError(
                "EXCLUGEN config git_dirty must be a JSON boolean"
            )
        if explicit_dirty != suffix_dirty:
            raise GeneratorProvenanceError(
                "EXCLUGEN config git_commit and git_dirty disagree"
            )
    return revision, "dirty" if suffix_dirty else "clean"


def _canonical_channel(stamp: Mapping[str, Any], model: str) -> str:
    sf = _require_mapping(stamp.get("sf"), "stamp.sf")
    raw_channel = str(sf.get("channel", "")).strip()
    if not raw_channel:
        raw_channel = model.split(".", 1)[0]
    target = str(sf.get("target", "")).strip()
    if raw_channel in {"pi0", "eta"} and target in {"p", "n"}:
        return f"{raw_channel}{target}"
    return raw_channel


def _revision_matches(expected: str, actual: str) -> bool:
    expected_base = expected.removesuffix("-dirty")
    actual_base = actual.removesuffix("-dirty")
    return expected_base.startswith(actual_base) or actual_base.startswith(expected_base)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise GeneratorProvenanceError(
            f"could not read EXCLUGEN config {path}: {error}"
        ) from error
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise GeneratorProvenanceError(
            f"invalid JSON in EXCLUGEN config {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise GeneratorProvenanceError(f"EXCLUGEN config {path} must be an object")
    return value, hashlib.sha256(payload).hexdigest()


def _record_from_pair(
    lund_path: Path,
    config_path: Path,
    *,
    mode: str,
    run_index: int,
    execution_path: str,
    expected_channel: str | None,
    expected_model: str | None,
    expected_revision: str | None,
) -> GeneratorRunRecord:
    config, config_sha256 = _load_config(config_path)
    card = _require_mapping(config.get("card"), "card")
    stamp = _require_mapping(config.get("stamp"), "stamp")
    run = _require_mapping(stamp.get("run"), "stamp.run")
    sf = _require_mapping(stamp.get("sf"), "stamp.sf")

    if str(stamp.get("code", "")) != "exclurad_py":
        raise GeneratorProvenanceError(
            f"{config_path}: expected stamp.code='exclurad_py'"
        )
    if run.get("mode") != mode:
        raise GeneratorProvenanceError(
            f"{config_path}: run.mode {run.get('mode')!r} does not match {mode!r}"
        )
    expected_iborn = 1 if mode == "born" else 0
    if int(card.get("iborn", -1)) != expected_iborn:
        raise GeneratorProvenanceError(
            f"{config_path}: card.iborn does not match {mode!r} mode"
        )

    model = str(card.get("model", "")).strip()
    if not model:
        raise GeneratorProvenanceError(f"{config_path}: card.model is missing")
    _require_matching(model, sf.get("model"), "card/stamp model")
    _require_matching(card.get("ebeam"), run.get("Ebeam"), "beam energy")
    _require_matching(card.get("ebeam"), sf.get("Ebeam"), "model beam energy")
    _require_matching(card.get("t_nucl"), run.get("t_nucl_card"), "t_nucl")
    _require_matching(card.get("vcut"), run.get("vcut"), "generation vcut")
    run_box = _require_mapping(run.get("box"), "stamp.run.box")
    for key in ("Q2", "W2", "t"):
        _require_matching(card.get(key), run_box.get(key), f"{key} generation box")

    requested_events = _require_positive_integer(card.get("nev"), "card.nev")
    _require_matching(requested_events, run.get("n_events"), "requested event count")
    seed = _require_nonnegative_integer(card.get("seed"), "card.seed")
    _require_matching(seed, run.get("seed"), "random seed")
    observed_events = count_lund_events(lund_path)
    if observed_events != requested_events:
        raise GeneratorProvenanceError(
            f"{lund_path}: observed {observed_events} events, card requests "
            f"{requested_events}"
        )

    channel = _canonical_channel(stamp, model)
    revision, dirty_state = _revision_and_dirty_state(stamp)
    branch = str(stamp.get("git_branch", "")).strip()
    if not branch:
        raise GeneratorProvenanceError(f"{config_path}: stamp.git_branch is missing")
    if expected_channel is not None and channel != expected_channel:
        raise GeneratorProvenanceError(
            f"{config_path}: expected channel {expected_channel!r}, found {channel!r}"
        )
    if expected_model is not None and model != expected_model:
        raise GeneratorProvenanceError(
            f"{config_path}: expected model {expected_model!r}, found {model!r}"
        )
    if expected_revision is not None and not _revision_matches(
        expected_revision, revision
    ):
        raise GeneratorProvenanceError(
            f"{config_path}: expected revision {expected_revision!r}, found "
            f"{revision!r}"
        )

    return GeneratorRunRecord(
        provider=PROVIDER_NAME,
        mode=mode,
        channel=channel,
        model=model,
        generator_revision=revision,
        generator_branch=branch,
        dirty_state=dirty_state,
        run_index=run_index,
        seed=seed,
        lund_path=lund_path,
        config_path=config_path,
        requested_events=requested_events,
        observed_events=observed_events,
        sigma_value=sigma_nb(config),
        sigma_units=SIGMA_UNITS,
        card_signature=card_signature(card),
        config_sha256=config_sha256,
        execution_path=execution_path,
        beam_energy_GeV=float(card["ebeam"]),
        generation_vcut_GeV2=float(card["vcut"]),
        t_nucl_GeV2=float(card["t_nucl"]),
        physics_card=physics_card(card),
    )


def _indexed_lund_files(directory: Path) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in sorted(directory.glob("*.lund")):
        index = lund_run_index(path)
        if index in indexed:
            raise GeneratorProvenanceError(
                f"duplicate EXCLUGEN LUND run index {index}: {indexed[index]} and {path}"
            )
        indexed[index] = path
    return indexed


def _indexed_config_files(directory: Path, mode: str) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in sorted(directory.glob(f"{mode}_*.config.json")):
        config_mode, index = config_run_identity(path)
        if config_mode != mode:
            raise GeneratorProvenanceError(
                f"{path}: config filename mode does not match {mode!r}"
            )
        if index in indexed:
            raise GeneratorProvenanceError(
                f"duplicate EXCLUGEN config run index {index}: {indexed[index]} and {path}"
            )
        indexed[index] = path
    return indexed


def discover_exclugen_mode(
    production_root: str | Path,
    mode: str,
    *,
    execution_path: str,
    expected_runs: int | None = None,
    expected_channel: str | None = "pi0p",
    expected_model: str | None = None,
    expected_revision: str | None = None,
    allow_mixed_dirty_state: bool = False,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
) -> tuple[GeneratorRunRecord, ...]:
    """Discover one EXCLUGEN mode and validate every LUND/config pair."""

    if mode not in MODES:
        raise GeneratorProvenanceError(f"unsupported EXCLUGEN mode {mode!r}")
    if not execution_path.strip():
        raise GeneratorProvenanceError("execution_path must not be empty")
    if progress_interval <= 0:
        raise GeneratorProvenanceError("progress_interval must be positive")
    root = Path(production_root)
    lund_dir = root / mode / "lund_osg"
    config_dir = root / mode / "prov"
    lund_files = _indexed_lund_files(lund_dir)
    config_files = _indexed_config_files(config_dir, mode)
    if not lund_files:
        raise GeneratorProvenanceError(f"{mode}: no LUND files found under {lund_dir}")
    if set(lund_files) != set(config_files):
        missing_configs = sorted(set(lund_files) - set(config_files))
        missing_lund = sorted(set(config_files) - set(lund_files))
        raise GeneratorProvenanceError(
            f"{mode}: LUND/config run-index mismatch; "
            f"missing configs={missing_configs}, missing LUND={missing_lund}"
        )
    if expected_runs is not None and len(lund_files) != expected_runs:
        raise GeneratorProvenanceError(
            f"{mode}: expected {expected_runs} runs, found {len(lund_files)}"
        )

    records_list: list[GeneratorRunRecord] = []
    ordered_indices = sorted(lund_files)
    for position, index in enumerate(ordered_indices, start=1):
        records_list.append(
            _record_from_pair(
                lund_files[index],
                config_files[index],
                mode=mode,
                run_index=index,
                execution_path=execution_path,
                expected_channel=expected_channel,
                expected_model=expected_model,
                expected_revision=expected_revision,
            )
        )
        if progress is not None and (
            position % progress_interval == 0 or position == len(ordered_indices)
        ):
            progress(
                f"[EXCLUGEN] {mode}: validated {position}/{len(ordered_indices)} "
                "LUND/config pairs"
            )
    records = tuple(records_list)
    consistent_fields = [
        "channel",
        "model",
        "generator_revision",
        "card_signature",
    ]
    if not allow_mixed_dirty_state:
        consistent_fields.append("dirty_state")
    for field_name in consistent_fields:
        values = {getattr(record, field_name) for record in records}
        if len(values) != 1:
            raise GeneratorProvenanceError(
                f"{mode}: mixed EXCLUGEN {field_name} values: {sorted(values)}"
            )
    return records


def discover_exclugen_campaign(
    production_root: str | Path,
    *,
    execution_path: str,
    expected_runs: int | None = None,
    expected_channel: str | None = "pi0p",
    expected_model: str | None = None,
    expected_revision: str | None = None,
    allow_revision_mismatch: bool = False,
    allow_mixed_dirty_state: bool = False,
    require_matching_run_indices: bool = True,
    progress: Callable[[str], None] | None = None,
    progress_interval: int = 25,
) -> GeneratorCampaign:
    """Discover paired Born/radiative EXCLUGEN modes as one campaign."""

    root = Path(production_root)
    born = discover_exclugen_mode(
        root,
        "born",
        execution_path=execution_path,
        expected_runs=expected_runs,
        expected_channel=expected_channel,
        expected_model=expected_model,
        expected_revision=expected_revision,
        allow_mixed_dirty_state=allow_mixed_dirty_state,
        progress=progress,
        progress_interval=progress_interval,
    )
    radiative = discover_exclugen_mode(
        root,
        "rad",
        execution_path=execution_path,
        expected_runs=expected_runs,
        expected_channel=expected_channel,
        expected_model=expected_model,
        expected_revision=expected_revision,
        allow_mixed_dirty_state=allow_mixed_dirty_state,
        progress=progress,
        progress_interval=progress_interval,
    )
    if require_matching_run_indices:
        born_indices = {record.run_index for record in born}
        radiative_indices = {record.run_index for record in radiative}
        if born_indices != radiative_indices:
            raise GeneratorProvenanceError(
                "Born and radiative EXCLUGEN run-index sets differ"
            )
    born_revisions = {record.generator_revision for record in born}
    radiative_revisions = {record.generator_revision for record in radiative}
    if not allow_revision_mismatch and born_revisions != radiative_revisions:
        raise GeneratorProvenanceError(
            "Born/radiative EXCLUGEN revision mismatch: "
            f"born={sorted(born_revisions)}, rad={sorted(radiative_revisions)}"
        )
    born_dirty_states = {record.dirty_state for record in born}
    radiative_dirty_states = {record.dirty_state for record in radiative}
    if (
        not allow_mixed_dirty_state
        and born_dirty_states != radiative_dirty_states
    ):
        raise GeneratorProvenanceError(
            "Born/radiative EXCLUGEN dirty-state mismatch: "
            f"born={sorted(born_dirty_states)}, "
            f"rad={sorted(radiative_dirty_states)}"
        )
    born_signature = {record.card_signature for record in born}
    radiative_signature = {record.card_signature for record in radiative}
    if born_signature != radiative_signature:
        raise GeneratorProvenanceError(
            "Born and radiative EXCLUGEN samples do not share one physics card/box"
        )
    for field_name in ("channel", "model"):
        born_values = {getattr(record, field_name) for record in born}
        radiative_values = {getattr(record, field_name) for record in radiative}
        if born_values != radiative_values:
            raise GeneratorProvenanceError(
                f"Born/radiative EXCLUGEN {field_name} mismatch"
            )
    return GeneratorCampaign(production_root=root, runs=born + radiative)


class ExclugenProvider:
    """Generator-provider adapter for EXCLUGEN/exclurad_py productions."""

    provider_name = PROVIDER_NAME

    def discover_campaign(
        self, production_root: str | Path, **kwargs: Any
    ) -> GeneratorCampaign:
        return discover_exclugen_campaign(production_root, **kwargs)
