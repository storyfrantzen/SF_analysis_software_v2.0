#!/usr/bin/env python3
"""Build and validate a durable catalog for an analysis campaign directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA = "sf-analysis-campaign-manifest-v1"
ENVIRONMENT_SCHEMA = "sf-analysis-campaign-environment-v1"
DEFAULT_MANIFEST = Path("01_provenance/campaign_manifest.json")
DEFAULT_MARKDOWN = Path("01_provenance/campaign_manifest.md")
DEFAULT_RESTORE_SCRIPT = Path(
    "01_provenance/environment/restore_campaign_environment.tcsh"
)

DEFAULT_ENVIRONMENT_VARIABLES = (
    "PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "PYTHONPATH",
    "CMAKE_PREFIX_PATH",
    "ROOTSYS",
    "CLAS12ROOT",
    "HIPO",
    "CCDB_HOME",
    "CCDB_CONNECTION",
    "RCDB_HOME",
    "RCDB_CONNECTION",
    "QADB",
    "JANA_HOME",
    "JANA_PLUGIN_PATH",
    "COATJAVA",
    "JAVA_HOME",
    "CLASSPATH",
    "MODULEPATH",
    "MODULESHOME",
    "LOADEDMODULES",
    "_LMFILES_",
)
DEFAULT_ENVIRONMENT_PREFIXES = (
    "CLAS12_",
    "HIPO_",
    "CCDB_",
    "RCDB_",
    "QADB_",
    "ROOT_",
    "JANA_",
    "COATJAVA_",
    "LMOD_",
    "__LMOD_",
)
PATH_LIKE_NAMES = {
    "PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "PYTHONPATH",
    "CMAKE_PREFIX_PATH",
    "ROOTSYS",
    "CLAS12ROOT",
    "HIPO",
    "CCDB_HOME",
    "RCDB_HOME",
    "QADB",
    "JANA_HOME",
    "JANA_PLUGIN_PATH",
    "COATJAVA",
    "JAVA_HOME",
    "MODULEPATH",
    "_LMFILES_",
}
CURATED_ARTIFACT_FIELDS = (
    "description",
    "role",
    "tags",
    "notes",
)
VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SENSITIVE_NAME = re.compile(
    r"(?:PASSWORD|PASSWD|TOKEN|SECRET|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY|API_KEY)",
    re.IGNORECASE,
)
PASSWORD_URI = re.compile(r"://[^/@:]+:[^/@]+@")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def mtime_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_variable_name(name: str) -> str:
    if not VARIABLE_NAME.fullmatch(name):
        raise ValueError(f"invalid shell variable name: {name!r}")
    if SENSITIVE_NAME.search(name):
        raise ValueError(f"refusing to record sensitive-looking variable: {name}")
    return name


def parse_assignment(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"expected NAME=VALUE, got: {raw!r}")
    name, value = raw.split("=", 1)
    return validate_variable_name(name), value


def looks_sensitive_value(value: str) -> bool:
    return PASSWORD_URI.search(value) is not None


def variable_kind(name: str, explicit_path: bool = False) -> str:
    if explicit_path or name in PATH_LIKE_NAMES or name.endswith(("_PATH", "_DIR")):
        return "path"
    return "value"


def capture_environment(
    environment: Mapping[str, str],
    *,
    names: Iterable[str],
    prefixes: Iterable[str],
    shell_variables: Sequence[tuple[str, str, str]],
    existing: Mapping[str, object] | None = None,
    replace: bool = False,
) -> dict[str, object]:
    if existing and not replace:
        payload = json.loads(json.dumps(existing))
    else:
        payload = {
            "schema": ENVIRONMENT_SCHEMA,
            "source_shell": "tcsh",
            "environment_variables": {},
            "shell_variables": {},
            "omitted_sensitive_variables": [],
        }

    environment_variables = payload.setdefault("environment_variables", {})
    local_variables = payload.setdefault("shell_variables", {})
    omitted = set(payload.setdefault("omitted_sensitive_variables", []))
    if not isinstance(environment_variables, dict) or not isinstance(
        local_variables, dict
    ):
        raise ValueError("environment snapshot contains invalid variable maps")

    selected = set(names)
    selected.update(
        name
        for name in environment
        if any(name.startswith(prefix) for prefix in prefixes)
    )
    for name in sorted(selected):
        try:
            validate_variable_name(name)
        except ValueError:
            omitted.add(name)
            continue
        present = name in environment
        value = environment.get(name, "")
        if present and looks_sensitive_value(value):
            omitted.add(name)
            environment_variables.pop(name, None)
            continue
        environment_variables[name] = {
            "present": present,
            "value": value if present else None,
            "kind": variable_kind(name),
        }

    for name, value, kind in shell_variables:
        validate_variable_name(name)
        if looks_sensitive_value(value):
            raise ValueError(
                f"refusing to record a credential-bearing value for {name}"
            )
        local_variables[name] = {"value": value, "kind": kind}

    payload["schema"] = ENVIRONMENT_SCHEMA
    payload["source_shell"] = "tcsh"
    payload["captured_at_utc"] = utc_now()
    payload["omitted_sensitive_variables"] = sorted(omitted)
    return payload


def tcsh_quote(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("tcsh restoration does not support newline-bearing values")
    # tcsh expands $, backticks, and double quotes inside double-quoted values.
    # Single quotes preserve them, but history expansion still requires \!.
    escaped = value.replace("!", r"\!").replace("'", r"'\''")
    return f"'{escaped}'"


def render_tcsh_environment(environment: Mapping[str, object]) -> str:
    exported = environment.get("environment_variables", {})
    local = environment.get("shell_variables", {})
    if not isinstance(exported, dict) or not isinstance(local, dict):
        raise ValueError("environment snapshot contains invalid variable maps")
    lines = [
        "#!/bin/tcsh -f",
        "# Generated from campaign_manifest.json.",
        "# Source this file; executing it cannot change the parent shell:",
        "#   source /path/to/restore_campaign_environment.tcsh",
        "",
    ]
    for name in sorted(exported):
        validate_variable_name(name)
        record = exported[name]
        if not isinstance(record, dict):
            raise ValueError(f"invalid environment record for {name}")
        if record.get("present"):
            value = record.get("value")
            if not isinstance(value, str):
                raise ValueError(f"invalid environment value for {name}")
            lines.append(f"setenv {name} {tcsh_quote(value)}")
        else:
            lines.append(f"unsetenv {name}")
    if exported and local:
        lines.append("")
    for name in sorted(local):
        validate_variable_name(name)
        record = local[name]
        if not isinstance(record, dict) or not isinstance(record.get("value"), str):
            raise ValueError(f"invalid shell-variable record for {name}")
        lines.append(f"set {name} = {tcsh_quote(record['value'])}")
    lines.append("")
    return "\n".join(lines)


def relative_catalog_path(path: Path, campaign_root: Path) -> str | None:
    try:
        return path.resolve().relative_to(campaign_root.resolve()).as_posix()
    except ValueError:
        return None


def file_record(path: Path, relative: str) -> dict[str, object]:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode):
        return {
            "path": relative,
            "type": "symlink",
            "link_target": os.readlink(path),
            "size_bytes": details.st_size,
            "modified_utc": mtime_utc(details.st_mtime),
            "mtime_ns": details.st_mtime_ns,
            "sha256": None,
        }
    return {
        "path": relative,
        "type": "file",
        "size_bytes": details.st_size,
        "modified_utc": mtime_utc(details.st_mtime),
        "mtime_ns": details.st_mtime_ns,
        "sha256": None,
    }


def discover_artifacts(
    campaign_root: Path, excluded_paths: set[str]
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for directory, directory_names, file_names in os.walk(
        campaign_root, followlinks=False
    ):
        directory_path = Path(directory)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            path = directory_path / name
            relative = path.relative_to(campaign_root).as_posix()
            if path.is_symlink():
                if relative not in excluded_paths:
                    artifacts.append(file_record(path, relative))
            else:
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            path = directory_path / name
            relative = path.relative_to(campaign_root).as_posix()
            if relative in excluded_paths:
                continue
            artifacts.append(file_record(path, relative))
    return sorted(artifacts, key=lambda item: str(item["path"]))


def metadata_matches(
    old: Mapping[str, object], current: Mapping[str, object]
) -> bool:
    if old.get("type") != current.get("type"):
        return False
    if current.get("type") == "symlink":
        return old.get("link_target") == current.get("link_target")
    return (
        old.get("size_bytes") == current.get("size_bytes")
        and old.get("mtime_ns") == current.get("mtime_ns")
    )


def hash_current_records(
    campaign_root: Path,
    records: Sequence[dict[str, object]],
    old_by_path: Mapping[str, Mapping[str, object]],
    checksum_mode: str,
) -> None:
    for record in records:
        if record["type"] != "file":
            continue
        path = str(record["path"])
        old = old_by_path.get(path)
        unchanged = old is not None and metadata_matches(old, record)
        if checksum_mode == "all" or (checksum_mode == "changed" and not unchanged):
            record["sha256"] = sha256_file(campaign_root / path)
        elif unchanged:
            record["sha256"] = old.get("sha256")


def curated_fields(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "description": record.get("description", ""),
        "role": record.get("role", ""),
        "tags": list(record.get("tags", []))
        if isinstance(record.get("tags", []), list)
        else [],
        "notes": record.get("notes", ""),
    }


def finalize_present_record(
    current: dict[str, object],
    *,
    now: str,
    old: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result = dict(current)
    if old is not None:
        result.update(curated_fields(old))
        result["first_seen_utc"] = old.get("first_seen_utc", now)
        previous = old.get("previous_paths", [])
        result["previous_paths"] = list(previous) if isinstance(previous, list) else []
    else:
        result.update({field: [] if field == "tags" else "" for field in CURATED_ARTIFACT_FIELDS})
        result["first_seen_utc"] = now
        result["previous_paths"] = []
    result["last_seen_utc"] = now
    result["status"] = "present"
    description = result.get("description")
    result["description_required"] = not (
        isinstance(description, str) and description.strip()
    )
    result.pop("missing_since_utc", None)
    return result


def reconcile_artifacts(
    current: list[dict[str, object]],
    existing: Sequence[Mapping[str, object]],
    *,
    now: str,
) -> list[dict[str, object]]:
    old_by_path = {
        str(record["path"]): record for record in existing if "path" in record
    }
    current_by_path = {str(record["path"]): record for record in current}
    reconciled: list[dict[str, object]] = []

    matched_paths = set(old_by_path).intersection(current_by_path)
    for path in sorted(matched_paths):
        reconciled.append(
            finalize_present_record(current_by_path[path], now=now, old=old_by_path[path])
        )

    missing = {
        path: record for path, record in old_by_path.items() if path not in matched_paths
    }
    new = {
        path: record for path, record in current_by_path.items() if path not in matched_paths
    }

    missing_by_hash: dict[tuple[object, object], list[str]] = {}
    for path, record in missing.items():
        checksum = record.get("sha256")
        if checksum:
            missing_by_hash.setdefault((record.get("type"), checksum), []).append(path)

    renamed_old_paths: set[str] = set()
    renamed_new_paths: set[str] = set()
    for path, record in sorted(new.items()):
        checksum = record.get("sha256")
        candidates = missing_by_hash.get((record.get("type"), checksum), []) if checksum else []
        candidates = [candidate for candidate in candidates if candidate not in renamed_old_paths]
        if len(candidates) != 1:
            continue
        old_path = candidates[0]
        old = missing[old_path]
        merged = finalize_present_record(record, now=now, old=old)
        previous = list(merged.get("previous_paths", []))
        if old_path not in previous:
            previous.append(old_path)
        merged["previous_paths"] = previous
        reconciled.append(merged)
        renamed_old_paths.add(old_path)
        renamed_new_paths.add(path)

    for path, record in sorted(new.items()):
        if path not in renamed_new_paths:
            reconciled.append(finalize_present_record(record, now=now))

    for path, old in sorted(missing.items()):
        if path in renamed_old_paths:
            continue
        record = dict(old)
        record["status"] = "missing"
        record.setdefault("missing_since_utc", now)
        reconciled.append(record)

    return sorted(reconciled, key=lambda item: str(item["path"]))


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def render_markdown(manifest: Mapping[str, object]) -> str:
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("manifest artifacts must be a list")
    present = [item for item in artifacts if item.get("status") == "present"]
    missing = [item for item in artifacts if item.get("status") == "missing"]
    undescribed = [item for item in present if item.get("description_required")]
    environment = manifest.get("environment", {})
    exported = environment.get("environment_variables", {}) if isinstance(environment, dict) else {}
    local = environment.get("shell_variables", {}) if isinstance(environment, dict) else {}
    catalog = manifest.get("catalog_files", {})
    restore_script = (
        catalog.get("restore_script", DEFAULT_RESTORE_SCRIPT.as_posix())
        if isinstance(catalog, dict)
        else DEFAULT_RESTORE_SCRIPT.as_posix()
    )
    lines = [
        "# Campaign artifact manifest",
        "",
        f"- Campaign root: `{markdown_escape(manifest.get('campaign_root', ''))}`",
        f"- Updated: `{markdown_escape(manifest.get('updated_at_utc', ''))}`",
        f"- Present artifacts: {len(present)}",
        f"- Missing historical artifacts: {len(missing)}",
        f"- Present artifacts requiring descriptions: {len(undescribed)}",
        f"- Captured environment variables: {len(exported) if isinstance(exported, dict) else 0}",
        f"- Captured shell variables: {len(local) if isinstance(local, dict) else 0}",
        "",
        "Restore the recorded shell state in `tcsh` with:",
        "",
        "```tcsh",
        f"source {restore_script}",
        "```",
        "",
        "## Artifacts",
        "",
        "| Path | Status | Role | Size (bytes) | SHA-256 | Description |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for artifact in artifacts:
        checksum = artifact.get("sha256") or ""
        lines.append(
            "| `{path}` | {status} | {role} | {size} | `{checksum}` | {description} |".format(
                path=markdown_escape(artifact.get("path", "")),
                status=markdown_escape(artifact.get("status", "")),
                role=markdown_escape(artifact.get("role", "")),
                size=artifact.get("size_bytes", ""),
                checksum=markdown_escape(checksum),
                description=markdown_escape(artifact.get("description", "")),
            )
        )
    lines.extend(["", "Generated from `campaign_manifest.json`; edit descriptions in JSON.", ""])
    return "\n".join(lines)


def write_text_atomic(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        temporary_path = Path(output.name)
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
    if mode is not None:
        temporary_path.chmod(mode)
    os.replace(temporary_path, path)


def load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported campaign manifest schema in {path}")
    if not isinstance(payload.get("artifacts"), list):
        raise ValueError(f"manifest has no artifact list: {path}")
    return payload


def resolve_output(campaign_root: Path, raw_path: Path | None, default: Path) -> Path:
    path = raw_path if raw_path is not None else default
    return path.resolve() if path.is_absolute() else (campaign_root / path).resolve()


def update_manifest(args: argparse.Namespace) -> int:
    campaign_root = args.campaign_root.resolve()
    if not campaign_root.is_dir():
        raise FileNotFoundError(f"campaign root is not a directory: {campaign_root}")
    manifest_path = resolve_output(campaign_root, args.manifest, DEFAULT_MANIFEST)
    markdown_path = resolve_output(campaign_root, args.markdown, DEFAULT_MARKDOWN)
    restore_path = resolve_output(
        campaign_root, args.restore_script, DEFAULT_RESTORE_SCRIPT
    )
    excluded = {
        relative
        for relative in (
            relative_catalog_path(manifest_path, campaign_root),
            relative_catalog_path(markdown_path, campaign_root),
            relative_catalog_path(restore_path, campaign_root),
        )
        if relative is not None
    }

    existing = load_manifest(manifest_path) if manifest_path.exists() else None
    old_artifacts = existing.get("artifacts", []) if existing else []
    if not isinstance(old_artifacts, list):
        raise ValueError("existing artifact catalog is invalid")
    old_by_path = {
        str(record["path"]): record for record in old_artifacts if "path" in record
    }
    current = discover_artifacts(campaign_root, excluded)
    hash_current_records(campaign_root, current, old_by_path, args.checksum)
    now = utc_now()
    artifacts = reconcile_artifacts(current, old_artifacts, now=now)

    explicit_names = [validate_variable_name(name) for name in args.environment_variable]
    explicit_prefixes = list(args.environment_prefix)
    shell_variables = [
        (*parse_assignment(raw), "value") for raw in args.shell_variable
    ]
    shell_variables.extend(
        (*parse_assignment(raw), "path") for raw in args.shell_path
    )
    shell_variables.extend(
        [
            ("campaign_root", str(campaign_root), "path"),
            ("campaign_manifest", str(manifest_path), "path"),
        ]
    )
    existing_environment = existing.get("environment") if existing else None
    should_capture = (
        existing is None
        or args.refresh_environment
        or bool(explicit_names or explicit_prefixes or args.shell_variable or args.shell_path)
    )
    if should_capture:
        refresh_defaults = existing is None or args.refresh_environment
        default_names = DEFAULT_ENVIRONMENT_VARIABLES if refresh_defaults else ()
        default_prefixes = DEFAULT_ENVIRONMENT_PREFIXES if refresh_defaults else ()
        environment = capture_environment(
            os.environ,
            names=(*default_names, *explicit_names),
            prefixes=(*default_prefixes, *explicit_prefixes),
            shell_variables=shell_variables,
            existing=existing_environment if isinstance(existing_environment, dict) else None,
            replace=args.refresh_environment,
        )
    elif isinstance(existing_environment, dict):
        environment = existing_environment
        local = environment.setdefault("shell_variables", {})
        if isinstance(local, dict):
            local["campaign_root"] = {"value": str(campaign_root), "kind": "path"}
            local["campaign_manifest"] = {"value": str(manifest_path), "kind": "path"}
    else:
        raise ValueError("existing manifest has no environment snapshot")

    created = existing.get("created_at_utc", now) if existing else now
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "created_at_utc": created,
        "updated_at_utc": now,
        "campaign_root": str(campaign_root),
        "catalog_files": {
            "json": relative_catalog_path(manifest_path, campaign_root)
            or str(manifest_path),
            "markdown": relative_catalog_path(markdown_path, campaign_root)
            or str(markdown_path),
            "restore_script": relative_catalog_path(restore_path, campaign_root)
            or str(restore_path),
        },
        "environment": environment,
        "artifacts": artifacts,
    }
    write_text_atomic(manifest_path, json.dumps(manifest, indent=2) + "\n")
    write_text_atomic(markdown_path, render_markdown(manifest))
    write_text_atomic(restore_path, render_tcsh_environment(environment), mode=0o755)

    present = sum(item["status"] == "present" for item in artifacts)
    missing = sum(item["status"] == "missing" for item in artifacts)
    undescribed = sum(
        item["status"] == "present" and item["description_required"]
        for item in artifacts
    )
    print(f"Wrote {manifest_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {restore_path}")
    print(
        f"Artifacts: {present} present, {missing} missing, "
        f"{undescribed} requiring descriptions"
    )
    omitted = environment.get("omitted_sensitive_variables", [])
    if omitted:
        print("Omitted sensitive variables: " + ", ".join(omitted))
    return 0


def check_manifest(args: argparse.Namespace) -> int:
    campaign_root = args.campaign_root.resolve()
    manifest_path = resolve_output(campaign_root, args.manifest, DEFAULT_MANIFEST)
    manifest = load_manifest(manifest_path)
    catalog = manifest.get("catalog_files", {})
    excluded: set[str] = set()
    if isinstance(catalog, dict):
        for raw in catalog.values():
            if isinstance(raw, str):
                path = Path(raw)
                resolved = path if path.is_absolute() else campaign_root / path
                relative = relative_catalog_path(resolved, campaign_root)
                if relative is not None:
                    excluded.add(relative)
    current = {str(item["path"]): item for item in discover_artifacts(campaign_root, excluded)}
    records = manifest["artifacts"]
    present = {
        str(item["path"]): item for item in records if item.get("status") == "present"
    }
    historical_missing = {
        str(item["path"]): item for item in records if item.get("status") == "missing"
    }
    problems: list[str] = []
    if not args.allow_missing:
        for path in sorted(set(historical_missing) - set(current)):
            problems.append(f"retained missing artifact: {path}")
    for path in sorted(set(current) - set(present)):
        qualifier = "returned" if path in historical_missing else "new"
        problems.append(f"{qualifier} artifact: {path}")
    for path in sorted(set(present) - set(current)):
        problems.append(f"missing artifact: {path}")
    for path in sorted(set(current).intersection(present)):
        if not metadata_matches(present[path], current[path]):
            problems.append(f"changed artifact: {path}")
            continue
        checksum = present[path].get("sha256")
        if args.verify_checksums and checksum and current[path]["type"] == "file":
            if sha256_file(campaign_root / path) != checksum:
                problems.append(f"checksum mismatch: {path}")
        if present[path].get("description_required") or not str(
            present[path].get("description", "")
        ).strip():
            problems.append(f"description required: {path}")

    if problems:
        print(f"Campaign manifest check failed with {len(problems)} problem(s):")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(
        f"Campaign manifest check passed: {len(present)} present artifacts, "
        f"{len(historical_missing)} retained missing records"
    )
    return 0


def render_environment(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest.resolve())
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("manifest has no environment snapshot")
    output = args.output
    if output is None:
        campaign_root = Path(str(manifest["campaign_root"]))
        catalog = manifest.get("catalog_files", {})
        raw = (
            catalog.get("restore_script", DEFAULT_RESTORE_SCRIPT.as_posix())
            if isinstance(catalog, dict)
            else DEFAULT_RESTORE_SCRIPT.as_posix()
        )
        output = Path(str(raw))
        if not output.is_absolute():
            output = campaign_root / output
    write_text_atomic(output.resolve(), render_tcsh_environment(environment), mode=0o755)
    print(f"Wrote {output.resolve()}")
    print(f"Restore with: source {output.resolve()}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Catalog campaign artifacts and preserve a restorable, deliberately "
            "scoped shell environment."
        )
    )
    commands = root.add_subparsers(dest="command", required=True)

    update = commands.add_parser("update", help="create or refresh a campaign catalog")
    update.add_argument("campaign_root", type=Path)
    update.add_argument("--manifest", type=Path)
    update.add_argument("--markdown", type=Path)
    update.add_argument("--restore-script", type=Path)
    update.add_argument(
        "--checksum",
        choices=("none", "changed", "all"),
        default="none",
        help=(
            "none preserves unchanged hashes; changed hashes new/changed files; "
            "all hashes every regular file"
        ),
    )
    update.add_argument(
        "--refresh-environment",
        action="store_true",
        help="replace the saved environment snapshot with the current one",
    )
    update.add_argument(
        "--environment-variable",
        action="append",
        default=[],
        metavar="NAME",
        help="capture an additional exported variable; may be repeated",
    )
    update.add_argument(
        "--environment-prefix",
        action="append",
        default=[],
        metavar="PREFIX",
        help="capture exported variables beginning with PREFIX; may be repeated",
    )
    update.add_argument(
        "--shell-variable",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="record a tcsh local variable; may be repeated",
    )
    update.add_argument(
        "--shell-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="record a path-valued tcsh local variable; may be repeated",
    )
    update.set_defaults(function=update_manifest)

    check = commands.add_parser("check", help="check the filesystem against a catalog")
    check.add_argument("campaign_root", type=Path)
    check.add_argument("--manifest", type=Path)
    check.add_argument("--verify-checksums", action="store_true")
    check.add_argument(
        "--allow-missing",
        action="store_true",
        help="allow retained historical records whose files remain absent",
    )
    check.set_defaults(function=check_manifest)

    restore = commands.add_parser(
        "render-environment", help="regenerate the tcsh restoration file"
    )
    restore.add_argument("manifest", type=Path)
    restore.add_argument("--output", type=Path)
    restore.set_defaults(function=render_environment)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.function(args))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
