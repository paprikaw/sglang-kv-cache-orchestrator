from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .sglang_format import (
    canonical_fingerprint,
    canonical_page_path,
    canonical_spec,
    materialization_fingerprint,
    materialization_spec,
    planned_materializations,
)

DEFAULT_STATE_ROOT = Path(
    os.environ.get(
        "SGLANG_KV_STATE_ROOT",
        "~/.local/state/sglang-kv-cache-orchestrator",
    )
).expanduser()
DEFAULT_REGISTRY = Path(
    os.environ.get("SGLANG_KV_REGISTRY", str(DEFAULT_STATE_ROOT / "registry.json"))
).expanduser()
REGISTRY_SCHEMA_VERSION = 2
SOURCE_ROOT = Path(__file__).resolve().parents[1]


# Public compatibility names retained for callers of the original prototype.
compatibility_spec = canonical_spec
compatibility_fingerprint = canonical_fingerprint
planned_locations = planned_materializations


def _empty_registry() -> dict[str, Any]:
    return {"schema_version": REGISTRY_SCHEMA_VERSION, "entries": {}}


@contextlib.contextmanager
def _locked_registry(path: Path) -> Iterator[dict[str, Any]]:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = json.loads(path.read_text()) if path.exists() else _empty_registry()
        version = registry.get("schema_version")
        if version != REGISTRY_SCHEMA_VERSION:
            raise ValueError(
                f"registry {path} uses schema {version}; expected "
                f"{REGISTRY_SCHEMA_VERSION}. Import legacy state explicitly instead of "
                "modifying it in place."
            )
        yield registry
        registry["updated_at"] = time.time()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


def _merge_cache_rows(
    existing: list[dict[str, Any]], additions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = {
        (
            row.get("materialization_fingerprint"),
            row.get("node"),
            row.get("instance_index"),
            row.get("node_rank"),
            row.get("path"),
        ): row
        for row in existing
    }
    for row in additions:
        key = (
            row.get("materialization_fingerprint"),
            row.get("node"),
            row.get("instance_index"),
            row.get("node_rank"),
            row.get("path"),
        )
        rows[key] = row
    return sorted(
        rows.values(),
        key=lambda row: tuple(
            str(value)
            for value in (
                row.get("materialization_fingerprint"),
                row.get("node"),
                row.get("instance_index"),
                row.get("node_rank"),
                row.get("path"),
            )
        ),
    )


def disk_backing_from_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1 or manifest.get("state") != "complete":
        raise ValueError(f"incomplete canonical manifest: {path}")
    return {
        "path": str(path.parent),
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_fingerprint": manifest.get("canonical_fingerprint"),
        "created_at": manifest.get("created_at"),
        "page_count": manifest.get("page_count"),
        "bytes": manifest.get("bytes"),
    }


def validate_disk_backing(
    entry: dict[str, Any], *, verify_files: bool = False
) -> tuple[bool, str, dict[str, Any] | None]:
    backing = entry.get("disk_backing")
    if not backing:
        return False, "canonical_disk_backing_missing", None
    manifest_path = Path(str(backing.get("manifest_path", "")))
    try:
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != backing.get("manifest_sha256"):
            return False, "canonical_manifest_digest_changed", None
        manifest = json.loads(raw)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"canonical_manifest_unreadable: {exc}", None
    if manifest.get("schema_version") != 1 or manifest.get("state") != "complete":
        return False, "canonical_manifest_incomplete", manifest
    fingerprint = entry.get("canonical_fingerprint")
    if manifest.get("canonical_fingerprint") != fingerprint:
        return False, "canonical_fingerprint_mismatch", manifest
    if manifest.get("canonical_format") != entry.get("compatibility"):
        return False, "canonical_format_mismatch", manifest
    if backing.get("canonical_fingerprint") not in {None, fingerprint}:
        return False, "disk_backing_fingerprint_mismatch", manifest
    if not (manifest_path.parent / "pages").is_dir():
        return False, "canonical_pages_missing", manifest
    if verify_files:
        expected_size = int(manifest["canonical_page_bytes"])
        keys = {
            key
            for prefix in manifest.get("prefixes", [])
            for key in prefix.get("page_keys", [])
        }
        if len(keys) != int(manifest.get("page_count", -1)):
            return False, "canonical_page_count_mismatch", manifest
        if int(manifest.get("bytes", -1)) != len(keys) * expected_size:
            return False, "canonical_byte_count_mismatch", manifest
        for key in keys:
            path = canonical_page_path(manifest_path.parent, key)
            try:
                if path.stat().st_size != expected_size:
                    return False, f"canonical_page_invalid: {key}", manifest
            except FileNotFoundError:
                return False, f"canonical_page_missing: {key}", manifest
    return True, "canonical_disk_backing_ready", manifest


def probe_materialization(
    config: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    """Inspect one node cache through SSH without requiring a daemon."""
    code = (
        "import json,sys;"
        "from kv_cache_orchestrator.sglang_format import inspect_local_materialization;"
        "c=json.load(sys.stdin);"
        "print(json.dumps(inspect_local_materialization("
        "sys.argv[3],c,int(sys.argv[1]),int(sys.argv[2]))))"
    )
    argv = [
        "env",
        f"PYTHONPATH={SOURCE_ROOT}",
        "python3",
        "-c",
        code,
        str(target["instance_index"]),
        str(target["node_rank"]),
        str(target["path"]),
    ]
    command = " ".join(shlex.quote(value) for value in argv)
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                str(target["node"]),
                command,
            ],
            input=json.dumps(config, separators=(",", ":")),
            text=True,
            capture_output=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {**target, "valid": False, "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode:
        return {
            **target,
            "valid": False,
            "error": proc.stderr.strip() or proc.stdout.strip(),
        }
    try:
        return {**target, **json.loads(proc.stdout)}
    except json.JSONDecodeError as exc:
        return {**target, "valid": False, "error": f"invalid worker JSON: {exc}"}


Probe = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _new_entry(config: dict[str, Any], now: float) -> dict[str, Any]:
    fingerprint = canonical_fingerprint(config)
    return {
        "fingerprint": fingerprint,
        "canonical_fingerprint": fingerprint,
        "state": "local_only",
        "created_at": now,
        "compatibility": canonical_spec(config),
        "disk_backing": None,
        "materializations": {},
        "checkpoint_ids": [],
    }


def _materialization_entry(
    entry: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    fingerprint = materialization_fingerprint(config)
    return entry.setdefault("materializations", {}).setdefault(
        fingerprint,
        {
            "fingerprint": fingerprint,
            "spec": materialization_spec(config),
            "local_caches": [],
        },
    )


def lookup_checkpoint(
    config: dict[str, Any],
    registry_path: Path = DEFAULT_REGISTRY,
    *,
    verify_live: bool = True,
    verify_disk_files: bool = False,
    probe: Probe = probe_materialization,
) -> dict[str, Any]:
    fingerprint = canonical_fingerprint(config)
    with _locked_registry(Path(registry_path)) as registry:
        entry = registry["entries"].get(fingerprint)
        if entry is None:
            if not verify_live:
                return {
                    "hit": False,
                    "reason": "not_registered",
                    "fingerprint": fingerprint,
                }
            statuses = [
                probe(config, target) for target in planned_materializations(config)
            ]
            local_ready = bool(statuses) and all(row.get("valid") for row in statuses)
            if not local_ready:
                return {
                    "hit": False,
                    "reason": "not_registered",
                    "fingerprint": fingerprint,
                    "local_ready": False,
                    "promotable": False,
                    "local_caches": statuses,
                }
            now = time.time()
            entry = _new_entry(config, now)
            registry["entries"][fingerprint] = entry
            materialized = _materialization_entry(entry, config)
            materialized["local_caches"] = _merge_cache_rows(
                [], [{**row, "last_verified_at": now} for row in statuses]
            )
            materialized["last_verified_at"] = now
            materialized["last_observation"] = statuses
            return {
                "hit": False,
                "reason": "canonical_disk_backing_missing",
                "fingerprint": fingerprint,
                "materialization_fingerprint": materialization_fingerprint(config),
                "local_ready": True,
                "promotable": True,
                "entry": entry,
                "local_caches": statuses,
            }
        if entry.get("state") == "invalid":
            return {
                "hit": False,
                "reason": "entry_invalid",
                "fingerprint": fingerprint,
                "entry": entry,
            }
        disk_valid, disk_reason, manifest = validate_disk_backing(
            entry, verify_files=verify_disk_files
        )
        materialized = _materialization_entry(entry, config)
        materialization_id = materialization_fingerprint(config)
        if not verify_live:
            return {
                "hit": disk_valid,
                "reason": disk_reason,
                "fingerprint": fingerprint,
                "materialization_fingerprint": materialization_id,
                "local_ready": None,
                "entry": entry,
                "disk_manifest": manifest,
            }

        statuses = [
            probe(config, target) for target in planned_materializations(config)
        ]
        now = time.time()
        valid_rows = [
            {**row, "last_verified_at": now} for row in statuses if row.get("valid")
        ]
        if valid_rows:
            materialized["local_caches"] = _merge_cache_rows(
                materialized.get("local_caches", []), valid_rows
            )
        materialized["last_verified_at"] = now
        materialized["last_observation"] = statuses
        local_ready = bool(statuses) and all(row.get("valid") for row in statuses)
        if not disk_valid:
            return {
                "hit": False,
                "reason": disk_reason,
                "fingerprint": fingerprint,
                "materialization_fingerprint": materialization_id,
                "local_ready": local_ready,
                "promotable": local_ready,
                "entry": entry,
                "local_caches": statuses,
            }
        return {
            "hit": True,
            "reason": (
                "canonical_and_materialization_ready"
                if local_ready
                else "canonical_ready_materialization_required"
            ),
            "fingerprint": fingerprint,
            "materialization_fingerprint": materialization_id,
            "local_ready": local_ready,
            "entry": entry,
            "disk_manifest": manifest,
            "local_caches": statuses,
            "missing_local_nodes": [
                row["node"] for row in statuses if not row.get("valid")
            ],
        }


def _inventory_rows(
    config: dict[str, Any], inventory: dict[str, Any]
) -> list[dict[str, Any]]:
    observed = {
        (
            row.get("node"),
            int(row.get("instance_index", -1)),
            int(row.get("node_rank", -1)),
        ): row
        for row in inventory.get("nodes", [])
    }
    rows = []
    for target in planned_materializations(config):
        key = (target["node"], target["instance_index"], target["node_rank"])
        row = observed.get(key)
        if row is None or not row.get("valid"):
            raise ValueError(
                f"materialization inventory is incomplete for {key}: {row}"
            )
        rows.append({**target, **row, "last_verified_at": time.time()})
    return rows


def register_checkpoint(
    config: dict[str, Any],
    inventory: dict[str, Any],
    run_dir: str | Path,
    registry_path: Path = DEFAULT_REGISTRY,
    *,
    disk_backing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fingerprint = canonical_fingerprint(config)
    additions = _inventory_rows(config, inventory)
    now = time.time()
    with _locked_registry(Path(registry_path)) as registry:
        entry = registry["entries"].setdefault(fingerprint, _new_entry(config, now))
        materialized = _materialization_entry(entry, config)
        materialized["local_caches"] = _merge_cache_rows(
            materialized.get("local_caches", []), additions
        )
        materialized["last_verified_at"] = now
        materialized["run_dir"] = str(Path(run_dir).resolve())
        if disk_backing is not None:
            entry["disk_backing"] = disk_backing
            valid, reason, _manifest = validate_disk_backing(entry)
            if not valid:
                raise ValueError(f"invalid canonical disk backing: {reason}")
            entry["state"] = "complete"
        checkpoint_ids = set(entry.get("checkpoint_ids", []))
        checkpoint_ids.add(str(config["checkpoint_id"]))
        entry["checkpoint_ids"] = sorted(checkpoint_ids)
        entry["last_verified_at"] = now
        entry["last_run_dir"] = str(Path(run_dir).resolve())
        return entry


def register_disk_checkpoint(
    config: dict[str, Any],
    disk_backing: dict[str, Any],
    run_dir: str | Path,
    registry_path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    """Register an already-complete Disk artifact without requiring a live cache."""
    fingerprint = canonical_fingerprint(config)
    now = time.time()
    with _locked_registry(Path(registry_path)) as registry:
        entry = registry["entries"].setdefault(fingerprint, _new_entry(config, now))
        entry["disk_backing"] = disk_backing
        valid, reason, _manifest = validate_disk_backing(entry)
        if not valid:
            raise ValueError(f"invalid canonical disk backing: {reason}")
        entry["state"] = "complete"
        checkpoint_ids = set(entry.get("checkpoint_ids", []))
        checkpoint_ids.add(str(config["checkpoint_id"]))
        entry["checkpoint_ids"] = sorted(checkpoint_ids)
        entry["last_verified_at"] = now
        entry["last_run_dir"] = str(Path(run_dir).resolve())
        return entry


def record_local_caches(
    config: dict[str, Any],
    inventory: dict[str, Any],
    registry_path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    return register_checkpoint(
        config,
        inventory,
        Path(inventory.get("run_dir", ".")),
        registry_path,
    )


def local_cache_records(
    config: dict[str, Any], registry_path: Path = DEFAULT_REGISTRY
) -> list[dict[str, Any]]:
    fingerprint = canonical_fingerprint(config)
    with _locked_registry(Path(registry_path)) as registry:
        entry = registry["entries"].get(fingerprint)
        if entry is None:
            return []
        materialized = entry.get("materializations", {}).get(
            materialization_fingerprint(config)
        )
        return list(materialized.get("local_caches", [])) if materialized else []


def list_checkpoints(registry_path: Path = DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    with _locked_registry(Path(registry_path)) as registry:
        return sorted(
            registry["entries"].values(),
            key=lambda row: float(row.get("created_at", 0)),
            reverse=True,
        )


def invalidate_checkpoint(
    fingerprint: str,
    reason: str,
    registry_path: Path = DEFAULT_REGISTRY,
) -> bool:
    with _locked_registry(Path(registry_path)) as registry:
        entry = registry["entries"].get(fingerprint)
        if entry is None:
            return False
        entry["state"] = "invalid"
        entry["invalid_reason"] = reason
        entry["invalidated_at"] = time.time()
        return True
