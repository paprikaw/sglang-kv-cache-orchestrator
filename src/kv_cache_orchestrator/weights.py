from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .sglang_format import LOCAL_CACHE_ROOT, json_fingerprint

WEIGHT_CACHE_FORMAT_VERSION = 1
DEFAULT_WEIGHT_CACHE_ROOT = Path(
    os.environ.get("SGLANG_WEIGHT_CACHE_ROOT", "/dev/shm/sglang-weight-cache")
).resolve()
IGNORED_SOURCE_DIRECTORIES = {".cache", ".git"}


def weight_cache_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("weight_cache", {}).get("enabled", False))


def weight_identity(config: dict[str, Any]) -> dict[str, Any]:
    common = config["common"]
    required = (
        "served_model_name",
        "model_revision",
        "model_config_sha256",
        "model_index_sha256",
        "dtype",
    )
    missing = [name for name in required if not common.get(name)]
    if missing:
        raise ValueError(f"weight identity is missing required fields: {missing}")
    return {
        "weight_cache_format_version": WEIGHT_CACHE_FORMAT_VERSION,
        "model": {
            "served_model_name": common["served_model_name"],
            "revision": common["model_revision"],
            "config_sha256": common["model_config_sha256"],
            "index_sha256": common["model_index_sha256"],
            "dtype": common["dtype"],
            "quantization": common.get("quantization"),
        },
    }


def weight_fingerprint(config: dict[str, Any]) -> str:
    """Return a model identity that is independent of TP/PP and node names."""
    return json_fingerprint(weight_identity(config))


def weight_cache_root(config: dict[str, Any]) -> Path:
    return Path(
        config.get("weight_cache", {}).get("local_root", DEFAULT_WEIGHT_CACHE_ROOT)
    ).resolve()


def weight_artifact_path(config: dict[str, Any]) -> Path:
    return weight_cache_root(config) / weight_fingerprint(config)


def runtime_model_path(config: dict[str, Any]) -> Path:
    return weight_artifact_path(config) / "model"


def source_model_path(config: dict[str, Any]) -> Path:
    return Path(config["common"]["model"]).resolve()


def _ensure_local_path(path: Path) -> None:
    allowed = LOCAL_CACHE_ROOT.resolve()
    resolved = path.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"weight cache path escapes local RAM root: {resolved}")


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text().strip() if path.is_file() else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_identity_files(config: dict[str, Any], root: Path) -> None:
    common = config["common"]
    for filename, key in (
        ("config.json", "model_config_sha256"),
        ("model.safetensors.index.json", "model_index_sha256"),
    ):
        path = root / filename
        if not path.is_file():
            raise ValueError(f"model identity file is unavailable: {path}")
        actual = _sha256(path)
        expected = str(common[key])
        if actual != expected:
            raise ValueError(
                f"model identity mismatch for {filename}: {actual} != {expected}"
            )


def source_weight_inventory(config: dict[str, Any]) -> dict[str, Any]:
    root = source_model_path(config)
    if not root.is_dir():
        raise ValueError(f"model source is unavailable: {root}")
    _validate_identity_files(config, root)
    files: list[dict[str, Any]] = []
    for current, directory_names, file_names in os.walk(root):
        current_path = Path(current)
        kept_directories = []
        for name in sorted(directory_names):
            child = current_path / name
            relative = child.relative_to(root)
            if relative.parts[0] in IGNORED_SOURCE_DIRECTORIES:
                continue
            if child.is_symlink():
                raise ValueError(f"model source contains a directory symlink: {child}")
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(root)
            if relative.parts[0] in IGNORED_SOURCE_DIRECTORIES:
                continue
            if path.is_symlink():
                raise ValueError(f"model source contains a file symlink: {path}")
            if not path.is_file():
                raise ValueError(f"model source contains a non-regular file: {path}")
            files.append({"path": relative.as_posix(), "bytes": path.stat().st_size})
    files.sort(key=lambda row: row["path"])
    if not files:
        raise ValueError(f"model source contains no regular files: {root}")
    return {
        "source_model_path": str(root),
        "file_count": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
        "files": files,
    }


def inspect_weight_cache(
    config: dict[str, Any], artifact: str | Path | None = None
) -> dict[str, Any]:
    artifact_path = Path(artifact or weight_artifact_path(config)).resolve()
    _ensure_local_path(artifact_path)
    result: dict[str, Any] = {
        "valid": False,
        "artifact_path": str(artifact_path),
        "runtime_model_path": str(artifact_path / "model"),
        "weight_fingerprint": weight_fingerprint(config),
        "matched_file_count": 0,
        "matched_bytes": 0,
    }
    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.is_file():
        return {**result, "reason": "manifest_missing"}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {**result, "reason": f"manifest_invalid: {exc}"}
    if (
        manifest.get("schema_version") != WEIGHT_CACHE_FORMAT_VERSION
        or manifest.get("state") != "complete"
    ):
        return {**result, "reason": "manifest_incomplete", "manifest": manifest}
    if manifest.get("weight_fingerprint") != weight_fingerprint(config):
        return {**result, "reason": "fingerprint_mismatch", "manifest": manifest}
    if manifest.get("weight_identity") != weight_identity(config):
        return {**result, "reason": "identity_mismatch", "manifest": manifest}
    if manifest.get("boot_id") != _boot_id():
        return {**result, "reason": "boot_id_mismatch", "manifest": manifest}
    model_root = artifact_path / "model"
    matched_files = 0
    matched_bytes = 0
    for row in manifest.get("files", []):
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            return {**result, "reason": "unsafe_manifest_path", "manifest": manifest}
        path = model_root / relative
        expected = int(row.get("bytes", -1))
        if not path.is_file():
            return {
                **result,
                "reason": f"file_missing:{relative.as_posix()}",
                "manifest": manifest,
                "matched_file_count": matched_files,
                "matched_bytes": matched_bytes,
            }
        actual = path.stat().st_size
        if actual != expected:
            return {
                **result,
                "reason": f"file_size_mismatch:{relative.as_posix()}",
                "manifest": manifest,
                "matched_file_count": matched_files,
                "matched_bytes": matched_bytes,
            }
        matched_files += 1
        matched_bytes += actual
    if matched_files != int(manifest.get("file_count", -1)):
        return {**result, "reason": "file_count_mismatch", "manifest": manifest}
    if matched_bytes != int(manifest.get("bytes", -1)):
        return {**result, "reason": "byte_count_mismatch", "manifest": manifest}
    try:
        _validate_identity_files(config, model_root)
    except ValueError as exc:
        return {**result, "reason": str(exc), "manifest": manifest}
    return {
        **result,
        "valid": True,
        "reason": "ready",
        "matched_file_count": matched_files,
        "matched_bytes": matched_bytes,
        "manifest": manifest,
    }


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def materialize_weight_cache(
    config: dict[str, Any],
    artifact: str | Path | None = None,
    *,
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Atomically copy the authoritative shared-Disk model into node CPU RAM."""
    artifact_path = Path(artifact or weight_artifact_path(config)).resolve()
    _ensure_local_path(artifact_path)
    current = inspect_weight_cache(config, artifact_path)
    if current["valid"]:
        return {"status": "already_present", "inventory": current}

    source = source_model_path(config)
    source_inventory = source_weight_inventory(config)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_path.parent / f".{artifact_path.name}.weights.lock"
    with _lock(lock_path):
        current = inspect_weight_cache(config, artifact_path)
        if current["valid"]:
            return {"status": "already_present", "inventory": current}
        free = shutil.disk_usage(artifact_path.parent).free
        required = int(source_inventory["bytes"]) + int(reserve_bytes)
        if free < required:
            raise OSError(
                f"insufficient /dev/shm space for weights: need {required}, have {free}"
            )
        token = uuid.uuid4().hex
        staging = artifact_path.parent / f".{artifact_path.name}.incoming-{token}"
        previous = artifact_path.parent / f".{artifact_path.name}.old-{token}"
        model_root = staging / "model"
        started_at = time.time()
        model_root.mkdir(parents=True, exist_ok=False)
        try:
            for row in source_inventory["files"]:
                relative = Path(str(row["path"]))
                source_file = source / relative
                destination = model_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, destination)
                if destination.stat().st_size != int(row["bytes"]):
                    raise RuntimeError(f"copied model file is incomplete: {relative}")
            _validate_identity_files(config, model_root)
            manifest = {
                "schema_version": WEIGHT_CACHE_FORMAT_VERSION,
                "state": "complete",
                "weight_fingerprint": weight_fingerprint(config),
                "weight_identity": weight_identity(config),
                "source_model_path": str(source),
                "created_at": time.time(),
                "copy_elapsed_seconds": time.time() - started_at,
                "boot_id": _boot_id(),
                "file_count": source_inventory["file_count"],
                "bytes": source_inventory["bytes"],
                "files": source_inventory["files"],
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            staged = inspect_weight_cache(config, staging)
            if not staged["valid"]:
                raise RuntimeError(f"staged weight cache failed validation: {staged}")
            if artifact_path.exists():
                artifact_path.replace(previous)
            staging.replace(artifact_path)
            shutil.rmtree(previous, ignore_errors=True)
            final = inspect_weight_cache(config, artifact_path)
            if not final["valid"]:
                raise RuntimeError(f"installed weight cache failed validation: {final}")
            return {
                "status": "materialized",
                "inventory": final,
                "copy_elapsed_seconds": manifest["copy_elapsed_seconds"],
                "finished_at": time.time(),
            }
        except BaseException:
            if previous.exists() and not artifact_path.exists():
                previous.replace(artifact_path)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(previous, ignore_errors=True)
