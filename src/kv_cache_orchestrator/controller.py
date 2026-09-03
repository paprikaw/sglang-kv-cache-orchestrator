from __future__ import annotations

import copy
import fcntl
import itertools
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .registry import (
    DEFAULT_REGISTRY,
    compatible_local_cache_records,
    disk_backing_from_manifest,
    lookup_checkpoint,
    probe_materialization,
    record_local_caches,
    register_checkpoint,
    register_disk_checkpoint,
    validate_disk_backing,
)
from .sglang_format import (
    artifact_path,
    canonical_fingerprint,
    canonical_page_path,
    canonical_spec,
    disk_root,
    materialization_fingerprint,
    model_kv_layout,
    planned_materializations,
    prompt_pages,
)
from .weights import (
    runtime_model_path,
    source_weight_inventory,
    weight_artifact_path,
    weight_cache_enabled,
    weight_fingerprint,
)

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SLURM_BIN = Path("/apps/slurm/latest/bin")
WorkerRunner = Callable[[str, list[str], float], subprocess.CompletedProcess[str]]


def remote_worker(
    node: str, arguments: list[str], timeout: float = 7200.0
) -> subprocess.CompletedProcess[str]:
    configured = os.environ.get("SGLANG_KV_WORKER_COMMAND")
    prefix = (
        shlex.split(configured)
        if configured
        else [
            "env",
            f"PYTHONPATH={SOURCE_ROOT}",
            "python3",
            "-m",
            "kv_cache_orchestrator.worker",
        ]
    )
    command = " ".join(shlex.quote(value) for value in [*prefix, *arguments])
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, command],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def worker_json(
    node: str,
    arguments: list[str],
    *,
    timeout: float = 7200.0,
    runner: WorkerRunner = remote_worker,
) -> dict[str, Any]:
    proc = runner(node, arguments, timeout)
    if proc.returncode:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"cache worker failed on {node}: {detail}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"cache worker returned invalid JSON on {node}: {proc.stdout}"
        ) from exc


@contextmanager
def _artifact_lock(root: Path, fingerprint: str) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f".{fingerprint}.publish.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _config_worker_args(
    command: str, config_path: Path, target: dict[str, Any]
) -> list[str]:
    return [
        command,
        "--config",
        str(config_path),
        "--instance-index",
        str(target["instance_index"]),
        "--node-rank",
        str(target["node_rank"]),
        "--path",
        str(target["path"]),
    ]


def inspect_configured_caches(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    runner: WorkerRunner = remote_worker,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    rows = []
    for target in planned_materializations(config):
        observed = worker_json(
            str(target["node"]),
            _config_worker_args("inspect", path, target),
            timeout=900,
            runner=runner,
        )
        rows.append({**target, **observed})
    return {"captured_at": time.time(), "nodes": rows}


def _existing_backing(config: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = artifact_path(config) / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        backing = disk_backing_from_manifest(manifest_path)
        entry = {
            "canonical_fingerprint": canonical_fingerprint(config),
            "compatibility": canonical_spec(config),
            "disk_backing": backing,
        }
        valid, _reason, _manifest = validate_disk_backing(entry, verify_files=True)
        return backing if valid else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def publish_checkpoint_to_disk(
    config: dict[str, Any],
    config_path: str | Path,
    run_dir: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    runner: WorkerRunner = remote_worker,
) -> dict[str, Any]:
    """Convert one complete TP/PP materialization into canonical shared-disk pages."""
    config_file = Path(config_path).resolve()
    fingerprint = canonical_fingerprint(config)
    root = disk_root(config)
    final = artifact_path(config)
    with _artifact_lock(root, fingerprint):
        existing = _existing_backing(config)
        if existing is not None:
            try:
                inventory = inspect_configured_caches(
                    config, config_file, runner=runner
                )
                entry = register_checkpoint(
                    config,
                    inventory,
                    run_dir,
                    Path(registry_path),
                    disk_backing=existing,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                inventory = {
                    "captured_at": time.time(),
                    "nodes": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                entry = register_disk_checkpoint(
                    config, existing, run_dir, Path(registry_path)
                )
            return {
                "status": "already_published",
                "canonical_fingerprint": fingerprint,
                "disk_backing": existing,
                "source_inventory": inventory,
                "registry_entry": entry,
            }

        staging = root / f".{fingerprint}.incoming-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        rows = []
        scatter = []
        try:
            for target in planned_materializations(config):
                arguments = _config_worker_args("scatter", config_file, target)
                arguments.extend(["--destination", str(staging)])
                result = worker_json(
                    str(target["node"]), arguments, timeout=14400, runner=runner
                )
                if int(result["written_rank_pieces"]) != int(
                    target["expected_file_count"]
                ):
                    raise RuntimeError(
                        f"worker {target['node']} wrote "
                        f"{result['written_rank_pieces']} rank pieces; expected "
                        f"{target['expected_file_count']}"
                    )
                rows.append({**target, **result["source"]})
                scatter.append(
                    {
                        "node": target["node"],
                        "instance_id": target["instance_id"],
                        "node_rank": target["node_rank"],
                        "page_count": result["page_count"],
                        "written_rank_pieces": result["written_rank_pieces"],
                    }
                )

            layout = model_kv_layout(config)
            canonical_page_bytes = (
                2
                * int(layout["page_size"])
                * int(layout["num_layers"])
                * int(layout["num_kv_heads"])
                * int(layout["head_dim"])
                * int(layout["item_size"])
            )
            prefixes = prompt_pages(config)
            unique_keys = {key for prefix in prefixes for key in prefix["page_keys"]}
            for key in unique_keys:
                path = canonical_page_path(staging, key)
                if not path.is_file() or path.stat().st_size != canonical_page_bytes:
                    raise RuntimeError(f"canonical page is incomplete: {key}")
            manifest = {
                "schema_version": 1,
                "state": "complete",
                "canonical_fingerprint": fingerprint,
                "canonical_format": canonical_spec(config),
                "created_at": time.time(),
                "producing_topology": config["topology"],
                "producing_materialization_fingerprint": materialization_fingerprint(
                    config
                ),
                "producing_run_dir": str(Path(run_dir).resolve()),
                "canonical_page_bytes": canonical_page_bytes,
                "page_count": len(unique_keys),
                "bytes": len(unique_keys) * canonical_page_bytes,
                "prefixes": prefixes,
                "scatter": scatter,
            }
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            staging.replace(final)
            backing = disk_backing_from_manifest(final / "manifest.json")
            inventory = {"captured_at": time.time(), "nodes": rows}
            entry = register_checkpoint(
                config,
                inventory,
                run_dir,
                Path(registry_path),
                disk_backing=backing,
            )
            return {
                "status": "published",
                "canonical_fingerprint": fingerprint,
                "disk_backing": backing,
                "manifest": manifest,
                "source_inventory": inventory,
                "registry_entry": entry,
            }
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def parse_byte_size(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)B?\s*", value, re.I)
    if not match:
        raise ValueError(f"invalid byte size: {value!r}")
    power = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4}[match.group(2).upper()]
    return int(float(match.group(1)) * (1024**power))


def hydrate_config_from_disk(
    config: dict[str, Any],
    config_path: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    run_dir: str | Path | None = None,
    allow_busy: bool = False,
    runner: WorkerRunner = remote_worker,
) -> dict[str, Any]:
    """Materialize the requested TP/PP layout from topology-neutral disk pages."""
    registry = Path(registry_path)
    lookup = lookup_checkpoint(config, registry, verify_live=False)
    if not lookup["hit"]:
        raise RuntimeError(f"canonical checkpoint unavailable: {lookup['reason']}")
    manifest_path = lookup["entry"]["disk_backing"]["manifest_path"]
    reserve = parse_byte_size(config["hicache"].get("storage_min_free_space", 0))
    config_file = Path(config_path).resolve()

    node_status = {}
    for node in sorted({str(row["node"]) for row in planned_materializations(config)}):
        status = worker_json(node, ["status"], timeout=180, runner=runner)
        node_status[node] = status
        if status.get("busy") and not allow_busy:
            raise RuntimeError(f"refusing to materialize on busy node {node}")

    rows = []
    transfers = []
    peer_policy = config.get("peer_transfer", {})
    peer_enabled = bool(peer_policy.get("enabled", False))
    fabric_regex = str(
        peer_policy.get("fabric_interface_regex", r"^(?:bond0(?:\.|$)|ib|mlx)")
    )
    fallback_to_disk = bool(peer_policy.get("fallback_to_disk", True))
    for target in planned_materializations(config):
        current = worker_json(
            str(target["node"]),
            _config_worker_args("inspect", config_file, target),
            timeout=900,
            runner=runner,
        )
        peer_attempts = []
        result = None
        if current.get("valid") and int(current.get("unexpected_file_count", 0)):
            arguments = _config_worker_args("materialize", config_file, target)
            arguments.extend(
                [
                    "--manifest",
                    str(manifest_path),
                    "--reserve-bytes",
                    str(reserve),
                ]
            )
            result = worker_json(
                str(target["node"]), arguments, timeout=14_400, runner=runner
            )
        elif current.get("valid"):
            result = {"status": "already_present", "inventory": current}
        elif peer_enabled:
            candidates = compatible_local_cache_records(config, target, registry)
            for source in candidates:
                if str(source.get("node")) == str(target["node"]) and str(
                    source.get("path")
                ) == str(target["path"]):
                    continue
                source_target = {**target, "path": source["path"]}
                attempt = {
                    "source_node": source.get("node"),
                    "source_path": source.get("path"),
                }
                try:
                    observed = worker_json(
                        str(source["node"]),
                        _config_worker_args("inspect", config_file, source_target),
                        timeout=900,
                        runner=runner,
                    )
                    attempt["source_valid"] = bool(observed.get("valid"))
                    if not observed.get("valid"):
                        attempt["status"] = "source_invalid"
                        peer_attempts.append(attempt)
                        continue
                    arguments = _config_worker_args(
                        "peer-materialize", config_file, target
                    )
                    arguments.extend(
                        [
                            "--source-node",
                            str(source["node"]),
                            "--source-path",
                            str(source["path"]),
                            "--fabric-interface-regex",
                            fabric_regex,
                            "--reserve-bytes",
                            str(reserve),
                        ]
                    )
                    result = worker_json(
                        str(target["node"]),
                        arguments,
                        timeout=14400,
                        runner=runner,
                    )
                    attempt["status"] = result["status"]
                    peer_attempts.append(attempt)
                    break
                except (OSError, RuntimeError, ValueError) as exc:
                    attempt["status"] = "failed"
                    attempt["error"] = f"{type(exc).__name__}: {exc}"
                    peer_attempts.append(attempt)
            if result is None and peer_attempts and not fallback_to_disk:
                raise RuntimeError(
                    f"all peer cache transfers failed for {target['node']}: "
                    f"{peer_attempts}"
                )
        if result is None:
            arguments = _config_worker_args("materialize", config_file, target)
            arguments.extend(
                [
                    "--manifest",
                    str(manifest_path),
                    "--reserve-bytes",
                    str(reserve),
                ]
            )
            result = worker_json(
                str(target["node"]), arguments, timeout=14400, runner=runner
            )
        rows.append({**target, **result["inventory"]})
        transfer = {
            "node": target["node"],
            "instance_id": target["instance_id"],
            "node_rank": target["node_rank"],
            "status": result["status"],
            "bytes": result["inventory"]["matched_bytes"],
            "peer_attempts": peer_attempts,
        }
        for key in (
            "source",
            "transport",
            "route",
            "duration_s",
            "throughput_bytes_s",
            "pruned_file_count",
            "pruned_bytes",
            "pruned_sample",
        ):
            if key in result:
                transfer[key] = result[key]
        transfers.append(transfer)
    inventory = {
        "captured_at": time.time(),
        "run_dir": str(Path(run_dir or ".").resolve()),
        "nodes": rows,
    }
    entry = record_local_caches(config, inventory, registry)
    return {
        "status": "ready",
        "canonical_fingerprint": lookup["fingerprint"],
        "materialization_fingerprint": materialization_fingerprint(config),
        "node_status": node_status,
        "transfers": transfers,
        "inventory": inventory,
        "registry_entry": entry,
    }


def parse_slurm_time_left(value: str) -> float | None:
    value = value.strip()
    if value.upper() in {"UNLIMITED", "NOT_SET", "N/A"}:
        return None
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        raise ValueError(f"invalid Slurm time-left value: {value!r}")
    return float((((days * 24) + hours) * 60 + minutes) * 60 + seconds)


def _slurm_command(name: str) -> str:
    candidate = SLURM_BIN / name
    return str(candidate if candidate.is_file() else name)


def _expand_nodes(nodelist: str) -> list[str]:
    if not any(char in nodelist for char in "[, "):
        return [nodelist]
    proc = subprocess.run(
        [_slurm_command("scontrol"), "show", "hostnames", nodelist],
        text=True,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def discover_slurm_candidates(
    config: dict[str, Any],
    *,
    candidate_nodes: set[str] | None = None,
    job_name_regex: str | None = None,
    runner: WorkerRunner = remote_worker,
) -> list[dict[str, Any]]:
    pattern = job_name_regex or config.get("placement", {}).get("slurm_job_name_regex")
    compiled = re.compile(pattern) if pattern else None
    proc = subprocess.run(
        [
            _slurm_command("squeue"),
            "--me",
            "-h",
            "-t",
            "RUNNING",
            "-o",
            "%i|%N|%L|%e|%T|%j",
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    candidates = []
    seen_nodes = set()
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        job_id, nodelist, time_left, end_time, state, job_name = line.split("|", 5)
        if compiled and not compiled.search(job_name):
            continue
        for node in _expand_nodes(nodelist):
            if candidate_nodes and node not in candidate_nodes:
                continue
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            try:
                health = worker_json(node, ["status"], timeout=180, runner=runner)
            except Exception as exc:
                health = {"error": f"{type(exc).__name__}: {exc}", "busy": True}
            try:
                node_ip = socket.gethostbyname(node)
            except socket.gaierror:
                node_ip = node
            candidates.append(
                {
                    "node": node,
                    "node_ip": node_ip,
                    "job_id": job_id,
                    "job_name": job_name,
                    "state": state,
                    "time_left": time_left,
                    "time_left_seconds": parse_slurm_time_left(time_left),
                    "end_time": end_time,
                    "busy": bool(health.get("busy", True)),
                    "health": health,
                    "cached_slots": [],
                }
            )
    return candidates


def _slot(target: dict[str, Any]) -> str:
    return f"{target['instance_index']}:{target['node_rank']}"


def discover_cache_hits(
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    probe: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ] = probe_materialization,
) -> list[dict[str, Any]]:
    # Probe the deterministic target path on every candidate. This discovers
    # caches created before the registry existed and caches surviving a
    # controller restart. The registry is still used for canonical Disk state.
    _ = registry_path
    targets = planned_materializations(config)
    for candidate in candidates:
        candidate.setdefault("cached_slots", [])
        candidate.setdefault("cache_observations", [])
        for target in targets:
            if candidate.get("busy"):
                continue
            if int(candidate.get("health", {}).get("gpu_count") or 0) != int(
                target["required_gpu_count"]
            ):
                continue
            candidate_target = {**target, "node": candidate["node"]}
            try:
                observed = probe(config, candidate_target)
            except Exception as exc:
                observed = {
                    **candidate_target,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            candidate["cache_observations"].append(observed)
            if observed.get("valid"):
                slot = _slot(target)
                if slot not in candidate["cached_slots"]:
                    candidate["cached_slots"].append(slot)
    return candidates


def discover_weight_cache_hits(
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    probe: Callable[[dict[str, Any], str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Probe the topology-independent model-weight cache on every idle node."""
    enabled = weight_cache_enabled(config)
    artifact = str(weight_artifact_path(config)) if enabled else None
    for candidate in candidates:
        candidate.setdefault("weight_cache_hit", False)
        candidate.setdefault("weight_cache_observation", None)
        if not enabled or candidate.get("busy"):
            continue
        try:
            observed = probe(config, str(candidate["node"]))
        except Exception as exc:
            observed = {
                "valid": False,
                "artifact_path": artifact,
                "error": f"{type(exc).__name__}: {exc}",
            }
        candidate["weight_cache_observation"] = observed
        candidate["weight_cache_hit"] = bool(observed.get("valid"))
    return candidates


def choose_placement(
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    min_time_left_seconds: float = 3600.0,
) -> dict[str, Any]:
    targets = planned_materializations(config)
    kv_reserve = parse_byte_size(config["hicache"].get("storage_min_free_space", 0))
    weights_enabled = weight_cache_enabled(config)
    weight_bytes = (
        int(source_weight_inventory(config)["bytes"]) if weights_enabled else 0
    )
    weight_reserve = (
        parse_byte_size(config["weight_cache"].get("storage_min_free_space", 0))
        if weights_enabled
        else 0
    )
    reserve = max(kv_reserve, weight_reserve)
    lookup = lookup_checkpoint(config, Path(registry_path), verify_live=False)
    eligible = []
    rejected = []
    for candidate in candidates:
        reasons = []
        left = candidate.get("time_left_seconds")
        if candidate.get("busy"):
            reasons.append("busy")
        if left is not None and float(left) < min_time_left_seconds:
            reasons.append("booking expires too soon")
        if reasons:
            rejected.append({"node": candidate["node"], "reasons": reasons})
        else:
            eligible.append(candidate)
    if len(eligible) < len(targets):
        summary = (
            f"need {len(targets)} idle booked nodes, found {len(eligible)}; "
            f"rejected={rejected}"
        )
        raise RuntimeError(summary)

    best = None
    for assignment in itertools.permutations(eligible, len(targets)):
        hits = 0
        weight_hits = 0
        lifetime = 0.0
        free_total = 0.0
        feasible = True
        for target, candidate in zip(targets, assignment, strict=True):
            required_gpus = int(target["required_gpu_count"])
            available_gpus = int(candidate.get("health", {}).get("gpu_count") or 0)
            if available_gpus != required_gpus:
                feasible = False
                break
            cached = _slot(target) in candidate.get("cached_slots", [])
            hits += int(cached)
            weight_cached = weights_enabled and bool(
                candidate.get("weight_cache_hit", False)
            )
            weight_hits += int(weight_cached)
            left = candidate.get("time_left_seconds")
            lifetime += 10**12 if left is None else float(left)
            free = float(candidate.get("health", {}).get("shm_free_bytes") or 0)
            free_total += free
            needed = (
                reserve
                + (0 if cached else int(target["expected_bytes"]))
                + (0 if weight_cached else weight_bytes)
            )
            if free < needed:
                feasible = False
                break
        if not feasible:
            continue
        score = (hits, weight_hits, lifetime, free_total)
        if best is None or score > best[0]:
            best = (score, assignment)
    if best is None:
        raise RuntimeError("no placement has enough /dev/shm capacity")

    decisions = []
    for target, candidate in zip(targets, best[1], strict=True):
        cached = _slot(target) in candidate.get("cached_slots", [])
        weight_cached = weights_enabled and bool(
            candidate.get("weight_cache_hit", False)
        )
        action = (
            "reuse_local"
            if cached
            else (
                "materialize_from_canonical_disk"
                if lookup["hit"]
                else "build_from_prefill"
            )
        )
        decisions.append(
            {
                "slot": _slot(target),
                "instance_index": target["instance_index"],
                "instance_id": target["instance_id"],
                "node_rank": target["node_rank"],
                "node": candidate["node"],
                "node_ip": candidate["node_ip"],
                "job_id": candidate["job_id"],
                "local_cache_hit": cached,
                "cache_action": action,
                "weight_cache_hit": weight_cached,
                "weight_cache_action": (
                    "disabled"
                    if not weights_enabled
                    else (
                        "reuse_local"
                        if weight_cached
                        else "materialize_from_shared_disk"
                    )
                ),
                "time_left": candidate["time_left"],
                "end_time": candidate["end_time"],
            }
        )
    return {
        "canonical_fingerprint": canonical_fingerprint(config),
        "materialization_fingerprint": materialization_fingerprint(config),
        "canonical_disk_available": bool(lookup["hit"]),
        "cache_hits": sum(row["local_cache_hit"] for row in decisions),
        "weight_cache_enabled": weights_enabled,
        "weight_fingerprint": (weight_fingerprint(config) if weights_enabled else None),
        "weight_cache_bytes": weight_bytes,
        "weight_cache_hits": sum(row["weight_cache_hit"] for row in decisions),
        "required_nodes": len(targets),
        "decisions": decisions,
        "rejected": rejected,
    }


def worker_probe(
    config_path: str | Path,
    *,
    runner: WorkerRunner = remote_worker,
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    path = Path(config_path).resolve()

    def probe(config: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
        del config
        try:
            observed = worker_json(
                str(target["node"]),
                _config_worker_args("inspect", path, target),
                timeout=900,
                runner=runner,
            )
            return {**target, **observed}
        except (OSError, RuntimeError, ValueError) as exc:
            return {
                **target,
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return probe


def worker_weight_probe(
    config_path: str | Path,
    *,
    runner: WorkerRunner = remote_worker,
) -> Callable[[dict[str, Any], str], dict[str, Any]]:
    path = Path(config_path).resolve()

    def probe(config: dict[str, Any], node: str) -> dict[str, Any]:
        del config
        try:
            return worker_json(
                node,
                ["weight-inspect", "--config", str(path)],
                timeout=900,
                runner=runner,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}

    return probe


def resolve_config(config: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    decisions = {row["slot"]: row for row in plan["decisions"]}
    node_rows = []
    seen = set()
    for instance_index, instance in enumerate(result["instances"]):
        assigned = []
        for node_rank in range(len(instance["nodes"])):
            decision = decisions[f"{instance_index}:{node_rank}"]
            assigned.append(decision["node"])
            if decision["node"] not in seen:
                seen.add(decision["node"])
                node_rows.append(
                    {
                        "node": decision["node"],
                        "node_ip": decision["node_ip"],
                        "job_id": str(decision["job_id"]),
                    }
                )
        instance["nodes"] = assigned
    result["nodes"] = node_rows
    result["client"] = {
        "node": node_rows[0]["node"],
        "job_id": node_rows[0]["job_id"],
    }
    result["placement_resolution"] = {
        "resolved_at": time.time(),
        "policy": "prefer_kv_then_weight_cache_then_lifetime_then_free_ram",
        "plan": plan,
    }
    if weight_cache_enabled(result):
        result["weight_cache"]["fingerprint"] = weight_fingerprint(result)
        result["weight_cache"]["runtime_model_path"] = str(runtime_model_path(result))
    return result


def inspect_configured_weights(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    runner: WorkerRunner = remote_worker,
) -> dict[str, Any]:
    if not weight_cache_enabled(config):
        return {"enabled": False, "nodes": []}
    config_file = Path(config_path).resolve()
    rows = []
    nodes = sorted(
        {str(node) for instance in config["instances"] for node in instance["nodes"]}
    )
    for node in nodes:
        observed = worker_json(
            node,
            ["weight-inspect", "--config", str(config_file)],
            timeout=900,
            runner=runner,
        )
        rows.append({"node": node, **observed})
    return {
        "enabled": True,
        "weight_fingerprint": weight_fingerprint(config),
        "runtime_model_path": str(runtime_model_path(config)),
        "nodes": rows,
    }


def materialize_configured_weights(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    allow_busy: bool = False,
    runner: WorkerRunner = remote_worker,
) -> dict[str, Any]:
    if not weight_cache_enabled(config):
        return {"status": "disabled", "nodes": []}
    config_file = Path(config_path).resolve()
    reserve = parse_byte_size(config["weight_cache"].get("storage_min_free_space", 0))
    nodes = sorted(
        {str(node) for instance in config["instances"] for node in instance["nodes"]}
    )
    statuses = {}
    for node in nodes:
        node_status = worker_json(node, ["status"], timeout=180, runner=runner)
        statuses[node] = node_status
        if node_status.get("busy") and not allow_busy:
            raise RuntimeError(f"refusing to materialize weights on busy node {node}")
    transfers = []
    for node in nodes:
        result = worker_json(
            node,
            [
                "weight-materialize",
                "--config",
                str(config_file),
                "--reserve-bytes",
                str(reserve),
            ],
            timeout=14400,
            runner=runner,
        )
        transfers.append({"node": node, **result})
    inspection = inspect_configured_weights(config, config_file, runner=runner)
    if not all(row.get("valid") for row in inspection["nodes"]):
        raise RuntimeError(f"weight materialization failed validation: {inspection}")
    return {
        "status": "ready",
        "weight_fingerprint": weight_fingerprint(config),
        "runtime_model_path": str(runtime_model_path(config)),
        "node_status": statuses,
        "transfers": transfers,
        "inspection": inspection,
    }


def prepare_cache_aware_config(
    config: dict[str, Any],
    source_config_path: str | Path,
    output_path: str | Path,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    candidate_nodes: set[str] | None = None,
    job_name_regex: str | None = None,
    min_time_left_seconds: float = 3600.0,
    dry_run: bool = False,
    runner: WorkerRunner = remote_worker,
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["orchestrator_registry"] = str(registry_path)
    candidates = discover_slurm_candidates(
        config,
        candidate_nodes=candidate_nodes,
        job_name_regex=job_name_regex,
        runner=runner,
    )
    discover_cache_hits(
        config,
        candidates,
        registry_path,
        probe=worker_probe(source_config_path, runner=runner),
    )
    discover_weight_cache_hits(
        config,
        candidates,
        probe=worker_weight_probe(source_config_path, runner=runner),
    )
    plan = choose_placement(
        config,
        candidates,
        registry_path,
        min_time_left_seconds=min_time_left_seconds,
    )
    resolved = resolve_config(config, plan)
    result = {
        "plan": plan,
        "candidates": candidates,
        "resolved_config": resolved,
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    live_probe = worker_probe(output, runner=runner)
    lookup = lookup_checkpoint(
        resolved,
        Path(registry_path),
        verify_live=True,
        probe=live_probe,
    )
    if not lookup["hit"] and lookup.get("promotable"):
        result["promotion"] = publish_checkpoint_to_disk(
            resolved, output, output.parent, registry_path, runner=runner
        )
        lookup = lookup_checkpoint(
            resolved,
            Path(registry_path),
            verify_live=True,
            probe=live_probe,
        )
    if lookup["hit"] and not lookup.get("local_ready"):
        result["materialization"] = hydrate_config_from_disk(
            resolved,
            output,
            registry_path,
            run_dir=output.parent,
            runner=runner,
        )
    elif not lookup["hit"]:
        result["next_action"] = "run_prefill_then_publish"
    if weight_cache_enabled(resolved):
        result["weight_materialization"] = materialize_configured_weights(
            resolved, output, runner=runner
        )
    return result
