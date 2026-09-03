from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import struct
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import checkpoint_path, endpoint_assignments, prefix_token_sequences

CANONICAL_FORMAT_VERSION = 1
LOCAL_CACHE_ROOT = Path(os.environ.get("SGLANG_KV_LOCAL_ROOT", "/dev/shm")).resolve()
DEFAULT_DISK_ROOT = Path(
    os.environ.get(
        "SGLANG_KV_DISK_ROOT",
        "~/.cache/sglang-kv-cache-orchestrator/canonical",
    )
).expanduser()
DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "half": 2,
    "float32": 4,
}


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def json_fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _prompt_digest(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def page_hashes(tokens: list[int], page_size: int) -> list[str]:
    """Match SGLang's chained SHA-256 page keys for ordinary token IDs."""
    if len(tokens) % page_size:
        raise ValueError("canonical checkpoints require page-aligned prompts")
    prior = b""
    result = []
    for start in range(0, len(tokens), page_size):
        page = tokens[start : start + page_size]
        raw = struct.pack(f"<{len(page)}I", *page)
        prior = hashlib.sha256(prior + raw).digest()
        result.append(prior.hex())
    return result


def prompt_pages(config: dict[str, Any]) -> list[dict[str, Any]]:
    page_size = int(config["common"]["page_size"])
    result = []
    for request_index, tokens in enumerate(prefix_token_sequences(config)):
        result.append(
            {
                "request_index": request_index,
                "input_tokens": len(tokens),
                "prompt_sha256": _prompt_digest(tokens),
                "page_keys": page_hashes(tokens, page_size),
            }
        )
    return result


def model_kv_layout(config: dict[str, Any]) -> dict[str, Any]:
    if config["hicache"].get("mem_layout") != "page_first":
        raise ValueError("only SGLang HiCache mem_layout=page_first is supported")
    model_path = Path(config["common"]["model"])
    model_config_path = model_path / "config.json"
    if not model_config_path.is_file():
        raise ValueError(f"model config is unavailable: {model_config_path}")
    model = json.loads(model_config_path.read_text())
    layers = model.get("num_hidden_layers")
    attention_heads = model.get("num_attention_heads")
    kv_heads = model.get("num_key_value_heads", attention_heads)
    head_dim = model.get("head_dim")
    if head_dim is None and model.get("hidden_size") and attention_heads:
        head_dim = int(model["hidden_size"]) // int(attention_heads)
    dtype = str(config["common"]["dtype"])
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported canonical KV dtype: {dtype}")
    if not all(
        isinstance(value, int) and value > 0 for value in (layers, kv_heads, head_dim)
    ):
        raise ValueError(
            "canonical MHA/GQA layout requires num_hidden_layers, "
            "num_key_value_heads, and head_dim"
        )
    if model.get("kv_lora_rank") or model.get("multi_latent_attention"):
        raise ValueError(
            "MLA canonical conversion is not implemented by this format version"
        )
    return {
        "kind": "mha_gqa",
        "num_layers": int(layers),
        "num_kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "dtype": dtype,
        "item_size": DTYPE_BYTES[dtype],
        "page_size": int(config["common"]["page_size"]),
        "axis_order": [
            "kv",
            "token",
            "layer",
            "kv_head",
            "head_dim",
            "item_byte",
        ],
    }


def canonical_spec(config: dict[str, Any]) -> dict[str, Any]:
    common = config["common"]
    required = (
        "source_commit",
        "model_revision",
        "model_config_sha256",
        "model_index_sha256",
    )
    missing = [name for name in required if not common.get(name)]
    if missing:
        raise ValueError(f"canonical identity is missing required fields: {missing}")
    return {
        "canonical_format_version": CANONICAL_FORMAT_VERSION,
        "page_hash_algorithm": "sglang_chained_sha256_u32le_v1",
        "engine": {
            "name": common["engine"],
            "version": common.get("engine_version"),
            "source_commit": common["source_commit"],
        },
        "model": {
            "served_model_name": common["served_model_name"],
            "revision": common["model_revision"],
            "config_sha256": common["model_config_sha256"],
            "index_sha256": common["model_index_sha256"],
            "dtype": common["dtype"],
            "quantization": common.get("quantization"),
        },
        "kv_layout": model_kv_layout(config),
        "prefixes": [
            {
                "request_index": row["request_index"],
                "input_tokens": row["input_tokens"],
                "prompt_sha256": row["prompt_sha256"],
            }
            for row in prompt_pages(config)
        ],
    }


def canonical_fingerprint(config: dict[str, Any]) -> str:
    return json_fingerprint(canonical_spec(config))


def layer_partition(config: dict[str, Any], instance: dict[str, Any]) -> list[int]:
    layout = model_kv_layout(config)
    pp_size = int(instance["pp_size"])
    explicit = instance.get("layer_partition")
    if explicit is not None:
        partition = [int(value) for value in explicit]
    elif pp_size == 1:
        partition = [int(layout["num_layers"])]
    else:
        raise ValueError(
            "cross-topology conversion requires an explicit PP layer_partition"
        )
    if len(partition) != pp_size or sum(partition) != int(layout["num_layers"]):
        raise ValueError(
            f"invalid layer partition for {instance['id']}: {partition}; "
            f"expected {pp_size} stages and {layout['num_layers']} total layers"
        )
    return partition


def materialization_spec(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_fingerprint": canonical_fingerprint(config),
        "file_format": "sglang_hicache_file_page_first_v1",
        "topology": config["topology"],
        "instances": [
            {
                "id": instance["id"],
                "node_count": len(instance["nodes"]),
                "tp_size": int(instance["tp_size"]),
                "pp_size": int(instance["pp_size"]),
                "layer_partition": layer_partition(config, instance),
                "request_indices": request_indices_for_instance(config, i),
            }
            for i, instance in enumerate(config["instances"])
        ],
        "mem_layout": config["hicache"]["mem_layout"],
    }


def materialization_fingerprint(config: dict[str, Any]) -> str:
    return json_fingerprint(materialization_spec(config))


def disk_root(config: dict[str, Any]) -> Path:
    return Path(
        config.get("checkpoint_store", {}).get("disk_root", DEFAULT_DISK_ROOT)
    ).resolve()


def artifact_path(config: dict[str, Any]) -> Path:
    return disk_root(config) / canonical_fingerprint(config)


def canonical_page_path(root: Path, page_key: str) -> Path:
    if len(page_key) != 64 or any(char not in "0123456789abcdef" for char in page_key):
        raise ValueError(f"invalid page key: {page_key!r}")
    return root / "pages" / page_key[:2] / f"{page_key}.bin"


def _model_suffix(config: dict[str, Any]) -> str:
    return "-".join(str(config["common"]["served_model_name"]).split("/"))


def local_rank_filename(
    config: dict[str, Any],
    page_key: str,
    tp_rank: int,
    tp_size: int,
    pp_rank: int,
    pp_size: int,
) -> str:
    suffix = f"_{_model_suffix(config)}_{tp_rank}_{tp_size}"
    if pp_size > 1:
        suffix += f"_{pp_size}_{pp_rank}"
    return f"{page_key}{suffix}.bin"


def _rank_coordinates(
    instance: dict[str, Any], node_rank: int
) -> list[tuple[int, int]]:
    tp_size = int(instance["tp_size"])
    pp_size = int(instance["pp_size"])
    world_size = tp_size * pp_size
    node_count = len(instance["nodes"])
    if node_count < 1 or world_size % node_count:
        raise ValueError(
            f"instance {instance['id']} cannot distribute {world_size} ranks "
            f"uniformly across {node_count} nodes"
        )
    ranks_per_node = world_size // node_count
    if not 0 <= node_rank < node_count:
        raise ValueError(f"node rank is outside instance {instance['id']}: {node_rank}")
    start = node_rank * ranks_per_node
    return [
        (global_rank % tp_size, global_rank // tp_size)
        for global_rank in range(start, start + ranks_per_node)
    ]


def _layer_range(partition: list[int], pp_rank: int) -> tuple[int, int]:
    start = sum(partition[:pp_rank])
    return start, start + partition[pp_rank]


def local_file_size(
    layout: dict[str, Any], partition: list[int], tp_size: int, pp_rank: int
) -> int:
    if int(layout["num_kv_heads"]) % tp_size:
        raise ValueError(
            f"num_kv_heads={layout['num_kv_heads']} is not divisible by TP={tp_size}"
        )
    local_heads = int(layout["num_kv_heads"]) // tp_size
    return (
        2
        * int(layout["page_size"])
        * int(partition[pp_rank])
        * local_heads
        * int(layout["head_dim"])
        * int(layout["item_size"])
    )


def request_indices_for_instance(
    config: dict[str, Any], instance_index: int
) -> list[int]:
    explicit = config["instances"][instance_index].get("cache_request_indices")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("instance.cache_request_indices must be a non-empty list")
        request_count = len(prefix_token_sequences(config))
        values = [int(value) for value in explicit]
        if len(values) != len(set(values)):
            raise ValueError(
                "instance.cache_request_indices must not contain duplicates"
            )
        if any(value < 0 or value >= request_count for value in values):
            raise ValueError(
                "instance.cache_request_indices contains an unavailable prefix index"
            )
        return values
    return [
        index
        for index, assigned in enumerate(endpoint_assignments(config))
        if assigned == instance_index
    ]


def page_keys_for_instance(config: dict[str, Any], instance_index: int) -> list[str]:
    prompts = prompt_pages(config)
    result = []
    seen = set()
    for request_index in request_indices_for_instance(config, instance_index):
        for page_key in prompts[request_index]["page_keys"]:
            if page_key not in seen:
                seen.add(page_key)
                result.append(page_key)
    return result


def expected_local_files(
    config: dict[str, Any], instance_index: int, node_rank: int
) -> list[dict[str, Any]]:
    instance = config["instances"][instance_index]
    layout = model_kv_layout(config)
    partition = layer_partition(config, instance)
    rows = {}
    for page_key in page_keys_for_instance(config, instance_index):
        for tp_rank, pp_rank in _rank_coordinates(instance, node_rank):
            filename = local_rank_filename(
                config,
                page_key,
                tp_rank,
                int(instance["tp_size"]),
                pp_rank,
                int(instance["pp_size"]),
            )
            rows[filename] = {
                "page_key": page_key,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "filename": filename,
                "bytes": local_file_size(
                    layout, partition, int(instance["tp_size"]), pp_rank
                ),
            }
    return [rows[name] for name in sorted(rows)]


def expected_local_inventory(
    config: dict[str, Any], instance_index: int, node_rank: int
) -> dict[str, Any]:
    files = expected_local_files(config, instance_index, node_rank)
    return _inventory_from_files(files)


def prune_unexpected_local_files(
    directory: str | Path,
    config: dict[str, Any],
    instance_index: int,
    node_rank: int,
) -> dict[str, Any]:
    """Remove pages accumulated after the immutable checkpoint was installed."""
    root = Path(directory)
    _ensure_below(root, LOCAL_CACHE_ROOT, "local materialization")
    if not root.is_dir():
        return {"pruned_file_count": 0, "pruned_bytes": 0, "pruned_sample": []}
    expected = {
        row["filename"]
        for row in expected_local_files(config, instance_index, node_rank)
    }
    unexpected = sorted(
        path for path in root.iterdir() if path.name not in expected
    )
    unsafe = [
        path.name for path in unexpected if path.is_symlink() or not path.is_file()
    ]
    if unsafe:
        raise ValueError(f"refusing unexpected non-file cache entries: {unsafe[:10]}")
    removed_bytes = sum(path.stat().st_size for path in unexpected)
    sample = [path.name for path in unexpected[:10]]
    for path in unexpected:
        path.unlink()
    return {
        "pruned_file_count": len(unexpected),
        "pruned_bytes": removed_bytes,
        "pruned_sample": sample,
    }


def _inventory_from_files(files: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for row in sorted(files, key=lambda item: item["filename"]):
        digest.update(row["filename"].encode())
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode())
        digest.update(b"\n")
    return {
        "expected_file_count": len(files),
        "expected_bytes": sum(int(row["bytes"]) for row in files),
        "expected_inventory_digest": digest.hexdigest(),
    }


def inspect_local_materialization(
    directory: str | Path,
    config: dict[str, Any],
    instance_index: int,
    node_rank: int,
) -> dict[str, Any]:
    root = Path(directory)
    expected = expected_local_files(config, instance_index, node_rank)
    expected_inventory = _inventory_from_files(expected)
    if not root.is_dir():
        return {
            "path": str(root),
            "exists": False,
            **expected_inventory,
            "matched_file_count": 0,
            "matched_bytes": 0,
            "missing_count": len(expected),
            "missing_sample": [row["filename"] for row in expected[:10]],
            "wrong_size_count": 0,
            "wrong_size_sample": [],
            "unexpected_file_count": 0,
            "unexpected_bytes": 0,
            "unexpected_sample": [],
            "oldest_mtime": None,
            "newest_mtime": None,
            "valid": False,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    missing = []
    wrong_size = []
    mtimes = []
    matched_bytes = 0
    expected_names = {row["filename"] for row in expected}
    unexpected = sorted(
        path for path in root.iterdir() if path.name not in expected_names
    )
    for row in expected:
        path = root / row["filename"]
        try:
            stat = path.stat()
        except FileNotFoundError:
            missing.append(row["filename"])
            continue
        if not path.is_file() or path.is_symlink() or stat.st_size != row["bytes"]:
            wrong_size.append(
                {
                    "file": row["filename"],
                    "expected": row["bytes"],
                    "actual": stat.st_size,
                }
            )
            continue
        matched_bytes += stat.st_size
        mtimes.append(stat.st_mtime)
    return {
        "path": str(root),
        "exists": root.is_dir(),
        **expected_inventory,
        "matched_file_count": len(expected) - len(missing) - len(wrong_size),
        "matched_bytes": matched_bytes,
        "missing_count": len(missing),
        "missing_sample": missing[:10],
        "wrong_size_count": len(wrong_size),
        "wrong_size_sample": wrong_size[:10],
        "unexpected_file_count": len(unexpected),
        "unexpected_bytes": sum(
            path.stat().st_size
            for path in unexpected
            if path.is_file() and not path.is_symlink()
        ),
        "unexpected_sample": [path.name for path in unexpected[:10]],
        "oldest_mtime": min(mtimes, default=None),
        "newest_mtime": max(mtimes, default=None),
        "valid": root.is_dir() and not missing and not wrong_size and bool(expected),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def _ensure_below(path: Path, root: Path, label: str) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"{label} must be below {resolved_root}: {resolved}")


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _shape(layout: dict[str, Any], layers: int, heads: int) -> tuple[int, ...]:
    return (
        2,
        int(layout["page_size"]),
        layers,
        heads,
        int(layout["head_dim"]),
        int(layout["item_size"]),
    )


def scatter_node_to_canonical(
    config: dict[str, Any],
    instance_index: int,
    node_rank: int,
    source: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Scatter one physical node's TP/PP rank files into canonical page tensors."""
    import numpy as np

    source_path = Path(source)
    destination_path = Path(destination)
    _ensure_below(source_path, LOCAL_CACHE_ROOT, "source")
    _ensure_below(destination_path, disk_root(config), "destination")
    inspection = inspect_local_materialization(
        source_path, config, instance_index, node_rank
    )
    if not inspection["valid"]:
        raise ValueError(f"source materialization is incomplete: {inspection}")

    instance = config["instances"][instance_index]
    layout = model_kv_layout(config)
    partition = layer_partition(config, instance)
    tp_size = int(instance["tp_size"])
    pp_size = int(instance["pp_size"])
    full_shape = _shape(layout, int(layout["num_layers"]), int(layout["num_kv_heads"]))
    full_bytes = int(np.prod(full_shape))
    local_heads = int(layout["num_kv_heads"]) // tp_size
    written_pieces = 0
    page_count = 0
    coordinates = _rank_coordinates(instance, node_rank)
    for page_key in page_keys_for_instance(config, instance_index):
        output = canonical_page_path(destination_path, page_key)
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(output, os.O_RDWR | os.O_CREAT, 0o664)
        try:
            if os.fstat(descriptor).st_size == 0:
                os.ftruncate(descriptor, full_bytes)
            elif os.fstat(descriptor).st_size != full_bytes:
                raise ValueError(f"canonical page has wrong size: {output}")
        finally:
            os.close(descriptor)
        target = np.memmap(output, mode="r+", dtype=np.uint8, shape=full_shape)
        for tp_rank, pp_rank in coordinates:
            layer_start, layer_end = _layer_range(partition, pp_rank)
            head_start = tp_rank * local_heads
            head_end = head_start + local_heads
            filename = local_rank_filename(
                config, page_key, tp_rank, tp_size, pp_rank, pp_size
            )
            local_shape = _shape(layout, layer_end - layer_start, local_heads)
            source_page = np.memmap(
                source_path / filename, mode="r", dtype=np.uint8, shape=local_shape
            )
            target[:, :, layer_start:layer_end, head_start:head_end, :, :] = source_page
            del source_page
            written_pieces += 1
        target.flush()
        del target
        page_count += 1
    return {
        "status": "scattered",
        "source": inspection,
        "instance_id": instance["id"],
        "node_rank": node_rank,
        "page_count": page_count,
        "written_rank_pieces": written_pieces,
        "canonical_page_bytes": full_bytes,
    }


def _canonical_manifest_pages(manifest: dict[str, Any]) -> dict[int, list[str]]:
    return {
        int(row["request_index"]): list(row["page_keys"])
        for row in manifest["prefixes"]
    }


def materialize_node_from_canonical(
    config: dict[str, Any],
    instance_index: int,
    node_rank: int,
    manifest_path: str | Path,
    destination: str | Path,
    *,
    reserve_bytes: int = 0,
) -> dict[str, Any]:
    """Slice canonical pages into the exact HiCache files needed on one node."""
    import numpy as np

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("state") != "complete":
        raise ValueError("canonical manifest is not complete format version 1")
    if manifest.get("canonical_fingerprint") != canonical_fingerprint(config):
        raise ValueError(
            "canonical manifest does not match the requested model/prefix set"
        )
    if manifest.get("canonical_format") != canonical_spec(config):
        raise ValueError("canonical manifest format metadata does not match the config")
    canonical_root = manifest_file.parent
    _ensure_below(canonical_root, disk_root(config), "canonical source")
    destination_path = Path(destination)
    _ensure_below(destination_path, LOCAL_CACHE_ROOT, "destination")

    expected = expected_local_inventory(config, instance_index, node_rank)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination_path.parent / f".{destination_path.name}.materialize.lock"
    with _lock(lock_path):
        normalization = prune_unexpected_local_files(
            destination_path, config, instance_index, node_rank
        )
        current = inspect_local_materialization(
            destination_path, config, instance_index, node_rank
        )
        if current["valid"]:
            return {
                "status": "already_present",
                "inventory": current,
                **normalization,
            }
        free = shutil.disk_usage(destination_path.parent).free
        required = int(expected["expected_bytes"]) + int(reserve_bytes)
        if free < required:
            raise OSError(f"insufficient /dev/shm space: need {required}, have {free}")
        token = uuid.uuid4().hex
        staging = destination_path.parent / f".{destination_path.name}.incoming-{token}"
        previous = destination_path.parent / f".{destination_path.name}.old-{token}"
        staging.mkdir(parents=False, exist_ok=False)
        try:
            instance = config["instances"][instance_index]
            layout = model_kv_layout(config)
            partition = layer_partition(config, instance)
            tp_size = int(instance["tp_size"])
            pp_size = int(instance["pp_size"])
            local_heads = int(layout["num_kv_heads"]) // tp_size
            full_shape = _shape(
                layout, int(layout["num_layers"]), int(layout["num_kv_heads"])
            )
            keys_by_request = _canonical_manifest_pages(manifest)
            page_keys = []
            seen = set()
            for request_index in request_indices_for_instance(config, instance_index):
                for page_key in keys_by_request[request_index]:
                    if page_key not in seen:
                        seen.add(page_key)
                        page_keys.append(page_key)
            for page_key in page_keys:
                canonical = np.memmap(
                    canonical_page_path(canonical_root, page_key),
                    mode="r",
                    dtype=np.uint8,
                    shape=full_shape,
                )
                for tp_rank, pp_rank in _rank_coordinates(instance, node_rank):
                    layer_start, layer_end = _layer_range(partition, pp_rank)
                    head_start = tp_rank * local_heads
                    head_end = head_start + local_heads
                    local = np.ascontiguousarray(
                        canonical[
                            :,
                            :,
                            layer_start:layer_end,
                            head_start:head_end,
                            :,
                            :,
                        ]
                    )
                    filename = local_rank_filename(
                        config, page_key, tp_rank, tp_size, pp_rank, pp_size
                    )
                    temporary = staging / f".{filename}.tmp"
                    local.tofile(temporary)
                    temporary.replace(staging / filename)
                del canonical
            installed = inspect_local_materialization(
                staging, config, instance_index, node_rank
            )
            if not installed["valid"]:
                raise RuntimeError(
                    f"generated materialization failed validation: {installed}"
                )
            if destination_path.exists():
                destination_path.replace(previous)
            staging.replace(destination_path)
            shutil.rmtree(previous, ignore_errors=True)
            final = inspect_local_materialization(
                destination_path, config, instance_index, node_rank
            )
            if not final["valid"]:
                raise RuntimeError(
                    f"installed materialization failed validation: {final}"
                )
            return {
                "status": "materialized",
                "inventory": final,
                "finished_at": time.time(),
            }
        except BaseException:
            if previous.exists() and not destination_path.exists():
                previous.replace(destination_path)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(previous, ignore_errors=True)


def planned_materializations(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    fingerprint = materialization_fingerprint(config)
    for instance_index, instance in enumerate(config["instances"]):
        path = checkpoint_path(config, instance["id"])
        for node_rank, node in enumerate(instance["nodes"]):
            rows.append(
                {
                    "materialization_fingerprint": fingerprint,
                    "instance_index": instance_index,
                    "instance_id": instance["id"],
                    "node_rank": node_rank,
                    "node": node,
                    "path": path,
                    "required_gpu_count": len(_rank_coordinates(instance, node_rank)),
                    **expected_local_inventory(config, instance_index, node_rank),
                }
            )
    return rows
