from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from kv_cache_orchestrator.sglang_format import (
    canonical_fingerprint,
    canonical_page_path,
    canonical_spec,
    expected_local_files,
    layer_partition,
    model_kv_layout,
    prompt_pages,
)


def make_config(
    root: Path,
    *,
    topology: str = "tp4",
    tp_size: int = 4,
    pp_size: int = 1,
    layer_partition_value: list[int] | None = None,
    prefixes: list[list[int]] | None = None,
    nodes: list[str] | None = None,
    num_layers: int = 4,
    num_kv_heads: int = 4,
) -> dict:
    model = root / "model"
    model.mkdir(parents=True, exist_ok=True)
    model_config = model / "config.json"
    model_config.write_text(
        json.dumps(
            {
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_kv_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": 2,
            }
        )
    )
    model_index = model / "model.safetensors.index.json"
    model_index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {"layer.weight": "model-00001-of-00001.safetensors"},
            }
        )
    )
    (model / "model-00001-of-00001.safetensors").write_bytes(bytes(range(16)))

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    prefixes = prefixes or [[1, 2, 3, 4]]
    nodes = nodes or ["node0"]
    instance = {
        "id": "service0",
        "nodes": nodes,
        "tp_size": tp_size,
        "pp_size": pp_size,
    }
    if layer_partition_value is not None:
        instance["layer_partition"] = layer_partition_value
    return {
        "schema_version": 1,
        "topology": topology,
        "checkpoint_id": f"tiny-{topology}",
        "checkpoint_store": {"disk_root": str(root / "disk")},
        "common": {
            "engine": "sglang",
            "engine_version": "test",
            "source_commit": "0123456789abcdef",
            "model": str(model),
            "model_revision": "model-revision",
            "model_config_sha256": digest(model_config),
            "model_index_sha256": digest(model_index),
            "served_model_name": "Tiny-GQA",
            "dtype": "float16",
            "quantization": None,
            "page_size": 2,
        },
        "hicache": {
            "mem_layout": "page_first",
            "storage_path_template": str(
                root / "local" / "{checkpoint_id}" / "{instance_id}"
            ),
            "storage_min_free_space": 0,
        },
        "workload": {
            "concurrency": len(prefixes),
            "prefixes": prefixes,
            "routing": "single",
        },
        "instances": [instance],
    }


def global_pages(config: dict) -> dict[str, np.ndarray]:
    layout = model_kv_layout(config)
    shape = (
        2,
        int(layout["page_size"]),
        int(layout["num_layers"]),
        int(layout["num_kv_heads"]),
        int(layout["head_dim"]),
        int(layout["item_size"]),
    )
    keys = sorted(
        {key for prefix in prompt_pages(config) for key in prefix["page_keys"]}
    )
    size = int(np.prod(shape))
    return {
        key: ((np.arange(size, dtype=np.uint16) + index * 29) % 256)
        .astype(np.uint8)
        .reshape(shape)
        for index, key in enumerate(keys)
    }


def expected_slice(config: dict, row: dict, page: np.ndarray) -> np.ndarray:
    instance = config["instances"][0]
    partition = layer_partition(config, instance)
    pp_rank = int(row["pp_rank"])
    tp_rank = int(row["tp_rank"])
    layer_start = sum(partition[:pp_rank])
    layer_end = layer_start + partition[pp_rank]
    local_heads = page.shape[3] // int(instance["tp_size"])
    head_start = tp_rank * local_heads
    head_end = head_start + local_heads
    return np.ascontiguousarray(
        page[:, :, layer_start:layer_end, head_start:head_end, :, :]
    )


def write_materialization(
    config: dict, node_directories: list[Path], pages: dict[str, np.ndarray]
) -> None:
    for node_rank, directory in enumerate(node_directories):
        directory.mkdir(parents=True, exist_ok=True)
        for row in expected_local_files(config, 0, node_rank):
            expected_slice(config, row, pages[row["page_key"]]).tofile(
                directory / row["filename"]
            )


def assert_materialization(
    config: dict, node_directories: list[Path], pages: dict[str, np.ndarray]
) -> None:
    for node_rank, directory in enumerate(node_directories):
        for row in expected_local_files(config, 0, node_rank):
            actual = (directory / row["filename"]).read_bytes()
            expected = expected_slice(config, row, pages[row["page_key"]]).tobytes()
            assert actual == expected


def write_manifest(config: dict, root: Path, pages: dict[str, np.ndarray]) -> Path:
    for key, page in pages.items():
        path = canonical_page_path(root, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        page.tofile(path)
    page_bytes = next(iter(pages.values())).nbytes
    prefixes = prompt_pages(config)
    manifest = {
        "schema_version": 1,
        "state": "complete",
        "canonical_fingerprint": canonical_fingerprint(config),
        "canonical_format": canonical_spec(config),
        "created_at": time.time(),
        "canonical_page_bytes": page_bytes,
        "page_count": len(pages),
        "bytes": len(pages) * page_bytes,
        "prefixes": prefixes,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def clone_topology(
    config: dict,
    *,
    topology: str,
    tp_size: int,
    pp_size: int,
    partition: list[int] | None,
    nodes: list[str] | None = None,
) -> dict:
    result = copy.deepcopy(config)
    result["topology"] = topology
    result["checkpoint_id"] = f"tiny-{topology}"
    instance = result["instances"][0]
    instance["nodes"] = nodes or ["node0"]
    instance["tp_size"] = tp_size
    instance["pp_size"] = pp_size
    if partition is None:
        instance.pop("layer_partition", None)
    else:
        instance["layer_partition"] = partition
    return result
