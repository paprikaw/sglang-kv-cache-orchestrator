from __future__ import annotations

from pathlib import Path

import pytest

from kv_cache_orchestrator import weights
from kv_cache_orchestrator.weights import (
    inspect_weight_cache,
    materialize_weight_cache,
    runtime_model_path,
    source_weight_inventory,
    weight_fingerprint,
)

from .helpers import clone_topology, make_config


def enable_weight_cache(config: dict, root: Path) -> None:
    config["weight_cache"] = {
        "enabled": True,
        "local_root": str(root / "ram" / "weights"),
        "storage_min_free_space": 0,
    }


def test_weight_identity_is_independent_of_tp_pp_and_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "LOCAL_CACHE_ROOT", tmp_path)
    source = make_config(tmp_path, topology="tp4", tp_size=4, pp_size=1)
    enable_weight_cache(source, tmp_path)
    target = clone_topology(
        source,
        topology="pp4",
        tp_size=1,
        pp_size=4,
        partition=[1, 1, 1, 1],
        nodes=["a", "b", "c", "d"],
    )
    target["hicache"]["storage_path_template"] = str(tmp_path / "other" / "kv")

    assert weight_fingerprint(source) == weight_fingerprint(target)
    assert runtime_model_path(source) == runtime_model_path(target)


def test_weight_cache_materialization_is_atomic_and_byte_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "LOCAL_CACHE_ROOT", tmp_path)
    config = make_config(tmp_path)
    enable_weight_cache(config, tmp_path)
    source = Path(config["common"]["model"])
    (source / ".cache").mkdir()
    (source / ".cache" / "ignored").write_bytes(b"not-runtime-data")

    result = materialize_weight_cache(config)
    observed = inspect_weight_cache(config)
    repeated = materialize_weight_cache(config)

    assert result["status"] == "materialized"
    assert observed["valid"] is True
    assert repeated["status"] == "already_present"
    assert observed["matched_bytes"] == source_weight_inventory(config)["bytes"]
    for row in observed["manifest"]["files"]:
        relative = Path(row["path"])
        assert (runtime_model_path(config) / relative).read_bytes() == (
            source / relative
        ).read_bytes()
    assert not (runtime_model_path(config) / ".cache").exists()


def test_weight_cache_inspection_rejects_truncated_file(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "LOCAL_CACHE_ROOT", tmp_path)
    config = make_config(tmp_path)
    enable_weight_cache(config, tmp_path)
    materialize_weight_cache(config)
    shard = runtime_model_path(config) / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"short")

    observed = inspect_weight_cache(config)

    assert observed["valid"] is False
    assert observed["reason"].startswith("file_size_mismatch:")


def test_weight_materialization_rejects_wrong_declared_revision_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(weights, "LOCAL_CACHE_ROOT", tmp_path)
    config = make_config(tmp_path)
    enable_weight_cache(config, tmp_path)
    config["common"]["model_config_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="model identity mismatch"):
        materialize_weight_cache(config)
