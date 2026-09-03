from __future__ import annotations

import pytest

from kv_cache_orchestrator.controller import (
    _existing_backing,
    choose_placement,
    discover_cache_hits,
    parse_slurm_time_left,
)
from kv_cache_orchestrator.registry import (
    disk_backing_from_manifest,
    lookup_checkpoint,
    register_checkpoint,
    register_disk_checkpoint,
)
from kv_cache_orchestrator.sglang_format import artifact_path, planned_materializations

from .helpers import clone_topology, global_pages, make_config, write_manifest


def inventory_for(config):
    return {
        "nodes": [
            {
                **target,
                "valid": True,
                "matched_file_count": target["expected_file_count"],
                "matched_bytes": target["expected_bytes"],
            }
            for target in planned_materializations(config)
        ]
    }


def candidate(
    node: str,
    *,
    cached: bool = False,
    seconds: float = 10_000,
    free: int = 1_000_000,
    busy: bool = False,
):
    return {
        "node": node,
        "node_ip": f"192.0.2.{len(node)}",
        "job_id": f"job-{node}",
        "job_name": "test",
        "state": "RUNNING",
        "time_left": "02:46:40",
        "time_left_seconds": seconds,
        "end_time": "later",
        "busy": busy,
        "health": {"gpu_count": 4, "shm_free_bytes": free},
        "cached_slots": ["0:0"] if cached else [],
    }


def test_disk_hit_is_independent_of_local_topology(tmp_path):
    source = make_config(tmp_path, topology="tp4", tp_size=4, pp_size=1)
    target = clone_topology(
        source,
        topology="pp4",
        tp_size=1,
        pp_size=4,
        partition=[1, 1, 1, 1],
    )
    pages = global_pages(source)
    manifest = write_manifest(source, tmp_path / "disk" / "artifact", pages)
    registry = tmp_path / "registry.json"
    backing = disk_backing_from_manifest(manifest)

    register_checkpoint(
        source,
        inventory_for(source),
        tmp_path,
        registry,
        disk_backing=backing,
    )
    result = lookup_checkpoint(
        target, registry, verify_live=False, verify_disk_files=True
    )

    assert result["hit"] is True
    assert result["local_ready"] is None


def test_existing_canonical_artifact_is_detected(tmp_path):
    config = make_config(tmp_path)
    pages = global_pages(config)
    write_manifest(config, artifact_path(config), pages)

    backing = _existing_backing(config)

    assert backing is not None
    assert backing["page_count"] == len(pages)


def test_corrupt_disk_page_invalidates_disk_hit(tmp_path):
    config = make_config(tmp_path)
    pages = global_pages(config)
    manifest = write_manifest(config, tmp_path / "disk" / "artifact", pages)
    registry = tmp_path / "registry.json"
    register_disk_checkpoint(
        config, disk_backing_from_manifest(manifest), tmp_path, registry
    )
    next((manifest.parent / "pages").rglob("*.bin")).write_bytes(b"bad")

    result = lookup_checkpoint(
        config, registry, verify_live=False, verify_disk_files=True
    )

    assert result["hit"] is False
    assert result["reason"].startswith("canonical_page_invalid")


def test_placement_prefers_idle_exact_cache_hit(tmp_path):
    config = make_config(tmp_path)
    registry = tmp_path / "registry.json"
    candidates = [
        candidate("cached", cached=True, seconds=5_000),
        candidate("newer", seconds=50_000),
        candidate("busy", cached=True, seconds=100_000, busy=True),
    ]

    plan = choose_placement(config, candidates, registry)

    assert plan["decisions"][0]["node"] == "cached"
    assert plan["decisions"][0]["cache_action"] == "reuse_local"


def test_cache_discovery_probes_unregistered_candidate_paths(tmp_path):
    config = make_config(tmp_path)
    candidates = [candidate("cached"), candidate("empty")]

    discover_cache_hits(
        config,
        candidates,
        tmp_path / "registry.json",
        probe=lambda _config, target: {
            **target,
            "valid": target["node"] == "cached",
        },
    )

    assert candidates[0]["cached_slots"] == ["0:0"]
    assert candidates[1]["cached_slots"] == []


def test_live_lookup_adopts_complete_unregistered_materialization(tmp_path):
    config = make_config(tmp_path)
    registry = tmp_path / "registry.json"

    result = lookup_checkpoint(
        config,
        registry,
        verify_live=True,
        probe=lambda _config, target: {**target, "valid": True},
    )

    assert result["hit"] is False
    assert result["local_ready"] is True
    assert result["promotable"] is True
    assert result["reason"] == "canonical_disk_backing_missing"


def test_placement_uses_disk_when_exact_view_is_absent(tmp_path):
    config = make_config(tmp_path)
    pages = global_pages(config)
    manifest = write_manifest(config, tmp_path / "disk" / "artifact", pages)
    registry = tmp_path / "registry.json"
    register_disk_checkpoint(
        config, disk_backing_from_manifest(manifest), tmp_path, registry
    )

    plan = choose_placement(config, [candidate("fresh")], registry)

    assert plan["canonical_disk_available"] is True
    assert plan["decisions"][0]["cache_action"] == "materialize_from_canonical_disk"


def test_placement_falls_back_to_prefill_without_disk(tmp_path):
    config = make_config(tmp_path)

    plan = choose_placement(config, [candidate("fresh")], tmp_path / "registry.json")

    assert plan["canonical_disk_available"] is False
    assert plan["decisions"][0]["cache_action"] == "build_from_prefill"


def test_placement_rejects_gpu_shape_mismatch(tmp_path):
    config = make_config(tmp_path)
    wrong = candidate("two-gpu")
    wrong["health"]["gpu_count"] = 2

    with pytest.raises(RuntimeError, match="no placement"):
        choose_placement(config, [wrong], tmp_path / "registry.json")


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [("1-02:03:04", 93_784.0), ("02:03:04", 7_384.0), ("03:04", 184.0)],
)
def test_parse_slurm_time_left(raw, seconds):
    assert parse_slurm_time_left(raw) == seconds
