from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import pytest

import kv_cache_orchestrator.sglang_format as sglang_format
from kv_cache_orchestrator.sglang_format import (
    canonical_fingerprint,
    canonical_page_path,
    expected_local_files,
    inspect_local_materialization,
    materialization_fingerprint,
    materialization_spec,
    materialize_node_from_canonical,
    page_hashes,
    scatter_node_to_canonical,
)

from .helpers import (
    assert_materialization,
    clone_topology,
    global_pages,
    make_config,
    write_manifest,
    write_materialization,
)

TOPOLOGIES = {
    "tp4": (4, 1, None),
    "pp4": (1, 4, [1, 1, 1, 1]),
    "tp2pp2": (2, 2, [2, 2]),
}


def test_page_hashes_match_chained_u32_little_endian():
    first = hashlib.sha256(struct.pack("<2I", 1, 2)).digest()
    second = hashlib.sha256(first + struct.pack("<2I", 3, 4)).hexdigest()

    assert page_hashes([1, 2, 3, 4], 2) == [first.hex(), second]


@pytest.mark.parametrize(
    ("source_name", "target_name"),
    [
        ("pp4", "tp4"),
        ("tp4", "pp4"),
        ("tp4", "tp2pp2"),
        ("tp2pp2", "tp4"),
        ("pp4", "tp2pp2"),
        ("tp2pp2", "pp4"),
    ],
)
def test_canonical_round_trip_across_tp_and_pp(
    tmp_path, monkeypatch, source_name, target_name
):
    local_root = tmp_path / "local"
    monkeypatch.setattr(sglang_format, "LOCAL_CACHE_ROOT", local_root)
    source_tp, source_pp, source_partition = TOPOLOGIES[source_name]
    target_tp, target_pp, target_partition = TOPOLOGIES[target_name]
    source = make_config(
        tmp_path,
        topology=source_name,
        tp_size=source_tp,
        pp_size=source_pp,
        layer_partition_value=source_partition,
    )
    target = clone_topology(
        source,
        topology=target_name,
        tp_size=target_tp,
        pp_size=target_pp,
        partition=target_partition,
    )
    pages = global_pages(source)
    source_dirs = [local_root / "source-node0"]
    write_materialization(source, source_dirs, pages)
    canonical_root = tmp_path / "disk" / ".incoming"
    canonical_root.mkdir(parents=True)

    result = scatter_node_to_canonical(source, 0, 0, source_dirs[0], canonical_root)

    assert result["written_rank_pieces"] == len(expected_local_files(source, 0, 0))
    shape = next(iter(pages.values())).shape
    for key, expected in pages.items():
        actual = np.memmap(
            canonical_page_path(canonical_root, key),
            mode="r",
            dtype=np.uint8,
            shape=shape,
        )
        np.testing.assert_array_equal(actual, expected)
    manifest = write_manifest(source, canonical_root, pages)
    target_dir = local_root / "target-node0"

    installed = materialize_node_from_canonical(target, 0, 0, manifest, target_dir)

    assert installed["status"] == "materialized"
    assert installed["inventory"]["valid"] is True
    assert_materialization(target, [target_dir], pages)
    assert canonical_fingerprint(source) == canonical_fingerprint(target)
    assert materialization_fingerprint(source) != materialization_fingerprint(target)


def test_canonical_round_trip_across_multi_node_tp4pp2_and_tp8(tmp_path, monkeypatch):
    local_root = tmp_path / "local"
    monkeypatch.setattr(sglang_format, "LOCAL_CACHE_ROOT", local_root)
    source = make_config(
        tmp_path,
        topology="tp4pp2",
        tp_size=4,
        pp_size=2,
        layer_partition_value=[2, 2],
        nodes=["node0", "node1"],
        num_kv_heads=8,
    )
    target = clone_topology(
        source,
        topology="tp8",
        tp_size=8,
        pp_size=1,
        partition=None,
        nodes=["node2", "node3"],
    )
    pages = global_pages(source)
    source_dirs = [local_root / "source-node0", local_root / "source-node1"]
    write_materialization(source, source_dirs, pages)
    canonical_root = tmp_path / "disk" / ".incoming"
    canonical_root.mkdir(parents=True)
    for node_rank, directory in enumerate(source_dirs):
        scatter_node_to_canonical(source, 0, node_rank, directory, canonical_root)
    shape = next(iter(pages.values())).shape
    for key, expected in pages.items():
        actual = np.memmap(
            canonical_page_path(canonical_root, key),
            mode="r",
            dtype=np.uint8,
            shape=shape,
        )
        np.testing.assert_array_equal(actual, expected)
    manifest = write_manifest(source, canonical_root, pages)
    target_dirs = [local_root / "target-node0", local_root / "target-node1"]
    for node_rank, directory in enumerate(target_dirs):
        materialize_node_from_canonical(target, 0, node_rank, manifest, directory)

    assert_materialization(target, target_dirs, pages)
    assert canonical_fingerprint(source) == canonical_fingerprint(target)


def test_shared_prefix_pages_are_counted_once(tmp_path):
    config = make_config(tmp_path, prefixes=[[1, 2, 3, 4], [1, 2, 5, 6]])

    files = expected_local_files(config, 0, 0)

    assert len(files) == 3 * 4


def test_inspection_rejects_wrong_sized_rank_file(tmp_path, monkeypatch):
    local_root = tmp_path / "local"
    monkeypatch.setattr(sglang_format, "LOCAL_CACHE_ROOT", local_root)
    config = make_config(tmp_path)
    pages = global_pages(config)
    directory = local_root / "node0"
    write_materialization(config, [directory], pages)
    first = expected_local_files(config, 0, 0)[0]
    (directory / first["filename"]).write_bytes(b"short")

    observed = inspect_local_materialization(directory, config, 0, 0)

    assert observed["valid"] is False
    assert observed["wrong_size_count"] == 1


def test_manifest_fingerprint_must_match(tmp_path, monkeypatch):
    local_root = tmp_path / "local"
    monkeypatch.setattr(sglang_format, "LOCAL_CACHE_ROOT", local_root)
    config = make_config(tmp_path)
    pages = global_pages(config)
    manifest = write_manifest(config, tmp_path / "disk" / "artifact", pages)
    raw = json.loads(manifest.read_text())
    raw["canonical_fingerprint"] = "0" * 64
    manifest.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="does not match"):
        materialize_node_from_canonical(config, 0, 0, manifest, local_root / "target")


def test_cache_request_indices_can_replicate_one_prefix_to_two_instances(tmp_path):
    config = make_config(tmp_path)
    first = config["instances"][0]
    first["cache_request_indices"] = [0]
    second = dict(first)
    second.update({"id": "service1", "nodes": ["node1"]})
    config["instances"].append(second)
    config["workload"].update({"routing": "explicit", "endpoint_assignments": [0]})

    spec = materialization_spec(config)

    assert spec["instances"][0]["request_indices"] == [0]
    assert spec["instances"][1]["request_indices"] == [0]
    assert expected_local_files(config, 0, 0)
    assert [row["filename"] for row in expected_local_files(config, 0, 0)] == [
        row["filename"] for row in expected_local_files(config, 1, 0)
    ]
