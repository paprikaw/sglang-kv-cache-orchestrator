from __future__ import annotations

import json
import shutil
import subprocess

import pytest

import kv_cache_orchestrator.sglang_format as sglang_format
from kv_cache_orchestrator.peer import materialize_node_from_peer, route_to_peer

from .helpers import global_pages, make_config, write_materialization


def test_route_guard_rejects_management_interface():
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, json.dumps([{"dev": "eno8303", "prefsrc": "192.0.2.1"}]), ""
        )

    with pytest.raises(RuntimeError, match="not required fabric"):
        route_to_peer(
            "source-node",
            interface_regex=r"^bond0(?:\.|$)",
            runner=runner,
            resolver=lambda _host: "192.0.2.2",
        )


def test_peer_materialization_is_validated_and_atomically_installed(
    tmp_path, monkeypatch
):
    local_root = tmp_path / "local"
    monkeypatch.setattr(sglang_format, "LOCAL_CACHE_ROOT", local_root)
    config = make_config(tmp_path)
    source = tmp_path / "fake-remote-source"
    write_materialization(config, [source], global_pages(config))
    (source / "post-checkpoint-page.bin").write_bytes(b"stale")
    destination = local_root / "target"

    def runner(argv, **_kwargs):
        if argv[0] == "ip":
            return subprocess.CompletedProcess(
                argv,
                0,
                json.dumps([{"dev": "bond0.3027", "prefsrc": "192.0.2.2"}]),
                "",
            )
        assert argv[0] == "rsync"
        shutil.copytree(source, argv[-1].rstrip("/"), dirs_exist_ok=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = materialize_node_from_peer(
        config,
        0,
        0,
        "source-node",
        "/dev/shm/source",
        destination,
        interface_regex=r"^bond0(?:\.|$)",
        runner=runner,
        resolver=lambda _host: "192.0.2.3",
    )

    assert result["status"] == "materialized_from_peer"
    assert result["route"]["device"] == "bond0.3027"
    assert result["inventory"]["valid"] is True
    assert result["pruned_file_count"] == 1
    assert result["pruned_bytes"] == 5
    assert not (destination / "post-checkpoint-page.bin").exists()
