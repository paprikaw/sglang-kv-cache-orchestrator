from __future__ import annotations

import json

import pytest

from kv_cache_orchestrator.config import (
    endpoint_assignments,
    load_config,
    prefix_token_sequences,
)

from .helpers import make_config


def test_explicit_token_file_is_resolved_relative_to_config(tmp_path):
    config = make_config(tmp_path)
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"token_ids": [7, 8, 9, 10]}))
    config["workload"]["prefixes"] = [{"token_file": "tokens.json"}]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    loaded = load_config(path)

    assert prefix_token_sequences(loaded) == [[7, 8, 9, 10]]
    assert loaded["workload"]["prefixes"][0]["token_file"] == str(tokens)


def test_explicit_routing_is_validated(tmp_path):
    config = make_config(tmp_path, prefixes=[[1, 2], [3, 4]])
    config["instances"].append(
        {"id": "service1", "nodes": ["node1"], "tp_size": 4, "pp_size": 1}
    )
    config["workload"].update({"routing": "explicit", "endpoint_assignments": [1, 0]})

    assert endpoint_assignments(config) == [1, 0]

    config["workload"]["endpoint_assignments"] = [2, 0]
    with pytest.raises(ValueError, match="outside the instance range"):
        endpoint_assignments(config)
