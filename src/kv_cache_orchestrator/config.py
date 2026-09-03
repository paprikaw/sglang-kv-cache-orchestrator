from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def deterministic_prompt(
    namespace: int, prefix_index: int, input_tokens: int
) -> list[int]:
    """Generate prefixes compatible with the original Decode replay harness."""
    if input_tokens < 1:
        raise ValueError("input_tokens must be positive")
    first = 1_000 + namespace * 100 + prefix_index
    seed = namespace * 911 + prefix_index * 101
    return [first] + [
        20_000 + ((seed + position * 17) % 8_191)
        for position in range(input_tokens - 1)
    ]


def _read_token_file(path: Path) -> list[int]:
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("token_ids")
    if not isinstance(value, list):
        raise ValueError(f"token file must contain a list or token_ids object: {path}")
    return value


def _validate_tokens(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty token ID list")
    tokens = []
    for token in value:
        if not isinstance(token, int) or not 0 <= token <= 0xFFFFFFFF:
            raise ValueError(f"{label} contains an invalid uint32 token ID: {token!r}")
        tokens.append(token)
    return tokens


def prefix_token_sequences(config: dict[str, Any]) -> list[list[int]]:
    """Resolve the topology-neutral prefix set described by a config.

    `workload.prefixes` accepts inline lists, `{\"token_ids\": [...]}` objects,
    or `{\"token_file\": \"...json\"}` objects. If it is absent, the legacy
    deterministic replay generator is used.
    """
    workload = config["workload"]
    prefixes = workload.get("prefixes")
    if prefixes is None:
        return [
            _validate_tokens(
                deterministic_prompt(
                    int(workload.get("prompt_namespace", 0)),
                    index,
                    int(workload["input_tokens"]),
                ),
                f"generated prefix {index}",
            )
            for index in range(int(workload["concurrency"]))
        ]
    if not isinstance(prefixes, list) or not prefixes:
        raise ValueError("workload.prefixes must be a non-empty list")
    result = []
    for index, row in enumerate(prefixes):
        label = f"workload.prefixes[{index}]"
        if isinstance(row, list):
            raw_tokens = row
        elif isinstance(row, dict) and "token_ids" in row:
            raw_tokens = row["token_ids"]
        elif isinstance(row, dict) and "token_file" in row:
            raw_tokens = _read_token_file(Path(str(row["token_file"])))
        else:
            raise ValueError(
                f"{label} must be a token list, token_ids object, or token_file object"
            )
        result.append(_validate_tokens(raw_tokens, label))
    configured_count = workload.get("concurrency")
    if configured_count is not None and int(configured_count) != len(result):
        raise ValueError(
            "workload.concurrency does not match the explicit prefix count: "
            f"{configured_count} != {len(result)}"
        )
    return result


def endpoint_assignments(config: dict[str, Any]) -> list[int]:
    count = len(prefix_token_sequences(config))
    instances = config["instances"]
    if not instances:
        raise ValueError("at least one instance is required")
    routing = config["workload"].get("routing", "round_robin")
    if routing == "round_robin":
        assignments = [index % len(instances) for index in range(count)]
    elif routing == "single":
        if len(instances) != 1:
            raise ValueError("single routing requires exactly one instance")
        assignments = [0] * count
    elif routing == "explicit":
        raw = config["workload"].get("endpoint_assignments")
        if not isinstance(raw, list) or len(raw) != count:
            raise ValueError(
                "explicit routing requires one endpoint assignment per prefix"
            )
        assignments = [int(value) for value in raw]
    else:
        raise ValueError(f"unsupported routing: {routing}")
    if any(index < 0 or index >= len(instances) for index in assignments):
        raise ValueError(
            f"endpoint assignment is outside the instance range: {assignments}"
        )
    return assignments


def checkpoint_path(config: dict[str, Any], instance_id: str) -> str:
    hicache = config["hicache"]
    return str(hicache["storage_path_template"]).format(
        checkpoint_id=config["checkpoint_id"],
        topology=config["topology"],
        instance_id=instance_id,
    )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config and make token-file paths portable across controller/workers."""
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text())
    prefixes = config.get("workload", {}).get("prefixes")
    if isinstance(prefixes, list):
        config = copy.deepcopy(config)
        for row in config["workload"]["prefixes"]:
            if isinstance(row, dict) and "token_file" in row:
                token_path = Path(str(row["token_file"]))
                if not token_path.is_absolute():
                    row["token_file"] = str((config_path.parent / token_path).resolve())
    return config
