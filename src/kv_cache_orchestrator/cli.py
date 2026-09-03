#!/usr/bin/env python3
"""Operate canonical Disk KV, topology views, and cache-aware node placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .controller import (
    choose_placement,
    discover_cache_hits,
    discover_slurm_candidates,
    discover_weight_cache_hits,
    hydrate_config_from_disk,
    inspect_configured_weights,
    materialize_configured_weights,
    prepare_cache_aware_config,
    publish_checkpoint_to_disk,
    resolve_config,
    worker_weight_probe,
)
from .registry import (
    DEFAULT_REGISTRY,
    invalidate_checkpoint,
    list_checkpoints,
    lookup_checkpoint,
)
from .sglang_format import (
    canonical_fingerprint,
    canonical_spec,
    materialization_fingerprint,
    materialization_spec,
)
from .weights import weight_cache_enabled, weight_fingerprint, weight_identity


def emit(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def add_placement_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-node", action="append", default=[])
    parser.add_argument("--job-name-regex")
    parser.add_argument("--min-time-left-s", type=float, default=3600.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    sub = parser.add_subparsers(dest="command", required=True)

    fingerprint = sub.add_parser(
        "fingerprint", help="Show canonical and topology materialization identities"
    )
    fingerprint.add_argument("--config", required=True)

    sub.add_parser("list", help="List canonical Disk entries")

    invalidate = sub.add_parser("invalidate", help="Mark a canonical entry invalid")
    invalidate.add_argument("fingerprint")
    invalidate.add_argument("--reason", required=True)

    status = sub.add_parser(
        "status", help="Inspect canonical and configured-node state"
    )
    status.add_argument("--config", required=True)
    status.add_argument("--no-live-verify", action="store_true")
    status.add_argument("--verify-disk-files", action="store_true")

    weight_status = sub.add_parser(
        "weight-status", help="Inspect configured nodes for the model-weight cache"
    )
    weight_status.add_argument("--config", required=True)

    weight_materialize = sub.add_parser(
        "weight-materialize",
        help="Copy shared-Disk model weights into configured nodes' CPU RAM",
    )
    weight_materialize.add_argument("--config", required=True)
    weight_materialize.add_argument("--allow-busy", action="store_true")

    publish = sub.add_parser(
        "publish", help="Convert the configured TP/PP cache into canonical Disk pages"
    )
    publish.add_argument("--config", required=True)
    publish.add_argument("--run-dir", required=True)

    hydrate = sub.add_parser(
        "materialize",
        help="Generate the configured TP/PP node cache from canonical Disk",
    )
    hydrate.add_argument("--config", required=True)
    hydrate.add_argument("--run-dir")
    hydrate.add_argument("--allow-busy", action="store_true")

    plan = sub.add_parser(
        "plan", help="Prefer idle nodes with the requested TP/PP view"
    )
    plan.add_argument("--config", required=True)
    add_placement_arguments(plan)

    prepare = sub.add_parser(
        "prepare", help="Resolve a placement and materialize any Disk-backed misses"
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--dry-run", action="store_true")
    add_placement_arguments(prepare)
    args = parser.parse_args()

    registry = Path(args.registry).expanduser().resolve()
    if args.command == "list":
        emit(list_checkpoints(registry))
        return 0
    if args.command == "invalidate":
        changed = invalidate_checkpoint(args.fingerprint, args.reason, registry)
        emit({"invalidated": changed, "fingerprint": args.fingerprint})
        return 0 if changed else 3

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config["orchestrator_registry"] = str(registry)
    if args.command == "fingerprint":
        emit(
            {
                "canonical_fingerprint": canonical_fingerprint(config),
                "canonical_spec": canonical_spec(config),
                "materialization_fingerprint": materialization_fingerprint(config),
                "materialization_spec": materialization_spec(config),
                "weight_cache_enabled": weight_cache_enabled(config),
                "weight_fingerprint": (
                    weight_fingerprint(config) if weight_cache_enabled(config) else None
                ),
                "weight_identity": (
                    weight_identity(config) if weight_cache_enabled(config) else None
                ),
            }
        )
        return 0
    if args.command == "status":
        result = lookup_checkpoint(
            config,
            registry,
            verify_live=not args.no_live_verify,
            verify_disk_files=args.verify_disk_files,
        )
        result["weight_cache"] = inspect_configured_weights(config, config_path)
        emit(result)
        return 0 if result["hit"] else 3
    if args.command == "weight-status":
        result = inspect_configured_weights(config, config_path)
        emit(result)
        return 0 if all(row.get("valid") for row in result["nodes"]) else 3
    if args.command == "weight-materialize":
        emit(
            materialize_configured_weights(
                config, config_path, allow_busy=args.allow_busy
            )
        )
        return 0
    if args.command == "publish":
        emit(publish_checkpoint_to_disk(config, config_path, args.run_dir, registry))
        return 0
    if args.command == "materialize":
        emit(
            hydrate_config_from_disk(
                config,
                config_path,
                registry,
                run_dir=args.run_dir,
                allow_busy=args.allow_busy,
            )
        )
        return 0
    candidates = set(args.candidate_node) or None
    if args.command == "prepare":
        emit(
            prepare_cache_aware_config(
                config,
                config_path,
                args.output,
                registry_path=registry,
                candidate_nodes=candidates,
                job_name_regex=args.job_name_regex,
                min_time_left_seconds=args.min_time_left_s,
                dry_run=args.dry_run,
            )
        )
        return 0

    discovered = discover_slurm_candidates(
        config,
        candidate_nodes=candidates,
        job_name_regex=args.job_name_regex,
    )
    discover_cache_hits(config, discovered, registry)
    discover_weight_cache_hits(
        config,
        discovered,
        probe=worker_weight_probe(config_path),
    )
    placement = choose_placement(
        config,
        discovered,
        registry,
        min_time_left_seconds=args.min_time_left_s,
    )
    emit(
        {
            "plan": placement,
            "candidates": discovered,
            "resolved_config": resolve_config(config, placement),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
