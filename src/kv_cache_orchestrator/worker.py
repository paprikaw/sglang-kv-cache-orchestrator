#!/usr/bin/env python3
"""Node worker for canonical KV scatter and topology-specific materialization."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .config import checkpoint_path, load_config
from .sglang_format import (
    LOCAL_CACHE_ROOT,
    inspect_local_materialization,
    materialize_node_from_canonical,
    scatter_node_to_canonical,
)


def load(path: str) -> dict:
    return load_config(path)


def emit(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def status() -> dict[str, object]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    if gpu.returncode:
        raise RuntimeError(gpu.stderr.strip() or gpu.stdout.strip())
    if compute.returncode:
        raise RuntimeError(compute.stderr.strip() or compute.stdout.strip())
    shm = os.statvfs(LOCAL_CACHE_ROOT)
    gpu_rows = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    processes = [line.strip() for line in compute.stdout.splitlines() if line.strip()]
    return {
        "gpu_rows": gpu_rows,
        "gpu_count": len(gpu_rows),
        "compute_processes": processes,
        "busy": bool(processes),
        "shm_free_bytes": shm.f_bavail * shm.f_frsize,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")

    for name in ("inspect", "scatter", "materialize"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--instance-index", type=int, required=True)
        command.add_argument("--node-rank", type=int, required=True)
        command.add_argument("--path")
        if name == "scatter":
            command.add_argument("--destination", required=True)
        if name == "materialize":
            command.add_argument("--manifest", required=True)
            command.add_argument("--reserve-bytes", type=int, default=0)
    args = parser.parse_args()

    if args.command == "status":
        emit(status())
        return 0
    config = load(args.config)
    instance = config["instances"][args.instance_index]
    path = args.path or checkpoint_path(config, instance["id"])
    if args.command == "inspect":
        emit(
            inspect_local_materialization(
                path, config, args.instance_index, args.node_rank
            )
        )
        return 0
    if args.command == "scatter":
        emit(
            scatter_node_to_canonical(
                config,
                args.instance_index,
                args.node_rank,
                path,
                args.destination,
            )
        )
        return 0
    emit(
        materialize_node_from_canonical(
            config,
            args.instance_index,
            args.node_rank,
            args.manifest,
            path,
            reserve_bytes=args.reserve_bytes,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
