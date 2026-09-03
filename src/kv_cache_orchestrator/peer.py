from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from . import sglang_format

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _remote_cache_path(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or path == PurePosixPath("/dev/shm"):
        raise ValueError(f"peer source must be below /dev/shm: {value}")
    if PurePosixPath("/dev/shm") not in path.parents or ".." in path.parts:
        raise ValueError(f"peer source must be below /dev/shm: {value}")
    return str(path)


def _peer_host(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"invalid peer hostname: {value!r}")
    return value


def route_to_peer(
    peer: str,
    *,
    interface_regex: str,
    runner: CommandRunner = subprocess.run,
    resolver: Callable[[str], str] = socket.gethostbyname,
) -> dict[str, Any]:
    """Resolve and verify that peer traffic uses the requested fast fabric."""
    host = _peer_host(peer)
    address = resolver(host)
    proc = runner(
        ["ip", "-json", "route", "get", address],
        text=True,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    rows = json.loads(proc.stdout)
    if not rows:
        raise RuntimeError(f"no route to peer {host} ({address})")
    route = rows[0]
    device = str(route.get("dev", ""))
    if not re.search(interface_regex, device):
        raise RuntimeError(
            f"route to {host} uses {device!r}, not required fabric {interface_regex!r}"
        )
    return {
        "peer": host,
        "address": address,
        "device": device,
        "source_address": route.get("prefsrc") or route.get("src"),
        "interface_regex": interface_regex,
    }


@contextmanager
def _materialization_lock(path: Path) -> Iterator[None]:
    # Reuse the exact lock convention used by canonical-Disk materialization.
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def peer_rsync_command(peer: str, source: str, destination: Path) -> list[str]:
    return [
        "rsync",
        "-a",
        "--whole-file",
        "--no-compress",
        "--delete",
        "--protect-args",
        "-e",
        (
            "ssh -o BatchMode=yes -o ConnectTimeout=8 "
            "-o Compression=no -o IPQoS=throughput"
        ),
        f"{_peer_host(peer)}:{_remote_cache_path(source).rstrip('/')}/",
        f"{destination}/",
    ]


def materialize_node_from_peer(
    config: dict[str, Any],
    instance_index: int,
    node_rank: int,
    source_node: str,
    source_path: str,
    destination: str | Path,
    *,
    reserve_bytes: int = 0,
    interface_regex: str = r"^(?:bond0(?:\.|$)|ib|mlx)",
    runner: CommandRunner = subprocess.run,
    resolver: Callable[[str], str] = socket.gethostbyname,
) -> dict[str, Any]:
    """Atomically pull an exact topology view from a peer node's CPU RAM.

    The data path is rsync over SSH.  The route guard ensures that the peer's
    service address resolves over the configured high-speed fabric before any
    bytes are copied; this supports IPoIB and TCP over a RoCE-capable NIC.
    """
    destination_path = Path(destination)
    sglang_format._ensure_below(  # noqa: SLF001
        destination_path, sglang_format.LOCAL_CACHE_ROOT, "destination"
    )
    source = _remote_cache_path(source_path)
    route = route_to_peer(
        source_node,
        interface_regex=interface_regex,
        runner=runner,
        resolver=resolver,
    )
    expected = sglang_format.expected_local_inventory(config, instance_index, node_rank)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination_path.parent / f".{destination_path.name}.materialize.lock"
    with _materialization_lock(lock_path):
        normalization = sglang_format.prune_unexpected_local_files(
            destination_path, config, instance_index, node_rank
        )
        current = sglang_format.inspect_local_materialization(
            destination_path, config, instance_index, node_rank
        )
        if current["valid"]:
            return {
                "status": "already_present",
                "source": {"node": source_node, "path": source},
                "route": route,
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
        started = time.time()
        try:
            proc = runner(
                peer_rsync_command(source_node, source, staging),
                text=True,
                capture_output=True,
                timeout=14_400,
            )
            if proc.returncode:
                raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
            normalization = sglang_format.prune_unexpected_local_files(
                staging, config, instance_index, node_rank
            )
            installed = sglang_format.inspect_local_materialization(
                staging, config, instance_index, node_rank
            )
            if not installed["valid"]:
                raise RuntimeError(f"peer copy failed validation: {installed}")
            if destination_path.exists():
                destination_path.replace(previous)
            staging.replace(destination_path)
            shutil.rmtree(previous, ignore_errors=True)
            final = sglang_format.inspect_local_materialization(
                destination_path, config, instance_index, node_rank
            )
            if not final["valid"]:
                raise RuntimeError(f"installed peer copy failed validation: {final}")
            finished = time.time()
            return {
                "status": "materialized_from_peer",
                "source": {"node": source_node, "path": source},
                "transport": "rsync_over_ssh",
                "route": route,
                "duration_s": finished - started,
                "throughput_bytes_s": int(final["matched_bytes"])
                / max(finished - started, 1e-9),
                "inventory": final,
                "finished_at": finished,
                **normalization,
            }
        except BaseException:
            if previous.exists() and not destination_path.exists():
                previous.replace(destination_path)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(previous, ignore_errors=True)
