# Architecture

## State hierarchy

The system has two data tiers and two identities.

| Object | Lifetime | Identity | Role |
| --- | --- | --- | --- |
| Canonical KV on shared Disk | Persistent | Canonical fingerprint | Authoritative backup and topology conversion boundary |
| SGLang rank files in node CPU RAM | Node/booking lifetime | Materialization fingerprint plus node slot | Fast local reuse by an exact TP/PP layout |

The registry losing a local cache record does not invalidate the Disk object.
Likewise, a restarted node only loses a materialization; it can reconstruct that
view from Disk.

## Controller

The controller is a CLI process rather than a resident service. It combines:

- registry state;
- current Slurm bookings and remaining time;
- GPU process and `/dev/shm` state returned by workers;
- live validation of the requested cache path on every eligible node.

The active probe also discovers caches created before the registry existed or
surviving a controller restart. A complete unregistered materialization is
adopted as `local_only` and becomes eligible for publication to Disk.

Placement scores complete node assignments lexicographically:

1. number of exact materialization hits;
2. total remaining booking lifetime;
3. total free `/dev/shm` capacity.

Busy nodes and bookings below the configured remaining-time threshold are
excluded. A miss requires enough free CPU RAM for the target rank files plus the
configured reserve.

## Worker

The worker has four commands:

| Command | Effect |
| --- | --- |
| `status` | Report GPU processes, GPU count, boot ID, and free local RAM |
| `inspect` | Validate every expected rank file and byte size |
| `scatter` | Copy this node's layer/head slices into staged canonical pages |
| `materialize` | Slice canonical pages into an atomic local SGLang directory |

The controller invokes one short-lived worker command over SSH. No worker daemon
or peer transport is required.

## Publish transaction

`publish` takes a complete topology-specific cache and creates one canonical
artifact.

1. Acquire a fingerprint-specific lock on shared Disk.
2. Create an incoming directory beside the final artifact.
3. Inspect each source node and scatter every expected rank piece sequentially.
4. Verify each worker wrote exactly its expected pieces.
5. Verify every unique canonical page exists with the exact byte size.
6. Write a complete manifest and atomically rename the incoming directory.
7. Register the Disk backing and observed source materialization.

Sequential scatter avoids concurrent writers to one shared page. A failed
publish removes the incoming directory and never exposes a complete manifest.

## Materialize transaction

`materialize` verifies the canonical fingerprint, checks capacity, and creates a
new local directory beside the current one. Every rank file is written through a
temporary name. After validation, the new directory atomically replaces the old
directory. If installation fails, the previous valid directory is restored.

## Deliberately omitted mechanisms

- node-to-node cache transfer;
- proactive migration based on reservation expiry;
- eviction policy for unrelated local caches;
- background reconciliation daemon.

These can be layered on later without changing canonical Disk identity. The
current design keeps the failure model simple: local RAM is disposable, shared
Disk is authoritative.
