# Streaming File Upload — Implementation Progress

> Master tracker for all tasks.  
> See individual task files in this folder for full details.  
> Architecture: [streaming_file_upload.md](file:///Users/andriistefaniv/Code/f1r3fly/f1r3fly/docs/streaming_file_upload.md)

---

## Task Overview

| # | Task | Layer | Status |
|---|------|-------|--------|
| 1 | [Protobuf & gRPC Definitions](./task-01-protobuf-grpc.md) | L1, L3 | ✅ Done |
| 2 | [Streaming File Ingestion](./task-02-streaming-ingestion.md) | L1 | ✅ Done |
| 3 | [Synthetic Deploy & Mempool Integration](./task-03-synthetic-deploy.md) | L1 | ✅ Done |
| 4 | [Orphan File Cleanup](./task-04-orphan-cleanup.md) | L1 | ✅ Done |
| 5 | [P2P File Sync Protocol](./task-05-p2p-file-sync.md) | L2 | ✅ Done |
| 6 | [FileSystemProcess — `rho:io:file` System Process](./task-06-file-system-process.md) | L3 | ✅ Done |
| 7 | [FileRegistry.rho — On-Chain Contract](./task-07-file-registry-contract.md) | L3 | ✅ Done |
| 8 | [gRPC Download (Observer-Only)](./task-08-grpc-download.md) | L3 | ✅ Done |
| 9 | [DA-Optimistic Consensus](./task-09-da-consensus.md) | L4 | ✅ Done |
| 10 | [Storage-Proportional Phlo Pricing](./task-10-phlo-pricing.md) | L5 | ✅ Done |
| 11 | [Configuration & CLI Flags](./task-11-configuration.md) | Cross-cutting | ✅ Done |
| 12 | [Integration Tests](./task-12-integration-tests.md) | Cross-cutting | ✅ Done |
| 13 | [Genesis Hash Mismatch Fix](./task-13-genesis-hash-fix.md) | L3 | ✅ Done |
| 14 | [P2P Replication Bug Fix](./task-14-p2p-replication-fix.md) | L2 | ✅ Done |

---

## Status Legend

| Icon | Meaning |
|------|---------|
| ⬜ | Not Started |
| 🔵 | In Progress |
| ✅ | Done |
| ⏸️ | Blocked |

---

## Parallel Execution Plan

Maximum parallelism: **3 lanes running simultaneously**.

### Phase 1 — Start in parallel, no dependencies

| Lane A (Ingestion) | Lane B (P2P / Consensus) | Lane C (On-Chain) |
|---|---|---|
| **T1** Protobuf & gRPC | **T1** _(shared with A)_ | **T7** FileRegistry.rho |
| **T11** Configuration | | |

> T1 and T11 are prerequisites for most Scala work. T7 is pure Rholang — zero Scala dependencies, can start on Day 1.

### Phase 2 — After T1 completes

| Lane A | Lane B | Lane C |
|---|---|---|
| **T2** Streaming Ingestion | **T5** P2P File Sync | **T6** FileSystemProcess |
| | **T8** gRPC Download | _(starts once T7 is done)_ |

> T2, T5, and T8 are all unblocked by T1 and independent of each other.  
> T6 can start as soon as T7 (pure Rholang) is done — in parallel with T2.

### Phase 3 — After T2 and T6 complete

| Lane A | Lane B | Lane C |
|---|---|---|
| **T3** Synthetic Deploy | **T9** DA Consensus | **T10** Phlo Pricing |

> T9 needs T5 + T6. T10 needs T6. T3 needs T2.

### Phase 4 — After T3

| Lane A | Lane B |
|---|---|
| **T4** Orphan Cleanup | _(T9 / T10 may still be running)_ |

### Phase 5 — After everything

| |
|---|
| **T12** Integration Tests |

---

## Dependency Graph

```
T1 (Proto) ──────┬──► T2 (Ingestion) ──► T3 (Synthetic Deploy) ──► T4 (Orphan Cleanup)
                 ├──► T5 (P2P Sync)  ──────────────────────────────┐
                 └──► T8 (Download)                                 │
                                                                    ▼
T7 (FileRegistry.rho) ──► T6 (FileSystemProcess) ──────────────► T9 (DA Consensus) ──► T12
                                                  └──────────► T10 (Phlo Pricing) ────► T12

T11 (Config) ──► T2, T9 (consumed by multiple)
```
