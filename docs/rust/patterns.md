> Last updated: 2026-03-23

# Key Patterns & Conventions

### Concurrency Model
- **Async/await** (Tokio) throughout for I/O-bound operations
- **`DashMap`** for lock-free concurrent hash maps (hot store, casper buffer, in-flight deduplication)
- **`imbl`** persistent collections for O(1) snapshot cloning (DAG state)
- **`Arc<Mutex<T>>`** for shared mutable state (connections, cost manager)
- **`mpsc::unbounded_channel`** for block processor queue (no backpressure)
- **`Semaphore`-bounded spawning** for background tasks (e.g., transfer extraction limited to 8 concurrent)
- **`Shared<BoxFuture>`** for in-flight request deduplication (CacheTransactionAPI)

### Error Handling
- `eyre::Result` at top level
- Domain-specific enums: `RSpaceError`, `CommError`, `InterpreterError`, `BlockError`, `KvStoreError`
- `thiserror` for derive-based error types

### Serialization
- **Protobuf** (prost) for network messages and RhoAPI types
- **Bincode** for LMDB key/value encoding
- **Serde JSON** for HTTP API responses
- **LZ4** compression for block storage (Java-compatible format)
- Custom binary encoding for radix trie nodes

### Storage
- **All LMDB**, not RocksDB
- Multiple environments: rspace/history, rspace/cold, blockstorage, dagstorage, eval/history, eval/cold, etc.
- `data_dir` defaults to `/var/lib/rnode` in Docker, `~/.rnode` locally

### Testing
- `proptest` for property-based testing (block generators, deploy generators)
- Test utils modules in casper, rholang, models
- `#[cfg(not(test))]` guards for production-only code (e.g., jemalloc reporter)

### Metrics
- `metrics` crate with Prometheus exposition
- Span-based tracing for operation timing
- Source prefixes: `f1r3fly.comm.*`, `f1r3fly.casper.*`, `f1r3fly.rspace.*`

### Environment Variables (Runtime Tuning)

| Variable | Default | Component |
|----------|---------|-----------|
| `F1R3_PROPOSER_MIN_INTERVAL_MS` | 250 | Proposer |
| `F1R3_HEARTBEAT_FRONTIER_CHASE_MAX_LAG` | 0 | Heartbeat |
| `F1R3_HEARTBEAT_PENDING_DEPLOY_MAX_LAG` | 20 | Heartbeat |
| `F1R3_HEARTBEAT_DEPLOY_RECOVERY_MAX_LAG` | 64 | Heartbeat |
| `F1R3_HEARTBEAT_SELF_PROPOSE_COOLDOWN_MS` | 0 | Heartbeat |
| `F1R3_HEARTBEAT_STALE_RECOVERY_MIN_INTERVAL_MS` | 12000 | Heartbeat |
| `F1R3_HEARTBEAT_DEPLOY_FINALIZATION_GRACE_MS` | 25000 | Heartbeat |
| `F1R3_FIND_DEPLOY_RETRY_INTERVAL_MS` | 50 | REST API |
| `F1R3_FIND_DEPLOY_MAX_ATTEMPTS` | 1 | REST API |
| `F1R3_GRPC_FIND_DEPLOY_RETRY_INTERVAL_MS` | 100 | gRPC API |
| `F1R3_GRPC_FIND_DEPLOY_MAX_ATTEMPTS` | 80 | gRPC API |
| `F1R3_ADAPTIVE_DEPLOY_CAP_ENABLED` | true | Block creation |
| `F1R3_ADAPTIVE_DEPLOY_CAP_TARGET_MS` | 1000 | Block creation |
| `F1R3_ADAPTIVE_DEPLOY_CAP_MIN` | 1 | Block creation |
| `F1R3_ADAPTIVE_DEPLOY_CAP_SMALL_BATCH_BYPASS` | 3 | Block creation |
| `F1R3_ADAPTIVE_DEPLOY_CAP_BACKLOG_FLOOR_ENABLED` | true | Block creation |
| `F1R3_ADAPTIVE_DEPLOY_CAP_BACKLOG_TRIGGER` | 2 | Block creation |
| `F1R3_ADAPTIVE_DEPLOY_CAP_BACKLOG_DIVISOR` | 2 | Block creation |
| `F1R3_ADAPTIVE_DEPLOY_CAP_BACKLOG_MIN` | 2 | Block creation |
| `F1R3_ADAPTIVE_DEPLOY_CAP_BACKLOG_MAX` | 8 | Block creation |
| `F1R3_CLIQUE_YIELD_CHECK_INTERVAL` | - | Safety oracle |
| `F1R3_CLIQUE_YIELD_TIMESLICE_MS` | - | Safety oracle |
| `F1R3_FINALIZER_WORK_BUDGET_MS` | - | Finalization |
| `F1R3_FINALIZER_STEP_TIMEOUT_MS` | - | Finalization |

### FFI Boundary (Scala Interop)
- C ABI via `extern "C"` functions in `rholang/src/lib.rs` and `rspace_rhotypes`
- Protobuf serialization across the FFI boundary
- 4-byte little-endian length prefix on returned buffers
- Manual memory management: Rust allocates (`Box::leak`), Scala must free
- `ALLOCATED_BYTES` atomic counter for leak tracking

[<- Back to overview](./README.md)
