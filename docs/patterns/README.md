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

### Hardcoded Runtime Parameters

The `F1R3_*` environment variables were removed in v0.4.10. These parameters are now hardcoded. Operator-tunable settings (heartbeat enabled/interval/max-lfb-age/cooldown) are in `defaults.conf` under `casper.heartbeat`.

Hardcoded values for reference:

| Parameter | Value | Component |
|-----------|-------|-----------|
| Proposer min interval | 250ms | Proposer |
| Heartbeat frontier chase max lag | 0 | Heartbeat |
| Heartbeat pending deploy max lag | 20 | Heartbeat |
| Heartbeat deploy recovery max lag | 64 | Heartbeat |
| Heartbeat stale recovery min interval | 12000ms | Heartbeat |
| Heartbeat deploy finalization grace | 25000ms | Heartbeat |
| REST find-deploy retry interval | 50ms | REST API |
| REST find-deploy max attempts | 1 | REST API |
| gRPC find-deploy retry interval | 100ms | gRPC API |
| gRPC find-deploy max attempts | 80 | gRPC API |
| Adaptive deploy cap enabled | true | Block creation |
| Adaptive deploy cap target | 1000ms | Block creation |
| Adaptive deploy cap min | 1 | Block creation |
| Adaptive deploy cap small batch bypass | 3 | Block creation |
| Adaptive deploy cap backlog floor enabled | true | Block creation |
| Adaptive deploy cap backlog trigger | 2 | Block creation |
| Adaptive deploy cap backlog divisor | 2 | Block creation |
| Adaptive deploy cap backlog min | 2 | Block creation |
| Adaptive deploy cap backlog max | 8 | Block creation |

### FFI Boundary (Scala Interop)
- C ABI via `extern "C"` functions in `rholang/src/lib.rs` and `rspace_rhotypes`
- Protobuf serialization across the FFI boundary
- 4-byte little-endian length prefix on returned buffers
- Manual memory management: Rust allocates (`Box::leak`), Scala must free
- `ALLOCATED_BYTES` atomic counter for leak tracking

[<- Back to overview](./README.md)
