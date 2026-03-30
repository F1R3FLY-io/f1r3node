> Last updated: 2026-03-23

# Workspace Overview

The Cargo workspace contains 11 crates:

| Crate | Role | Lines of Rust (approx) |
|-------|------|----------------------|
| `shared` | Foundation types, KV store abstraction, LMDB bindings | Base layer |
| `crypto` | Hashing (Blake2b, Keccak256, SHA256), signing (Secp256k1, Ed25519), TLS certs | Base layer |
| `models` | Protobuf-generated types, domain structs (blocks, deploys, validators), Rholang AST, sorted collections | Core types |
| `rspace++` | Tuple space engine: produce/consume matching, LMDB-backed trie history, checkpointing | Storage engine |
| `rholang` | Rholang interpreter: parser, normalizer, reducer, cost accounting, system processes | Execution engine |
| `casper` | CBC Casper consensus: block creation/validation, DAG, safety oracle, finalization | Consensus |
| `block-storage` | Block persistence, DAG storage (imbl), casper buffer, deploy index | Persistence |
| `comm` | P2P networking: Kademlia DHT, TLS transport, connection management | Networking |
| `node` | Binary entry point: boot sequence, gRPC/HTTP servers, CLI, diagnostics | Orchestrator |
| `graphz` | Graphviz DOT generation for DAG visualization | Utility |
| `rspace++/libs/rspace_rhotypes` | C FFI bindings for RSpace <-> Scala JNA interop | FFI bridge |

**Workspace-level dependencies** (from root `Cargo.toml`):
- Async: `tokio` (full), `tokio-stream`, `futures`, `async-trait`
- gRPC/Proto: `tonic`, `prost`, `tonic-prost`, `tonic-prost-build`
- Serialization: `bincode`, `serde`, `serde_json`
- Observability: `tracing`, `tracing-subscriber`, `metrics`
- Concurrency: `dashmap`, `imbl` (persistent collections)
- Crypto: `rand`
- Error handling: `eyre`, `thiserror`
- Misc: `hex`, `itertools`, `regex`, `stacker`, `tower`

---

# Architecture & Dependency Graph

```
                         ┌──────────┐
                         │   node   │  (binary, orchestrator)
                         └────┬─────┘
              ┌───────┬───────┼───────┬──────────┐
              v       v       v       v          v
          ┌──────┐ ┌──────┐ ┌──────┐ ┌───────┐ ┌──────┐
          │casper│ │ comm │ │graphz│ │rholang│ │block-│
          │      │ │      │ │      │ │       │ │store │
          └──┬───┘ └──┬───┘ └──────┘ └───┬───┘ └──┬───┘
             │        │                   │        │
     ┌───────┼────────┤                   │        │
     v       v        v                   v        v
  ┌──────┐ ┌──────┐ ┌──────┐        ┌────────┐ ┌──────┐
  │models│ │crypto│ │shared│        │rspace++│ │shared│
  └──┬───┘ └──┬───┘ └──────┘        └────┬───┘ │      │
     │        │                          │      └──────┘
     v        v                          v
  ┌──────┐ ┌──────┐                   ┌──────┐
  │crypto│ │shared│                   │shared│
  └──┬───┘ └──────┘                   └──────┘
     v
  ┌──────┐
  │shared│
  └──────┘
```

**Dependency direction**: `shared` is the leaf dependency; `node` is the root.

---

## Module Documentation

| Module | Description |
|--------|-------------|
| [shared](./shared.md) | Foundation types, KV store abstraction, LMDB bindings |
| [crypto](./crypto.md) | Hashing, signing, certificates |
| [models](./models.md) | Protobuf types, Rholang AST, sorted collections |
| [rspace](./rspace.md) | Tuple space engine, produce/consume matching, trie history |
| [rholang](./rholang.md) | Interpreter, reducer, cost accounting, system processes |
| [casper](./casper.md) | CBC Casper consensus, block creation/validation, finalization |
| [block-storage](./block-storage.md) | Block persistence, DAG storage, deploy index |
| [comm](./comm.md) | P2P networking, Kademlia DHT, TLS transport |
| [node](./node.md) | Binary entry point, gRPC/HTTP servers, CLI, diagnostics |
| [graphz](./graphz.md) | Graphviz DOT generation for DAG visualization |

## Cross-Cutting Documentation

| Document | Description |
|----------|-------------|
| [Data Flows](./data-flows.md) | Block lifecycle and deploy execution flows |
| [Patterns & Conventions](./patterns.md) | Concurrency, error handling, serialization, env vars |
| [Consensus Configuration](https://github.com/F1R3FLY-io/system-integration/blob/main/docs/consensus-configuration.md) | FTT and synchrony threshold semantics, finalization formula, recommended values per validator set size |
