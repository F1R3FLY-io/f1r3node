# F1r3fly Documentation

## Workspace Overview

The Cargo workspace contains 11 crates:

| Crate | Role | Docs |
|-------|------|------|
| [`shared`](../shared/) | Foundation types, KV store abstraction, LMDB bindings | [docs](./shared/) |
| [`crypto`](../crypto/) | Hashing (Blake2b, Keccak256, SHA256), signing (Secp256k1, Ed25519), TLS certs | [docs](./crypto/) |
| [`models`](../models/) | Protobuf-generated types, domain structs, Rholang AST, sorted collections | [docs](./models/) |
| [`rspace++`](../rspace++/) | Tuple space engine: produce/consume matching, LMDB-backed trie history | [docs](./rspace/) |
| [`rholang`](../rholang/) | Rholang interpreter: parser, normalizer, reducer, cost accounting, system processes | [docs](./rholang/) |
| [`casper`](../casper/) | CBC Casper consensus: block creation/validation, DAG, safety oracle, finalization | [docs](./casper/) |
| [`block-storage`](../block-storage/) | Block persistence, DAG storage (imbl), casper buffer, deploy index | [docs](./block-storage/) |
| [`comm`](../comm/) | P2P networking: Kademlia DHT, TLS transport, connection management | [docs](./comm/) |
| [`node`](../node/) | Binary entry point: boot sequence, gRPC/HTTP servers, CLI, diagnostics | [docs](./node/) |
| [`graphz`](../graphz/) | Graphviz DOT generation for DAG visualization | [docs](./graphz/) |

## Architecture & Dependency Graph

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

### Core Crates

| Module | Description |
|--------|-------------|
| [shared](./shared/) | Foundation types, KV store abstraction, LMDB bindings |
| [crypto](./crypto/) | Hashing, signing, certificates |
| [models](./models/) | Protobuf types, Rholang AST, sorted collections |
| [rspace](./rspace/) | Tuple space engine, produce/consume matching, trie history |
| [rholang](./rholang/) | Interpreter, reducer, cost accounting, system processes |
| [casper](./casper/) | CBC Casper consensus, block creation/validation, finalization |
| [block-storage](./block-storage/) | Block persistence, DAG storage, deploy index |
| [comm](./comm/) | P2P networking, Kademlia DHT, TLS transport |
| [node](./node/) | Binary entry point, gRPC/HTTP servers, CLI, diagnostics |
| [graphz](./graphz/) | Graphviz DOT generation for DAG visualization |

### Cross-Cutting

| Document | Description |
|----------|-------------|
| [Data Flows](./data-flows/) | Block lifecycle and deploy execution flows |
| [Patterns & Conventions](./patterns/) | Concurrency, error handling, serialization, env vars |

### Consensus

| Document | Description |
|----------|-------------|
| [Casper Overview](./casper/) | Block creation, validation, DAG merging, finalization |
| [Consensus Protocol](./casper/CONSENSUS_PROTOCOL.md) | End-to-end protocol walkthrough, abstraction boundaries for adding new consensus |
| [Byzantine Fault Tolerance](./casper/BYZANTINE_FAULT_TOLERANCE.md) | BFT architecture, clique oracle, equivocation detection, slashing |
| [Synchrony Constraint](./casper/SYNC_CONSTRAINT.md) | Synchrony constraint mechanism, configuration, troubleshooting |
| [Consensus Configuration](https://github.com/F1R3FLY-io/system-integration/blob/main/docs/consensus-configuration.md) | FTT and synchrony threshold semantics, recommended values |

### Rholang Language

| Document | Description |
|----------|-------------|
| [Rholang Evaluator](../rholang/README.md) | Language overview, CLI usage, known issues |
| [Rholang Module Docs](./rholang/) | Interpreter internals, reducer, system processes |
| [Rholang Tutorial](./rholang/rholangtut.md) | Language tutorial |
| [Pattern Matching](./rholang/rholangmatchingtut.md) | Pattern matching guide |
| [Ollama Integration](./rholang/ollama.md) | Local LLM integration via Ollama |
| [Reference Documentation](../rholang/reference_doc/README.md) | Language reference by topic |

### Cryptography

| Document | Description |
|----------|-------------|
| [Crypto Module](./crypto/) | Hashing, signing, certificates |
| [Schnorr/FROST Design](./schnorr-frost-secp256k1-design.md) | Schnorr and FROST signature scheme design |
| [Schnorr/FROST Status](./schnorr-frost-secp256k1-status.md) | Implementation status |

### Architecture & Design

| Document | Description |
|----------|-------------|
| [F1r3fly Architecture](./f1r3fly_architecture.md) | High-level architecture overview |
| [F1r3fly State Diagram](./f1r3fly_state_diagram.md) | Node lifecycle state diagrams |
| [Namespaces & Scaling](./namespaces-scaling-mercury.md) | Namespace organization, regional namespaces |
| [Rholang Language Analysis](./rholang-language-analysis.md) | Language design analysis |
| [Features](./features.md) | Feature requirements and status |

### Genesis & Token Identity

The native token's name, symbol, and decimals are configured before genesis and baked into the on-chain `TokenMetadata` contract at `rho:system:tokenMetadata`. They are immutable after genesis and exposed via `/api/status`. See:

- [defaults.conf](../node/src/main/resources/defaults.conf) — `native-token-name`, `native-token-symbol`, `native-token-decimals` under `casper.genesis-block-data`
- [Docker Genesis Configuration](../docker/README.md#native-token) — compose env var overrides, API and on-chain query examples
- [Consensus Protocol § Configuration](./casper/CONSENSUS_PROTOCOL.md#10-configuration) — genesis-locked parameters

### Infrastructure

| Document | Description |
|----------|-------------|
| [WebSocket Events](./node/websocket-events.md) | `/ws/events` endpoint: 9 event types, startup replay, payload schemas |
| [Docker Setup](../docker/README.md) | Docker compose for shard, standalone, monitoring |
| [RNode API](./rnode-api/) | Protocol Buffer API documentation |
| [LFS Requester Architecture](./plans/lfs_tuple_space_requester_concurrency_architecture.md) | LFS tuple space concurrency design |
| [Whiteblock Test Plan](./whiteblock/whiteblock-test-plan.md) | Network testing plan |

### Archive

Legacy and superseded documents are in [archive/](./archive/).
