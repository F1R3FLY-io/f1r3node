# WebSocket Event Streaming

The node streams real-time events to clients via the `/ws/events` WebSocket endpoint.

## Connection

```
ws://<host>:<http-port>/ws/events
```

On connect, the server sends:
1. A `started` handshake: `{"event": "started", "schema-version": 1}`
2. Buffered startup events (replayed from the startup buffer)
3. Live events as they occur

Startup events that fired before the client connected are replayed from an in-memory buffer. The buffer is sealed when the node reaches Running state, after which only live events are streamed.

## Event Types

All events use the envelope format:

```json
{
  "event": "<event-type>",
  "schema-version": 1,
  "payload": { ... }
}
```

### Block Lifecycle

These fire continuously as the node processes blocks.

| Event | Description | Payload |
|-------|-------------|---------|
| `block-created` | Block proposed by this validator (before validation) | block-hash, parent-hashes, justification-hashes, deploys, creator, seq-num |
| `block-added` | Block validated and added to DAG | Same as block-created |
| `block-finalised` | Block finalized by consensus | Same as block-created |

The `deploys` array contains objects with `id`, `cost`, `deployer`, and `errored` fields.

### Genesis Ceremony

These fire once during node startup when the genesis block is being created and approved.

| Event | Description | Payload |
|-------|-------------|---------|
| `sent-unapproved-block` | Boot node broadcasts genesis candidate | block-hash |
| `block-approval-received` | Boot node receives approval from a validator | block-hash, sender |
| `sent-approved-block` | Boot node broadcasts the approved genesis block | block-hash |
| `approved-block-received` | Validator receives the approved genesis block | block-hash |

### Node Lifecycle

| Event | Description | Payload |
|-------|-------------|---------|
| `entered-running-state` | Engine transitions from Initializing to Running | block-hash |
| `node-started` | HTTP/gRPC servers are ready | address |

## Startup Event Replay

Events published during startup (before any WebSocket client can connect) are buffered in memory. When a client connects, these events are replayed before entering the live stream. The buffer is sealed when the engine completes initialization.

This ensures clients receive the full node lifecycle regardless of when they connect:
- `node-started` (HTTP server ready)
- Genesis ceremony events (`sent-unapproved-block`, `block-approval-received`, etc.)
- `entered-running-state` (node is operational)

Events that arrive both via replay and live stream are deduplicated.

## Publish Sites

| Event | Source File |
|-------|------------|
| `block-created` | `casper/src/rust/blocks/proposer/proposer.rs` |
| `block-added` | `casper/src/rust/multi_parent_casper_impl.rs` |
| `block-finalised` | `casper/src/rust/multi_parent_casper_impl.rs` |
| `sent-unapproved-block` | `casper/src/rust/engine/approve_block_protocol.rs` |
| `block-approval-received` | `casper/src/rust/engine/approve_block_protocol.rs` |
| `sent-approved-block` | `casper/src/rust/engine/approve_block_protocol.rs` |
| `approved-block-received` | `casper/src/rust/engine/initializing.rs` |
| `entered-running-state` | `casper/src/rust/engine/engine.rs` |
| `node-started` | `node/src/rust/runtime/node_runtime.rs` |

## Implementation

- **Event bus**: `shared/src/rust/shared/f1r3fly_events.rs` — `F1r3flyEvents` struct with tokio broadcast channel (capacity 100) and startup buffer
- **Event types**: `shared/src/rust/shared/f1r3fly_event.rs` — `F1r3flyEvent` enum with 9 variants
- **WebSocket handler**: `node/src/rust/web/events_info.rs` — handles connection, replay, dedup, and live streaming
- **Startup seal**: `node/src/rust/runtime/node_runtime.rs` — calls `seal_startup()` after engine_init completes
