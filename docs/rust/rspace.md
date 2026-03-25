> Last updated: 2026-03-23

# Crate: rspace++ (Tuple Space Storage)

**Path**: `rspace++/`

High-performance, persistent tuple space engine. The core storage substrate for Rholang smart contract communication.

## Core Concept

Processes communicate by placing data on **channels** and waiting for data with **patterns**:
- **Produce**: Put data on a channel. If a waiting consumer matches, fire a COMM event.
- **Consume**: Wait on channel(s) with pattern(s). If data already present matches, fire immediately.
- **COMM event**: A matched produce+consume pair triggers continuation execution.

## ISpace Trait (Core API)

```rust
pub trait ISpace<C, P, A, K>: Send + Sync {
    // Tuple space operations
    fn produce(&mut self, channel: C, data: A, persist: bool)
        -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError>;
    fn consume(&mut self, channels: Vec<C>, patterns: Vec<P>,
               continuation: K, persist: bool, peeks: BTreeSet<i32>)
        -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError>;
    fn install(&mut self, channels: Vec<C>, patterns: Vec<P>, continuation: K)
        -> Result<Option<(K, Vec<A>)>, RSpaceError>;

    // Checkpointing
    fn create_checkpoint(&mut self) -> Result<Checkpoint, RSpaceError>;
    fn create_soft_checkpoint(&mut self) -> SoftCheckpoint<C, P, A, K>;
    fn revert_to_soft_checkpoint(&mut self, cp: SoftCheckpoint<C, P, A, K>)
        -> Result<(), RSpaceError>;
    fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), RSpaceError>;

    // State inspection
    fn get_data(&self, channel: &C) -> Vec<Datum<A>>;
    fn get_waiting_continuations(&self, channels: Vec<C>)
        -> Vec<WaitingContinuation<P, K>>;
    fn get_joins(&self, channel: C) -> Vec<Vec<C>>;
    fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>>;

    // Replay support
    fn rig_and_reset(&mut self, start_root: Blake2b256Hash, log: Log)
        -> Result<(), RSpaceError>;
    fn check_replay_data(&self) -> Result<(), RSpaceError>;
    fn is_replay(&self) -> bool;
}
```

## Core Data Types

```rust
pub struct Datum<A: Clone> {
    pub a: A,              // The data value
    pub persist: bool,     // Survives matching if true
    pub source: Produce,   // Hash reference to origin event
}

pub struct WaitingContinuation<P: Clone, K: Clone> {
    pub patterns: Vec<P>,
    pub continuation: K,
    pub persist: bool,
    pub peeks: BTreeSet<i32>,  // Non-consuming read indices
    pub source: Consume,
}

pub struct Checkpoint {
    pub root: Blake2b256Hash,  // Trie root hash
    pub log: Log,              // Event log from this checkpoint
}
```

## Match Trait

```rust
pub trait Match<P, A>: Send + Sync {
    fn get(&self, p: P, a: A) -> Option<A>;  // None = no match
}
```

For Rholang, this implements spatial pattern matching on `Par` processes.

## Implementations

| Struct | Purpose |
|--------|---------|
| `RSpace<C,P,A,K>` | Primary implementation for block execution. `is_replay()` returns `false`. |
| `ReplayRSpace<C,P,A,K>` | Deterministic replay variant (tracks `replay_data: MultisetMultiMap<IOEvent, COMM>`). `is_replay()` returns `true`. |
| `ReportingRspace<C,P,A,K>` | Debugging variant that records all COMM events separated by soft checkpoints. |

The `is_replay()` distinction is critical for non-deterministic system processes (e.g., external API calls to OpenAI, Ollama, gRPC services). When `is_replay()` returns `true`, these processes use cached output from the original execution stored in `Produce::output_value` instead of re-executing the external call, ensuring deterministic block replay.

## Storage Architecture

```
HotStore (in-memory, DashMap)         <- Fast working set
    |
    | create_checkpoint()
    v
HistoryRepository                     <- Converts actions to trie mutations
    |
    v
RadixHistory (256-ary trie)           <- Immutable content-addressed trie
    |
    v
LMDB (3 databases per instance)       <- Durable persistence
  - {prefix}-history (trie nodes)
  - {prefix}-roots (checkpoint root hashes)
  - {prefix}-cold (leaf data: datums, continuations, joins)
```

**Hot Store** (`InMemHotStore`):
- `HotStoreState` contains 5 `DashMap`s: continuations, installed_continuations, data, joins, installed_joins
- Lock-free concurrent access
- Two caches: primary and history (lazy-loaded from cold store)

**Cold Store** (LMDB):
- `PersistedData` enum: `Data(DataLeaf)`, `Continuations(ContinuationsLeaf)`, `Joins(JoinsLeaf)`
- Each leaf stores serialized bytes

## Radix Trie

256-ary trie for checkpointing. Each node has 256 slots (one per byte value):

```rust
pub enum Item {
    EmptyItem,
    Leaf { prefix: ByteVector, value: ByteVector },   // value = 32-byte hash
    NodePtr { prefix: ByteVector, ptr: ByteVector },   // ptr = 32-byte child hash
}
pub type Node = Vec<Item>;  // Always 256 items
```

**Key paths**: `[PREFIX_DATUM|PREFIX_KONT|PREFIX_JOINS] + [hash_bytes]`

Compact binary encoding: item index (1B) + type/prefix-length (1B) + prefix (0-127B) + value/ptr (32B).

Root hash = `Blake2b256(encoded_root_node)`.

## Event Tracking

```rust
pub struct Produce {
    pub channel_hash: Blake2b256Hash,
    pub hash: Blake2b256Hash,
    pub persistent: bool,
    pub is_deterministic: bool,
    pub output_value: Vec<Vec<u8>>,  // For non-deterministic processes
    pub failed: bool,
}

pub struct Consume {
    pub channel_hashes: Vec<Blake2b256Hash>,
    pub hash: Blake2b256Hash,
    pub persistent: bool,
}

pub struct COMM {
    pub consume: Consume,
    pub produces: Vec<Produce>,
    pub peeks: BTreeSet<i32>,
    pub times_repeated: BTreeMap<Produce, i32>,
}
```

### Produce Identity Semantics

`Produce` has custom `PartialEq`/`Eq`, `Hash`, and `Ord`/`PartialOrd` implementations that compare and order by the `hash` field only. Metadata fields (`is_deterministic`, `output_value`, `failed`) are set after creation (e.g., via `mark_as_non_deterministic`) and must NOT affect identity, which would break replay event matching in `ReplayRSpace`. The `hash` field is a cryptographic hash of the channel, data, and persist flag computed at creation time.

### COMM Sort Order

`COMM::new` sorts `produce_refs` by `(channel_hash, hash, persistent)` for COMM event identity, which intentionally differs from `Produce::Ord` (hash-only). Do not replace with `.sort()`.

## Merger (Consensus Support)

`merger/merging_logic.rs` analyzes event logs for conflicts:

- `depends(target, source) -> bool` -- Checks if source events are prerequisites for target
- `are_conflicting(a, b) -> bool` -- Detects races on non-persistent operations
- `conflict_reason(a, b) -> Option<String>` -- Human-readable conflict explanation

`ChannelChange<A>` tracks `added` and `removed` items per channel for merge analysis.

## FFI Sub-crate: rspace_rhotypes

**Path**: `rspace++/libs/rspace_rhotypes/`

C FFI bindings for Scala JNA interop:

```
Type aliases:
  Channel = Par
  Pattern = BindPattern
  Data = ListParWithRandom
  Continuation = TaggedContinuation
```

Exports: `create_rspace()`, `produce()`, `consume()`, `install()`, `spatial_match_result()`, `reset_rspace()`, `free_space()`, etc.

Memory management: Rust allocates via `Box::leak()`, returns raw pointer. Scala must call `free_allocated_memory()`. `ALLOCATED_BYTES` atomic counter tracks leaked memory.

## Tests

Extensive test suites in `tests/`: `hot_store_spec.rs` (property-based), `storage_actions_test.rs`, `replay_rspace_tests.rs`, `reporting_rspace_tests.rs`, `export_import_tests.rs`, `install_test.rs`, plus `history/` subdirectory tests.

[← Back to overview](./README.md)
