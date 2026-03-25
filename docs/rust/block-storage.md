> Last updated: 2026-03-23

# Crate: block-storage

**Path**: `block-storage/`

Block persistence, DAG state management, casper buffer, and deploy indexing.

## Block Store

**`KeyValueBlockStore`**:
- Two stores: blocks and approved_blocks
- LZ4 compression with varint length prefix (Java-compatible)
- `get(hash)`, `put(hash, block)`, `contains(hash)`
- `get_approved_block()` / `put_approved_block()` -- Singleton approved block

## DAG Storage

**`KeyValueDagRepresentation`** -- Immutable DAG snapshot (O(1) clones via `imbl`):
- `dag_set`, `latest_messages_map`, `child_map`, `height_map`
- `invalid_blocks_set`, `last_finalized_block`, `finalized_blocks_set`
- Queries: `lookup`, `children`, `parents_unsafe`, `latest_messages`, `topo_sort`, `main_parent_chain`, `ancestors`, `descendants`, `non_finalized_blocks`

**`BlockDagKeyValueStorage`** -- Live mutable DAG with global `Mutex`:
- `get_representation()` -- Atomic snapshot (acquires lock)
- `insert(block, invalid, approved)` -- Add block with metadata updates
- `record_directly_finalized(hash, effect)` -- Async finalization

**`BlockMetadataStore`** -- Per-block metadata with in-memory DAG state:
- Uses `imbl` persistent collections (HashSet, OrdMap, HashMap) for structural sharing
- `add(metadata)`, `record_finalized(directly, indirectly)`, `contains(hash)`
- **DAG metadata caches**: In-memory indices avoid repeated LMDB deserialization on hot paths:
  - `block_number_map`: BlockHash → block_num
  - `main_parent_map`: BlockHash → parent BlockHash
  - `self_justification_map`: BlockHash → justification BlockHash
  - `finalized_block_set`: Bounded HashSet (cap 50k, prunes to 25k) of finalized blocks

## Casper Buffer

**`CasperBufferKeyValueStorage`** -- Tracks unprocessed block dependencies:
- Two `DashMap`s: child_to_parent, parent_to_child adjacency lists
- `BlockDependencyDag` (doubly-linked DAG) rebuilt from persistent store on startup
- `add_relation(parent, child)`, `put_pendant(block)`, `remove(hash)`, `get_pendants()`

## Finality Storage

**`LastFinalizedStorage` trait**: `put(hash)`, `get()`, `get_or_else(default)`
- `LastFinalizedKeyValueStorage` (persistent)
- `LastFinalizedMemoryStorage` (in-memory)

## Deploy Storage

**`KeyValueDeployStorage`** -- Stores `Signed<DeployData>` indexed by deploy signature. Methods: `add()`, `remove()`, `read_all()`, `non_empty()`. Deploy index maps deploy signature to block hash.

## Tests

`block_dag_storage_test.rs` (proptest integration), `key_value_block_store.rs` (proptest unit), `casper_buffer_key_value_storage.rs` (tokio async), `doubly_linked_dag_operations.rs` (DAG unit tests).

[← Back to overview](./README.md)
