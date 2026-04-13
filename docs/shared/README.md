> Last updated: 2026-03-23

# Crate: shared

**Path**: `shared/`

Foundation types and storage abstractions used by every other crate.

## Type Aliases

```rust
pub type ByteVector = Vec<u8>;
pub type ByteBuffer = Vec<u8>;
pub type Byte = u8;
pub type ByteString = Vec<u8>;
pub type BitSet = Vec<u8>;
pub type BitVector = Vec<u8>;
```

## Key-Value Store Abstraction

The KV store layer is the persistence backbone of the entire system.

**`KeyValueStore` trait** -- Low-level byte-oriented interface:
- `get(keys: Vec<Vec<u8>>) -> Vec<Option<Vec<u8>>>`
- `put(kv_pairs: Vec<(Vec<u8>, Vec<u8>)>)`
- `delete(keys: Vec<Vec<u8>>) -> usize`
- `iterate(fn(key, value))`
- `to_map() -> BTreeMap`
- Convenience: `get_one`, `put_one`, `put_if_absent`, `contains`

**`KeyValueTypedStore<K, V>` trait** -- Generic typed interface:
- `get`, `put`, `delete`, `collect`, `to_map`
- Impl: `KeyValueTypedStoreImpl<K, V>` wraps `KeyValueStore` with `bincode` serialization

**`LmdbKeyValueStore`** -- LMDB backend via `heed` crate:
- Memory-mapped B+ tree
- `SerdeBincode<Vec<u8>>` encoding

**Error type**: `KvStoreError` (KeyNotFound, IoError, SerializationError, InvalidArgument, LockError)

## Other Modules

- `compression.rs` -- LZ4 compression/decompression
- `f1r3fly_event.rs` / `f1r3fly_events.rs` -- Event data structures
- `grpc_server.rs` -- gRPC server utilities
- `metrics_constants.rs` / `metrics_semaphore.rs` -- Observability
- `env.rs` -- Environment variable parsing utilities (`var_parsed`, `var_or`, `var_bool`)
- `dag/dag_ops.rs` -- DAG utility operations
- `hashable_set.rs` -- HashSet utilities
- Serde helpers: `serde_bytes`, `serde_vec_bytes`, `serde_btreemap_bytes_i64`, `serde_hex_bytes`, `serde_hex_vec_u8`, `serde_always_equal_bitset`

## Tests

Unit tests in: `compression.rs`, `f1r3fly_events.rs`, `metrics_semaphore.rs`, `list_ops.rs`, `recent_hash_filter.rs`, `dag_ops.rs`, `grpc_server.rs`.

**See also:** [shared/ crate](../../shared/)

[← Back to docs index](../README.md)
