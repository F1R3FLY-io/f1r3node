# Reified RSpaces Examples

This directory contains pedagogical examples demonstrating the new Reified RSpaces
system and its integration with Rholang syntax.

## Reified RSpaces Examples

These examples demonstrate the new Reified RSpaces system using **pure Rholang files**
that can be executed end-to-end (parsing through interpretation).

### Collection Types

| File | Description |
|------|-------------|
| `reified_rspaces/queue_fifo.rho` | Queue: First-In-First-Out message processing |
| `reified_rspaces/stack_lifo.rho` | Stack: Last-In-First-Out (undo/redo pattern) |
| `reified_rspaces/set_idempotent.rho` | Set: Idempotent storage (deduplication) |
| `reified_rspaces/cell_exactly_once.rho` | Cell: Exactly-once write semantics |
| `reified_rspaces/priority_queue.rho` | PriorityQueue: Priority-based ordering |
| `reified_rspaces/vectordb_similarity.rho` | VectorDB: Basic similarity concepts |
| `reified_rspaces/vectordb_semantic_search.rho` | VectorDB: Native `~` and `~>` similarity operators |

### Storage Types

| File | Description |
|------|-------------|
| `reified_rspaces/pathmap_prefix_aggregation.rho` | PathMap: Hierarchical prefix channel routing |

### Advanced Features

| File | Description |
|------|-------------|
| `reified_rspaces/multi_space_isolation.rho` | Multiple isolated namespaces |
| `reified_rspaces/speculative_execution.rho` | Checkpoints and rollback |
| `reified_rspaces/gas_metering.rho` | Phlogiston resource limits |
| `reified_rspaces/temp_vs_default_qualifier.rho` | Space qualifier semantics |

## Running Reified RSpaces Examples

```bash
# Execute a Reified RSpaces example
rholang --execute reified_rspaces/queue_fifo.rho

# Or use the interpreter directly
cargo run --bin rholang -- reified_rspaces/set_idempotent.rho
```

## New Rholang Syntax

### Factory Pattern

Create spaces using factory URNs:

```rholang
new QueueFactory(`rho:space:queue:hashmap:default`), myQueue in {
  QueueFactory!({}, *myQueue) |
  use myQueue {
    // Operations target myQueue
  }
}
```

### Factory URN Format

```
rho:space:{collection}:{storage}:{qualifier}
```

**Collections**:
- `bag` - Unordered multiset (default)
- `queue` - FIFO ordering
- `stack` - LIFO ordering
- `set` - Idempotent (no duplicates)
- `cell` - Exactly-once write
- `priorityqueue` - Priority-based ordering
- `vectordb` - Similarity matching

**Storage**:
- `hashmap` - Hash table (default, exact channel matching)
- `pathmap` - Hierarchical paths (prefix aggregation semantics)
- `array` - Fixed-size indexed
- `vector` - Growable indexed
- `hashset` - Set-optimized

**Qualifiers**:
- `default` - Persistent, mobile, concurrent
- `temp` - Ephemeral (cleared on checkpoint)
- `seq` - Non-mobile, sequential access

### Use Blocks

Switch the active space context:

```rholang
use myQueue {
  // All operations here target myQueue
  task!(1) |
  for (@t <- task) { process!(t) }
}
```

### Priority Sends

Send with explicit priority (for PriorityQueue):

```rholang
tasks!!priority(3, {"urgent": true})  // Priority 3 (high)
tasks!!priority(0, {"urgent": false}) // Priority 0 (low)
```

### Similarity Matching

Query by embedding similarity (for VectorDB) using native `~` operator:

```rholang
// Store document with embedding (embeddings use 0-100 integer scale)
// Note: 'docs' is a name (channel) created with 'new docs in { ... }'
docs!([0, "Cat article", [90, 5, 10, 20]]) |

// Implicit threshold (uses space default, typically 0.8)
// Pattern syntax: @[id, title, emb] - bare vars inside quoted list
for (@[id, title, emb] <- docs ~ [85, 10, 5, 15]) {
  stdout!(["Found similar:", title])
} |

// Explicit threshold (80 = 0.80 cosine similarity)
for (@[id, title, emb] <- docs ~> 80 ~ [85, 10, 5, 15]) {
  stdout!(["High confidence match:", title])
}
```

The `~` operator triggers cosine similarity matching against stored embeddings.
The `~>` operator sets an explicit threshold (0-100 scale maps to 0.0-1.0).

See `reified_rspaces/vectordb_semantic_search.rho` for comprehensive examples.

### PathMap Prefix Aggregation

With PathMap storage, data sent on a child path is visible at ALL prefix paths:

```rholang
new PathMapFactory(`rho:space:pathmap:hashmap:default`), space in {
  PathMapFactory!({}, *space) |
  use space {
    // Send on child path [0, 1, 2]
    @[0, 1, 2]!("hello") |

    // Receive on prefix [0, 1] - THIS WORKS with PathMap!
    // (Would BLOCK forever with HashMap)
    for (@msg <- @[0, 1]) {
      stdout!("Received: " ++ msg)  // "Received: hello"
    }
  }
}
```

Use cases: hierarchical event routing, logging systems, monitoring dashboards.

See `reified_rspaces/pathmap_prefix_aggregation.rho` for a comprehensive tutorial.

### Checkpoints

Create and restore checkpoints:

```rholang
@"checkpoint"!("soft", *cpId) |
for (@id <- cpId) {
  // Speculative work...
  @"rollback"!(*id)  // OR @"commit"!(*id)
}
```

### Gas Metering

Query and manage gas balance:

```rholang
@"phlogiston:balance"!(*balance) |
for (@b <- balance) {
  if (b > 100) {
    expensiveOperation!()
  } else {
    stdout!("Low on gas!")
  }
}
```

## Collection Semantics Summary

| Collection | Ordering | Duplicates | Capacity |
|------------|----------|------------|----------|
| Bag | Unordered | Yes | Unbounded |
| Queue | FIFO | Yes | Unbounded |
| Stack | LIFO | Yes | Unbounded |
| Set | Unordered | No | Unbounded |
| Cell | N/A | No (1 max) | 1 |
| PriorityQueue | By priority | Yes | Unbounded |
| VectorDB | By similarity | Yes | Unbounded |

## Qualifier Semantics Summary

| Qualifier | Persists | Mobile | Concurrent |
|-----------|----------|--------|------------|
| Default | Yes | Yes | Yes |
| Temp | No | Yes | Yes |
| Seq | No | No | No |

## Storage Semantics Summary

| Storage | Channel Matching | Prefix Aggregation | Use Case |
|---------|------------------|-------------------|----------|
| HashMap | Exact only | No | General purpose |
| PathMap | Exact + prefix | Yes | Hierarchical routing |
| Array | By index | No | Fixed-size buffers |
| Vector | By index | No | Growable indexed |
| HashSet | Exact only | No | Presence tracking |

## Formal Verification

These examples correspond to properties verified in the Rocq formal proofs:

- `GenericRSpace.v` - Core invariants (NoPendingMatch, Produce/Consume exclusivity)
- `Collections/DataCollection.v` - Collection semantics (FIFO, LIFO, idempotent)
- `Phlogiston.v` - Gas accounting invariants
- `Checkpoint.v` - Checkpoint/replay determinism
- `Safety/Properties.v` - Qualifier semantics
