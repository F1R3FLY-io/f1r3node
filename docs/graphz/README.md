> Last updated: 2026-03-23

# Crate: graphz (Visualization)

**Path**: `graphz/`

Async Graphviz DOT language generator for DAG visualization.

### API

```rust
pub trait GraphSerializer: Send + Sync {
    async fn push(&self, str: &str, suffix: &str) -> Result<(), GraphzError>;
}
```

Implementations: `StringSerializer`, `ListSerializer` (sends via tokio oneshot), `FileSerializer`.

**`Graphz` builder**:
- `edge(src, dst, style, arrow_head, constraint)`
- `node(name, shape, style, color, label)`
- `close()`

**Top-level functions**:
- `apply(name, type, serializer, ...)` -- Create graph
- `subgraph(name, type, serializer, ...)` -- Nested subgraph

Enums: `GraphType` (Graph, DiGraph), `GraphShape` (Circle, DoubleCircle, Box, PlainText, Msquare, Record), `GraphStyle` (Solid, Bold, Filled, Invis, Dotted, Dashed), `GraphRank`, `GraphRankDir`, `GraphArrowType`.

### Tests

12 async tests in `tests/graphz_tests.rs` covering simple graphs, digraphs, edges, nodes, subgraphs, and performance (1000-edge test).

[<- Back to overview](./README.md)
