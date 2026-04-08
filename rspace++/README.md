# rspace++

Rust implementation of the F1r3fly tuple space engine. Provides produce/consume pattern matching, LMDB-backed trie history, and checkpointing for the Rholang interpreter.

See [rspace/README.md](../rspace/README.md) for the original Scala implementation and conceptual overview.

## Building

```bash
cargo build --release -p rspace_plus_plus
cargo build --profile dev -p rspace_plus_plus   # debug mode
```

## Testing

```bash
cargo test -p rspace_plus_plus
cargo test --release -p rspace_plus_plus
cargo test --test <test_file_name>               # specific test file
cargo test --test <folder>::<test_file_name>      # specific test in folder
```

## Documentation

- [RSpace Module Overview](../docs/rspace/README.md) — Tuple space engine, produce/consume matching, trie history
