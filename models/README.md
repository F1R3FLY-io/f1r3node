# Models

Protobuf-generated types, domain structs (blocks, deploys, validators), Rholang AST, and sorted collections for the F1r3fly blockchain.

## Building

```bash
cargo build --release -p models
cargo build --profile dev -p models   # debug mode
```

## Testing

```bash
cargo test -p models
cargo test --release -p models
cargo test --test <test_file_name>               # specific test file
cargo test --test <folder>::<test_file_name>      # specific test in folder
```

## Documentation

- [Models Module Overview](../docs/models/README.md) — Protobuf types, Rholang AST, sorted collections
