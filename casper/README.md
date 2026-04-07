# Casper

CBC Casper consensus engine: block creation, validation, DAG management, safety oracle, and finalization.

## Building

```bash
cargo build --release -p casper
cargo build --profile dev -p casper   # debug mode
```

## Testing

```bash
cargo test -p casper                             # all tests
cargo test --release -p casper                   # release mode
cargo test --test <test_file_name>               # specific test file
cargo test --test <folder>::<test_file_name>      # specific test in folder
```

## Documentation

- [Consensus Protocol](../docs/casper/CONSENSUS_PROTOCOL.md) — End-to-end protocol walkthrough
- [Casper Module Overview](../docs/casper/README.md) — Block creation, validation, DAG, safety oracle, finalization
- [Byzantine Fault Tolerance](../docs/casper/BYZANTINE_FAULT_TOLERANCE.md) — BFT architecture, clique oracle, slashing
- [Synchrony Constraint](../docs/casper/SYNC_CONSTRAINT.md) — Synchrony constraint mechanism and configuration
