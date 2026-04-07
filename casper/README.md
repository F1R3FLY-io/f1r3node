# RChain Node

## slowcooker

Set of generative stateful tests that take a long time.

command:
`sbt casper/slowcooker:test`

### The idea

- have a set of commands that are valid actions on RChain
- have mechanics that generate and run these commands
- gather results and validate those

# Casper

## Rust

Parts of this directory are ported to Rust.

### Building

To build the `casper` Rust library, run `cargo build --release -p casper`
  - `cargo build --profile dev -p casper` will build the library in debug mode

### Testing

To run all tests: `cargo test`

Run all tests in release mode: `cargo test --release`

To run specific test file: `cargo test --test <test_file_name>`

To run specific test in specific folder: `cargo test --test <test_folder_name>::<test_file_name>`

## Documentation

- [Casper Module Overview](../docs/casper/README.md) — Block creation, validation, DAG, safety oracle, finalization
- [Byzantine Fault Tolerance](../docs/casper/BYZANTINE_FAULT_TOLERANCE.md) — BFT architecture, clique oracle, slashing
- [Synchrony Constraint](../docs/casper/SYNC_CONSTRAINT.md) — Synchrony constraint mechanism and configuration