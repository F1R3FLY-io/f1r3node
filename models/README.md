# Models

Common data model types common for the RChain blockchain.

### Prerequisites

* [sbt](http://www.scala-sbt.org/download.html)

### Building

```
sbt compile
```

### Testing

```
sbt test
```

Testing with coverage:

```
sbt clean coverage test
```

Generating a coverage report:

```
sbt coverageReport
```

The HTML version of the generated report is located at:

 `./target/scala-<version>/scoverage-report/index.html`
 
 ## Rust

Parts of this directory are ported to Rust.

### Building

To build the `models` Rust library, run `cargo build --release -p models`
  - `cargo build --profile dev -p models` will build the library in debug mode

### Testing

To run all tests: `cargo test`

Run all tests in release mode: `cargo test --release`

To run specific test file: `cargo test --test <test_file_name>`

To run specific test in specific folder: `cargo test --test <test_folder_name>::<test_file_name>`

## Documentation

- [Models Module Overview](../docs/models/README.md) — Protobuf types, Rholang AST, sorted collections
