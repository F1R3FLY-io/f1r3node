# Rholang

Rholang is a concurrent programming language, with a focus on message-passing and formally modeled by the ρ-calculus, a reflective, higher-order extension of the π-calculus. It is designed to be used to implement protocols and "smart contracts" on a general-purpose blockchain, but could be used in other settings as well.

## Command Line Interface

The Rholang CLI provides a command-line interface for executing Rholang programs and compiling them to various formats.

### Building the CLI

```bash
cargo build --release --bin rholang-cli
```

The binary will be available at `target/release/rholang-cli`

### Running the CLI

```bash
# Evaluate a Rholang file
./target/release/rholang-cli rholang/examples/stdout.rho

# Start the REPL (interactive mode)
./target/release/rholang-cli

# Compile to binary protobuf format
./target/release/rholang-cli --binary rholang/examples/stdout.rho

# Compile to text protobuf format
./target/release/rholang-cli --text rholang/examples/stdout.rho

# Evaluate quietly (no storage output)
./target/release/rholang-cli --quiet rholang/examples/stdout.rho

# Show only unmatched sends
./target/release/rholang-cli --unmatched-sends-only rholang/examples/stdout.rho
```

### Options

- `--binary` - outputs binary protobuf serialization
- `--text` - outputs textual protobuf serialization
- `--quiet` - don't print tuplespace after evaluation
- `--unmatched-sends-only` - only print unmatched sends after evaluation
- `--data-dir <DATA_DIR>` - Path to data directory
- `--map-size <MAP_SIZE>` - Map size (in bytes) [default: 1073741824]
- `-h, --help` - Print help
- `-V, --version` - Print version

## Building the Library

```bash
cargo build --release -p rholang
cargo build --profile dev -p rholang   # debug mode
```

## Testing

```bash
cargo test -p rholang
cargo test --release -p rholang
cargo test --test <test_file_name>               # specific test file
cargo test --test <folder>::<test_file_name>      # specific test in folder
```

## Known Limitations

- Guarded patterns for channel receive (e.g. `for (@x <- y if x > 0)`) don't work
- 0-arity send and receive is currently broken
- Match cases are not pre-evaluated (matching `7 + 8` as a pattern doesn't work — match against `15` instead)

Working examples are included in the `examples/` directory and the [Rholang tutorial](../docs/rholang/rholangtut.md).

## Documentation

- [Rholang Module Overview](../docs/rholang/README.md) — Interpreter, reducer, cost accounting, system processes
- [Rholang Tutorial](../docs/rholang/rholangtut.md) — Language tutorial
- [Pattern Matching Tutorial](../docs/rholang/rholangmatchingtut.md) — Pattern matching guide
- [Ollama Integration](../docs/rholang/ollama.md) — Local LLM integration via Ollama
- [Reference Documentation](./reference_doc/README.md) — Language reference by topic
