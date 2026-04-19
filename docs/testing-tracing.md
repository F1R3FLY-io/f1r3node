# Tracing in Tests

## Casper Tests

The casper test suite (`casper/tests/mod.rs`) initializes tracing via `init_logger()` with a default filter of `ERROR`. This means `tracing::info!`, `tracing::debug!`, and `tracing::warn!` calls in tests are **silent by default**.

To see tracing output, set `RUST_LOG`:

```bash
# All info-level output
RUST_LOG=info cargo test -p casper --release --test mod -- my_test --nocapture

# Targeted (less noise)
RUST_LOG=casper=info cargo test -p casper --release --test mod -- my_test --nocapture

# Debug level for a specific module
RUST_LOG=casper::rust::rholang=debug cargo test -p casper --release --test mod -- my_test --nocapture
```

The `--nocapture` flag is required — without it, pytest/cargo captures stderr and the tracing output is hidden.

## Rholang Tests

Rholang tests do NOT have a shared `init_logger()`. To use tracing in rholang tests, add `tracing_subscriber` initialization in the test function:

```rust
let _ = tracing_subscriber::fmt()
    .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
    .with_target(true)
    .try_init();
```

Then run with `RUST_LOG=info` or `RUST_LOG=replay_debug=debug`.

## Common Gotchas

- **Release mode does NOT strip tracing** — the `tracing` crate in `Cargo.toml` has no `max_level_*` features set, so all levels compile in
- **`try_init()` silently fails** if another subscriber is already set (e.g., another test in the same process initialized first). Use `try_init()` (not `init()`) to avoid panics
- **JSON format** — casper's `init_logger` uses `.json()` format. Log lines look like `{"timestamp":"...","level":"INFO","message":"..."}`, not plain text
