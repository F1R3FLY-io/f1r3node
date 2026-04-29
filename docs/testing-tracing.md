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

## Tracing Rholang Execution via Block Report

`RUST_LOG` shows what the runtime did. It does *not* show what happened inside the tuplespace — which produces matched which consumes, which sends were orphaned, which contracts received messages. For that, use the block report API.

When a deploy reports `success: true` but the expected effect didn't materialize (a continuation never fires, a balance didn't change, a `deployId` channel returns empty), the block report is the canonical way to inspect what actually happened.

### API

gRPC: `getEventByHash` (`ReportQuery` / `EventInfoResponse`)
HTTP: `POST /api/trace`

```json
{
  "blockHash": "<hex>",
  "forceReplay": true
}
```

Returns the full event log for the block: every produce, consume, and COMM event for every deploy and system deploy, with channel hashes, random states, and originating deploy ID. Read-only nodes serve this endpoint; validator nodes do not.

See [node API reference](node/api-reference.md) for the full schema.

### Detecting orphan sends

The signature of an orphan send (a send whose receiver was never installed):

- A `produce` event on channel hash X
- No `comm` event whose `produces` include that produce's `random_state`

A short Python correlation against the trace JSON:

```python
def find_orphan_produces(trace):
    comm_consumed = set()
    for evt in trace["events"]:
        if evt["type"] == "comm":
            for p in evt["produces"]:
                comm_consumed.add(p["random_state"])
    return [
        evt for evt in trace["events"]
        if evt["type"] == "produce"
        and evt["random_state"] not in comm_consumed
    ]
```

For multi-hop call chains (A -> B -> C -> D), follow the response channels: each `for` waits on a specific channel hash, which can be correlated against produces by hash. Channel hashes are Blake2b prefixes — you'll need to map them back to source by replaying the contract's structure.

### Limitations

This is a manual workflow today: trace dumps are several MB of JSON and require correlation scripts. There is no first-class "show me the orphan sends in this block" tool. A planned improvement is to surface orphan continuations directly in deploy results — see the "How to make this easier next time" section of the bridge-deposit-orphan investigation report (in the `system-integration` repo, `docs/`).
