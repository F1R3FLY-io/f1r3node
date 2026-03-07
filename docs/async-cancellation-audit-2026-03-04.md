## Async Cancellation Audit (2026-03-04)

This document records the async cancellation / lock-across-await issues that were fixed,
in strict sequence with a successful test gate before proceeding to the next issue.

### 1) Inbound transport worker lifecycle: detached branch risk

- Problem:
  - `tokio::select!` in `comm/src/rust/transport/grpc_transport_receiver.rs` only awaited one worker.
  - The sibling `JoinHandle` could be dropped/detached, and cleanup aborted only the combined task.
- Fix:
  - Replaced the single combined handle with explicit per-worker handles.
  - `MessageBuffers` now stores both `tell` and `blob` worker task handles.
  - Stale-peer cleanup now aborts both worker tasks directly.
- Validation:
  - `cargo test -p comm "transport::transport_layer_spec::concurrent_streams_to_same_peer_should_all_succeed" -- --nocapture`
  - Result: `1 passed, 0 failed`.

### 2) LMDB dir manager lock held across `.await`

- Problem:
  - `rspace++/src/rspace/shared/lmdb_dir_store_manager.rs` held `managers_state` lock while awaiting:
    - manager receiver in `store()`
    - manager shutdown in `shutdown()`
- Fix:
  - `store()`: remove receiver under lock, drop lock, then await receiver.
  - `shutdown()`: drain all receivers under lock into a vec, drop lock, then await and shutdown each manager.
- Validation:
  - `cargo test -p casper batch2::lmdb_key_value_store_spec:: -- --nocapture`
  - Result: `3 passed, 0 failed`.

### 3) Proposer timeout canceled in-flight propose future

- Problem:
  - `node/src/rust/instances/proposer_instance.rs` used `tokio::time::timeout` directly on propose future.
  - Timeout dropped the future, canceling in-flight propose work mid-operation.
- Fix:
  - Removed timeout-based cancellation path.
  - Propose now awaits directly to completion under the proposer lock.
  - Cleaned imports/variable naming for warning-free local changes.
- Validation:
  - `cargo test -p node proposer_instance::tests:: -- --nocapture`
  - Result: `2 passed, 0 failed`.

### 4) Initialization used `try_join!` (early sibling cancellation)

- Problem:
  - `casper/src/rust/engine/initializing.rs` used `tokio::try_join!` for block-request and tuple-space futures.
  - On one branch error, the other branch would be canceled immediately.
- Fix:
  - Replaced with `tokio::join!`, then propagated errors after both branches completed.
- Validation:
  - `cargo test -p casper engine::initializing_spec:: -- --nocapture`
  - Result: `3 passed, 0 failed`.

### 5) System process service lock held across remote awaits

- Problem:
  - `rholang/src/rust/interpreter/system_processes.rs` held outer `Mutex` guards while awaiting OpenAI/Ollama requests.
- Fix:
  - Clone service state under lock, release lock immediately, then perform async remote call.
  - Applied to: `gpt4`, `dalle3`, `text_to_audio`, `ollama_chat`, `ollama_generate`, `ollama_models`.
- Validation:
  - `cargo test -p rholang non_deterministic_processes_spec:: -- --nocapture`
  - Result: `8 passed, 0 failed`.

