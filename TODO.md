# TODO

## Bug: Validator permanently stuck if initial bootstrap connection fails

When a validator starts before the boot node's P2P listener is ready, the initial TCP connection to boot fails with "Connection refused." The validator discovers other validators via Kademlia, connects to them, but **never retries the connection to boot**. Since the genesis ceremony only broadcasts the UnapprovedBlock from boot, the validator never receives the genesis candidate and loops forever on "Casper engine present but Casper not initialized yet."

**Root cause (confirmed via logs):**
1. Validator1 starts at T+0, tries to connect to boot — "Connection refused (os error 111)"
2. Boot starts listening at T+1, computes genesis, broadcasts UnapprovedBlock at T+10
3. Validator2 and validator3 are connected to boot, receive genesis, approve it
4. Boot gets 2/2 required signatures, transitions to Running
5. Validator2 and validator3 request ApprovedBlock from boot, initialize successfully
6. **Validator1 never reconnects to boot** — stuck permanently in "Casper not initialized yet" loop
7. Validator1 connects to validator3 via discovery at T+20 but this doesn't help — genesis ceremony only comes from boot

**Impact:** ~50% of integration test runs fail due to startup timeout. All 5 containers report healthy (Docker healthcheck passes) but the shard is non-functional.

**Fix locations:**
- `comm/src/rust/rp/connect.rs:325` — connection failure handling (should retry bootstrap peer)
- `casper/src/rust/engine/casper_launch.rs:770` — genesis validator mode (should retry fetching approved block from any peer, not just boot)
- `node/src/rust/runtime/node_runtime.rs:572` — "Waiting for first connection" (should have retry logic for bootstrap)

**Fix options:**
1. **Retry bootstrap connection**: After initial connection failure, retry boot peer with exponential backoff
2. **Request ApprovedBlock from any peer**: If validator connects to other validators that already have the approved block, request it from them instead of waiting for boot's broadcast
3. **Increase boot startup priority**: Use Docker `depends_on` with healthcheck to ensure boot is fully ready before validators start (integration test workaround only)

- Location: `node/src/rust/runtime/setup.rs:677` (the warning loop)
- Reproduction: `F1R3FLY_NODE_IMAGE=... pytest test_bridge_admin.py -v -s --keep-running` — fails ~50% of runs

## Review: `compute_parents_post_state_regression_spec.rs`

This test file needs review and cleanup:

### Merge scope test (`visible_blocks_should_not_grow_unbounded_with_dag_depth`)

The test at line 564 is marked `#[ignore]` with a comment saying `visible_blocks` grows to ~60 at round 20 instead of staying bounded at ~3 per round. The comment says the current code includes everything back to `ancestor_min_block_number` (effectively genesis when `max_parent_depth` is large). Need to:
- [ ] Update test for async ISpace — currently uses manual thread spawn with sync runtime, needs migration to `#[tokio::test(flavor = "multi_thread")]`
- [ ] Verify if this is still the case after recent merge optimizations (PR #473 conflict detection caching)
- [ ] Determine if this is a correctness issue (too many blocks in scope = slower merge) or a bug (wrong blocks included = incorrect merge results)
- [ ] Un-ignore and fix, or document why the current behavior is acceptable

### Stack size env var (`F1R3_COMPUTE_PARENTS_REGRESSION_STACK_BYTES`)

Three tests spawn threads with `F1R3_COMPUTE_PARENTS_REGRESSION_STACK_BYTES` (default 64MB) for deep recursion during genesis evaluation. This env var handling was removed from the rest of the codebase (replaced by `StackGrowingFuture` with `stacker::maybe_grow()`). Need to:
- [ ] Update all three tests to use `#[tokio::test(flavor = "multi_thread")]` with async ISpace
- [ ] Determine if 64MB is still necessary or if `StackGrowingFuture` covers these code paths in tests
- [ ] If still needed, document the env var in the test file and CI config
- [ ] If not needed, replace the manual thread spawn with standard `#[tokio::test]` using `StackGrowingFuture`

## Regression Spec Genesis Needs Wallets

`compute_parents_post_state_regression_spec.rs` creates its own minimal genesis without wallets. Tests that need deploy execution (bridge merge repro, deploy-level merge tests) can't run because precharge fails — deployer has no vault. Add wallets to the regression spec's `Genesis` struct or provide a shared `genesis_context()` variant that returns a genesis compatible with the DAG/block store infrastructure.

## DAG Snapshot Finalization State Race

`KeyValueDagRepresentation` is designed as an immutable snapshot, but `is_finalized` breaks this by falling back to a live LMDB read via the shared `block_metadata_index`. This creates a hybrid where some state is snapshot-consistent and some is live, leading to race conditions between `record_finalized` (which updates in-memory `DagState` before persisting to LMDB) and concurrent `is_finalized` calls.

**Current workaround:** The LMDB persist in `record_finalized` now runs inside the `dag_state` write lock to prevent the window between in-memory and persistent state updates. This is correct but holds the write lock slightly longer.

**Long-term fix:** Make `KeyValueDagRepresentation` a true self-contained snapshot:
1. `finalized_blocks_set` becomes authoritative — remove the LMDB fallback from `is_finalized`
2. `lookup` returns metadata from the snapshot, not from live LMDB
3. `record_finalized` updates `DagState` and LMDB atomically, and new snapshots capture the consistent state
4. No shared mutable reads from snapshots — eliminates the entire class of race conditions

**Cost:** Snapshots become larger (carry metadata in memory). The existing `finalized_blocks_set` pruning (cap 50k, prune to 25k) would need similar bounding for metadata. This is a DAG architecture refactor.

**Files:** `block-storage/src/rust/dag/block_dag_key_value_storage.rs` (`is_finalized`, `get_representation`), `block-storage/src/rust/dag/block_metadata_store.rs` (`record_finalized`)

## Flaky Test: `approve_block_protocol_test::should_send_approved_block_message_to_peers_once_approved_block_is_created`

This genesis ceremony test fails intermittently (~1 in 3 runs). Root cause: all 300+ casper tests share a single LMDB database via `SHARED_LMDB_LOCK` (see `casper/tests/helper/block_dag_storage_fixture.rs` lines 1-27). The global mutex serializes tests within a single run, but the shared LMDB state can accumulate artifacts across tests. The approve_block_protocol tests are particularly sensitive because they operate on genesis state.

**Reproduction:** Run `cargo test --package casper --test mod --release` multiple times — fails ~33% of runs.

**Fix options:**
1. Per-test LMDB scope IDs (already used via `generate_scope_id()` but the underlying LMDB env is shared)
2. Isolate genesis ceremony tests into a separate test binary with their own LMDB env
3. Add explicit LMDB cleanup between test runs

**File:** `casper/tests/helper/block_dag_storage_fixture.rs`, `casper/tests/engine/approve_block_protocol_test.rs`

## RSpace Lock Granularity

`event_log` and `produce_counter` in `rspace.rs` and `replay_rspace.rs` are always accessed together in `log_produce`/`log_consume` but use separate `Mutex` locks. Combining them into a single lock would reduce lock acquisitions per RSpace operation. Low priority — the critical sections are short and uncontended under per-channel locks, so overhead is ~5ns per extra lock.

## RSpace Per-Channel Lock Contention

Popular channels (e.g. vault channel during many transfers) could become a bottleneck under the per-channel `tokio::sync::Mutex` locking scheme. Inherent to the design and matches Scala behavior. Monitor under high-load testing.

## RSpace DashMap Lock Growth

`phase_a_locks` and `phase_b_locks` in `rspace.rs` and `replay_rspace.rs` grow unboundedly as new channels are created. Each entry is ~64 bytes (`Arc<tokio::sync::Mutex<()>>`). Fine for typical workloads but could consume memory under adversarial workloads creating millions of unique channels. Consider periodic eviction of locks for channels no longer in the hot store.

## Sync/Async Lock Mixing in RSpace

The hot store uses blocking `std::sync::RwLock` while channel locks use `tokio::sync::Mutex`. Safe as long as the sync lock is never held across an `.await` point (currently true — only short hash lookups). Document this invariant and consider adding a lint or code comment to prevent future regressions.

## uint256 Primitive Type

Add a fixed-width `uint256` type to the Rholang language as a new primitive, following the pattern established by PR #425 (Float, BigInt, BigRat, FixedPoint).

### Parser / Grammar (rholang-rs)
- [ ] New literal syntax (e.g. `42u256`)
- [ ] Grammar rule in rholang-rs parser

### Protobuf
- [ ] `GUint256` message in `RhoTypes.proto`

### Normalizer
- [ ] `Uint256Literal` → `GUint256` proto expression
- [ ] Range enforcement: `0 <= value < 2^256`

### Reducer
- [ ] Arithmetic: `+`, `-`, `*`, `/`, `%` with overflow/underflow checks
- [ ] Comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=`
- [ ] Unary negation rejected (unsigned)

### Supporting
- [ ] Pretty printer rendering
- [ ] Spatial matcher `has_locally_free`
- [ ] Cost functions proportional to operand size
- [ ] Sorter and score tree support for deterministic ordering

### Tests
- [ ] Unit tests (direct AST construction, reducer in isolation)
- [ ] End-to-end eval tests (Rholang source through parse -> normalize -> reduce -> rspace)
- [ ] Overflow/underflow boundary tests
- [ ] Cross-type error tests (uint256 + BigInt -> error)

### Related Future Work

Items not covered by the numeric types implementation (PR #425) that may be needed:

- [ ] API exposure — typed BigInt/uint256 in HTTP/gRPC responses
- [ ] Wallet/token balances using BigInt — currently hardwired to i64 throughout (PoS, vault, transfer, APIs)
- [ ] Cross-type coercion rules (e.g. BigInt + Float promotion)
- [ ] Standard library functions (pow, floor, sqrt, etc.)
