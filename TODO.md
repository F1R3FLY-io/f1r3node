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

## RSpace Lock Granularity

`event_log` and `produce_counter` in `rspace.rs` and `replay_rspace.rs` are always accessed together in `log_produce`/`log_consume` but use separate `Mutex` locks. Combining them into a single lock would reduce lock acquisitions per RSpace operation. Low priority — the critical sections are short and uncontended under per-channel locks, so overhead is ~5ns per extra lock.
