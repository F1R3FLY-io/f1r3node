# TODO

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
