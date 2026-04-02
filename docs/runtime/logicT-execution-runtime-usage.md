# LogicT Execution Runtime (Experimental Branch) - Functionality and Usage

## Purpose

This document describes what was added in the `experimental/logicT-execution-runtime` branch and how to use the split post-match continuation runtime.

The branch implements the LogicT runtime phases for:

- explicit post-match execution state
- bounded step execution
- continuation persistence and resume
- executor/public and bridge continuation flows
- operational hardening (caps, expiry, metrics, replay/recovery tests)

## Current rollout status

- Split post-match execution is feature-gated and configured via `SplitPostMatchConfig`.
- Default config is still conservative (`enabled: false`) so migration remains opt-in.

## Functionality added

### 1. Explicit machine state for post-match continuation execution

The reducer now models post-match work as serializable machine state:

- dispatch payload and replay metadata
- control state for dispatch/reconsume/reproduce/peek-restore follow-ups

This is the core prerequisite for deterministic suspension and later resume.

### 2. Step-based reducer execution

A bounded stepping API exists internally:

- `reduce_step(exec_state, gas_limit) -> StepResult`
- explicit terminal status (`Completed` or `Suspended`)
- explicit consumed gas and effect emission marker

Stepwise execution is validated against full execution for covered fragments.

### 3. Continuation persistence layer

A dedicated continuation store persists suspended state with canonical identity and integrity:

- handle (`origin` + `nonce`)
- serialized machine state
- gas limit per step
- funding policy and visibility
- optional bounty and TTL-derived expiry
- subtype metadata (`Standard` or `Bridge`)
- lifecycle status (`Active`, `Completed`, `Expired`, `Failed`)
- optimistic versioning and state root

The store supports create/load/update/complete/expire/fail lifecycle transitions and stale-version rejection.

### 4. Split execution integration on consume-match path

When enabled:

1. consume-match succeeds
2. runtime executes one bounded step
3. immediate effects are committed
4. remainder is persisted as continuation state

When disabled, runtime stays on full legacy post-match execution.

### 5. Continuation execution path

Suspended continuations can be resumed by handle/version with bounded gas:

- `execute_continuation`
- stale version is rejected
- terminal states are not resumable
- partial progress is persisted

Public-executor path adds policy checks:

- `execute_public_continuation`
- requires `Public` visibility + `ExecutorPays` funding policy

### 6. Funding policy and queue visibility

Supported policy knobs:

- `FundingPolicy::{ProducerOnly, ExecutorPays}`
- `ContinuationVisibility::{Private, Public}`
- optional `bounty`
- optional `ttl_epochs` with deterministic epoch-based expiry

Queue visibility/filtering:

- `list_public_continuation_queue(epoch)` returns executor-eligible active continuations

### 7. Bridge continuation subtype and scheduler hooks

Bridge-specific subtype metadata is modeled in the same continuation runtime:

- lane ID, source/target domains, message nonce
- finality phase
- deadline epoch
- reward policy
- rescue epoch

Scheduler/ops hooks:

- `list_bridge_continuation_queue_by_deadline(epoch)`
- `list_bridge_continuation_queue_by_reward(epoch)`
- `list_bridge_rescue_candidates(epoch)`
- `update_bridge_finality_phase(handle, expected_version, phase)`
- `execute_bridge_continuation(...)`
- `execute_next_bridge_by_deadline(...)`
- `execute_next_bridge_by_reward(...)`

### 8. Operational hardening

Added protections and observability:

- continuation state size cap: `128 KiB`
- continuation branch fan-out cap: `128`
- deterministic expiry sweeps (`set_continuation_epoch`, `expire_due`)
- snapshot/restore support in continuation store for crash-recovery validation
- runtime metrics for creation, resume paths, gas per step, steps-to-completion, expiry, and bridge queue/execution counters

## How to use

### A. Enable split post-match runtime

Configure reducer:

```rust
use rholang::rust::interpreter::reduce::SplitPostMatchConfig;
use rholang::rust::interpreter::storage::continuation_store::{
    ContinuationSubtype, ContinuationVisibility, FundingPolicy,
};

reducer.set_split_post_match_config(SplitPostMatchConfig {
    enabled: true,
    initial_step_gas_limit: i64::MAX,
    continuation_funding_policy: FundingPolicy::ExecutorPays, // or ProducerOnly
    continuation_visibility: ContinuationVisibility::Public,   // or Private
    continuation_bounty: Some(42),                             // optional
    continuation_ttl_epochs: Some(10),                         // optional
    continuation_subtype: ContinuationSubtype::Standard,       // or Bridge(...)
});
```

What happens next:

- ordinary receive/send matching can persist suspended continuation remainder automatically.
- immediate effects from the first bounded step are committed before suspension.

### B. Inspect and resume standard continuations

```rust
let handles = reducer.list_continuation_handles();
let handle = handles[0].clone();
let persisted = reducer.load_continuation(&handle)?;

let result = reducer
    .execute_continuation(&handle, persisted.version, i64::MAX)
    .await?;
```

Important semantics:

- pass the latest `version`; stale versions are rejected.
- `result.status` is `Completed` or `Suspended`.
- if suspended, call again with latest version to continue progress.

### C. Use public executor queue (ExecutorPays + Public only)

```rust
let queue = reducer.list_public_continuation_queue(epoch)?;
if let Some(item) = queue.first() {
    let _res = reducer
        .execute_public_continuation(&item.handle, item.version, i64::MAX, epoch)
        .await?;
}
```

### D. Drive expiry sweeps

```rust
let expired_handles = reducer.set_continuation_epoch(next_epoch)?;
```

This advances deterministic epoch and expires due active continuations with TTL.

### E. Configure and execute bridge continuations

Use bridge subtype in config:

```rust
use rholang::rust::interpreter::storage::continuation_store::{
    BridgeContinuationMetadata, BridgeFinalityPhase, BridgeRewardPolicy, ContinuationSubtype,
};

let bridge_subtype = ContinuationSubtype::Bridge(BridgeContinuationMetadata {
    lane_id: "lane-1".to_string(),
    source_domain: "eth-mainnet".to_string(),
    target_domain: "f1r3node".to_string(),
    message_nonce: 7,
    finality_phase: BridgeFinalityPhase::Pending,
    deadline_epoch: Some(100),
    reward_policy: BridgeRewardPolicy::Fixed(25),
    rescue_epoch: Some(120),
});
```

Queue and execution flows:

```rust
let by_deadline = reducer.list_bridge_continuation_queue_by_deadline(epoch)?;
let by_reward = reducer.list_bridge_continuation_queue_by_reward(epoch)?;
let rescue_due = reducer.list_bridge_rescue_candidates(epoch)?;

if let Some(item) = by_deadline.first() {
    let _res = reducer
        .execute_bridge_continuation(&item.handle, item.version, i64::MAX, epoch)
        .await?;
}

let _next_deadline = reducer.execute_next_bridge_by_deadline(epoch, i64::MAX).await?;
let _next_reward = reducer.execute_next_bridge_by_reward(epoch, i64::MAX).await?;
```

Finality phase updates are version-checked:

```rust
let _updated = reducer.update_bridge_finality_phase(
    &handle,
    expected_version,
    BridgeFinalityPhase::Retrying,
)?;
```

## Metrics emitted

Core continuation metrics include:

- `rholang.continuation.created`
- `rholang.continuation.state_bytes`
- `rholang.continuation.resumed`
- `rholang.continuation.resume.completed`
- `rholang.continuation.resume.suspended`
- `rholang.continuation.resume.failed`
- `rholang.continuation.gas_per_step`
- `rholang.continuation.steps_to_completion`
- `rholang.continuation.expired`

Bridge-related metrics include queue request counters and execute request counter:

- `rholang.bridge.queue.deadline_requests`
- `rholang.bridge.queue.reward_requests`
- `rholang.bridge.queue.rescue_requests`
- `rholang.bridge.execute.requests`

## Verification and test entrypoints

Primary verification:

```bash
./scripts/verify-runtime.sh
```

Targeted runtime examples:

```bash
cargo test -p rholang split_post_match_enabled_should_commit_dispatch_effects_and_persist_remainder -- --nocapture
cargo test -p rholang execute_continuation_should_resume_to_completion_and_reject_terminal_resume -- --nocapture
cargo test -p rholang split_replay_runs_should_be_deterministic_for_multiple_cases -- --nocapture
cargo test -p rholang continuation_replay_retries_should_be_idempotent_and_non_mutating -- --nocapture
cargo test -p rholang bridge_followup_scheduler_hooks_should_support_phase_updates_and_next_execution -- --nocapture
```

## Notes and constraints

- Deterministic replay is a hard invariant; continuation behavior must remain deterministic across validators.
- Continuation status/version transitions are optimistic and explicit; external callers must carry and refresh version.
- The split path is still opt-in via `SplitPostMatchConfig.enabled`.
