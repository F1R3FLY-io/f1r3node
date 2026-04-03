# Heartbeat Stale-LFB Recovery (2026-03-02)

## Problem

Validators could stall with `last-finalized-block` (LFB) stuck even though heartbeat was running.
The stall happened in a low-lag window:

- `frontier_chase_max_lag = 0` (hardcoded) blocks frontier-chasing once a validator is ahead.
- Leader lag recovery only started at `lag > pending_deploy_max_lag` (hardcoded default `20`).
- For stale LFB with lag in `1..20`, no validator proposed, so finality could not advance.

## Correct behavior

When LFB is stale and lag is non-zero, the shard must still produce recovery blocks so clique
agreement can progress and LFB can move forward, while avoiding proposal spam.

## Fix

Implemented in:

- `node/src/rust/instances/heartbeat_proposer.rs`

Behavior change:

- Keep existing stale-LFB recovery and frontier-chase throttling.
- Add a leader-only fallback when regular stale recovery is throttled:
  - Conditions:
    - LFB is stale
    - no pending deploys
    - lag `> 0`
    - current validator is deterministic recovery leader
    - regular stale recovery is currently throttled
- Keep existing high-lag leader recovery (`lag > threshold`) unchanged.

This removes the low-lag dead zone without weakening finality safety rules.

## Safety/Liveness rationale

- Safety unchanged: finalization still depends on finalizer + clique fault-tolerance checks.
- Liveness improved: at least one validator can keep recovery progress moving when stale-LFB
  throttles would otherwise stop all proposals.

