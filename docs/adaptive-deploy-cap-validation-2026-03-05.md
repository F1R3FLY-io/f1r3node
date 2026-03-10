# Adaptive Per-Block Deploy Capping (Validation, 2026-03-05)

## Summary

Implemented adaptive per-block user deploy capping in Rust block creation so latency stays under a 1s target without hard-coding a fixed cap of `2`.

The cap now self-tunes based on observed block creation time:

- When `total_create_block_ms` is above target, cap decreases proportionally.
- When cap is saturated and latency is comfortably below target, cap increases.
- Bounded by configured min/max.

## Code changes

Primary implementation:

- `casper/src/rust/blocks/proposer/block_creator.rs`

Key additions:

- `PreparedUserDeploys` return type for selection metadata (`effective_cap`, `cap_hit`)
- Adaptive cap state (`current_cap`, EMA of create-block latency)
- Control logic:
  - `effective_user_deploys_per_block_cap()`
  - `next_adaptive_cap(...)`
  - `update_adaptive_user_deploy_cap(...)`
- Feedback wiring from measured `total_create_block_ms` after block creation

Unit tests added in the same file:

- `adaptive_cap_reduces_when_latency_exceeds_target`
- `adaptive_cap_increases_when_capped_and_headroom_exists`
- `adaptive_cap_does_not_increase_when_not_capped`
- `adaptive_cap_respects_min_and_max_bounds`

## Runtime configuration

New env knobs:

- `F1R3_ADAPTIVE_DEPLOY_CAP_ENABLED` (default: enabled)
- `F1R3_ADAPTIVE_DEPLOY_CAP_TARGET_MS` (default: `1000`)
- `F1R3_ADAPTIVE_DEPLOY_CAP_MIN` (default: `1`)

Existing max bound remains:

- `F1R3_MAX_USER_DEPLOYS_PER_BLOCK` (default: `32`)

## Validation performed

### 1) Unit validation

Command:

```bash
cargo test -p casper adaptive_cap -- --nocapture
```

Result:

- Passed: `4`
- Failed: `0`

### 2) Runtime latency validation on patched image

Patched image build:

```bash
docker build -t f1r3fly-rust-node:local -f node/Dockerfile .
```

Cluster:

- Compose file: `docker/shard-with-autopropose.local.yml`
- Image: `f1r3fly-rust-node:local`

Benchmark command:

```bash
./scripts/ci/run-latency-benchmark-mode.sh strict-ci docker/shard-with-autopropose.local.yml 60 /tmp/casper-latency-benchmark-adaptive-20260305T083909Z
```

Key results:

- `propose_total`: avg `312.01ms`, p95 `577ms`
- `block_creator_total_create_block`: avg `351.35ms`, p95 `498ms`
- `block_creator_compute_deploys_checkpoint`: avg `322.79ms`, p95 `464ms`
- `checkpoint_compute_state`: avg `318.54ms`, p95 `460ms`

Target check:

- p95 for propose and block creation both below `1000ms`: **PASS**

Runtime log evidence (validator1/2/3):

- `prepare_user_deploys_ms=..., user_deploys_count=..., user_deploy_cap=32, user_deploy_cap_hit=false`

Interpretation:

- Effective cap in this run was `32` (configured max), confirming no hard-coded fixed cap of `2`.
- This load profile did not saturate cap (`cap_hit=false`), so no downward adaptation was required.

Artifacts:

- `/tmp/casper-latency-benchmark-adaptive-20260305T083909Z/load-summary.txt`
- `/tmp/casper-latency-benchmark-adaptive-20260305T083909Z/profile/summary.txt`

### 3) Short memory soak validation

Command:

```bash
./scripts/ci/run-validator-leak-soak.sh docker/shard-with-autopropose.local.yml 120 10 /tmp/casper-validator-leak-soak-adaptive-20260305T084036Z
```

Observed RSS deltas over ~115s:

- validator1: `-0.72 MiB`
- validator2: `-5.45 MiB`
- validator3: `+0.84 MiB`

Interpretation:

- No positive unbounded growth trend in this short run.
- Finalizer activity completed normally (`finalizer_run_timed_out=0` for all validators).

Artifacts:

- `/tmp/casper-validator-leak-soak-adaptive-20260305T084036Z/summary.txt`
- `/tmp/casper-validator-leak-soak-adaptive-20260305T084036Z/finalizer-summary.txt`

## Conclusion

The adaptive capping change is implemented, unit-tested, and validated on a runtime built from the patched workspace:

- Meets sub-1s latency target in this strict 60s run without forcing a static cap of `2`.
- No memory regression signal in the short soak.
