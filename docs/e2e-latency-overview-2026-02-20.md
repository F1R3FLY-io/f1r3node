# E2E Latency Profile and Correctness-First Plan (2026-02-20)

## Latest commit verification (2026-02-20T13:59Z)
- Initialization correctness gate:
  - `./scripts/ci/check-casper-init-sla.sh docker/shard-with-autopropose.yml 180`
  - Result: `SLA PASSED` on `validator1/2/3` with `attempts=1, approved=1, transitions=1, time_to_running_count=1` per validator.
- External regression suite:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - Result: `12 passing`, `1 pending` (latest rerun complete).
- Conclusion:
  - latest commit remains correct on startup/genesis + end-to-end deploy/finality regression.

## Current end-to-end time breakdown (latest profile snapshot)
- Run: `./scripts/ci/profile-casper-latency.sh docker/shard-with-autopropose.yml /tmp/casper-latency-profile-latest`
- Time: `2026-02-20T13:59:17Z`
- Top-level timings:
  - `propose_total`: `avg=567.16ms`, `p95=1151ms`
  - `block_creator_total_create_block`: `avg=331.41ms`, `p95=914ms`
  - `block_creator_compute_deploys_checkpoint`: `avg=274.47ms`, `p95=862ms`
  - `finalizer_total`: `avg=58.21ms`, `p95=190ms`
  - `block_validation_mean_ms=280.34`, `block_replay_mean_ms=137.95`
  - `block_requests_retry_ratio=47.53`
- `compute_parents_post_state` (last 20m, validators 1-3):
  - path counts:
    - `single_parent=362`
    - `merged=299`
    - `descendant_fast_path=41`
    - `cache_hit=8`
  - merged-path hotspot:
    - `merged_total_ms`: `avg=265.88ms`, `p95=940ms`, `p99=1089ms`, `max=1111ms`
    - `merge_ms`: `avg=264.40ms`, `p95=938ms`, `p99=1087ms`, `max=1108ms`
    - `visible_blocks`: `avg=63.20`, `p95=163`, `p99=175`, `max=177`
- Interpretation:
  - checkpoint/merge dominates block creation cost and drives propose tail.
  - finalizer remains secondary relative to checkpoint + validation/replay.

## Latest status (after correctness patches)
- Build/deploy: `f1r3node:latest` retagged to `f1r3flyindustries/f1r3fly-rust-node:latest` and cluster recreated.
- External suite: `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - Result: `12 passing`, `1 pending` (completed around `2026-02-20T13:22Z`).
- Regressions fixed:
  - Removed proposer wedge from `Runtime error: NumberChannel must have singleton value.` by deterministic mergeable-channel sanitization.
  - Treated `InvalidRepeatDeploy` self-validation failure as recoverable propose skip (not hard bug failure).
- Verification evidence:
  - No `BugError (seqNum ...)` / `NumberChannel must have singleton value` matches in current `validator1` logs after rerun.

## E2E latency reduction plan (correctness-first)

### Phase 0: correctness gates first (must stay green)
- Keep these as non-negotiable acceptance checks for every optimization candidate:
  - `scripts/ci/check-casper-init-sla.sh` passes on all validators.
  - external suite `firefly-rholang-tests-finality-suite-v2/test.sh` passes (`12 passing`, `1 pending`).
  - no proposer wedge signatures in logs:
    - `NumberChannel must have singleton value`
    - `BugError (seqNum`
  - recoverable self-validation failures only (`propose_recoverable_self_validation_failures_total` observed, no hard-fail escalation).

### Phase 1: checkpoint/merge tail reduction (highest ROI)
- Target area:
  - `compute_parents_post_state` merged path (`merge_ms`, `visible_blocks` growth).
- Concrete actions:
  - bound merge input set using a deterministic visibility horizon (tip-distance cap) for merged parents.
  - add memoization for merged post-state by normalized parent-set key to increase effective `cache_hit`.
  - keep/extend descendant/single-parent fast paths; avoid shortcuts that drop rejection context.
- Success criteria:
  - `block_creator_compute_deploys_checkpoint p95` reduced by >=25%.
  - `propose_total p95` reduced by >=20%.
  - no increase in invalid/rejected deploy correctness regressions.

### Phase 2: validation/replay + dependency churn
- Target area:
  - `block_validation_mean_ms`, `block_replay_mean_ms`, `block_requests_retry_ratio`.
- Concrete actions:
  - extend dependency recovery throttling with in-flight coalescing (single requester per dependency hash).
  - avoid duplicate replay/validation on recently seen blocks in short windows.
  - add retry budget telemetry to identify top hashes driving retries.
- Success criteria:
  - `block_validation_mean_ms` <= 220ms
  - `block_replay_mean_ms` <= 110ms
  - `block_requests_retry_ratio` < 5 under benchmark load.

### Phase 3: propose/finalization loop polish
- Target area:
  - residual proposer races and finalizer tail spikes.
- Concrete actions:
  - tighten proposer scheduling/backoff to reduce transient race churn.
  - sample and cap high-volume warning logs in hot paths.
  - ensure finalizer wakeups are event-driven where possible (minimize periodic wake overhead).
- Success criteria:
  - `propose_total avg` <= 450ms and `p95` <= 900ms
  - `finalizer_total p95` <= 150ms
  - no regression in suite pass rate or init SLA.

## Clean benchmark after fixes
- Run: `./scripts/ci/run-latency-benchmark.sh docker/shard-with-autopropose.yml 120 /tmp/casper-latency-benchmark-current-H`
- Time: `2026-02-20T13:12:45Z`
- Load:
  - `deploy_attempts=54`, `deploy_success=54`, `deploy_failure=0`
  - `propose_ok=3`, `propose_fail=0`
- Profile:
  - `propose_total`: `n=250`, `avg=479.71ms`, `p50=465ms`, `p95=794ms`
  - `block_creator_total_create_block`: `n=251`, `avg=318.27ms`, `p50=312ms`, `p95=516ms`
  - `block_creator_compute_deploys_checkpoint`: `n=251`, `avg=263.24ms`, `p50=263ms`, `p95=463ms`
  - `finalizer_total`: `n=378`, `avg=6.53ms`, `p50=2ms`, `p95=32ms`
  - `block_validation_mean_ms=377.32`, `block_replay_mean_ms=214.17`
  - `block_requests_retry_ratio=6.71`

## `compute_parents_post_state` timing breakdown (instrumented)
- Source target: `f1r3fly.compute_parents_post_state.timing`
- Path mix (validators 1-3, benchmark window):
  - `path=merged`: `count=156`, `avg total=215.21ms`, `max=1877ms`
  - `path=cache_hit`: `count=18`, `avg total=0ms`
  - `path=single_parent`: `count=76`, `avg total=0ms`
- Dominant sub-phase on merged path:
  - `merge_ms`: `avg=214.56ms`, `p95=1204ms`, `p99=1808ms`, `max=1877ms`
  - `visible_blocks`: `avg=28.74`, `p95=71`, `max=78`

Interpretation:
- Remaining tail is primarily in multi-parent merge work (`compute_parents_post_state -> merge_ms`) and secondarily in block validation/replay.
- Finalization is not the bottleneck.

## Follow-up optimization experiments (same day)

### Experiment A (rejected): `same_post_state` fast-path in `compute_parents_post_state`
- Change tried:
  - skip merge when all selected parents had identical `post_state_hash`.
- Result:
  - path hit frequently (`same_post_state_fast_path` observed), but benchmark tails regressed.
  - `current-I` vs `current-H`:
    - `propose_total p95`: `794ms -> 1077ms` (worse)
    - `block_creator_compute_deploys_checkpoint p95`: `463ms -> 707ms` (worse)
- Decision:
  - reverted this optimization.
- Likely cause:
  - skipping merge also skipped rejection context (`rejected_deploys`), increasing downstream churn.

### Experiment B (kept): dependency recovery re-request cooldown
- File: `casper/src/rust/engine/block_retriever.rs`
- Change:
  - added per-hash cooldown for `recover_dependency` rebroadcast (`1000ms`) to avoid re-request storms.
- Benchmarks:
  - `current-H` (baseline after correctness fixes)
    - `propose_total avg/p95`: `479.71 / 794 ms`
    - `compute_deploys_checkpoint avg/p95`: `263.24 / 463 ms`
    - `block_validation_mean_ms`: `377.32`
    - `block_replay_mean_ms`: `214.17`
    - `block_requests_retry_ratio`: `6.71`
  - `current-J` (with cooldown)
    - `propose_total avg/p95`: `478.52 / 807 ms`
    - `compute_deploys_checkpoint avg/p95`: `238.54 / 479 ms`
    - `block_validation_mean_ms`: `282.46`
    - `block_replay_mean_ms`: `137.95`
    - `block_requests_retry_ratio`: `6.07`
- Interpretation:
  - meaningful improvement in replay/validation means and retry ratio.
  - propose/checkpoint p95 are near baseline (slightly noisier), but not regressed like Experiment A.

### New observability added
- `casper/src/rust/rholang/runtime.rs`
  - counter: `mergeable_channel_number_sanitized_total{source="casper_runtime"}`
- `casper/src/rust/blocks/proposer/proposer.rs`
  - counter: `propose_recoverable_self_validation_failures_total{source="casper_proposer",reason=...}`
- Example observed in `current-J`:
  - `propose_recoverable_self_validation_failures_total{reason="neglected_invalid_block"} = 9` on validator2.

### Experiment C (rejected): partial-eviction strategy for `parents_post_state_cache`
- File tried: `casper/src/rust/util/rholang/runtime_manager.rs`
- Change tried:
  - replace full cache clear-at-limit with partial random-sample eviction to preserve working set.
- Validation conditions:
  - rebuilt image (`node/Dockerfile`), retagged to `f1r3flyindustries/f1r3fly-rust-node:latest`, full compose recreate.
  - correctness suite rerun passed (`12 passing`, `1 pending`).
- Benchmarks:
  - baseline (`current-J`):
    - `propose_total avg/p95`: `478.52 / 807 ms`
    - `compute_deploys_checkpoint avg/p95`: `238.54 / 479 ms`
    - `finalizer_total avg/p95`: `21.09 / 69 ms`
    - `block_validation_mean_ms`: `282.46`
    - `block_replay_mean_ms`: `137.95`
    - `block_requests_retry_ratio`: `6.07`
  - candidate run 1 (`current-L`):
    - `propose_total avg/p95`: `724.23 / 1820 ms`
    - `compute_deploys_checkpoint avg/p95`: `245.40 / 517 ms`
    - `finalizer_total avg/p95`: `552.09 / 675 ms`
    - `block_validation_mean_ms`: `560.93`
    - `block_replay_mean_ms`: `565.63`
    - `block_requests_retry_ratio`: `9.34`
  - candidate run 2 (`current-M`):
    - `propose_total avg/p95`: `794.31 / 2142 ms`
    - `compute_deploys_checkpoint avg/p95`: `239.05 / 512 ms`
    - `finalizer_total avg/p95`: `563.83 / 681 ms`
    - `block_validation_mean_ms`: `654.35`
    - `block_replay_mean_ms`: `837.23`
    - `block_requests_retry_ratio`: `59.40`
- Additional trace observation (last 15m) showed improved `compute_parents_post_state` path mix (`cache_hit=66`, merged p95 `153ms`), but this did not translate to better E2E latency.
- Decision:
  - rejected and reverted (do not keep this patch).

## Latest verification run (current branch + initialization fix)
- Test: `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
- Result: `12 passing`, `1 pending` (completed around `2026-02-20T09:58Z`)
- Startup correctness:
  - `bootstrap` transitioned to running at `2026-02-20T09:54:09Z`
  - `validator1/2/3` transitioned to running at `2026-02-20T09:56:12Z`
  - Validators initially logged `No approved block available ... Will request again in 10 seconds`, then recovered and entered `Running`

Interpretation: genesis ceremony / Casper init no longer deadlocks; validators eventually initialize and process deploys.

## Revalidation after CI + init-metrics hardening
- Test: `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
- Result: `12 passing`, `1 pending` (completed around `2026-02-20T10:41Z`)
- Observed behavior:
  - Core deploy tests continued to pass with sub-second deploy submission times in this run.
  - Negative syntax/unbound-name tests failed deploy as expected.
  - Existing suite warnings remained non-fatal (`No blockHash found for deploy`, optional strict assertions disabled).

Interpretation: latest correctness-focused changes still pass the same E2E regression suite and did not regress functional behavior.

## Clean-state revalidation after regression triage
- Test: `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
- Result: `12 passing`, `1 pending` (completed around `2026-02-20T12:14Z`)
- Run conditions:
  - fresh compose reset (`down -v`, `up -d`)
  - rust-client tip cache cleared before run
- Observation:
  - suite passed, but warnings remained (`No blockHash found for deploy`, intermittent propose nudge failures).
  - these warnings are treated as non-fatal by the current external suite settings.

Correctness hardening added:
- New script: `scripts/ci/clear-rust-client-tip-cache.sh`
- `scripts/ci/run-latency-benchmark.sh` now calls the cache-clear step before load.
- Reason: stale rust-client monotonic tip cache after chain reset can force deploys into future-only validity windows and create false negatives in correctness/perf checks.

Additional correctness hardening (`2026-02-20`):
- File: `casper/src/rust/api/block_api.rs`
- Change:
  - treat recoverable propose outcomes (`NoNewDeploys`, transient proposer race currently surfaced as `InternalDeployError`) as non-error API responses in both `create_block` and `get_propose_result`.
- Effect:
  - rust-client `propose` no longer fails hard on these recoverable cases; it returns success text such as `Propose skipped due to transient proposal race (...)`.
  - clean-state external suite still passes (`12 passing`, `1 pending`), with fewer disruptive propose-nudge failures.

Operational pitfall discovered:
- `docker/shard-with-autopropose.yml` uses image tag `f1r3flyindustries/f1r3fly-rust-node:latest`.
- Local builds to `f1r3node:latest` are ignored unless retagged.
- For local validation, retag before compose restart:
  - `docker tag f1r3node:latest f1r3flyindustries/f1r3fly-rust-node:latest`

## Post-fix benchmark snapshot (patched image + corrected harness/image flow)
- Run: `./scripts/ci/run-latency-benchmark.sh docker/shard-with-autopropose.yml 60 /tmp/casper-latency-benchmark-current-F`
- Time: `2026-02-20T12:38:30Z`
- Load:
  - `deploy_attempts=28`, `deploy_success=28`, `deploy_failure=0`
  - `propose_ok=1`, `propose_fail=0`
- Profile:
  - `propose_total`: avg=`942.63ms`, p50=`445ms`, p95=`3586ms`
  - `block_creator_total_create_block`: avg=`658.76ms`, p50=`223ms`, p95=`2905ms`
  - `block_creator_compute_deploys_checkpoint`: avg=`567.79ms`, p50=`138ms`, p95=`2728ms`
  - `block_validation_mean_ms`: `208.75ms`
  - `block_requests_retry_ratio`: `0.20`

Directional comparison vs earlier clean baseline (`baseline-new-E`):
- `propose_total avg`: `1565.17ms -> 942.63ms` (`~39.8%` lower)
- `block_creator_total_create_block avg`: `1018.65ms -> 658.76ms` (`~35.3%` lower)
- `block_creator_compute_deploys_checkpoint avg`: `938.79ms -> 567.79ms` (`~39.5%` lower)
- `propose_fail`: `2 -> 0` in this 60s run

Interpretation:
- Correctness-first fixes reduced operational failure modes (stale tip cache, recoverable propose errors surfacing as hard failures).
- Tail latency (`p95`) remains high and bursty; next optimizations should focus on tail trimming in `compute_deploys_checkpoint` and proposal race/validation churn.

## Automated profiling command (Phase B baseline)
- Script: `scripts/ci/profile-casper-latency.sh`
- Example:
  - `./scripts/ci/profile-casper-latency.sh docker/shard-with-autopropose.yml /tmp/casper-latency-profile`
- Output:
  - `summary.txt` with propose/block-creator/finalizer log-derived p50/p95 and replay/validation/retriever metrics from Prometheus.

## Controlled A/B benchmark harness
- Script: `scripts/ci/run-latency-benchmark.sh`
- Usage:
  - `./scripts/ci/run-latency-benchmark.sh docker/shard-with-autopropose.yml 120 /tmp/casper-latency-benchmark-A`
- What it does:
  - runs init SLA gate,
  - applies fixed-duration deploy load,
  - captures profile summary via `profile-casper-latency.sh`.
- A/B workflow:
  1. run on baseline commit (`...benchmark-A`)
  2. run on candidate commit (`...benchmark-B`)
  3. compare `profile/summary.txt` and `load-summary.txt`.

Latest baseline snapshot (`2026-02-20T11:11Z`, validators 1-3):
- `propose_total`: n=`6458`, avg=`282.27ms`, p50=`241ms`, p95=`495ms`
- `block_creator_total_create_block`: avg=`190.80ms`, p50=`161ms`, p95=`384ms`
- `block_creator_compute_deploys_checkpoint`: avg=`132.52ms`, p50=`102ms`, p95=`323ms`
- `finalizer_total`: n=`1851`, avg=`164.72ms`, p50=`61ms`, p95=`433ms`
- `block_validation_mean_ms`: `292.39ms`
- `block_replay_mean_ms`: `135.53ms`
- `block_requests_retry_ratio`: `1177.21` (`3,938,948 / 3,346`)

## First optimization attempt (low risk, deploy-heavy path)
- File: `casper/src/rust/blocks/proposer/block_creator.rs`
- Changes:
  - rewrote `prepare_user_deploys` from multi-pass filtering to single-pass classification.
  - bounded per-reason filtered deploy warning logs to sampled output (cap) instead of logging every filtered deploy.
  - removed redundant deploy-in-scope filtering when constructing `all_deploys`.
- Rationale:
  - reduces CPU and log amplification under large deploy pools; preserves filtering semantics and expired-deploy cleanup.

Post-rebuild profile snapshot (`2026-02-20T11:26Z`, validators 1-3):
- `propose_total`: n=`477`, avg=`284.13ms`, p50=`230ms`, p95=`493ms`
- `block_creator_total_create_block`: avg=`179.29ms`, p50=`147ms`, p95=`363ms`
- `block_creator_compute_deploys_checkpoint`: avg=`126.12ms`, p50=`94ms`, p95=`305ms`
- `block_validation_mean_ms`: `258.54ms`
- `block_replay_mean_ms`: `146.26ms`
- `block_requests_retry_ratio`: `35.55` (`17,810 / 501`)

Note:
- This profile was taken after rebuild/restart and during a run that showed repeated heartbeat `internal deploy error` propose failures; treat it as directional only, not a final A/B conclusion.

First controlled run with harness (`2026-02-20T11:29Z`, duration 60s):
- Load: `deploy_attempts=29`, `deploy_success=29`, `deploy_failure=0`, `deploy_attempt_rate_per_s=0.48`
- `propose_total`: avg=`273.06ms`, p50=`227ms`, p95=`476ms`
- `block_creator_total_create_block`: avg=`175.49ms`, p50=`148ms`, p95=`362ms`
- `block_creator_compute_deploys_checkpoint`: avg=`121.56ms`, p50=`94ms`, p95=`303ms`

Controlled comparison (`baseline-A` vs restored candidate `block_creator.rs`, `2026-02-20T11:39Z` vs `11:48Z`):
- Baseline (`/tmp/casper-latency-benchmark-baseline-A`, harness fully passed):
  - `propose_total`: avg=`433.26ms`, p95=`851ms`
  - `block_creator_total_create_block`: avg=`271.75ms`, p95=`483ms`
  - `block_creator_compute_deploys_checkpoint`: avg=`218.61ms`, p95=`428ms`
  - `block_validation_mean_ms`: `263.85ms`
  - `block_replay_mean_ms`: `119.62ms`
  - `block_requests_retry_ratio`: `0.31`
- Candidate (`/tmp/casper-latency-benchmark-candidate-B`, manual load/profile only):
  - `propose_total`: avg=`1680.30ms`, p95=`2578ms`
  - `block_creator_total_create_block`: avg=`1195.64ms`, p95=`1650ms`
  - `block_creator_compute_deploys_checkpoint`: avg=`1116.57ms`, p95=`1557ms`
  - `block_validation_mean_ms`: `945.27ms`
  - `block_replay_mean_ms`: `151.72ms`
  - `block_requests_retry_ratio`: `7.24`
- Relative delta (candidate vs baseline):
  - `propose_total avg`: `+287.8%` (`3.88x`)
  - `block_creator_total_create_block avg`: `+340.1%` (`4.40x`)
  - `block_creator_compute_deploys_checkpoint avg`: `+410.8%` (`5.11x`)
  - `block_validation_mean_ms`: `+258.3%` (`3.58x`)
  - `block_replay_mean_ms`: `+26.8%` (`1.27x`)
  - `block_requests_retry_ratio`: `+2235%` (`23.35x`)

Correctness caveat for candidate run:
- `scripts/ci/check-casper-init-sla.sh` failed because required metrics `casper_init_attempts`, `casper_init_approved_block_received`, and `casper_init_time_to_running_count` were absent/zero.
- Only `casper_init_transition_to_running` was exported on all validators.
- Because correctness gate failed, this candidate performance sample is valid for diagnosis but not release acceptance.

Fair clean-to-clean rerun after SLA checker fix (`2026-02-20T11:53Z` vs `11:54Z`):
- Candidate clean (`/tmp/casper-latency-benchmark-candidate-clean-D`):
  - `propose_total`: avg=`3604.76ms`, p95=`6557ms`
  - `block_creator_total_create_block`: avg=`2547.55ms`, p95=`4971ms`
  - `block_creator_compute_deploys_checkpoint`: avg=`2463.41ms`, p95=`4851ms`
  - `block_validation_mean_ms`: `1536.10ms`
  - `block_replay_mean_ms`: `149.34ms`
  - `block_requests_retry_ratio`: `0.16`
- Baseline clean (`/tmp/casper-latency-benchmark-baseline-new-E`):
  - `propose_total`: avg=`1565.17ms`, p95=`5374ms`
  - `block_creator_total_create_block`: avg=`1018.65ms`, p95=`2914ms`
  - `block_creator_compute_deploys_checkpoint`: avg=`938.79ms`, p95=`2815ms`
  - `block_validation_mean_ms`: `2052.08ms`
  - `block_replay_mean_ms`: `148.69ms`
  - `block_requests_retry_ratio`: `0.29`
- Delta (candidate vs baseline, same methodology):
  - `propose_total avg`: `+130.3%` (`2.30x`)
  - `block_creator_total_create_block avg`: `+150.1%` (`2.50x`)
  - `block_creator_compute_deploys_checkpoint avg`: `+162.4%` (`2.62x`)

Conclusion:
- The candidate `block_creator.rs` variant is a clear propose/create-block regression under matched clean-run conditions and should not be adopted.

## Where time is spent

### 1) Propose loop (`f1r3fly.propose.timing`, 3 validators, last 30m, n=218)
- `propose_core_ms`: avg `429.54`, p50 `302`, p95 `1065`
- `total_ms`: avg `431.53`, p50 `305`, p95 `1067`
- `snapshot_ms`: low and not a dominant cost

### 2) Block creation inside propose (`f1r3fly.block_creator.timing`, n=218)
- `total_create_block_ms`: avg `267.44`, p50 `182`, p95 `639`
- `compute_deploys_checkpoint_ms`: avg `212.21`, p50 `123`, p95 `580`
- `compute_bonds_ms`: avg `53.83`, p50 `50`, p95 `73`
- `package_ms`/`sign_ms`: ~`0ms`

Derived:
- Non-block-creation propose overhead (`propose_total - block_total`) avg: `~164ms`
- Dominant propose hot path remains `compute_deploys_checkpoint`.

### 3) Block replay/validation loop (Prometheus aggregated over validators)
- `block_processing_stage_replay_time`: mean `0.1546s` (`154.6ms`)
- `block_validation_time`: mean `0.3491s` (`349.1ms`)
- `block_processing_stage_validation_setup_time`: mean `0.0411s` (`41.1ms`)
- `block_processing_stage_storage_time`: mean `0.0055s` (`5.5ms`)

Replay sub-phases:
- `block_replay_phase_user_deploys_time`: `242.9ms`
- `block_replay_phase_system_deploys_time`: `90.2ms`
- `block_replay_phase_create_checkpoint_time`: `23.4ms`
- `block_replay_phase_reset_time`: `4.9ms`

Validation step metrics (instrumented in ms, histogram buckets in seconds):
- `block_validation_step_checkpoint_time`: mean `178.8ms`
- `block_validation_step_bonds_cache_time`: mean `100.4ms`
- `block_validation_step_block_summary_time`: mean `16.7ms`

### 4) Finalization loop (`f1r3fly.finalizer.timing`, n=246)
- `total_ms`: avg `5.51`, p50 `4`, p95 `19`
- `clique_ms`: avg `4.64`, p50 `3`, p95 `16`
- `found_new_lfb=true`: avg `6.07ms` (n=41)
- `found_new_lfb=false`: avg `5.40ms` (n=205)

Interpretation: finalizer is no longer a dominant cost in this run.

### 5) Dependency/retrieval pressure
- Log events in last 30m:
  - `Missing dependencies`: `314` total
  - `Recovery re-request issued for dependency`: `442` total
- Retriever counters:
  - v1: `block_requests_total=103`, `block_requests_retries=651`
  - v2: `106`, `1185`
  - v3: `103`, `626`

Interpretation: dependency retry churn is still high and likely contributes to inclusion/finality tail latency.

## Bottleneck ranking (current successful run)
1. Block validation/replay on receiving nodes (`~154ms replay + ~349ms validation` mean).
2. Propose path p95 spikes from `compute_deploys_checkpoint` (up to `~580ms`) and resulting `propose_total` p95 (`~1067ms`).
3. Dependency retry/re-request churn.
4. Finalizer (now small, low priority for optimization).

## Correctness-first, then latency reduction plan

### Phase A: Correctness hardening (first priority)
1. Genesis approval handoff robustness:
   - Add explicit startup metric/event: `casper_init_attempt`, `approved_block_received`, `transition_to_running`.
   - Keep retry loop but expose retry count and elapsed time until running per validator.
2. Init race prevention:
   - Preserve current `transition_to_initializing -> init()` immediate call path.
   - Add regression test for delayed bootstrap approved-block availability.
3. Gating checks:
   - CI smoke: fail if validator doesn’t enter `Running` within fixed SLA (for example 3 minutes in local compose).

Exit criteria:
- 0 stuck validators across repeated clean-cluster starts.
- Deterministic transition-to-running timing envelope.

### Phase B: E2E latency reduction (after Phase A passes)
1. Propose/block creation:
   - Optimize `compute_deploys_checkpoint` (cache common path for low deploy counts).
   - Reuse bonds/cache by post-state where safe.
2. Replay/validation:
   - Target checkpoint + bonds-cache validation steps first.
   - Reduce repeated lookups in user deploy replay path.
3. Dependency retrieval:
   - Per-hash in-flight dedup, bounded retries with jitter/backoff.
   - Batch missing-dependency requests.
4. Measurement hygiene:
   - Fix unit mismatch for `block_validation_step_*`.
   - Emit per-deploy lifecycle timestamps (`accepted -> included -> validated -> finalized`).

Target KPIs:
- `propose_total_ms` p95 < `700ms`
- `block_validation_time` mean < `250ms`
- `block_requests_retries / block_requests_total` reduced by at least `60%`
- `submit -> finalized` p95 reduced by at least `30%` on the same suite.
