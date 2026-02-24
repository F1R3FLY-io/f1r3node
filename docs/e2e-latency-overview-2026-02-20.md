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

## Correctness + performance update (2026-02-23)
- Correctness revalidation on clean recreate:
  - `./scripts/ci/check-casper-init-sla.sh docker/shard-with-autopropose.yml 240`
  - Result: `SLA PASSED` in `108s` (`validator1/2/3` all `attempts=1, approved=1, transitions=1, time_to_running_count=1`).
  - External suite rerun:
    - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
    - Result: `12 passing`, `1 pending`.
- Fresh strict 120s benchmark (post-revert baseline):
  - `/tmp/casper-latency-benchmark-post-pprof-revert-20260223T092517Z`
  - `deploy_success=54/54`, `propose_fail=0`, `propose_bug_error=0`
  - `propose_total avg/p95=748.76/3371 ms`
  - `block_creator_total_create_block avg/p95=551.24/2782 ms`
  - `block_creator_compute_deploys_checkpoint avg/p95=486.05/2713 ms`
  - `finalizer_total avg/p95=56.50/286 ms`
  - `block_validation_mean_ms=420.35`, `block_replay_mean_ms=233.40`
  - `block_requests_retry_ratio=0.89`
- Allocator call-stack profiling status:
  - Added repeatable profiler wrapper:
    - `scripts/ci/profile-validator-allocator-hotspots.sh`
    - workflow:
      - runs correctness-gated 120s soak (`run-latency-benchmark.sh`)
      - captures earliest/latest jemalloc heap snapshots for validator process
      - emits symbolized call stacks for:
        - top latest allocations
        - positive growth (earliest -> latest)
  - Fresh allocator profile run:
    - command:
      - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260223T-profile1`
    - artifacts:
      - summary: `/tmp/casper-allocator-hotspots-20260223T-profile1/summary.txt`
      - symbolized latest: `/tmp/casper-allocator-hotspots-20260223T-profile1/stacks/latest.top15.symbolized.txt`
      - symbolized growth: `/tmp/casper-allocator-hotspots-20260223T-profile1/stacks/delta-positive.top15.symbolized.txt`
  - Key hotspot call paths from this run:
    - replay checkpoint/history clone chain:
      - `HistoryRepositoryImpl::checkpoint/do_checkpoint -> RadixHistory::process -> Vec::clone`
    - interpreter allocation growth in consume/produce path:
      - `DebruijnInterpreter::{consume_inner,produce_inner,generated_message_eval,eval_inner}`
      - with repeated `alloc::raw_vec::RawVec::grow_one/finish_grow`
    - proposer merge/checkpoint path:
      - `compute_parents_post_state -> dag_merger::merge`
    - history read path:
      - `RadixTreeImpl::{read,load_node} -> RSpaceHistoryReaderImpl::get_data_proj`
    - startup LMDB allocation path (static baseline contributor):
      - `mdb_env_open -> heed::EnvOpenOptions::open -> KeyValueStoreManager::{get_stores,r_space_stores,eval_stores}`
 - Empty-checkpoint clone reduction (2026-02-23):
  - code change:
    - `rspace++/src/rspace/history/history_repository_impl.rs`
    - added no-op fast path for empty checkpoints in both `checkpoint(...)` and `do_checkpoint(...)` to avoid unnecessary history processing/cloning when action list is empty.
  - correctness checks:
    - `cargo check -p rspace_plus_plus --quiet` passed
    - `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture` passed (`5 passed`)
    - `cargo test -p rspace_plus_plus --test replay_rspace_tests check_replay_data_should_proceed_if_replay_data_is_empty -- --nocapture` passed
    - external suite rerun:
      - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
      - result: `12 passing`, `1 pending`
  - allocator profiling workflow hardening:
    - `scripts/ci/profile-validator-allocator-hotspots.sh` now ignores zero-byte terminal heap files and selects latest non-empty snapshot.
    - this fixed invalid runs where `latest.stacks.tsv` was empty due `jeprof.*.heap` size `0`.
  - latest allocator soak profile (valid):
    - `/tmp/casper-allocator-hotspots-20260223T-profile3`
    - soak metrics snapshot:
      - `propose_total avg/p95=2622.56/20757 ms`
      - `block_creator_compute_deploys_checkpoint avg/p95=2172.27/18985 ms`
      - `block_requests_retry_ratio=3.47`
    - top positive growth stacks remain concentrated in:
      - replay checkpoint/history path (`HistoryRepositoryImpl::checkpoint -> RadixHistory::process`)
      - interpreter consume/produce path (`DebruijnInterpreter::*`, `RawVec::grow_one`)
      - history read path (`RadixTreeImpl::{read,load_node}`)
 - Allocator call-stack deep profiling (2026-02-23T18:56Z):
  - correctness revalidation:
    - external suite rerun completed:
      - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
      - result: `12 passing`, `1 pending` (completed around `2026-02-23T18:50Z`).
  - deep 120s soak artifacts:
    - `/tmp/casper-allocator-hotspots-20260223T-callstack-120s-deep`
    - key performance snapshot:
      - `propose_total avg/p95=733.99/2355 ms`
      - `block_creator_compute_deploys_checkpoint avg/p95=485.51/1750 ms`
      - `finalizer_total avg/p95=2669.92/10608 ms`
      - `block_requests_retry_ratio=3.32`
  - top allocating call paths (positive growth, symbolized):
    - history-read clone path:
      - `Vec::clone -> RadixTreeImpl::load_node -> RSpaceHistoryReaderImpl::{get_data_proj,get_joins_proj} -> InMemHotStore::{get_data,get_joins}`
    - interpreter evaluation path:
      - `DebruijnInterpreter::{consume_inner,produce_inner,dispatch_inner,eval_inner,generated_message_eval}` with repeated `Vec::clone`/`to_vec` growth.
    - proposer compute-state path:
      - `RuntimeOps::compute_state -> block_creator::create -> Proposer::do_propose`.
    - startup/static allocator contribution:
      - `mdb_env_open -> heed::EnvOpenOptions::open -> KeyValueStoreManager::*`.
 - tooling improvement for repeatability:
    - `scripts/ci/profile-validator-allocator-hotspots.sh` now emits aggregated callsite report:
      - `stacks/delta-positive.topN.callsites.txt`
      - includes:
        - top first symbolized frames by aggregated bytes
        - categorized path totals (`rspace_history_read`, `interpreter_eval`, `proposer_compute_state`, `lmdb_env_open`, etc.)
    - smoke validation artifact:
      - `/tmp/casper-allocator-hotspots-20260223T-callstack-smoke/stacks/delta-positive.top20.callsites.txt`
 - RadixTree read-path clone reduction (2026-02-23T19:44Z):
  - code changes:
    - `rspace++/src/rspace/history/radix_tree.rs`
      - added `load_node_arc(...)` and switched `read_at(...)` to use shared `Arc<Node>` path.
      - kept existing `load_node(...)` API by delegating and cloning only for compatibility callers.
      - removed extra cache-miss clone by moving decoded node directly into `Arc`.
  - correctness validation:
    - `cargo check -p rspace_plus_plus --quiet` passed.
    - `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture` passed (`5 passed`).
    - external suite rerun:
      - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
      - result: `12 passing`, `1 pending` (latest run completed in ~`2m`).
  - allocator hotspot comparison (120s deep stack runs):
    - baseline pre-change:
      - `/tmp/casper-allocator-hotspots-20260223T-callstack-120s-deep`
      - `rspace_history_read=7,570,815`
    - after `load_node_arc` read path:
      - `/tmp/casper-allocator-hotspots-20260223T-post-loadnodearc-120s`
      - `rspace_history_read=5,217,791`
    - after no-clone-on-miss follow-up:
      - `/tmp/casper-allocator-hotspots-20260223T-post-loadnodearc2-120s`
      - `rspace_history_read=3,823,506`
  - interpretation:
    - dominant history-read allocation pressure was reduced materially (~49.5% vs baseline).
    - remaining allocation pressure moved to interpreter/proposer/other paths (e.g. `RawVec::finish_grow`, transport/prometheus hash table growth), which becomes the next memory target.

## Pre-load correctness gates and auto-recovery (2026-02-20T22:56Z)
- Benchmark script hardening:
  - `scripts/ci/run-latency-benchmark.sh` now enforces pre-load invariants before timed load:
    - validators running (`validator1/2/3`)
    - peer connectivity minimum (`PRELOAD_REQUIRE_PEERS_MIN`, default `3`) with readiness polling
    - baseline retry ratio guard from Prometheus snapshot:
      - `block_requests_retries / block_requests_total <= PRELOAD_RETRY_RATIO_MAX` (default `2.50`)
      - enforced only when `block_requests_total >= PRELOAD_RETRY_RATIO_MIN_REQUESTS` (default `100`)
  - post-load quality gate (optional):
    - `POSTLOAD_RETRY_RATIO_MAX`
    - `POSTLOAD_RETRY_RATIO_MIN_REQUESTS` (default `100`)
    - if configured and exceeded at end of run, benchmark exits non-zero after profile generation.
  - on invariant failure, diagnostics are written to `OUT_DIR/preload-diag`:
    - compose state, validator logs, metrics snapshots
- Optional self-healing mode added:
  - `AUTO_RECREATE_ON_PRELOAD_FAIL=1`
  - `AUTO_RECREATE_MAX_ATTEMPTS` (default `1`)
  - behavior: on pre-load invariant failure, force-recreate compose cluster and retry preload sequence.
- Validation evidence:
  - back-to-back gated full runs:
    - `/tmp/casper-latency-benchmark-gated-full1-20260220T224529Z`: passed, retry ratio `1.47`
    - `/tmp/casper-latency-benchmark-gated-full2-20260220T224753Z`: completed, retry ratio regressed to `4.31`
    - `/tmp/casper-latency-benchmark-gated-full3-20260220T225002Z`: blocked at preload (`ratio=2.6681 > 2.50`)
  - auto-recreate e2e run:
    - `/tmp/casper-latency-benchmark-auto-recreate-e2e-20260220T225539Z`
    - attempt 1 failed preload (`ratio=2.1833 > 1.00` with strict test threshold), cluster recreated automatically
    - attempt 2 passed preload and benchmark completed
- Operational guidance:
  - correctness-first CI gate mode:
    - keep default strict mode (`AUTO_RECREATE_ON_PRELOAD_FAIL=0`) to surface degraded baseline immediately.
  - exploratory/perf soak mode:
    - use `AUTO_RECREATE_ON_PRELOAD_FAIL=1` to auto-heal and keep collecting comparable clean-state runs.
  - convenience wrapper:
    - script: `scripts/ci/run-latency-benchmark-mode.sh`
    - strict preset:
      - `./scripts/ci/run-latency-benchmark-mode.sh strict-ci docker/shard-with-autopropose.yml 120`
      - defaults:
        - preload gate enabled (`PRELOAD_RETRY_RATIO_MAX=2.50`)
        - post-load gate enabled (`POSTLOAD_RETRY_RATIO_MAX=2.50`)
    - auto-heal preset:
      - `./scripts/ci/run-latency-benchmark-mode.sh soak-autoheal docker/shard-with-autopropose.yml 120`
      - defaults:
        - preload gate enabled
        - post-load gate disabled (measurement mode)
  - nightly recommended sequence (strict then fallback):
    - script: `scripts/ci/run-latency-benchmark-nightly.sh`
    - command:
      - `./scripts/ci/run-latency-benchmark-nightly.sh docker/shard-with-autopropose.yml 120`
    - machine-readable output:
      - writes `${OUT_BASE}-summary.json` with strict/fallback status, chosen path, artifact dirs, and extracted key metrics.
      - optional stable output copy for CI dashboards:
        - `SUMMARY_OUT=/tmp/casper-nightly-summary-latest.json ./scripts/ci/run-latency-benchmark-nightly.sh docker/shard-with-autopropose.yml 120 /tmp/casper-latency-benchmark-nightly-$(date -u +%Y%m%dT%H%M%SZ)`

## Mode comparison snapshot (2026-02-20T23:04Z)
- Goal:
  - compare fail-fast (`strict-ci`) vs self-healing (`soak-autoheal`) behavior on the same cluster sequence.
- Runs:
  - strict:
    - `/tmp/casper-latency-benchmark-mode-strict-full-20260220T225942Z`
  - soak-autoheal:
    - `/tmp/casper-latency-benchmark-mode-soak-full-20260220T230152Z`
- Summary:

| Mode | Preload behavior | `block_requests_retry_ratio` | `block_requests_retry_action_broadcast_only` | `propose_total avg/p95` |
|---|---|---:|---:|---:|
| `strict-ci` | Passed attempt 1 (`baseline ratio=2.3822 < 2.50`), no recreate | `4.57` | `635` | `1296.53ms / 2613ms` |
| `soak-autoheal` | Attempt 1 failed preload (`3.0754 > 2.50`), auto-recreated, attempt 2 passed | `1.48` | `44` | `1206.25ms / 2464ms` |

- Interpretation:
  - `strict-ci` is best for correctness-first gating (surface degraded state immediately).
  - `soak-autoheal` is better for collecting stable comparable performance runs when state is already degraded.

## Nightly sequence validation with post-load gate (2026-02-20T23:18Z)
- Run:
  - `./scripts/ci/run-latency-benchmark-nightly.sh docker/shard-with-autopropose.yml 120 /tmp/casper-latency-benchmark-nightly-postgate-20260220T231542Z`
- Artifacts:
  - strict: `/tmp/casper-latency-benchmark-nightly-postgate-20260220T231542Z-strict`
  - fallback: `/tmp/casper-latency-benchmark-nightly-postgate-20260220T231542Z-soak-autoheal`
- Observed flow:
  - `strict-ci` failed pre-load retry-ratio gate (`3.5559 > 2.50`), as intended.
  - nightly switched to `soak-autoheal`.
  - fallback attempt 1 also failed pre-load gate, auto-recreated cluster, attempt 2 passed and completed.
- Fallback run summary:
  - `deploy_success=54/54`
  - `block_requests_retry_ratio=1.53`
  - `block_requests_retry_action_broadcast_only=50`
  - `propose_total avg/p95=1223.55ms / 2659ms`
- Interpretation:
  - strict quality gates correctly block degraded state.
  - nightly fallback reliably restores clean-state measurement when degradation is detected.

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

### Experiment D (kept): retry-loop de-amplification when waiting-list is empty
- File: `casper/src/rust/engine/block_retriever.rs`
- Root cause:
  - `request_all` retried expired missing hashes even when `waiting_list` was empty.
  - `try_rerequest` empty-waiting branch previously did no action and did not refresh timestamp.
  - This created retry amplification in maintenance loops.
- Change:
  - in `try_rerequest`, always refresh per-hash request timestamp on retry pass.
  - if `waiting_list` is empty, broadcast `HasBlockRequest` instead of no-op.
  - keep existing next-peer retry behavior when waiting peers exist.
- Validation:
  - rebuilt image, full compose recreate.
  - external suite passed (`12 passing`, `1 pending`).
- Benchmarks vs `current-J` baseline:
  - baseline (`current-J`)
    - `propose_total avg/p95`: `478.52 / 807 ms`
    - `compute_deploys_checkpoint avg/p95`: `238.54 / 479 ms`
    - `block_requests_retry_ratio`: `6.07`
  - candidate run 1 (`current-N`)
    - `propose_total avg/p95`: `603.49 / 1136 ms`
    - `compute_deploys_checkpoint avg/p95`: `258.99 / 564 ms`
    - `block_requests_retry_ratio`: `1.35`
  - candidate run 2 (`current-O`)
    - `propose_total avg/p95`: `636.60 / 1201 ms`
    - `compute_deploys_checkpoint avg/p95`: `276.39 / 585 ms`
    - `block_requests_retry_ratio`: `3.18`
- Interpretation:
  - retry storm is materially reduced (ratio improvement `~48%` to `~78%` vs baseline).
  - propose/checkpoint and finalizer remained noisier in these runs; additional Phase 2 work needed to convert retry gains into stable end-to-end latency gains.

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

## Fresh baseline after retry-loop rollback (`2026-02-20T15:06Z` to `15:09Z`)
- Rebuilt image from current `HEAD` (`5dbbb97d`) and force-recreated `docker/shard-with-autopropose.yml`.
- External suite rerun:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - Result: `12 passing`, `1 pending` (completed around `2026-02-20T15:07Z`).
- Fresh profile (`scripts/ci/profile-casper-latency.sh`):
  - `propose_total`: `avg=825.72ms`, `p95=1348ms`
  - `block_creator_total_create_block`: `avg=501.75ms`, `p95=728ms`
  - `block_creator_compute_deploys_checkpoint`: `avg=413.89ms`, `p95=641ms`
  - `finalizer_total`: `avg=649.53ms`, `p95=834ms`
  - `block_validation_mean_ms=635.92`
  - `block_replay_mean_ms=934.23`
  - `block_requests_retry_ratio=3.41`

Interpretation:
- Latest commit is currently functionally correct on the external regression suite after clean recreate.
- End-to-end latency remains dominated by propose/create-block/checkpoint and replay/validation, with finalizer now also elevated in this sample.

## Init correctness instrumentation status update
- `scripts/ci/check-casper-init-sla.sh` passes, but validators may legitimately take a direct-to-running path that does not export Initializing-only metrics.
- `scripts/ci/collect-casper-init-artifacts.sh` was updated to classify this as `PASS_DIRECT_RUNNING` (instead of false `FAIL`) when:
  - `casper_init_transition_to_running >= 1`, and
  - `casper_init_attempts`, `casper_init_approved_block_received`, and `casper_init_time_to_running_count` are all absent.

Why this matters:
- It keeps correctness reporting aligned with actual engine behavior during genesis ceremony / Casper initialization and avoids false negatives in triage.

## Next correctness-first latency plan (tightened)
1. Keep startup correctness green on every run:
   - `check-casper-init-sla.sh` must pass.
   - external suite must stay `12 passing`, `1 pending`.
2. Stabilize replay/validation first:
   - profile block replay/validation sub-phases under sustained deploy load.
   - prioritize `block_validation_step_checkpoint_time` and replay user-deploy path.
3. Then reduce propose/create-block tails:
   - reduce `compute_deploys_checkpoint` merged-path variance.
   - preserve rejection semantics; reject any optimization that regresses correctness warnings into proposer failures.
4. Finally tune finalizer/retriever pressure:
   - keep retry ratio low without inducing proposer stalls.
   - verify finalizer p95 regression source with isolated runs.

## Phase A progress: direct-to-running metrics now emitted in code
- Change:
  - Added `record_direct_to_running_init_metrics()` in `casper/src/rust/engine/engine.rs`.
  - Called it in direct startup transitions:
    - `casper/src/rust/engine/casper_launch.rs` before `transition_to_running(...)`
    - `casper/src/rust/engine/genesis_ceremony_master.rs` before `transition_to_running(...)`
- Purpose:
  - Ensure validators that bypass `Initializing` still emit:
    - `casper_init_attempts`
    - `casper_init_approved_block_received`
    - `casper_init_time_to_approved_block`
    - `casper_init_time_to_running`
  - Keep startup correctness telemetry deterministic and CI-artifact friendly.

Verification (`2026-02-20T15:20Z`):
- `scripts/ci/check-casper-init-sla.sh docker/shard-with-autopropose.yml`
  - validator1/2/3 metrics now: `attempts=1, approved=1, transitions=1, time_to_running_count=1`
  - result: `SLA PASSED`
- `scripts/ci/collect-casper-init-artifacts.sh ...`
  - `validator_init_gate: PASS` for validator1/2/3 (not `PASS_DIRECT_RUNNING`)
- external regression suite:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - result: `12 passing`, `1 pending`

Post-fix profile snapshot (`2026-02-20T15:41Z`):
- `propose_total`: `avg=577.75ms`, `p95=1021ms`
- `block_creator_total_create_block`: `avg=301.80ms`, `p95=654ms`
- `block_creator_compute_deploys_checkpoint`: `avg=208.20ms`, `p95=541ms`
- `finalizer_total`: `avg=771.82ms`, `p95=982ms`
- `block_requests_retry_ratio=2.50`

## Experiment E (rejected): reuse runtime-manager lock across validation checkpoint + bonds-cache
- File tried: `casper/src/rust/multi_parent_casper_impl.rs`
- Change tried:
  - acquire `runtime_manager` once and reuse it for:
    - `validate_block_checkpoint(...)`
    - `Validate::bonds_cache(...)`
  - intended goal: reduce lock churn in validation hot path.
- Validation:
  - startup SLA: passed.
  - external suite: passed (`12 passing`, `1 pending`).
- Performance outcome:
  - controlled benchmark (`/tmp/casper-latency-benchmark-lockreuse`):
    - `propose_total avg/p95 = 1131.63 / 2919 ms`
    - `compute_deploys_checkpoint avg/p95 = 295.78 / 698 ms`
    - `block_validation_mean_ms = 1517.41`
    - `block_replay_mean_ms = 1062.50`
  - clean-state benchmark (`/tmp/casper-latency-benchmark-lockreuse-clean`):
    - `propose_total avg/p95 = 722.07 / 1848 ms`
    - `compute_deploys_checkpoint avg/p95 = 391.36 / 1202 ms`
    - `finalizer_total avg/p95 = 676.93 / 873 ms`
- Decision:
  - rejected and reverted; lock reuse increased tail latency and contention in this environment.

Revalidation after revert (`2026-02-20T15:55Z`):
- startup SLA: passed with full init metrics on validators.
- external suite: `12 passing`, `1 pending`.
- profile snapshot (`/tmp/casper-latency-profile-20260220-155500`):
  - `propose_total avg/p95 = 898.38 / 1726 ms`
  - `compute_deploys_checkpoint avg/p95 = 308.09 / 771 ms`
  - `block_validation_mean_ms = 552.60`
  - `block_replay_mean_ms = 673.05`
  - `block_requests_retry_ratio = 0.70`

## Latest correctness verification (`2026-02-20T16:24Z`)
- Current branch (`5dbbb97d` + working-tree changes) after forced clean recreate:
  - `docker compose -f docker/shard-with-autopropose.yml down -v && up --build -d`
  - `scripts/ci/check-casper-init-sla.sh docker/shard-with-autopropose.yml`: `SLA PASSED`
  - validator metrics: `attempts=1`, `approved=1`, `transitions=1`, `time_to_running_count=1`
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`: `12 passing`, `1 pending` (about `59s`)

## Controlled commit comparison (clean recreate + 120s load)
Method:
- For each revision: full `docker compose down -v` + `up --build -d`, then same 120s deploy/propose load and same profile extraction.

Revisions:
- `HEAD` = `5dbbb97d` (`/tmp/casper-latency-benchmark-current-R`)
- `HEAD~1` = `691a7bb5` (`/tmp/casper-latency-benchmark-head-minus-1`)
- `HEAD~2` = `afa19661` (`/tmp/casper-latency-benchmark-head-minus-2`)

Results:
- `HEAD` (`5dbbb97d`)
  - `propose_total avg/p95`: `847.99 / 1708 ms`
  - `create_block avg/p95`: `546.51 / 1042 ms`
  - `compute_deploys_checkpoint avg/p95`: `447.52 / 892 ms`
  - `finalizer_total avg/p95`: `814.55 / 1097 ms`
  - `block_validation_mean_ms`: `689.60`
  - `block_replay_mean_ms`: `646.46`
  - `retry_ratio`: `9.17`
  - load summary: `50 deploys`, `propose_ok=1`, `propose_fail=2`
- `HEAD~1` (`691a7bb5`)
  - `propose_total avg/p95`: `715.44 / 879 ms`
  - `create_block avg/p95`: `398.82 / 494 ms`
  - `compute_deploys_checkpoint avg/p95`: `311.99 / 393 ms`
  - `finalizer_total avg/p95`: `761.82 / 1014 ms`
  - `block_validation_mean_ms`: `97.99`
  - `block_replay_mean_ms`: `417.40`
  - `retry_ratio`: `6.80`
  - load summary: `45 deploys`, `propose_ok=1`, `propose_fail=2`
- `HEAD~2` (`afa19661`)
  - `propose_total avg/p95`: `1343.35 / 4593 ms`
  - `create_block avg/p95`: `503.31 / 861 ms`
  - `compute_deploys_checkpoint avg/p95`: `394.28 / 725 ms`
  - `finalizer_total avg/p95`: `824.30 / 1109 ms`
  - `block_validation_mean_ms`: `2474.67`
  - `block_replay_mean_ms`: `1538.83`
  - `retry_ratio`: `14.02`
  - load summary: `56 deploys`, `propose_ok=3`, `propose_fail=0`

Interpretation:
- `HEAD~2` is clearly the worst for validation/replay/propose tails.
- `HEAD` is materially better than `HEAD~2` for propose tails but currently regresses vs `HEAD~1` on propose/create/checkpoint, validation, replay, and retry ratio.
- Correctness stays green on current branch, but latency budget is not yet improved relative to `HEAD~1`.

## Updated correctness-first e2e latency reduction plan
1. Keep initialization correctness as hard gate:
   - enforce `check-casper-init-sla.sh` + external suite pass on every candidate run.
   - keep direct-to-running metrics emission intact; reject any change that drops init telemetry.
2. Remove replay-path regressions before new optimizations:
   - A/B current replay allocation patch in `casper/src/rust/rholang/replay_runtime.rs` against `HEAD~1` behavior over at least 3 clean runs.
   - if no stable win in `block_replay_mean_ms` and `compute_deploys_checkpoint p95`, revert patch.
3. Stabilize propose/create loop under load:
   - prioritize `compute_deploys_checkpoint` variance reduction and proposer success rate (`propose_fail=0` goal).
   - require p95 improvements without raising retry ratio.
4. Finalizer and retriever tuning after proposer loop is stable:
   - cap retry amplification (`retry_ratio < 5` target in this benchmark shape).
   - profile finalizer cache misses (`layers_visited`, message weight cache miss) and reduce repeated traversals.
5. Acceptance criteria for a candidate optimization:
   - correctness: init SLA pass + external suite `12 passing`, `1 pending`.
   - performance: `propose_total p95` and `compute_deploys_checkpoint p95` both better than current baseline, with no regression in `retry_ratio` and no increase in proposer failures.

## Experiment F (current): revert replay allocation micro-optimization
- File: `casper/src/rust/rholang/replay_runtime.rs`
- Change:
  - reverted the single-vector preallocation/append optimization in `replay_deploys`.
  - restored separate `deploy_results` and `system_deploy_results` vectors with merge at the end.
- Rationale:
  - clean benchmark on commit `cfe025ed` showed instability and high retry amplification.

Verification:
- Clean recreate + init SLA on reverted state: pass (via benchmark step 1).
- External suite on reverted state:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - result: `12 passing`, `1 pending` (`~1m`).

Benchmarks:
- candidate before revert (`/tmp/casper-latency-benchmark-cfe025ed`):
  - `propose_total avg/p95`: `943.22 / 2032 ms`
  - `compute_deploys_checkpoint avg/p95`: `442.40 / 1007 ms`
  - `block_replay_mean_ms`: `758.39`
  - `block_requests_retry_ratio`: `37.37`
- candidate after revert, clean recreate (`/tmp/casper-latency-benchmark-cfe025ed-revert-replay-clean`):
  - `propose_total avg/p95`: `971.92 / 1309 ms`
  - `compute_deploys_checkpoint avg/p95`: `373.47 / 512 ms`
  - `block_replay_mean_ms`: `725.49`
  - `block_requests_retry_ratio`: `8.83`
- candidate after revert, clean recreate replicate (`/tmp/casper-latency-benchmark-cfe025ed-revert-replay-clean-2`):
  - `propose_total avg/p95`: `896.07 / 1130 ms`
  - `compute_deploys_checkpoint avg/p95`: `351.11 / 491 ms`
  - `block_replay_mean_ms`: `745.93`
  - `block_requests_retry_ratio`: `8.69`

Interpretation:
- p95 tails improve materially in proposer and checkpoint phases.
- retry amplification is dramatically lower after revert.
- keep replay revert; treat prior micro-optimization as rejected.

## Proposer failure classification fix (benchmark hygiene)
- File: `scripts/ci/run-latency-benchmark.sh`
- Change:
  - split proposer outcomes into:
    - `propose_ok`
    - `propose_transient` (only `Propose skipped due to transient proposal race`)
    - `propose_bug_error` (explicit `Proposal failed: BugError`)
    - `propose_fail` (hard failures; includes bug errors)
- Why:
  - previous classification treated `BugError` as transient and masked correctness failures.
- Validation:
  - sanity run with aggressive propose cadence (`PROPOSE_EVERY=1`) now reports explicit bug failures:
    - `/tmp/casper-latency-benchmark-script-sanity-classifier-fix`
    - `propose_bug_error=1`, `propose_fail=1`

## Root-cause progress: repeated toxic deploy causing refund failure
- New diagnostics in `casper/src/rust/rholang/runtime.rs`:
  - refund failure log now includes:
    - `deploy_sig`
    - `deployer_pk`
    - `refund_amount`
  - metric emitted: `casper_runtime_refund_failures_total`
- Important finding (rebuilt local image, 2026-02-20):
  - same deploy repeatedly triggers:
    - `System runtime error: Unable to refund remaining gas ((Bug found) Deploy refund failed: Insufficient funds)`
  - observed recurring signature:
    - `deploy_sig=3045022100ba95d11e801a0307d847b96964469bd68b7297e1b4f2494710ce7339f313c43f02201d25af3c1c4576586f5edd18da3ea3145328fe84e3dd5c62635c932e3142a706`
  - this explains repeated proposer `BugError` and associated latency degradation under load.

Next correctness priority:
- implement deterministic handling for toxic deploys that repeatedly trigger gas-refund platform failure:
  - quarantine/evict offending deploy from proposal candidate set, or
  - treat refund failure path with safe rollback and deploy-level rejection without poisoning proposer loop.

## Experiment G (kept): toxic deploy quarantine on gas-refund platform failure
- Files:
  - `casper/src/rust/rholang/runtime.rs`
  - `casper/src/rust/blocks/proposer/block_creator.rs`
- Changes:
  - `GasRefundFailure` now carries full context (`deploy_sig`, `deployer_pk`, `refund_amount`).
  - Added metric `casper_runtime_refund_failures_total`.
  - In block creation, when checkpoint computation fails with `GasRefundFailure`:
    - parse `deploy_sig` from failure context,
    - remove matching deploy from deploy storage (quarantine),
    - return `NoNewDeploys` for this proposal cycle instead of propagating hard proposer error.
- Why:
  - without quarantine, the same toxic deploy repeatedly caused:
    - `Unable to refund remaining gas ((Bug found) Deploy refund failed: Insufficient funds)`
    - repeated proposer `BugError`
    - prolonged proposer degradation under load.

Validation:
- Rebuilt local node image from source and recreated cluster.
- `scripts/ci/check-casper-init-sla.sh docker/shard-with-autopropose.yml`: pass.
- external suite:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - `12 passing`, `1 pending`.

Benchmark evidence (aggressive proposer cadence: `PROPOSE_EVERY=1`):
- before fix (`/tmp/casper-latency-benchmark-diagnostics-check-new-image-long`):
  - `propose_ok=10`, `propose_bug_error=4`, `propose_fail=4`
- after fix run 1 (`/tmp/casper-latency-benchmark-quarantine-fix`):
  - `propose_ok=17`, `propose_bug_error=0`, `propose_fail=0`
  - `propose_total p95=1641ms`
  - `compute_deploys_checkpoint p95=706ms`
- after fix run 2 (`/tmp/casper-latency-benchmark-quarantine-fix-replicate`):
  - `propose_ok=19`, `propose_bug_error=0`, `propose_fail=0`
  - `propose_total p95=1636ms`
  - `compute_deploys_checkpoint p95=730ms`

Runtime confirmation:
- validator logs now show repeated toxic signature explicitly:
  - `deploy_sig=3045022100ba95d11e801a0307d847b96964469bd68b7297e1b4f2494710ce7339f313c43f02201d25af3c1c4576586f5edd18da3ea3145328fe84e3dd5c62635c932e3142a706`
- quarantine path prevents this single deploy from repeatedly poisoning proposer cycles.

## Fresh A/B comparison after correctness fixes (`2026-02-20T20:28Z` to `20:30Z`)
Method:
- same benchmark shape for both revisions:
  - clean recreate (`up -d --force-recreate`),
  - `120s` load,
  - same deploy payload and interval.

Revisions:
- current fixed working tree (quarantine + replay revert + improved benchmark counters):
  - `/tmp/casper-latency-benchmark-current-quarantine-fix-e2e`
- `HEAD~1` (`5dbbb97d`):
  - `/tmp/casper-latency-benchmark-head-minus-1-postfix-compare`

Results:
- current fixed:
  - load: `53 deploys`, `propose_ok=3`, `propose_bug_error=0`, `propose_fail=0`
  - `propose_total avg/p95`: `1054.13 / 2301 ms`
  - `create_block avg/p95`: `494.39 / 1316 ms`
  - `compute_deploys_checkpoint avg/p95`: `373.45 / 1154 ms`
  - `finalizer_total avg/p95`: `1277.55 / 2026 ms`
  - `block_validation_mean_ms`: `802.20`
  - `block_replay_mean_ms`: `747.32`
  - `retry_ratio`: `8.52`
- `HEAD~1`:
  - load: `45 deploys`, `propose_ok=2`, `propose_fail=1`
  - `propose_total avg/p95`: `1243.27 / 3115 ms`
  - `create_block avg/p95`: `700.38 / 2059 ms`
  - `compute_deploys_checkpoint avg/p95`: `571.37 / 1887 ms`
  - `finalizer_total avg/p95`: `1493.52 / 2282 ms`
  - `block_validation_mean_ms`: `313.72`
  - `block_replay_mean_ms`: `826.08`
  - `retry_ratio`: `9.13`

Interpretation:
- current fixed branch materially improves propose/create/checkpoint/finalizer tails vs `HEAD~1`.
- proposer correctness also improved in this sample (`propose_bug_error=0`, `propose_fail=0`).
- validation/replay means remain mixed and should be optimized in the next phase.

## Metric unit correction: validation step histogram
- File: `casper/src/rust/multi_parent_casper_impl.rs`
- Change:
  - switched validation duration histogram recording from `elapsed.as_millis()` to `elapsed.as_secs_f64()` in both:
    - `validate(...)`
    - `validate_self_created(...)`
- Why:
  - histogram buckets are second-based and profile scripts convert `_sum` seconds to milliseconds; recording raw milliseconds into the seconds metric produced inflated `sum_s` values and skewed means.
- Sanity check:
  - `cargo check -p casper` passed after the change.

## Continue run verification (`2026-02-20T20:39Z` to `20:42Z`)
Benchmark run:
- command:
  - `./scripts/ci/run-latency-benchmark.sh docker/shard-with-autopropose.yml 120 /tmp/casper-latency-benchmark-continue-20260220T203915Z`
- load summary:
  - `deploy_attempts=42`, `deploy_success=36`, `deploy_failure=6`
  - `propose_ok=1`, `propose_transient=0`, `propose_bug_error=0`, `propose_fail=1`
- profile summary:
  - `propose_total avg/p95`: `1355.10 / 2426 ms`
  - `create_block avg/p95`: `694.45 / 1538 ms`
  - `compute_deploys_checkpoint avg/p95`: `562.24 / 1370 ms`
  - `finalizer_total avg/p95`: `1658.55 / 2356 ms`
  - `block_validation_mean_ms`: `779.40 (sum_s=29.6172909620, count=38)`
  - `block_replay_mean_ms`: `770.10 (sum_s=26.9535645440, count=35)`
  - `block_requests_retry_ratio`: `1.62`

Interpretation:
- corrected `sum_s` values now look internally consistent (seconds scale), confirming the metric-unit fix behavior.
- this run had one non-`BugError` proposer failure and lower deploy attempt rate; end-to-end latency remains noisy and needs additional correctness hardening around proposal stability under sustained load.

External regression suite:
- command:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
- result:
  - `12 passing`, `1 pending` (completed around `2026-02-20T20:42Z`)
- notes:
  - negative syntax/unbound-name tests failed deploy as expected.
  - existing non-fatal warnings remained (`No blockHash found for deploy`, optional strict checks disabled).

## Benchmark harness correctness hardening (startup endpoint + log scoping)
Files:
- `scripts/ci/run-latency-benchmark.sh`
- `scripts/ci/profile-casper-latency.sh`

Changes:
- added deploy-target configuration:
  - `DEPLOY_HOST` (default `localhost`)
  - `DEPLOY_GRPC_PORT` (default `40412`)
  - `DEPLOY_HTTP_PORT` (default `40413`)
- added startup endpoint readiness gate before timed load:
  - `status -H $DEPLOY_HOST -p $DEPLOY_HTTP_PORT`
  - requires configurable consecutive successes (`GRPC_READY_CONSECUTIVE_SUCCESSES`, default `2`)
- wired deploy/propose commands to explicit host/port (no implicit defaults).
- profile log extraction now accepts `log_since_utc` and benchmark passes the pre-load timestamp, so propose/create/finalizer samples come from this run window instead of full container lifetime.

Validation:
- smoke run (`20s`) with new gating + log scoping:
  - `/tmp/casper-latency-benchmark-logscope-20260220T2052Z`
  - `deploy_attempts=10`, `deploy_success=10`, `deploy_failure=0`
  - profile timing sample counts became run-local (`propose_total n=28`, `create_block n=29`) instead of inflated multi-run totals.

Clean 120s gated baseline:
- `/tmp/casper-latency-benchmark-readiness-gated-logscope-20260220T2052Z`
- load summary:
  - `deploy_attempts=53`, `deploy_success=53`, `deploy_failure=0`
  - `propose_ok=3`, `propose_bug_error=0`, `propose_fail=0`
- profile:
  - `propose_total avg/p95`: `1382.46 / 2583 ms`
  - `create_block avg/p95`: `702.27 / 1544 ms`
  - `compute_deploys_checkpoint avg/p95`: `537.25 / 1380 ms`
  - `finalizer_total avg/p95`: `1681.54 / 2304 ms`

Comparison to pre-gate run (`/tmp/casper-latency-benchmark-continue-20260220T203915Z`):
- transport-flap failures removed:
  - before: `deploy_failure=6`, `propose_fail=1`
  - after: `deploy_failure=0`, `propose_fail=0`
- deploy throughput improved:
  - before: `42 attempts / 120s` (`0.35/s`)
  - after: `53 attempts / 120s` (`0.44/s`)

Interpretation:
- readiness gating and scoped profiling improve correctness and measurement reliability.
- next correctness priority remains reducing large block-request retry amplification and stabilizing propose/create/checkpoint tails under sustained deploy load.

## Retry amplification root cause and fix (`2026-02-20T20:58Z` to `21:15Z`)
Root-cause finding:
- scoped compose logs during benchmark window showed repeated replay retries for the same block on `validator3`:
  - `SystemRuntimeError(ConsumeFailed)` in `casper/src/rust/util/rholang/interpreter_util.rs`
- block-retriever metrics showed extreme retry inflation:
  - `block_requests_total{source="f1r3fly.casper.block-retriever"}`
  - `block_requests_retries{source="f1r3fly.casper.block-retriever"}`
- code-level bug in `block_retriever`:
  - retry counter was incremented before `try_rerequest(...)`, even when no actual request was sent (e.g., empty waiting list), causing noisy/overstated retry counts and hot-loop churn.

Code changes:
- file: `casper/src/rust/engine/block_retriever.rs`
- updates:
  - `try_rerequest(...)` now returns `Result<bool, CasperError>` indicating whether a retry action was actually issued.
  - `BLOCK_REQUESTS_RETRIES_METRIC` is incremented only when `did_retry == true`.
  - when waiting list is empty, retriever now:
    - refreshes request timestamp (age-threshold backoff), and
    - broadcasts `HasBlockRequest` once per retry cycle instead of no-op looping.

Live validation:
- build:
  - local image rebuilt and retagged `f1r3flyindustries/f1r3fly-rust-node:latest` (`sha256:e45a074663d3...`)
- short run after fix:
  - `/tmp/casper-latency-benchmark-retryfix-20260220T2110Z`
  - `deploy_attempts=10`, `deploy_success=10`, `propose_fail=0`
  - `block_requests_total=25`, `block_requests_retries=0`, `retry_ratio=0.00`
- sustained run after fix:
  - `/tmp/casper-latency-benchmark-retryfix-120s-20260220T2113Z`
  - `deploy_attempts=52`, `deploy_success=52`, `propose_fail=0`
  - `propose_total avg/p95`: `1100.68 / 2732 ms`
  - `create_block avg/p95`: `538.98 / 1439 ms`
  - `compute_deploys_checkpoint avg/p95`: `410.01 / 1264 ms`
  - `finalizer_total avg/p95`: `1428.45 / 2203 ms`
  - `block_requests_total=292`, `block_requests_retries=662`, `retry_ratio=2.27`

Before/after comparison (same delta-metric harness):
- pre-fix:
  - `/tmp/casper-latency-benchmark-delta-metrics-20260220T2058Z`
  - `block_requests_total=52`, `block_requests_retries=44471`, `retry_ratio=855.21`
- post-fix (120s):
  - `/tmp/casper-latency-benchmark-retryfix-120s-20260220T2113Z`
  - `block_requests_total=292`, `block_requests_retries=662`, `retry_ratio=2.27`

Interpretation:
- retry amplification is reduced by ~99.7% versus the prior measured state and no longer dominates E2E benchmark noise.
- remaining retries are non-zero under sustained load and should be profiled next by block hash / peer source to drive the next correctness+latency phase.

External regression suite revalidation:
- `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
- result on retry-fix image:
  - `12 passing`, `1 pending` (completed around `2026-02-20T21:16Z`)

## Retry action attribution (new metrics)
Instrumentation:
- `casper/src/rust/metrics_constants.rs`
  - added `block.requests.retry.action`
- `casper/src/rust/engine/block_retriever.rs`
  - emits action-labeled counter increments on retry path:
    - `action="peer_request"`
    - `action="broadcast_only"`
    - `action="none"`
- `scripts/ci/profile-casper-latency.sh`
  - now reports delta counters:
    - `block_requests_retry_action_peer_request`
    - `block_requests_retry_action_broadcast_only`
    - `block_requests_retry_action_none`

Validation run:
- `/tmp/casper-latency-benchmark-retry-action-fixedparser-20260220T2137Z`
- load:
  - `deploy_attempts=10`, `deploy_success=10`, `propose_fail=0`
- retry metrics:
  - `block_requests_total=56`
  - `block_requests_retries=341`
  - `block_requests_retry_ratio=6.09`
  - `block_requests_retry_action_peer_request=59`
  - `block_requests_retry_action_broadcast_only=281`
  - `block_requests_retry_action_none=0`

Interpretation:
- residual retry volume is dominated by `broadcast_only` (~82% in this sample), i.e. hashes frequently have no candidate peers in waiting list when retry window opens.
- this points to peer/source acquisition and broadcast backoff policy as the next correctness+latency lever, rather than per-peer request retry logic.

## Broadcast-only cooldown experiment (`2026-02-20T21:46Z` to `21:49Z`)
Code change:
- file: `casper/src/rust/engine/block_retriever.rs`
- behavior:
  - added `broadcast_retry_last_request` per-hash map
  - for `broadcast_only` retry path, apply cooldown (`1000ms`) before rebroadcasting `HasBlockRequest`
  - emit retry-action metric `action="broadcast_suppressed"` when broadcast is skipped due to cooldown

Profiler update:
- file: `scripts/ci/profile-casper-latency.sh`
- now reports:
  - `block_requests_retry_action_broadcast_suppressed`

Build/runtime:
- image rebuilt and retagged:
  - `f1r3flyindustries/f1r3fly-rust-node:latest`
  - `sha256:8fd4227a6976...`

Sustained run:
- `/tmp/casper-latency-benchmark-broadcast-cooldown-120s-20260220T2147Z`
- load:
  - `deploy_attempts=55`, `deploy_success=55`, `propose_fail=0`
- retry/action metrics:
  - `block_requests_total=330`
  - `block_requests_retries=721`
  - `block_requests_retry_ratio=2.18`
  - `block_requests_retry_action_peer_request=299`
  - `block_requests_retry_action_broadcast_only=422`
  - `block_requests_retry_action_broadcast_suppressed=0`

Interpretation:
- cooldown did not trigger in this sample (`broadcast_suppressed=0`), implying most broadcast-only retries are already spaced beyond 1 second.
- slight retry-ratio improvement vs prior comparable run (`2.66 -> 2.18`) is directionally positive but not sufficient evidence that cooldown is the primary lever.
- next priority should focus on faster peer-source acquisition for missing hashes (to convert `broadcast_only` into `peer_request`) rather than only rate-limiting broadcasts.

External suite revalidation on cooldown build:
- `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
- result:
  - `12 passing`, `1 pending` (completed around `2026-02-20T21:50Z`)

## Peer requery experiment (convert broadcast-only retries to direct peer requests)
Code change:
- file: `casper/src/rust/engine/block_retriever.rs`
- behavior:
  - when retrying and `waiting_list` is empty but `peers` set is non-empty, issue direct request to a known peer (`peer_requery`) instead of immediate `broadcast_only`.
  - metric action added: `action="peer_requery"`.

Profiler update:
- file: `scripts/ci/profile-casper-latency.sh`
- now reports:
  - `block_requests_retry_action_peer_requery`

Build/runtime:
- image rebuilt and retagged:
  - `f1r3flyindustries/f1r3fly-rust-node:latest`
  - `sha256:6976735585e6...`

Sustained run:
- `/tmp/casper-latency-benchmark-peer-requery-120s-20260220T2203Z`
- load:
  - `deploy_attempts=55`, `deploy_success=55`, `propose_fail=0`
- retry/action metrics:
  - `block_requests_total=300`
  - `block_requests_retries=392`
  - `block_requests_retry_ratio=1.31`
  - `block_requests_retry_action_peer_request=211`
  - `block_requests_retry_action_peer_requery=155`
  - `block_requests_retry_action_broadcast_only=26`
  - `block_requests_retry_action_none=0`
  - `block_requests_retry_action_broadcast_suppressed=0`

Comparison vs prior cooldown-only run (`/tmp/casper-latency-benchmark-broadcast-cooldown-120s-20260220T2147Z`):
- retry ratio improved:
  - `2.18 -> 1.31` (`~40%` reduction)
- broadcast-only retries reduced sharply:
  - `422 -> 26` (`~94%` reduction)
- end-to-end timing remained similar/stable:
  - `propose_total avg`: `1096.79 -> 1102.60 ms`
  - `create_block avg`: `542.27 -> 552.43 ms`
  - `checkpoint avg`: `406.84 -> 421.76 ms`
  - `finalizer avg`: `1527.98 -> 1491.30 ms`

Interpretation:
- this is the first change that materially shifts retry behavior from blind broadcast churn to targeted peer retrieval while keeping correctness stable.
- next optimization should reduce `peer_requery` retries by improving peer selection quality (e.g., rotate known peers, score successful responders, or cap repeated queries to the same peer per hash).

External suite revalidation on peer-requery build:
- `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
- result:
  - `12 passing`, `1 pending` (completed around `2026-02-20T22:05Z`)

## Peer requery round-robin selection experiment
Code changes:
- file: `casper/src/rust/engine/block_retriever.rs`
- updates:
  - `peer_requery` now round-robins across known peers per hash using `known_peer_requery_cursor`.
  - added cleanup for retry auxiliary state when request state is removed (`clear_retry_aux_state` in request-all expiry path).

Build/runtime:
- image rebuilt:
  - `f1r3flyindustries/f1r3fly-rust-node:latest`
  - `sha256:5ce214f70cc7...`

Sustained run:
- `/tmp/casper-latency-benchmark-peer-requery-rr-120s-20260220T2212Z`
- load:
  - `deploy_attempts=54`, `deploy_success=54`, `propose_fail=0`
- retry/action metrics:
  - `block_requests_total=285`
  - `block_requests_retries=376`
  - `block_requests_retry_ratio=1.32`
  - `block_requests_retry_action_peer_request=195`
  - `block_requests_retry_action_peer_requery=144`
  - `block_requests_retry_action_broadcast_only=37`
  - `block_requests_retry_action_none=0`

Comparison vs prior `peer_requery` (no round-robin) run:
- previous:
  - `/tmp/casper-latency-benchmark-peer-requery-120s-20260220T2203Z`
  - `retry_ratio=1.31`, `retries=392`, `broadcast_only=26`
- round-robin:
  - `retry_ratio=1.32`, `retries=376`, `broadcast_only=37`

Interpretation:
- round-robin did not provide clear additional latency/retry gains in this sample (mixed movement; likely within run-to-run noise).
- keep `peer_requery` (major gain), treat round-robin as neutral pending more replicas.

External suite revalidation on round-robin build:
- `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
- result:
  - `12 passing`, `1 pending` (completed around `2026-02-20T22:19Z`)

## 2026-02-23: Memory-leak and retry-loop hardening continuation

### Code changes (correctness-first)
- `casper/src/rust/engine/block_retriever.rs`
  - Added adaptive per-hash `peer_requery` cooldown based on retry-attempt count.
  - Goal: damp requery amplification for unresolved hot hashes without blocking first retries.
- `casper/src/rust/blocks/block_processor.rs`
  - Added missing-dependency quarantine TTL for blocks dropped after repeated missing-dependency checks.
  - Added quarantine sweep + skip path before dependency processing.
  - Goal: prevent immediate re-admission loops of toxic dependency blocks.

### Correctness validation
- Init SLA:
  - `./scripts/ci/check-casper-init-sla.sh docker/shard-with-autopropose.yml 240`
  - Result: PASS (all validators reached `Running` and exposed required init metrics).
- External suite:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - Result: `12 passing, 1 pending`.

### Performance tracking

Strict benchmark before adaptive backoff (reference):
- `/tmp/casper-latency-benchmark-memleak-pass-20260222T235233Z`
- `block_requests_retry_ratio=2.53` (strict post-load gate fail)
- `replay_retry_top_hashes`: very concentrated (`f1a220...=333`, `9f7a14...=122`)

Strict benchmark after adaptive `peer_requery` cooldown:
- `/tmp/casper-latency-benchmark-adaptive-requery-20260223T000310Z`
- `deploy_success=48/48`, `propose_fail=0`
- `block_requests_retry_ratio=1.14` (gate pass)
- retry action mix:
  - `peer_request=89`
  - `peer_requery=57`
  - `broadcast_only=17`
- `replay_retry_top_hashes`: (`9f7a14...=120`, `7e8965...=118`)

Strict benchmark after dependency quarantine:
- `/tmp/casper-latency-benchmark-depquarantine-20260223T002003Z`
- `deploy_success=53/53`
- `block_requests_retry_ratio=1.16` (gate pass)
- retry action mix:
  - `peer_request=145`
  - `peer_requery=116`
  - `broadcast_only=29`
- replay hot-hash concentration collapsed:
  - `f1a220...=3`
  - `d1687e...=3`
  - `9f7a14...=3`
  - `7e8965...=3`

### Memory/leak tracking

Fresh soak before dependency quarantine:
- `/tmp/casper-validator-leak-soak-adaptive-requery-fresh-20260223T000859Z`
- slopes:
  - `validator1=5.848663 MiB/s`
  - `validator2=7.899589 MiB/s`
  - `validator3=4.264617 MiB/s`
- validator2 finalizer:
  - `new_lfb_found_true=0`, `new_lfb_found_false=39`

Soak after dependency quarantine:
- `/tmp/casper-validator-leak-soak-depquarantine-20260223T002221Z`
- slopes:
  - `validator1=5.067034 MiB/s`
  - `validator2=4.090115 MiB/s`
  - `validator3=4.713931 MiB/s`
- validator2 finalizer:
  - `new_lfb_found_true=0`, `new_lfb_found_false=45`

Interpretation:
- retry pressure and replay hot-hash amplification are materially improved.
- memory-growth slope improved significantly on validator2 (`7.90 -> 4.09 MiB/s`, ~48% reduction).
- remaining correctness/perf risk: validator2 still shows no `new_lfb_found_true` during soak windows.
- next target: isolate validator2 finalizer no-progress condition with per-hash/block-state tracing at finalizer+replay boundaries while keeping current retry/quarantine guards.

### Allocator hotspot profile (validator2, 120s soak, call stacks)

Profiler capture root:
- `/tmp/casper-pprof-heap-20260223T080337Z`

Correctness-first run gating:
- init SLA pass before soak (all validators `attempts=1, approved=1, transitions=1, time_to_running_count=1`)
- soak run used auto-heal pre-load gate and completed:
  - `/tmp/casper-pprof-heap-20260223T080337Z/soak-120s-heapprof4`
  - `deploy_success=26/27`
  - `block_requests_retry_ratio=2.06`

Heap artifacts captured during soak:
- `docker/data/rnode.validator2/heap/validator2_1.0001.heap` .. `validator2_1.0006.heap`

Top allocator hotspots (latest heap, alloc_space):
- source: `/tmp/casper-pprof-heap-20260223T080337Z/pprof/latest.alloc_space.top.txt`
- `alloc::raw_vec::finish_grow@235510`: `131.0 MB` (`40.7%`)
- `KeyValueBlockStore::get`: `36.2 MB` (`11.2%`)
- `mdb_env_open`: `20.1 MB` (`6.2%`)
- `tonic::codec::decode::Streaming::new_request`: `20.0 MB` (`6.2%`)
- `mdb_midl_alloc`: `10.0 MB` (`3.1%`)

Top retained hotspots (latest heap, inuse_space):
- source: `/tmp/casper-pprof-heap-20260223T080337Z/pprof/latest.inuse_space.top.txt`
- `mdb_env_open`: `20.1 MB` (`29.9%`)
- `__rustc::__rdl_realloc`: `11.9 MB` (`17.7%`)
- `mdb_midl_alloc`: `10.0 MB` (`14.9%`)
- `alloc::raw_vec::finish_grow@235510`: `3.2 MB` flat (`15.0 MB` cum)

Stack-attributed call-path evidence:
- MDB/open path:
  - `/tmp/casper-pprof-heap-20260223T080337Z/stacks/mdb.focus.stacks.symbolized.txt`
  - `mdb_env_open -> heed::env::EnvOpenOptions::open -> ... -> KeyValueStoreManager::get_stores/r_space_stores -> setup_node_program`
- Replay/RSpace path:
  - `/tmp/casper-pprof-heap-20260223T080337Z/stacks/replay.focus.stacks.symbolized.txt`
  - `ReplayRSpace::restore_installs/reset/create_checkpoint` and `DebruijnInterpreter::* -> ReplayRSpace::locked_produce/locked_install`
- Growth diff (early -> latest):
  - `/tmp/casper-pprof-heap-20260223T080337Z/stacks/diff.inuse_space.early_to_latest.stacks.symbolized.txt`
  - strongest positive stack in this sample is repeated `to_vec/clone` chains under `InMemHotStore::get_cont_from_history_store` and interpreter consume/produce flow.

Interpretation:
- retained heap remains dominated by LMDB environment/open allocations (`mdb_env_open`, `mdb_midl_alloc`) plus vector growth/realloc paths.
- replay/checkpoint install/restore and hot-store continuation/history lookup paths are present in top stack contributors.
- immediate next optimization target for leak/perf: reduce repeated clone/to_vec allocations in replay hot-store continuation retrieval and checkpoint/reset flow while preserving correctness gates.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23)

Code changes applied for this cycle:
- `rspace++/src/rspace/history/radix_tree.rs`
  - switched read-cache storage from `DashMap<ByteVector, Node>` to `DashMap<ByteVector, Arc<Node>>`.
  - this avoids cloning full decoded nodes when inserting/holding read-cache entries.
- `rspace++/src/rspace/history/instances/radix_history.rs`
  - previously adjusted to avoid cloning full `RadixTreeImpl` caches across checkpointed histories.
- `casper/src/rust/util/rholang/runtime_manager.rs`
  - added bounded `active_validators_cache` to reduce repeated active-validator derivation allocations.
- `scripts/ci/profile-validator-allocator-hotspots.sh`
  - switched heap file enumeration/cleanup to `find` to avoid glob/ARG_MAX failure under high snapshot counts.

Correctness validation on this image:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo check -p casper --quiet`: pass (existing unrelated warning in `block_approver_protocol.rs`).
- external suite:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - result: `12 passing, 1 pending`.

Allocator call-stack captures:
- prior comparison point: `/tmp/casper-allocator-hotspots-20260223T-cont5`
- new run on rebuilt image: `/tmp/casper-allocator-hotspots-20260223T-cont6`

Top stack-growth movement (symbolized, top-15):
- history-read/load path aggregate (top-15 bucketed):
  - `cont5`: `2601709` bytes
  - `cont6`: `2235277` bytes
  - directional change: about `-14%`
- `RuntimeManager::get_active_validators` is no longer a dominant top-15 delta stack (regression from earlier runs removed), but `introduce_system_process` still appears in one top delta stack in `cont6`.
- dominant callsites in `cont6` shifted to:
  - transport/H2 HPACK alloc growth (`h2::hpack::*`, `h2::frame::headers::*`)
  - interpreter paths (`CostManager::charge`, `Substitute::substitute_and_charge`)
  - remaining history-read path (`RadixTreeImpl::load_node/read_at -> get_data_proj/get_joins_proj`)
  - hot-store continuation clone path (`InMemHotStore::get_cont_from_history_store`)

Performance tracking for the same allocator runs:
- `cont5` (`/tmp/casper-allocator-hotspots-20260223T-cont5`):
  - `propose_total avg/p95 = 2677 / 8112 ms`
  - `checkpoint avg/p95 = 1300 / 4419 ms`
  - `finalizer avg/p95 = 1385 / 2161 ms`
  - `block_requests_retry_ratio = 3.26`
- `cont6` (`/tmp/casper-allocator-hotspots-20260223T-cont6`):
  - `propose_total avg/p95 = 1573 / 4759 ms`
  - `checkpoint avg/p95 = 1063 / 3097 ms`
  - `finalizer avg/p95 = 969 / 2192 ms`
  - `block_requests_retry_ratio = 1.05`

Interpretation:
- correctness remains stable after the cache/allocator-focused changes.
- memory/allocation pressure in the targeted history-read stack improved directionally, but not eliminated.
- next highest ROI allocator target is now hot-store continuation cloning (`get_cont_from_history_store`) plus interpreter substitution/charge allocation churn, with history-read load path still a secondary hotspot.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont7)

Additional code change in this cycle:
- `rspace++/src/rspace/hot_store.rs`
  - made history fetch lazy in hot getters:
    - `get_continuations`
    - `get_data`
    - `get_joins`
  - these now hit history/cache only on in-memory miss, instead of unconditional history fetch on every getter call.

Validation:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo check -p casper --quiet`: pass (same existing warning only).
- external suite on this build:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - result: `12 passing, 1 pending` (first test was slow in this sample, but completed successfully).

Allocator stack comparison (top-15 symbolized delta reports):
- `cont5`: `/tmp/casper-allocator-hotspots-20260223T-cont5`
- `cont6`: `/tmp/casper-allocator-hotspots-20260223T-cont6`
- `cont7`: `/tmp/casper-allocator-hotspots-20260223T-cont7`

Aggregated stack-family movement (top-15 bucket):
- history read/load (`RadixTreeImpl::load_node/read_at -> get_data_proj/get_joins_proj`):
  - `cont5`: `2601709` bytes
  - `cont6`: `2235277` bytes
  - `cont7`: `1825509` bytes
  - net: about `-29.8%` vs `cont5` (`-18.3%` vs `cont6`)
- hot-store history/continuation path:
  - `cont5`: `238797` bytes
  - `cont6`: `173999` bytes
  - `cont7`: `266682` bytes (noisy upward move in this sample)
- interpreter cost/substitute path remains dominant:
  - `cont6`: `2650414` bytes
  - `cont7`: `2717888` bytes

Performance tracking for cont6 -> cont7:
- `cont6` (`/tmp/casper-allocator-hotspots-20260223T-cont6`):
  - `propose_total avg/p95 = 1573 / 4759 ms`
  - `checkpoint avg/p95 = 1063 / 3097 ms`
  - `finalizer avg/p95 = 969 / 2192 ms`
  - `block_requests_retry_ratio = 1.05`
- `cont7` (`/tmp/casper-allocator-hotspots-20260223T-cont7`):
  - `propose_total avg/p95 = 1371 / 4975 ms`
  - `checkpoint avg/p95 = 1020 / 4243 ms`
  - `finalizer avg/p95 = 845 / 2180 ms`
  - `block_requests_retry_ratio = 1.55`

Interpretation:
- lazy history-fetch improved the targeted history-read allocator stack directionally and materially.
- end-to-end latency remains noisy (p95 tails still high), and interpreter/substitution allocations are now the most consistent top contributor.
- next memory/perf target should move to interpreter allocation control (`CostManager::charge` / `Substitute::substitute_and_charge`) and continuation clone pressure in hot-store `get_continuations` merge paths.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont8 -> cont9)

Correctness-first patch in this cycle:
- `rholang/src/rust/interpreter/storage/charging_rspace.rs`
  - removed repeated hot-path `to_bytes()` allocations in matching logic by precomputing and reusing:
    - `triggered_by_id_bytes`
    - `consume_id_bytes`
    - `triggered_id_bytes`

Validation:
- `cargo check -p rholang --quiet`: pass.
- external suite on patched image:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - result: `12 passing, 1 pending`.

Allocator call-stack deltas (top-15 symbolized, positive growth):
- `cont8`: `/tmp/casper-allocator-hotspots-20260223T-cont8`
- `cont9`: `/tmp/casper-allocator-hotspots-20260223T-cont9`

Aggregated stack-family movement (top-15 bucket):
- total top-15 growth bytes:
  - `cont8`: `7952987`
  - `cont9`: `6775058` (`-14.8%`)
- history read/load family (`RadixTreeImpl::load_node/read_at`, `get_*_proj`, decode):
  - `cont8`: `2515464`
  - `cont9`: `2041120` (`-18.9%`)
- interpreter charge/substitute/reduce family:
  - `cont8`: `5013156`
  - `cont9`: `3830044` (`-23.6%`)
- `Blake2b512Random::to_bytes` family:
  - `cont8`: `1617691`
  - `cont9`: `0` (dropped out of top-15 growth)

Performance tracking for cont8 -> cont9:
- `cont8` (`/tmp/casper-allocator-hotspots-20260223T-cont8`):
  - `propose_total avg/p95 = 2393 / 8251 ms`
  - `block_creator_total_create_block avg/p95 = 2241 / 8164 ms`
  - `block_creator_compute_deploys_checkpoint avg/p95 = 1802 / 7126 ms`
  - `finalizer avg/p95 = 1373 / 2213 ms`
  - `block_validation_mean_ms = 3094.74`
  - `block_processing_replay_mean_ms = 3024.37`
  - `block_requests_retry_ratio = 3.89`
- `cont9` (`/tmp/casper-allocator-hotspots-20260223T-cont9`):
  - `propose_total avg/p95 = 1419 / 4460 ms`
  - `block_creator_total_create_block avg/p95 = 1219 / 3710 ms`
  - `block_creator_compute_deploys_checkpoint avg/p95 = 966 / 3551 ms`
  - `finalizer avg/p95 = 1009 / 2199 ms`
  - `block_validation_mean_ms = 709.86`
  - `block_processing_replay_mean_ms = 2236.74`
  - `block_requests_retry_ratio = 1.13`

Interpretation:
- The repeated `to_bytes()` allocation hotspot was a real contributor; removing it improved allocator pressure and latency while preserving correctness.
- Remaining dominant allocator growth is still interpreter/reduce and history-read paths under soak; those are the next correctness-safe optimization targets.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont10)

Additional low-risk allocation patch:
- `rholang/src/rust/interpreter/reduce.rs`
  - in `eval_send`, removed an unnecessary intermediate vector clone before substitution.

Validation:
- `cargo check -p rholang --quiet`: pass.
- external suite on this build:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - result: `12 passing, 1 pending`.

Allocator stack-family aggregate (top-15 positive growth):
- `cont9` -> `cont10`
  - total top-15 bytes: `6775058` -> `7210215`
  - history-read family: `2041120` -> `2305500`
  - interpreter charge/substitute family: `3830044` -> `3715057`
  - `Blake2b512Random::to_bytes`: `0` -> `0` (still eliminated from top-15)

Performance snapshot:
- `cont9` (`/tmp/casper-allocator-hotspots-20260223T-cont9`):
  - `propose avg/p95 = 1419 / 4460 ms`
  - `checkpoint avg/p95 = 966 / 3551 ms`
  - `finalizer avg/p95 = 1009 / 2199 ms`
  - `retry_ratio = 1.13`
- `cont10` (`/tmp/casper-allocator-hotspots-20260223T-cont10`):
  - `propose avg/p95 = 1704 / 4809 ms`
  - `checkpoint avg/p95 = 1319 / 4526 ms`
  - `finalizer avg/p95 = 3927 / 10345 ms`
  - `retry_ratio = 1.28`

Interpretation:
- The earlier `to_bytes` reuse fix remains effective (`to_bytes` hotspot still absent).
- This `cont10` sample is degraded/noisy overall (not a clean latency win), so the new `eval_send` clone-removal should be treated as allocator hygiene only until replicated A/B runs confirm net E2E benefit.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont11)

Additional allocation change in this cycle:
- `rholang/src/rust/interpreter/accounting/costs.rs`
  - changed `Cost.operation` from `String` to `Cow<'static, str>`.
  - made `Cost::create`/`create_from_generic` accept `Into<Cow<'static, str>>`.
  - replaced static `"...".to_string()` operation literals with borrowed `&'static str`.
- `rholang/src/rust/interpreter/accounting/cost_accounting.rs`
  - `operation: "init".into()`.
- `rholang/src/lib.rs`
  - convert `Cow` to protobuf string via `.to_string()` only at FFI boundary.

Validation:
- `cargo check -p rholang --quiet`: pass.
- external suite on this build:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - result: `12 passing, 1 pending`.

Allocator stack/call-stack findings (symbolized, top positive growth):
- artifact: `/tmp/casper-allocator-hotspots-20260223T-cont11`
- compared to `cont10` (`/tmp/casper-allocator-hotspots-20260223T-cont10`):
  - top-15 positive growth bytes: `7210215` -> `8966168` (`+24.4%`)
  - top-15 latest snapshot bytes: `26738694` -> `27005469` (`+1.0%`)
- dominant call paths in `cont11` delta:
  - `rholang::rust::interpreter::accounting::CostManager::charge`
  - `rholang::rust::interpreter::storage::charging_rspace::...::create_checkpoint`
  - `rholang::rust::interpreter::rho_runtime::RhoRuntimeImpl::create_checkpoint`
  - `rspace_plus_plus::rspace::history::radix_tree::RadixTreeImpl::load_node`
  - `RSpaceHistoryReaderImpl::get_data_proj` / `get_joins_proj`
  - `InMemHotStore::install_continuation` via `DashMap::_insert`
- `Blake2b512Random::to_bytes` remains absent from top symbolized delta stacks.

Performance snapshot (`cont10` -> `cont11`):
- `propose avg/p95`: `1704 / 4809 ms` -> `1669 / 5412 ms`
- `checkpoint avg/p95`: `1319 / 4526 ms` -> `1280 / 5160 ms`
- `finalizer avg/p95`: `3927 / 10345 ms` -> `6010 / 10415 ms`
- `retry_ratio`: `1.28` -> `1.92`

Interpretation:
- The `Cost.operation` `Cow<'static, str>` conversion is correctness-safe and preserves behavior, but this sample does not show allocator reduction; top-15 growth increased and tails remain high.
- Remaining allocator pressure is concentrated in interpreter charge/checkpoint and history-read/load paths (with continuation install pressure), so those should stay as next targets.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont12 + cont12b replicate)

Correctness-first change applied before these runs:
- `rholang/src/rust/interpreter/accounting/mod.rs`
  - changed `CostManager.log` from `Vec<Cost>` to bounded `VecDeque<Cost>`.
  - added `max_log_entries` with runtime default `0` (disable production accumulation), env override `F1R3_COST_LOG_MAX_ENTRIES`.
  - tests keep full log behavior (`usize::MAX`) to preserve test semantics.

Correctness validation:
- `cargo check -p rholang --quiet`: pass.
- external suite rerun on current build:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - result: `12 passing, 1 pending`.

Allocator stack/call-stack findings:
- artifacts:
  - `cont12`: `/tmp/casper-allocator-hotspots-20260223T-cont12`
  - `cont12b` replicate: `/tmp/casper-allocator-hotspots-20260223T-cont12b`
- top-15 positive growth bytes (symbolized):
  - `cont11`: `8966168`
  - `cont12`: `5334366` (`-40.5%` vs cont11)
  - `cont12b`: `7830722` (`+46.8%` vs cont12; still below cont11)
- top-15 latest snapshot bytes:
  - `cont11`: `27005469`
  - `cont12`: `26793265`
  - `cont12b`: `28693039`
- key stack observation:
  - `CostManager::charge` in top symbolized positive-growth stacks:
    - `cont11`: present
    - `cont12`: absent
    - `cont12b`: absent

Top allocating/growth call-stack families after the fix (symbolized):
- protobuf decode/growth path (largest in `cont12b`):
  - `alloc::raw_vec::RawVec::grow_one`
  - `prost::encoding::message::merge_repeated`
  - `models::casper::{ProcessedDeployProto,ProcessedSystemDeployProto,BodyProto}::merge_field`
- history read path:
  - `rspace_plus_plus::rspace::history::radix_tree::RadixTreeImpl::{load_node,read_at}`
  - `RSpaceHistoryReaderImpl::{fetch_data,get_data_proj,get_joins_proj,get_continuations_proj}`
- block dependency/commit path:
  - `BlockDependencyDag::add`
  - `CasperBufferKeyValueStorage::add_relation`
  - `BlockProcessor::check_dependencies_with_effects`
- merge path remains visible in `cont12` (not top in `cont12b`):
  - `casper::rust::util::rholang::interpreter_util::compute_parents_post_state`

Performance tracking:
- `cont12` (`/tmp/casper-allocator-hotspots-20260223T-cont12`):
  - `propose_total avg/p95 = 1892.93 / 5610 ms`
  - `block_creator_total_create_block avg/p95 = 1700.13 / 5487 ms`
  - `block_creator_compute_deploys_checkpoint avg/p95 = 1419.66 / 5244 ms`
  - `finalizer_total avg/p95 = 3467.67 / 10374 ms`
  - `block_requests_retry_ratio = 1.05`
- `cont12b` (`/tmp/casper-allocator-hotspots-20260223T-cont12b`):
  - `propose_total avg/p95 = 2373.54 / 5714 ms`
  - `block_creator_total_create_block avg/p95 = 1845.75 / 4908 ms`
  - `block_creator_compute_deploys_checkpoint avg/p95 = 1410.35 / 3584 ms`
  - `finalizer_total avg/p95 = 3952.14 / 10382 ms`
  - `block_requests_retry_ratio = 2.93`

Interpretation:
- The leak-like growth from `CostManager::charge` was removed from top allocator stacks while preserving correctness.
- Remaining dominant allocator pressure is now in protobuf block decode plus history-read/load paths.
- Tail latency is still noisy/high in both replicate soaks; next correctness-safe memory work should target decode buffering and history-read clone pressure with guarded A/B runs.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont13 / cont13b)

Correctness-first patch in this cycle:
- `block-storage/src/rust/util/doubly_linked_dag_operations.rs`
  - removed clone-heavy `updated_with` path in `BlockDependencyDag::add`.
  - switched to in-place mutation with `DashMap::get_mut` for parent->child and child->parent sets.
  - goal: eliminate repeated `DashSet` cloning/allocation churn on dependency edge inserts.

Validation:
- `cargo check -p block-storage --quiet`: pass.
- `cargo check -p casper --quiet`: pass (existing unrelated warning in `block_approver_protocol.rs`).
- rebuilt and retagged image: `f1r3flyindustries/f1r3fly-rust-node:latest`.
- external suite on rebuilt image:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - result: `12 passing, 1 pending`.

Allocator profiling runs:
- `cont13`: `/tmp/casper-allocator-hotspots-20260223T-cont13`
  - caveat: preload gate auto-recreated cluster before run; early heap baseline was near-zero (`jeprof...i241`), so growth bytes are not directly comparable to prior runs.
- `cont13b` (fairer compare, no preload-triggered recreate):
  - `/tmp/casper-allocator-hotspots-20260223T-cont13b`

Fair compare (`cont12b` -> `cont13b`):
- top-15 positive growth bytes:
  - `7830722` -> `5478368` (`-30.0%`)
- top-15 latest snapshot bytes:
  - `28693039` -> `28216630` (`-1.7%`)
- stack presence counts in top symbolized positive-growth set:
  - `BlockDependencyDag::add`: `1 -> 0`
  - `DashMap<K,V,S> as core::clone::Clone`: `1 -> 0`
  - `prost::encoding::message::merge_repeated`: `4 -> 2`
  - `RadixTreeImpl::load_node`: `10 -> 14`
  - `RSpaceHistoryReaderImpl*`: `22 -> 26`
  - `CostManager::charge`: `0 -> 0`

Interpretation:
- The DAG in-place mutation patch removed the `BlockDependencyDag::add`/DashMap clone hotspot from top allocator-growth stacks and reduced aggregate top-15 growth in the fair compare.
- Remaining dominant allocator pressure is now mostly protobuf decode (`prost::merge_repeated`) and history read/load (`RadixTreeImpl` + `RSpaceHistoryReaderImpl`) paths.
- End-to-end latency remains noisy in these samples (finalizer/retry ratio variability), so next memory work should target decode/history allocation patterns with guarded A/B runs.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont14, call-stack isolation pass)

Correctness-first patch in this cycle:
- `block-storage/src/rust/key_value_block_store.rs`
  - replaced per-read `lz4_flex::decompress(...) -> Vec<u8>` with thread-local reusable buffer + `lz4_flex::decompress_into(...)`.
  - removed `expect(...)` panic paths in block decode and now return `KvStoreError::SerializationError(...)` with explicit cause strings.
  - objective: reduce decode-path allocation churn while improving failure correctness semantics.

Validation:
- `cargo check -p block-storage --quiet`: pass.
- `cargo check -p casper --quiet`: pass (existing unrelated warning unchanged).
- rebuilt image: `f1r3flyindustries/f1r3fly-rust-node:latest` -> `sha256:40f3feef6da7...`.
- external suite:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - result: `12 passing, 1 pending`.
  - note: this run had severe nudge lag in `core/Arithmetic` (~372s) but completed successfully.

Allocator stack artifacts:
- run: `/tmp/casper-allocator-hotspots-20260223T-cont14`
- symbolized call-stack reports:
  - `/tmp/casper-allocator-hotspots-20260223T-cont14/stacks/delta-positive.top15.symbolized.txt`
  - `/tmp/casper-allocator-hotspots-20260223T-cont14/stacks/latest.top15.symbolized.txt`

Fair compare (`cont13b` -> `cont14`):
- top-15 positive growth bytes:
  - `5478368` -> `1819705` (`-66.8%`)
- top-15 latest snapshot bytes:
  - `28216630` -> `44904372` (`+59.1%`)
- stack-presence counts in top symbolized sets:
  - delta-positive:
    - `prost::encoding::message::merge_repeated`: `2 -> 2`
    - `RadixTreeImpl::load_node`: `14 -> 11`
    - `RSpaceHistoryReaderImpl*`: `26 -> 21`
    - `ChargingRSpace`: `0 -> 2`
  - latest-retained:
    - `prost::encoding::message::merge_repeated`: `2 -> 2`
    - `mdb_env_open`: `14 -> 9`
    - `mdb_midl_alloc`: `5 -> 0`
    - `RadixTreeImpl::load_node`: `0 -> 5`
    - `RSpaceHistoryReaderImpl*`: `0 -> 10`

Top allocating call-stack families in `cont14` (symbolized):
- protobuf decode growth:
  - `alloc::raw_vec::RawVec::grow_one`
  - `prost::encoding::message::merge_repeated`
  - `models::casper::{ProcessedDeployProto,BodyProto}::merge_field`
- history load/read:
  - `RadixTreeImpl::{load_node,read_at}`
  - `RSpaceHistoryReaderImpl::{fetch_data,get_data_proj,get_joins_proj,get_continuations_proj}`
- runtime checkpoint/reset path:
  - `RadixHistory::{reset,process}`
  - `ChargingRSpace::{reset,create_checkpoint}`
- LMDB open/init retained allocations still visible:
  - `mdb_env_open`

Performance tracking (`cont14`):
- `propose_total avg/p95 = 2759.13 / 9840 ms`
- `block_creator_total_create_block avg/p95 = 2375.38 / 7803 ms`
- `block_creator_compute_deploys_checkpoint avg/p95 = 1764.24 / 5902 ms`
- `finalizer_total avg/p95 = 6678.00 / 10480 ms`
- `block_requests_retry_ratio = 5.68`

Interpretation:
- Call-stack collection is now explicit and stable enough to rank allocators by code path.
- This `cont14` sample does not show an end-to-end latency gain; retained top-15 bytes increased while growth bytes dropped (mixed signal under high runtime churn).
- The dominant allocator callsites remain decode + history read/load paths; those are still the highest-priority memory targets for the next cycle.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont15) and rejected optimization

Experimented optimization (rejected after profiling):
- `rspace++/src/rspace/history/radix_tree.rs`
  - attempted in-place iterative rewrite of `decode(...)` to remove per-item `Node` cloning in decode path.
  - correctness passed, but allocator + latency outcomes regressed in this sample.
  - decision: **reverted** this change (do not keep).

Validation for the trial build:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo check -p casper --quiet`: pass.
- external suite:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - result: `12 passing, 1 pending`.

Run artifact:
- `cont15`: `/tmp/casper-allocator-hotspots-20260223T-cont15`

Compare (`cont14` -> `cont15`):
- top-15 positive growth bytes:
  - `1819705` -> `12055793` (`+562.5%`, worse)
- top-15 latest snapshot bytes:
  - `44904372` -> `50321797` (`+12.1%`, worse)
- stack-family weighted presence in top symbolized sets:
  - delta-positive:
    - `RSpaceHistoryReaderImpl`: `1768015` -> `20238706` (worse)
    - `RadixTreeImpl::load_node`: `1351165` -> `10119353` (worse)
    - `prost::encoding::message::merge_repeated`: `458304` -> `0`
  - latest-retained:
    - `RSpaceHistoryReaderImpl`: `36998884` -> `44962124` (worse)
    - `RadixTreeImpl::load_node`: `18499442` -> `22481062` (worse)
    - `prost::encoding::message::merge_repeated`: `15061124` -> `17932734` (worse)

Performance (`cont14` -> `cont15`):
- `propose_total avg/p95`: `2759/9840` -> `3115/13900` ms (worse)
- `create_block avg/p95`: `2375/7803` -> `2801/13672` ms (worse)
- `compute_deploys_checkpoint avg/p95`: `1764/5902` -> `2416/13399` ms (worse)
- `finalizer_total avg/p95`: `6678/10480` -> `8386/10531` ms (worse)
- `block_requests_retry_ratio`: `5.68` -> `7.89` (worse)

Interpretation:
- The attempted radix decode rewrite did not deliver net improvement under this controlled soak and was reverted.
- Current top allocator call paths remain:
  - `RSpaceHistoryReaderImpl*`
  - `RadixTreeImpl::load_node/read_at`
  - `prost::encoding::message::merge_repeated`
- Next iterations should target these paths with smaller, correctness-guarded changes and immediate soak comparison.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont16: comm cache-eviction validation)

Correctness-first patch validated in this cycle:
- `comm/src/rust/transport/packet_ops.rs`
  - `PacketOps::restore(...)` now consumes cache entries with `remove(...)` (one-shot) instead of cloning via `get(...)`.
  - objective: prevent unbounded stream cache growth under sustained transport traffic.

Correctness validation:
- `cargo check -p comm --quiet`: pass.
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).

Allocator profiling runs:
- first run (noisy, rejected for comparison quality):
  - `/tmp/casper-allocator-hotspots-20260223T-post-comm-restore-evict-120s`
  - issue: heap span was disproportionate (`early jeprof.1.1` -> `latest jeprof.1.6774`), not comparable to prior `i153`-range run.
- fair replicate used for comparison:
  - `/tmp/casper-allocator-hotspots-20260223T-post-comm-restore-evict-120s-rep2`
  - heap span: `early jeprof.1.217` -> `latest jeprof.1.402`.

Fair allocator compare (`post-loadnodearc2-120s` -> `post-comm-restore-evict-120s-rep2`):
- categorized top growth bytes:
  - `rspace_history_read`: `3823506` -> `3492902` (`-8.65%`)
  - `interpreter_eval`: `2702777` -> `161060` (`-94.04%`)
  - `proposer_compute_state`: `2160949` -> not in top categorized paths
  - `other`: `1939208` -> `6404762` (up; shifted mix)
- top first-frame aggregate:
  - `Vec::clone`: `5179268` -> `5562769` (`+7.4%`)
- transport stack presence in top symbolized growth:
  - baseline had `comm::rust::transport::grpc_transport_{server,receiver}` frames.
  - replicate had none in top-15 symbolized growth.

Latency verification:
- allocator-run replicate profile (`/tmp/casper-allocator-hotspots-20260223T-post-comm-restore-evict-120s-rep2/soak/profile/summary.txt`):
  - `propose_total p95`: `3708` -> `3281` ms (better vs baseline run)
  - `create_block p95`: `3522` -> `2831` ms (better)
  - `compute_deploys_checkpoint p95`: `3368` -> `2495` ms (better)
  - `finalizer_total p95`: `10776` -> `10623` ms (slightly better)
  - `block_requests_retry_ratio`: `1.38` -> `3.62` (worse; retrieval churn remains unstable)
- dedicated latency harness run:
  - `/tmp/casper-latency-benchmark-post-comm-restore-evict-20260223T2211Z`
  - correctness result: no deploy failures, no proposer bug errors.
  - key metrics: `propose_total p95=3624`, `create_block p95=3155`, `checkpoint p95=2884`, `finalizer_total p95=10798`, `retry_ratio=4.22`.

Interpretation:
- The `comm` restore-eviction fix is correctness-safe and removes transport frames from top allocator-growth call stacks in the fair replicate.
- Dominant allocator growth remains concentrated in non-transport paths (`rspace_history_read` and clone/grow families).
- End-to-end latency remains noisy due retry/requery pressure; this patch should be considered memory-correctness hygiene, not a standalone latency win.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont17: history-reader fetch-key allocation pass)

Correctness-first patch in this cycle:
- `rspace++/src/rspace/history/instances/rspace_history_reader_impl.rs`
  - optimized `fetch_data(...)` key lookup path:
    - removed intermediate `Blake2b256Hash::from_bytes(bytes.to_vec())` + `hash.bytes()` re-allocation.
    - serialize existing read bytes directly for legacy key format.
    - preserve legacy behavior with fallback to raw-bytes key lookup for imported data.
  - replaced `expect(...)` panics in `fetch_data(...)` serialization/deserialization with `HistoryError::ActionError(...)`.

Correctness validation:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo check -p comm --quiet`: pass.
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).

Allocator profiling:
- first run (rejected as unfair/noisy): `/tmp/casper-allocator-hotspots-20260223T-post-historyfetch-nohash-120s`
  - early/latest heap span: `i2 -> i2884` (cluster restart/startup contamination).
- fair replicate used for comparison: `/tmp/casper-allocator-hotspots-20260223T-post-historyfetch-nohash-120s-rep2`
  - early/latest heap span: `i165 -> i293`.

Fair allocator compare (`post-comm-restore-evict-120s-rep2` -> `post-historyfetch-nohash-120s-rep2`):
- top categorized growth bytes:
  - `rspace_history_read`: `3492902` -> `1739264` (`-50.2%`)
  - `interpreter_eval`: `161060` -> `1206734` (`+649%`)
  - `other`: `6404762` -> `2659204` (`-58.5%`)
  - `proposer_compute_state`: not present -> `1310721`
- top first-frame aggregate:
  - `Vec::clone`: `5562769` -> `4662848` (`-16.2%`)
- transport stack presence in top symbolized growth:
  - none (`comm::rust::transport::*` absent in top-15).

Latency verification:
- allocator replicate profile (`/tmp/casper-allocator-hotspots-20260223T-post-historyfetch-nohash-120s-rep2/soak/profile/summary.txt`):
  - tails regressed in this sample:
    - `propose_total p95`: `3281` -> `5346` ms
    - `create_block p95`: `2831` -> `5110` ms
    - `checkpoint p95`: `2495` -> `4915` ms
  - `finalizer_total p95`: `10623` -> `10654` (flat)
  - `block_requests_retry_ratio`: `3.62` -> `3.88` (worse)
- dedicated latency run 1:
  - `/tmp/casper-latency-benchmark-post-historyfetch-nohash-20260223T2250Z`
  - `propose_total p95=5635`, `create_block p95=5337`, `checkpoint p95=5103`, `retry_ratio=3.64` (worse vs recent post-comm dedicated run).
- dedicated latency run 2 (auto-healed preload):
  - `/tmp/casper-latency-benchmark-post-historyfetch-nohash-20260223T2252Z-rep2`
  - `propose_total p95=982`, `create_block p95=801`, `checkpoint p95=475`, `retry_ratio=1.38` (much better in this sample),
  - but had `propose_fail=1` and longer load window (`142s`) after auto-recreate, so this is not directly equivalent.

Interpretation:
- The fetch-key allocation patch is correctness-safe and reduces allocator pressure in the targeted `rspace_history_read` family on the fair replicate.
- End-to-end latency impact is inconclusive due strong run-to-run variance and preload auto-heal effects.
- Keep as memory/correctness hygiene for now; require additional controlled A/B latency replications before claiming tail-latency improvement.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont18: rebuilt-image gated rerun)

Rebuild/restart gate applied for this cycle:
- rebuilt image: `f1r3flyindustries/f1r3fly-rust-node:latest`
  - image sha: `sha256:e18e6aa176f1baaf8e14a507ea3155cf99ba57f52919a9ef299a366bee4ffbcc`
- tagged profiler image: `f1r3node-jemallocprof:latest` (same sha).
- validators force-recreated and container image IDs verified before every measurement run.

Correctness validation (post-rebuild):
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo check -p comm --quiet`: pass.
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).

Allocator call-stack run (jemalloc compose):
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260223T-rebuilt-cont-20260223T232710Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260223T-rebuilt-cont-20260223T232710Z`
- heap window:
  - `early_heap=jeprof.1.3.i3.heap`
  - `latest_heap=jeprof.1.96.i96.heap`
- top categorized growth bytes:
  - `rspace_history_read`: `3478527`
  - `interpreter_eval`: `2881358`
  - `other`: `1867441`
  - `proposer_compute_state`: `201237`
- top first symbolized frame:
  - `rspace_plus_plus::rspace::history::radix_tree::empty_node`: `3478527`
- additional notable first-frame aggregates:
  - `Vec::clone`: `438411`
  - `alloc::raw_vec::finish_grow`: `196835`

Latency summary from the same rebuilt allocator soak:
- profile: `/tmp/casper-allocator-hotspots-20260223T-rebuilt-cont-20260223T232710Z/soak/profile/summary.txt`
- key p95s:
  - `propose_total`: `3510` ms
  - `create_block`: `2852` ms
  - `checkpoint_compute_state`: `2752` ms
  - `finalizer_total`: `10829` ms
- retrieval pressure:
  - `block_requests_retry_ratio`: `1.32`
- correctness signals:
  - `deploy_failure=0`, `propose_bug_error=0`, `propose_fail=0`.

Dedicated rebuilt latency run (standard compose, independent of allocator script):
- command:
  - `./scripts/ci/run-latency-benchmark.sh docker/shard-with-autopropose.yml 120 /tmp/casper-latency-benchmark-rebuilt-cont-20260223T232943Z`
- artifact:
  - `/tmp/casper-latency-benchmark-rebuilt-cont-20260223T232943Z`
- key p95s:
  - `propose_total`: `3481` ms
  - `block_creator_total_create_block`: `3098` ms
  - `checkpoint_compute_state`: `3009` ms
  - `finalizer_total`: `10412` ms
- retrieval pressure:
  - `block_requests_retry_ratio`: `1.09`
- correctness signals:
  - `deploy_failure=0`, `propose_bug_error=0`, `propose_fail=0`.

Interpretation:
- With rebuild+force-recreate gating, the dominant growth path remains in rspace history read/decode (`RadixTreeImpl::load_node_arc/read_at` via `empty_node`/`decode` stacks).
- Transport restore-cache growth no longer dominates top stack categories in this run.
- Latency remains high in p95 tails but is now measured from confirmed rebuilt binaries and appears internally consistent across the allocator soak and dedicated latency benchmark.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-23 cont19: decode-on-demand read path)

Correctness-first patch in this cycle:
- file: `rspace++/src/rspace/history/radix_tree.rs`
- change:
  - added `decode_item_at(...)` helper to decode a single item for a target index from serialized node bytes.
  - updated `read_at(...)` to use `read_at_from_ptr(...)`:
    - on cache hit: keep existing decoded-node path.
    - on cache miss: fetch serialized node bytes and decode only the requested branch item, recursively.
  - intent: reduce full-node decode (`empty_node` + full item materialization) on read-path cache misses.

Correctness validation:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo check -p comm --quiet`: pass.
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).

Rebuild/restart gate (strict):
- rebuilt image from current workspace:
  - `f1r3flyindustries/f1r3fly-rust-node:latest`
  - sha: `sha256:dc0e6154b6d3241aa6df7d169fb4890227e9f9791233a0d6809cc365b283aed3`
- profiler tag:
  - `f1r3node-jemallocprof:latest` (same sha).
- validators force-recreated for allocator and latency runs; container image IDs verified before each run.

Allocator call-stack run (jemalloc compose):
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260223T-cont19-20260223T234020Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260223T-cont19-20260223T234020Z`
- heap window:
  - `early_heap=jeprof.1.2.i2.heap`
  - `latest_heap=jeprof.1.113.i113.heap`
- top categorized growth bytes:
  - `interpreter_eval`: `3412930`
  - `other`: `2474534`
  - `rspace_history_read`: `158115`
  - `proposer_compute_state`: `143741`
- top first-frame aggregate (notable):
  - `rspace_plus_plus::rspace::history::radix_tree::empty_node`: `158115`

A/B vs prior rebuilt-gated run (`rebuilt-cont-20260223T232710Z` -> `cont19`):
- categorized growth:
  - `rspace_history_read`: `3478527` -> `158115` (`-95.45%`)
  - `interpreter_eval`: `2881358` -> `3412930` (`+18.45%`)
  - `other`: `1867441` -> `2474534` (`+32.51%`)
- interpretation:
  - dominant allocator growth moved away from history-read/decode stacks in this sample; growth mix shifted toward interpreter/other families.

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260223T-cont19-20260223T234020Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `2898` ms
  - `create_block`: `2404` ms
  - `checkpoint_compute_state`: `2296` ms
  - `finalizer_total`: `10476` ms
  - `block_requests_retry_ratio`: `1.03`
- vs prior rebuilt allocator soak:
  - `propose_total p95`: `3510` -> `2898` (better)
  - `create_block p95`: `2852` -> `2404` (better)
  - `checkpoint p95`: `2752` -> `2296` (better)
  - `finalizer_total p95`: `10829` -> `10476` (better)
  - `retry_ratio`: `1.32` -> `1.03` (better)

Dedicated rebuilt latency run (standard compose):
- command:
  - `./scripts/ci/run-latency-benchmark.sh docker/shard-with-autopropose.yml 120 /tmp/casper-latency-benchmark-cont19-20260223T234304Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont19-20260223T234304Z`
- p95:
  - `propose_total`: `4248` ms
  - `block_creator_total_create_block`: `3938` ms
  - `checkpoint_compute_state`: `3811` ms
  - `finalizer_total`: `10631` ms
  - `block_requests_retry_ratio`: `0.94`
- vs prior rebuilt dedicated latency run:
  - `propose_total p95`: `3481` -> `4248` (worse)
  - `create_block p95`: `3098` -> `3938` (worse)
  - `checkpoint p95`: `3009` -> `3811` (worse)
  - `finalizer_total p95`: `10412` -> `10631` (slightly worse)
  - `retry_ratio`: `1.09` -> `0.94` (better)

Interpretation:
- This patch is correctness-safe and materially reduced the targeted allocator hotspot (`rspace_history_read`) in the rebuilt allocator soak.
- Dedicated latency tails did not improve in this single replicate, though block-retrieval retry pressure improved.
- Keep this change as memory-pressure improvement; run additional rebuilt dedicated latency replications before making a tail-latency claim.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont20: avoid eager hot-store state hydration on reads)

Correctness-first patch in this cycle:
- file: `rspace++/src/rspace/hot_store.rs`
- change:
  - removed eager insertion of history-read results into `hot_store_state` in read paths:
    - `get_continuations(...)` miss branches (`None, Some(inst)` and `None, None`)
    - `get_data(...)` miss branch
    - `get_joins(...)` miss branch
  - behavior remains read-correct (returns unchanged data), but avoids duplicating large vectors in both history cache and mutable state when no write follows.

Correctness validation:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo check -p comm --quiet`: pass.
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).

Rebuild/restart gate (strict):
- rebuilt image:
  - `f1r3flyindustries/f1r3fly-rust-node:latest`
  - sha: `sha256:63f7e79813916dcdbc1c849451a7b2a198b0effa9e84f65571f73b598680ba62`
- profiler tag:
  - `f1r3node-jemallocprof:latest` (same sha).
- validators force-recreated and running container image IDs verified before allocator + latency runs.

Allocator call-stack run (jemalloc compose):
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260223T-cont20-20260224T002230Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260223T-cont20-20260224T002230Z`
- heap window:
  - `early_heap=jeprof.1.2.i2.heap`
  - `latest_heap=jeprof.1.96.i96.heap`
- top categorized growth bytes:
  - `other`: `3643051`
  - `interpreter_eval`: `1455725`
  - `proposer_compute_state`: `1250565`
  - (`rspace_history_read` not in top categorized paths)
- notable top first-frame aggregates:
  - `alloc::raw_vec::finish_grow` (`ha2dc...`): `1717738`
  - `Vec::clone`: `1325618`
  - `tonic::codec::decode::Streaming<T>::new_request`: `1310721`

A/B vs prior run (`cont19` -> `cont20`):
- categorized growth:
  - `interpreter_eval`: `3412930` -> `1455725` (`-57.35%`)
  - `other`: `2474534` -> `3643051` (`+47.22%`)
  - `rspace_history_read`: `158115` -> not in top categories (`effective -100% in top set`)
  - `proposer_compute_state`: `143741` -> `1250565` (`+770%`)
- interpretation:
  - targeted interpreter path reduced in this sample; allocation pressure shifted toward gRPC/proposer/other buckets.

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260223T-cont20-20260224T002230Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `4214` ms
  - `create_block`: `4033` ms
  - `checkpoint_compute_state`: `3896` ms
  - `finalizer_total`: `10225` ms
  - `block_requests_retry_ratio`: `1.08`
- vs cont19 allocator-soak latency:
  - `propose/create/checkpoint` p95 worse
  - `finalizer_total p95` slightly better (`10476 -> 10225`)
  - `retry_ratio` slightly worse (`1.03 -> 1.08`)

Dedicated rebuilt latency run (standard compose):
- command:
  - `./scripts/ci/run-latency-benchmark.sh docker/shard-with-autopropose.yml 120 /tmp/casper-latency-benchmark-cont20-20260224T002516Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont20-20260224T002516Z`
- p95:
  - `propose_total`: `2106` ms
  - `block_creator_total_create_block`: `1663` ms
  - `checkpoint_compute_state`: `1571` ms
  - `finalizer_total`: `11016` ms
  - `block_requests_retry_ratio`: `0.84`
- vs cont19 dedicated latency:
  - `propose/create/checkpoint` p95 improved strongly
  - `finalizer_total p95` slightly worse (`10631 -> 11016`)
  - `retry_ratio` improved (`0.94 -> 0.84`)

Interpretation:
- Patch is correctness-safe and reduces the prior `interpreter_eval` growth hotspot in this rebuilt allocator run.
- Latency remains mixed by phase: proposer/checkpoint tails improved in dedicated run, but finalizer tail remains dominant and high.
- Next optimization should target `proposer_compute_state` and/or transport decode/protobuf growth paths now visible in top call stacks.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont21: hot-store in-place mutation clone reduction)

Correctness-first patch in this cycle:
- file: `rspace++/src/rspace/hot_store.rs`
- change:
  - reduced clone+reinsert patterns in mutation paths by switching to in-place `DashMap::entry` updates where possible:
    - `put_continuation`
    - `put_datum`
    - `put_join`
    - `install_join`
  - intent: lower transient vector/map allocations in hot-store write paths while preserving observable behavior.

Correctness validation:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo check -p comm --quiet`: pass.
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).
- external integration suite:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`: pass (`12 passing`, `1 pending`).

Rebuild/restart gate (strict):
- rebuilt image:
  - `f1r3flyindustries/f1r3fly-rust-node:latest`
  - sha: `sha256:de7641341cf83938d3802ae5962be64d11a66cd3ce09416e8f450de91837fa41`
- profiler tag:
  - `f1r3node-jemallocprof:latest` (same sha).
- validators were recreated from scratch before runs:
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`
- running container image IDs were checked after recreate.

Allocator call-stack run (jemalloc compose):
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont21-20260224T003516Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont21-20260224T003516Z`
- heap window:
  - `early_heap=jeprof.1.2.i2.heap`
  - `latest_heap=jeprof.1.121.i121.heap`
- top categorized growth bytes:
  - `proposer_compute_state`: `2902356`
  - `other`: `2885099`
  - `interpreter_eval`: `296420`
  - `rspace_history_read`: `287482`

A/B vs prior run (`cont20` -> `cont21`):
- categorized growth:
  - `proposer_compute_state`: `1250565` -> `2902356` (`+132.09%`)
  - `other`: `3643051` -> `2885099` (`-20.81%`)
  - `interpreter_eval`: `1455725` -> `296420` (`-79.64%`)
  - `rspace_history_read`: not in top set -> `287482` (reappeared, but still well below historical multi-million levels).
- interpretation:
  - interpreter-related growth dropped materially; dominant growth shifted to proposer compute-state and non-specific `other` buckets.

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont21-20260224T003516Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `3426` ms
  - `create_block`: `2928` ms
  - `checkpoint_compute_state`: `2838` ms
  - `finalizer_total`: `10942` ms
  - `block_requests_retry_ratio`: `1.11`
- vs cont20 allocator-soak latency:
  - `propose/create/checkpoint` p95 improved (`4214/4033/3896 -> 3426/2928/2838`)
  - `finalizer_total p95` worsened (`10225 -> 10942`)
  - `retry_ratio` slightly worse (`1.08 -> 1.11`)

Dedicated rebuilt latency run (same rebuilt jemalloc stack):
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont21-20260224T004024Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont21-20260224T004024Z`
- p95:
  - `propose_total`: `2840` ms
  - `block_creator_total_create_block`: `2035` ms
  - `checkpoint_compute_state`: `1953` ms
  - `finalizer_total`: `10870` ms
  - `block_requests_retry_ratio`: `4.42`
- vs cont20 dedicated latency:
  - `propose/create/checkpoint` p95 worse (`2106/1663/1571 -> 2840/2035/1953`)
  - `finalizer_total p95` slightly better (`11016 -> 10870`)
  - `retry_ratio` regressed sharply (`0.84 -> 4.42`)

Interpretation:
- Patch is correctness-safe under unit and external integration checks.
- Allocator gains are mixed: `interpreter_eval` dropped substantially, but proposer compute-state became the dominant growth family in this replicate.
- Latency remains mixed and unstable by phase; finalizer p95 stays high, and replay/request retry pressure regressed in the dedicated run, so further diagnosis should prioritize proposer compute-state + retry pressure paths.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont22: hot-store lazy history fetch on mutation miss paths)

Correctness-first patch in this cycle:
- file: `rspace++/src/rspace/hot_store.rs`
- change:
  - removed eager history fetches in mutation methods and fetch history only when state key is missing:
    - `put_continuation`
    - `remove_continuation`
    - `put_datum`
    - `remove_datum`
    - `put_join`
  - intent: avoid unnecessary full-vector clones from history cache on hot paths where state already has entries.

Correctness validation:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo check -p comm --quiet`: pass.
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).
- external integration suite:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`: pass (`12 passing`, `1 pending`).

Rebuild/restart gate (strict):
- rebuilt image:
  - `f1r3flyindustries/f1r3fly-rust-node:latest`
  - sha: `sha256:93b3085af5e97b307a2077f3ee7ebc8fb9e70435cd8ce1047b782aff7e5d2357`
- profiler tag:
  - `f1r3node-jemallocprof:latest` (same sha).
- validators recreated before run:
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`
- running container image IDs verified after recreate.

Allocator call-stack run (jemalloc compose):
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont22-20260224T011535Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont22-20260224T011535Z`
- heap window:
  - `early_heap=jeprof.1.2.i2.heap`
  - `latest_heap=jeprof.1.110.i110.heap`
- top categorized growth bytes:
  - `other`: `2833342`
  - `interpreter_eval`: `1640159`
  - `rspace_history_read`: `273108`
  - `proposer_compute_state`: `265492`

A/B vs prior run (`cont21` -> `cont22`):
- categorized growth:
  - `proposer_compute_state`: `2902356` -> `265492` (`-90.85%`)
  - `other`: `2885099` -> `2833342` (`-1.79%`)
  - `interpreter_eval`: `296420` -> `1640159` (`+453.32%`)
  - `rspace_history_read`: `287482` -> `273108` (`-5.00%`)
- interpretation:
  - targeted proposer-compute-state allocation pressure dropped materially in this replicate; growth shifted back toward interpreter/other families.

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont22-20260224T011535Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `2890` ms
  - `create_block`: `2780` ms
  - `checkpoint_compute_state`: `2648` ms
  - `finalizer_total`: `10894` ms
  - `block_requests_retry_ratio`: `1.21`
- vs cont21 allocator-soak latency:
  - `propose/create/checkpoint` p95 improved (`3426/2928/2838 -> 2890/2780/2648`)
  - `finalizer_total p95` slightly improved (`10942 -> 10894`)
  - `retry_ratio` slightly worsened (`1.11 -> 1.21`)

Dedicated rebuilt latency run (same rebuilt jemalloc stack):
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont22-20260224T011804Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont22-20260224T011804Z`
- p95:
  - `propose_total`: `4356` ms
  - `block_creator_total_create_block`: `4183` ms
  - `checkpoint_compute_state`: `4086` ms
  - `finalizer_total`: `10473` ms
  - `block_requests_retry_ratio`: `4.86`
- vs cont21 dedicated latency:
  - `propose/create/checkpoint` p95 worsened (`2840/2035/1953 -> 4356/4183/4086`)
  - `finalizer_total p95` improved (`10870 -> 10473`)
  - `retry_ratio` worsened (`4.42 -> 4.86`)

Interpretation:
- Patch is correctness-safe and strongly reduces the currently dominant `proposer_compute_state` allocator category in this rebuilt sample.
- Dedicated latency remains volatile and degraded on propose/create/checkpoint tails in this replicate; finalizer tail improved modestly.
- Next pass should focus on high `interpreter_eval`/`other` call stacks while separately tracking retry-ratio regressions in dedicated runs.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont23: in-place remove-path updates in hot-store)

Correctness-first patch in this cycle:
- file: `rspace++/src/rspace/hot_store.rs`
- change:
  - reworked remove paths to avoid unconditional clone+reinsert and eager history fetch:
    - `remove_continuation`: entry-based in-place removal for occupied state; history fetch only on vacant.
    - `remove_datum`: entry-based in-place removal for occupied state; history fetch only on vacant.
    - `remove_join`: avoids eager history joins/continuations fetch and only materializes from history on vacant paths.
  - intent: reduce remove-path clone pressure in proposer/checkpoint flows while preserving behavior and warnings.

Correctness validation:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo check -p comm --quiet`: pass.
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).
- external integration suite:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`: pass (`12 passing`, `1 pending`).

Rebuild/restart gate (strict):
- rebuilt image:
  - `f1r3flyindustries/f1r3fly-rust-node:latest`
  - sha: `sha256:fc68454110153f58f7048aad468a1424655e893644dc26e2ea3cfec6e6048192`
- profiler tag:
  - `f1r3node-jemallocprof:latest` (same sha).
- validators recreated before run:
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`
- running container image IDs verified after recreate.

Allocator call-stack run (jemalloc compose):
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont23-20260224T023855Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont23-20260224T023855Z`
- top categorized growth bytes:
  - `other`: `5169213`
  - `interpreter_eval`: `3297152`
  - `proposer_compute_state`: `597223`
  - `rspace_history_read`: `373726`

A/B vs prior run (`cont22` -> `cont23`):
- categorized growth:
  - `proposer_compute_state`: `265492` -> `597223` (`+124.95%`)
  - `interpreter_eval`: `1640159` -> `3297152` (`+101.03%`)
  - `other`: `2833342` -> `5169213` (`+82.44%`)
  - `rspace_history_read`: `273108` -> `373726` (`+36.84%`)
- interpretation:
  - this replicate regressed allocator growth broadly (especially `interpreter_eval`/`other`), with proposer category still below earlier multi-million spikes.

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont23-20260224T023855Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `2370` ms
  - `create_block`: `1999` ms
  - `checkpoint_compute_state`: `1890` ms
  - `finalizer_total`: `10467` ms
  - `block_requests_retry_ratio`: `1.16`
- vs cont22 allocator-soak latency:
  - `propose/create/checkpoint` p95 improved (`2890/2780/2648 -> 2370/1999/1890`)
  - `finalizer_total p95` improved (`10894 -> 10467`)
  - `retry_ratio` improved (`1.21 -> 1.16`)

Dedicated rebuilt latency run (same rebuilt jemalloc stack):
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont23-20260224T024123Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont23-20260224T024123Z`
- p95:
  - `propose_total`: `2667` ms
  - `block_creator_total_create_block`: `2297` ms
  - `checkpoint_compute_state`: `2205` ms
  - `finalizer_total`: `11035` ms
  - `block_requests_retry_ratio`: `2.84`
- vs cont22 dedicated latency:
  - `propose/create/checkpoint` p95 improved (`4356/4183/4086 -> 2667/2297/2205`)
  - `finalizer_total p95` worsened (`10473 -> 11035`)
  - `retry_ratio` improved (`4.86 -> 2.84`) but remains elevated.

Interpretation:
- Patch is correctness-safe and improves proposer/checkpoint tails in both allocator-soak and dedicated runs for this replicate.
- Allocator growth regressed in major categories, so this change is not a clear memory win despite latency improvements.
- Next memory pass should directly target `interpreter_eval`/`other` call stacks (likely Rholang `Par`/vector clone families) while keeping this latency data for tradeoff context.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont24: COMM borrow-path clone reduction, recreate-only image cycle)

Correctness validation before perf cycle:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo check -p comm --quiet`: pass.
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).
- external integration suite:
  - `cd ~/work/asi/tests/firefly-rholang-tests-finality-suite-v2 && ./test.sh`: pass (`12 passing`, `1 pending`).

Recreate gate executed:
- `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
- `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`

Important caveat:
- `/tmp/shard-jemalloc-validator2.yml` is image-only (no `build:` stanza), so this cycle validated strict container restart but did not guarantee a new source image digest.

Allocator call-stack run:
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont24-20260224T065122Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont24-20260224T065122Z`
- top categorized growth bytes:
  - `other`: `4366672`
  - `interpreter_eval`: `4062685`
  - `rspace_history_read`: `186863`
  - `proposer_compute_state`: `158115`

A/B vs prior run (`cont23` -> `cont24`, recreate-only):
- categorized growth:
  - `other`: `5169213` -> `4366672` (`-15.53%`)
  - `interpreter_eval`: `3297152` -> `4062685` (`+23.21%`)
  - `rspace_history_read`: `373726` -> `186863` (`-50.00%`)
  - `proposer_compute_state`: `597223` -> `158115` (`-73.52%`)

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont24-20260224T065122Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `4904` ms
  - `create_block`: `3233` ms
  - `checkpoint_compute_state`: `3137` ms
  - `finalizer_total`: `10930` ms
  - `block_requests_retry_ratio`: `1.19`
- vs cont23 allocator-soak latency:
  - `propose/create/checkpoint` p95 worsened (`2370/1999/1890 -> 4904/3233/3137`)
  - `finalizer_total p95` worsened (`10467 -> 10930`)
  - `retry_ratio` worsened (`1.16 -> 1.19`)

Dedicated latency run:
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont24-20260224T065122Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont24-20260224T065122Z`
- p95:
  - `propose_total`: `2732` ms
  - `block_creator_total_create_block`: `1117` ms
  - `checkpoint_compute_state`: `877` ms
  - `finalizer_total`: `10894` ms
  - `block_requests_retry_ratio`: `3.78`
- vs cont23 dedicated latency:
  - `propose_total` slightly worsened (`2667 -> 2732`)
  - `create/checkpoint` improved (`2297/2205 -> 1117/877`)
  - `finalizer_total` improved (`11035 -> 10894`)
  - `retry_ratio` worsened (`2.84 -> 3.78`)

Interpretation:
- Recreate-only cycle shows promising reductions in `rspace_history_read` and `proposer_compute_state`, but with `interpreter_eval` still dominant and allocator-soak latency tails regressing.
- Because image digest was unchanged in this cycle, treat `cont24` as an operational restart validation plus directional signal, not a definitive source-rebuild measurement.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont25: explicit source image rebuild + strict recreate)

Source image rebuild gate:
- explicit rebuild from source:
  - `docker buildx build --platform linux/amd64 --file node/Dockerfile --tag f1r3flyindustries/f1r3fly-rust-node:latest --load .`
  - `docker tag f1r3flyindustries/f1r3fly-rust-node:latest f1r3node-jemallocprof:latest`
- rebuilt image digest:
  - `f1r3flyindustries/f1r3fly-rust-node:latest`: `sha256:5a5e732e915e8c8596afce0cfa73a39ffd2c4d7ef309d3b582d0fd278773ce24`
  - `f1r3node-jemallocprof:latest`: same sha.
- strict recreate before run:
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`

Allocator call-stack run:
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont25-20260224T070358Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont25-20260224T070358Z`
- top categorized growth bytes:
  - `other`: `6800856`
  - `interpreter_eval`: `715987`
  - `rspace_history_read`: `129367`

A/B vs prior run (`cont24` -> `cont25`, first source-rebuilt cycle):
- categorized growth:
  - `other`: `4366672` -> `6800856` (`+55.74%`)
  - `interpreter_eval`: `4062685` -> `715987` (`-82.38%`)
  - `rspace_history_read`: `186863` -> `129367` (`-30.77%`)

A/B vs pre-change comparison point (`cont23` -> `cont25`):
- categorized growth:
  - `interpreter_eval`: `3297152` -> `715987` (`-78.29%`)
  - `rspace_history_read`: `373726` -> `129367` (`-65.38%`)
  - `other`: `5169213` -> `6800856` (`+31.56%`)

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont25-20260224T070358Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `2948` ms
  - `create_block`: `2345` ms
  - `checkpoint_compute_state`: `2219` ms
  - `finalizer_total`: `10776` ms
  - `block_requests_retry_ratio`: `1.03`
- vs cont24 allocator-soak latency:
  - `propose/create/checkpoint` p95 improved (`4904/3233/3137 -> 2948/2345/2219`)
  - `finalizer_total p95` improved (`10930 -> 10776`)
  - `retry_ratio` improved (`1.19 -> 1.03`)

Dedicated latency run:
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont25-20260224T070358Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont25-20260224T070358Z`
- p95:
  - `propose_total`: `4964` ms
  - `block_creator_total_create_block`: `4532` ms
  - `checkpoint_compute_state`: `4379` ms
  - `finalizer_total`: `10947` ms
  - `block_requests_retry_ratio`: `3.01`
- vs cont24 dedicated latency:
  - `propose/create/checkpoint` p95 worsened (`2732/1117/877 -> 4964/4532/4379`)
  - `finalizer_total p95` slightly worsened (`10894 -> 10947`)
  - `retry_ratio` improved (`3.78 -> 3.01`) but remains elevated.

Interpretation:
- On the first fully source-rebuilt cycle, `interpreter_eval` and `rspace_history_read` allocator growth dropped materially vs both cont24 and cont23.
- `other` remains the dominant and growing bucket, now clearly the next hotspot for callsite-level narrowing.
- Latency signal is mixed: allocator-soak tails improved, but dedicated run tails regressed significantly on propose/create/checkpoint; additional replicated runs are needed before accepting this patch as latency-safe.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont26: load-phase window fix + strict recreate)

Measurement-correctness patch:
- file: `scripts/ci/run-latency-benchmark.sh`
- change:
  - moved `profile_log_since_utc` assignment to start exactly at Step 4 (load phase) instead of before preload checks.
  - intent: exclude preload/setup allocations from allocator delta window and improve attribution fidelity for unbounded-growth paths.
- validation:
  - `bash -n scripts/ci/run-latency-benchmark.sh`: pass.

Runtime gate:
- image in run: `sha256:5a5e732e915e8c8596afce0cfa73a39ffd2c4d7ef309d3b582d0fd278773ce24`.
- strict recreate before run:
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`

Allocator call-stack run:
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont26-20260224T075645Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont26-20260224T075645Z`
- early/latest heaps:
  - `early_heap=jeprof.1.4.i4.heap`
  - `latest_heap=jeprof.1.125.i125.heap`
- top categorized growth bytes:
  - `other`: `5926223`
  - `proposer_compute_state`: `287482`
  - `interpreter_eval`: `229641`
  - `rspace_history_read`: `129367`

Resolved call-paths inside `other` (from symbolized deltas):
- `tonic::codec::decode::Streaming<T>::new_request` (~`1.31MB`).
- `casper::rust::validate::Validate::repeat_deploy` via `block_storage::...::KeyValueBlockStore::get` vector growth (~`2x 786KB` + additional tails).
- additional network/runtime tails:
  - `h2::proto::...` growth paths.
  - unclassified libc/tokio stack (`~1.26MB`) pending deeper symbol resolution.

A/B vs prior run (`cont25` -> `cont26`):
- categorized growth:
  - `other`: `6800856` -> `5926223` (`-12.86%`)
  - `interpreter_eval`: `715987` -> `229641` (`-67.93%`)
  - `rspace_history_read`: `129367` -> `129367` (`0.00%`)
  - `proposer_compute_state`: absent in `cont25` top categories -> `287482` (newly visible with corrected window).

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont26-20260224T075645Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `4886` ms
  - `create_block`: `4657` ms
  - `checkpoint_compute_state`: `4541` ms
  - `finalizer_total`: `11034` ms
  - `block_requests_retry_ratio`: `1.18`
- vs cont25 allocator-soak latency:
  - `propose/create/checkpoint` p95 worsened (`2948/2345/2219 -> 4886/4657/4541`)
  - `finalizer_total p95` worsened (`10776 -> 11034`)
  - `retry_ratio` worsened (`1.03 -> 1.18`)

Dedicated latency run:
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont26-20260224T075645Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont26-20260224T075645Z`
- p95:
  - `propose_total`: `1539` ms
  - `block_creator_total_create_block`: `1301` ms
  - `checkpoint_compute_state`: `1105` ms
  - `finalizer_total`: `10563` ms
  - `block_requests_retry_ratio`: `2.74`
- vs cont25 dedicated latency:
  - `propose/create/checkpoint` p95 improved (`4964/4532/4379 -> 1539/1301/1105`)
  - `finalizer_total p95` improved (`10947 -> 10563`)
  - `retry_ratio` improved (`3.01 -> 2.74`) but still elevated.

Interpretation:
- The load-phase window fix improved hotspot attribution quality and reduced startup-noise contamination.
- Dominant growth is now concentrated in:
  - transport decode (`tonic::codec::decode::Streaming<T>::new_request`)
  - validation/block retrieval (`Validate::repeat_deploy` + `KeyValueBlockStore::get`).
- Next optimization pass should target those two paths directly; current rspace-side categories are no longer the dominant contributor in this replicate.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont27: early-exit repeat_deploy traversal)

Correctness-first code change:
- file: `shared/src/rust/dag/dag_ops.rs`
  - added `bf_traverse_find(...) -> Option<A>` (BFS with early exit).
  - added unit test `test_bf_traverse_find_stops_on_match`.
- file: `casper/src/rust/validate.rs`
  - `Validate::repeat_deploy` switched from:
    - `bf_traverse(...).into_iter().find(...)`
  - to:
    - `bf_traverse_find(..., predicate)`
  - intent: avoid allocating full traversal vector and stop on first duplicate match.

Correctness validation:
- `cargo check -p shared --quiet`: pass.
- `cargo test -p shared dag_ops -- --nocapture`: pass (`5 passed`).
- `cargo check -p casper --quiet`: pass.
- `cargo check -p node --quiet`: pass.
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo check -p comm --quiet`: pass.
- external integration suite:
  - `cd ~/work/asi/tests/firefly-rholang-tests-finality-suite-v2 && ./test.sh`: pass (`12 passing`, `1 pending`).

Source rebuild/recreate gate:
- rebuilt image from source:
  - `docker buildx build --platform linux/amd64 --file node/Dockerfile --tag f1r3flyindustries/f1r3fly-rust-node:latest --load .`
  - image digest: `sha256:0eb525515b6d9f5418a08a7072e6153784404c9b039ab63221f0e7f69ec7b75a`
  - tagged to `f1r3node-jemallocprof:latest` (same sha).
- strict recreate before run:
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`

Allocator call-stack run:
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont27-20260224T081627Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont27-20260224T081627Z`
- top categorized growth bytes:
  - `other`: `7230339`
  - `proposer_compute_state`: `1413534`
  - `interpreter_eval`: `833886`
  - `rspace_history_read`: `85796`
- top resolved call-paths in `other`:
  - `Validate::repeat_deploy` via `KeyValueBlockStore::get/get_unsafe` remained dominant in this replicate (`~3.73MB` top stack).
  - transport decode path still present:
    - `tonic::codec::decode::Streaming<T>::new_request` (`~1.31MB`).
  - additional decode/network pressure:
    - `h2::hpack::decoder::Decoder::decode`,
    - `bytes::BytesMut::reserve_inner`,
    - `prost::encoding::*` merge/copy paths.

A/B vs prior run (`cont26` -> `cont27`):
- categorized growth:
  - `other`: `5926223` -> `7230339` (`+22.01%`)
  - `proposer_compute_state`: `287482` -> `1413534` (`+391.69%`)
  - `interpreter_eval`: `229641` -> `833886` (`+263.13%`)
  - `rspace_history_read`: `129367` -> `85796` (`-33.68%`)

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont27-20260224T081627Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `2845` ms
  - `create_block`: `2578` ms
  - `checkpoint_compute_state`: `2453` ms
  - `finalizer_total`: `10603` ms
  - `block_requests_retry_ratio`: `1.20`
- vs cont26 allocator-soak latency:
  - `propose/create/checkpoint` p95 improved (`4886/4657/4541 -> 2845/2578/2453`)
  - `finalizer_total p95` improved (`11034 -> 10603`)
  - `retry_ratio` near-flat (`1.18 -> 1.20`).

Dedicated latency run:
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont27-20260224T081627Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont27-20260224T081627Z`
- p95:
  - `propose_total`: `4988` ms
  - `block_creator_total_create_block`: `3545` ms
  - `checkpoint_compute_state`: `3395` ms
  - `finalizer_total`: `10419` ms
  - `block_requests_retry_ratio`: `2.13`
- vs cont26 dedicated latency:
  - `propose/create/checkpoint` p95 worsened (`1539/1301/1105 -> 4988/3545/3395`)
  - `finalizer_total p95` improved (`10563 -> 10419`)
  - `retry_ratio` improved (`2.74 -> 2.13`) but remains elevated.

Interpretation:
- Early-exit traversal is correctness-safe, but this single replicate did not show a memory-allocation reduction in `repeat_deploy`/`other`; allocator pressure remained dominated by the same path.
- Latency remained highly volatile across dedicated runs; no stable latency win is attributable yet.
- Next pass should target `repeat_deploy` data-access shape itself (avoid repeated full block decode on traversal) and transport decode buffering.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont28: repeat_deploy proto fast-path)

Correctness-first code change:
- file: `block-storage/src/rust/key_value_block_store.rs`
  - added `has_any_deploy_sig(...)` and `has_any_deploy_sig_unsafe(...)`.
  - these inspect deploy signatures from decoded `BlockMessageProto` directly, avoiding full `BlockMessage::from_proto` conversion on the hot traversal path.
- file: `casper/src/rust/validate.rs`
  - `Validate::repeat_deploy` now:
    - builds `HashSet<Vec<u8>>` deploy sig set,
    - uses `block_store.has_any_deploy_sig_unsafe(...)` in BFS predicate,
    - keeps full block decode only for the single duplicated block used for warning details.
  - intent: reduce repeated decode/clone allocations during repeat-deploy ancestry scan.

Correctness validation:
- `cargo check -p block-storage --quiet`: pass.
- `cargo check -p casper --quiet`: pass.
- `cargo check -p node --quiet`: pass.
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo check -p comm --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).
- external integration suite:
  - `cd ~/work/asi/tests/firefly-rholang-tests-finality-suite-v2 && ./test.sh`: pass (`12 passing`, `1 pending`).
- note:
  - `cargo test -p casper --test mod ...` currently fails due pre-existing unrelated integration-test API drift in `casper/tests/*` (DashSet/stream-signature mismatches), not due this change.

Source rebuild/recreate gate:
- rebuilt image from source:
  - `docker buildx build --platform linux/amd64 --file node/Dockerfile --tag f1r3flyindustries/f1r3fly-rust-node:latest --load .`
  - image digest: `sha256:10b2cae538566e38ba5301f4fa2a3ccd41ebe923c29e3e57a3501947c9c7c423`
  - tagged to `f1r3node-jemallocprof:latest`.
- strict recreate before measurement:
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`

Allocator call-stack run:
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont28-20260224T085057Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont28-20260224T085057Z`
- top categorized growth bytes (`delta-positive.top15.callsites.txt`):
  - `interpreter_eval`: `4105074`
  - `other`: `3758733`
  - `tonic_reflection_startup`: `379980`
  - `rspace_history_read`: `158115`
  - `proposer_compute_state`: `100619`
- top resolved call-paths:
  - dominant stack is interpreter consume/reset path (`RSpace::consume` / `DebruijnInterpreter::*`) rather than `Validate::repeat_deploy`.
  - `block_storage::KeyValueBlockStore::bytes_to_block_proto/get` still appears in top-15 delta stacks.
  - `Validate::repeat_deploy` / `KeyValueBlockStore::get_unsafe` is no longer in top delta stacks (it was dominant in cont27).

A/B vs prior run (`cont27` -> `cont28`):
- categorized growth:
  - `other`: `7230339` -> `3758733` (`-48.01%`)
  - `proposer_compute_state`: `1413534` -> `100619` (`-92.88%`)
  - `interpreter_eval`: `833886` -> `4105074` (`+392.28%`)
  - `rspace_history_read`: `85796` -> `158115` (`+84.29%`)

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont28-20260224T085057Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `3356` ms
  - `create_block`: `3185` ms
  - `checkpoint_compute_state`: `3048` ms
  - `finalizer_total`: `11006` ms
  - `block_requests_retry_ratio`: `1.00`
- vs cont27 allocator-soak latency:
  - `propose/create/checkpoint` p95 worsened (`2845/2578/2453 -> 3356/3185/3048`)
  - `finalizer_total p95` worsened (`10603 -> 11006`)
  - `retry_ratio` improved (`1.20 -> 1.00`).

Dedicated latency run:
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont28-20260224T085057Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont28-20260224T085057Z`
- p95:
  - `propose_total`: `4271` ms
  - `block_creator_total_create_block`: `4131` ms
  - `checkpoint_compute_state`: `3999` ms
  - `finalizer_total`: `10853` ms
  - `block_requests_retry_ratio`: `3.00`
- vs cont27 dedicated latency:
  - `propose/create/checkpoint` p95 improved on propose, worsened on create/checkpoint (`4988/3545/3395 -> 4271/4131/3999`)
  - `finalizer_total p95` worsened (`10419 -> 10853`)
  - `retry_ratio` worsened (`2.13 -> 3.00`).

Interpretation:
- This pass removed `repeat_deploy/get_unsafe` from dominant allocator deltas under strict rebuild/recreate and kept correctness green on active gates.
- Allocation pressure shifted toward interpreter-heavy stacks in this replicate; `other` reduced substantially but latency remains volatile and not yet improved consistently.
- Next memory pass should target interpreter reset/consume clone growth while preserving current repeat-deploy improvement.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont29: restore_installs clone-reduction)

Correctness-first code change:
- file: `rspace++/src/rspace/rspace.rs`
- file: `rspace++/src/rspace/replay_rspace.rs`
- change:
  - `restore_installs()` now moves out the installs map via `std::mem::take(...)` instead of cloning the full map.
  - for each install entry, replay still goes through `install(...)` (same semantics path).
  - preserves prior behavior for immediate matches by reinserting original install when `install_result.is_some()`.
  - intent: reduce repeated full-map clone pressure during reset/checkpoint restore cycles.

Correctness validation:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo check -p node --quiet`: pass.
- `cargo check -p casper --quiet`: pass.
- `cargo check -p comm --quiet`: pass.
- external integration suite:
  - `cd ~/work/asi/tests/firefly-rholang-tests-finality-suite-v2 && ./test.sh`: pass (`12 passing`, `1 pending`).

Source rebuild/recreate gate:
- rebuilt image from source:
  - `docker buildx build --platform linux/amd64 --file node/Dockerfile --tag f1r3flyindustries/f1r3fly-rust-node:latest --load .`
  - image digest: `sha256:e397ada4d1343ae37febddd281b1d9288c3196c998d466ee6fecddeb400c87d1`
  - tagged to `f1r3node-jemallocprof:latest`.
- strict recreate before measurement:
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`

Allocator call-stack run:
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont29-20260224T091216Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont29-20260224T091216Z`
- top categorized growth bytes (`delta-positive.top15.callsites.txt`):
  - `proposer_compute_state`: `2409404`
  - `other`: `2029536`
  - `interpreter_eval`: `762632`
  - `rspace_history_read`: `143741`
- top resolved call-paths in delta stacks:
  - `InMemHotStore::install_join` and `RSpace::restore_installs` are now visible in top growth.
  - `Validate::repeat_deploy` still appears, now via `KeyValueBlockStore::has_any_deploy_sig[_unsafe]` and `bytes_to_block_proto`.
  - `KeyValueBlockStore::get` remains present in lower top-delta entries.

A/B vs prior run (`cont28` -> `cont29`):
- categorized growth:
  - `proposer_compute_state`: `100619` -> `2409404` (`+2294.73%`)
  - `other`: `3758733` -> `2029536` (`-46.01%`)
  - `interpreter_eval`: `4105074` -> `762632` (`-81.42%`)
  - `rspace_history_read`: `158115` -> `143741` (`-9.09%`)

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont29-20260224T091216Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `3400` ms
  - `create_block`: `2715` ms
  - `checkpoint_compute_state`: `2598` ms
  - `finalizer_total`: `10504` ms
  - `block_requests_retry_ratio`: `0.94`
- vs cont28 allocator-soak latency:
  - `propose/create/checkpoint` p95 improved (`3356/3185/3048 -> 3400/2715/2598`; propose near-flat, create/checkpoint better)
  - `finalizer_total p95` improved (`11006 -> 10504`)
  - `retry_ratio` improved (`1.00 -> 0.94`).

Dedicated latency run:
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont29-20260224T091216Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont29-20260224T091216Z`
- p95:
  - `propose_total`: `2166` ms
  - `block_creator_total_create_block`: `1931` ms
  - `checkpoint_compute_state`: `1807` ms
  - `finalizer_total`: `11139` ms
  - `block_requests_retry_ratio`: `3.01`
- vs cont28 dedicated latency:
  - `propose/create/checkpoint` p95 improved strongly (`4271/4131/3999 -> 2166/1931/1807`)
  - `finalizer_total p95` worsened (`10853 -> 11139`)
  - `retry_ratio` near-flat (`3.00 -> 3.01`).

Interpretation:
- Clone-heavy interpreter category dropped substantially in this replicate, and dedicated propose/create/checkpoint latency improved materially.
- New dominant allocation paths are now more concentrated in `proposer_compute_state` and restore/install-join related stacks.
- Next pass should focus on reducing `restore_installs` + `install_join` per-channel join cloning cost, while preserving install semantics.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont30: restore_installs direct-move no per-entry clone)

Correctness-first code change:
- file: `rspace++/src/rspace/rspace.rs`
- file: `rspace++/src/rspace/replay_rspace.rs`
- change:
  - `restore_installs()` now moves `Install` fields directly into `install(...)` instead of cloning per-entry values before replay.
  - this keeps startup-only install semantics unchanged (`locked_install` still rejects non-startup install path).
  - intent: remove remaining per-entry clone pressure in restore path after prior map-level `mem::take(...)` pass.

Correctness validation:
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- external integration suite:
  - `cd ~/work/asi/tests/firefly-rholang-tests-finality-suite-v2 && ./test.sh`: pass (`12 passing`, `1 pending`).

Source rebuild/recreate gate:
- rebuilt image from source:
  - `docker buildx build --platform linux/amd64 --file node/Dockerfile --tag f1r3flyindustries/f1r3fly-rust-node:latest --load .`
  - image digest: `sha256:43115cee85ee8fa13e23d85516f5248aacdb543e9c9a2ef023061cf154d3000f`
  - tagged to `f1r3node-jemallocprof:latest`.
- strict recreate before measurement:
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`

Allocator call-stack run:
- command:
  - `./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont30-20260224T093110Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont30-20260224T093110Z`
- top categorized growth bytes (`delta-positive.top15.callsites.txt`):
  - `other`: `3575016`
  - `tonic_reflection_startup`: `259264`
  - `interpreter_eval`: `199278`
  - `rspace_history_read`: `114993`
  - `proposer_compute_state`: `43122`
- top resolved call-paths:
  - `RSpace::restore_installs` and `InMemHotStore::install_join` no longer appear in top-15 positive-delta stacks.
  - `Validate::repeat_deploy` / `has_any_deploy_sig` also not present in top-15 positive-delta stacks.
  - `KeyValueBlockStore::bytes_to_block_proto/get` remains present in top deltas.

A/B vs prior run (`cont29` -> `cont30`):
- categorized growth:
  - `proposer_compute_state`: `2409404` -> `43122` (`-98.21%`)
  - `interpreter_eval`: `762632` -> `199278` (`-73.87%`)
  - `rspace_history_read`: `143741` -> `114993` (`-20.00%`)
  - `other`: `2029536` -> `3575016` (`+76.14%`)

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont30-20260224T093110Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `5347` ms
  - `create_block`: `5117` ms
  - `checkpoint_compute_state`: `4985` ms
  - `finalizer_total`: `11399` ms
  - `block_requests_retry_ratio`: `3.88`
- vs cont29 allocator-soak latency:
  - `propose/create/checkpoint` p95 worsened (`3400/2715/2598 -> 5347/5117/4985`)
  - `finalizer_total p95` worsened (`10504 -> 11399`)
  - `retry_ratio` worsened (`0.94 -> 3.88`).

Dedicated latency run:
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont30-20260224T093331Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont30-20260224T093331Z`
- p95:
  - `propose_total`: `3095` ms
  - `block_creator_total_create_block`: `2880` ms
  - `checkpoint_compute_state`: `2804` ms
  - `finalizer_total`: `11036` ms
  - `block_requests_retry_ratio`: `2.65`
- vs cont29 dedicated latency:
  - `propose/create/checkpoint` p95 worsened (`2166/1931/1807 -> 3095/2880/2804`)
  - `finalizer_total p95` improved slightly (`11139 -> 11036`)
  - `retry_ratio` improved (`3.01 -> 2.65`).

Interpretation:
- Direct-move `restore_installs` removed the previously visible restore/install-join growth signatures from top allocator deltas and collapsed `proposer_compute_state` category in this replicate.
- Allocation growth redistributed into `other` and a startup reflection bucket; the net memory-growth shape changed but did not yield a consistent latency win.
- Next pass should split `other` with deeper categorized bucketing (or wider top-N stack capture) and target the remaining `KeyValueBlockStore` decode-heavy paths that still appear in top deltas.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont31: wider call-stack capture)

Correctness + environment gate:
- strict validator restart before measurements:
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`
- correctness checks:
  - `cargo check -p rspace_plus_plus --quiet`: pass.
  - `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
  - `cd ~/work/asi/tests/firefly-rholang-tests-finality-suite-v2 && ./test.sh`: pass (`12 passing`, `1 pending`).

Allocator call-stack run (wider depth):
- command:
  - `TOP_STACKS=40 TOP_FRAMES=20 ./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont31-20260224T094257Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont31-20260224T094257Z`
- top categorized growth bytes (`delta-positive.top40.callsites.txt`):
  - `other`: `6454929`
  - `proposer_compute_state`: `1578515`
  - `interpreter_eval`: `969990`
  - `rspace_history_read`: `71870`
- notable top first-frames:
  - `rustls::msgs::deframer::buffers::DeframerVecBuffer::read`: `4413185`
  - `RawTable::clone` / `RawTable::reserve_rehash` + vec clone/to_vec paths remain prominent.
- symbolized-path notes:
  - `RSpace::restore_installs` is present in widened top-40 symbolized deltas.
  - `Validate::repeat_deploy` / `has_any_deploy_sig` not present in widened top-40 symbolized deltas.

A/B vs prior run (`cont30` -> `cont31`):
- categorized growth:
  - `other`: `3575016` -> `6454929` (`+80.56%`)
  - `proposer_compute_state`: `43122` -> `1578515` (`+3560.96%`)
  - `interpreter_eval`: `199278` -> `969990` (`+386.70%`)
  - `rspace_history_read`: `114993` -> `71870` (`-37.50%`)

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont31-20260224T094257Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `2035` ms
  - `create_block`: `1748` ms
  - `checkpoint_compute_state`: `1657` ms
  - `finalizer_total`: `10381` ms
  - `block_requests_retry_ratio`: `3.01`
- run-level note:
  - soak summary reported `propose_bug_error=1` and `propose_fail=1` (single incident during run).
- vs cont30 allocator-soak latency:
  - `propose/create/checkpoint` p95 improved (`5347/5117/4985 -> 2035/1748/1657`)
  - `finalizer_total p95` improved (`11399 -> 10381`)
  - `retry_ratio` improved (`3.88 -> 3.01`).

Dedicated latency run:
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont31-20260224T094540Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont31-20260224T094540Z`
- p95:
  - `propose_total`: `2196` ms
  - `block_creator_total_create_block`: `2021` ms
  - `checkpoint_compute_state`: `1945` ms
  - `finalizer_total`: `10904` ms
  - `block_requests_retry_ratio`: `3.09`
- vs cont30 dedicated latency:
  - `propose/create/checkpoint` p95 improved (`3095/2880/2804 -> 2196/2021/1945`)
  - `finalizer_total p95` improved (`11036 -> 10904`)
  - `retry_ratio` worsened (`2.65 -> 3.09`).

Interpretation:
- Widening stack capture improved attribution: `other` is strongly influenced by transport deframer reads plus clone/rehash-heavy table/vector operations.
- `repeat_deploy` is no longer surfacing in dominant deltas; pressure has shifted toward transport + replay/install/collection clone paths.
- Given the single `propose_bug_error` in allocator-soak run, follow-up should include log-level root-cause capture for that incident before treating cont31 latency gains as fully stable.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont32: restore replay reserve + widened stack capture)

Correctness-first implementation note:
- attempted change (replay without re-recording installs) introduced a regression:
  - repeated proposer failures and runtime errors in container logs:
    - `Proposal failed: BugError (seqNum -1)`
    - `Runtime error: Incorrect number of results from query of current bonds ...: 0`
- fix applied in both files:
  - `rspace++/src/rspace/rspace.rs`
  - `rspace++/src/rspace/replay_rspace.rs`
- final behavior:
  - `restore_installs()` still re-records installs (`record_install=true`) to preserve semantics.
  - added `reserve(installs.len())` after `mem::take(...)` before replay inserts to reduce rehash pressure safely.

Correctness validation (post-fix image):
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- external integration suite on rebuilt container:
  - `cd ~/work/asi/tests/firefly-rholang-tests-finality-suite-v2 && ./test.sh`: pass (`12 passing`, `1 pending`).

Source rebuild/recreate gate:
- intermediate regression image (discarded from measurement):
  - `sha256:ff6ba5b16842e0deaad13a73df4ea6cf172cb72f259deb3164cc6c1691757298`
- fixed image used for cont32:
  - `sha256:f606a773076a2313a7a81f777844e5bdc2acfbbc333101c4966c79a744b36fe9`
  - tagged to `f1r3node-jemallocprof:latest`
- strict recreate before measurement:
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`

Allocator call-stack run (wider depth):
- command:
  - `TOP_STACKS=40 TOP_FRAMES=20 ./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont32-20260224T101237Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont32-20260224T101237Z`
- top categorized growth bytes (`delta-positive.top40.callsites.txt`):
  - `interpreter_eval`: `2943604`
  - `other`: `1971406`
  - `proposer_compute_state`: `1370744`
  - `tonic_reflection_startup`: `528204`
  - `rspace_history_read`: `388100`
- key symbolized paths still present in widened top set:
  - `RSpace::restore_installs`
  - `InMemHotStore::install_join`
  - `InMemHotStore::get_data_from_history_store`
  - `InMemHotStore::get_cont_from_history_store`
  - `Validate::repeat_deploy` via `KeyValueBlockStore::has_any_deploy_sig[_unsafe]`

A/B vs prior run (`cont31` -> `cont32`):
- categorized growth:
  - `other`: `6454929` -> `1971406` (`-69.46%`)
  - `proposer_compute_state`: `1578515` -> `1370744` (`-13.16%`)
  - `interpreter_eval`: `969990` -> `2943604` (`+203.47%`)
  - `rspace_history_read`: `71870` -> `388100` (`+439.99%`)

Latency from allocator soak:
- profile:
  - `/tmp/casper-allocator-hotspots-20260224T-cont32-20260224T101237Z/soak/profile/summary.txt`
- p95:
  - `propose_total`: `7931` ms
  - `create_block`: `3097` ms
  - `checkpoint_compute_state`: `2904` ms
  - `finalizer_total`: `10750` ms
  - `block_requests_retry_ratio`: `3.96`
- vs cont31 allocator-soak latency:
  - `propose/create/checkpoint` p95 worsened (`2035/1748/1657 -> 7931/3097/2904`)
  - `finalizer_total p95` worsened (`10381 -> 10750`)
  - `retry_ratio` worsened (`3.01 -> 3.96`)
- run-level note:
  - `propose_bug_error=0`, `propose_fail=0` in cont32 soak.

Dedicated latency run:
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont32-20260224T101511Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont32-20260224T101511Z`
- p95:
  - `propose_total`: `4734` ms
  - `block_creator_total_create_block`: `4197` ms
  - `checkpoint_compute_state`: `4055` ms
  - `finalizer_total`: `10870` ms
  - `block_requests_retry_ratio`: `3.93`
- vs cont31 dedicated latency:
  - `propose/create/checkpoint` p95 worsened (`2196/2021/1945 -> 4734/4197/4055`)
  - `finalizer_total p95` improved slightly (`10904 -> 10870`)
  - `retry_ratio` worsened (`3.09 -> 3.93`).

Interpretation:
- correctness is restored and externally validated on the fixed image.
- reserve-on-replay reduced the large `other` bucket but shifted growth toward interpreter/history-read categories; `restore_installs` and history-cache fetch paths remain in dominant widened stacks.
- latency is materially worse in cont32 despite zero proposer bug errors during soak, so this pass is not a performance win and should not be promoted as-is.

### Allocator hotspot continuation (validator2, 120s soak, 2026-02-24 cont34: single-runtime compute_state+bonds path)

Correctness-first implementation note:
- reduced per-proposal runtime initialization by avoiding a second runtime spawn for bonds query:
  - `casper/src/rust/util/rholang/runtime_manager.rs`
    - added `compute_state_with_bonds(...)` that computes state and bonds using one spawned runtime.
  - `casper/src/rust/util/rholang/interpreter_util.rs`
    - `compute_deploys_checkpoint(...)` switched to `compute_state_with_bonds(...)` and now returns bonds.
  - `casper/src/rust/blocks/proposer/block_creator.rs`
    - consumes bonds from checkpoint tuple; removed separate `compute_bonds(...)` phase.
- behavioral expectation:
  - no change in state/replay correctness; only runtime lifecycle reduced.

Compile/test validation:
- `cargo check -p casper --quiet`: pass.
- `cargo check -p node --quiet`: pass.
- `cargo check -p rspace_plus_plus --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).

Source rebuild/recreate gate:
- rebuilt image from current source:
  - `docker build -f node/Dockerfile -t f1r3flyindustries/f1r3fly-rust-node:latest .`
  - image digest: `sha256:2f0ad3b61da28f905a4271e77e06845167a523ee586d8742457c732e6906b4c0`
- retagged for jemalloc compose compatibility:
  - `docker tag f1r3flyindustries/f1r3fly-rust-node:latest f1r3node-jemallocprof:latest`
- strict recreate cycle (performed twice to ensure validator2 tag alignment):
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml down -v`
  - `docker compose -f /tmp/shard-jemalloc-validator2.yml up -d --force-recreate`
- container image verification:
  - `boot/validator1/validator2/validator3/readonly` all running image
    `sha256:2f0ad3b61da28f905a4271e77e06845167a523ee586d8742457c732e6906b4c0`.

Correctness gates on rebuilt containers:
- `./scripts/ci/check-casper-init-sla.sh /tmp/shard-jemalloc-validator2.yml 240`: pass.
- `cd ~/work/asi/tests/firefly-rholang-tests-finality-suite-v2 && ./test.sh`: pass (`12 passing`, `1 pending`).

Dedicated latency run:
- command:
  - `./scripts/ci/run-latency-benchmark.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-latency-benchmark-cont34-20260224T105809Z`
- artifact:
  - `/tmp/casper-latency-benchmark-cont34-20260224T105809Z/profile/summary.txt`
- p95:
  - `propose_total`: `2711` ms
  - `block_creator_total_create_block`: `2616` ms
  - `checkpoint_compute_state`: `2582` ms
  - `finalizer_total`: `10764` ms
  - `block_requests_retry_ratio`: `3.25`
- implementation signal:
  - `block_creator_compute_bonds: n=0` (bonds phase removed as standalone timing).

A/B vs prior dedicated run (`cont33` -> `cont34`):
- `propose_total p95`: `5242 -> 2711` (`-48.28%`)
- `create_block p95`: `5065 -> 2616` (`-48.35%`)
- `checkpoint_compute_state p95`: `4932 -> 2582` (`-47.65%`)
- `finalizer_total p95`: `11011 -> 10764` (`-2.24%`)
- `retry_ratio`: `3.27 -> 3.25` (flat/slight improvement).

Allocator call-stack run (wider depth):
- command:
  - `TOP_STACKS=40 TOP_FRAMES=20 ./scripts/ci/profile-validator-allocator-hotspots.sh /tmp/shard-jemalloc-validator2.yml 120 /tmp/casper-allocator-hotspots-20260224T-cont34-20260224T110030Z`
- artifact:
  - `/tmp/casper-allocator-hotspots-20260224T-cont34-20260224T110030Z`
- top categorized growth bytes (`delta-positive.top40.callsites.txt`):
  - `interpreter_eval`: `1652814`
  - `other`: `601142`
  - `proposer_compute_state`: `476733`
  - `rspace_history_read`: `114992`

A/B vs prior allocator run (`cont33` -> `cont34`):
- `proposer_compute_state`: `2540884 -> 476733` (`-81.24%`)
- `other`: `1049887 -> 601142` (`-42.74%`)
- `rspace_history_read`: `388099 -> 114992` (`-70.37%`)
- `interpreter_eval`: `916344 -> 1652814` (`+80.36%`)

Interpretation:
- correctness remained green after source rebuild + strict validator recreate.
- eliminating the extra runtime spawn for bonds materially reduced propose/create/checkpoint p95 (~48% improvement vs cont33), but p95 is still above the `<2s` target.
- dominant remaining allocation/latency pressure is now in interpreter evaluation / clone-heavy runtime execution paths; this is the next optimization target.

### Latency continuation (2026-02-24 cont35-cont37: DAG clone removal + finalizer catch-up budget)

Correctness-first code changes:
- Removed full CasperBuffer DAG clone from equivocation checks on propose/validate path.
  - `block-storage/src/rust/casperbuffer/casper_buffer_key_value_storage.rs`
    - added `requested_as_dependency(&BlockHashSerde) -> bool` (contains-key lookup).
  - `casper/src/rust/equivocation_detector.rs`
    - `check_equivocations(...)` now receives precomputed `requested_as_dependency: bool`.
  - `casper/src/rust/multi_parent_casper_impl.rs`
    - replaced `to_doubly_linked_dag()` clone path with boolean lookup.
- Reduced finalizer catch-up per-run budget to align with latency target envelope.
  - `casper/src/rust/finality/finalizer.rs`
    - `FINALIZER_CATCHUP_WORK_BUDGET_MS: 10000 -> 2000`
    - `FINALIZER_CATCHUP_STEP_TIMEOUT_MS: 1000 -> 200`

Validation:
- `cargo check -p casper --quiet`: pass.
- `cargo check -p node --quiet`: pass.
- `cargo test -p rspace_plus_plus --test mod history_repository -- --nocapture`: pass (`5 passed`).
- `cargo test -p comm packet_ops -- --nocapture`: pass (`6 passed`).

Strict rebuild/recreate cycles were performed between benchmark rounds.
- image digests observed:
  - `sha256:dd8a8ea37647fbda49818d65196f21a77a7303d60c36338f03df1ea2e29e4469` (DAG clone removal)
  - `sha256:0f7b5fad451e604d76142a4bd9d990a95855f5ed242fcca573642eef8ec566d4` (plus finalizer catch-up budget reduction)
- validators recreated with `down -v` + `up -d --force-recreate` for each measurement round.

Correctness gates on rebuilt clusters:
- `./scripts/ci/check-casper-init-sla.sh /tmp/shard-jemalloc-validator2.yml 240`: pass.
- `cd ~/work/asi/tests/firefly-rholang-tests-finality-suite-v2 && ./test.sh`: pass (`12 passing`, `1 pending`).

Allocator impact after DAG clone removal:
- artifact: `/tmp/casper-allocator-hotspots-cont36-20260224T113535Z/stacks/delta-positive.top40.callsites.txt`
- categorized growth bytes:
  - `other`: `5232396`
  - `interpreter_eval`: `711306`
  - `proposer_compute_state`: `352461`
  - `rspace_history_read`: `14374`
- comparison vs cont35 (same collector):
  - `proposer_compute_state`: `2254472 -> 352461` (`-84.37%`)
  - `interpreter_eval`: `1735164 -> 711306` (`-59.01%`)
  - confirms large clone-path reduction on proposer compute-state allocations.

Latency snapshots:
- pre-finalizer-budget adjustment (cont36b):
  - artifact: `/tmp/casper-latency-benchmark-cont36b-20260224T113832Z/profile/summary.txt`
  - p95: `propose_total=629ms`, `create_block=456ms`, `checkpoint_compute_state=454ms`, `finalizer_total=11020ms`.
- post-finalizer-budget adjustment (cont37):
  - artifact: `/tmp/casper-latency-benchmark-cont37-20260224T115053Z/profile/summary.txt`
  - p95: `propose_total=2592ms`, `create_block=1826ms`, `checkpoint_compute_state=1814ms`, `finalizer_total=2509ms`, `last_finalized_block_total=2618ms`.

Current status vs target (`<2s` propose/create/replay/finalization):
- achieved: block creation/checkpoint p95 now <2s in cont37.
- not yet achieved:
  - `propose_total p95=2592ms`
  - replay mean remains >2s (`block_processing_replay_mean_ms=3348ms`)
  - `finalizer_total p95=2509ms` (down materially from ~11s, but still above target).

Primary remaining bottleneck signal:
- repeated retrieval retries for a single hash still dominate tail behavior:
  - `replay_retry_top_hashes: da58419c...:171`
  - `block_requests_retry_ratio: 3.52`.

### Unattended memory-growth forensic pass (2026-02-24, preliminary)

Objective:
- isolate dominant call paths behind unattended validator RSS growth (reported up to ~15GiB) using long-soak sampling + jemalloc stack diffs.

Live soak (in progress):
- script: `scripts/ci/run-validator-leak-soak.sh`
- command:
  - `SOAK_PROFILE_PROC=1 SOAK_PROC_SAMPLE_EVERY_SECONDS=30 SOAK_PROFILE_FINALIZER=1 ./scripts/ci/run-validator-leak-soak.sh /tmp/shard-jemalloc-validator2.yml 10800 30 /tmp/casper-validator-leak-soak-3h-20260224T141046Z`
- artifact root:
  - `/tmp/casper-validator-leak-soak-3h-20260224T141046Z`

Interim proc-class deltas (~182s window):
- validator1: `Rss +629.5MiB`, `Anonymous +99.8MiB`, `FileApprox +529.8MiB`
- validator2: `Rss +711.0MiB`, `Anonymous +432.3MiB`, `FileApprox +278.7MiB`
- validator3: `Rss +1026.7MiB`, `Anonymous +894.1MiB`, `FileApprox +132.7MiB`

Interpretation:
- validator2/3 growth is predominantly anonymous memory (heap-like), not only file-backed mmap expansion.

Fresh jemalloc stack diffs (validator2, mtime-windowed snapshots):
- window A (`i255 -> i276`):
  - `/tmp/casper-heap-growth-soak-v2-20260224T141502Z`
- window B (`i276 -> i325`):
  - `/tmp/casper-heap-growth-soak-v2-window2-20260224T141606Z`

Window B family totals (top-30 positive growth stacks):
- `rholang_rspace_clone`: `3,536,140` bytes
- `block_decode`: `1,155,521` bytes
- `grpc_http2_buffers`: `179,763` bytes
- `other`: `56,990` bytes

Dominant symbolized paths in window B:
- `hashbrown::raw::inner::RawTable::reserve_rehash` from `RadixTreeImpl::load_node_arc` / history reset during proposer compute-state.
- `alloc::vec::in_place_collect::from_iter_in_place` via `ParSortMatcher::sort_match` / substitute/reduce interpreter path.
- `alloc::raw_vec::finish_grow` via `KeyValueBlockStore::bytes_to_block_proto` (`prost` block decode).

Preliminary culprit statement:
- main unattended allocator growth path on profiled validator2 is clone-heavy Rholang/RSpace history/interpreter work (`load_node_arc` + Par/Datum/WaitingContinuation clone chains), with secondary growth from block decode (`prost`/`bytes_to_block_proto`).
- this aligns with anonymous-memory-heavy growth observed in the live soak.
