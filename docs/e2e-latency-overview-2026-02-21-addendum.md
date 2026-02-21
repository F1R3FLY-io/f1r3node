# E2E Latency + Memory Soak Addendum (2026-02-21)

## Scope
This addendum captures validator memory-growth and finalization-loop behavior after iterative Rust-side queue/cache fixes, with repeated clean-container soak tests and finality-suite correctness checks.

## Code-Level Changes In Scope

1. `casper/src/rust/engine/running.rs`
- Added ingress dedup for in-flight blocks:
  - Skip enqueue if block hash already exists in `blocks_in_processing`.
  - Enforce in-flight cap before enqueue.
  - Roll back pre-enqueue mark if channel send fails.

2. `casper/src/rust/engine/block_retriever.rs`
- Added runtime-toggleable dedup for already-queried peers in `admit_hash`:
  - Env: `F1R3_BLOCK_RETRIEVER_DEDUP_QUERIED_PEERS`
  - `1`: avoid re-adding peers already queried for same hash.
  - `0`: preserve prior behavior.
- Default set to `0` (disabled) pending longer-run validation.

3. `docker/shard-with-autopropose.yml`
- Added env:
  - `F1R3_BLOCK_RETRIEVER_DEDUP_QUERIED_PEERS=${F1R3_BLOCK_RETRIEVER_DEDUP_QUERIED_PEERS:-0}`
- Existing trim env retained:
  - `F1R3_MALLOC_TRIM_EVERY_BLOCKS=${F1R3_MALLOC_TRIM_EVERY_BLOCKS:-16}`

4. `scripts/ci/run-validator-leak-soak.sh`
- Added clean restart + data wipe + readiness gate + proc sampler mode.
- Extended sampled metrics to include DAG cardinalities and cache/hot-store gauges.
- Added finalizer/log health summary output (`finalizer-summary.txt`) with per-validator counts:
  - `finalizer-run-started`, `finalizer-run-finished`
  - `new_lfb_found=true/false`
  - `filtered_agreements=0`
  - finalizer skip/timeout indicators

## Repro Command Template

```bash
SOAK_RESTART_CLEAN=1 \
SOAK_CLEAN_DATA_DIR=/home/purplezky/work/asi/f1r3node/docker/data \
SOAK_WAIT_FOR_READY=1 \
SOAK_PROFILE_PROC=1 \
SOAK_PROC_SAMPLE_EVERY_SECONDS=10 \
F1R3_MALLOC_TRIM_EVERY_BLOCKS=16 \
F1R3_BLOCK_RETRIEVER_DEDUP_QUERIED_PEERS=0 \
scripts/ci/run-validator-leak-soak.sh docker/shard-with-autopropose.yml 120 10 /tmp/<out>
```

## Key 120s Runs (Mean RSS Slope, MiB/s)

- Baseline (no newer queue fixes): `6.505706`
  - `/tmp/casper-validator-leak-soak-structure-20260221T202940Z/summary.txt`
- After ingress duplicate enqueue fix (`running.rs`): `6.232432`
  - `/tmp/casper-validator-leak-soak-structure-afterdupfix-20260221T204031Z/summary.txt`
- With retriever dedup enabled (full run #1): `7.535135` (regression/outlier)
  - `/tmp/casper-validator-leak-soak-after2fixes-full120-20260221T211509Z/summary.txt`
- With retriever dedup enabled (full run #2): `6.488589`
  - `/tmp/casper-validator-leak-soak-after2fixes-full120-rep2-20260221T212001Z/summary.txt`
- Final current default (`dedup=0`): `6.456156`
  - `/tmp/casper-validator-leak-soak-final-defaultdedup0-full120-20260221T213950Z/summary.txt`

## Controlled A/B (90s) for Retriever Dedup Flag

- `dedup=1`: `7.306329`
  - `/tmp/casper-validator-leak-soak-ab-dedup1-20260221T213059Z/summary.txt`
- `dedup=0`: `7.051250` (better by ~3.49%)
  - `/tmp/casper-validator-leak-soak-ab-dedup0-20260221T213449Z/summary.txt`

## Observations

1. Anonymous/private-dirty dominates RSS growth across runs (proc sampler confirms file-backed growth is small).
2. `running.rs` ingress dedup is consistently useful and should stay enabled.
3. Retriever dedup of already-queried peers is not yet robustly positive under full-load variance; keep runtime-toggleable and default-disabled.
4. Run-to-run variance strongly tracks finalization health (`new_lfb_found` cadence / `filtered_agreements=0` bursts), so memory slope must be interpreted together with finalization progress.

## Correctness Status

Finality suite re-run after latest changes:
- Command: `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
- Result: `12 passing`, `1 pending`

## Latest Commit Re-Validation (2026-02-21, rebuilt image)

Environment:
- Image rebuilt from current HEAD (`362d37fd`) and retagged: `f1r3flyindustries/f1r3fly-rust-node:latest`
- Image digest: `sha256:09f45ba197f3a5a9c21f0b260ca059e292b46a9177ed4952f18d6e42d0cf7aac`

Fresh clean 120s soaks (`SOAK_RESTART_CLEAN=1`, data wipe, proc+finalizer profiling):
- Run A mean slope: `6.824024`
  - `/tmp/casper-validator-leak-soak-latest-20260221T220717Z/summary.txt`
  - `/tmp/casper-validator-leak-soak-latest-20260221T220717Z/proc-summary.txt`
  - `/tmp/casper-validator-leak-soak-latest-20260221T220717Z/finalizer-summary.txt`
- Run B mean slope: `6.184985`
  - `/tmp/casper-validator-leak-soak-latest-rep2-20260221T221249Z/summary.txt`
  - `/tmp/casper-validator-leak-soak-latest-rep2-20260221T221249Z/proc-summary.txt`
  - `/tmp/casper-validator-leak-soak-latest-rep2-20260221T221249Z/finalizer-summary.txt`

Comparison vs prior reference (`dedup=0` full 120s: `6.456156`):
- Run A: `+5.70%`
- Run B: `-4.20%`
- Two-run mean: `6.504505` (`+0.75%`)

Interpretation:
- Latest commit remains correct (`12 passing`, `1 pending`) and memory growth remains highly variant run-to-run.
- Aggregate behavior is effectively flat versus prior reference; no robust leak elimination yet.
- Proc summaries again show growth mostly in anonymous/private-dirty pages.

## Phase Breakdown Profile (2026-02-21)

Profiler run:
- Command: `scripts/ci/profile-casper-latency.sh docker/shard-with-autopropose.yml /tmp/casper-latency-profile-e2e-verify2-20260221T222431Z "<since_utc>"`
- Summary: `/tmp/casper-latency-profile-e2e-verify2-20260221T222431Z/summary.txt`

Observed hotspots:
- Propose loop:
  - `propose_total avg ~977 ms` (p95 ~4489 ms)
  - `propose_core avg ~976 ms` (snapshot is negligible)
- Block creation:
  - `total_create_block avg ~600 ms`
  - `compute_deploys_checkpoint avg ~542 ms`
  - `checkpoint_parents_post_state avg ~401 ms`
  - path mix shows `merged` path is dominant expensive path:
    - `compute_parents_post_state path=merged avg ~1310 ms`
    - `path=single_parent` / `descendant_fast_path` / `cache_hit` near `0 ms`
- Replay/validation (histograms):
  - `block_processing_replay_mean ~183 ms`
  - replay phase means:
    - `user_deploys ~397 ms`
    - `system_deploys ~112 ms`
    - `create_checkpoint ~38 ms`
- Finalization loop:
  - `finalizer_total avg ~12.5 ms` (p95 ~53 ms), `budget_exhausted=true` not observed
  - `last_finalized_block_total avg ~14.8 ms`
- Retrieval pressure remains high:
  - `block_requests_retry_ratio ~3.11`

## Correctness-First E2E Latency Reduction Plan

Priority 1: Keep finalization and DAG shape stable (correctness gate).
- Goal: maximize single-parent / descendant-fast-path and minimize expensive merged-parent path.
- Actions:
  - enforce healthy finalizer cadence and track `new_lfb_found` trend as release gate.
  - track `compute_parents_post_state path=*` distribution in CI (fail if merged-path share spikes beyond baseline).
  - investigate/resolve high block-retrieval retries (`retry_ratio ~3+`) because retry storms increase DAG divergence and merged-parent work.

Priority 2: Reduce propose-core critical path.
- Goal: lower `propose_core` p95 by shrinking checkpoint time.
- Actions:
  - optimize parent-state selection first (fast-path eligibility and cache hit retention).
  - bound expensive merge scope and add guardrails around pathological multi-parent merges.
  - add regression guard: `propose_total` and `checkpoint_parents_post_state` p95 thresholds in profiling CI.

Priority 3: Replay cost reduction after DAG-shape stabilization.
- Goal: lower replay mean and replay phase p95.
- Actions:
  - profile user-deploy replay segment (`block_replay_phase_user_deploys_time`) and prioritize top deploy/runtime hotspots.
  - keep checkpoint creation cost bounded and verify post-change replay histogram deltas against baseline snapshot.

Priority 4: Re-validate end-to-end every change set.
- Required gate for each optimization PR:
  - clean 120s soak x2 (memory slope + proc + finalizer summaries),
  - latency profile report (`profile-casper-latency.sh`),
  - finality suite (`12 passing`, `1 pending` expected).

## Recommended Next Steps

1. Add per-iteration finalizer health summary to soak output:
- counts of `new_lfb_found=true/false`, `filtered_agreements=0`, and finalized block deltas per validator.

2. Add optional pprof/heap snapshots around high `filtered_agreements=0` windows:
- correlate allocator growth with finalizer/search path activity.

3. Keep retriever dedup feature-gated until 3+ stable 120s A/B runs show clear net benefit.

4. Prioritize end-to-end latency plan around correctness-first guardrails:
- stable genesis/finalization progression,
- bounded queue growth,
- deterministic propose trigger behavior under high deploy load.
