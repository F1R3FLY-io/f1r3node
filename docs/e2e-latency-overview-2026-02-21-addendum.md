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

## Priority 1 Implementation #1 (Block Retriever Requery Cooldown)

Change:
- `casper/src/rust/engine/block_retriever.rs`
  - Added per-hash `peer_requery_last_request` cooldown tracking.
  - Added configurable peer requery cooldown guard for `peer_requery` path.
    - Env: `F1R3_BLOCK_RETRIEVER_PEER_REQUERY_COOLDOWN_MS`
    - Default: `1000` ms
  - Added retry action metric label: `peer_requery_suppressed`.
  - Added cleanup/sweep for the new cooldown map.
 - `docker/shard-with-autopropose.yml`
   - Added env default:
     - `F1R3_BLOCK_RETRIEVER_PEER_REQUERY_COOLDOWN_MS=${F1R3_BLOCK_RETRIEVER_PEER_REQUERY_COOLDOWN_MS:-1000}`

Validation:
- Build: `cargo check -p node` passed.
- Clean soak (120s):
  - `/tmp/casper-validator-leak-soak-peerrequerycooldown-20260221T223644Z/summary.txt`
  - Mean RSS slope: `6.250450 MiB/s` (vs prior reference `6.456156`, delta `-3.19%`).
- Latency profile window:
  - `/tmp/casper-latency-profile-peerrequerycooldown-20260221T224107Z/summary.txt`
  - `block_requests_retry_ratio: 1.26`
  - `compute_parents_post_state path=merged avg_total_ms: 19.57`
- Correctness:
  - `~/work/asi/tests/firefly-rholang-tests-finality-suite-v2/test.sh`
  - Result: `12 passing`, `1 pending`

Note on comparability:
- Retry ratio / merged-path timing are workload-window sensitive.
- Compared with earlier long-window snapshot (`retry_ratio 3.11`, merged avg `1309.77 ms`), this run is strongly better directionally, but we should still run A/B replicated windows for final decisioning.

### Replicated Stability Check (3 runs, 2026-02-21)

Additional clean runs:
- Soak summaries:
  - `/tmp/casper-validator-leak-soak-peerrequerycooldown-rep2-20260221T224426Z/summary.txt`
  - `/tmp/casper-validator-leak-soak-peerrequerycooldown-rep3-20260221T224901Z/summary.txt`
- Latency summaries:
  - `/tmp/casper-latency-profile-peerrequerycooldown-rep2-20260221T224851Z/summary.txt`
  - `/tmp/casper-latency-profile-peerrequerycooldown-rep3-20260221T225321Z/summary.txt`

Aggregates across 3 cooldown runs:
- Memory slope means:
  - run1 `6.250450`
  - run2 `6.359459`
  - run3 `6.400000`
  - 3-run mean `6.336636` (`-1.85%` vs prior reference `6.456156`)
- Block-retriever retry ratio:
  - mean `1.327` (min `1.26`, max `1.37`)
  - baseline reference window: `3.11`
- `compute_parents_post_state path=merged` avg_total_ms:
  - mean `138.26` (min `19.57`, max `205.52`)
  - baseline reference window: `1309.77`

Interpretation update:
- Retry pressure reduction appears stable and substantial.
- Merged-path cost remains variable but clearly improved vs prior reference windows.
- Memory growth remains noisy but trends modestly better on aggregate with cooldown enabled.

### Cooldown Tuning Snapshot (90s A/B, preliminary)

Build under test:
- `1203b92b` (`F1R3_BLOCK_RETRIEVER_PEER_REQUERY_COOLDOWN_MS` configurable)
- Image digest: `sha256:30b7f0a0bdda6463f727bf20202a97d2280a23a1b00419a8f17c947e395d02d3`

Runs:
- `1000ms` cooldown:
  - Soak: `/tmp/casper-validator-leak-soak-peerrequerycooldown-1000-20260221T230514Z/summary.txt`
  - Profile: `/tmp/casper-latency-profile-peerrequerycooldown-1000-20260221T230906Z/summary.txt`
  - Mean slope: `7.365401`
  - Retry ratio: `0.97`
  - `merged` path avg_total_ms: `29.37`
- `1500ms` cooldown:
  - Soak: `/tmp/casper-validator-leak-soak-peerrequerycooldown-1500-20260221T230924Z/summary.txt`
  - Profile: `/tmp/casper-latency-profile-peerrequerycooldown-1500-20260221T231314Z/summary.txt`
  - Mean slope: `9.002532`
  - Retry ratio: `0.65`
  - `merged` path avg_total_ms: `126.95`

Observation:
- `1500ms` reduces retry ratio further in this short run, but has worse memory slope and merged-path cost in the same window.
- `peer_requery_suppressed` remained `0` in both windows, so this pair does not yet show cooldown suppression events directly.
- Treat this as inconclusive; we need longer replicated windows (>=120s) before selecting a new default beyond `1000ms`.

### Cooldown Sweep (120s, clean runs: 1000 / 1500 / 2000)

Artifacts:
- `1000ms`
  - Soak: `/tmp/casper-validator-leak-soak-peerrequerycooldown-1000-120s-20260221T231450Z/summary.txt`
  - Profile: `/tmp/casper-latency-profile-peerrequerycooldown-1000-120s-20260221T231915Z/summary.txt`
- `1500ms`
  - Soak: `/tmp/casper-validator-leak-soak-peerrequerycooldown-1500-120s-20260221T231927Z/summary.txt`
  - Profile: `/tmp/casper-latency-profile-peerrequerycooldown-1500-120s-20260221T232347Z/summary.txt`
- `2000ms`
  - Soak: `/tmp/casper-validator-leak-soak-peerrequerycooldown-2000-120s-20260221T232417Z/summary.txt`
  - Profile: `/tmp/casper-latency-profile-peerrequerycooldown-2000-120s-20260221T232838Z/summary.txt`

Headline metrics:
- Memory slope mean (MiB/s):
  - `1000`: `6.791369`
  - `1500`: `6.710811`
  - `2000`: `5.542643` (best on this single run)
- Block retriever retry ratio:
  - `1000`: `1.26`
  - `1500`: `1.10` (best)
  - `2000`: `1.69` (worst)
- `compute_parents_post_state path=merged` avg_total_ms:
  - `1000`: `51.76`
  - `1500`: `19.79` (best)
  - `2000`: `298.43` (worst)
- `peer_requery_suppressed`:
  - `1000`: `0`
  - `1500`: `2`
  - `2000`: `4`

Interpretation:
- `2000ms` improves memory slope in this sample but clearly harms retry pressure and merged-path cost (likely over-throttling).
- `1500ms` is the best balanced setting in this sweep: lower retry ratio than `1000` and significantly better merged-path latency than both `1000` and `2000`, with slightly better memory slope than `1000`.
- Recommended next action: run one more replicated 120s pair (`1000` vs `1500`) before changing default fleet value.

### Replicated 120s Confirmation Pair (`1000` vs `1500`)

Artifacts:
- `1000ms` rep2:
  - Soak: `/tmp/casper-validator-leak-soak-peerrequerycooldown-1000-120s-rep2-20260221T233023Z/summary.txt`
  - Profile: `/tmp/casper-latency-profile-peerrequerycooldown-1000-120s-rep2-20260221T233445Z/summary.txt`
- `1500ms` rep2:
  - Soak: `/tmp/casper-validator-leak-soak-peerrequerycooldown-1500-120s-rep2-20260221T233500Z/summary.txt`
  - Profile: `/tmp/casper-latency-profile-peerrequerycooldown-1500-120s-rep2-20260221T233923Z/summary.txt`

Two-run means (using both 120s runs per setting):
- Memory slope mean:
  - `1000`: `6.328568`
  - `1500`: `6.574899` (`+3.89%` vs `1000`, worse)
- Retry ratio mean:
  - `1000`: `1.325`
  - `1500`: `1.210` (better)
- `compute_parents_post_state path=merged` avg_total_ms mean:
  - `1000`: `34.97`
  - `1500`: `111.52` (worse)
- `peer_requery_suppressed` mean:
  - `1000`: `0.00`
  - `1500`: `2.50`

Final interpretation for now:
- `1500` consistently lowers retry ratio, but worsens merged-path latency and memory slope on replicated 120s averages.
- Keep default at `1000ms` pending deeper investigation (likely interaction with finalizer cadence / DAG shape dominates).

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
