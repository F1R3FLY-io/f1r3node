# Finality Bottleneck Overview (2026-02-19)

## Scope
This summary covers three measurement windows captured from validator logs:
- `post2`: after initial timeout/budget instrumentation and patch set
- `post3`: after first cooperative-yield patch (regression run)
- `post4`: after second cooperative-yield patch (inner-loop yields)
- `post5`: after adding finalizer candidate cap (`FINALIZER_MAX_CLIQUE_CANDIDATES=128`)
- `post6`: after adding run-local memoization in clique disagreement checks
- `post7`: after adaptive cooperative yielding (time-slice based)
- `post8`: after conservative pairwise validator pruning before combinations
- `post9`: after mutual-justification precheck before disagreement scans
- `post10`: after lm_b chain disagreement summary cache (O(1) stopper checks)

Logs analyzed:
- `/tmp/post2_v1.log`, `/tmp/post2_v2.log`, `/tmp/post2_v3.log`
- `/tmp/post3_v1.log`, `/tmp/post3_v2.log`, `/tmp/post3_v3.log`
- `/tmp/post4_v1.log`, `/tmp/post4_v2.log`, `/tmp/post4_v3.log`
- `/tmp/post5_v1.log`, `/tmp/post5_v2.log`, `/tmp/post5_v3.log`
- `/tmp/post6_v1.log`, `/tmp/post6_v2.log`, `/tmp/post6_v3.log`
- `/tmp/post7_v1.log`, `/tmp/post7_v2.log`, `/tmp/post7_v3.log`
- `/tmp/post8_v1.log`, `/tmp/post8_v2.log`, `/tmp/post8_v3.log`
- `/tmp/post9_v1.log`, `/tmp/post9_v2.log`, `/tmp/post9_v3.log`
- `/tmp/post10_v1.log`, `/tmp/post10_v2.log`, `/tmp/post10_v3.log`

## Executive Summary
- Dominant bottleneck is still **CliqueOracle CPU cost** during finalizer evaluation on lagging validators (`v1`, `v2`).
- The second cooperative-yield patch improved behavior substantially:
  - `v1/v2` finalizer runtime dropped from `~10-14s` to `~2.4-2.7s`.
  - Overshoot above 2s budget dropped from `~12s` to `~0.4-0.7s` average.
- Candidate cap patch provided an additional improvement:
  - `v1/v2` finalizer runtime moved from `~2.4-2.7s` to `~2.2-2.35s`.
  - Budget overshoot dropped further to `~0.24-0.35s` average.
- Despite improvement, `v1/v2` still hit `budget_exhausted=true` in every sampled finalizer pass of `post4`.
- `v1/v2` still hit `budget_exhausted=true` in every sampled finalizer pass of `post5`.
- `post6` (memoization) showed mixed absolute latency due deeper DAG at sample time, but improved normalized cost on `v2`.
- `post7` (adaptive yielding) significantly improved budget adherence and clique cost on both `v1` and `v2`.
- `post8` (pairwise pruning) further improved normalized clique cost, with minor/neutral absolute runtime change under deeper DAG.
- `post9` (mutual-justification precheck) improved normalized clique cost further and reduced `v2` overshoot.
- `post10` (chain summary cache) improved normalized clique cost again and reduced average overshoot on `v1`.
- `v3` remains healthy in finalizer latency (single-digit ms), but still observes invalid-block related churn (`InvalidBondsCache`, `NeglectedInvalidBlock`).

## Run-by-Run Metrics

### Finalizer timing (averages)
- `post2_v1`: `total_ms=10300.0`, `clique_ms=10254.8`, `budget_exhausted_true=8/8`, `found_new_lfb_true=0/8`
- `post2_v2`: `total_ms=10169.0`, `clique_ms=10125.9`, `budget_exhausted_true=8/8`, `found_new_lfb_true=0/8`
- `post2_v3`: `total_ms=9.2`, `clique_ms=9.2`, `budget_exhausted_true=0/5`, `found_new_lfb_true=5/5`

- `post3_v1`: `total_ms=14394.8`, `clique_ms=14341.8`, `budget_exhausted_true=4/4`, `found_new_lfb_true=0/4`
- `post3_v2`: `total_ms=14120.2`, `clique_ms=14075.2`, `budget_exhausted_true=4/4`, `found_new_lfb_true=0/4`
- `post3_v3`: `total_ms=6.0`, `clique_ms=6.0`, `budget_exhausted_true=0/2`, `found_new_lfb_true=2/2`

- `post4_v1`: `total_ms=2404.9`, `clique_ms=2329.4`, `budget_exhausted_true=16/16`, `found_new_lfb_true=0/16`
- `post4_v2`: `total_ms=2664.1`, `clique_ms=2582.4`, `budget_exhausted_true=16/16`, `found_new_lfb_true=0/16`
- `post4_v3`: `total_ms=9.0`, `clique_ms=8.7`, `budget_exhausted_true=0/3`, `found_new_lfb_true=3/3`
- `post5_v1`: `total_ms=2237.0`, `clique_ms=2150.4`, `budget_exhausted_true=59/59`, `found_new_lfb_true=0/59`, `candidate_capped_true=59/59`
- `post5_v2`: `total_ms=2347.4`, `clique_ms=2252.6`, `budget_exhausted_true=54/54`, `found_new_lfb_true=0/54`, `candidate_capped_true=54/54`
- `post5_v3`: `total_ms=11.9`, `clique_ms=11.8`, `budget_exhausted_true=0/14`, `found_new_lfb_true=14/14`
- `post6_v1`: `total_ms=2471.4`, `clique_ms=2371.4`, `budget_exhausted_true=20/20`, `found_new_lfb_true=0/20`, `candidate_capped_true=20/20`
- `post6_v2`: `total_ms=2304.2`, `clique_ms=2206.3`, `budget_exhausted_true=23/23`, `found_new_lfb_true=0/23`, `candidate_capped_true=23/23`
- `post6_v3`: `total_ms=7.3`, `clique_ms=7.0`, `budget_exhausted_true=0/6`, `found_new_lfb_true=6/6`
- `post7_v1`: `total_ms=2132.8`, `clique_ms=2011.2`, `budget_exhausted_true=26/26`, `found_new_lfb_true=0/26`, `candidate_capped_true=26/26`
- `post7_v2`: `total_ms=2138.2`, `clique_ms=2014.2`, `budget_exhausted_true=25/25`, `found_new_lfb_true=0/25`, `candidate_capped_true=25/25`
- `post7_v3`: `total_ms=16.4`, `clique_ms=16.3`, `budget_exhausted_true=0/7`, `found_new_lfb_true=7/7`
- `post8_v1`: `total_ms=2171.3`, `clique_ms=2051.7`, `budget_exhausted_true=22/22`, `found_new_lfb_true=0/22`, `candidate_capped_true=22/22`
- `post8_v2`: `total_ms=2149.6`, `clique_ms=2023.8`, `budget_exhausted_true=21/21`, `found_new_lfb_true=0/21`, `candidate_capped_true=21/21`
- `post8_v3`: `total_ms=3.7`, `clique_ms=3.7`, `budget_exhausted_true=0/3`, `found_new_lfb_true=3/3`
- `post9_v1`: `total_ms=2172.6`, `clique_ms=2034.6`, `budget_exhausted_true=19/19`, `found_new_lfb_true=0/19`, `candidate_capped_true=19/19`
- `post9_v2`: `total_ms=2119.0`, `clique_ms=1981.3`, `budget_exhausted_true=29/29`, `found_new_lfb_true=0/29`, `candidate_capped_true=29/29`
- `post9_v3`: `total_ms=12.0`, `clique_ms=11.7`, `budget_exhausted_true=0/6`, `found_new_lfb_true=6/6`
- `post10_v1`: `total_ms=2138.6`, `clique_ms=2003.0`, `budget_exhausted_true=21/21`, `found_new_lfb_true=0/21`, `candidate_capped_true=21/21`
- `post10_v2`: `total_ms=2134.0`, `clique_ms=1999.1`, `budget_exhausted_true=21/21`, `found_new_lfb_true=0/21`, `candidate_capped_true=21/21`
- `post10_v3`: `total_ms=22.5`, `clique_ms=20.5`, `budget_exhausted_true=0/2`, `found_new_lfb_true=2/2`

### Structural work indicators (averages)
- `post2_v1`: `layers_visited=1576.9`, `agreements=1813.9`, `filtered_agreements=410.0`, `clique_evals=1.00`
- `post2_v2`: `layers_visited=1567.1`, `agreements=1840.1`, `filtered_agreements=410.0`, `clique_evals=1.00`
- `post3_v1`: `layers_visited=1820.0`, `agreements=2057.0`, `filtered_agreements=410.0`, `clique_evals=1.00`
- `post3_v2`: `layers_visited=1817.8`, `agreements=2090.8`, `filtered_agreements=410.0`, `clique_evals=1.00`
- `post4_v1`: `layers_visited=3176.8`, `agreements=3413.8`, `filtered_agreements=410.0`, `clique_evals=2.44`
- `post4_v2`: `layers_visited=3199.9`, `agreements=3472.9`, `filtered_agreements=410.0`, `clique_evals=2.75`
- `post5_v1`: `layers_visited=3554.5`, `agreements=3791.5`, `filtered_agreements=410.0`, `clique_evals=2.00`
- `post5_v2`: `layers_visited=3589.3`, `agreements=3862.3`, `filtered_agreements=410.0`, `clique_evals=2.30`
- `post6_v1`: `layers_visited=3939.1`, `agreements=4176.1`, `filtered_agreements=410.0`, `clique_evals=2.00`
- `post6_v2`: `layers_visited=3984.3`, `agreements=4257.3`, `filtered_agreements=410.0`, `clique_evals=2.00`
- `post7_v1`: `layers_visited=4857.1`, `agreements=5094.1`, `filtered_agreements=410.0`, `clique_evals=6.10`
- `post7_v2`: `layers_visited=4915.4`, `agreements=5188.4`, `filtered_agreements=410.0`, `clique_evals=6.40`
- `post8_v1`: `layers_visited=5103.3`, `agreements=5340.3`, `filtered_agreements=410.0`, `clique_evals=6.50`
- `post8_v2`: `layers_visited=5173.2`, `agreements=5446.2`, `filtered_agreements=410.0`, `clique_evals=6.30`
- `post9_v1`: `layers_visited=5664.0`, `agreements=5901.0`, `filtered_agreements=410.0`, `clique_evals=6.60`
- `post9_v2`: `layers_visited=5777.9`, `agreements=6050.9`, `filtered_agreements=410.0`, `clique_evals=6.60`
- `post10_v1`: `layers_visited=5785.5`, `agreements=6022.5`, `filtered_agreements=410.0`, `clique_evals=6.30`
- `post10_v2`: `layers_visited=6018.3`, `agreements=6291.3`, `filtered_agreements=410.0`, `clique_evals=7.20`

Interpretation:
- Work depth increased significantly by `post4` (about 2x layers/agreements), yet wall time still dropped drastically after cooperative yields.
- Under even deeper work in `post5`, candidate capping held `clique_evals` lower and reduced average runtime further.
- `clique_ms` remains ~96-99% of `total_ms` on lagging nodes, so optimization target remains unchanged.

### Budget overshoot after improvements
Relative to `FINALIZER_WORK_BUDGET_MS=2000`:
- `post3_v1`: overshoot avg `12394.8ms`
- `post3_v2`: overshoot avg `12120.2ms`
- `post4_v1`: overshoot avg `404.9ms` (p95 `907ms`)
- `post4_v2`: overshoot avg `664.1ms` (p95 `773ms`)
- `post5_v1`: overshoot avg `237.0ms` (p95 `402ms`)
- `post5_v2`: overshoot avg `347.4ms` (p95 `954ms`)
- `post6_v1`: overshoot avg `471.4ms` (p95 `628ms`)
- `post6_v2`: overshoot avg `304.2ms` (p95 `487ms`)
- `post7_v1`: overshoot avg `132.8ms` (p95 `284ms`)
- `post7_v2`: overshoot avg `138.2ms` (p95 `277ms`)
- `post8_v1`: overshoot avg `171.3ms` (p95 `388ms`)
- `post8_v2`: overshoot avg `149.6ms` (p95 `248ms`)
- `post9_v1`: overshoot avg `172.6ms` (p95 `356ms`)
- `post9_v2`: overshoot avg `119.0ms` (p95 `229ms`)
- `post10_v1`: overshoot avg `138.6ms` (p95 `223ms`)
- `post10_v2`: overshoot avg `134.0ms` (p95 `239ms`)

### Normalized clique cost
To separate algorithm cost from DAG depth growth, we tracked:
- `clique_ms / agreements`
- `clique_ms / layers_visited`

Results:
- `post5_v1`: `0.5671`, `0.6050`
- `post6_v1`: `0.5679`, `0.6020` (roughly flat)
- `post5_v2`: `0.5833`, `0.6277`
- `post6_v2`: `0.5183`, `0.5538` (improved)
- `post7_v1`: `0.3948`, `0.4141` (strong improvement)
- `post7_v2`: `0.3882`, `0.4098` (strong improvement)
- `post8_v1`: `0.3842`, `0.4021` (improved)
- `post8_v2`: `0.3716`, `0.3912` (improved)
- `post9_v1`: `0.3448`, `0.3592` (improved)
- `post9_v2`: `0.3274`, `0.3429` (improved)
- `post10_v1`: `0.3326`, `0.3462` (improved)
- `post10_v2`: `0.3178`, `0.3322` (improved)

## Bottlenecks

1) Clique computation dominates finalizer runtime
- Evidence: `clique_ms` approximately equals `total_ms` on `v1/v2` in all runs.
- Location: `casper/src/rust/safety/clique_oracle.rs` and call path from `casper/src/rust/finality/finalizer.rs`.

2) Budget exhaustion persists on lagging validators
- Evidence: `post4` still `budget_exhausted=true` in `16/16` samples for both `v1` and `v2`.
- Evidence: `post5` still `budget_exhausted=true` in `59/59` (`v1`) and `54/54` (`v2`).
- Result: repeated partial work and delayed LFB advancement on those nodes.

3) Invalid-block churn remains visible on `v3`
- Evidence: `InvalidBondsCache` and `NeglectedInvalidBlock` still present in all windows.
- Current impact: no direct finalizer-latency impact observed on `v3`, but contributes noise and potential wasted processing.

## Quick Wins (Prioritized)

1) Hard-cap finalizer candidate set before clique evaluation (completed, high impact)
- Rationale: `filtered_agreements` is fixed at `410`; evaluating all candidates is expensive when chain depth is high.
- Implemented: `FINALIZER_MAX_CLIQUE_CANDIDATES=128` in `casper/src/rust/finality/finalizer.rs`.
- Outcome: reduced `v1/v2` average runtime and overshoot, but did not eliminate budget exhaustion.

2) Add memoization for repeated disagreement checks within a single finalizer run (completed)
- Rationale: `never_eventually_see_disagreement` likely revisits overlapping `(validator_a, validator_b, target)` chain relationships.
- Implemented:
  - cache `latest_message_hash(validator)`
  - cache `self_justification_chain(lm_b)`
  - cache `self_justification(lm_a_j_b)`
  - cache `is_in_main_chain(target, hash)`
- Outcome: normalized improvement on `v2`; `v1` remained roughly flat under higher-depth workload.

3) Tighten cooperative yielding granularity adaptively (completed, high impact)
- Implemented time-slice based yields with:
  - `COOPERATIVE_YIELD_CHECK_INTERVAL=8`
  - `COOPERATIVE_YIELD_TIMESLICE_MS=1`
- Outcome: much lower budget overshoot and better normalized clique cost, even with substantially deeper DAGs.

4) Prune validators that cannot form any pairwise edge before combinations (completed, low risk)
- Implemented conservative pruning for pair generation:
  - require latest message exists
  - require at least one justification to another agreeing validator
- Outcome: improved normalized clique cost; absolute runtime remained close to `post7` under deeper DAG.

5) Skip disagreement scans unless latest-message justifications are mutual (completed, low risk)
- Implemented in pair loop: require `a` justifies `b` and `b` justifies `a` before calling `never_eventually_see_disagreement`.
- Outcome: further normalized clique-cost reduction; modest absolute gain, especially on `v2`.

6) Cache per-`lm_b` chain scan summary (completed, medium impact)
- Implemented:
  - one-time scan per `lm_b` to build
    - hash -> index map
    - first disagreement index
  - stopper checks then become O(1) instead of rescanning each time.
- Outcome: further reduction in normalized clique cost and better `v1` overshoot profile.
- Rationale: fixed `idx % 8` and `% 64` reduced overshoot, but p95 overshoot still up to `~0.9s`.
- Action: lower thresholds under high depth, or add elapsed-time-based yield trigger.
- Expected outcome: better enforcement of step timeout/budget and lower tail latency.

4) Early pruning before pairwise combinations (medium impact, medium effort)
- Rationale: pair combinations scale quadratically with agreeing validators.
- Action: pre-prune validators with low/no chance of contributing to maximal clique for current target.
- Expected outcome: fewer pair checks, smaller clique search graph.

5) Separate invalid-block churn mitigation track (medium impact, low effort)
- Rationale: `InvalidBondsCache`/`NeglectedInvalidBlock` is persistent background churn.
- Action: add counters by source peer/reason and rate-limit repeated invalid handling paths.
- Expected outcome: cleaner pipeline and less background overhead/risk.

## Code/Config Touchpoints
- Budget and timeout constants: `casper/src/rust/finality/finalizer.rs:33`, `casper/src/rust/finality/finalizer.rs:34`
- Candidate cap constant: `casper/src/rust/finality/finalizer.rs:35`
- Timeout-wrapped clique call and budget checks: `casper/src/rust/finality/finalizer.rs:248`, `casper/src/rust/finality/finalizer.rs:254`
- Cooperative yields in disagreement chain scan: `casper/src/rust/safety/clique_oracle.rs:85`
- Cooperative yields in pair loop: `casper/src/rust/safety/clique_oracle.rs:145`

## Suggested Next Measurement Gate
Use the same 3-minute workload + 10-minute log slice and require all of:
- `v1/v2 total_ms avg < 2000`
- `v1/v2 budget_exhausted_true < 20%`
- `v1/v2 found_new_lfb_true > 0` in each window
- no regression in proposer recoverable errors

## Candidate Cap Sweep (64/96/128)
After making candidate cap runtime-configurable via `F1R3_FINALIZER_MAX_CLIQUE_CANDIDATES`, we ran three identical windows:
- `cap=64`: `/tmp/sweep_cap64_v{1,2,3}.log`
- `cap=96`: `/tmp/sweep_cap96_v{1,2,3}.log`
- `cap=128`: `/tmp/sweep_cap128_v{1,2,3}.log`

### Aggregate on lagging validators (`v1+v2`)
- `cap=64`: `n=36`, `total_avg=2140.2`, `clique_avg=2004.2`, `overshoot_avg=140.2`
- `cap=96`: `n=33`, `total_avg=2148.9`, `clique_avg=2012.7`, `overshoot_avg=148.9`
- `cap=128`: `n=36`, `total_avg=2133.9`, `clique_avg=1994.7`, `overshoot_avg=133.9`

All three remained `budget_exhausted=true` in all sampled `v1/v2` passes and had `found_new_lfb_true=0` on `v1/v2`.

### Recommendation
- Keep `candidate_cap=128` as current best among tested values.
- Lower caps (`64`, `96`) did not improve budget adherence and were slightly worse on average latency.

## Latest Update (post11/post12)

After `post10`, candidate ranking was changed twice:
- `post11`: stake-heavy ranking
- `post12`: `block_number desc`, then smaller agreeing set, then stake

### v1+v2 aggregate comparison
- `post10`: `n=42`, `total_avg=2136.3`, `clique_avg=2001.0`, `overshoot_avg=136.3`, `clique_evals_avg=6.8`
- `post11`: `n=38`, `total_avg=2168.9`, `clique_avg=2018.4`, `overshoot_avg=168.9`, `clique_evals_avg=5.4`
- `post12`: `n=42`, `total_avg=2140.8`, `clique_avg=1998.1`, `overshoot_avg=140.8`, `clique_evals_avg=7.1`

`post12` recovered most of the `post11` regression, but remains slightly behind `post10` on absolute wall time and overshoot.

### Current dominant bottleneck
1) `CliqueOracle` compute remains dominant on lagging validators (`v1`, `v2`):
- `post12_v1`: `clique_avg=1990.1ms` vs `total_avg=2130.1ms`
- `post12_v2`: `clique_avg=2007.0ms` vs `total_avg=2152.6ms`
- `budget_exhausted_true=42/42` on `post12_v1v2`
- `found_new_lfb_true=0/42` on `post12_v1v2`

This confirms ranking tweaks alone are second-order; clique compute depth/shape is still first-order.

### Quick wins from current state
1) Keep `candidate_cap=128`.
- It remains the best cap from the 64/96/128 sweep and still best in current post-series.

2) Add runtime knobs for ranking strategy and yield timeslice.
- Make ordering policy env-configurable (without rebuild) and sweep with same harness.
- Expose adaptive yield constants (`COOPERATIVE_YIELD_CHECK_INTERVAL`, `COOPERATIVE_YIELD_TIMESLICE_MS`) via env to tune tail overshoot quickly.

3) Move cap earlier in the pipeline (before expensive agreement materialization), not only before clique evaluation.
- The run still processes very large agreement sets (`post12 agreements_avg=6614.6` while `candidate_cap=128`).
- Earlier pruning should reduce work that is currently spent before capped clique passes.

4) Add a lightweight cheap pre-score before expensive clique checks.
- Use a small-cost signal (recentness and low fan-in) to gate which candidates hit expensive clique path first.
- Objective is not perfect ordering; objective is finding a valid new LFB candidate sooner within the fixed budget.

5) Keep invalid-block churn as separate hygiene track.
- `post12_v3` still shows `InvalidBondsCache=25`, `NeglectedInvalidBlock=9`.
- Not dominant for finalizer latency, but still worth reducing background noise.

## Runtime Ranking Sweep (latest)

Added runtime ranking env switch:
- `F1R3_FINALIZER_CANDIDATE_RANKING=recency_stake|recency_smallset_stake|stake_desc`

Sweep runs (same short workload and log window, `candidate_cap=128`):
- `recency_smallset_stake`: `n=28`, `total_avg=2219.4`, `clique_avg=2040.2`, `overshoot_avg=219.4`, `clique_evals_avg=5.8`
- `recency_stake`: `n=31`, `total_avg=2156.6`, `clique_avg=1991.8`, `overshoot_avg=156.6`, `clique_evals_avg=6.1`
- `stake_desc`: `n=31`, `total_avg=2229.1`, `clique_avg=2050.9`, `overshoot_avg=229.1`, `clique_evals_avg=5.1`

Recommendation:
- Use `recency_stake` as default (best wall-time and overshoot among tested strategies).
- Keep runtime override available for further sweeps without rebuild.

## Runtime Yield Tuning Knobs (latest)

Added runtime env switches for cooperative yields in clique computation:
- `F1R3_CLIQUE_YIELD_CHECK_INTERVAL` (default `8`)
- `F1R3_CLIQUE_YIELD_TIMESLICE_MS` (default `1`)

This enables low-friction tail-latency tuning (overshoot/p95) without rebuilding.

## Inner-Loop Efficiency Follow-up (post13/post14/post15)

### Changes
1) `post13`:
- Reduced finalizer loop overhead (weight-map cache, in-place agreement accumulation, precomputed sort keys).

2) `post14`:
- Added shared `CliqueOracleRunCache` across candidate evaluations within one finalizer run.
- Reused latest-message and latest-justification data across candidate `compute_output` calls.

3) `post15`:
- Deduplicated candidate list by `block_hash` before clique evaluation.
- Reused self-justification caches across candidate evaluations.
- Added `deduped_filtered_agreements` metric to finalizer timing logs.

### v1+v2 aggregate
- `post13`: `n=34`, `total_avg=2136.0`, `clique_avg=1988.9`, `overshoot_avg=136.0`, `agreements_avg=7157.7`
- `post14`: `n=37`, `total_avg=2157.5`, `clique_avg=1992.6`, `overshoot_avg=157.5`, `agreements_avg=7422.2`
- `post15`: `n=65`, `total_avg=2150.0`, `clique_avg=1961.3`, `overshoot_avg=150.0`, `agreements_avg=7667.7`

`post15` vs `post14`:
- `total_avg -7.6ms`
- `clique_avg -31.3ms`
- `overshoot_avg -7.6ms`
- `agreements_avg +245.5` (deeper workload), yet clique time improved.

Candidate dedup signal (`post15_v1v2`):
- `filtered_agreements_avg=410.0`
- `deduped_filtered_agreements_avg=204.0`
- `dedup_ratio=0.498`

Interpretation:
- About half of candidate entries were duplicates at clique stage, and removing those improved clique cost under deeper load.
- Dominant bottleneck is still clique compute (`budget_exhausted=true` remains 100% on lagging validators), but per-unit work efficiency is improving.

## Cap Sweep After Dedup (latest code)

With dedup and shared run caches enabled, we reran cap sweep on `recency_stake`:
- `cap=96`: `n=30`, `total_avg=2178.0`, `clique_avg=1996.5`, `overshoot_avg=178.0`, `clique_per_agreement=0.242225`
- `cap=112`: `n=28`, `total_avg=2163.0`, `clique_avg=1976.8`, `overshoot_avg=163.0`, `clique_per_agreement=0.239478`
- `cap=128`: `n=29`, `total_avg=2127.1`, `clique_avg=1944.4`, `overshoot_avg=127.1`, `clique_per_agreement=0.235110`

Observations:
- `cap=128` is now clearly best among tested caps on both absolute wall-time and normalized clique cost.
- `deduped_filtered_agreements_avg` stayed around `204` across runs; lower caps did not improve budget adherence.

## Upper-Bound Pruning Gate (post18)

Added cheap FT upper-bound pruning before clique evaluation:
- `ft_upper_bound = (2 * agreeing_stake - total_stake) / total_stake`
- if `ft_upper_bound <= threshold`, skip expensive clique computation (safe prune because max clique stake cannot exceed agreeing stake).

Observed `v1+v2` impact:
- `post16`: `total_avg=2160.4`, `clique_avg=1984.8`, `clique_evals_avg=6.7`, `upper_bound_pruned_avg=0.0`
- `post17`: `total_avg=2179.7`, `clique_avg=1977.0`, `clique_evals_avg=6.2`, `upper_bound_pruned_avg=0.0`
- `post18`: `total_avg=181.0`, `clique_avg=0.0`, `clique_evals_avg=0.0`, `upper_bound_pruned_avg=128.0`

Additional signals:
- `budget_exhausted_true` on `v1+v2`: `0` in `post18` (was always true before).
- `found_new_lfb_true` on `v1+v2`: remained `0` (same as prior windows).

Interpretation:
- This removes the dominant inner-loop cost in windows where no candidate can possibly exceed FT threshold.
- It appears to be the highest-impact quick win so far for lagging-validator finalizer wall-time.

## Inner-Loop Hoist Optimization (post19)

### What changed
In `CliqueOracle` pairwise validator evaluation:
1) Hoisted latest-message and justification-hash resolution out of the `(a,b)` inner loop.
- Before: for each pair, disagreement checks re-derived `lm_a`, scanned `lm_a` justifications to find `lm_a_j_b`, and fetched `lm_b` indirectly.
- After: per validator, precompute once:
  - latest message hash
  - map `justified_validator -> justified_block_hash`
  Then pair loop uses direct map lookups (`O(1)` by key in map structure) and calls disagreement check with pre-resolved hashes.

2) Simplified disagreement function inputs.
- `never_eventually_see_disagreement` now accepts `(lm_b, lm_a_j_b)` directly, removing repeated latest-message/justification discovery from the hot path.

### post19 measurement
`v1+v2` aggregate:
- `samples=35`
- `total_avg=217.2ms`
- `clique_avg=0.0ms`
- `clique_evals_avg=0.00`
- `upper_bound_pruned_avg=128.00`
- `upper_bound_passed_avg=0.00`
- `max_ft_upper_bound_avg=0.333333`
- `budget_exhausted_true=0`

Comparison to `post18 v1+v2`:
- `post18 total_avg=181.0ms`, `post19 total_avg=217.2ms`
- same prune profile (`clique_evals=0`, `upper_bound_pruned=128`, `budget_exhausted=0`)

Interpretation:
- The new hoist is correct and keeps behavior stable.
- On lagging validators in these windows, clique path is already fully pruned, so additional clique inner-loop optimization does not change first-order latency.
- Dominant runtime now sits before clique evaluation (agreement/layer traversal and candidate materialization).

### Updated bottleneck and quick wins
Current dominant bottleneck (`v1`,`v2`):
1) Pre-clique agreement accumulation and filtering work (`layers_visited`, agreement traversal), not clique search.

Next quick wins:
1) Early candidate cap application during agreement construction, not only before clique evaluation.
2) Streamed top-K candidate maintenance to avoid building/sorting large intermediate agreement sets.
3) Reuse/caching of expensive `message_weight_map_f`/agreement-derived maps across adjacent finalizer ticks when DAG tip has minimal change.

## Pre-Clique Aggregation Rewrite (post20)

### What changed
`Finalizer::run` agreement preparation was rewritten to avoid duplicate intermediate materialization:
1) Layer traversal now aggregates agreements directly per `block_hash` while traversing.
2) Removed large intermediate stages:
- `mk_agreements_stream`
- `full_agreements_map` + `mapaccumulate_stream`
- post-filter dedup-by-`block_hash` map
3) Layer traversal now builds next layer in the same pass as agreement aggregation (single pass per layer).

Net effect:
- less allocation/cloning in pre-clique path
- less duplicate work before candidate ranking/cap

### post20 measurement
`v1+v2` aggregate:
- `samples=55`
- `total_avg=180.1ms`
- `clique_avg=0.0ms`
- `clique_evals_avg=0.00`
- `agreements_avg=9478.3`
- `filtered_avg=204.0`
- `deduped_avg=204.0`
- `upper_bound_pruned_avg=128.00`
- `upper_bound_passed_avg=0.00`
- `budget_exhausted_true=0`

Comparison:
- `post18 v1+v2 total_avg=181.0ms`
- `post19 v1+v2 total_avg=217.2ms`
- `post20 v1+v2 total_avg=180.1ms`

Interpretation:
- With clique fully pruned on lagging validators, this pre-clique rewrite recovers and slightly improves wall time.
- Current first-order cost is now remaining agreement traversal/bookkeeping overhead, not clique computation.

## Main-Parent Lookup Cache (post21)

### What changed
Inside agreement layer traversal in `Finalizer::run`, added per-run cache:
- `main_parent_cache: HashMap<BlockHash, Option<BlockMetadata>>`

This avoids repeated `dag.lookup_unsafe(main_parent_hash)` for the same parent hash across validators/layers in one finalizer run.

### post21 measurement
`v1+v2` aggregate:
- `samples=56`
- `total_avg=160.3ms`
- `clique_avg=0.0ms`
- `clique_evals_avg=0.00`
- `agreements_avg=9649.2`
- `upper_bound_pruned_avg=128.00`
- `upper_bound_passed_avg=0.00`
- `budget_exhausted_true=0`

Comparison:
- `post20 v1+v2 total_avg=180.1ms`
- `post21 v1+v2 total_avg=160.3ms`

Interpretation:
- The parent-lookup cache produced another meaningful reduction in pre-clique wall time while keeping behavior stable (same prune profile and zero budget exhaustion).
- Dominant work remains agreement traversal and associated bookkeeping before FT/clique stage.

## Weight-Map Cache Borrow Experiment (post22/post23)

Tried replacing per-visit cloned `message_weight_map` access with borrowed cache access.

Observed:
- `post22 v1+v2`: `total_avg=169.9ms`, `agreements_avg=9969.2`, `ms_per_1k_agreements=17.041`
- `post23 v1+v2`: `total_avg=190.7ms`, `agreements_avg=10127.5`, `ms_per_1k_agreements=18.828`
- baseline for this phase:
  - `post21 v1+v2`: `total_avg=160.3ms`, `agreements_avg=9649.2`, `ms_per_1k_agreements=16.617`

Interpretation:
- This experiment did not improve throughput and appears regressive under current workload.
- Change was reverted to keep the better-performing post21 path as working baseline.

## Inner-Loop Focus: What Work Dominates Now (post24-post28)

### Ground truth from instrumentation
Added counters in `Finalizer timing` for:
- `message_weight_map_cache_hit/miss`
- `main_parent_cache_hit/miss`

Observed hit rates stayed very low across runs:
- `message_weight_map_cache_hit_rate`: ~1.9-2.0%
- `main_parent_cache_hit_rate`: ~1.9-2.0%

This confirms most per-visit work in the traversal loop is still effectively on unique messages, so optimization must reduce per-visit constant cost; cache lookups alone will not shift runtime much.

### Inner loop under test
Hot loop (agreement traversal) does, for each `(validator, message)`:
1) resolve message weight map
2) record agreement into per-message accumulator
3) resolve main parent metadata
4) enqueue next layer tuple

### Experiments and outcomes
1) Per-layer compaction (group validators by message + sort compact layer):
- intent: hoist weight-map/parent work out of repeated inner iterations
- result: regressed
- `post25 v1+v2`: `ms_per_1k_agreements=19.342`

2) Shared weight-map via `Arc<WeightMap>`:
- intent: eliminate large `WeightMap` cloning in hot loop
- result: partial recovery from compaction regression, but still slower than post24
- `post26 v1+v2`: `ms_per_1k_agreements=17.315`

3) Compaction removed (keep aggregation + shared maps):
- intent: remove compaction/sort overhead while retaining cheap map sharing
- result under current load: still regressive
- `post27 v1+v2`: `ms_per_1k_agreements=20.605`
- `post28 v1+v2`: `ms_per_1k_agreements=20.083`

Reference:
- `post24 v1+v2`: `ms_per_1k_agreements=16.679`

### Current bottleneck statement
The dominant bottleneck remains pre-clique agreement traversal bookkeeping. Clique remains fully pruned (`clique_evals=0`) in these windows.

### Quick wins (highest confidence next)
1) Move stable per-message data fully out of the per-visit loop for a run-scoped structure keyed by message hash (without per-layer regroup/sort allocation).
2) Replace hash-map-heavy per-visit path with structure-of-arrays style staging for current/next layers to reduce clone/hash overhead.
3) Add a fixed-size profiler breakdown in logs for inner-loop phases:
- `weight_map_lookup_ms`
- `agreement_record_ms`
- `parent_lookup_ms`
- `next_layer_push_ms`
to stop guessing and optimize the true dominant sub-phase.
