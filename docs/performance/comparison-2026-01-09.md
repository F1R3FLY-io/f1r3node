# Performance Comparison: Baseline vs Post-Optimization

**Date:** 2026-01-09
**Phases Completed:** 1, 2, 3

---

## Executive Summary

| Category | Result | Key Improvement |
|----------|--------|-----------------|
| Phlogiston | **+130% throughput** | Atomic ordering + code optimization |
| Checkpoint (small) | **+295% revert speed** | im::Vector O(1) clone |
| Multi-channel | **+142% throughput** | DashMap sharded locking |
| High contention (8t) | **+38% throughput** | DashMap reduces lock contention |
| consume_immediate | **-50% throughput** | Trade-off for parallel gains |

**Net Assessment:** Optimizations successful. Parallel/concurrent workloads see major improvements. Sequential single-channel consume has regressed but this is an acceptable trade-off.

---

## Detailed Comparison

### Phlogiston (Gas) Accounting

```
                    Baseline         Post-Optimization      Change
                    --------         -----------------      ------
100 ops             8.61 µs          3.79 µs                -56% time
                    11.6 Melem/s     26.2 Melem/s           +126% throughput

1,000 ops           87.1 µs          38.2 µs                -56% time
                    11.5 Melem/s     26.2 Melem/s           +128% throughput

10,000 ops          875 µs           367 µs                 -58% time
                    11.4 Melem/s     27.1 Melem/s           +138% throughput
```

**Verdict:** Exceeds expectations. 2.3x improvement in throughput.

---

### Checkpoint Operations

```
                         Baseline      Post-Opt       Change
                         --------      --------       ------
create/10 channels       34.3 µs       23.6 µs        -32%  IMPROVED
revert/10 channels       34.2 µs       8.65 µs        -75%  IMPROVED (+295% thrpt)

create/100 channels      35.2 µs       32.4 µs        -9%   IMPROVED
revert/100 channels      17.7 µs       17.4 µs        ~0%   NO CHANGE

create/1000 channels     168.4 µs      121.6 µs       -28%  IMPROVED
revert/1000 channels     90.1 µs       114.2 µs       +27%  REGRESSED
```

**Verdict:** Small/medium checkpoints significantly improved. Large checkpoint revert needs investigation.

---

### Multi-Channel Produce

```
                    Baseline         Post-Optimization      Change
                    --------         -----------------      ------
10 channels         5.63 µs          2.12 µs                -63% time (+165% thrpt)
100 channels        19.6 µs          8.08 µs                -59% time (+142% thrpt)
1,000 channels      196.3 µs         85.6 µs                -56% time (+129% thrpt)
```

**Verdict:** Excellent. DashMap sharded locking provides 2x+ improvement.

---

### Lock Contention (Same Channel)

```
                         Baseline      Post-Opt       Change
                         --------      --------       ------
2 threads                ~656 µs       931 µs         +43%  REGRESSED
4 threads                ~1.50 ms      1.24 ms        -17%  IMPROVED
8 threads                ~3.25 ms      2.34 ms        -28%  IMPROVED
```

**Verdict:** DashMap overhead at low thread counts. Improvement at 4+ threads as expected.

---

### Produce/Consume Cycle

```
                              Baseline      Post-Opt       Change
                              --------      --------       ------
produce_only/10K              436.6 µs      415.8 µs       -5%   IMPROVED
consume_immediate/100         21.0 µs       36.2 µs        +72%  REGRESSED
consume_immediate/1K          186.5 µs      353.7 µs       +90%  REGRESSED
consume_immediate/10K         1.84 ms       3.77 ms        +105% REGRESSED
```

**Root Cause:** im::Vector O(log n) access + DashMap lookup overhead
**Trade-off:** Acceptable for checkpoint/concurrent gains

---

## Summary Table

| Benchmark | Baseline | Post-Opt | Δ Time | Δ Throughput |
|-----------|----------|----------|--------|--------------|
| phlogiston/10K | 875 µs | 367 µs | -58% | **+138%** |
| checkpoint_create/1K | 168 µs | 122 µs | -28% | +39% |
| checkpoint_revert/10 | 34.2 µs | 8.65 µs | -75% | **+295%** |
| multi_channel/100 | 19.6 µs | 8.08 µs | -59% | **+142%** |
| lock_contention/8t | 3.25 ms | 2.34 ms | -28% | +38% |
| consume_immediate/10K | 1.84 ms | 3.77 ms | +105% | -51% |

---

## Optimization Impact by Use Case

### Best For:
- **High-throughput gas metering** - 2.3x faster
- **Speculative execution with rollback** - 4x faster checkpoint revert
- **Multi-space operations** - 2.4x faster multi-channel produce
- **Concurrent access (4+ threads)** - 20-40% improvement

### Trade-off For:
- **Sequential single-channel consume** - 2x slower
- **Low-thread contention (2 threads)** - 40% slower

### Neutral:
- **Registry lookups** - Similar performance
- **Pattern matching** - Similar performance

---

## Next Steps (Phase 4)

1. Investigate consume_immediate regression
2. Add hybrid Vec/im::Vector for collection size-based switching
3. Implement Rayon parallel matching for large continuations
4. Profile and optimize large checkpoint revert

---

## Files

- Baseline: `docs/performance/baseline-2026-01-09.md`
- Post-Optimization: `docs/performance/post-optimization-2026-01-09.md`
- This Comparison: `docs/performance/comparison-2026-01-09.md`
