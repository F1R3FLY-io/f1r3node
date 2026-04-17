# Rholang Evaluation Order and Consensus Determinism

> Last updated: 2026-04-14

## Rholang Is Non-Deterministic By Design

Rholang parallel composition (`A | B`) means A and B can execute in any order,
and different orderings may produce different observable results. This is a
feature — it models real concurrent systems where processes execute independently
without a global clock.

For example, `@x!(1) | @x!(2) | for(@v <- x) { ... }` can match either
`1` or `2` to the consumer. Both are valid executions.

## Blockchain Consensus Requires Reproducibility

When a proposer evaluates a deploy, it produces a specific state hash, cost,
and event log. Every validator must replay that deploy and produce the EXACT
same result. The non-determinism is resolved at the implementation level:

1. **Evaluation order**: The `eval(Par)` term vector ordering (sends first,
   receives second) determines which `tokio::spawn` tasks are created first
2. **Candidate matching**: `Random.shuffle` with a deploy-derived seed selects
   the same candidate deterministically
3. **Replay oracle**: `ReplayRSpace` uses the play event log to force the
   exact same COMMs during replay, regardless of evaluation order

## Implementation: tokio::spawn with FIFO Per-Channel Locks

The Rust node uses `tokio::spawn` for parallel Par branch evaluation, matching
Scala's `parTraverse`. Per-channel locks use `tokio::sync::Mutex` which
guarantees FIFO ordering for waiters, matching Scala's cats-effect `Semaphore`.

Both `RSpace` and `ReplayRSpace` have per-channel two-phase locks, matching
Scala's `RSpaceOps` which provides locks to both via inheritance.

## Scala Empirical Analysis

Instrumented logging on the Scala node (`RSpaceOps.scala`, `RSpace.scala`)
confirmed across 4 independent runs:

- Both produces ALWAYS store before the consume fires COMM for the join
  contract `@0!!(0) | @1!!(1) | for (_ <- @0 & _ <- @1) { 0 }`
- The sends-first term vector ordering combined with Monix ForkJoinPool
  task registration produces consistent operation ordering
- The pattern holds across 20+ parallel repetitions per run on threads 27-44

## Known Limitations

- **RCHAIN-3917**: Scala acknowledges non-deterministic cost for some contracts
  involving persistent operations with same-channel joins. The Scala test for
  randomly generated contracts is `ignore`d.
- **RCHAIN-4032**: `SpaceMatcher.extractDataCandidates` causes unmatched COMMs
  with overlapping join patterns. Joins like `for(<- x & <- x)` are explicitly
  blocked with `ReceiveOnSameChannelsError`.

## Cost Accounting (ChargingRSpace)

The `ChargingRSpace` wrapper uses order-independent charging:

1. Pre-charge storage cost before each `produce`/`consume`
2. When COMM fires: refund all pre-charges, charge unified COMM cost, re-charge
   persistent items
3. When no COMM: no additional event storage charge (removed for order-independence)

Persistent operation re-issues (from `continue_consume_process` /
`continue_produce_process`) are pre-credited in the reducer to make the net
cost of re-installation zero.

## References

- Scala `Reduce.scala` line 256-289: term vector construction
- Scala `RSpaceOps.scala` lines 107-158: per-channel Semaphore locks
- Scala `CostAccountingSpec.scala` line 303-317: determinism test with `.par`
- `cost-accounted-rho.pdf`: Future architecture for in-language cost accounting
