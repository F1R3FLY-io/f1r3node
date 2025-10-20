# Design Document: Heartbeat Block Proposer

## Overview

Add a heartbeat mechanism to ensure liveness of the blockchain by proposing empty blocks when no user activity occurs. This ensures the DAG continues to advance and blocks can be finalized even during periods of zero transaction traffic.

## Goals

1. **Ensure Liveness**: Propose blocks periodically to keep the DAG advancing and enable finalization
2. **Remove Empty Block Restriction**: Allow blocks to be created with no user deploys (only system deploys)
3. **Safe Integration**: Leverage existing synchronization mechanisms to avoid race conditions
4. **Configurable**: Allow operators to control heartbeat behavior

## Components

### 1. Remove Empty Block Check (BlockCreator.scala)

**Current Code** (BlockCreator.scala:117-162):
```scala
r <- if (deploys.nonEmpty || slashingDeploys.nonEmpty)
      for {
        // ... create block
      } yield BlockCreatorResult.created(signedBlock)
    else
      BlockCreatorResult.noNewDeploys.pure[F]
```

**New Code**:
```scala
// Always create blocks - they will always have system deploys (CloseBlockDeploy)
r <- for {
  // ... create block
} yield BlockCreatorResult.created(signedBlock)
```

**Rationale**:
- Every block already contains `CloseBlockDeploy` system deploy (line 112-115)
- Empty blocks are valid - they advance the DAG and record validator agreement via justifications
- The validation code has no checks requiring non-empty user deploys
- This is purely a policy decision, not a fundamental constraint

### 2. Add Heartbeat Configuration (shared-rnode.conf)

```hocon
casper {
  heartbeat {
    # Enable heartbeat block proposing for liveness
    enabled = true

    # Check interval - how often to check if heartbeat is needed
    check-interval = 30 seconds

    # Maximum age of last finalized block before triggering heartbeat
    # If no block has been finalized in this duration, propose an empty block
    max-lfb-age = 60 seconds
  }
}
```

**Configuration Parameters**:
- `enabled`: Master switch for heartbeat functionality
- `check-interval`: How often the heartbeat thread wakes up to check
- `max-lfb-age`: If LFB is older than this, trigger a propose

### 3. Heartbeat Proposer Implementation

**File**: `node/src/main/scala/coop/rchain/node/instances/HeartbeatProposer.scala`

```scala
package coop.rchain.node.instances

import cats.effect.{Concurrent, Timer}
import cats.syntax.all._
import coop.rchain.casper.{Casper, ProposeFunction}
import coop.rchain.shared.{Log, Time}
import fs2.Stream
import scala.concurrent.duration.FiniteDuration

final case class HeartbeatConfig(
    enabled: Boolean,
    checkInterval: FiniteDuration,
    maxLfbAge: FiniteDuration
)

object HeartbeatProposer {

  /**
   * Create a heartbeat proposer stream that periodically checks if a block
   * needs to be proposed to maintain liveness.
   *
   * @param triggerPropose The propose function from Setup.scala
   * @param casper The Casper instance
   * @param config Heartbeat configuration
   * @return A stream that runs the heartbeat loop
   */
  def create[F[_]: Concurrent: Timer: Time: Log](
      triggerPropose: ProposeFunction[F],
      casper: Casper[F],
      config: HeartbeatConfig
  ): Stream[F, Unit] = {

    if (!config.enabled) {
      Stream.empty
    } else {
      Stream
        .awakeEvery[F](config.checkInterval)
        .evalMap(_ => checkAndMaybePropose(triggerPropose, casper, config))
    }
  }

  private def checkAndMaybePropose[F[_]: Concurrent: Time: Log](
      triggerPropose: ProposeFunction[F],
      casper: Casper[F],
      config: HeartbeatConfig
  ): F[Unit] = {
    for {
      // Get current DAG state
      dag <- casper.blockDag
      lfbHash = dag.lastFinalizedBlock
      lfbMeta <- dag.lookup(lfbHash)

      // Check if LFB is stale
      now <- Time[F].currentMillis
      timeSinceLFB = lfbMeta.map(meta => now - meta.timestamp).getOrElse(0L)
      shouldPropose = timeSinceLFB > config.maxLfbAge.toMillis

      _ <- if (shouldPropose) {
             for {
               _ <- Log[F].info(
                     s"Heartbeat: LFB is ${timeSinceLFB}ms old (threshold: ${config.maxLfbAge.toMillis}ms), triggering propose"
                   )
               // Trigger propose - this goes through the same queue as user proposes
               result <- triggerPropose(casper, false)
               _ <- result match {
                     case ProposerEmpty =>
                       Log[F].debug("Heartbeat: Propose already in progress")
                     case ProposerFailure(status, seqNum) =>
                       Log[F].warn(s"Heartbeat: Propose failed with $status (seqNum $seqNum)")
                     case ProposerSuccess(_, block) =>
                       Log[F].info(s"Heartbeat: Successfully created block ${block.blockHash.toHexString}")
                     case ProposerStarted(seqNum) =>
                       Log[F].info(s"Heartbeat: Async propose started (seqNum $seqNum)")
                   }
             } yield ()
           } else {
             Log[F].debug(s"Heartbeat: LFB age is ${timeSinceLFB}ms, no action needed")
           }
    } yield ()
  }
}
```

### 4. Integration into Node Runtime

**File**: `node/src/main/scala/coop/rchain/node/runtime/Setup.scala`

Add heartbeat configuration loading after line 254:

```scala
// After proposerStateRefOpt initialization
heartbeatConfig = HeartbeatConfig(
  enabled = conf.casper.heartbeat.enabled,
  checkInterval = conf.casper.heartbeat.checkInterval,
  maxLfbAge = conf.casper.heartbeat.maxLfbAge
)
```

**File**: `node/src/main/scala/coop/rchain/node/runtime/NodeRuntime.scala`

Add heartbeat stream after line 367:

```scala
heartbeatStream = if (proposer.isDefined && heartbeatConfig.enabled)
  HeartbeatProposer.create[F](
    triggerProposeFOpt.get,
    // Need to get Casper instance - this requires reading from EngineCell
    // See implementation note below
    heartbeatConfig
  )
else
  fs2.Stream.empty
```

**Implementation Note**: Getting the Casper instance for heartbeat requires reading from EngineCell. Two approaches:

**Option A**: Pass EngineCell to heartbeat and read it each iteration:
```scala
private def checkAndMaybePropose[F[_]: Concurrent: Time: Log: EngineCell](
    triggerPropose: ProposeFunction[F],
    config: HeartbeatConfig
): F[Unit] = {
  EngineCell[F].read >>= {
    _.withCasper(
      casper => doHeartbeatCheck(triggerPropose, casper, config),
      Log[F].debug("Heartbeat: Casper not available yet")
    )
  }
}
```

**Option B**: Wait for Casper to be available once at startup, then use it:
```scala
// In NodeRuntime
heartbeatStream = if (proposer.isDefined && heartbeatConfig.enabled)
  Stream.eval(waitForCasper) // Wait for Casper to be initialized
    .flatMap(casper => HeartbeatProposer.create[F](triggerProposeFOpt.get, casper, heartbeatConfig))
else
  fs2.Stream.empty
```

**Recommendation**: Use Option A for robustness - handle the case where Casper might be temporarily unavailable.

## Synchronization Design

### Architecture Overview

```
┌─────────────────┐
│ User Deploy API │──┐
└─────────────────┘  │
                     │
┌─────────────────┐  │    ┌──────────────────┐
│ Propose API     │──┼───▶│ proposerQueue    │
└─────────────────┘  │    │ (Unbounded)      │
                     │    └────────┬─────────┘
┌─────────────────┐  │             │
│ Block Processor │──┘             │ dequeue
└─────────────────┘                ▼
                            ┌──────────────────┐
┌─────────────────┐         │ ProposerInstance │
│ Heartbeat       │────────▶│                  │
└─────────────────┘         │ Semaphore(1)     │
                            │ ┌──────────────┐ │
                            │ │ Only ONE     │ │
                            │ │ propose      │ │
                            │ │ executes     │ │
                            │ │ at a time    │ │
                            │ └──────────────┘ │
                            └──────────────────┘
```

### Why This Approach is Safe

#### 1. Single Queue Serializes All Requests

**All four sources** of propose requests go through the same `proposerQueue`:
- User Deploy API (with auto-propose enabled)
- Explicit Propose API
- Block Processing (with auto-propose enabled)
- **NEW: Heartbeat Proposer**

The queue is created once at startup (Setup.scala:241):
```scala
proposerQueue <- Queue.unbounded[F, (Casper[F], Boolean, Deferred[F, ProposerResult])]
```

All callers use the same `triggerProposeFOpt` function which enqueues to this queue:
```scala
triggerProposeFOpt: Option[ProposeFunction[F]] =
  Some((casper: Casper[F], isAsync: Boolean) =>
    for {
      d <- Deferred[F, ProposerResult]
      _ <- proposerQueue.enqueue1((casper, isAsync, d))  // Same queue!
      r <- d.get
    } yield r
  )
```

**Safety Property**: The queue guarantees FIFO ordering and thread-safe enqueue/dequeue operations.

#### 2. Semaphore Ensures Only One Actual Propose

ProposerInstance creates a Semaphore with 1 permit (ProposerInstance.scala:30):
```scala
lock <- Semaphore[F](1)
```

For each dequeued request, it tries to acquire the lock (line 41):
```scala
Stream.eval(lock.tryAcquire)
```

**If lock is available** (no propose in progress):
- `tryAcquire` returns `true`
- Proceeds to execute `proposer.propose()` (line 59)
- Inside `proposer.propose()`:
  - Gets CasperSnapshot (with current maxSeqNums)
  - Creates block with next seqNum
  - Signs and validates block
  - Inserts into DAG
- Releases lock (line 66)

**If lock is held** (propose already in progress):
- `tryAcquire` returns `false`
- Immediately completes the Deferred with `ProposerEmpty` (line 45)
- Does NOT call `proposer.propose()`
- The calling thread gets `ProposerEmpty` result

**Safety Property**: The Semaphore guarantees that only ONE thread can be inside `proposer.propose()` at any time, preventing the race condition where two threads get the same snapshot and create blocks with duplicate seqNums.

#### 3. getCasperSnapshot is Safe Under Serialization

The race condition we were concerned about:
```
Thread A: getCasperSnapshot() → maxSeqNum = 10
Thread B: getCasperSnapshot() → maxSeqNum = 10
Thread A: Create block with seqNum = 11
Thread B: Create block with seqNum = 11  ← EQUIVOCATION!
```

**This CANNOT happen because**:
- Semaphore ensures only Thread A OR Thread B can be inside `proposer.propose()`
- If Thread A enters first:
  - Gets snapshot with maxSeqNum = 10
  - Creates block with seqNum = 11
  - Inserts into DAG (updates maxSeqNum to 11)
  - Releases lock
- When Thread B enters (after Thread A finishes):
  - Gets snapshot with maxSeqNum = 11 (updated by Thread A)
  - Creates block with seqNum = 12
  - No collision!

**Safety Property**: Serial execution of `proposer.propose()` means each propose sees the effects of all previous proposes.

#### 4. Heartbeat Integration Safety

The heartbeat simply calls the same `triggerPropose` function:
```scala
result <- triggerPropose(casper, false)
```

This:
1. Creates a Deferred
2. Enqueues to the same `proposerQueue`
3. Blocks waiting for result

**The heartbeat has no special privileges** - it waits in line like everyone else:
- If a user deploy is being processed, heartbeat waits
- If an explicit propose is running, heartbeat gets `ProposerEmpty`
- If heartbeat is running, user requests wait their turn

**Safety Property**: The heartbeat participates in the same serialization mechanism as all other callers, with no additional synchronization needed.

#### 5. Trigger Mechanism Prevents Work Loss

If multiple requests arrive while a propose is in progress (ProposerInstance.scala:44-47):
```scala
.evalFilter { v =>
  (trigger.tryPut(()) >> proposeIDDef.complete(ProposerResult.empty))
    .unlessA(v)
    .as(v)
}
```

- Failed lock attempts "cock the trigger" (put a value in the MVar)
- Return `ProposerEmpty` to the caller
- After the current propose finishes (lines 78-84):
  ```scala
  _ <- trigger.tryTake.flatMap {
    case Some(_) =>  // Someone tried to propose while we were busy
      Deferred[F, ProposerResult] >>= { d =>
        proposeRequestsQueue.enqueue1(c, false, d)  // Re-enqueue!
      }
    case None => ().pure[F]
  }
  ```

**Safety Property**: If work arrives during a propose, it doesn't get lost - one additional propose is automatically triggered.

#### 6. No Additional Locking Required

The heartbeat implementation requires **zero additional synchronization primitives**:
- No new Semaphores
- No new Refs
- No new MVars
- No locks of any kind

It simply calls an existing function (`triggerPropose`) that already has proper synchronization.

**Safety Property**: Reusing existing, proven synchronization mechanisms reduces the risk of deadlocks, race conditions, or other concurrency bugs.

## Testing Strategy

### Unit Tests

1. **Test Empty Block Creation**
   - Verify blocks can be created with no user deploys
   - Verify CloseBlockDeploy is always present
   - Verify empty blocks validate correctly

2. **Test Heartbeat Timing**
   - Mock time to advance LFB age
   - Verify heartbeat triggers propose when threshold exceeded
   - Verify heartbeat doesn't trigger when LFB is fresh

3. **Test Heartbeat Synchronization**
   - Trigger heartbeat while user propose is in progress
   - Verify heartbeat gets ProposerEmpty
   - Verify no blocks with duplicate seqNums are created

### Integration Tests

1. **Zero Traffic Scenario**
   - Start network with no user deploys
   - Verify blocks are proposed every max-lfb-age interval
   - Verify finalization continues to advance

2. **Mixed Traffic Scenario**
   - Generate sporadic user deploys
   - Verify heartbeat fills gaps when needed
   - Verify heartbeat doesn't interfere with normal proposes

3. **Concurrent Proposes**
   - Trigger user deploy, explicit propose, and heartbeat simultaneously
   - Verify only one block created per validator
   - Verify all seqNums are sequential with no gaps or duplicates

## Deployment Strategy

### Phase 1: Remove Empty Block Check (Low Risk)
1. Modify BlockCreator.scala to remove the empty check
2. Deploy to testnet
3. Verify empty blocks can be created via explicit propose API
4. Monitor for any validation issues

### Phase 2: Add Heartbeat (Medium Risk)
1. Add heartbeat configuration with `enabled = false` by default
2. Add HeartbeatProposer implementation
3. Integrate into NodeRuntime
4. Deploy to testnet with heartbeat disabled
5. Enable heartbeat on single validator, monitor for issues
6. Gradually enable on more validators

### Phase 3: Production Rollout (Low Risk)
1. Deploy to production with heartbeat disabled
2. Enable on validators with low stake first
3. Monitor metrics: heartbeat triggers, propose failures, seqNum gaps
4. Enable on all validators once proven stable

## Metrics and Monitoring

Add the following metrics:

```scala
// In HeartbeatProposer
Metrics.counter("heartbeat.checks.total")         // Total heartbeat checks
Metrics.counter("heartbeat.proposes.triggered")   // How many times heartbeat triggered propose
Metrics.counter("heartbeat.proposes.empty")       // How many times got ProposerEmpty
Metrics.histogram("heartbeat.lfb.age.ms")         // Distribution of LFB ages
```

Alert on:
- `heartbeat.proposes.empty` high rate (may indicate propose is stuck)
- `heartbeat.lfb.age.ms` exceeding 2x threshold (may indicate heartbeat not working)

## Risks and Mitigations

### Risk 1: Heartbeat Creates Too Many Empty Blocks
**Impact**: Blockchain bloat, storage waste
**Probability**: Medium
**Mitigation**:
- Configure reasonable max-lfb-age (e.g., 60 seconds)
- Monitor block size and empty block rate
- Adjust configuration based on metrics

### Risk 2: Heartbeat Interferes with Normal Proposes
**Impact**: User transactions delayed
**Probability**: Low
**Mitigation**:
- Heartbeat uses same queue, waits its turn
- If normal traffic resumes, LFB advances naturally and heartbeat stops triggering
- Heartbeat only triggers when LFB is truly stale

### Risk 3: Configuration Error Causes Propose Storm
**Impact**: Network overload, resources exhausted
**Probability**: Low
**Mitigation**:
- Semaphore ensures only one propose at a time regardless of triggers
- Add rate limiting: track heartbeat triggers per minute, alert if excessive
- Minimum check-interval enforced (e.g., >= 10 seconds)

## Conclusion

This design leverages the existing, well-tested ProposerInstance serialization mechanism to safely add heartbeat functionality. By removing the empty block restriction and adding a simple timer-based proposer, we ensure blockchain liveness during periods of zero traffic without introducing race conditions or requiring complex new synchronization primitives.

The key insight is that **all propose sources flow through the same queue and semaphore**, making the addition of a heartbeat source trivial from a synchronization perspective. The existing architecture was already designed to handle multiple concurrent propose sources safely.
