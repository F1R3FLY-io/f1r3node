# Trait-Based Logging Architecture Proposal for PR #84

### Current Issue

**Scala (Working):**
```scala
class ReportingRspace extends ReplayRSpace {
  override def logComm(...) = {
    val result = super.logComm(...)  // Call parent
    // Add reporting logic
    result
  }
}
```
When `ReplayRSpace.lockedConsume()` calls `self.logComm()`, it polymorphically dispatches to the overridden `ReportingRspace.logComm()`.

**Rust (Broken):**
```rust
pub struct ReportingRspace<C, P, A, K> {
    replay_rspace: ReplayRSpace<C, P, A, K>,  // Composition
    // ...
}

impl<C, P, A, K> ReportingRspace<C, P, A, K> {
    fn log_comm(&mut self, ...) -> COMM {
        // This method exists but is NEVER CALLED!
    }
}
```
When `ReplayRSpace.locked_consume()` calls `self.log_comm()`, it calls `ReplayRSpace.log_comm()` directly, never reaching the reporting logic.

## Proposed Solution: Generic Trait-Based Architecture

This solution replicates Scala's polymorphic dispatch using Rust's trait system, providing the same semantic behavior with zero runtime cost.

### Core Design

#### 1. Logging Trait Definition

```rust
/// Core logging operations that can be overridden
pub trait RSpaceLogger<C, P, A, K> {
    fn log_comm(
        &mut self,
        data_candidates: &[ConsumeCandidate<C, A>],
        channels: &[C],
        wk: WaitingContinuation<P, K>,
        comm: COMM,
        label: &str,
    ) -> COMM;

    fn log_consume(
        &mut self,
        consume_ref: Consume,
        channels: &[C],
        patterns: &[P],
        continuation: &K,
        persist: bool,
        peeks: &BTreeSet<i32>,
    ) -> Consume;

    fn log_produce(
        &mut self,
        produce_ref: Produce,
        channel: &C,
        data: &A,
        persist: bool,
    ) -> Produce;
}
```

#### 2. Generic ReplayRSpace

```rust
/// ReplayRSpace becomes generic over the logger implementation
pub struct ReplayRSpace<C, P, A, K, L> 
where
    L: RSpaceLogger<C, P, A, K>,
{
    pub history_repository: Arc<Box<dyn HistoryRepository<C, P, A, K>>>,
    pub store: Arc<Box<dyn HotStore<C, P, A, K>>>,
    installs: Arc<Mutex<HashMap<Vec<C>, Install<P, K>>>>,
    event_log: Log,
    produce_counter: BTreeMap<Produce, i32>,
    matcher: Arc<Box<dyn Match<P, A>>>,
    pub replay_data: MultisetMultiMap<IOEvent, COMM>,
    
    // Logger dependency injection - this enables polymorphism
    logger: L,
}

impl<C, P, A, K, L> ReplayRSpace<C, P, A, K, L>
where
    L: RSpaceLogger<C, P, A, K>,
{
    /// Critical change: calls logger.log_comm() instead of self.log_comm()
    fn locked_consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
        consume_ref: Consume,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError> {
        // ... existing matching logic ...
        
        match get_comm_and_consume_candidates {
            Some((_, data_candidates)) => {
                // 🎯 KEY CHANGE: Polymorphic dispatch through trait
                let comm_ref = self.logger.log_comm(
                    &data_candidates,
                    &channels,
                    wk.clone(),
                    COMM::new(/*...*/),
                    CONSUME_COMM_LABEL,
                );
                // ... rest of method unchanged
            }
        }
    }
}
```

#### 3. Basic Logger Implementation

```rust
/// Default logger that replicates current ReplayRSpace behavior
pub struct BasicLogger;

impl<C, P, A, K> RSpaceLogger<C, P, A, K> for BasicLogger {
    fn log_comm(
        &mut self,
        _data_candidates: &[ConsumeCandidate<C, A>],
        _channels: &[C],
        _wk: WaitingContinuation<P, K>,
        comm: COMM,
        _label: &str,
    ) -> COMM {
        // Current ReplayRSpace logic - just return the COMM
        // TODO: Add metrics
        comm
    }

    fn log_consume(&mut self, consume_ref: Consume, ...) -> Consume {
        consume_ref
    }

    fn log_produce(&mut self, produce_ref: Produce, ...) -> Produce {
        // TODO: Handle produce counter logic
        produce_ref
    }
}
```

#### 4. Reporting Logger Implementation

```rust
/// Logger that adds reporting functionality (equivalent to Scala ReportingRspace)
pub struct ReportingLogger<C, P, A, K> {
    base_logger: BasicLogger,
    report: Arc<Mutex<Vec<Vec<ReportingEvent<C, P, A, K>>>>>,
    soft_report: Arc<Mutex<Vec<ReportingEvent<C, P, A, K>>>>,
}

impl<C, P, A, K> RSpaceLogger<C, P, A, K> for ReportingLogger<C, P, A, K>
where
    C: Clone + Debug,
    P: Clone + Debug,
    A: Clone + Debug,
    K: Clone + Debug,
{
    fn log_comm(
        &mut self,
        data_candidates: &[ConsumeCandidate<C, A>],
        channels: &[C],
        wk: WaitingContinuation<P, K>,
        comm: COMM,
        label: &str,
    ) -> COMM {
        // First call base implementation (equivalent to super.logComm())
        let comm_ref = self.base_logger.log_comm(
            data_candidates, channels, wk.clone(), comm, label
        );

        // Then add reporting logic (equivalent to ReportingRspace override)
        let reporting_consume = ReportingConsume {
            channels: channels.to_vec(),
            patterns: wk.patterns,
            continuation: wk.continuation,
            peeks: wk.peeks.into_iter().collect(),
        };

        let reporting_produces = data_candidates
            .iter()
            .map(|dc| ReportingProduce {
                channel: dc.channel.clone(),
                data: dc.datum.a.clone(),
            })
            .collect();

        let reporting_comm = ReportingEvent::ReportingComm(ReportingComm {
            consume: reporting_consume,
            produces: reporting_produces,
        });

        if let Ok(mut soft_report_guard) = self.soft_report.lock() {
            soft_report_guard.push(reporting_comm);
        }

        comm_ref
    }

    // Similar implementations for log_consume and log_produce...
}
```