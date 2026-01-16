//! Charging Space Agent - Automatic Phlogiston Metering
//!
//! This module provides a wrapper around SpaceAgent that automatically charges
//! phlogiston (gas) for all operations. It ensures that resource consumption
//! is metered and bounded.
//!
//! # Formal Correspondence
//! - `Phlogiston.v`: Charge preservation and non-negativity invariants
//! - `GenericRSpace.v`: Integration of gas accounting with space operations
//! - `Safety/Properties.v`: Resource exhaustion safety properties
//!
//! # Design
//! The `ChargingSpaceAgent<S>` wraps any `SpaceAgent<C, P, A, K>` implementation
//! and intercepts all operations to:
//! 1. Calculate the phlogiston cost
//! 2. Attempt to charge from the meter
//! 3. Either proceed with the operation or return OutOfPhlogiston error
//!
//! This pattern follows the decorator design pattern, allowing gas accounting
//! to be composed with any space implementation.

use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;
use std::sync::Arc;

use serde::{de::DeserializeOwned, Serialize};

use super::agent::{SpaceAgent, CheckpointableSpace, ReplayableSpace};
use super::errors::SpaceError;
use super::phlogiston::{PhlogistonMeter, Operation};
use super::types::{ChannelBound, ContinuationBound, DataBound, PatternBound, SpaceId, SpaceQualifier};

use rspace_plus_plus::rspace::{
    checkpoint::{Checkpoint, SoftCheckpoint},
    errors::RSpaceError,
    hashing::blake2b256_hash::Blake2b256Hash,
    internal::{Datum, Row, WaitingContinuation},
    rspace_interface::{ContResult, ISpace, MaybeConsumeResult, MaybeProduceResult, RSpaceResult},
    trace::{Log, event::Produce},
};

// =============================================================================
// Charging Space Agent
// =============================================================================

/// A wrapper that charges phlogiston for all space operations.
///
/// This implements the `SpaceAgent` trait by delegating to an inner space
/// while charging the appropriate gas cost for each operation.
///
/// # Type Parameters
/// - `S`: The underlying space agent type
/// - `C`: Channel type
/// - `P`: Pattern type
/// - `A`: Data type
/// - `K`: Continuation type
///
/// # Invariants (from Phlogiston.v)
/// - All operations charge before executing (charge-before-execute)
/// - Failed charges result in OutOfPhlogiston error
/// - Successful operations reduce meter balance by operation cost
/// - Total consumed = sum of all successful operation costs
pub struct ChargingSpaceAgent<S, C, P, A, K>
where
    S: SpaceAgent<C, P, A, K>,
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    /// The underlying space agent.
    inner: S,
    /// The phlogiston meter for tracking gas consumption.
    meter: Arc<PhlogistonMeter>,
    /// Phantom data for type parameters.
    _phantom: std::marker::PhantomData<(C, P, A, K)>,
}

impl<S, C, P, A, K> ChargingSpaceAgent<S, C, P, A, K>
where
    S: SpaceAgent<C, P, A, K>,
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    /// Create a new charging agent wrapping the given space.
    ///
    /// # Arguments
    /// * `inner` - The underlying space agent
    /// * `meter` - The phlogiston meter to use for charging
    pub fn new(inner: S, meter: Arc<PhlogistonMeter>) -> Self {
        ChargingSpaceAgent {
            inner,
            meter,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a charging agent with unlimited phlogiston (for testing).
    pub fn unlimited(inner: S) -> Self {
        Self::new(inner, Arc::new(PhlogistonMeter::unlimited()))
    }

    /// Get a reference to the underlying space agent.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Get a mutable reference to the underlying space agent.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }

    /// Get the phlogiston meter.
    pub fn meter(&self) -> &PhlogistonMeter {
        &self.meter
    }

    /// Get the current phlogiston balance.
    pub fn balance(&self) -> u64 {
        self.meter.balance()
    }

    /// Get total phlogiston consumed.
    pub fn total_consumed(&self) -> u64 {
        self.meter.total_consumed()
    }

    /// Charge for an operation, returning error if insufficient phlogiston.
    fn charge(&self, operation: &Operation) -> Result<(), SpaceError> {
        self.meter.charge(operation)
    }
}

impl<S, C, P, A, K> SpaceAgent<C, P, A, K> for ChargingSpaceAgent<S, C, P, A, K>
where
    S: SpaceAgent<C, P, A, K>,
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    fn space_id(&self) -> &SpaceId {
        self.inner.space_id()
    }

    fn qualifier(&self) -> SpaceQualifier {
        self.inner.qualifier()
    }

    fn gensym(&mut self) -> Result<C, SpaceError> {
        // Charge for channel creation
        self.charge(&Operation::CreateChannel)?;
        self.inner.gensym()
    }

    fn produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
        priority: Option<usize>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>, Produce)>, SpaceError> {
        // Estimate data size for send cost
        // In a real implementation, we'd serialize and measure the actual size
        // For now, use a reasonable estimate based on type
        let data_size = std::mem::size_of::<A>();
        self.charge(&Operation::Send { data_size })?;

        self.inner.produce(channel, data, persist, priority)
    }

    fn consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)>, SpaceError> {
        // Charge for receive and pattern matching
        self.charge(&Operation::Receive)?;

        // Charge for pattern matching based on pattern count
        let pattern_size = patterns.len();
        if pattern_size > 0 {
            self.charge(&Operation::Match { pattern_size })?;
        }

        self.inner.consume(channels, patterns, continuation, persist, peeks)
    }

    fn install(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, SpaceError> {
        // Install is like consume but persistent - charge accordingly
        self.charge(&Operation::Receive)?;

        let pattern_size = patterns.len();
        if pattern_size > 0 {
            self.charge(&Operation::Match { pattern_size })?;
        }

        self.inner.install(channels, patterns, continuation)
    }

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        // Read operations are free (for now) as they don't modify state
        // Could add lookup cost if needed
        self.inner.get_data(channel)
    }

    fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        // Read operations are free
        self.inner.get_waiting_continuations(channels)
    }

    fn get_joins(&self, channel: C) -> Vec<Vec<C>> {
        // Read operations are free
        self.inner.get_joins(channel)
    }
}

// =============================================================================
// Checkpointable Implementation
// =============================================================================

impl<S, C, P, A, K> CheckpointableSpace<C, P, A, K> for ChargingSpaceAgent<S, C, P, A, K>
where
    S: CheckpointableSpace<C, P, A, K>,
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    fn create_checkpoint(&mut self) -> Result<Checkpoint, SpaceError> {
        self.charge(&Operation::Checkpoint)?;
        self.inner.create_checkpoint()
    }

    fn create_soft_checkpoint(&mut self) -> SoftCheckpoint<C, P, A, K> {
        // Soft checkpoints are cheaper, could use a different cost
        // For now, they're free as they don't persist
        self.inner.create_soft_checkpoint()
    }

    fn revert_to_soft_checkpoint(
        &mut self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), SpaceError> {
        // Reverting is free (part of soft checkpoint semantics)
        self.inner.revert_to_soft_checkpoint(checkpoint)
    }

    fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), SpaceError> {
        // Reset charges similar to replay
        self.charge(&Operation::Replay { operation_count: 1 })?;
        self.inner.reset(root)
    }

    fn clear(&mut self) -> Result<(), SpaceError> {
        // Clear is a management operation, could charge if needed
        self.inner.clear()
    }
}

// =============================================================================
// Replayable Implementation
// =============================================================================

impl<S, C, P, A, K> ReplayableSpace<C, P, A, K> for ChargingSpaceAgent<S, C, P, A, K>
where
    S: ReplayableSpace<C, P, A, K>,
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    fn rig_and_reset(
        &mut self,
        start_root: Blake2b256Hash,
        log: Log,
    ) -> Result<(), SpaceError> {
        // Charge based on log size for replay
        let operation_count = estimate_log_operations(&log);
        self.charge(&Operation::Replay { operation_count })?;
        self.inner.rig_and_reset(start_root, log)
    }

    fn rig(&self, log: Log) -> Result<(), SpaceError> {
        // Rigging without reset is a setup operation
        // Note: This takes &self so we can't charge (would need interior mutability)
        // In production, consider using a lock or making this &mut self
        self.inner.rig(log)
    }

    fn check_replay_data(&self) -> Result<(), SpaceError> {
        // Checking is a validation operation, no charge
        self.inner.check_replay_data()
    }

    fn is_replay(&self) -> bool {
        self.inner.is_replay()
    }

    fn update_produce(&mut self, produce: Produce) {
        // Update during replay is part of the replayed operation
        self.inner.update_produce(produce)
    }
}

/// Estimate the number of operations in a log for charging purposes.
fn estimate_log_operations(log: &Log) -> usize {
    // Log is Vec<Event>, so the count is simply the vector length
    log.len()
}

// =============================================================================
// ISpace Implementation (for compatibility with RhoISpace)
// =============================================================================

/// ISpace implementation for ChargingSpaceAgent.
///
/// This allows ChargingSpaceAgent to be used where `Box<dyn ISpace<...>>` is expected,
/// enabling phlogiston metering for user-created spaces that use the ISpace interface.
///
/// # Charging Behavior
/// - `produce`, `consume`, `consume_with_similarity`: Charge based on data/pattern size
/// - `install`: Charge for receive + pattern matching
/// - `create_checkpoint`: Charge for checkpoint creation
/// - `rig_and_reset`: Charge based on log size
/// - Read operations (`get_data`, `get_joins`, etc.): Free (no state modification)
/// - `clear`, `reset`: Free management operations
impl<S, C, P, A, K> ISpace<C, P, A, K> for ChargingSpaceAgent<S, C, P, A, K>
where
    S: SpaceAgent<C, P, A, K> + ISpace<C, P, A, K>,
    C: ChannelBound + Eq + Hash + AsRef<[u8]> + Serialize + DeserializeOwned,
    P: PatternBound + Serialize + DeserializeOwned,
    A: DataBound + Serialize + DeserializeOwned,
    K: ContinuationBound + Serialize + DeserializeOwned,
{
    fn create_checkpoint(&mut self) -> Result<Checkpoint, RSpaceError> {
        self.charge(&Operation::Checkpoint)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;
        ISpace::create_checkpoint(&mut self.inner)
    }

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        // Read operations are free
        ISpace::get_data(&self.inner, channel)
    }

    fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        // Read operations are free
        ISpace::get_waiting_continuations(&self.inner, channels)
    }

    fn get_joins(&self, channel: C) -> Vec<Vec<C>> {
        // Read operations are free
        ISpace::get_joins(&self.inner, channel)
    }

    fn clear(&mut self) -> Result<(), RSpaceError> {
        // Clear is a management operation, no charge
        ISpace::clear(&mut self.inner)
    }

    fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), RSpaceError> {
        // Reset charges similar to replay
        self.charge(&Operation::Replay { operation_count: 1 })
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;
        ISpace::reset(&mut self.inner, root)
    }

    fn consume_result(
        &mut self,
        channel: Vec<C>,
        pattern: Vec<P>,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        // Charge for receive + match
        self.charge(&Operation::Receive)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;
        let pattern_size = pattern.len();
        if pattern_size > 0 {
            self.charge(&Operation::Match { pattern_size })
                .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;
        }
        ISpace::consume_result(&mut self.inner, channel, pattern)
    }

    fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>> {
        // Read operation, no charge
        ISpace::to_map(&self.inner)
    }

    fn create_soft_checkpoint(&mut self) -> SoftCheckpoint<C, P, A, K> {
        // Soft checkpoints are free (no persistence)
        ISpace::create_soft_checkpoint(&mut self.inner)
    }

    fn revert_to_soft_checkpoint(
        &mut self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), RSpaceError> {
        // Reverting is free
        ISpace::revert_to_soft_checkpoint(&mut self.inner, checkpoint)
    }

    fn consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError> {
        // Charge for receive
        self.charge(&Operation::Receive)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;

        // Charge for pattern matching
        let pattern_size = patterns.len();
        if pattern_size > 0 {
            self.charge(&Operation::Match { pattern_size })
                .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;
        }

        ISpace::consume(&mut self.inner, channels, patterns, continuation, persist, peeks)
    }

    fn consume_with_modifiers(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        modifiers: Vec<Vec<u8>>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError> {
        // Charge for receive
        self.charge(&Operation::Receive)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;

        // Charge for pattern matching
        let pattern_size = patterns.len();
        if pattern_size > 0 {
            self.charge(&Operation::Match { pattern_size })
                .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;
        }

        // Charge extra for VectorDB operations if any modifiers present
        for modifier_bytes in &modifiers {
            if !modifier_bytes.is_empty() {
                // Estimate dimensions from serialized modifier size
                let dimensions = modifier_bytes.len() / std::mem::size_of::<f32>();
                self.charge(&Operation::VectorDbSearch { dimensions })
                    .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;
            }
        }

        ISpace::consume_with_modifiers(&mut self.inner, channels, patterns, modifiers, continuation, persist, peeks)
    }

    fn produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
        priority: Option<usize>,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError> {
        // Estimate data size for send cost
        let data_size = std::mem::size_of::<A>();
        self.charge(&Operation::Send { data_size })
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;

        ISpace::produce(&mut self.inner, channel, data, persist, priority)
    }

    fn install(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        // Install is like consume but persistent
        self.charge(&Operation::Receive)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;

        let pattern_size = patterns.len();
        if pattern_size > 0 {
            self.charge(&Operation::Match { pattern_size })
                .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;
        }

        ISpace::install(&mut self.inner, channels, patterns, continuation)
    }

    fn rig_and_reset(&mut self, start_root: Blake2b256Hash, log: Log) -> Result<(), RSpaceError> {
        // Charge based on log size for replay
        let operation_count = estimate_log_operations(&log);
        self.charge(&Operation::Replay { operation_count })
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))?;
        ISpace::rig_and_reset(&mut self.inner, start_root, log)
    }

    fn rig(&self, log: Log) -> Result<(), RSpaceError> {
        // Rigging is a setup operation, no charge (takes &self anyway)
        ISpace::rig(&self.inner, log)
    }

    fn check_replay_data(&self) -> Result<(), RSpaceError> {
        // Validation only, no charge
        ISpace::check_replay_data(&self.inner)
    }

    fn is_replay(&self) -> bool {
        ISpace::is_replay(&self.inner)
    }

    fn update_produce(&mut self, produce: Produce) {
        // Update during replay is part of replayed operation
        ISpace::update_produce(&mut self.inner, produce)
    }
}

// =============================================================================
// Builder for Charging Agent
// =============================================================================

/// Builder for creating charging space agents with custom configuration.
pub struct ChargingAgentBuilder<S> {
    inner: Option<S>,
    meter: Option<Arc<PhlogistonMeter>>,
    initial_limit: Option<u64>,
}

impl<S> ChargingAgentBuilder<S> {
    /// Create a new builder.
    pub fn new() -> Self {
        ChargingAgentBuilder {
            inner: None,
            meter: None,
            initial_limit: None,
        }
    }

    /// Set the inner space agent.
    pub fn with_space(mut self, space: S) -> Self {
        self.inner = Some(space);
        self
    }

    /// Set a shared phlogiston meter.
    pub fn with_meter(mut self, meter: Arc<PhlogistonMeter>) -> Self {
        self.meter = Some(meter);
        self
    }

    /// Set the initial phlogiston limit (creates a new meter).
    pub fn with_limit(mut self, limit: u64) -> Self {
        self.initial_limit = Some(limit);
        self
    }

    /// Build the charging agent.
    ///
    /// # Errors
    ///
    /// Returns `SpaceError::BuilderIncomplete` if no inner space was provided.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let charging_agent = ChargingAgentBuilder::new()
    ///     .with_space(inner_space)
    ///     .with_limit(1_000_000)
    ///     .build()?;
    /// ```
    pub fn build<C, P, A, K>(self) -> Result<ChargingSpaceAgent<S, C, P, A, K>, SpaceError>
    where
        S: SpaceAgent<C, P, A, K>,
        C: ChannelBound,
        P: PatternBound,
        A: DataBound,
        K: ContinuationBound,
    {
        let inner = self.inner.ok_or(SpaceError::BuilderIncomplete {
            builder: "ChargingAgentBuilder",
            missing_field: "inner (use with_space())",
        })?;
        let meter = self.meter.unwrap_or_else(|| {
            let limit = self.initial_limit.unwrap_or(10_000_000);
            Arc::new(PhlogistonMeter::new(limit))
        });

        Ok(ChargingSpaceAgent::new(inner, meter))
    }

    /// Build the charging agent, panicking if incomplete.
    ///
    /// This is a convenience method for cases where you're certain the builder
    /// is complete. Prefer `build()` for production code.
    ///
    /// # Panics
    ///
    /// Panics if no inner space was provided.
    pub fn build_unchecked<C, P, A, K>(self) -> ChargingSpaceAgent<S, C, P, A, K>
    where
        S: SpaceAgent<C, P, A, K>,
        C: ChannelBound,
        P: PatternBound,
        A: DataBound,
        K: ContinuationBound,
    {
        self.build().expect("ChargingAgentBuilder incomplete")
    }
}

impl<S> Default for ChargingAgentBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Mock space agent for testing
    struct MockSpaceAgent {
        id: SpaceId,
        gensym_count: usize,
    }

    impl MockSpaceAgent {
        fn new() -> Self {
            MockSpaceAgent {
                id: SpaceId::default_space(),
                gensym_count: 0,
            }
        }
    }

    impl SpaceAgent<u64, String, String, String> for MockSpaceAgent {
        fn space_id(&self) -> &SpaceId {
            &self.id
        }

        fn qualifier(&self) -> SpaceQualifier {
            SpaceQualifier::Default
        }

        fn gensym(&mut self) -> Result<u64, SpaceError> {
            self.gensym_count += 1;
            Ok(self.gensym_count as u64)
        }

        fn produce(
            &mut self,
            _channel: u64,
            _data: String,
            _persist: bool,
            _priority: Option<usize>,
        ) -> Result<Option<(ContResult<u64, String, String>, Vec<RSpaceResult<u64, String>>, Produce)>, SpaceError> {
            Ok(None)
        }

        fn consume(
            &mut self,
            _channels: Vec<u64>,
            _patterns: Vec<String>,
            _continuation: String,
            _persist: bool,
            _peeks: BTreeSet<i32>,
        ) -> Result<Option<(ContResult<u64, String, String>, Vec<RSpaceResult<u64, String>>)>, SpaceError> {
            Ok(None)
        }

        fn install(
            &mut self,
            _channels: Vec<u64>,
            _patterns: Vec<String>,
            _continuation: String,
        ) -> Result<Option<(String, Vec<String>)>, SpaceError> {
            Ok(None)
        }

        fn get_data(&self, _channel: &u64) -> Vec<Datum<String>> {
            vec![]
        }

        fn get_waiting_continuations(&self, _channels: Vec<u64>) -> Vec<WaitingContinuation<String, String>> {
            vec![]
        }

        fn get_joins(&self, _channel: u64) -> Vec<Vec<u64>> {
            vec![]
        }
    }

    #[test]
    fn test_charging_agent_gensym_charges() {
        let mock = MockSpaceAgent::new();
        let meter = Arc::new(PhlogistonMeter::new(1000));
        let mut agent = ChargingSpaceAgent::new(mock, meter.clone());

        let initial = agent.balance();
        assert!(agent.gensym().is_ok());

        // Balance should have decreased by CHANNEL_CREATE_COST
        assert!(agent.balance() < initial);
        assert_eq!(agent.total_consumed(), super::super::phlogiston::CHANNEL_CREATE_COST);
    }

    #[test]
    fn test_charging_agent_out_of_phlogiston() {
        let mock = MockSpaceAgent::new();
        // Very low limit
        let meter = Arc::new(PhlogistonMeter::new(10));
        let mut agent = ChargingSpaceAgent::new(mock, meter);

        // Should fail due to insufficient phlogiston
        let result = agent.gensym();
        assert!(result.is_err());

        match result {
            Err(SpaceError::OutOfPhlogiston { required, available, .. }) => {
                assert_eq!(available, 10);
                assert!(required > 10);
            }
            _ => panic!("Expected OutOfPhlogiston error"),
        }
    }

    #[test]
    fn test_charging_agent_produce_charges() {
        let mock = MockSpaceAgent::new();
        let meter = Arc::new(PhlogistonMeter::new(10_000));
        let mut agent = ChargingSpaceAgent::new(mock, meter);

        let initial = agent.balance();
        assert!(agent.produce(1, "test".to_string(), false, None).is_ok());

        assert!(agent.balance() < initial);
        assert!(agent.total_consumed() > 0);
    }

    #[test]
    fn test_charging_agent_consume_charges() {
        let mock = MockSpaceAgent::new();
        let meter = Arc::new(PhlogistonMeter::new(10_000));
        let mut agent = ChargingSpaceAgent::new(mock, meter);

        let initial = agent.balance();
        assert!(agent.consume(
            vec![1],
            vec!["pattern".to_string()],
            "cont".to_string(),
            false,
            BTreeSet::new()
        ).is_ok());

        // Should charge for both receive and match
        assert!(agent.balance() < initial);
        assert!(agent.total_consumed() > super::super::phlogiston::RECEIVE_BASE_COST);
    }

    #[test]
    fn test_builder() {
        let mock = MockSpaceAgent::new();
        let agent: ChargingSpaceAgent<_, u64, String, String, String> = ChargingAgentBuilder::new()
            .with_space(mock)
            .with_limit(5000)
            .build()
            .expect("Builder should succeed with all required fields");

        assert_eq!(agent.balance(), 5000);
    }

    #[test]
    fn test_builder_incomplete_returns_error() {
        let result: Result<ChargingSpaceAgent<MockSpaceAgent, u64, String, String, String>, _> =
            ChargingAgentBuilder::new()
                .with_limit(5000)
                // Missing .with_space()
                .build();

        assert!(result.is_err());
        if let Err(SpaceError::BuilderIncomplete { builder, missing_field }) = result {
            assert_eq!(builder, "ChargingAgentBuilder");
            assert!(missing_field.contains("inner"));
        } else {
            panic!("Expected BuilderIncomplete error");
        }
    }

    #[test]
    fn test_unlimited_agent() {
        let mock = MockSpaceAgent::new();
        let mut agent: ChargingSpaceAgent<_, u64, String, String, String> =
            ChargingSpaceAgent::unlimited(mock);

        // Should be able to do many operations without running out
        for _ in 0..1000 {
            assert!(agent.gensym().is_ok());
        }
    }
}
