//! Layer 3-4: Space Agent Traits
//!
//! This module defines the core traits for space operations:
//! - `SpaceAgent`: Core produce/consume operations (Layer 3)
//! - `CheckpointableSpace`: State management (Layer 4)
//! - `ReplayableSpace`: Deterministic replay (Layer 4)
//!
//! These traits abstract over the underlying storage implementation,
//! allowing different space configurations to be used interchangeably.

use std::collections::BTreeSet;

use super::errors::SpaceError;
use super::types::{ChannelBound, ContinuationBound, DataBound, PatternBound, SpaceId, SpaceQualifier};

use models::rhoapi::EFunction;
use rspace_plus_plus::rspace::{
    checkpoint::{Checkpoint, SoftCheckpoint},
    hashing::blake2b256_hash::Blake2b256Hash,
    internal::{Datum, WaitingContinuation},
    rspace_interface::{ContResult, RSpaceResult},
    trace::{Log, event::Produce},
};

// ==========================================================================
// Layer 3: Space Agent Core Trait
// ==========================================================================

/// Core trait for space operations.
///
/// This is the primary interface for interacting with a space. It provides:
/// - `produce`: Send data to a channel
/// - `consume`: Receive data from channels (with pattern matching)
/// - `gensym`: Generate unique channel names
/// - `install`: Install persistent continuations
///
/// Type parameters:
/// - `C`: Channel type
/// - `P`: Pattern type
/// - `A`: Data type
/// - `K`: Continuation type
pub trait SpaceAgent<C, P, A, K>: Send + Sync
where
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    /// Get the space ID for this agent.
    fn space_id(&self) -> &SpaceId;

    /// Get the qualifier for this space.
    fn qualifier(&self) -> SpaceQualifier;

    /// Generate a new unique channel name in this space.
    ///
    /// The behavior depends on the outer storage type:
    /// - HashMap: Returns a unique random-based name
    /// - Array: Returns the next index, or `OutOfNames` if full
    /// - Vector: Returns the next index, growing the vector
    fn gensym(&mut self) -> Result<C, SpaceError>;

    /// Produce data on a channel.
    ///
    /// If a matching continuation is waiting, it is triggered and the match
    /// result is returned. Otherwise, the data is stored at the channel.
    ///
    /// # Arguments
    /// - `channel`: The channel to send on
    /// - `data`: The data to send
    /// - `persist`: Whether the data should persist after being consumed
    /// - `priority`: Optional priority level for PriorityQueue collections (0 = highest)
    ///
    /// # Returns
    /// - `Ok(Some(...))` if a continuation was triggered
    /// - `Ok(None)` if the data was stored
    /// - `Err(...)` on error (e.g., Cell already full)
    fn produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
        priority: Option<usize>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>, Produce)>, SpaceError>;

    /// Consume data from channels.
    ///
    /// Searches for data matching the given patterns on the given channels.
    /// If matches are found, the continuation is triggered immediately.
    /// Otherwise, the continuation is stored to wait for matching data.
    ///
    /// # Arguments
    /// - `channels`: The channels to receive from
    /// - `patterns`: The patterns to match (one per channel)
    /// - `continuation`: The continuation to execute on match
    /// - `persist`: Whether the continuation should persist after triggering
    /// - `peeks`: Set of channel indices to peek (don't consume)
    ///
    /// # Returns
    /// - `Ok(Some(...))` if matching data was found
    /// - `Ok(None)` if the continuation was stored
    /// - `Err(...)` on error
    fn consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)>, SpaceError>;

    /// Consume data from channels with pattern modifiers (sim, rank, etc.).
    ///
    /// This method extends the standard consume operation with modifier-based
    /// matching. Modifiers are represented as EFunction calls:
    /// - `sim(query, metric, threshold, ...)`: VectorDB similarity matching
    /// - `rank(query, function, params...)`: Result ranking/filtering
    ///
    /// # Arguments
    /// - `channels`: The channels to receive from
    /// - `patterns`: The patterns to match (one per channel)
    /// - `modifiers`: Pattern modifiers for each channel (as EFunction calls)
    /// - `continuation`: The continuation to execute on match
    /// - `persist`: Whether the continuation should persist after triggering
    /// - `peeks`: Set of channel indices to peek (don't consume)
    ///
    /// # Returns
    /// - `Ok(Some(...))` if matching data was found
    /// - `Ok(None)` if the continuation was stored
    /// - `Err(...)` on error
    ///
    /// # Default Implementation
    /// Falls back to regular consume, ignoring modifiers.
    /// Spaces that support VectorDB collections should override this.
    fn consume_with_modifiers(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        _modifiers: Vec<Vec<EFunction>>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)>, SpaceError> {
        // Default implementation: fall back to standard consume
        // VectorDB spaces should override this to implement modifier-based matching
        self.consume(channels, patterns, continuation, persist, peeks)
    }

    /// Install a persistent continuation.
    ///
    /// Similar to consume with `persist=true`, but always stores the continuation
    /// even if matching data exists.
    ///
    /// # Arguments
    /// - `channels`: The channels to receive from
    /// - `patterns`: The patterns to match
    /// - `continuation`: The continuation to install
    ///
    /// # Returns
    /// - `Ok(Some(...))` if matching data was found (continuation still installed)
    /// - `Ok(None)` if no matching data
    fn install(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, SpaceError>;

    /// Get data stored at a channel.
    fn get_data(&self, channel: &C) -> Vec<Datum<A>>;

    /// Get waiting continuations for channels.
    fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>>;

    /// Get join patterns for a channel.
    fn get_joins(&self, channel: C) -> Vec<Vec<C>>;
}

// ==========================================================================
// Layer 4: Checkpointable Space
// ==========================================================================

/// Extension trait for checkpointing capabilities.
///
/// Checkpoints capture the state of a space at a point in time, allowing
/// rollback for speculative execution or recovery from failures.
pub trait CheckpointableSpace<C, P, A, K>: SpaceAgent<C, P, A, K>
where
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    /// Create a persistent checkpoint.
    ///
    /// This captures the current state and writes it to the history trie.
    /// The checkpoint can be used to reset the space to this state later.
    fn create_checkpoint(&mut self) -> Result<Checkpoint, SpaceError>;

    /// Create a soft (non-persistent) checkpoint.
    ///
    /// This is faster than a full checkpoint but doesn't persist to storage.
    /// Useful for speculative execution that may be rolled back.
    fn create_soft_checkpoint(&mut self) -> SoftCheckpoint<C, P, A, K>;

    /// Revert to a soft checkpoint.
    ///
    /// Restores the space to the state captured in the soft checkpoint.
    fn revert_to_soft_checkpoint(
        &mut self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), SpaceError>;

    /// Reset to a checkpoint by its merkle root.
    ///
    /// Restores the space to a previously created checkpoint.
    fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), SpaceError>;

    /// Clear all data and continuations.
    ///
    /// Does not affect the history trie.
    fn clear(&mut self) -> Result<(), SpaceError>;
}

// ==========================================================================
// Layer 4: Replayable Space
// ==========================================================================

/// Extension trait for deterministic replay.
///
/// Replay is used to re-execute a sequence of operations deterministically,
/// typically for validation or recovery. The space is "rigged" with a log
/// of expected operations and verifies that replay matches the log.
pub trait ReplayableSpace<C, P, A, K>: CheckpointableSpace<C, P, A, K>
where
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    /// Rig the space for replay and reset to a starting state.
    ///
    /// # Arguments
    /// - `start_root`: The merkle root to reset to before replay
    /// - `log`: The log of operations to replay
    fn rig_and_reset(
        &mut self,
        start_root: Blake2b256Hash,
        log: Log,
    ) -> Result<(), SpaceError>;

    /// Rig the space for replay without resetting.
    ///
    /// Note: Takes `&self` not `&mut self` per spec.
    fn rig(&self, log: Log) -> Result<(), SpaceError>;

    /// Check that replay data matches expectations.
    ///
    /// Verifies that all operations in the log were replayed correctly.
    fn check_replay_data(&self) -> Result<(), SpaceError>;

    /// Check if the space is in replay mode.
    fn is_replay(&self) -> bool;

    /// Update produce result during replay.
    ///
    /// Called after a produce operation in replay mode to record the result.
    fn update_produce(&mut self, produce: Produce);
}

// ==========================================================================
// Dynamic Space Agent (Type-Erased)
// ==========================================================================

/// Type-erased space agent for use in the registry.
///
/// This allows different space implementations to be stored in the same
/// collection. Uses the same type parameters as the interpreter.
pub type DynSpaceAgent<C, P, A, K> = Box<dyn SpaceAgent<C, P, A, K>>;

/// Blanket implementation for boxed space agents.
///
/// This allows `Box<dyn SpaceAgent>` to be used where `SpaceAgent` is expected,
/// enabling type-erased storage in registries and factories.
impl<C, P, A, K> SpaceAgent<C, P, A, K> for Box<dyn SpaceAgent<C, P, A, K>>
where
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    fn space_id(&self) -> &SpaceId {
        (**self).space_id()
    }

    fn qualifier(&self) -> SpaceQualifier {
        (**self).qualifier()
    }

    fn gensym(&mut self) -> Result<C, SpaceError> {
        (**self).gensym()
    }

    fn produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
        priority: Option<usize>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>, Produce)>, SpaceError> {
        (**self).produce(channel, data, persist, priority)
    }

    fn consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)>, SpaceError> {
        (**self).consume(channels, patterns, continuation, persist, peeks)
    }

    fn consume_with_modifiers(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        modifiers: Vec<Vec<EFunction>>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)>, SpaceError> {
        (**self).consume_with_modifiers(channels, patterns, modifiers, continuation, persist, peeks)
    }

    fn install(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, SpaceError> {
        (**self).install(channels, patterns, continuation)
    }

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        (**self).get_data(channel)
    }

    fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        (**self).get_waiting_continuations(channels)
    }

    fn get_joins(&self, channel: C) -> Vec<Vec<C>> {
        (**self).get_joins(channel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic compile-time check that traits are object-safe
    fn _check_object_safety<C, P, A, K>(_: &dyn SpaceAgent<C, P, A, K>)
    where
        C: ChannelBound,
        P: PatternBound,
        A: DataBound,
        K: ContinuationBound,
    {
    }
}
