//! Async Space Agent Traits
//!
//! This module provides async versions of the SpaceAgent trait hierarchy
//! for use with tokio-based runtime where mutex acquisition is async.
//!
//! These traits mirror the synchronous versions in `agent.rs` but use
//! async methods, allowing proper integration with the async ISpace
//! implementation from rspace++.

use std::collections::BTreeSet;

use async_trait::async_trait;

use super::errors::SpaceError;
use super::types::{ChannelBound, ContinuationBound, DataBound, PatternBound, SpaceId, SpaceQualifier};

use rspace_plus_plus::rspace::{
    checkpoint::{Checkpoint, SoftCheckpoint},
    hashing::blake2b256_hash::Blake2b256Hash,
    internal::{Datum, WaitingContinuation},
    rspace_interface::{ContResult, RSpaceResult},
    trace::{Log, event::Produce},
};

// ==========================================================================
// Async Layer 3: Space Agent Core Trait
// ==========================================================================

/// Async version of the core trait for space operations.
///
/// This is the primary async interface for interacting with a space. It provides:
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
#[async_trait]
pub trait AsyncSpaceAgent<C, P, A, K>: Send + Sync
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
    async fn gensym(&mut self) -> Result<C, SpaceError>;

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
    async fn produce(
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
    async fn consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)>, SpaceError>;

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
    async fn install(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, SpaceError>;

    /// Get data stored at a channel.
    async fn get_data(&self, channel: &C) -> Vec<Datum<A>>;

    /// Get waiting continuations for channels.
    async fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>>;

    /// Get join patterns for a channel.
    async fn get_joins(&self, channel: C) -> Vec<Vec<C>>;
}

// ==========================================================================
// Async Layer 4: Checkpointable Space
// ==========================================================================

/// Async extension trait for checkpointing capabilities.
///
/// Checkpoints capture the state of a space at a point in time, allowing
/// rollback for speculative execution or recovery from failures.
#[async_trait]
pub trait AsyncCheckpointableSpace<C, P, A, K>: AsyncSpaceAgent<C, P, A, K>
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
    async fn create_checkpoint(&mut self) -> Result<Checkpoint, SpaceError>;

    /// Create a soft (non-persistent) checkpoint.
    ///
    /// This is faster than a full checkpoint but doesn't persist to storage.
    /// Useful for speculative execution that may be rolled back.
    async fn create_soft_checkpoint(&mut self) -> SoftCheckpoint<C, P, A, K>;

    /// Revert to a soft checkpoint.
    ///
    /// Restores the space to the state captured in the soft checkpoint.
    async fn revert_to_soft_checkpoint(
        &mut self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), SpaceError>;

    /// Reset to a checkpoint by its merkle root.
    ///
    /// Restores the space to a previously created checkpoint.
    async fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), SpaceError>;

    /// Clear all data and continuations.
    ///
    /// Does not affect the history trie.
    async fn clear(&mut self) -> Result<(), SpaceError>;
}

// ==========================================================================
// Async Layer 4: Replayable Space
// ==========================================================================

/// Async extension trait for deterministic replay.
///
/// Replay is used to re-execute a sequence of operations deterministically,
/// typically for validation or recovery. The space is "rigged" with a log
/// of expected operations and verifies that replay matches the log.
#[async_trait]
pub trait AsyncReplayableSpace<C, P, A, K>: AsyncCheckpointableSpace<C, P, A, K>
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
    async fn rig_and_reset(
        &mut self,
        start_root: Blake2b256Hash,
        log: Log,
    ) -> Result<(), SpaceError>;

    /// Rig the space for replay without resetting.
    async fn rig(&self, log: Log) -> Result<(), SpaceError>;

    /// Check that replay data matches expectations.
    ///
    /// Verifies that all operations in the log were replayed correctly.
    async fn check_replay_data(&self) -> Result<(), SpaceError>;

    /// Check if the space is in replay mode.
    fn is_replay(&self) -> bool;

    /// Update produce result during replay.
    ///
    /// Called after a produce operation in replay mode to record the result.
    async fn update_produce(&mut self, produce: Produce);
}

// ==========================================================================
// Dynamic Async Space Agent (Type-Erased)
// ==========================================================================

/// Type-erased async space agent for use in the registry.
///
/// This allows different async space implementations to be stored in the same
/// collection. Uses the same type parameters as the interpreter.
pub type DynAsyncSpaceAgent<C, P, A, K> = Box<dyn AsyncSpaceAgent<C, P, A, K>>;

#[cfg(test)]
mod tests {
    use super::*;

    // Basic compile-time check that async traits are object-safe
    fn _check_object_safety<C, P, A, K>(_: &dyn AsyncSpaceAgent<C, P, A, K>)
    where
        C: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
        P: Clone + Send + Sync + 'static,
        A: Clone + Send + Sync + 'static,
        K: Clone + Send + Sync + 'static,
    {
    }
}
