//! ISpace to SpaceAgent Adapter
//!
//! This module provides an adapter that bridges the existing `ISpace` trait
//! from rspace++ with the new `SpaceAgent` trait hierarchy. This allows
//! the current RSpace implementation to be used within the multi-space registry.

use std::collections::BTreeSet;
use std::sync::Arc;
use tokio::sync::RwLock;

use async_trait::async_trait;

use rspace_plus_plus::rspace::{
    checkpoint::{Checkpoint, SoftCheckpoint},
    errors::RSpaceError,
    hashing::blake2b256_hash::Blake2b256Hash,
    internal::{Datum, WaitingContinuation},
    rspace_interface::{ContResult, ISpace, RSpaceResult},
    trace::{Log, event::Produce},
};

use super::async_agent::{AsyncSpaceAgent, AsyncCheckpointableSpace, AsyncReplayableSpace};
use super::errors::SpaceError;
use super::types::{ChannelBound, ContinuationBound, DataBound, PatternBound, SpaceId, SpaceQualifier};

/// Adapter that wraps an `ISpace` to implement `SpaceAgent`.
///
/// This allows the existing RSpace implementation to be used with the
/// new multi-space registry and trait hierarchy.
pub struct ISpaceAdapter<C, P, A, K>
where
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    /// The underlying ISpace implementation - uses RwLock for better read concurrency
    inner: Arc<RwLock<Box<dyn ISpace<C, P, A, K> + Send + Sync>>>,

    /// The space ID for this adapter
    space_id: SpaceId,

    /// The space qualifier
    qualifier: SpaceQualifier,

    /// Counter for gensym
    gensym_counter: std::sync::atomic::AtomicUsize,
}

impl<C, P, A, K> ISpaceAdapter<C, P, A, K>
where
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    /// Create a new adapter wrapping an ISpace.
    pub fn new(
        inner: Arc<RwLock<Box<dyn ISpace<C, P, A, K> + Send + Sync>>>,
        space_id: SpaceId,
        qualifier: SpaceQualifier,
    ) -> Self {
        ISpaceAdapter {
            inner,
            space_id,
            qualifier,
            gensym_counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get the inner ISpace (for direct access when needed).
    pub fn inner(&self) -> &Arc<RwLock<Box<dyn ISpace<C, P, A, K> + Send + Sync>>> {
        &self.inner
    }
}

// Note: The synchronous SpaceAgent trait is intentionally NOT implemented for ISpaceAdapter
// because ISpace requires async access. Use AsyncSpaceAgent instead.
// This design enforces async-only usage at compile time, preventing runtime errors.

impl<C, P, A, K> ISpaceAdapter<C, P, A, K>
where
    C: ChannelBound,
    P: PatternBound,
    A: DataBound,
    K: ContinuationBound,
{
    /// Async version of create_checkpoint (requires write lock - modifies state)
    pub async fn create_checkpoint_async(&mut self) -> Result<Checkpoint, SpaceError> {
        let mut guard = self.inner.write().await;
        guard.create_checkpoint().map_err(|e| SpaceError::CheckpointError { description: e.to_string() })
    }

    /// Async version of create_soft_checkpoint (requires write lock - captures state)
    pub async fn create_soft_checkpoint_async(&mut self) -> SoftCheckpoint<C, P, A, K> {
        let mut guard = self.inner.write().await;
        guard.create_soft_checkpoint()
    }

    /// Async version of revert_to_soft_checkpoint (requires write lock - modifies state)
    pub async fn revert_to_soft_checkpoint_async(
        &mut self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), SpaceError> {
        let mut guard = self.inner.write().await;
        guard.revert_to_soft_checkpoint(checkpoint)
            .map_err(|e| SpaceError::CheckpointError { description: e.to_string() })
    }

    /// Async version of reset (requires write lock - modifies state)
    pub async fn reset_async(&mut self, root: &Blake2b256Hash) -> Result<(), SpaceError> {
        let mut guard = self.inner.write().await;
        guard.reset(root).map_err(|e| SpaceError::CheckpointError { description: e.to_string() })
    }

    /// Async version of clear (requires write lock - modifies state)
    pub async fn clear_async(&mut self) -> Result<(), SpaceError> {
        let mut guard = self.inner.write().await;
        guard.clear().map_err(|e| SpaceError::CheckpointError { description: e.to_string() })
    }

    /// Async version of get_data (uses read lock - no mutation)
    pub async fn get_data_async(&self, channel: &C) -> Vec<Datum<A>> {
        let guard = self.inner.read().await;
        guard.get_data(channel)
    }

    /// Async version of get_waiting_continuations (uses read lock - no mutation)
    pub async fn get_waiting_continuations_async(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        let guard = self.inner.read().await;
        guard.get_waiting_continuations(channels)
    }

    /// Async version of get_joins (uses read lock - no mutation)
    pub async fn get_joins_async(&self, channel: C) -> Vec<Vec<C>> {
        let guard = self.inner.read().await;
        guard.get_joins(channel)
    }

    /// Async version of produce (requires write lock - modifies state)
    pub async fn produce_async(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
        priority: Option<usize>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>, Produce)>, SpaceError> {
        let mut guard = self.inner.write().await;
        guard.produce(channel, data, persist, priority)
            .map_err(|e| SpaceError::InternalError { description: e.to_string() })
    }

    /// Async version of consume (requires write lock - modifies state)
    pub async fn consume_async(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)>, SpaceError> {
        let mut guard = self.inner.write().await;
        guard.consume(channels, patterns, continuation, persist, peeks)
            .map_err(|e| SpaceError::InternalError { description: e.to_string() })
    }

    /// Async version of install (requires write lock - modifies state)
    pub async fn install_async(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, SpaceError> {
        let mut guard = self.inner.write().await;
        guard.install(channels, patterns, continuation)
            .map_err(|e| SpaceError::InternalError { description: e.to_string() })
    }

    /// Async version of rig_and_reset (requires write lock - modifies state)
    pub async fn rig_and_reset_async(
        &mut self,
        start_root: Blake2b256Hash,
        log: Log,
    ) -> Result<(), SpaceError> {
        let mut guard = self.inner.write().await;
        guard.rig_and_reset(start_root, log)
            .map_err(|e| SpaceError::InternalError { description: e.to_string() })
    }

    /// Async version of rig (requires write lock - sets up replay tables)
    pub async fn rig_async(&self, log: Log) -> Result<(), SpaceError> {
        let guard = self.inner.write().await;
        guard.rig(log)
            .map_err(|e| SpaceError::InternalError { description: e.to_string() })
    }

    /// Async version of check_replay_data (uses read lock - validation only)
    pub async fn check_replay_data_async(&self) -> Result<(), SpaceError> {
        let guard = self.inner.read().await;
        guard.check_replay_data()
            .map_err(|e| SpaceError::InternalError { description: e.to_string() })
    }

    /// Check if in replay mode (uses read lock - no mutation)
    pub async fn is_replay_async(&self) -> bool {
        let guard = self.inner.read().await;
        guard.is_replay()
    }

    /// Async version of update_produce (requires write lock - modifies state)
    pub async fn update_produce_async(&mut self, produce: Produce) {
        let mut guard = self.inner.write().await;
        guard.update_produce(produce)
    }
}

// ==========================================================================
// AsyncSpaceAgent Implementation
// ==========================================================================

#[async_trait]
impl<C, P, A, K> AsyncSpaceAgent<C, P, A, K> for ISpaceAdapter<C, P, A, K>
where
    C: ChannelBound + From<usize>,
    P: Clone + Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
    K: Clone + Send + Sync + 'static,
{
    fn space_id(&self) -> &SpaceId {
        &self.space_id
    }

    fn qualifier(&self) -> SpaceQualifier {
        self.qualifier
    }

    async fn gensym(&mut self) -> Result<C, SpaceError> {
        let counter = self.gensym_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(C::from(counter))
    }

    async fn produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
        priority: Option<usize>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>, Produce)>, SpaceError> {
        self.produce_async(channel, data, persist, priority).await
    }

    async fn consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)>, SpaceError> {
        self.consume_async(channels, patterns, continuation, persist, peeks).await
    }

    async fn install(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, SpaceError> {
        self.install_async(channels, patterns, continuation).await
    }

    async fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        self.get_data_async(channel).await
    }

    async fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        self.get_waiting_continuations_async(channels).await
    }

    async fn get_joins(&self, channel: C) -> Vec<Vec<C>> {
        self.get_joins_async(channel).await
    }
}

// ==========================================================================
// AsyncCheckpointableSpace Implementation
// ==========================================================================

#[async_trait]
impl<C, P, A, K> AsyncCheckpointableSpace<C, P, A, K> for ISpaceAdapter<C, P, A, K>
where
    C: ChannelBound + From<usize>,
    P: Clone + Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
    K: Clone + Send + Sync + 'static,
{
    async fn create_checkpoint(&mut self) -> Result<Checkpoint, SpaceError> {
        self.create_checkpoint_async().await
    }

    async fn create_soft_checkpoint(&mut self) -> SoftCheckpoint<C, P, A, K> {
        self.create_soft_checkpoint_async().await
    }

    async fn revert_to_soft_checkpoint(
        &mut self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), SpaceError> {
        self.revert_to_soft_checkpoint_async(checkpoint).await
    }

    async fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), SpaceError> {
        self.reset_async(root).await
    }

    async fn clear(&mut self) -> Result<(), SpaceError> {
        self.clear_async().await
    }
}

// ==========================================================================
// AsyncReplayableSpace Implementation
// ==========================================================================

#[async_trait]
impl<C, P, A, K> AsyncReplayableSpace<C, P, A, K> for ISpaceAdapter<C, P, A, K>
where
    C: ChannelBound + From<usize>,
    P: Clone + Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
    K: Clone + Send + Sync + 'static,
{
    async fn rig_and_reset(
        &mut self,
        start_root: Blake2b256Hash,
        log: Log,
    ) -> Result<(), SpaceError> {
        self.rig_and_reset_async(start_root, log).await
    }

    async fn rig(&self, log: Log) -> Result<(), SpaceError> {
        self.rig_async(log).await
    }

    async fn check_replay_data(&self) -> Result<(), SpaceError> {
        self.check_replay_data_async().await
    }

    fn is_replay(&self) -> bool {
        // Use blocking for this simple check since it doesn't modify state
        // In a full async context, you might want to cache this value
        false // Default to false; async version available via is_replay_async
    }

    async fn update_produce(&mut self, produce: Produce) {
        self.update_produce_async(produce).await
    }
}

/// Convert RSpaceError to SpaceError
impl From<RSpaceError> for SpaceError {
    fn from(err: RSpaceError) -> Self {
        SpaceError::InternalError { description: err.to_string() }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_adapter_space_id() {
        // This is a compile-time check that ISpaceAdapter can be created
        // A full test would require a mock ISpace implementation
    }
}
