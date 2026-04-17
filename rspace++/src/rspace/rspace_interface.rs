// See rspace/src/main/scala/coop/rchain/rspace/ISpace.scala

use std::collections::{BTreeSet, HashMap};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::checkpoint::Checkpoint;
use super::errors::RSpaceError;
use super::hashing::blake2b256_hash::Blake2b256Hash;
use super::internal::{Datum, ProduceCandidate, Row, WaitingContinuation};
use super::trace::Log;
use super::trace::event::Produce;
use crate::rspace::checkpoint::SoftCheckpoint;

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq, Hash)]
pub struct RSpaceResult<C, A> {
    pub channel: C,
    pub matched_datum: A,
    pub removed_datum: A,
    pub persistent: bool,
}

// NOTE: On Scala side, they are defaulting "peek" to false
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub struct ContResult<C, P, K> {
    pub continuation: K,
    pub persistent: bool,
    pub channels: Vec<C>,
    pub patterns: Vec<P>,
    pub peek: bool,
}

pub type MaybeProduceCandidate<C, P, A, K> = Option<ProduceCandidate<C, P, A, K>>;
pub type MaybeConsumeResult<C, P, A, K> = Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)>;
pub type MaybeProduceResult<C, P, A, K> =
    Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>, Produce)>;

/** The interface for RSpace
 *
 * @tparam C a type representing a channel
 * @tparam P a type representing a pattern
 * @tparam A a type representing an arbitrary piece of data and match result
 * @tparam K a type representing a continuation
 *
 * The traits 'Tuplespace' and 'IReplayRSpace' have been combined into this
 * trait
 *
 */
#[async_trait]
pub trait ISpace<C: Eq + std::hash::Hash + Send + Sync, P: Clone + Send + Sync, A: Clone + Send + Sync, K: Clone + Send + Sync>: Send + Sync {
    /** Creates a checkpoint.
     *
     * @return A [[Checkpoint]]
     */
    async fn create_checkpoint(&self) -> Result<Checkpoint, RSpaceError>;

    async fn get_data(&self, channel: &C) -> Vec<Datum<A>>;

    async fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>>;

    async fn get_joins(&self, channel: C) -> Vec<Vec<C>>;

    /** Clears the store.  Does not affect the history trie.
     */
    async fn clear(&self) -> Result<(), RSpaceError>;

    /// Return current history root hash without creating a checkpoint.
    async fn get_root(&self) -> Blake2b256Hash;

    /** Resets the store to the given root.
     *
     * @param root A BLAKE2b256 Hash representing the checkpoint
     */
    async fn reset(&self, root: &Blake2b256Hash) -> Result<(), RSpaceError>;

    async fn consume_result(
        &self,
        channel: Vec<C>,
        pattern: Vec<P>,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError>;

    // TODO: this should not be exposed - OLD
    async fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>>;

    /**
    Allows to create a "soft" checkpoint which doesn't persist the checkpointed data into history.
    This operation is significantly faster than {@link #createCheckpoint()} because the computationally
    expensive operation of creating the history trie is avoided.
    */
    async fn create_soft_checkpoint(&self) -> SoftCheckpoint<C, P, A, K>;

    /// Drain and return the in-memory event log without cloning the hot-store
    /// snapshot. This is a lightweight alternative when only logs are
    /// needed.
    async fn take_event_log(&self) -> Log;

    /**
    Reverts the ISpace to the state checkpointed using {@link #createSoftCheckpoint()}
    */
    async fn revert_to_soft_checkpoint(
        &self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), RSpaceError>;

    /* TUPLESPACE */

    /** Searches the store for data matching all the given patterns at the
     * given channels.
     *
     * If no match is found, then the continuation and patterns are put in
     * the store at the given channels.
     *
     * If a match is found, then the continuation is returned along with the
     * matching data.
     *
     * Matching data stored with the `persist` flag set to `true` will not
     * be removed when it is retrieved. See below for more information
     * about using the `persist` flag.
     *
     * '''NOTE''':
     *
     * A call to [[consume]] that is made with the persist flag set to
     * `true` only persists when there is no matching data.
     *
     * This means that in order to make a continuation "stick" in the store,
     * the user will have to continue to call [[consume]] until a `None`
     * is received.
     *
     * @param channels A Seq of channels on which to search for matching
     * data @param patterns A Seq of patterns with which to search for
     * matching data @param continuation A continuation
     * @param persist Whether or not to attempt to persist the data
     */
    async fn consume(
        &self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError>;

    /** Searches the store for a continuation that has patterns that match
     * the given data at the given channel.
     *
     * If no match is found, then the data is put in the store at the given
     * channel.
     *
     * If a match is found, then the continuation is returned along with the
     * matching data.
     *
     * Matching data or continuations stored with the `persist` flag set to
     * `true` will not be removed when they are retrieved. See below for
     * more information about using the `persist` flag.
     *
     * '''NOTE''':
     *
     * A call to [[produce]] that is made with the persist flag set to
     * `true` only persists when there are no matching continuations.
     *
     * This means that in order to make a piece of data "stick" in the
     * store, the user will have to continue to call [[produce]] until a
     * `None` is received.
     *
     * @param channel A channel on which to search for matching
     * continuations and/or store data @param data A piece of data
     * @param persist Whether or not to attempt to persist the data
     */
    async fn produce(
        &self,
        channel: C,
        data: A,
        persist: bool,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError>;

    async fn install(
        &self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError>;

    /* REPLAY */

    async fn rig_and_reset(&self, start_root: Blake2b256Hash, log: Log) -> Result<(), RSpaceError>;

    async fn rig(&self, log: Log) -> Result<(), RSpaceError>;

    async fn check_replay_data(&self) -> Result<(), RSpaceError>;

    async fn is_replay(&self) -> bool;

    async fn update_produce(&self, produce: Produce) -> ();
}
