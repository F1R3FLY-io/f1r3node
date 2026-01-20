// See rspace/src/main/scala/coop/rchain/rspace/ISpace.scala

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

use crate::rspace::checkpoint::SoftCheckpoint;

use super::{
    checkpoint::Checkpoint,
    errors::RSpaceError,
    hashing::blake2b256_hash::Blake2b256Hash,
    internal::{Datum, ProduceCandidate, Row, WaitingContinuation},
    trace::{Log, event::Produce},
};

/// Result of a successful consume or produce operation.
///
/// When prefix semantics are enabled (PathMap stores), `suffix_key` contains
/// the path suffix for data matched at a descendant path. For exact matches,
/// `suffix_key` is `None`.
///
/// # Suffix Key Semantics (from "Reifying RSpaces" design spec)
///
/// Data at `@[0, 1, 2]` consumed at prefix `@[0, 1]` should be wrapped as
/// `[suffix_element, original_data]`. For example:
/// - Original: `"hi"` at `@[0, 1, 2]`
/// - Consumed at: `@[0, 1]`
/// - Result: `[2, "hi"]` (suffix key element prepended)
///
/// The wrapping is performed by `wrap_with_suffix_key` in the util module.
#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq, Hash)]
pub struct RSpaceResult<C, A> {
    pub channel: C,
    pub matched_datum: A,
    pub removed_datum: A,
    pub persistent: bool,
    /// Suffix key for prefix matches. `None` for exact matches.
    /// Contains the path elements between the consume prefix and the actual data channel.
    /// E.g., for data at `@[0,1,2]` consumed at `@[0,1]`, this is `Some(vec![2])`.
    #[serde(default)]
    pub suffix_key: Option<Vec<u8>>,
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
 * The traits 'Tuplespace' and 'IReplayRSpace' have been combined into this trait
 *
 */
pub trait ISpace<C: Eq + std::hash::Hash, P: Clone, A: Clone, K: Clone> {
    /** Creates a checkpoint.
     *
     * @return A [[Checkpoint]]
     */
    fn create_checkpoint(&mut self) -> Result<Checkpoint, RSpaceError>;

    fn get_data(&self, channel: &C) -> Vec<Datum<A>>;

    fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>>;

    fn get_joins(&self, channel: C) -> Vec<Vec<C>>;

    /** Clears the store.  Does not affect the history trie.
     */
    fn clear(&mut self) -> Result<(), RSpaceError>;

    /** Resets the store to the given root.
     *
     * @param root A BLAKE2b256 Hash representing the checkpoint
     */
    fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), RSpaceError>;

    fn consume_result(
        &mut self,
        channel: Vec<C>,
        pattern: Vec<P>,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError>;

    // TODO: this should not be exposed - OLD
    fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>>;

    /**
    Allows to create a "soft" checkpoint which doesn't persist the checkpointed data into history.
    This operation is significantly faster than {@link #createCheckpoint()} because the computationally
    expensive operation of creating the history trie is avoided.
    */
    fn create_soft_checkpoint(&mut self) -> SoftCheckpoint<C, P, A, K>;

    /**
    Reverts the ISpace to the state checkpointed using {@link #createSoftCheckpoint()}
    */
    fn revert_to_soft_checkpoint(
        &mut self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), RSpaceError>;

    /* TUPLESPACE */

    /** Searches the store for data matching all the given patterns at the given channels.
     *
     * If no match is found, then the continuation and patterns are put in the store at the given
     * channels.
     *
     * If a match is found, then the continuation is returned along with the matching data.
     *
     * Matching data stored with the `persist` flag set to `true` will not be removed when it is
     * retrieved. See below for more information about using the `persist` flag.
     *
     * '''NOTE''':
     *
     * A call to [[consume]] that is made with the persist flag set to `true` only persists when
     * there is no matching data.
     *
     * This means that in order to make a continuation "stick" in the store, the user will have to
     * continue to call [[consume]] until a `None` is received.
     *
     * @param channels A Seq of channels on which to search for matching data
     * @param patterns A Seq of patterns with which to search for matching data
     * @param continuation A continuation
     * @param persist Whether or not to attempt to persist the data
     */
    fn consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError>;

    /** Searches for matching data with optional similarity-based pattern matching.
     *
     * This extends the standard consume operation with VectorDB-style similarity matching.
     * When similarity data is provided for a channel, data is matched based on embedding
     * similarity rather than exact pattern matching.
     *
     * Default implementation ignores similarity and delegates to regular consume.
     * Spaces that support VectorDB collections should override this.
     *
     * @param channels A Seq of channels on which to search for matching data
     * @param patterns A Seq of patterns with which to search for matching data
     * @param modifiers Serialized pattern modifiers (EFunctions per channel, empty for exact match)
     * @param continuation A continuation
     * @param persist Whether or not to attempt to persist the data
     * @param peeks Set of channel indices to peek (don't consume)
     */
    fn consume_with_modifiers(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        _modifiers: Vec<Vec<u8>>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError> {
        // Default: ignore modifiers, use regular consume
        self.consume(channels, patterns, continuation, persist, peeks)
    }

    /** Searches the store for a continuation that has patterns that match the given data at the
     * given channel.
     *
     * If no match is found, then the data is put in the store at the given channel.
     *
     * If a match is found, then the continuation is returned along with the matching data.
     *
     * Matching data or continuations stored with the `persist` flag set to `true` will not be
     * removed when they are retrieved. See below for more information about using the `persist`
     * flag.
     *
     * '''NOTE''':
     *
     * A call to [[produce]] that is made with the persist flag set to `true` only persists when
     * there are no matching continuations.
     *
     * This means that in order to make a piece of data "stick" in the store, the user will have to
     * continue to call [[produce]] until a `None` is received.
     *
     * @param channel A channel on which to search for matching continuations and/or store data
     * @param data A piece of data
     * @param persist Whether or not to attempt to persist the data
     * @param priority Optional priority level for PriorityQueue collections (0 = highest)
     */
    fn produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
        priority: Option<usize>,
    ) -> Result<MaybeProduceResult<C, P, A, K>, RSpaceError>;

    fn install(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError>;

    /* REPLAY */

    fn rig_and_reset(&mut self, start_root: Blake2b256Hash, log: Log) -> Result<(), RSpaceError>;

    fn rig(&self, log: Log) -> Result<(), RSpaceError>;

    fn check_replay_data(&self) -> Result<(), RSpaceError>;

    fn is_replay(&self) -> bool;

    fn update_produce(&mut self, produce: Produce) -> ();
}
