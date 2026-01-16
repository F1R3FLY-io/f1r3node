//! GenericRSpace<CS, M, C, P, A, K, DC, CC> - Parameterized RSpace Implementation
//!
//! This module implements the core `GenericRSpace` struct as specified in the
//! "Reifying RSpaces" specification (Lines 641-715). It provides a flexible,
//! parameterized implementation of the RSpace tuple space.
//!
//! # Type Parameters
//!
//! - `CS`: ChannelStore - How channels are indexed (HashMap, PathMap, Array, etc.)
//! - `M`: Match - Pattern matching strategy (RholangMatch, VectorDBMatch, etc.)
//! - `C`: Channel type
//! - `P`: Pattern type
//! - `A`: Data type
//! - `K`: Continuation type
//! - `DC`: DataCollection - How data is stored at each channel (Bag, Queue, Stack, etc.)
//! - `CC`: ContinuationCollection - How continuations are stored (Bag, Queue, Stack, etc.)
//!
//! # Design
//!
//! The GenericRSpace combines:
//! - A ChannelStore for channel indexing and data/continuation storage
//! - A Matcher for pattern matching semantics
//! - Optional HistoryStore for checkpointing
//! - SpaceQualifier for persistence and concurrency behavior
//!
//! This design enables creating spaces with different combinations of:
//! - Storage strategies (HashMap for O(1), PathMap for hierarchical, etc.)
//! - Matching semantics (structural, similarity-based, etc.)
//! - Persistence behavior (Default, Temp, Seq)

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

use smallvec::SmallVec;
use dashmap::DashMap;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use std::collections::HashSet;
use rspace_plus_plus::rspace::{
    checkpoint::{Checkpoint, SoftCheckpoint},
    hashing::blake2b256_hash::Blake2b256Hash,
    hot_store::HotStoreState,
    internal::{Datum, WaitingContinuation, MultisetMultiMap},
    rspace_interface::{ContResult, RSpaceResult},
    trace::{Log, event::{Produce, Event, IOEvent, COMM}},
};

use models::rhoapi::{EFunction, GPrivate, GUnforgeable, ListParWithRandom, Par, g_unforgeable::UnfInstance};
use uuid::Uuid;

use super::agent::{CheckpointableSpace, ReplayableSpace, SpaceAgent};
use super::channel_store::ChannelStore;
use super::collections::{ContinuationCollection, DataCollection, EmbeddingType, SimilarityCollection, SimilarityMetric};
use super::errors::SpaceError;
use super::history::BoxedHistoryStore;
use super::matcher::Match;
use super::types::{BoxedTheory, SpaceConfig, SpaceId, SpaceQualifier, Validatable, TheoryValidator, get_path_suffix, SuffixKey};
use super::similarity_extraction::{
    extract_embedding_from_map, extract_channel_id_from_par,
    compute_cosine_similarity, compute_dot_product, compute_euclidean_similarity,
    compute_manhattan_similarity, compute_hamming_similarity, compute_jaccard_similarity,
};

// Re-export for backward compatibility
pub use super::similarity_extraction::ExtractedModifiers;

// Also re-export extract_modifiers_from_efunctions for backward compatibility
pub use super::similarity_extraction::extract_modifiers_from_efunctions;

// =============================================================================
// GenericRSpace Struct
// =============================================================================

/// Generic RSpace parameterized by storage strategy and matcher.
///
/// This is the core implementation of a reified RSpace. It provides the full
/// `SpaceAgent` and `CheckpointableSpace` interfaces while being flexible
/// enough to support different storage and matching strategies.
///
/// # Examples
///
/// ```ignore
/// // Create a HashMap-based space with exact matching
/// let store = HashMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
/// let matcher = ExactMatch::new();
/// let space = GenericRSpace::new(
///     store,
///     matcher,
///     SpaceId::default_space(),
///     SpaceQualifier::Default,
/// );
/// ```
pub struct GenericRSpace<CS, M>
where
    CS: ChannelStore,
    M: Match<CS::Pattern, CS::Data>,
{
    /// The channel store for data and continuation storage
    channel_store: CS,

    /// The pattern matcher
    matcher: M,

    /// Unique identifier for this space
    space_id: SpaceId,

    /// Qualifier determining persistence and concurrency behavior
    qualifier: SpaceQualifier,

    /// Optional history store for checkpointing (None for temp spaces)
    history_store: Option<BoxedHistoryStore>,

    /// Soft checkpoint for speculative execution rollback
    soft_checkpoint: Option<SoftCheckpoint<CS::Channel, CS::Pattern, CS::Data, CS::Continuation>>,

    /// Replay log for deterministic replay (if in replay mode)
    replay_log: Option<Log>,

    /// Whether this space is in replay mode
    is_replay: bool,

    /// Replay data for tracking COMM events during replay verification.
    /// Maps IOEvents to their associated COMM events from the log.
    ///
    /// Note: MultisetMultiMap uses DashMap internally, providing lock-free concurrent access.
    /// No outer RwLock needed since all MultisetMultiMap methods take &self.
    replay_data: MultisetMultiMap<IOEvent, COMM>,

    /// Optional theory for data validation before storage.
    ///
    /// If set, all data sent to this space will be validated against the theory
    /// before being stored. Invalid data will be rejected with a TheoryValidationError.
    ///
    /// Formal Correspondence: GenericRSpace.v (produce_validates_data)
    theory: Option<BoxedTheory>,

    /// Stack of channel store snapshots for nested soft checkpoints.
    ///
    /// Each soft checkpoint stores a clone of the channel store at that point.
    /// Reverting pops and restores from this stack.
    soft_checkpoint_stack: Vec<(CS, usize)>, // (channel_store_snapshot, gensym_counter_at_checkpoint)

    /// Similarity query matrices for efficient batch similarity computation.
    ///
    /// Keyed by channel, each matrix stores normalized query embeddings from
    /// waiting similarity-based continuations. When `produce()` stores new data,
    /// it queries these matrices using SIMD-optimized matrix-vector multiplication
    /// to find matching continuations.
    ///
    /// # Architecture
    ///
    /// This enables the store-first approach for similarity matching:
    /// 1. `produce()` stores data in VectorDB (embedding gets normalized)
    /// 2. `produce()` queries the channel's matrix with the normalized embedding
    /// 3. SIMD batch computation finds all queries with similarity >= threshold
    /// 4. Best-matching continuation is fired
    ///
    /// The counter tracks the next continuation ID to assign.
    ///
    /// # Lazy Allocation
    /// Wrapped in Option to save ~100 bytes per space that doesn't use similarity queries.
    /// Most spaces don't use VectorDB/similarity features, so this is None by default.
    similarity_queries: Option<HashMap<CS::Channel, super::collections::SimilarityQueryMatrix>>,

    /// Counter for generating unique continuation IDs for similarity queries.
    next_continuation_id: std::sync::atomic::AtomicUsize,

    /// Registry of lazy result producers for similarity query result channels.
    ///
    /// When a similarity query matches, instead of returning all results immediately,
    /// we create a lazy result channel and store a `LazyResultProducer` here.
    /// When consumers try to read from the lazy channel, the producer provides
    /// documents one at a time, enabling:
    ///
    /// - **Lazy evaluation**: Documents computed on-demand
    /// - **Early termination**: Consumer can stop after first N results
    /// - **Backpressure**: Production rate controlled by consumption
    ///
    /// # Key Format
    ///
    /// The key is the result channel ID (GPrivate bytes). When `consume()` is called
    /// on a channel, we check if it's a lazy result channel by looking up its ID here.
    ///
    /// # Lifecycle
    ///
    /// 1. `consume_with_similarity()` creates result channel + `LazyResultProducer`
    /// 2. Producer is stored here with result channel ID as key
    /// 3. Consumer calls `for (@doc <- resultCh)` which triggers `consume()`
    /// 4. `consume()` detects lazy channel, calls `producer.produce_next()`
    /// 5. Next document is produced to the channel
    /// 6. When exhausted, Nil is produced and entry is removed
    ///
    /// # Lazy Allocation
    /// Wrapped in Option to save memory when not using lazy result channels.
    /// Only allocated when the first similarity query creates a lazy channel.
    lazy_producers: Option<HashMap<Vec<u8>, super::collections::LazyResultProducer<CS::Channel>>>,
}

// =============================================================================
// Type Aliases for Common Collection Types
// =============================================================================
//
// These aliases reduce the 8 type parameters of GenericRSpace to more manageable
// configurations for common use cases.

// Collection type aliases (internal)
type BagDC<A> = super::collections::BagDataCollection<A>;
type BagCC<P, K> = super::collections::BagContinuationCollection<P, K>;
type QueueDC<A> = super::collections::QueueDataCollection<A>;
type QueueCC<P, K> = super::collections::QueueContinuationCollection<P, K>;
type StackDC<A> = super::collections::StackDataCollection<A>;
type StackCC<P, K> = super::collections::StackContinuationCollection<P, K>;
type SetDC<A> = super::collections::SetDataCollection<A>;
type SetCC<P, K> = super::collections::SetContinuationCollection<P, K>;

/// Backward-compatible type alias for GenericRSpace with Bag collections.
///
/// This preserves a convenient type alias for GenericRSpace with a specific
/// channel store that uses Bag collections.
///
/// # Example
/// ```ignore
/// // Using BagRSpace with HashMapChannelStore
/// type MySpace = BagRSpace<HashMapBagChannelStore<C, P, A, K>, ExactMatch<P, A>>;
/// ```
pub type BagRSpace<CS, M> = GenericRSpace<CS, M>;

/// GenericRSpace with FIFO Queue collections.
///
/// Use this for spaces where data and continuation processing order matters.
/// First-in-first-out semantics ensure fair processing.
pub type QueueRSpace<CS, M> = GenericRSpace<CS, M>;

/// GenericRSpace with LIFO Stack collections.
///
/// Use this for spaces where last-in-first-out processing is desired,
/// such as recursive computation or depth-first exploration.
pub type StackRSpace<CS, M> = GenericRSpace<CS, M>;

/// GenericRSpace with Set collections (idempotent operations).
///
/// Use this for spaces where duplicate data is not meaningful.
/// Produces to the same channel with identical data are idempotent.
pub type SetRSpace<CS, M> = GenericRSpace<CS, M>;

// =============================================================================
// Concrete Channel Store + Collection Type Aliases
// =============================================================================
//
// These aliases fix both the channel store and collection types, requiring only
// the matcher, channel, pattern, data, and continuation types.

use super::channel_store::{HashMapChannelStore, PathMapChannelStore, VectorChannelStore};

/// HashMap-based space with Bag collections.
///
/// The most common configuration: O(1) channel lookup with unordered data storage.
///
/// # Type Parameters
/// - `M`: Matcher (e.g., ExactMatch, VectorDBMatch)
/// - `C`: Channel type
/// - `P`: Pattern type
/// - `A`: Data type
/// - `K`: Continuation type
///
/// # Example
/// ```ignore
/// type MySpace = HashMapBagSpace<ExactMatch<Par>, Par, Par, ListParWithRandom, TaggedContinuation>;
/// ```
pub type HashMapBagSpace<M, C, P, A, K> = GenericRSpace<
    HashMapChannelStore<C, P, A, K, BagDC<A>, BagCC<P, K>>,
    M,
>;

/// HashMap-based space with Queue collections (FIFO).
///
/// O(1) channel lookup with FIFO data processing order.
pub type HashMapQueueSpace<M, C, P, A, K> = GenericRSpace<
    HashMapChannelStore<C, P, A, K, QueueDC<A>, QueueCC<P, K>>,
    M,
>;

/// HashMap-based space with Stack collections (LIFO).
///
/// O(1) channel lookup with LIFO data processing order.
pub type HashMapStackSpace<M, C, P, A, K> = GenericRSpace<
    HashMapChannelStore<C, P, A, K, StackDC<A>, StackCC<P, K>>,
    M,
>;

/// HashMap-based space with Set collections (idempotent).
///
/// O(1) channel lookup with idempotent data storage.
pub type HashMapSetSpace<M, C, P, A, K> = GenericRSpace<
    HashMapChannelStore<C, P, A, K, SetDC<A>, SetCC<P, K>>,
    M,
>;

/// PathMap-based space with Bag collections.
///
/// Hierarchical channel addressing with prefix semantics.
/// Channels are `Vec<u8>` paths, enabling prefix aggregation.
///
/// # Example
/// ```ignore
/// // A consume on @[0,1] can match data at @[0,1,2], @[0,1,3], etc.
/// type HierarchicalSpace = PathMapBagSpace<ExactMatch<Par>, Par, ListParWithRandom, TaggedContinuation>;
/// ```
pub type PathMapBagSpace<M, P, A, K> = GenericRSpace<
    PathMapChannelStore<P, A, K, BagDC<A>, BagCC<P, K>>,
    M,
>;

/// PathMap-based space with Queue collections.
pub type PathMapQueueSpace<M, P, A, K> = GenericRSpace<
    PathMapChannelStore<P, A, K, QueueDC<A>, QueueCC<P, K>>,
    M,
>;

/// Vector-based space with Bag collections.
///
/// Uses integer channel indices for O(1) lookup with dense allocation.
/// Best when channels are sequential integers starting from 0.
/// For Rholang integration, use `VectorChannelStore<Par, ...>` directly.
pub type VectorBagSpace<M, C, P, A, K> = GenericRSpace<
    VectorChannelStore<C, P, A, K, BagDC<A>, BagCC<P, K>>,
    M,
>;

/// Vector-based space with Queue collections.
/// For Rholang integration, use `VectorChannelStore<Par, ...>` directly.
pub type VectorQueueSpace<M, C, P, A, K> = GenericRSpace<
    VectorChannelStore<C, P, A, K, QueueDC<A>, QueueCC<P, K>>,
    M,
>;

// =============================================================================
// Prefix Match Result
// =============================================================================

/// Internal struct for tracking match results with prefix semantics.
///
/// When prefix semantics are enabled (e.g., PathMap storage), a consume on
/// channel `@[0,1]` may find data at `@[0,1,2]`. This struct tracks both
/// the consume pattern channel and the actual channel where data was found.
///
/// # Fields
/// - `consume_channel_idx`: Index into the consume channels array
/// - `actual_channel`: The channel where data was actually found
/// - `data`: The matched data
/// - `suffix_key`: The path suffix (empty for exact matches)
/// - `is_peek`: Whether this was a peek operation
#[derive(Clone, Debug)]
struct PrefixMatchResult<C, A> {
    /// Index of the consume channel in the pattern
    consume_channel_idx: usize,
    /// The actual channel where data was found (may differ from consume channel)
    actual_channel: C,
    /// The matched data
    data: A,
    /// Suffix key for prefix matches (empty if exact match)
    suffix_key: SuffixKey,
    /// Whether this was a peek operation
    is_peek: bool,
}

// =============================================================================
// Serialized State
// =============================================================================

/// Serializable representation of GenericRSpace state for checkpointing.
///
/// This struct captures the essential state that needs to be persisted:
/// - All data collections indexed by channel
/// - All continuation collections indexed by channel pattern
/// - All join patterns
/// - The gensym counter for generating unique names
/// - The space qualifier for persistence behavior
/// - The space ID
///
/// # Formal Correspondence
/// - `Checkpoint.v`: checkpoint_preserves_state theorem
/// - `CheckpointReplay.tla`: HardCheckpoint action
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SerializedState<C, P, A, K> {
    /// Data collections: (channel, list of (data, persist_flag))
    pub data: Vec<(C, Vec<(A, bool)>)>,

    /// Continuation collections: (channel_pattern, list of (patterns, continuation, persist_flag))
    pub continuations: Vec<(Vec<C>, Vec<(Vec<P>, K, bool)>)>,

    /// Join patterns: (channel, list of join patterns involving this channel)
    pub joins: Vec<(C, Vec<Vec<C>>)>,

    /// The gensym counter for unique name generation
    pub gensym_counter: usize,

    /// The space qualifier (Default, Temp, Seq)
    pub qualifier: SpaceQualifier,

    /// The space ID
    pub space_id: Vec<u8>,
}

impl<CS, M, C, P, A, K, DC, CC> GenericRSpace<CS, M>
where
    CS: ChannelStore<Channel = C, Pattern = P, Data = A, Continuation = K, DataColl = DC, ContColl = CC>,
    M: Match<P, A>,
    C: Clone + Eq + Hash + Send + Sync,
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync,
    DC: DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: ContinuationCollection<P, K> + Default + Clone + Send + Sync,
{
    /// Create a new GenericRSpace with the given configuration.
    ///
    /// # Arguments
    /// - `channel_store`: The channel store implementation
    /// - `matcher`: The pattern matcher implementation
    /// - `space_id`: Unique identifier for this space
    /// - `qualifier`: Persistence and concurrency behavior
    pub fn new(
        channel_store: CS,
        matcher: M,
        space_id: SpaceId,
        qualifier: SpaceQualifier,
    ) -> Self {
        GenericRSpace {
            channel_store,
            matcher,
            space_id,
            qualifier,
            history_store: None,
            soft_checkpoint: None,
            replay_log: None,
            is_replay: false,
            replay_data: MultisetMultiMap::empty(),
            theory: None,
            soft_checkpoint_stack: Vec::new(),
            similarity_queries: None,  // Lazy allocation - only allocated when needed
            next_continuation_id: std::sync::atomic::AtomicUsize::new(0),
            lazy_producers: None,  // Lazy allocation - only allocated when needed
        }
    }

    /// Create a new GenericRSpace with a history store for checkpointing.
    ///
    /// # Arguments
    /// - `channel_store`: The channel store implementation
    /// - `matcher`: The pattern matcher implementation
    /// - `space_id`: Unique identifier for this space
    /// - `qualifier`: Persistence and concurrency behavior
    /// - `history_store`: The history store for checkpointing
    pub fn with_history(
        channel_store: CS,
        matcher: M,
        space_id: SpaceId,
        qualifier: SpaceQualifier,
        history_store: BoxedHistoryStore,
    ) -> Self {
        GenericRSpace {
            channel_store,
            matcher,
            space_id,
            qualifier,
            history_store: Some(history_store),
            soft_checkpoint: None,
            replay_log: None,
            is_replay: false,
            replay_data: MultisetMultiMap::empty(),
            theory: None,
            soft_checkpoint_stack: Vec::new(),
            similarity_queries: None,  // Lazy allocation - only allocated when needed
            next_continuation_id: std::sync::atomic::AtomicUsize::new(0),
            lazy_producers: None,  // Lazy allocation - only allocated when needed
        }
    }

    /// Set a theory for data validation on this space.
    ///
    /// When a theory is set, all data produced to this space will be validated
    /// against the theory before being stored. Invalid data will be rejected
    /// with a `SpaceError::TheoryValidationError`.
    ///
    /// # Arguments
    /// - `theory`: The theory to validate data against
    ///
    /// # Example
    /// ```ignore
    /// let theory = SimpleTypeTheory::new("NatTheory", vec!["Nat".to_string()]);
    /// space.set_theory(Some(Box::new(theory)));
    /// ```
    pub fn set_theory(&mut self, theory: Option<BoxedTheory>) {
        self.theory = theory;
    }

    /// Get a reference to the theory (if any).
    pub fn theory(&self) -> Option<&BoxedTheory> {
        self.theory.as_ref()
    }

    /// Check if this space has a theory configured.
    pub fn has_theory(&self) -> bool {
        self.theory.is_some()
    }

    /// Validate data against this space's theory.
    ///
    /// If no theory is configured, validation always succeeds.
    /// If a theory is configured, the data is validated against it.
    ///
    /// # Arguments
    /// - `data`: The data to validate, implementing Validatable
    ///
    /// # Returns
    /// - `Ok(())` if validation passes or no theory is configured
    /// - `Err(SpaceError::TheoryValidationError)` if validation fails
    ///
    /// # Formal Correspondence
    /// GenericRSpace.v: `validate_data_sound` theorem
    pub fn validate<V: Validatable>(&self, data: &V) -> Result<(), SpaceError> {
        match &self.theory {
            Some(theory) => theory.validate_data(data),
            None => Ok(()), // No theory = accept everything
        }
    }

    /// Validate data using a string representation.
    ///
    /// This is useful when the data type doesn't implement Validatable,
    /// but you can provide a string representation for validation.
    ///
    /// # Arguments
    /// - `term`: The string representation of the data to validate
    ///
    /// # Returns
    /// - `Ok(())` if validation passes or no theory is configured
    /// - `Err(SpaceError::TheoryValidationError)` if validation fails
    pub fn validate_term(&self, term: &str) -> Result<(), SpaceError> {
        match &self.theory {
            Some(theory) => {
                theory.validate(term).map_err(|validation_error| {
                    SpaceError::TheoryValidationError {
                        theory_name: theory.name().to_string(),
                        validation_error,
                        term: term.to_string(),
                    }
                })
            }
            None => Ok(()), // No theory = accept everything
        }
    }

    /// Get a reference to the matcher.
    pub fn matcher(&self) -> &M {
        &self.matcher
    }

    /// Get a reference to the channel store.
    pub fn channel_store(&self) -> &CS {
        &self.channel_store
    }

    /// Get a mutable reference to the channel store.
    pub fn channel_store_mut(&mut self) -> &mut CS {
        &mut self.channel_store
    }

    /// Check if this space has a history store configured.
    pub fn has_history_store(&self) -> bool {
        self.history_store.is_some()
    }

    /// Get the allocation mode for `new` bindings within this space.
    ///
    /// Delegates to the channel store's allocation mode. Used by `eval_new()`
    /// to determine whether to use random IDs or index-based allocation.
    pub fn allocation_mode(&self) -> super::types::AllocationMode {
        self.channel_store.allocation_mode()
    }

    /// Get the space configuration.
    ///
    /// Returns a configuration reflecting the current space settings.
    /// The theory (if any) is cloned into the returned configuration.
    pub fn config(&self) -> SpaceConfig {
        SpaceConfig {
            outer: super::types::OuterStorageType::HashMap, // Default, would need runtime detection
            data_collection: super::types::InnerCollectionType::Bag,
            continuation_collection: super::types::InnerCollectionType::Bag,
            qualifier: self.qualifier,
            theory: self.theory.as_ref().map(|t| t.clone_box()),
            gas_config: super::types::GasConfiguration::default(), // Gas config tracked by ChargingSpaceAgent wrapper
        }
    }

    // =========================================================================
    // Internal Helper Methods
    // =========================================================================

    /// Find a fireable continuation for the given channel and data, checking similarity requirements.
    ///
    /// This method verifies that ALL channels in the join pattern have matching data
    /// before returning a result. It also checks similarity requirements stored with
    /// continuations. When a continuation was registered with similarity patterns
    /// (via `consume_with_similarity`), this method verifies that the incoming data
    /// meets the similarity threshold before allowing the continuation to fire.
    ///
    /// NOTE: For VectorDB channels with waiting similarity queries, prefer using
    /// `try_produce_with_similarity_matrix()` first, which uses SIMD-optimized
    /// matrix operations for batch similarity computation.
    ///
    /// # Arguments
    /// * `channel` - The channel where data is being produced
    /// * `data` - The data being produced
    ///
    /// # Returns
    /// * `Some((patterns, continuation, persist, join_channels))` - A continuation that can fire
    /// * `None` - No fireable continuation found
    fn find_fireable_continuation_with_similarity(
        &self,
        channel: &C,
        data: &A,
    ) -> Option<(Vec<P>, K, bool, Vec<C>)>
    where
        P: Clone,
        K: Clone,
        A: 'static,
    {
        use std::any::Any;
        use super::collections::{StoredSimilarityInfo, VectorDBDataCollection};

        // Get all join patterns that include this channel
        let joins = self.channel_store.get_joins(channel);

        for join_channels in joins {
            // Get the continuation collection for this join pattern
            if let Some(cont_coll) = self.channel_store.get_continuation_collection(&join_channels)
            {
                // Check each continuation with its similarity info
                for (patterns, cont, persist, similarity_opt) in cont_coll.all_continuations_with_similarity() {
                    // Find which pattern corresponds to our channel
                    if let Some(idx) = join_channels.iter().position(|c| c == channel) {
                        if idx < patterns.len() {
                            // Check if our data matches this pattern
                            if !self.matcher.matches(&patterns[idx], data) {
                                continue;
                            }

                            // Check similarity requirements if present
                            if let Some(similarity_info) = similarity_opt {
                                // Get the similarity requirement for this channel
                                if idx < similarity_info.embeddings.len() {
                                    if let Some((query_embedding, threshold, resolved_metric, _top_k)) = &similarity_info.embeddings[idx] {
                                        // Extract embedding from the incoming data
                                        // First, try to get VectorDB collection for embedding type info
                                        let data_embedding = self.extract_embedding_from_data(channel, data);

                                        if let Ok(data_emb) = data_embedding {
                                            // Use resolved metric if provided, otherwise default to cosine
                                            let similarity = match resolved_metric.as_ref().unwrap_or(&SimilarityMetric::Cosine) {
                                                SimilarityMetric::Cosine => compute_cosine_similarity(query_embedding, &data_emb),
                                                SimilarityMetric::DotProduct => compute_dot_product(query_embedding, &data_emb),
                                                SimilarityMetric::Euclidean => compute_euclidean_similarity(query_embedding, &data_emb),
                                                SimilarityMetric::Manhattan => compute_manhattan_similarity(query_embedding, &data_emb),
                                                SimilarityMetric::Hamming => compute_hamming_similarity(query_embedding, &data_emb),
                                                SimilarityMetric::Jaccard => compute_jaccard_similarity(query_embedding, &data_emb),
                                            };

                                            if similarity < *threshold {
                                                // Similarity not met - skip this continuation
                                                continue;
                                            }
                                        } else {
                                            // Failed to extract embedding - skip this continuation
                                            tracing::warn!(
                                                "find_fireable_continuation_with_similarity: failed to extract embedding from data"
                                            );
                                            continue;
                                        }
                                    }
                                }
                            }

                            // Pattern matches and similarity satisfied - check if all OTHER channels have data
                            let mut all_channels_have_data = true;
                            for (i, ch) in join_channels.iter().enumerate() {
                                if ch == channel {
                                    // This is our channel - we have the data being produced
                                    continue;
                                }
                                if i >= patterns.len() {
                                    continue;
                                }

                                // Check if this other channel has matching data
                                // Note: For joins with multiple channels, we'd also need to check
                                // similarity on those channels. For now, we use standard pattern matching.
                                let pattern = &patterns[i];
                                let has_data = self.channel_store
                                    .get_data_collection(ch)
                                    .map(|dc| dc.peek(|a| self.matcher.matches(pattern, a)).is_some())
                                    .unwrap_or(false);
                                if !has_data {
                                    all_channels_have_data = false;
                                    break;
                                }
                            }

                            // Only return this continuation if all channels have data
                            if all_channels_have_data {
                                return Some((
                                    patterns.to_vec(),
                                    cont.clone(),
                                    persist,
                                    join_channels.clone(),
                                ));
                            }
                            // Otherwise, continue searching for another continuation
                        }
                    }
                }
            }
        }

        None
    }

    /// Extract embedding from data for similarity comparison.
    ///
    /// This helper attempts to extract the embedding from the data being produced,
    /// using the VectorDB collection's configuration to determine the embedding type.
    fn extract_embedding_from_data(&self, channel: &C, data: &A) -> Result<Vec<f32>, SpaceError>
    where
        A: 'static,
    {
        use std::any::Any;
        use super::collections::VectorDBDataCollection;

        // Get the data collection for this channel
        let dc = self.channel_store.get_data_collection(channel).ok_or_else(|| {
            SpaceError::InvalidConfiguration {
                description: "No data collection found for channel".to_string(),
            }
        })?;

        // Downcast to VectorDB collection to get embedding config
        let dc_any: &dyn Any = dc;
        if let Some(vec_dc) = dc_any.downcast_ref::<VectorDBDataCollection<A>>() {
            let embedding_type = vec_dc.embedding_type();
            let dimensions = vec_dc.embedding_dimensions();

            // Try to downcast data to extract embedding
            let data_any: &dyn Any = data;

            // First try: ListParWithRandom (actual type used by RhoISpace)
            if let Some(list_par) = data_any.downcast_ref::<ListParWithRandom>() {
                if let Some(data_par) = list_par.pars.first() {
                    return extract_embedding_from_map(data_par, embedding_type, dimensions);
                }
            }

            // Fallback: direct Par downcast
            if let Some(data_par) = data_any.downcast_ref::<Par>() {
                return extract_embedding_from_map(data_par, embedding_type, dimensions);
            }

            Err(SpaceError::EmbeddingExtractionError {
                description: "Data type not supported for embedding extraction".to_string(),
            })
        } else {
            Err(SpaceError::InvalidConfiguration {
                description: "Channel does not have a VectorDB collection".to_string(),
            })
        }
    }

    /// Try to produce data using the store-first similarity matrix approach.
    ///
    /// This method implements efficient similarity matching by:
    /// 1. Storing the data first in the VectorDB collection
    /// 2. Querying the similarity matrix for matching continuations
    /// 3. Firing the best-matching continuation if found
    ///
    /// This is more efficient than the on-the-fly approach because:
    /// - Data embedding is normalized once during storage
    /// - Similarity is computed via SIMD-optimized matrix-vector multiplication
    /// - All waiting queries are checked in a single batch operation
    ///
    /// # Arguments
    /// - `channel`: The channel to produce on
    /// - `data`: The data being produced
    /// - `persist`: Whether the data persists after matching
    ///
    /// # Returns
    /// - `Some(Ok(...))`: A matching continuation was found and fired
    /// - `Some(Err(...))`: An error occurred during processing
    /// - `None`: No similarity queries for this channel (caller should use fallback)
    fn try_produce_with_similarity_matrix(
        &mut self,
        channel: &C,
        data: A,
        persist: bool,
    ) -> Option<Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>, Produce)>, SpaceError>>
    where
        A: Clone + 'static,
        P: Clone,
        K: Clone,
    {
        use std::any::Any;
        use super::collections::{ContinuationId, SimilarityQueryMatrix, VectorDBDataCollection};

        // Check if this channel has any waiting similarity queries
        // Lazy allocation: return None early if similarity_queries hasn't been allocated
        let query_matrix = self.similarity_queries.as_ref()?.get(channel)?;
        if query_matrix.is_empty() {
            return None;
        }

        // Get the VectorDB data collection for this channel
        let dc = self.channel_store.get_or_create_data_collection(channel);
        let dc_any: &mut dyn Any = dc;
        let vec_dc = dc_any.downcast_mut::<VectorDBDataCollection<A>>()?;

        // Extract embedding from data
        let embedding_type = vec_dc.embedding_type();
        let dimensions = vec_dc.embedding_dimensions();

        // Try to extract embedding from data
        let data_any: &dyn Any = &data;
        let embedding = if let Some(list_par) = data_any.downcast_ref::<ListParWithRandom>() {
            if let Some(data_par) = list_par.pars.first() {
                match extract_embedding_from_map(data_par, embedding_type, dimensions) {
                    Ok(emb) => emb,
                    Err(e) => return Some(Err(e)),
                }
            } else {
                return Some(Err(SpaceError::InvalidConfiguration {
                    description: "ListParWithRandom contains no Par values for embedding extraction".to_string(),
                }));
            }
        } else if let Some(data_par) = data_any.downcast_ref::<Par>() {
            match extract_embedding_from_map(data_par, embedding_type, dimensions) {
                Ok(emb) => emb,
                Err(e) => return Some(Err(e)),
            }
        } else {
            // Not a VectorDB-compatible data type
            return None;
        };

        // Store data first and get the index
        let data_idx = match vec_dc.put_with_embedding_returning_index(data.clone(), embedding, persist) {
            Ok(idx) => idx,
            Err(e) => return Some(Err(e)),
        };

        // Get the normalized embedding we just stored
        let normalized_embedding = match vec_dc.get_normalized_embedding(data_idx) {
            Some(emb) => emb,
            None => {
                return Some(Err(SpaceError::InvalidConfiguration {
                    description: "Failed to retrieve normalized embedding after storage".to_string(),
                }));
            }
        };

        // Query the similarity matrix for matching continuations (SIMD-optimized)
        // Re-borrow query_matrix since we dropped the mutable borrow above
        let query_matrix = self.similarity_queries.as_ref()?.get(channel)?;
        let matches = query_matrix.find_matching_queries(&normalized_embedding);

        if matches.is_empty() {
            // No matching continuation found - data stays in VectorDB
            return Some(Ok(None));
        }

        // Find the best match (highest similarity score)
        let (cont_id, channel_idx, similarity, cont_persist) = matches
            .into_iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))?;

        // Get the continuation from the continuation collection
        // For single-channel queries, the join pattern is just [channel]
        let join_channels = vec![channel.clone()];

        // Find and retrieve the continuation (will be removed if not persistent)
        let cont_coll = self.channel_store.get_continuation_collection_mut(&join_channels)?;
        let (patterns, cont, _persist, _similarity_info) = if cont_persist {
            // Persistent - just peek, don't remove
            let all = cont_coll.all_continuations_with_similarity();
            // Find by ContinuationId - we need to match the correct one
            // For now, just take the first matching one (we'll refine this later)
            // Note: all.first() returns Option<&(...)> so we need (*k).clone() to get K (owned)
            all.first().map(|(p, k, persist, sim)| (p.to_vec(), (*k).clone(), *persist, sim.cloned()))?
        } else {
            // Non-persistent - remove the continuation
            cont_coll.find_and_remove_with_similarity(|_p, _k| true)?
        };

        // Remove data from VectorDB if not persistent
        if !persist {
            // Re-borrow as mutable
            let dc = self.channel_store.get_data_collection_mut(channel)?;
            let dc_any: &mut dyn Any = dc;
            if let Some(vec_dc) = dc_any.downcast_mut::<VectorDBDataCollection<A>>() {
                vec_dc.remove_by_index(data_idx);
            }
        }

        // Remove the query from the similarity matrix if not persistent
        if !cont_persist {
            if let Some(query_matrix) = self.similarity_queries.as_mut().and_then(|m| m.get_mut(channel)) {
                query_matrix.remove_query(cont_id);
            }

            // Remove join pattern if no more continuations
            let should_remove_join = self.channel_store
                .get_continuation_collection(&join_channels)
                .map(|cc| cc.is_empty())
                .unwrap_or(true);

            if should_remove_join {
                self.channel_store.remove_join(&join_channels);
            }
        }

        // Build the result
        let matched_data = vec![RSpaceResult {
            channel: channel.clone(),
            matched_datum: data.clone(),
            removed_datum: data,
            persistent: persist,
            suffix_key: None,
        }];

        let cont_result = ContResult {
            continuation: cont,
            persistent: cont_persist,
            peek: false,
            channels: join_channels,
            patterns,
        };

        Some(Ok(Some((cont_result, matched_data, Produce::default()))))
    }

    /// Find a matching continuation at a prefix path for the given channel and data.
    ///
    /// When prefix semantics are enabled (PathMap storage), this method searches
    /// for continuations waiting at prefix paths of the produce channel.
    ///
    /// # Example
    /// If produce is on `@[0,1,2]`, this will check for continuations at:
    /// - `@[0,1]` (immediate prefix)
    /// - `@[0]` (longer prefix)
    /// - `@[]` (root prefix, if applicable)
    ///
    /// # Returns
    /// - `Some((patterns, continuation, persist, join_channels, prefix_channel, match_idx))`:
    ///   The matching continuation with the prefix path where it was found
    /// - `None`: No matching continuation found at any prefix
    ///
    /// # Formal Correspondence
    /// - `PathMapStore.v`: `produce_finds_prefix_continuation` theorem
    /// Find a matching continuation at a prefix path for the given channel and data.
    ///
    /// # Returns
    /// - `Some((patterns, continuation, persist, join_channels, prefix_channel, match_idx, suffix_key))`:
    ///   The matching continuation with the prefix path and computed suffix key
    /// - `None`: No matching continuation found at any prefix
    fn find_matching_continuation_at_prefix(
        &self,
        channel: &C,
        data: &A,
    ) -> Option<(Vec<P>, K, bool, Vec<C>, C, usize, SuffixKey)>
    where
        P: Clone,
        K: Clone,
        C: AsRef<[u8]>,
    {
        if !self.channel_store.supports_prefix_semantics() {
            return None;
        }

        // Get all prefix paths for the produce channel
        let prefixes = self.channel_store.channel_prefixes(channel);

        for prefix in prefixes {
            // Skip the exact channel (already checked by find_matching_continuation)
            if &prefix == channel {
                continue;
            }

            // First, check for single-channel continuations at the prefix
            // Single-channel consumes like `for (@x <- @[0,1])` don't register joins,
            // they just store a continuation at [prefix] directly.
            let single_channel_key = vec![prefix.clone()];
            if let Some(cont_coll) = self.channel_store.get_continuation_collection(&single_channel_key) {
                for (patterns, cont, persist) in cont_coll.all_continuations() {
                    if !patterns.is_empty() {
                        // Check if our data matches this pattern
                        if self.matcher.matches(&patterns[0], data) {
                            // Compute suffix key: difference between produce channel and prefix
                            // E.g., channel=[0,1,2], prefix=[0,1] -> suffix=[2]
                            let suffix_key = get_path_suffix(prefix.as_ref(), channel.as_ref())
                                .unwrap_or_default();
                            return Some((
                                patterns.to_vec(),
                                cont.clone(),
                                persist,
                                single_channel_key,
                                prefix.clone(),
                                0,  // idx is always 0 for single-channel
                                suffix_key,
                            ));
                        }
                    }
                }
            }

            // Then, check join patterns that include this prefix channel
            let joins = self.channel_store.get_joins(&prefix);

            for join_channels in joins {
                // Get the continuation collection for this join pattern
                if let Some(cont_coll) = self.channel_store.get_continuation_collection(&join_channels)
                {
                    // Check each continuation
                    for (patterns, cont, persist) in cont_coll.all_continuations() {
                        // Find which pattern corresponds to the prefix channel
                        if let Some(idx) = join_channels.iter().position(|c| c == &prefix) {
                            if idx < patterns.len() {
                                // Check if our data matches this pattern
                                if self.matcher.matches(&patterns[idx], data) {
                                    // Compute suffix key: difference between produce channel and prefix
                                    // E.g., channel=[0,1,2], prefix=[0,1] -> suffix=[2]
                                    let suffix_key = get_path_suffix(prefix.as_ref(), channel.as_ref())
                                        .unwrap_or_default();
                                    return Some((
                                        patterns.to_vec(),
                                        cont.clone(),
                                        persist,
                                        join_channels.clone(),
                                        prefix.clone(),
                                        idx,
                                        suffix_key,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Check if all channels in a join have matching data.
    fn check_all_channels_match(
        &self,
        channels: &[C],
        patterns: &[P],
        peeks: &BTreeSet<i32>,
    ) -> Option<Vec<A>>
    where
        A: Clone,
    {
        let mut matched_data = Vec::with_capacity(channels.len());

        for (i, (ch, pattern)) in channels.iter().zip(patterns.iter()).enumerate() {
            let is_peek = peeks.contains(&(i as i32));

            if let Some(data_coll) = self.channel_store.get_data_collection(ch) {
                // Try to find matching data
                let matcher = &self.matcher;
                let found = if is_peek {
                    data_coll.peek(|a| matcher.matches(pattern, a)).cloned()
                } else {
                    // For non-peek, we need the data but can't remove yet
                    // (we'll remove in a second pass if all channels match)
                    data_coll.peek(|a| matcher.matches(pattern, a)).cloned()
                };

                if let Some(data) = found {
                    matched_data.push(data);
                } else {
                    return None; // This channel has no matching data
                }
            } else {
                return None; // No data collection for this channel
            }
        }

        Some(matched_data)
    }

    /// Remove matched data from all channels (for successful consume).
    fn remove_matched_data(
        &mut self,
        channels: &[C],
        patterns: &[P],
        peeks: &BTreeSet<i32>,
    ) -> Vec<A>
    where
        A: Clone,
    {
        let mut removed = Vec::with_capacity(channels.len());

        for (i, (ch, pattern)) in channels.iter().zip(patterns.iter()).enumerate() {
            let is_peek = peeks.contains(&(i as i32));

            if let Some(data_coll) = self.channel_store.get_data_collection_mut(ch) {
                if is_peek {
                    // For peek, just get the data without removing
                    if let Some(data) = data_coll.peek(|a| self.matcher.matches(pattern, a)) {
                        removed.push(data.clone());
                    }
                } else {
                    // For non-peek, remove the data
                    if let Some(data) =
                        data_coll.find_and_remove(|a| self.matcher.matches(pattern, a))
                    {
                        removed.push(data);
                    }
                }
            }
        }

        removed
    }

    /// Atomically check and remove matching data from all channels in a single pass.
    ///
    /// This combines `check_all_channels_match` and `remove_matched_data` into a
    /// single atomic operation, eliminating the TOCTOU race condition and improving
    /// performance by avoiding the second pass.
    ///
    /// # Algorithm
    /// 1. For each channel, atomically find and remove matching data (or peek for peek channels)
    /// 2. If any channel fails to match, rollback all previously removed data
    /// 3. Return None if rollback occurred, Some(data) if all channels matched
    ///
    /// # Performance
    /// - Single pass instead of two passes
    /// - O(n) where n is the number of channels
    /// - Uses SmartDataStorage's O(1) swap_remove in Eager mode
    ///
    /// # Formal Correspondence
    /// - Implements atomic consume semantics from `GenericRSpace.v`
    /// - Ensures linearizability: either all channels match or none are modified
    fn check_and_remove_matched_data_atomic(
        &mut self,
        channels: &[C],
        patterns: &[P],
        peeks: &BTreeSet<i32>,
    ) -> Option<Vec<A>>
    where
        A: Clone,
    {
        let mut matched_data: Vec<A> = Vec::with_capacity(channels.len());
        let mut removed_from: Vec<(usize, C)> = Vec::with_capacity(channels.len());

        for (i, (ch, pattern)) in channels.iter().zip(patterns.iter()).enumerate() {
            let is_peek = peeks.contains(&(i as i32));

            if let Some(data_coll) = self.channel_store.get_data_collection_mut(ch) {
                let matcher = &self.matcher;

                if is_peek {
                    // For peek, just find without removing
                    if let Some(data) = data_coll.peek(|a| matcher.matches(pattern, a)) {
                        matched_data.push(data.clone());
                    } else {
                        // No match - rollback and return None
                        self.rollback_removed_data(&removed_from, &matched_data, peeks);
                        return None;
                    }
                } else {
                    // Atomic find and remove
                    if let Some(data) = data_coll.find_and_remove(|a| matcher.matches(pattern, a)) {
                        matched_data.push(data);
                        removed_from.push((i, ch.clone()));
                    } else {
                        // No match - rollback and return None
                        self.rollback_removed_data(&removed_from, &matched_data, peeks);
                        return None;
                    }
                }
            } else {
                // No data collection for this channel - rollback and return None
                self.rollback_removed_data(&removed_from, &matched_data, peeks);
                return None;
            }
        }

        Some(matched_data)
    }

    /// Rollback helper: put back removed data on failure.
    ///
    /// Called when atomic matching fails mid-operation to restore the space
    /// to its previous state.
    #[inline]
    fn rollback_removed_data(
        &mut self,
        removed_from: &[(usize, C)],
        matched_data: &[A],
        peeks: &BTreeSet<i32>,
    ) where
        A: Clone,
    {
        for (idx, (i, ch)) in removed_from.iter().enumerate() {
            // Only rollback non-peek removes (peeks didn't modify anything)
            if !peeks.contains(&(*i as i32)) {
                if let Some(data_coll) = self.channel_store.get_data_collection_mut(ch) {
                    // Best effort rollback - ignore errors
                    let _ = data_coll.put(matched_data[idx].clone());
                }
            }
        }
    }

    // =========================================================================
    // Prefix-Aware Matching Methods
    // =========================================================================

    /// Check if all channels have matching data, including prefix descendants.
    ///
    /// When prefix semantics are enabled, this method searches not just the
    /// exact channels but also all descendant channels (paths that have the
    /// consume channel as a prefix).
    ///
    /// # Example
    /// For PathMap with consume on `@[0,1]`, this will find data at:
    /// - `@[0,1]` (exact match)
    /// - `@[0,1,2]` (descendant with suffix `[2]`)
    /// - `@[0,1,3,4]` (descendant with suffix `[3,4]`)
    ///
    /// # Formal Correspondence
    /// - `PathMapStore.v`: `consume_finds_descendant_data` theorem
    fn check_all_channels_match_with_prefix(
        &self,
        channels: &[C],
        patterns: &[P],
        peeks: &BTreeSet<i32>,
    ) -> Option<Vec<PrefixMatchResult<C, A>>>
    where
        A: Clone,
    {
        let mut results = Vec::with_capacity(channels.len());

        for (i, (ch, pattern)) in channels.iter().zip(patterns.iter()).enumerate() {
            let is_peek = peeks.contains(&(i as i32));

            // First, try exact match on the consume channel
            if let Some(data_coll) = self.channel_store.get_data_collection(ch) {
                if let Some(data) = data_coll.peek(|a| self.matcher.matches(pattern, a)) {
                    results.push(PrefixMatchResult {
                        consume_channel_idx: i,
                        actual_channel: ch.clone(),
                        data: data.clone(),
                        suffix_key: Vec::new(), // Empty suffix = exact match
                        is_peek,
                    });
                    continue;
                }
            }

            // If no exact match and prefix semantics enabled, check descendants
            if self.channel_store.supports_prefix_semantics() {
                let descendants = self.channel_store.channels_with_prefix(ch);
                let mut found = false;

                for descendant in descendants {
                    // Skip the exact channel (already checked above)
                    if &descendant == ch {
                        continue;
                    }

                    if let Some(data_coll) = self.channel_store.get_data_collection(&descendant) {
                        if let Some(data) = data_coll.peek(|a| self.matcher.matches(pattern, a)) {
                            // Compute suffix key using the channel store's implementation
                            // For PathMap stores, this converts channels to paths and computes the suffix
                            let suffix_key = self.channel_store.compute_suffix_key(ch, &descendant)
                                .unwrap_or_default();

                            results.push(PrefixMatchResult {
                                consume_channel_idx: i,
                                actual_channel: descendant.clone(),
                                data: data.clone(),
                                suffix_key,
                                is_peek,
                            });
                            found = true;
                            break; // Found a match for this channel
                        }
                    }
                }

                if !found {
                    return None; // No match for this channel
                }
            } else {
                return None; // No exact match and no prefix semantics
            }
        }

        Some(results)
    }

    /// Check if all channels have matching data with suffix key computation.
    ///
    /// This is the specialized version for PathMap stores where the channel
    /// type is `Vec<u8>` and suffix keys can be computed.
    ///
    /// # Formal Correspondence
    /// - `PathMapStore.v`: `consume_computes_suffix_key` theorem
    #[allow(dead_code)] // May be needed for PathMap prefix semantics integration
    fn check_all_channels_match_with_suffix(
        &self,
        channels: &[C],
        patterns: &[P],
        peeks: &BTreeSet<i32>,
    ) -> Option<Vec<PrefixMatchResult<C, A>>>
    where
        A: Clone,
        C: AsRef<[u8]>,
    {
        let mut results = Vec::with_capacity(channels.len());

        for (i, (ch, pattern)) in channels.iter().zip(patterns.iter()).enumerate() {
            let is_peek = peeks.contains(&(i as i32));

            // First, try exact match on the consume channel
            if let Some(data_coll) = self.channel_store.get_data_collection(ch) {
                if let Some(data) = data_coll.peek(|a| self.matcher.matches(pattern, a)) {
                    results.push(PrefixMatchResult {
                        consume_channel_idx: i,
                        actual_channel: ch.clone(),
                        data: data.clone(),
                        suffix_key: Vec::new(), // Empty suffix = exact match
                        is_peek,
                    });
                    continue;
                }
            }

            // If no exact match and prefix semantics enabled, check descendants
            if self.channel_store.supports_prefix_semantics() {
                let descendants = self.channel_store.channels_with_prefix(ch);
                let mut found = false;

                for descendant in descendants {
                    // Skip the exact channel (already checked above)
                    if &descendant == ch {
                        continue;
                    }

                    if let Some(data_coll) = self.channel_store.get_data_collection(&descendant) {
                        if let Some(data) = data_coll.peek(|a| self.matcher.matches(pattern, a)) {
                            // Compute suffix key
                            let suffix = get_path_suffix(ch.as_ref(), descendant.as_ref())
                                .unwrap_or_default();

                            results.push(PrefixMatchResult {
                                consume_channel_idx: i,
                                actual_channel: descendant.clone(),
                                data: data.clone(),
                                suffix_key: suffix,
                                is_peek,
                            });
                            found = true;
                            break; // Found a match for this channel
                        }
                    }
                }

                if !found {
                    return None; // No match for this channel
                }
            } else {
                return None; // No exact match and no prefix semantics
            }
        }

        Some(results)
    }

    /// Remove matched data based on prefix match results.
    ///
    /// Unlike `remove_matched_data`, this method removes data from the
    /// actual channels where matches were found, which may differ from
    /// the consume pattern channels when prefix semantics are enabled.
    ///
    /// # Formal Correspondence
    /// - `PathMapStore.v`: `consume_removes_from_actual_path` theorem
    #[allow(dead_code)] // Kept for reference; replaced by atomic version
    fn remove_matched_data_from_prefix_results(
        &mut self,
        results: &[PrefixMatchResult<C, A>],
        patterns: &[P],
    ) -> Vec<(A, SuffixKey)>
    where
        A: Clone,
    {
        let mut removed = Vec::with_capacity(results.len());

        for result in results {
            if result.is_peek {
                // For peek, return data without removing
                removed.push((result.data.clone(), result.suffix_key.clone()));
            } else {
                // Remove from the actual channel where data was found
                if let Some(data_coll) = self.channel_store.get_data_collection_mut(&result.actual_channel) {
                    let pattern = &patterns[result.consume_channel_idx];
                    if let Some(data) = data_coll.find_and_remove(|a| self.matcher.matches(pattern, a)) {
                        removed.push((data, result.suffix_key.clone()));
                    }
                }
            }
        }

        removed
    }

    /// Atomic prefix-aware consume that fuses find and remove operations.
    ///
    /// This method solves the TOCTOU race condition in prefix-aware consume where
    /// concurrent consumers with overlapping prefixes (e.g., @[0] and @[0,1]) could
    /// both peek the same data at a descendant channel (@[0,1,2]), but only one
    /// would successfully remove it, leaving the other with incomplete results.
    ///
    /// # Algorithm
    ///
    /// For each consume channel:
    /// 1. Try exact match first using `find_and_remove` (atomic)
    /// 2. If no exact match and prefix semantics enabled, try descendants
    /// 3. If removal fails mid-operation (race detected), rollback and return None
    ///
    /// When this method returns `None`, the caller should store the continuation
    /// as a wait pattern rather than firing with incomplete bindings.
    ///
    /// # Formal Correspondence
    /// - `PathMapStore.v`: `consume_atomic` lemma
    /// - Fixes TOCTOU race between `check_all_channels_match_with_prefix` and
    ///   `remove_matched_data_from_prefix_results`
    fn check_and_consume_with_prefix_atomic(
        &mut self,
        channels: &[C],
        patterns: &[P],
        peeks: &BTreeSet<i32>,
    ) -> Option<Vec<(C, A, SuffixKey)>>
    where
        A: Clone,
    {
        let mut results: Vec<(C, A, SuffixKey)> = Vec::with_capacity(channels.len());
        let mut removed_channels: Vec<C> = Vec::with_capacity(channels.len());

        for (i, (ch, pattern)) in channels.iter().zip(patterns.iter()).enumerate() {
            let is_peek = peeks.contains(&(i as i32));

            // Track whether we found a match for this channel
            let mut found = false;

            // First, try exact match on the consume channel
            if let Some(data_coll) = self.channel_store.get_data_collection_mut(ch) {
                if is_peek {
                    // For peek, just check if matching data exists
                    if let Some(data) = data_coll.peek(|a| self.matcher.matches(pattern, a)) {
                        results.push((ch.clone(), data.clone(), Vec::new()));
                        found = true;
                    }
                } else {
                    // Atomic find and remove for exact match
                    if let Some(data) = data_coll.find_and_remove(|a| self.matcher.matches(pattern, a)) {
                        results.push((ch.clone(), data, Vec::new()));
                        removed_channels.push(ch.clone());
                        found = true;
                    }
                }
            }

            // If no exact match and prefix semantics enabled, check descendants
            if !found && self.channel_store.supports_prefix_semantics() {
                let descendants = self.channel_store.channels_with_prefix(ch);

                for descendant in descendants {
                    // Skip the exact channel (already checked above)
                    if &descendant == ch {
                        continue;
                    }

                    if is_peek {
                        // For peek, just check if matching data exists
                        if let Some(data_coll) = self.channel_store.get_data_collection(&descendant) {
                            if let Some(data) = data_coll.peek(|a| self.matcher.matches(pattern, a)) {
                                let suffix_key = self.channel_store.compute_suffix_key(ch, &descendant)
                                    .unwrap_or_default();
                                results.push((descendant.clone(), data.clone(), suffix_key));
                                found = true;
                                break;
                            }
                        }
                    } else {
                        // Atomic find and remove for prefix match
                        if let Some(data_coll) = self.channel_store.get_data_collection_mut(&descendant) {
                            if let Some(data) = data_coll.find_and_remove(|a| self.matcher.matches(pattern, a)) {
                                let suffix_key = self.channel_store.compute_suffix_key(ch, &descendant)
                                    .unwrap_or_default();
                                results.push((descendant.clone(), data, suffix_key));
                                removed_channels.push(descendant.clone());
                                found = true;
                                break;
                            }
                        }
                    }
                }
            }

            // If no match found for this channel, we must rollback and return None
            if !found {
                // Rollback: put back all removed data
                for (idx, (removed_ch, removed_data, _)) in results.into_iter().enumerate() {
                    // Only rollback non-peek removes
                    if !peeks.contains(&(idx as i32)) {
                        if let Some(data_coll) = self.channel_store.get_data_collection_mut(&removed_ch) {
                            // Ignore errors during rollback - best effort
                            let _ = data_coll.put(removed_data);
                        }
                    }
                }
                return None;
            }
        }

        Some(results)
    }

    /// Serialize the current state for checkpointing.
    ///
    /// Exports all data, continuations, joins, and the gensym counter to a
    /// serializable format, then encodes it using bincode.
    ///
    /// # Type Requirements
    /// - `C`, `P`, `A`, `K` must implement `Serialize`
    ///
    /// # Formal Correspondence
    /// - `Checkpoint.v`: checkpoint_preserves_state theorem
    /// - `CheckpointReplay.tla`: HardCheckpoint action
    ///
    /// # Returns
    /// The serialized state as bytes, or an empty vector if serialization fails.
    fn serialize_state(&self) -> Vec<u8>
    where
        C: Serialize,
        P: Serialize,
        A: Serialize,
        K: Serialize,
    {
        // Export data from channel store
        let exported_data = self.channel_store.export_data();
        let data: Vec<(C, Vec<(A, bool)>)> = exported_data
            .into_iter()
            .map(|(channel, data_collection)| {
                // Convert data collection to serializable format
                // Each item in BagDataCollection is just the data itself
                let items: Vec<(A, bool)> = data_collection
                    .all_data()
                    .into_iter()
                    .map(|a| (a.clone(), false)) // Default persist = false
                    .collect();
                (channel, items)
            })
            .collect();

        // Export continuations from channel store
        let exported_conts = self.channel_store.export_continuations();
        let continuations: Vec<(Vec<C>, Vec<(Vec<P>, K, bool)>)> = exported_conts
            .into_iter()
            .map(|(channels, cont_collection)| {
                // Convert continuation collection to serializable format
                let items: Vec<(Vec<P>, K, bool)> = cont_collection
                    .all_continuations()
                    .into_iter()
                    .map(|(patterns, k, persist)| (patterns.to_vec(), k.clone(), persist))
                    .collect();
                (channels, items)
            })
            .collect();

        // Export joins from channel store
        let joins = self.channel_store.export_joins();

        // Create serialized state
        let state = SerializedState {
            data,
            continuations,
            joins,
            gensym_counter: self.channel_store.gensym_counter(),
            qualifier: self.qualifier,
            space_id: self.space_id.0.clone(),
        };

        // Serialize using bincode
        bincode::serialize(&state).unwrap_or_else(|e| {
            tracing::error!("Failed to serialize space state: {}", e);
            Vec::new()
        })
    }

    /// Deserialize state from bytes.
    ///
    /// Restores data, continuations, joins, and the gensym counter from the
    /// serialized state.
    ///
    /// # Type Requirements
    /// - `C`, `P`, `A`, `K` must implement `DeserializeOwned`
    ///
    /// # Formal Correspondence
    /// - `Checkpoint.v`: replay_restores_state theorem
    /// - `CheckpointReplay.tla`: Replay action
    ///
    /// # Arguments
    /// - `state`: The serialized state bytes
    ///
    /// # Errors
    /// Returns `SpaceError::DeserializationError` if the bytes cannot be decoded.
    fn deserialize_state(&mut self, state: &[u8]) -> Result<(), SpaceError>
    where
        C: DeserializeOwned,
        P: DeserializeOwned,
        A: DeserializeOwned,
        K: DeserializeOwned,
    {
        // Deserialize using bincode
        let serialized: SerializedState<C, P, A, K> = bincode::deserialize(state)
            .map_err(|e| SpaceError::DeserializationError {
                message: format!("Failed to deserialize space state: {}", e),
            })?;

        // Validate space ID matches (optional, for safety)
        if serialized.space_id != self.space_id.0 {
            return Err(SpaceError::DeserializationError {
                message: format!(
                    "Space ID mismatch: expected {:?}, got {:?}",
                    self.space_id.0, serialized.space_id
                ),
            });
        }

        // Import data into channel store
        // Convert serialized format back to the appropriate DataCollection type
        let data_collections: Vec<(C, DC)> = serialized
            .data
            .into_iter()
            .map(|(channel, items)| {
                let mut collection = DC::default();
                for (data, _persist) in items {
                    let _ = collection.put(data);
                }
                (channel, collection)
            })
            .collect();
        self.channel_store.import_data(data_collections);

        // Import continuations into channel store
        // Convert serialized format back to the appropriate ContinuationCollection type
        let cont_collections: Vec<(Vec<C>, CC)> = serialized
            .continuations
            .into_iter()
            .map(|(channels, items)| {
                let mut collection = CC::default();
                for (patterns, continuation, persist) in items {
                    collection.put(patterns, continuation, persist);
                }
                (channels, collection)
            })
            .collect();
        self.channel_store.import_continuations(cont_collections);

        // Import joins
        self.channel_store.import_joins(serialized.joins);

        // Restore gensym counter - use max to prevent name collisions
        // with channels that were generated after the checkpoint.
        // This ensures restored state never reuses channel names that
        // were temporarily in use before the rollback.
        let current_counter = self.channel_store.gensym_counter();
        self.channel_store.set_gensym_counter(std::cmp::max(current_counter, serialized.gensym_counter));

        // Restore qualifier
        self.qualifier = serialized.qualifier;

        Ok(())
    }
}

// =============================================================================
// Clone Implementation
// =============================================================================

impl<CS, M, C, P, A, K, DC, CC> Clone for GenericRSpace<CS, M>
where
    CS: ChannelStore<Channel = C, Pattern = P, Data = A, Continuation = K, DataColl = DC, ContColl = CC> + Clone,
    M: Match<P, A> + Clone,
    C: Clone + Eq + Hash + Send + Sync + 'static,
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync,
    DC: DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: ContinuationCollection<P, K> + Default + Clone + Send + Sync,
{
    fn clone(&self) -> Self {
        GenericRSpace {
            channel_store: self.channel_store.clone(),
            matcher: self.matcher.clone(),
            space_id: self.space_id.clone(),
            qualifier: self.qualifier,
            history_store: self.history_store.clone(),
            soft_checkpoint: self.soft_checkpoint.clone(),
            replay_log: self.replay_log.clone(),
            is_replay: self.is_replay,
            replay_data: self.replay_data.clone(),
            theory: self.theory.as_ref().map(|t| t.clone_box()),
            soft_checkpoint_stack: self.soft_checkpoint_stack.clone(),
            similarity_queries: self.similarity_queries.clone(),
            next_continuation_id: std::sync::atomic::AtomicUsize::new(
                self.next_continuation_id.load(std::sync::atomic::Ordering::Relaxed)  // Relaxed sufficient for clone
            ),
            lazy_producers: self.lazy_producers.clone(),
        }
    }
}

// =============================================================================
// Debug Implementation
// =============================================================================

impl<CS, M, C, P, A, K, DC, CC> Debug for GenericRSpace<CS, M>
where
    CS: ChannelStore<Channel = C, Pattern = P, Data = A, Continuation = K, DataColl = DC, ContColl = CC> + Debug,
    M: Match<P, A> + Debug,
    C: Clone + Eq + Hash + Send + Sync + Debug + 'static,
    P: Clone + Send + Sync + Debug,
    A: Clone + Send + Sync + std::fmt::Debug,
    K: Clone + Send + Sync + Debug,
    DC: DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: ContinuationCollection<P, K> + Default + Clone + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenericRSpace")
            .field("space_id", &self.space_id)
            .field("qualifier", &self.qualifier)
            .field("has_history_store", &self.history_store.is_some())
            .field("is_replay", &self.is_replay)
            .field("matcher", &self.matcher.matcher_name())
            .field("theory", &self.theory.as_ref().map(|t| t.name()))
            .finish()
    }
}

// =============================================================================
// SpaceAgent Implementation
// =============================================================================

impl<CS, M, C, P, A, K, DC, CC> SpaceAgent<C, P, A, K> for GenericRSpace<CS, M>
where
    CS: ChannelStore<Channel = C, Pattern = P, Data = A, Continuation = K, DataColl = DC, ContColl = CC>,
    M: Match<P, A>,
    C: Clone + Eq + Hash + Send + Sync + AsRef<[u8]> + 'static,
    P: Clone + PartialEq + Send + Sync + 'static,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync + 'static,
    DC: DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: ContinuationCollection<P, K> + Default + Clone + Send + Sync,
{
    fn space_id(&self) -> &SpaceId {
        &self.space_id
    }

    fn qualifier(&self) -> SpaceQualifier {
        self.qualifier
    }

    fn gensym(&mut self) -> Result<C, SpaceError> {
        self.channel_store.gensym(&self.space_id)
    }

    fn produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
        priority: Option<usize>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>, Produce)>, SpaceError> {
        // Store priority for later use when storing data
        let _priority = priority;

        // Validate data against theory before storing
        if let Some(ref theory) = self.theory {
            // Use Validatable trait for proper type-aware validation if available.
            // For ListParWithRandom, this produces strings like "Nat(42)" or "String(hello)"
            // that the theory can properly validate.
            use std::any::TypeId;
            use models::rhoapi::ListParWithRandom;
            use super::types::Validatable;

            let term = if TypeId::of::<A>() == TypeId::of::<ListParWithRandom>() {
                // SAFETY: We've verified the type matches
                let data_ref: &A = &data;
                let ptr = data_ref as *const A as *const ListParWithRandom;
                let lp = unsafe { &*ptr };
                lp.to_validatable_string()
            } else {
                // Fallback to Debug format for other types
                format!("{:?}", data)
            };

            self.validate_term(&term)?;
        }

        // 0. Try the efficient matrix-based approach first for VectorDB similarity queries
        //
        // This uses the store-first architecture with SIMD-optimized matrix operations:
        // 1. Store data in VectorDB (normalizes embedding, adds to matrix)
        // 2. Query similarity matrix for matching continuations
        // 3. Fire best match if found, otherwise data stays stored
        //
        // Returns:
        // - Some(result): Handled by matrix approach (success or error)
        // - None: No similarity queries for this channel, use standard produce
        if let Some(result) = self.try_produce_with_similarity_matrix(&channel, data.clone(), persist) {
            return result;
        }

        // 1. Check for a continuation that can actually fire (all join channels have data)
        // This uses find_fireable_continuation_with_similarity which:
        // - Checks data availability on all channels
        // - Verifies similarity requirements for VectorDB queries
        if let Some((patterns, cont, cont_persist, join_channels)) =
            self.find_fireable_continuation_with_similarity(&channel, &data)
        {
            // 2. Found a fireable match - remove data from all channels
            // SmallVec: most joins have ≤4 channels, avoiding heap allocation
            let mut matched_data: SmallVec<[RSpaceResult<C, A>; 4]> = SmallVec::new();
            for (i, ch) in join_channels.iter().enumerate() {
                if ch == &channel {
                    // This is our channel - use the data we're producing
                    matched_data.push(RSpaceResult {
                        channel: channel.clone(),
                        matched_datum: data.clone(),
                        removed_datum: data.clone(),
                        persistent: persist,
                        suffix_key: None, // Exact match
                    });
                } else if i < patterns.len() {
                    // Another channel in the join - remove matching data
                    if let Some(dc) = self.channel_store.get_data_collection_mut(ch) {
                        let pattern = &patterns[i];
                        if let Some(found_data) =
                            dc.find_and_remove(|a| self.matcher.matches(pattern, a))
                        {
                            matched_data.push(RSpaceResult {
                                channel: ch.clone(),
                                matched_datum: found_data.clone(),
                                removed_datum: found_data,
                                persistent: false, // Retrieved data is not persistent
                                suffix_key: None, // Exact match
                            });
                        }
                    }
                }
            }

            // 3. Remove the continuation if not persistent
            if !cont_persist {
                let should_remove_join = if let Some(cc) = self.channel_store.get_continuation_collection_mut(&join_channels)
                {
                    // Note: We compare patterns only since cont is a clone and pointer comparison won't work
                    cc.find_and_remove(|p, _k| p == patterns.as_slice());
                    // Only remove join if no more continuations remain for this join pattern
                    cc.is_empty()
                } else {
                    true // No collection means join can be removed
                };

                // Only remove join pattern if no more continuations exist for it
                if should_remove_join {
                    self.channel_store.remove_join(&join_channels);
                }
            }

            // 4. Build ContResult
            let cont_result = ContResult {
                continuation: cont,
                persistent: cont_persist,
                peek: false,
                channels: join_channels,
                patterns,
            };

            let produce_event = Produce::default();

            return Ok(Some((cont_result, matched_data.into_vec(), produce_event)));
        }

        // 5. Check for matching continuation at prefix paths (PathMap prefix semantics)
        // This enables produce on @[0,1,2] to trigger continuation at @[0,1]
        //
        // Formal Correspondence: PathMapStore.v (produce_triggers_prefix_continuation)
        if let Some((patterns, cont, cont_persist, join_channels, _prefix_channel, match_idx, suffix_key)) =
            self.find_matching_continuation_at_prefix(&channel, &data)
        {
            // Found a match at a prefix path - collect data from all channels in the join
            // SmallVec: most joins have ≤4 channels, avoiding heap allocation
            let mut matched_data: SmallVec<[RSpaceResult<C, A>; 4]> = SmallVec::new();

            for (i, ch) in join_channels.iter().enumerate() {
                if i == match_idx {
                    // This is the prefix channel - use the data we're producing
                    // The result channel is the actual produce channel, not the prefix
                    // Include suffix key for data wrapping per design spec:
                    // Data at @[0,1,2] consumed at @[0,1] should become [2, data]
                    matched_data.push(RSpaceResult {
                        channel: channel.clone(),
                        matched_datum: data.clone(),
                        removed_datum: data.clone(),
                        persistent: persist,
                        suffix_key: if suffix_key.is_empty() { None } else { Some(suffix_key.clone()) },
                    });
                } else if i < patterns.len() {
                    // Another channel in the join - try to find matching data
                    if let Some(dc) = self.channel_store.get_data_collection_mut(ch) {
                        let pattern = &patterns[i];
                        if let Some(found_data) =
                            dc.find_and_remove(|a| self.matcher.matches(pattern, a))
                        {
                            matched_data.push(RSpaceResult {
                                channel: ch.clone(),
                                matched_datum: found_data.clone(),
                                removed_datum: found_data,
                                persistent: false,
                                suffix_key: None, // Other channels in join are exact matches
                            });
                        }
                    }
                }
            }

            // Remove the continuation if not persistent
            if !cont_persist {
                let should_remove_join = if let Some(cc) = self.channel_store.get_continuation_collection_mut(&join_channels)
                {
                    // Note: We compare patterns only since cont is a clone and pointer comparison won't work
                    cc.find_and_remove(|p, _k| p == patterns.as_slice());
                    // Only remove join if no more continuations remain for this join pattern
                    cc.is_empty()
                } else {
                    true // No collection means join can be removed
                };

                // Only remove join pattern if no more continuations exist for it
                if should_remove_join {
                    self.channel_store.remove_join(&join_channels);
                }
            }

            // Build ContResult - note that channels refer to the continuation's join channels
            // (which includes the prefix), but the result contains the actual produce channel
            let cont_result = ContResult {
                continuation: cont,
                persistent: cont_persist,
                peek: false,
                channels: join_channels,
                patterns,
            };

            let produce_event = Produce::default();

            return Ok(Some((cont_result, matched_data.into_vec(), produce_event)));
        }

        // 6. No match found at exact or prefix paths - store the data
        let dc = self.channel_store.get_or_create_data_collection(&channel);

        // Check if this is a VectorDB collection requiring embedding extraction
        // Uses runtime type downcasting to detect VectorDBDataCollection
        use std::any::Any;
        use super::collections::VectorDBDataCollection;

        let dc_any: &mut dyn Any = dc;
        if let Some(vec_dc) = dc_any.downcast_mut::<VectorDBDataCollection<A>>() {
            // VectorDB collection detected - try to extract embedding from data
            // The data type A in RhoISpace is ListParWithRandom, not Par directly.
            // ListParWithRandom wraps Vec<Par> with random state for deterministic execution.
            let data_any: &dyn Any = &data;

            // Try to extract embedding from data
            // Note: Not all data sent to a VectorDB space has embeddings (e.g., sync channels like ready!(Nil))
            // We gracefully fallback to standard storage for non-vector data.
            let embedding_result = if let Some(list_par) = data_any.downcast_ref::<ListParWithRandom>() {
                // Extract the first Par from the list
                if let Some(data_par) = list_par.pars.first() {
                    let embedding_type = vec_dc.embedding_type();
                    let dimensions = vec_dc.embedding_dimensions();
                    extract_embedding_from_map(data_par, embedding_type, dimensions)
                } else {
                    Err(SpaceError::EmbeddingExtractionError {
                        description: "ListParWithRandom contains no Par values".to_string(),
                    })
                }
            }
            // Fallback: try direct Par downcast (for unit tests or other contexts)
            else if let Some(data_par) = data_any.downcast_ref::<Par>() {
                let embedding_type = vec_dc.embedding_type();
                let dimensions = vec_dc.embedding_dimensions();
                extract_embedding_from_map(data_par, embedding_type, dimensions)
            }
            else {
                Err(SpaceError::EmbeddingExtractionError {
                    description: format!(
                        "Data type not supported for embedding extraction: {:?}",
                        std::any::type_name::<A>()
                    ),
                })
            };

            match embedding_result {
                Ok(embedding) => {
                    // Store with embedding for similarity indexing
                    vec_dc.put_with_embedding_and_persist(data, embedding, persist)?;
                }
                Err(_) => {
                    // Graceful fallback: store without embedding for non-vector data
                    // This allows sync channels like ready!(Nil) to work within VectorDB spaces
                    vec_dc.put_with_persist(data, persist)?;
                }
            }
        } else {
            // Check if this is a PriorityQueue collection that needs priority handling
            use super::collections::PriorityQueueDataCollection;

            let dc_any: &mut dyn Any = dc;
            if let Some(pq_dc) = dc_any.downcast_mut::<PriorityQueueDataCollection<A>>() {
                // PriorityQueue collection - use priority if specified
                match priority {
                    Some(p) => pq_dc.put_with_priority_and_persist(data, p, persist)?,
                    None => pq_dc.put_with_persist(data, persist)?,
                }
            } else {
                // Standard data collection - use put with persistence flag
                // Priority is ignored for non-PriorityQueue collections
                dc.put_with_persist(data, persist)?;
            }
        }

        Ok(None)
    }

    fn consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)>, SpaceError> {
        if channels.len() != patterns.len() {
            return Err(SpaceError::InvalidConfiguration {
                description: format!(
                    "Channels and patterns length mismatch: {} vs {}",
                    channels.len(),
                    patterns.len()
                ),
            });
        }

        // =====================================================================
        // LAZY CHANNEL HANDLING
        // =====================================================================
        // Check if any channel is a lazy result channel from a similarity query.
        // Lazy channels are created in consume_with_similarity and registered
        // in lazy_producers. When a consumer does `for (@doc <- resultCh)` on
        // such a channel, we intercept here and lazily retrieve the next document.
        //
        // This enables TRUE lazy evaluation:
        // - Documents are retrieved on-demand (not eagerly)
        // - Early termination works (stop consuming = no more retrieval)
        // - Memory efficient (only requested documents are cloned)
        use std::any::Any;
        use super::collections::VectorDBDataCollection;

        // For single-channel consumes, check if it's a lazy result channel
        if channels.len() == 1 {
            // Try to extract channel ID by downcasting to Par
            let channel_ref: &C = &channels[0];
            let channel_any: &dyn Any = channel_ref;
            if let Some(par) = channel_any.downcast_ref::<Par>() {
                if let Some(channel_id) = extract_channel_id_from_par(par) {
                    // Check if this channel has a lazy producer
                    // Lazy allocation: skip if lazy_producers hasn't been allocated
                    let mut should_remove_producer = false;
                    let mut lazy_result: Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)> = None;

                    if let Some(producer) = self.lazy_producers.as_mut().and_then(|m| m.get_mut(&channel_id)) {
                        // Get the next document index from the producer
                        if let Some((doc_idx, score)) = producer.next_index() {
                            // Retrieve the document from the source VectorDB
                            let source_channel = producer.source_channel().clone();

                            // Get the VectorDB collection via the source channel
                            if let Some(dc) = self.channel_store.get_data_collection_mut(&source_channel) {
                                let dc_any: &mut dyn Any = dc;
                                if let Some(vec_dc) = dc_any.downcast_mut::<VectorDBDataCollection<A>>() {
                                    // Retrieve document by index, respecting persistence semantics
                                    // Persistent docs (stored with !!) return clone without removal
                                    // Non-persistent docs (stored with !) are tombstoned
                                    if let Some((data, _embedding)) = vec_dc.get_or_remove_by_index(doc_idx) {
                                        tracing::info!(
                                            "consume: lazy retrieval of doc {} (score={:.3}) from VectorDB",
                                            doc_idx, score
                                        );

                                        // Build result with the retrieved document
                                        // The data is already of type A (e.g., ListParWithRandom for RhoISpace)
                                        let result = RSpaceResult {
                                            channel: channels[0].clone(),
                                            matched_datum: data.clone(),
                                            removed_datum: data,
                                            persistent: false,
                                            suffix_key: None,
                                        };

                                        let cont_result = ContResult {
                                            continuation: continuation.clone(),
                                            persistent: persist,
                                            peek: !peeks.is_empty(),
                                            channels: channels.clone(),
                                            patterns: patterns.clone(),
                                        };

                                        // Check if producer is exhausted after this retrieval
                                        if producer.is_exhausted() {
                                            should_remove_producer = true;
                                        }

                                        lazy_result = Some((cont_result, vec![result]));
                                    } else {
                                        tracing::warn!(
                                            "consume: lazy channel doc {} not found in VectorDB (may have been consumed)",
                                            doc_idx
                                        );
                                    }
                                }
                            }
                        } else {
                            // Producer exhausted - mark for removal
                            // The consume will then block waiting for data on this channel,
                            // which is the correct Rholang semantics for an empty channel.
                            should_remove_producer = true;
                            // Fall through to normal consume behavior (will block waiting)
                        }
                    }

                    // Remove producer outside the mutable borrow scope if needed
                    if should_remove_producer {
                        if let Some(m) = self.lazy_producers.as_mut() {
                            m.remove(&channel_id);
                        }
                    }

                    // Return lazy result if we got one
                    if let Some(result) = lazy_result {
                        return Ok(Some(result));
                    }
                }
            }
        }

        // Try prefix-aware matching first if the store supports it
        // This enables consume on @[0,1] to find data at @[0,1,2], @[0,1,3], etc.
        //
        // Uses atomic check_and_consume_with_prefix_atomic to avoid TOCTOU race:
        // Previously, check_all_channels_match_with_prefix would peek data, then
        // remove_matched_data_from_prefix_results would remove it. Concurrent
        // consumers with overlapping prefixes could both peek the same data,
        // but only one would successfully remove, leaving the other incomplete.
        if self.channel_store.supports_prefix_semantics() {
            if let Some(atomic_results) = self.check_and_consume_with_prefix_atomic(&channels, &patterns, &peeks) {
                // Build result - use the actual channel where data was found
                // Include suffix key for data wrapping per design spec:
                // Data at @[0,1,2] consumed at @[0,1] should become [2, data]
                let results: Vec<RSpaceResult<C, A>> = atomic_results
                    .into_iter()
                    .map(|(channel, data, suffix_key)| {
                        RSpaceResult {
                            channel,
                            matched_datum: data.clone(),
                            removed_datum: data,
                            persistent: false,
                            // Suffix key: None for exact matches, Some([...]) for prefix matches
                            suffix_key: if suffix_key.is_empty() { None } else { Some(suffix_key) },
                        }
                    })
                    .collect();

                let cont_result = ContResult {
                    continuation,
                    persistent: persist,
                    peek: !peeks.is_empty(),
                    channels: channels.clone(),
                    patterns,
                };

                return Ok(Some((cont_result, results)));
            }
        } else {
            // Use exact-match semantics for non-prefix stores (HashMap, Array, etc.)
            // Single-pass atomic check-and-remove eliminates TOCTOU race and improves performance
            if let Some(removed_data) =
                self.check_and_remove_matched_data_atomic(&channels, &patterns, &peeks)
            {
                // Build result - no suffix key for exact matches
                let results: Vec<RSpaceResult<C, A>> = channels
                    .iter()
                    .zip(removed_data.into_iter())
                    .map(|(ch, data)| RSpaceResult {
                        channel: ch.clone(),
                        matched_datum: data.clone(),
                        removed_datum: data,
                        persistent: false,
                        suffix_key: None, // Exact match - no suffix key
                    })
                    .collect();

                let cont_result = ContResult {
                    continuation,
                    persistent: persist,
                    peek: !peeks.is_empty(),
                    channels: channels.clone(),
                    patterns,
                };

                return Ok(Some((cont_result, results)));
            }
        }

        // No match found - store the continuation
        // First, register join patterns
        for _ch in &channels {
            self.channel_store.put_join(channels.clone());
            // Break after first - put_join adds to all channels in the join
            break;
        }

        // Then store the continuation
        let cc = self
            .channel_store
            .get_or_create_continuation_collection(&channels);
        cc.put(patterns, continuation, persist);

        Ok(None)
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
        // Check if any pattern modifier is present
        let has_modifiers = modifiers.iter().any(|m| !m.is_empty());

        if !has_modifiers {
            // No pattern modifiers - use standard consume
            return self.consume(channels, patterns, continuation, persist, peeks);
        }

        // Extract and validate pattern modifiers
        // This prepares the data for VectorDB matching and provides early error detection
        // Store (embedding, resolved_threshold, optional_metric, optional_top_k)
        // threshold is resolved now so we can store it with the continuation
        use std::any::Any;
        use super::collections::{SimilarityMetric, VectorDBDataCollection, StoredSimilarityInfo};

        let mut extracted_patterns: Vec<Option<(Vec<f32>, f32, Option<SimilarityMetric>, Option<usize>)>> = Vec::with_capacity(modifiers.len());

        for (mods, channel) in modifiers.iter().zip(channels.iter()) {
            if !mods.is_empty() {
                // Extract modifier information from EFunction calls
                let extracted = extract_modifiers_from_efunctions(mods)?;

                // Get the query embedding (required)
                let embedding = extracted.query_embedding.ok_or_else(|| SpaceError::SimilarityMatchError {
                    reason: "Pattern modifier requires a query embedding".to_string(),
                })?;

                // Use extracted metric or default
                let resolved_metric = extracted.metric;

                // Use extracted threshold or resolve from collection's default
                let resolved_threshold = if let Some(threshold) = extracted.threshold {
                    threshold
                } else {
                    // Get threshold from VectorDB collection if available
                    if let Some(dc) = self.channel_store.get_data_collection(channel) {
                        let dc_any: &dyn Any = dc;
                        if let Some(vec_dc) = dc_any.downcast_ref::<VectorDBDataCollection<A>>() {
                            vec_dc.default_threshold()
                        } else {
                            0.5 // Fallback for non-VectorDB collections
                        }
                    } else {
                        0.5 // Fallback if collection doesn't exist yet
                    }
                };

                // Use extracted top_k if present
                let top_k = extracted.top_k;

                extracted_patterns.push(Some((embedding, resolved_threshold, resolved_metric, top_k)));
            } else {
                extracted_patterns.push(None);
            }
        }

        // For each channel with a similarity pattern, try VectorDB matching
        // When top-K is specified, we collect indices for lazy retrieval
        //
        // LAZY CHANNEL SEMANTICS:
        // Instead of eagerly retrieving all documents, we:
        // 1. Compute similarity scores and collect indices (not documents)
        // 2. Create a lazy result channel (GPrivate)
        // 3. Store a LazyResultProducer in the registry
        // 4. Return the lazy channel to the continuation
        // 5. Documents are retrieved on-demand when the channel is consumed
        let mut similarity_results: Vec<(usize, Vec<(usize, f32)>, C)> = Vec::new();  // (channel_idx, sorted_indices, source_channel)

        for (i, (channel, extracted)) in channels.iter().zip(extracted_patterns.iter()).enumerate() {
            if let Some((embedding, resolved_threshold, resolved_metric, top_k)) = extracted {
                if let Some(dc) = self.channel_store.get_data_collection_mut(channel) {
                    // Try to downcast to VectorDBDataCollection
                    let dc_any: &mut dyn Any = dc;
                    if let Some(vec_dc) = dc_any.downcast_mut::<VectorDBDataCollection<A>>() {
                        let k = top_k.unwrap_or(1);

                        // Use the new query() method which delegates to backend.
                        // This supports per-query metric override via resolved_metric.
                        let similarity_fn = resolved_metric.as_ref().map(|m| m.as_str());
                        let ranking_fn = Some("topk");
                        // Use backend ResolvedArg (Int variant, not Integer)
                        let params = vec![super::ResolvedArg::Int(k as i64)];

                        // LAZY: Get indices only (no document retrieval or removal yet)
                        let indices = match vec_dc.query(
                            embedding,
                            similarity_fn,
                            Some(*resolved_threshold),
                            ranking_fn,
                            &params,
                        ) {
                            Ok(results) => results,
                            Err(e) => {
                                tracing::warn!(
                                    "VectorDB query failed on channel {}: {}",
                                    i, e
                                );
                                vec![]
                            }
                        };

                        if !indices.is_empty() {
                            tracing::info!(
                                "VectorDB similarity match on channel {}: found {} indices (threshold: {}, k: {}, metric: {:?})",
                                i, indices.len(), resolved_threshold, k, similarity_fn
                            );
                            // Store indices along with source channel for lazy retrieval
                            similarity_results.push((i, indices, channel.clone()));
                            continue;
                        }
                    } else {
                        tracing::warn!(
                            "Similarity pattern on non-VectorDB collection at channel {} - using standard match",
                            i
                        );
                    }
                }
                // No similarity match found for this channel - leave empty (will fail all_required_matched check)
            }
            // No similarity pattern for this channel - skip (will use standard matching later if needed)
        }

        // Check if we got similarity matches for all channels that required them
        let channels_with_patterns: Vec<usize> = extracted_patterns.iter()
            .enumerate()
            .filter_map(|(i, p)| if p.is_some() { Some(i) } else { None })
            .collect();
        let matched_channels: std::collections::HashSet<usize> = similarity_results.iter()
            .map(|(i, _, _)| *i)
            .collect();
        let all_required_matched = channels_with_patterns.iter()
            .all(|i| matched_channels.contains(i));

        if all_required_matched && !similarity_results.is_empty() {
            // LAZY CHANNEL SEMANTICS:
            // For each channel with similarity matches, create a lazy result channel.
            // The continuation receives the lazy channel(s), and documents are retrieved
            // on-demand when the channel is consumed via `for (@doc <- resultCh)`.
            // SmallVec: most joins have ≤4 channels, avoiding heap allocation
            let mut results: SmallVec<[RSpaceResult<C, A>; 4]> = SmallVec::new();

            for (channel_idx, sorted_indices, source_channel) in similarity_results {
                let pattern_channel = &channels[channel_idx];

                // Create unforgeable result channel using UUID
                let result_channel_id = Uuid::new_v4().as_bytes().to_vec();
                let result_channel_par = Par::default().with_unforgeables(vec![GUnforgeable {
                    unf_instance: Some(UnfInstance::GPrivateBody(GPrivate {
                        id: result_channel_id.clone(),
                    })),
                }]);

                // Create lazy producer with indices and source channel
                let producer = super::collections::LazyResultProducer::new(
                    sorted_indices.clone(),
                    source_channel,
                );

                tracing::info!(
                    "consume_with_similarity: created lazy channel {:?} with {} indices for channel {}",
                    &result_channel_id[..8], producer.total_matches(), channel_idx
                );

                // Store producer in registry keyed by result channel ID
                // Lazy allocation: create HashMap on first use
                self.lazy_producers
                    .get_or_insert_with(HashMap::new)
                    .insert(result_channel_id, producer);

                // Create result with lazy channel wrapped in the data type A
                // For RhoISpace, A is ListParWithRandom, so we construct it and use Any to convert.
                // The continuation body will receive this channel and consume from it.
                //
                // We use runtime type checking since A is generic but we know it must be
                // ListParWithRandom for VectorDB lazy channels to work.
                use std::any::Any;
                let lazy_result_lpwr = ListParWithRandom {
                    pars: vec![result_channel_par],
                    random_state: Vec::new(),  // No random state needed for lazy channel
                };

                // Convert ListParWithRandom to type A using Any downcast
                // This works because A is ListParWithRandom for RhoISpace
                let lazy_result_any: Box<dyn Any> = Box::new(lazy_result_lpwr.clone());
                if let Ok(lazy_result) = lazy_result_any.downcast::<A>() {
                    results.push(RSpaceResult {
                        channel: pattern_channel.clone(),
                        matched_datum: (*lazy_result).clone(),
                        removed_datum: *lazy_result,
                        persistent: false,
                        suffix_key: None,
                    });
                } else {
                    tracing::warn!(
                        "consume_with_similarity: lazy channel feature requires A=ListParWithRandom, \
                         falling back to standard matching for channel {}",
                        channel_idx
                    );
                    // Skip this channel - will fall through to standard matching
                    continue;
                }
            }

            if !results.is_empty() {
                tracing::info!(
                    "consume_with_similarity: returning {} lazy result channel(s)",
                    results.len()
                );
                let cont_result = ContResult {
                    continuation,
                    persistent: persist,
                    peek: !peeks.is_empty(),
                    channels: channels.clone(),
                    patterns,
                };
                return Ok(Some((cont_result, results.into_vec())));
            }
        }

        // No similarity match found - store the continuation with similarity info
        // so that produce() can wake it up when matching data arrives.
        //
        // Previously this was an interim fix that just returned None without storing,
        // but now we implement the full fix by storing similarity patterns with the
        // continuation and checking them in find_fireable_continuation().
        if extracted_patterns.iter().any(|p| p.is_some()) {
            // Register join patterns
            for _ch in &channels {
                self.channel_store.put_join(channels.clone());
                break;
            }

            // Generate a unique continuation ID for the query matrix
            use super::collections::{ContinuationId, SimilarityQueryMatrix};
            let cont_id = ContinuationId(
                self.next_continuation_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed)  // Relaxed sufficient - uniqueness only
            );

            // Add queries to the SimilarityQueryMatrix for each channel with a similarity pattern
            // This enables the efficient matrix-based matching in produce()
            for (channel_idx, (channel, extracted)) in channels.iter().zip(extracted_patterns.iter()).enumerate() {
                if let Some((embedding, resolved_threshold, _resolved_metric, _top_k)) = extracted {
                    // Get embedding dimensions from the VectorDB collection
                    let dimensions = if let Some(dc) = self.channel_store.get_data_collection(channel) {
                        let dc_any: &dyn Any = dc;
                        if let Some(vec_dc) = dc_any.downcast_ref::<VectorDBDataCollection<A>>() {
                            vec_dc.embedding_dimensions()
                        } else {
                            embedding.len() // Use query embedding dimensions as fallback
                        }
                    } else {
                        embedding.len()
                    };

                    // Get or create the SimilarityQueryMatrix for this channel
                    // Lazy allocation: create HashMap on first use
                    let query_matrix = self.similarity_queries
                        .get_or_insert_with(HashMap::new)
                        .entry(channel.clone())
                        .or_insert_with(|| SimilarityQueryMatrix::new(dimensions));

                    // Add the query to the matrix
                    if let Err(e) = query_matrix.add_query(embedding, *resolved_threshold, cont_id, channel_idx, persist) {
                        tracing::warn!(
                            "consume_with_similarity: failed to add query to matrix: {:?}",
                            e
                        );
                    }
                }
            }

            // Store the continuation with similarity info for later matching
            let similarity_info = StoredSimilarityInfo::new(extracted_patterns);
            let cc = self
                .channel_store
                .get_or_create_continuation_collection(&channels);
            cc.put_with_similarity(patterns, continuation, persist, Some(similarity_info));

            return Ok(None);
        }

        // No similarity patterns at all - use standard consume
        self.consume(channels, patterns, continuation, persist, peeks)
    }

    fn install(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, SpaceError> {
        // Install is like consume with persist=true, but always stores
        // First, register join patterns
        for _ch in &channels {
            self.channel_store.put_join(channels.clone());
            break;
        }

        // Store as persistent continuation
        let cc = self
            .channel_store
            .get_or_create_continuation_collection(&channels);
        cc.put(patterns.clone(), continuation.clone(), true);

        // Check if there's already matching data (single-pass atomic operation)
        let peeks = BTreeSet::new();
        if let Some(removed_data) =
            self.check_and_remove_matched_data_atomic(&channels, &patterns, &peeks)
        {
            return Ok(Some((continuation, removed_data)));
        }

        Ok(None)
    }

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        self.channel_store
            .get_data_collection(channel)
            .map(|dc| {
                dc.all_data()
                    .into_iter()
                    .map(|a| Datum {
                        a: a.clone(),
                        persist: false,
                        source: Produce::default(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        self.channel_store
            .get_continuation_collection(&channels)
            .map(|cc| {
                cc.all_continuations()
                    .into_iter()
                    .map(|(patterns, cont, persist)| WaitingContinuation {
                        patterns: patterns.to_vec(),
                        continuation: cont.clone(),
                        persist,
                        source: Default::default(),
                        peeks: BTreeSet::new(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn get_joins(&self, channel: C) -> Vec<Vec<C>> {
        self.channel_store.get_joins(&channel)
    }
}

// =============================================================================
// CheckpointableSpace Implementation
// =============================================================================

impl<CS, M, C, P, A, K, DC, CC> CheckpointableSpace<C, P, A, K> for GenericRSpace<CS, M>
where
    CS: ChannelStore<Channel = C, Pattern = P, Data = A, Continuation = K, DataColl = DC, ContColl = CC> + Clone,
    M: Match<P, A> + Clone,
    C: Clone + Eq + Hash + Send + Sync + Serialize + DeserializeOwned + AsRef<[u8]> + 'static,
    P: Clone + PartialEq + Send + Sync + Serialize + DeserializeOwned + 'static,
    A: Clone + Send + Sync + std::fmt::Debug + Serialize + DeserializeOwned + 'static,
    K: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
    DC: DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: ContinuationCollection<P, K> + Default + Clone + Send + Sync,
{
    fn create_checkpoint(&mut self) -> Result<Checkpoint, SpaceError> {
        // All spaces support checkpointing, but Temp spaces have ephemeral data
        // that is cleared on restore (only the qualifier and empty collections persist)
        let state = self.serialize_state();
        let root = Blake2b256Hash::new(&state);

        // Store in history if available
        if let Some(ref store) = self.history_store {
            store.store(root.clone(), &state)?;
        }

        Ok(Checkpoint {
            root,
            log: Log::default(),
        })
    }

    fn create_soft_checkpoint(&mut self) -> SoftCheckpoint<C, P, A, K> {
        // Store a snapshot of the channel store for later restoration
        let gensym_counter = self.channel_store.gensym_counter();
        self.soft_checkpoint_stack.push((self.channel_store.snapshot(), gensym_counter));

        // Return a SoftCheckpoint token (the actual data is in our stack)
        // The HotStoreState is left empty since we use our own snapshotting
        let cache_snapshot = HotStoreState {
            continuations: DashMap::new(),
            installed_continuations: DashMap::new(),
            data: DashMap::new(),
            joins: DashMap::new(),
            installed_joins: DashMap::new(),
        };
        SoftCheckpoint {
            cache_snapshot,
            log: Log::default(),
            produce_counter: BTreeMap::new(),
        }
    }

    fn revert_to_soft_checkpoint(
        &mut self,
        _checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), SpaceError> {
        // Pop and restore from our snapshot stack
        if let Some((snapshot, snapshot_gensym_counter)) = self.soft_checkpoint_stack.pop() {
            // Keep track of current counter before restoring
            let current_counter = self.channel_store.gensym_counter();
            self.channel_store = snapshot;
            // Use max to prevent name collisions with channels generated after checkpoint
            self.channel_store.set_gensym_counter(std::cmp::max(current_counter, snapshot_gensym_counter));
            Ok(())
        } else {
            Err(SpaceError::CheckpointError {
                description: "No soft checkpoint to revert to".to_string(),
            })
        }
    }

    fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), SpaceError> {
        // Retrieve state from history store
        if let Some(ref store) = self.history_store {
            if !store.contains(root) {
                return Err(SpaceError::CheckpointError {
                    description: format!("Checkpoint root not found: {}", root),
                });
            }

            let state = store.retrieve(root)?;
            self.deserialize_state(&state)?;

            Ok(())
        } else {
            Err(SpaceError::CheckpointError {
                description: "No history store configured for reset".to_string(),
            })
        }
    }

    fn clear(&mut self) -> Result<(), SpaceError> {
        self.channel_store.clear();
        self.soft_checkpoint = None;
        Ok(())
    }
}

// =============================================================================
// ReplayableSpace Implementation
// =============================================================================

impl<CS, M, C, P, A, K, DC, CC> ReplayableSpace<C, P, A, K> for GenericRSpace<CS, M>
where
    CS: ChannelStore<Channel = C, Pattern = P, Data = A, Continuation = K, DataColl = DC, ContColl = CC> + Clone,
    M: Match<P, A> + Clone,
    C: Clone + Eq + Hash + Send + Sync + Serialize + DeserializeOwned + AsRef<[u8]> + 'static,
    P: Clone + PartialEq + Send + Sync + Serialize + DeserializeOwned + 'static,
    A: Clone + Send + Sync + std::fmt::Debug + Serialize + DeserializeOwned + 'static,
    K: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
    DC: DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: ContinuationCollection<P, K> + Default + Clone + Send + Sync,
{
    fn rig_and_reset(&mut self, start_root: Blake2b256Hash, log: Log) -> Result<(), SpaceError> {
        // Call rig first to populate replay_data
        ReplayableSpace::rig(self, log.clone())?;

        // Reset to the starting state
        CheckpointableSpace::reset(self, &start_root)?;

        // Store the log for replay verification
        self.replay_log = Some(log);
        self.is_replay = true;

        Ok(())
    }

    fn rig(&self, log: Log) -> Result<(), SpaceError> {
        // Partition log into io_events and comm_events
        let (io_events, comm_events): (Vec<_>, Vec<_>) =
            log.iter().partition(|event| match event {
                Event::IoEvent(IOEvent::Produce(_)) => true,
                Event::IoEvent(IOEvent::Consume(_)) => true,
                Event::Comm(_) => false,
            });

        // Create set of IOEvents for lookup
        let new_stuff: HashSet<_> = io_events.into_iter().collect();

        // Clear and populate replay_data (MultisetMultiMap uses DashMap, methods take &self)
        self.replay_data.clear();

        for event in comm_events {
            match event {
                Event::Comm(comm) => {
                    let (consume, produces) = (comm.consume.clone(), comm.produces.clone());
                    let mut io_events: Vec<IOEvent> = produces
                        .into_iter()
                        .map(IOEvent::Produce)
                        .collect();
                    io_events.insert(0, IOEvent::Consume(consume));

                    for io_event in io_events {
                        let io_event_converted = Event::IoEvent(io_event.clone());
                        if new_stuff.contains(&io_event_converted) {
                            self.replay_data.add_binding(io_event, comm.clone());
                        }
                    }
                }
                _ => return Err(SpaceError::ReplayError {
                    description: "Only COMM events expected in replay log".to_string(),
                }),
            }
        }

        Ok(())
    }

    fn check_replay_data(&self) -> Result<(), SpaceError> {
        if !self.is_replay {
            return Err(SpaceError::ReplayError {
                description: "Not in replay mode".to_string(),
            });
        }

        // Verify that all COMM events from replay_log were consumed
        // (MultisetMultiMap uses DashMap, methods take &self - no lock needed)
        if self.replay_data.is_empty() {
            Ok(())
        } else {
            Err(SpaceError::ReplayError {
                description: format!(
                    "Unused COMM event: replay_data has {} elements left",
                    self.replay_data.map.len()
                ),
            })
        }
    }

    fn is_replay(&self) -> bool {
        self.is_replay
    }

    fn update_produce(&mut self, produce_ref: Produce) {
        // Record the produce result for replay verification by updating
        // matching produce refs in the replay log
        if let Some(ref mut log) = self.replay_log {
            for event in log.iter_mut() {
                match event {
                    Event::IoEvent(IOEvent::Produce(produce)) => {
                        if produce.hash == produce_ref.hash {
                            *produce = produce_ref.clone();
                        }
                    }
                    Event::Comm(comm) => {
                        for produce in comm.produces.iter_mut() {
                            if produce.hash == produce_ref.hash {
                                *produce = produce_ref.clone();
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

// =============================================================================
// Builder Pattern
// =============================================================================

/// Builder for constructing GenericRSpace instances.
///
/// The builder supports two usage patterns:
/// 1. **Simple (Bag collections)**: Use default DC/CC parameters for backward compatibility
/// 2. **Advanced (custom collections)**: Specify DC/CC to use Queue, Stack, Set, etc.
///
/// # Example with default Bag collections
/// ```ignore
/// let space = GenericRSpaceBuilder::<_, _, Channel, Pattern, Data, Continuation>::new()
///     .with_channel_store(store)
///     .with_matcher(matcher)
///     .build()?;
/// ```
///
/// # Example with custom collections
/// ```ignore
/// let space = GenericRSpaceBuilder::<_, _, Channel, Pattern, Data, Cont, QueueDC<Data>, QueueCC<Pattern, Cont>>::new()
///     .with_channel_store(store)
///     .with_matcher(matcher)
///     .build()?;
/// ```
pub struct GenericRSpaceBuilder<CS, M>
where
    CS: ChannelStore,
    M: Match<CS::Pattern, CS::Data>,
{
    channel_store: Option<CS>,
    matcher: Option<M>,
    space_id: Option<SpaceId>,
    qualifier: SpaceQualifier,
    history_store: Option<BoxedHistoryStore>,
    theory: Option<BoxedTheory>,
}

impl<CS, M> GenericRSpaceBuilder<CS, M>
where
    CS: ChannelStore,
    M: Match<CS::Pattern, CS::Data>,
{
    /// Create a new builder with default values.
    pub fn new() -> Self {
        GenericRSpaceBuilder {
            channel_store: None,
            matcher: None,
            space_id: None,
            qualifier: SpaceQualifier::Default,
            history_store: None,
            theory: None,
        }
    }

    /// Set the channel store.
    pub fn with_channel_store(mut self, store: CS) -> Self {
        self.channel_store = Some(store);
        self
    }

    /// Set the matcher.
    pub fn with_matcher(mut self, matcher: M) -> Self {
        self.matcher = Some(matcher);
        self
    }

    /// Set the space ID.
    pub fn with_space_id(mut self, id: SpaceId) -> Self {
        self.space_id = Some(id);
        self
    }

    /// Set the qualifier.
    pub fn with_qualifier(mut self, qualifier: SpaceQualifier) -> Self {
        self.qualifier = qualifier;
        self
    }

    /// Set the history store.
    pub fn with_history_store(mut self, store: BoxedHistoryStore) -> Self {
        self.history_store = Some(store);
        self
    }

    /// Set the theory for data validation.
    ///
    /// When a theory is set, all data produced to the space will be validated
    /// against it before being stored. Invalid data will be rejected.
    pub fn with_theory(mut self, theory: BoxedTheory) -> Self {
        self.theory = Some(theory);
        self
    }

    /// Build the GenericRSpace.
    ///
    /// # Errors
    ///
    /// Returns `SpaceError::BuilderIncomplete` if channel_store or matcher is not set.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let space = GenericRSpaceBuilder::new()
    ///     .with_channel_store(store)
    ///     .with_matcher(matcher)
    ///     .build()?;
    /// ```
    pub fn build(self) -> Result<GenericRSpace<CS, M>, SpaceError> {
        let channel_store = self.channel_store.ok_or(SpaceError::BuilderIncomplete {
            builder: "GenericRSpaceBuilder",
            missing_field: "channel_store",
        })?;
        let matcher = self.matcher.ok_or(SpaceError::BuilderIncomplete {
            builder: "GenericRSpaceBuilder",
            missing_field: "matcher",
        })?;
        let space_id = self.space_id.unwrap_or_else(SpaceId::default_space);

        let mut space = GenericRSpace::new(channel_store, matcher, space_id, self.qualifier);

        if let Some(history_store) = self.history_store {
            space.history_store = Some(history_store);
        }

        if let Some(theory) = self.theory {
            space.theory = Some(theory);
        }

        Ok(space)
    }

    /// Build the GenericRSpace, panicking if incomplete.
    ///
    /// This is a convenience method for cases where you're certain the builder
    /// is complete. Prefer `build()` for production code.
    ///
    /// # Panics
    ///
    /// Panics if channel_store or matcher is not set.
    pub fn build_unchecked(self) -> GenericRSpace<CS, M> {
        self.build().expect("GenericRSpaceBuilder incomplete")
    }
}

impl<CS, M> Default for GenericRSpaceBuilder<CS, M>
where
    CS: ChannelStore,
    M: Match<CS::Pattern, CS::Data>,
{
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// ISpace Implementation
// =============================================================================
//
// This implementation allows GenericRSpace to be used where ISpace is expected,
// enabling integration with the existing reducer infrastructure.

use rspace_plus_plus::rspace::{
    errors::RSpaceError,
    internal::Row,
    rspace_interface::{ISpace, MaybeConsumeResult},
};

impl<CS, M, C, P, A, K, DC, CC> ISpace<C, P, A, K> for GenericRSpace<CS, M>
where
    CS: ChannelStore<Channel = C, Pattern = P, Data = A, Continuation = K, DataColl = DC, ContColl = CC> + Clone,
    M: Match<P, A> + Clone,
    C: Clone + Eq + Hash + Send + Sync + Serialize + DeserializeOwned + AsRef<[u8]> + 'static,
    P: Clone + PartialEq + Send + Sync + Serialize + DeserializeOwned + 'static,
    A: Clone + Send + Sync + std::fmt::Debug + Serialize + DeserializeOwned + 'static,
    K: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
    DC: DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: ContinuationCollection<P, K> + Default + Clone + Send + Sync,
{
    fn create_checkpoint(&mut self) -> Result<Checkpoint, RSpaceError> {
        CheckpointableSpace::create_checkpoint(self)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))
    }

    fn get_data(&self, channel: &C) -> Vec<Datum<A>> {
        SpaceAgent::get_data(self, channel)
    }

    fn get_waiting_continuations(&self, channels: Vec<C>) -> Vec<WaitingContinuation<P, K>> {
        SpaceAgent::get_waiting_continuations(self, channels)
    }

    fn get_joins(&self, channel: C) -> Vec<Vec<C>> {
        SpaceAgent::get_joins(self, channel)
    }

    fn clear(&mut self) -> Result<(), RSpaceError> {
        CheckpointableSpace::clear(self)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))
    }

    fn reset(&mut self, root: &Blake2b256Hash) -> Result<(), RSpaceError> {
        CheckpointableSpace::reset(self, root)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))
    }

    fn consume_result(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        // consume_result is a peek-style operation that returns data without storing continuation.
        // It finds a waiting continuation for the channels and returns matching data if available.
        //
        // Semantics:
        // 1. Find waiting continuations for these channels
        // 2. Find one whose patterns match the provided patterns
        // 3. Look for data matching those patterns on each channel
        // 4. Return (continuation.k, matched_data) if all succeed

        // Step 1: Get waiting continuations for these channels
        let waiting_continuations = SpaceAgent::get_waiting_continuations(self, channels.clone());

        // Step 2: Find a continuation with matching patterns
        let matching_continuation = waiting_continuations.iter().find(|wc| {
            wc.patterns.len() == patterns.len()
                && wc
                    .patterns
                    .iter()
                    .zip(patterns.iter())
                    .all(|(wc_pat, pat)| wc_pat == pat)
        });

        let continuation = match matching_continuation {
            Some(wc) => wc.continuation.clone(),
            None => {
                // No matching continuation found - nothing to return
                return Ok(None);
            }
        };

        // Step 3: Find matching data on each channel
        // SmallVec optimization: most joins have ≤4 channels, stack-allocate for common case
        let mut matched_data: SmallVec<[A; 4]> = SmallVec::new();
        for (ch, pattern) in channels.iter().zip(patterns.iter()) {
            let data = SpaceAgent::get_data(self, ch);
            let matching = data.iter().find(|d| self.matcher.matches(pattern, &d.a));
            if let Some(datum) = matching {
                matched_data.push(datum.a.clone());
            } else {
                // No matching data on this channel
                return Ok(None);
            }
        }

        // Step 4: Return the continuation and matched data
        Ok(Some((continuation, matched_data.into_vec())))
    }

    fn to_map(&self) -> HashMap<Vec<C>, Row<P, A, K>> {
        // Convert channel store state to a HashMap
        // This is primarily for debugging and inspection
        let map = HashMap::new();

        // For each channel, collect data and continuations
        // The channel_store doesn't expose iteration directly, so this is approximate
        // A full implementation would need ChannelStore to expose its channels

        map
    }

    fn create_soft_checkpoint(&mut self) -> SoftCheckpoint<C, P, A, K> {
        CheckpointableSpace::create_soft_checkpoint(self)
    }

    fn revert_to_soft_checkpoint(
        &mut self,
        checkpoint: SoftCheckpoint<C, P, A, K>,
    ) -> Result<(), RSpaceError> {
        CheckpointableSpace::revert_to_soft_checkpoint(self, checkpoint)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))
    }

    fn consume(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>)>, RSpaceError> {
        SpaceAgent::consume(self, channels, patterns, continuation, persist, peeks)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))
    }

    fn consume_with_modifiers(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        modifiers_bytes: Vec<Vec<u8>>,
        continuation: K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> Result<MaybeConsumeResult<C, P, A, K>, RSpaceError> {
        use prost::Message;

        // Deserialize the pattern modifiers from bytes
        // Each Vec<u8> contains length-prefixed serialized EFunction messages
        let deserialized_modifiers: Vec<Vec<EFunction>> = modifiers_bytes
            .into_iter()
            .map(|bytes| {
                if bytes.is_empty() {
                    vec![]
                } else {
                    // Decode length-prefixed EFunction messages
                    let mut efunctions = Vec::new();
                    let mut offset = 0;
                    while offset + 4 <= bytes.len() {
                        // Read 4-byte little-endian length
                        let len = u32::from_le_bytes([
                            bytes[offset],
                            bytes[offset + 1],
                            bytes[offset + 2],
                            bytes[offset + 3],
                        ]) as usize;
                        offset += 4;

                        if offset + len <= bytes.len() {
                            if let Ok(ef) = EFunction::decode(&bytes[offset..offset + len]) {
                                efunctions.push(ef);
                            }
                            offset += len;
                        } else {
                            break;
                        }
                    }
                    efunctions
                }
            })
            .collect();

        // Call the SpaceAgent consume_with_modifiers method with deserialized modifiers
        SpaceAgent::consume_with_modifiers(self, channels, patterns, deserialized_modifiers, continuation, persist, peeks)
            .map_err(|e: SpaceError| RSpaceError::InterpreterError(e.to_string()))
    }

    fn produce(
        &mut self,
        channel: C,
        data: A,
        persist: bool,
        priority: Option<usize>,
    ) -> Result<Option<(ContResult<C, P, K>, Vec<RSpaceResult<C, A>>, Produce)>, RSpaceError> {
        SpaceAgent::produce(self, channel, data, persist, priority)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))
    }

    fn install(
        &mut self,
        channels: Vec<C>,
        patterns: Vec<P>,
        continuation: K,
    ) -> Result<Option<(K, Vec<A>)>, RSpaceError> {
        SpaceAgent::install(self, channels, patterns, continuation)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))
    }

    fn rig_and_reset(&mut self, start_root: Blake2b256Hash, log: Log) -> Result<(), RSpaceError> {
        ReplayableSpace::rig_and_reset(self, start_root, log)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))
    }

    fn rig(&self, log: Log) -> Result<(), RSpaceError> {
        ReplayableSpace::rig(self, log)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))
    }

    fn check_replay_data(&self) -> Result<(), RSpaceError> {
        ReplayableSpace::check_replay_data(self)
            .map_err(|e| RSpaceError::InterpreterError(e.to_string()))
    }

    fn is_replay(&self) -> bool {
        ReplayableSpace::is_replay(self)
    }

    fn update_produce(&mut self, produce: Produce) {
        ReplayableSpace::update_produce(self, produce)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::interpreter::spaces::channel_store::HashMapChannelStore;
    use crate::rust::interpreter::spaces::collections::{
        BagContinuationCollection, BagDataCollection,
    };
    use crate::rust::interpreter::spaces::matcher::WildcardMatch;

    use serde::{Serialize, Deserialize};

    /// Test channel type - newtype wrapper around Vec<u8> that implements necessary traits
    /// for both HashMap store (From<usize> for gensym) and PathMap suffix key semantics (AsRef<[u8]>).
    #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
    struct TestChannel(Vec<u8>);

    impl From<usize> for TestChannel {
        fn from(n: usize) -> Self {
            TestChannel(vec![n as u8])
        }
    }

    impl AsRef<[u8]> for TestChannel {
        fn as_ref(&self) -> &[u8] {
            &self.0
        }
    }

    type TestPattern = i32;  // Use same type as data for simpler testing
    type TestData = i32;
    type TestCont = String;

    /// Helper to create a test channel from a usize for convenience.
    fn chan(n: usize) -> TestChannel {
        TestChannel::from(n)
    }

    fn create_test_space() -> GenericRSpace<
        HashMapChannelStore<TestChannel, TestPattern, TestData, TestCont, BagDataCollection<TestData>, BagContinuationCollection<TestPattern, TestCont>>,
        WildcardMatch<TestPattern, TestData>,
    > {
        let store = HashMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
        let matcher = WildcardMatch::<TestPattern, TestData>::new();
        GenericRSpace::new(store, matcher, SpaceId::default_space(), SpaceQualifier::Default)
    }

    #[test]
    fn test_generic_rspace_creation() {
        let space = create_test_space();
        assert_eq!(space.qualifier(), SpaceQualifier::Default);
        assert!(!space.has_history_store());
    }

    #[test]
    fn test_gensym() {
        let mut space = create_test_space();

        let ch1 = space.gensym().expect("gensym should succeed");
        let ch2 = space.gensym().expect("gensym should succeed");
        let ch3 = space.gensym().expect("gensym should succeed");

        assert_ne!(ch1, ch2);
        assert_ne!(ch2, ch3);
        assert_ne!(ch1, ch3);
    }

    #[test]
    fn test_produce_stores_data() {
        let mut space = create_test_space();

        let result = SpaceAgent::produce(&mut space, chan(0), 42, false, None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // No matching continuation

        let data = SpaceAgent::get_data(&space, &chan(0));
        assert_eq!(data.len(), 1);
        assert_eq!(data[0].a, 42);
    }

    #[test]
    fn test_consume_stores_continuation() {
        let mut space = create_test_space();

        let channels = vec![chan(0)];
        let patterns = vec![0i32];  // Use i32 pattern
        let continuation = "cont".to_string();

        let result = SpaceAgent::consume(&mut space, channels.clone(), patterns, continuation, false, BTreeSet::new());
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // No matching data

        let conts = SpaceAgent::get_waiting_continuations(&space, channels);
        assert_eq!(conts.len(), 1);
    }

    #[test]
    fn test_get_joins() {
        let mut space = create_test_space();

        // Register a join by consuming on multiple channels
        let channels = vec![chan(0), chan(1)];
        let patterns = vec![1i32, 2i32];  // Use i32 patterns
        let continuation = "cont".to_string();

        let _ = SpaceAgent::consume(&mut space, channels.clone(), patterns, continuation, false, BTreeSet::new());

        let joins = SpaceAgent::get_joins(&space, chan(0));
        assert!(!joins.is_empty());
        assert!(joins.contains(&channels));
    }

    #[test]
    fn test_clear() {
        let mut space = create_test_space();

        SpaceAgent::produce(&mut space, chan(0), 42, false, None).expect("produce should succeed");
        assert!(!SpaceAgent::get_data(&space, &chan(0)).is_empty());

        CheckpointableSpace::clear(&mut space).expect("clear should succeed");
        assert!(SpaceAgent::get_data(&space, &chan(0)).is_empty());
    }

    #[test]
    fn test_builder() {
        let store = HashMapChannelStore::<TestChannel, TestPattern, TestData, TestCont, _, _>::new(
            BagDataCollection::new, BagContinuationCollection::new,
        );
        let matcher = WildcardMatch::<TestPattern, TestData>::new();

        let space: GenericRSpace<_, _> =
            GenericRSpaceBuilder::new()
                .with_channel_store(store)
                .with_matcher(matcher)
                .with_space_id(SpaceId::new(vec![1, 2, 3]))
                .with_qualifier(SpaceQualifier::Temp)
                .build()
                .expect("Builder should succeed with all required fields");

        assert_eq!(space.qualifier(), SpaceQualifier::Temp);
    }

    #[test]
    fn test_builder_incomplete_returns_error() {
        // Missing channel_store
        let result = GenericRSpaceBuilder::<
            HashMapChannelStore<TestChannel, TestPattern, TestData, TestCont,
                BagDataCollection<TestData>, BagContinuationCollection<TestPattern, TestCont>>,
            WildcardMatch<TestPattern, TestData>
        >::new()
            .with_matcher(WildcardMatch::<TestPattern, TestData>::new())
            .build();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SpaceError::BuilderIncomplete { builder: "GenericRSpaceBuilder", .. }),
            "Expected BuilderIncomplete error for missing channel_store, got: {:?}", err
        );

        // Missing matcher
        let store = HashMapChannelStore::<TestChannel, TestPattern, TestData, TestCont, _, _>::new(
            BagDataCollection::new, BagContinuationCollection::new,
        );
        let result = GenericRSpaceBuilder::<_, WildcardMatch<TestPattern, TestData>>::new()
            .with_channel_store(store)
            .build();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SpaceError::BuilderIncomplete { builder: "GenericRSpaceBuilder", .. }),
            "Expected BuilderIncomplete error for missing matcher, got: {:?}", err
        );
    }

    #[test]
    fn test_temp_space_checkpoint_succeeds() {
        // Temp spaces now support checkpointing - data is ephemeral and cleared on restore
        let store = HashMapChannelStore::<TestChannel, TestPattern, TestData, TestCont, _, _>::new(
            BagDataCollection::new, BagContinuationCollection::new,
        );
        let matcher = WildcardMatch::<TestPattern, TestData>::new();

        let mut space: GenericRSpace<_, _> =
            GenericRSpace::new(store, matcher, SpaceId::default_space(), SpaceQualifier::Temp);

        let result: Result<Checkpoint, SpaceError> = CheckpointableSpace::create_checkpoint(&mut space);
        assert!(result.is_ok(), "Temp spaces should support checkpointing");
    }

    #[test]
    fn test_debug_output() {
        let space = create_test_space();
        let debug = format!("{:?}", space);
        assert!(debug.contains("GenericRSpace"));
        assert!(debug.contains("Default"));
    }

    #[test]
    fn test_theory_not_configured() {
        let space = create_test_space();
        assert!(!space.has_theory());
        assert!(space.theory().is_none());

        // Validation should pass when no theory is configured
        let result = space.validate_term("anything");
        assert!(result.is_ok());
    }

    #[test]
    fn test_theory_validation_passes() {
        use crate::rust::interpreter::spaces::types::SimpleTypeTheory;

        let mut space = create_test_space();
        let theory = SimpleTypeTheory::new("TestTheory", vec!["Nat".to_string(), "Int".to_string()]);
        space.set_theory(Some(Box::new(theory)));

        assert!(space.has_theory());
        assert!(space.theory().is_some());
        assert_eq!(space.theory().unwrap().name(), "TestTheory");

        // These should pass validation
        assert!(space.validate_term("Nat(42)").is_ok());
        assert!(space.validate_term("Int(-5)").is_ok());
    }

    #[test]
    fn test_theory_validation_fails() {
        use crate::rust::interpreter::spaces::types::SimpleTypeTheory;

        let mut space = create_test_space();
        let theory = SimpleTypeTheory::new("NatOnly", vec!["Nat".to_string()]);
        space.set_theory(Some(Box::new(theory)));

        // This should fail validation
        let result = space.validate_term("String(hello)");
        assert!(result.is_err());

        if let Err(SpaceError::TheoryValidationError { theory_name, term, .. }) = result {
            assert_eq!(theory_name, "NatOnly");
            assert_eq!(term, "String(hello)");
        } else {
            panic!("Expected TheoryValidationError");
        }
    }

    #[test]
    fn test_builder_with_theory() {
        use crate::rust::interpreter::spaces::types::SimpleTypeTheory;

        let store = HashMapChannelStore::<TestChannel, TestPattern, TestData, TestCont, _, _>::new(
            BagDataCollection::new, BagContinuationCollection::new,
        );
        let matcher = WildcardMatch::<TestPattern, TestData>::new();
        let theory = SimpleTypeTheory::new("BuilderTheory", vec!["Test".to_string()]);

        let space: GenericRSpace<_, _> =
            GenericRSpaceBuilder::new()
                .with_channel_store(store)
                .with_matcher(matcher)
                .with_theory(Box::new(theory))
                .build()
                .expect("Builder should succeed with all required fields");

        assert!(space.has_theory());
        assert_eq!(space.theory().unwrap().name(), "BuilderTheory");
    }

    #[test]
    fn test_config_includes_theory() {
        use crate::rust::interpreter::spaces::types::SimpleTypeTheory;

        let mut space = create_test_space();
        let theory = SimpleTypeTheory::new("ConfigTheory", vec!["Test".to_string()]);
        space.set_theory(Some(Box::new(theory)));

        let config = space.config();
        assert!(config.theory.is_some());
        assert_eq!(config.theory.as_ref().unwrap().name(), "ConfigTheory");
    }

    #[test]
    fn test_debug_with_theory() {
        use crate::rust::interpreter::spaces::types::SimpleTypeTheory;

        let mut space = create_test_space();
        let theory = SimpleTypeTheory::new("DebugTheory", vec![]);
        space.set_theory(Some(Box::new(theory)));

        let debug = format!("{:?}", space);
        assert!(debug.contains("DebugTheory"));
    }

    // =========================================================================
    // PathMap Prefix Semantics Tests
    // =========================================================================

    use crate::rust::interpreter::spaces::channel_store::PathMapChannelStore;

    /// Create a PathMap-based space for testing prefix semantics.
    fn create_pathmap_space() -> GenericRSpace<
        PathMapChannelStore<String, String, String, BagDataCollection<String>, BagContinuationCollection<String, String>>,
        WildcardMatch<String, String>,
    > {
        let store = PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
        let matcher = WildcardMatch::<String, String>::new();
        GenericRSpace::new(store, matcher, SpaceId::default_space(), SpaceQualifier::Default)
    }

    #[test]
    fn test_pathmap_space_supports_prefix_semantics() {
        let space = create_pathmap_space();
        assert!(space.channel_store().supports_prefix_semantics());
    }

    #[test]
    fn test_consume_finds_data_at_descendant_path() {
        // Test that consume on @[0,1] finds data at @[0,1,2]
        let mut space = create_pathmap_space();

        // Produce data at @[0,1,2] (a child path)
        let child_path = vec![0u8, 1, 2];
        SpaceAgent::produce(&mut space, child_path.clone(), "hello".to_string(), false, None)
            .expect("produce should succeed");

        // Verify data exists at the child path
        assert_eq!(SpaceAgent::get_data(&space, &child_path).len(), 1);

        // Consume on @[0,1] (a prefix path)
        let prefix_path = vec![0u8, 1];
        let result = SpaceAgent::consume(
            &mut space,
            vec![prefix_path.clone()],
            vec!["*".to_string()], // Wildcard pattern
            "continuation".to_string(),
            false,
            BTreeSet::new(),
        ).expect("consume should succeed");

        // Should find the data at the child path
        assert!(result.is_some());
        let (cont_result, rspace_results) = result.unwrap();

        // The result should reference the actual path where data was found
        assert_eq!(rspace_results.len(), 1);
        assert_eq!(rspace_results[0].channel, child_path);
        assert_eq!(rspace_results[0].matched_datum, "hello");

        // Data should be removed from the child path
        assert_eq!(SpaceAgent::get_data(&space, &child_path).len(), 0);

        // The continuation patterns should reference the original consume channels
        assert_eq!(cont_result.channels, vec![prefix_path]);
    }

    #[test]
    fn test_consume_exact_match_takes_priority() {
        // When data exists at both the exact path and a descendant,
        // exact match should be found first
        let mut space = create_pathmap_space();

        // Produce data at both @[0,1] and @[0,1,2]
        let exact_path = vec![0u8, 1];
        let child_path = vec![0u8, 1, 2];

        SpaceAgent::produce(&mut space, child_path.clone(), "from_child".to_string(), false, None)
            .expect("produce should succeed");
        SpaceAgent::produce(&mut space, exact_path.clone(), "from_exact".to_string(), false, None)
            .expect("produce should succeed");

        // Consume on @[0,1] should find the exact match first
        let result = SpaceAgent::consume(
            &mut space,
            vec![exact_path.clone()],
            vec!["*".to_string()],
            "continuation".to_string(),
            false,
            BTreeSet::new(),
        ).expect("consume should succeed");

        assert!(result.is_some());
        let (_, rspace_results) = result.unwrap();
        assert_eq!(rspace_results[0].channel, exact_path);
        assert_eq!(rspace_results[0].matched_datum, "from_exact");

        // Exact path data should be removed, child path data should remain
        assert_eq!(SpaceAgent::get_data(&space, &exact_path).len(), 0);
        assert_eq!(SpaceAgent::get_data(&space, &child_path).len(), 1);
    }

    #[test]
    fn test_consume_stores_continuation_when_no_data() {
        // When no matching data exists, continuation should be stored
        let mut space = create_pathmap_space();

        let prefix_path = vec![0u8, 1];
        let result = SpaceAgent::consume(
            &mut space,
            vec![prefix_path.clone()],
            vec!["*".to_string()],
            "waiting".to_string(),
            false,
            BTreeSet::new(),
        ).expect("consume should succeed");

        // No immediate match
        assert!(result.is_none());

        // Continuation should be stored
        let conts = SpaceAgent::get_waiting_continuations(&space, vec![prefix_path]);
        assert_eq!(conts.len(), 1);
        assert_eq!(conts[0].continuation, "waiting");
    }

    #[test]
    fn test_consume_with_multiple_descendants() {
        // Test that consume finds data when multiple descendants exist
        let mut space = create_pathmap_space();

        // Produce data at @[0,1,2] and @[0,1,3]
        SpaceAgent::produce(&mut space, vec![0u8, 1, 2], "data_2".to_string(), false, None)
            .expect("produce should succeed");
        SpaceAgent::produce(&mut space, vec![0u8, 1, 3], "data_3".to_string(), false, None)
            .expect("produce should succeed");

        // Consume on @[0,1] should find one of them
        let prefix_path = vec![0u8, 1];
        let result = SpaceAgent::consume(
            &mut space,
            vec![prefix_path],
            vec!["*".to_string()],
            "continuation".to_string(),
            false,
            BTreeSet::new(),
        ).expect("consume should succeed");

        assert!(result.is_some());
        let (_, rspace_results) = result.unwrap();

        // Should have consumed one piece of data
        assert_eq!(rspace_results.len(), 1);

        // One data should remain, one should be consumed
        let remaining_2 = SpaceAgent::get_data(&space, &vec![0u8, 1, 2]).len();
        let remaining_3 = SpaceAgent::get_data(&space, &vec![0u8, 1, 3]).len();
        assert_eq!(remaining_2 + remaining_3, 1);
    }

    #[test]
    fn test_hashmap_space_unchanged() {
        // Verify that HashMap-based spaces still use exact matching
        let mut space = create_test_space();

        // Produce data at channel 1
        SpaceAgent::produce(&mut space, chan(1), 42, false, None).expect("produce should succeed");

        // Consume on channel 0 should NOT find data at channel 1
        let result = SpaceAgent::consume(
            &mut space,
            vec![chan(0)],
            vec![0i32], // Pattern that would match anything in WildcardMatch
            "continuation".to_string(),
            false,
            BTreeSet::new(),
        ).expect("consume should succeed");

        // No match should be found (no prefix semantics)
        assert!(result.is_none());

        // Data at channel 1 should still exist
        assert_eq!(SpaceAgent::get_data(&space, &chan(1)).len(), 1);
    }

    #[test]
    fn test_pathmap_spec_example() {
        // Test from spec lines 159-192:
        // @[0, 1, 2]!({|"hi"|}) | @[0, 1, 2]!({|"hello"|}) | @[0, 1, 3]!({|"there"|})
        // When receiving on @[0,1]:
        //   for( x <- @[0, 1] ) { P }
        // Gets x bound to data from one of the descendant paths
        let mut space = create_pathmap_space();

        // Store data at child paths as per spec
        SpaceAgent::produce(&mut space, vec![0u8, 1, 2], "hi".to_string(), false, None)
            .expect("produce should succeed");
        SpaceAgent::produce(&mut space, vec![0u8, 1, 2], "hello".to_string(), false, None)
            .expect("produce should succeed");
        SpaceAgent::produce(&mut space, vec![0u8, 1, 3], "there".to_string(), false, None)
            .expect("produce should succeed");

        // Verify data is stored
        assert_eq!(SpaceAgent::get_data(&space, &vec![0u8, 1, 2]).len(), 2);
        assert_eq!(SpaceAgent::get_data(&space, &vec![0u8, 1, 3]).len(), 1);

        // Consume on prefix @[0,1] - should find data from descendants
        let result = SpaceAgent::consume(
            &mut space,
            vec![vec![0u8, 1]],
            vec!["*".to_string()],
            "consumer".to_string(),
            false,
            BTreeSet::new(),
        ).expect("consume should succeed");

        assert!(result.is_some());
        let (_, rspace_results) = result.unwrap();

        // Should get one piece of data from one of the child paths
        assert_eq!(rspace_results.len(), 1);
        let data = &rspace_results[0].matched_datum;
        assert!(data == "hi" || data == "hello" || data == "there");

        // One less total data after consume
        let remaining = SpaceAgent::get_data(&space, &vec![0u8, 1, 2]).len()
            + SpaceAgent::get_data(&space, &vec![0u8, 1, 3]).len();
        assert_eq!(remaining, 2);
    }

    // =========================================================================
    // Phase 4 Tests: Produce triggers continuations at prefix paths
    // =========================================================================
    // Phase 4 is IMPLEMENTED: produce() checks prefix paths for continuations
    // via find_matching_continuation_at_prefix(). Data at @[0,1,2] triggers
    // continuations waiting on @[0,1].

    #[test]
    fn test_produce_on_child_stores_data_when_no_prefix_continuation() {
        // When there's no waiting continuation, produce stores data
        let mut space = create_pathmap_space();

        // Produce on @[0,1,2] with no waiting continuations
        let result = SpaceAgent::produce(&mut space, vec![0u8, 1, 2], "hello".to_string(), false, None)
            .expect("produce should succeed");

        // No continuation triggered - data stored
        assert!(result.is_none());
        assert_eq!(SpaceAgent::get_data(&space, &vec![0u8, 1, 2]).len(), 1);
    }

    #[test]
    fn test_produce_finds_continuation_at_exact_path() {
        // Produce should trigger continuation at exact path (existing behavior)
        let mut space = create_pathmap_space();

        let path = vec![0u8, 1, 2];

        // Set up a waiting continuation at exact path
        let consume_result = SpaceAgent::consume(
            &mut space,
            vec![path.clone()],
            vec!["*".to_string()],
            "waiting".to_string(),
            false,
            BTreeSet::new(),
        ).expect("consume should succeed");
        assert!(consume_result.is_none()); // No data yet, continuation stored

        // Produce on the same path should trigger the continuation
        let produce_result = SpaceAgent::produce(&mut space, path.clone(), "data".to_string(), false, None)
            .expect("produce should succeed");

        assert!(produce_result.is_some());
        let (cont_result, rspace_results, _) = produce_result.unwrap();
        assert_eq!(cont_result.continuation, "waiting");
        assert_eq!(rspace_results[0].matched_datum, "data");
    }

    #[test]
    fn test_produce_on_child_triggers_prefix_continuation() {
        // After Phase 4: produce on @[0,1,2] should trigger continuation at @[0,1]
        let mut space = create_pathmap_space();

        // Set up a waiting continuation at prefix path @[0,1]
        let consume_result = SpaceAgent::consume(
            &mut space,
            vec![vec![0u8, 1]],
            vec!["*".to_string()],
            "prefix_consumer".to_string(),
            false,
            BTreeSet::new(),
        ).expect("consume should succeed");
        assert!(consume_result.is_none()); // No data yet, continuation stored

        // Verify continuation is stored
        assert_eq!(SpaceAgent::get_waiting_continuations(&space, vec![vec![0u8, 1]]).len(), 1);

        // Produce on child path @[0,1,2] should trigger the prefix continuation
        let produce_result = SpaceAgent::produce(&mut space, vec![0u8, 1, 2], "hello".to_string(), false, None)
            .expect("produce should succeed");

        // This assertion will pass after Phase 4 is implemented
        assert!(produce_result.is_some());
        let (cont_result, rspace_results, _) = produce_result.unwrap();
        assert_eq!(cont_result.continuation, "prefix_consumer");
        assert_eq!(rspace_results[0].matched_datum, "hello");
        // The actual channel should be the child path
        assert_eq!(rspace_results[0].channel, vec![0u8, 1, 2]);

        // Continuation should be removed
        assert_eq!(SpaceAgent::get_waiting_continuations(&space, vec![vec![0u8, 1]]).len(), 0);
    }
}
