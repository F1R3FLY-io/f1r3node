//! Multi-Space RSpace Integration Module
//!
//! This module implements the 6-layer trait hierarchy for reified RSpaces as specified
//! in the "Reifying RSpaces" specification. It enables support for multiple distinct
//! tuple spaces within a single Rholang runtime.
//!
//! # Layer Architecture
//!
//! 1. **Inner Collections** (`collections.rs`): Data/continuation storage at channels
//!    - Bag, Queue, Stack, Set, Cell, PriorityQueue, VectorDB
//!
//! 2. **Outer Storage** (`channel_store.rs`): Channel indexing structures
//!    - HashMap, PathMap, Array, Vector, HashSet
//!
//! 3. **Space Agent Core** (`agent.rs`): Core space operations
//!    - produce, consume, gensym, install
//!
//! 4. **Checkpointing** (`checkpoint.rs`): State management
//!    - create_checkpoint, reset, soft checkpoints
//!
//! 5. **Generic RSpace** (`generic_rspace.rs`): Parameterized implementation
//!    - Combines storage + matching strategies
//!
//! 6. **Space Factories** (`factory.rs`): Space construction
//!    - Create spaces from configurations
//!
//! # Supporting Modules
//!
//! - **Matcher** (`matcher.rs`): Pattern matching trait and implementations
//!   - Match<P, A> trait for pluggable matching strategies
//!   - ExactMatch, VectorDBMatch, WildcardMatch, etc.
//!
//! - **History** (`history.rs`): Checkpoint storage abstraction
//!   - HistoryStore trait for state persistence
//!   - InMemoryHistoryStore, BoundedHistoryStore, NullHistoryStore
//!
//! # Registry
//!
//! The `SpaceRegistry` (`registry.rs`) manages all space instances and provides:
//! - Space creation and lookup
//! - Channel-to-space routing
//! - Use block stack for scoped default spaces
//!
//! # Usage
//!
//! ```rholang
//! // Create a new space
//! new HMB(`rho:space:hashmap`), mySpace in {
//!   HMB!({}, *mySpace) |
//!   for (space <- mySpace) {
//!     // Use the space as default for nested operations
//!     use space {
//!       new ch in { ch!(42) }
//!     }
//!   }
//! }
//! ```

pub mod adapter;
pub mod agent;
pub mod async_agent;
pub mod channel_store;
pub mod charging_agent;
pub mod collections;
pub mod errors;
pub mod factory;
pub mod generic_rspace;
pub mod history;
pub mod matcher;
pub mod phlogiston;
pub mod prelude;
pub mod registry;
pub mod similarity_extraction;
pub mod types;
pub mod vectordb;

// Re-exports for convenience
pub use types::{
    InnerCollectionType,
    OuterStorageType,
    SpaceConfig,
    SpaceId,
    SpaceQualifier,
    // Type bound trait aliases (reduces where clause boilerplate)
    ChannelBound,
    PatternBound,
    DataBound,
    ContinuationBound,
    SpaceParamBound,
    // Theory types for MeTTaIL integration
    Theory,
    NullTheory,
    SimpleTypeTheory,
    BoxedTheory,
    // Validation traits for typed tuple spaces
    Validatable,
    TheoryValidator,
    ValidationResult,
    // Gas/Phlogiston configuration
    GasConfiguration,
    // PathMap prefix aggregation types
    SuffixKey,
    AggregatedDatum,
    get_path_suffix,
    path_prefixes,
    is_path_prefix,
    path_element_boundaries,
    // Par-to-Path conversion for Rholang PathMap integration
    par_to_path,
    path_to_par,
    is_par_path,
};
pub use errors::SpaceError;
pub use registry::SpaceRegistry;
pub use adapter::ISpaceAdapter;
pub use async_agent::{AsyncSpaceAgent, AsyncCheckpointableSpace, AsyncReplayableSpace};
pub use agent::{SpaceAgent, CheckpointableSpace, ReplayableSpace};

// New exports for spec alignment
pub use matcher::{
    Match, ExactMatch, VectorDBMatch, WildcardMatch, PredicateMatcher,
    BoxedMatch, VectorPattern, AndMatch, OrMatch, boxed,
};
pub use history::{HistoryStore, InMemoryHistoryStore, BoundedHistoryStore, NullHistoryStore, BoxedHistoryStore};
pub use generic_rspace::{GenericRSpace, GenericRSpaceBuilder, BagRSpace, ExtractedModifiers};
pub use similarity_extraction::{
    extract_embedding_from_par, extract_number_from_par, extract_threshold_from_par,
    extract_top_k_from_par, extract_metric_from_par, extract_rank_function_from_par,
    extract_modifiers_from_efunctions, extract_channel_id_from_par, extract_embedding_from_map,
    compute_cosine_similarity, compute_dot_product, compute_euclidean_similarity,
    compute_manhattan_similarity, compute_hamming_similarity, compute_jaccard_similarity,
    compute_similarity,
};
pub use collections::{DataCollectionExt, ContinuationCollectionExt, SimilarityCollection};
pub use phlogiston::{
    PhlogistonMeter, GasConfig, Operation,
    SEND_BASE_COST, SEND_PER_BYTE_COST, RECEIVE_BASE_COST,
    MATCH_BASE_COST, MATCH_PER_ELEMENT_COST, CHANNEL_CREATE_COST,
    CHECKPOINT_COST, SPACE_CREATE_COST,
};
pub use charging_agent::{ChargingSpaceAgent, ChargingAgentBuilder};

// Backend registry for VectorDB factory pattern
// Note: Backend implementations (like rho-vectordb) depend on rholang and register
// themselves with the BackendRegistry via their own `register_with_rholang()` functions.
pub use vectordb::registry::{
    BackendConfig, BackendRegistry, VectorBackendDyn, VectorBackendFactory, ResolvedArg,
};

// PathMap channel store for Rholang integration
pub use channel_store::RholangPathMapStore;

// Re-exports from rspace_plus_plus for checkpoint operations
pub use rspace_plus_plus::rspace::checkpoint::{Checkpoint, SoftCheckpoint};
pub use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;

// Factory exports including MeTTaIL integration hooks
pub use factory::{
    config_from_urn, urn_from_config, config_from_full_urn, parse_urn_with_theory,
    // Theory loading infrastructure
    TheorySpec, TheoryLoader, BuiltinTheoryLoader, SharedTheoryLoader,
    // Factory traits
    SpaceFactory, FactoryRegistry,
};

// Vector and tensor operations from the tensor module
// These operations are essential for vector/matrix computations across the crate.
pub use super::tensor::{
    // Element-wise operations
    sigmoid, temperature_sigmoid, softmax, heaviside, heaviside_f32,
    l2_normalize, l2_normalize_safe,
    // Binary vector operations
    majority,
    // Similarity operations
    cosine_similarity, cosine_similarity_safe, euclidean_distance, dot_product,
    // Matrix operations
    gram_matrix, cosine_similarity_matrix,
    // Tensor logic operations
    superposition, retrieval, top_k_similar,
    // Einsum operations
    einsum_2d, einsum_vm, einsum_mv,
    // Batch operations
    batch_matmul, batch_cosine_similarity,
    // Utility functions
    vec_to_array1, slice_to_array1, array1_to_vec, rows_to_array2,
    // Constants
    PARALLEL_THRESHOLD,
    // Hypervector operations (High-Dimensional Computing)
    bind, unbind, bundle, permute, unpermute, hamming_similarity, resonance,
};
