//! rho-vectordb - Vector database for Rholang tuple space operations
//!
//! This crate provides a pluggable vector database abstraction for similarity-based
//! pattern matching in Rholang's reified RSpaces. It supports multiple backend
//! implementations through a two-layer trait architecture.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  Rholang RSpace (uses VectorDB<Par>)            │
//! ├─────────────────────────────────────────────────┤
//! │  VectorDB<A> trait                              │
//! │  - DefaultVectorDB<A, B: VectorBackend>         │
//! ├─────────────────────────────────────────────────┤
//! │  VectorBackend trait                            │
//! │  - InMemoryBackend (default, uses ndarray/SIMD) │
//! │  - PineconeBackend (future)                     │
//! │  - QdrantBackend (future)                       │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **`full`** (default): Full implementation with SIMD-optimized in-memory backend
//! - **`types-only`**: Type definitions only (traits, enums, errors) - no implementations
//!
//! # Usage
//!
//! ```ignore
//! use rho_vectordb::prelude::*;
//!
//! // Create an in-memory backend
//! let backend = InMemoryBackend::new(384); // 384 dimensions
//!
//! // Create a VectorDB with the backend
//! let mut db: DefaultVectorDB<String, _> = DefaultVectorDB::new(backend);
//!
//! // Store data with embedding
//! db.put("document1".to_string(), vec![0.1, 0.2, ...]).unwrap();
//!
//! // Find similar items
//! if let Some((doc, score)) = db.consume_most_similar(&query_embedding, 0.8) {
//!     println!("Found: {} with score {}", doc, score);
//! }
//! ```

// ============================================================================
// Core Modules (always available)
// ============================================================================

pub mod error;
pub mod metrics;
pub mod backend;
pub mod db;
pub mod utils;

// ============================================================================
// Feature-Gated Modules
// ============================================================================

pub mod handlers;
pub mod ranking;
pub mod registry;

// Rholang integration - provides factory implementation for rholang's BackendRegistry
pub mod rholang_backend;

// ============================================================================
// Public Re-exports
// ============================================================================

pub use error::VectorDBError;
pub use metrics::{EmbeddingType, IndexConfig, SimilarityMetric};
pub use backend::VectorBackend;
pub use db::VectorDB;

pub use backend::{HandlerBackend, InMemoryBackend};
pub use db::DefaultVectorDB;

pub use handlers::{
    // Traits
    SimilarityMetricHandler, RankingFunctionHandler,
    // Types
    FunctionContext, IndexOptimizationData, SimilarityResult, RankingResult, ResolvedArg,
    // Similarity handlers
    CosineMetricHandler, DotProductMetricHandler, EuclideanMetricHandler,
    ManhattanMetricHandler, HammingMetricHandler, JaccardMetricHandler,
    // Ranking handlers
    TopKRankingHandler, AllRankingHandler, ThresholdRankingHandler,
    // Registries
    FunctionHandlerRegistry, SimilarityMetricRegistry, RankingFunctionRegistry,
};

// Rholang integration re-exports
pub use rholang_backend::register_with_rholang;

// ============================================================================
// Prelude Module for Ergonomic Imports
// ============================================================================

/// Convenient imports for common VectorDB usage.
///
/// ```ignore
/// use rho_vectordb::prelude::*;
/// ```
pub mod prelude {
    pub use super::backend::{HandlerBackend, InMemoryBackend};
    pub use super::backend::VectorBackend;
    pub use super::db::DefaultVectorDB;
    pub use super::db::VectorDB;
    pub use super::error::VectorDBError;
    pub use super::metrics::{EmbeddingType, SimilarityMetric};
    pub use super::handlers::{
        SimilarityMetricHandler, RankingFunctionHandler,
        FunctionContext, SimilarityResult, RankingResult, ResolvedArg,
        FunctionHandlerRegistry,
    };
}
