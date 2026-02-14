//! Similarity metrics and index configuration for VectorDB operations.
//!
//! This module provides type definitions for similarity metrics, embedding types,
//! and index configuration for optimization strategies.

mod types;

// Re-export EmbeddingType from the parent vectordb types module
// to ensure there's only one EmbeddingType type in the crate.
pub use super::super::types::EmbeddingType;

pub use types::{
    SimilarityMetric,
    // Index configuration types
    IndexConfig,
    IndexType,
    HnswConfig,
    ScalarQuantizationConfig,
};
