//! Function handler registries for extensible VectorDB operations.
//!
//! This module provides registries for similarity metric handlers and
//! ranking function handlers, allowing extensible VectorDB operations.
//!
//! # Registries
//!
//! - [`SimilarityMetricRegistry`]: Registry for similarity metric handlers
//! - [`RankingFunctionRegistry`]: Registry for ranking function handlers
//! - [`FunctionHandlerRegistry`]: Combined registry for both
//!
//! See the [`handlers`](super::handlers) module for the full handler system.

// Re-export from handlers module
pub use super::handlers::registry::{
    FunctionHandlerRegistry, RankingFunctionRegistry, SimilarityMetricRegistry,
};
