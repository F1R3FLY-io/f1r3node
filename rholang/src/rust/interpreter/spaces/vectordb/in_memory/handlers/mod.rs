//! Handler system for VectorDB similarity and ranking operations.
//!
//! This module provides a pluggable handler system for custom similarity metrics
//! and ranking functions. Backends can register handlers and use them for
//! query processing.
//!
//! # Architecture
//!
//! - **Types**: Core data structures (`ResolvedArg`, `FunctionContext`, etc.)
//! - **Traits**: Handler interfaces (`SimilarityMetricHandler`, `RankingFunctionHandler`)
//! - **Similarity**: Built-in similarity metric handlers (Cosine, Euclidean, etc.)
//! - **Ranking**: Built-in ranking function handlers (TopK, All, Threshold)
//! - **Registry**: Handler lookup and registration
//!
//! # Usage
//!
//! ```ignore
//! use rholang::rust::interpreter::spaces::vectordb::in_memory::handlers::{
//!     FunctionHandlerRegistry, SimilarityMetricHandler, CosineMetricHandler,
//!     FunctionContext, SimilarityResult,
//! };
//! use ndarray::{Array1, Array2};
//! use std::sync::Arc;
//!
//! // Create registry with default handlers
//! let registry = FunctionHandlerRegistry::with_defaults();
//!
//! // Look up a handler by name or alias
//! let handler = registry.get_similarity("cos").unwrap();
//!
//! // Compute similarity
//! let embeddings = Array2::zeros((10, 128));
//! let live_mask = Array1::ones(10);
//! let query = vec![0.0; 128];
//! let context = FunctionContext::default();
//!
//! let result = handler.compute(
//!     &query,
//!     &embeddings,
//!     &live_mask,
//!     0.8, // threshold
//!     &[],
//!     &context,
//! ).unwrap();
//! ```
//!
//! # Custom Handlers
//!
//! Implement `SimilarityMetricHandler` or `RankingFunctionHandler` to create
//! custom handlers:
//!
//! ```ignore
//! use rholang::rust::interpreter::spaces::vectordb::in_memory::handlers::{
//!     SimilarityMetricHandler, SimilarityResult, ResolvedArg, FunctionContext,
//! };
//! use rholang::rust::interpreter::spaces::vectordb::in_memory::error::VectorDBError;
//! use ndarray::{Array1, Array2};
//!
//! struct CustomMetricHandler;
//!
//! impl SimilarityMetricHandler for CustomMetricHandler {
//!     fn name(&self) -> &str { "custom" }
//!     fn aliases(&self) -> &[&str] { &["my_metric"] }
//!
//!     fn compute(
//!         &self,
//!         query: &[f32],
//!         embeddings: &Array2<f32>,
//!         live_mask: &Array1<f32>,
//!         threshold: f32,
//!         extra_params: &[ResolvedArg],
//!         context: &FunctionContext,
//!     ) -> Result<SimilarityResult, VectorDBError> {
//!         // Custom similarity computation
//!         let scores = Array1::ones(embeddings.nrows()) * live_mask;
//!         Ok(SimilarityResult::new(scores, threshold))
//!     }
//! }
//! ```

pub mod ranking;
pub mod registry;
pub mod similarity;
pub mod traits;
pub mod types;

// Re-export core types
pub use types::{
    FunctionContext, IndexOptimizationData, RankingResult, ResolvedArg, SimilarityResult,
};

// Re-export traits
pub use traits::{RankingFunctionHandler, SimilarityMetricHandler};

// Re-export similarity handlers
pub use similarity::{
    CosineMetricHandler, DotProductMetricHandler, EuclideanMetricHandler, HammingMetricHandler,
    JaccardMetricHandler, ManhattanMetricHandler,
};

// Re-export ranking handlers
pub use ranking::{AllRankingHandler, ThresholdRankingHandler, TopKRankingHandler};

// Re-export registries
pub use registry::{FunctionHandlerRegistry, RankingFunctionRegistry, SimilarityMetricRegistry};
