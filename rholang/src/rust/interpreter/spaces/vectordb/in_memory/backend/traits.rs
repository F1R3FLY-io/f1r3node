//! VectorBackend trait definition.
//!
//! This trait provides a high-level, backend-agnostic interface for vector
//! storage operations. Implementations can use any optimization strategy
//! internally (tombstones, SIMD, etc.) without exposing those details.
//!
//! # Trait Hierarchy
//!
//! - [`VectorBackend`]: Core backend trait for vector storage and retrieval
//! - [`HandlerBackend`]: Extension trait for backends that support custom handlers

use std::hash::Hash;

use super::super::error::VectorDBError;
use super::super::metrics::SimilarityMetric;

use super::super::handlers::{
    FunctionContext, FunctionHandlerRegistry, RankingResult, ResolvedArg, SimilarityResult,
};

use ndarray::{Array1, Array2};

/// Backend-agnostic vector storage abstraction.
///
/// This trait is intentionally minimal and high-level so that arbitrary
/// vector databases can implement it without being coupled to any specific
/// optimization strategy.
///
/// # Type Parameters
///
/// The associated `Id` type allows backends to use their own identifier scheme
/// (e.g., `usize` for in-memory, `String` for cloud services, `Uuid` for distributed).
///
/// # Implementation Notes
///
/// - Backends should pre-normalize vectors for cosine similarity efficiency
/// - The `find_similar` method should return results sorted by similarity descending
/// - Implementations can use any internal optimization (tombstones, SIMD, etc.)
///
/// # Example
///
/// ```ignore
/// use rho_vectordb::backend::VectorBackend;
///
/// struct MyBackend { /* ... */ }
///
/// impl VectorBackend for MyBackend {
///     type Id = usize;
///     // ... implement methods ...
/// }
/// ```
pub trait VectorBackend: Clone + Send + Sync {
    /// Opaque identifier for stored embeddings.
    ///
    /// This allows backends to use their preferred ID scheme.
    type Id: Clone + Eq + Hash + Send + Sync + std::fmt::Debug;

    /// Store an embedding and return its identifier.
    ///
    /// The backend may normalize the embedding internally for efficiency
    /// with certain similarity metrics.
    fn store(&mut self, embedding: &[f32]) -> Result<Self::Id, VectorDBError>;

    /// Retrieve an embedding by identifier.
    ///
    /// Returns `None` if the ID is not found or has been removed.
    fn get(&self, id: &Self::Id) -> Option<Vec<f32>>;

    /// Remove an embedding by identifier.
    ///
    /// Returns `true` if the embedding was found and removed, `false` otherwise.
    fn remove(&mut self, id: &Self::Id) -> bool;

    /// Find embeddings similar to the query.
    ///
    /// Returns (id, similarity_score) pairs sorted by similarity descending.
    /// Only results meeting the threshold are returned.
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding vector
    /// * `metric` - Similarity metric to use
    /// * `threshold` - Minimum similarity threshold (results below are excluded)
    /// * `limit` - Maximum number of results to return (`None` for unlimited)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The query dimensions don't match the backend's configured dimensions
    /// - The metric is not supported by this backend
    fn find_similar(
        &self,
        query: &[f32],
        metric: SimilarityMetric,
        threshold: f32,
        limit: Option<usize>,
    ) -> Result<Vec<(Self::Id, f32)>, VectorDBError>;

    /// Get the embedding dimensions configured for this backend.
    fn dimensions(&self) -> usize;

    /// Get the number of stored embeddings.
    fn len(&self) -> usize;

    /// Check if the backend is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all embeddings from the backend.
    fn clear(&mut self);

    /// Get the list of similarity metrics supported by this backend.
    ///
    /// The default implementation returns all metrics, but backends can
    /// override this to restrict support (e.g., cloud services may only
    /// support cosine and dot product).
    fn supported_metrics(&self) -> Vec<SimilarityMetric> {
        SimilarityMetric::all().to_vec()
    }

    /// Check if a specific metric is supported.
    fn supports_metric(&self, metric: SimilarityMetric) -> bool {
        self.supported_metrics().contains(&metric)
    }
}

// ============================================================================
// Handler Backend Extension Trait
// ============================================================================

/// Extension trait for backends that support custom similarity/ranking handlers.
///
/// This trait provides access to a pluggable handler system where similarity metrics
/// and ranking functions can be registered and looked up by name. This allows
/// Rholang patterns like:
///
/// ```rholang
/// for (resultCh <- docs ~ sim("cos", "0.8") ~ query) { ... }
/// ```
///
/// Where `"cos"` resolves to a registered similarity handler.
///
/// # Handler System
///
/// Backends implementing this trait maintain a [`FunctionHandlerRegistry`] that maps:
/// - Metric IDs → [`SimilarityMetricHandler`](super::super::handlers::SimilarityMetricHandler) implementations
/// - Ranking IDs → [`RankingFunctionHandler`](super::super::handlers::RankingFunctionHandler) implementations
///
/// # Usage
///
/// ```ignore
/// use rho_vectordb::{InMemoryBackend, HandlerBackend, FunctionContext, ResolvedArg};
///
/// let mut backend = InMemoryBackend::new(128);
///
/// // Register some embeddings
/// backend.store(&embedding1)?;
/// backend.store(&embedding2)?;
///
/// // Compute similarity using a handler
/// let context = FunctionContext::new(0.8, "cosine", 128, EmbeddingType::Float);
/// let result = backend.compute_similarity_with_handler(
///     &query,
///     "cosine",
///     0.8,
///     &[],
///     &context,
/// )?;
///
/// // Rank the results
/// let ranked = backend.rank_with_handler(&result, "topk", &[ResolvedArg::Integer(5)], &context)?;
/// ```
pub trait HandlerBackend: VectorBackend {
    /// Get the handler registry for this backend.
    fn handler_registry(&self) -> &FunctionHandlerRegistry;

    /// Get mutable handler registry for registration.
    fn handler_registry_mut(&mut self) -> &mut FunctionHandlerRegistry;

    /// Get the embeddings matrix (N x D).
    ///
    /// Returns the raw embedding matrix. For backends that normalize embeddings
    /// on storage, this returns the normalized values.
    fn embeddings_matrix(&self) -> &Array2<f32>;

    /// Get the live mask for filtering tombstoned entries.
    ///
    /// Returns an array where 1.0 = live entry, 0.0 = tombstoned.
    /// For compact backends (no tombstones), returns all 1.0s.
    fn live_mask(&self) -> Array1<f32>;

    /// Compute similarity using a registered handler.
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding vector
    /// * `metric_id` - Identifier of the similarity metric handler (e.g., "cosine", "cos")
    /// * `threshold` - Minimum similarity threshold
    /// * `extra_params` - Additional parameters for the handler
    /// * `context` - Function context with collection configuration
    ///
    /// # Returns
    ///
    /// `SimilarityResult` with per-embedding scores and effective threshold.
    fn compute_similarity_with_handler(
        &self,
        query: &[f32],
        metric_id: &str,
        threshold: f32,
        extra_params: &[ResolvedArg],
        context: &FunctionContext,
    ) -> Result<SimilarityResult, VectorDBError> {
        // Validate dimensions
        if query.len() != self.dimensions() {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimensions(),
                actual: query.len(),
            });
        }

        // Look up the handler
        let handler = self.handler_registry().get_similarity(metric_id).ok_or_else(|| {
            VectorDBError::UnsupportedMetric(format!(
                "Unknown similarity metric: '{}'. Available: {:?}",
                metric_id,
                self.handler_registry().similarity.names()
            ))
        })?;

        // Compute similarity
        let mask = self.live_mask();
        handler.compute(
            query,
            self.embeddings_matrix(),
            &mask,
            threshold,
            extra_params,
            context,
        )
    }

    /// Rank results using a registered handler.
    ///
    /// # Arguments
    ///
    /// * `similarity` - Result from similarity computation
    /// * `ranking_id` - Identifier of the ranking handler (e.g., "topk", "all")
    /// * `params` - Parameters for the ranking function
    /// * `context` - Function context with collection configuration
    ///
    /// # Returns
    ///
    /// `RankingResult` with selected indices and scores.
    fn rank_with_handler(
        &self,
        similarity: &SimilarityResult,
        ranking_id: &str,
        params: &[ResolvedArg],
        context: &FunctionContext,
    ) -> Result<RankingResult, VectorDBError> {
        // Look up the handler
        let handler = self.handler_registry().get_ranking(ranking_id).ok_or_else(|| {
            VectorDBError::InvalidArgument(format!(
                "Unknown ranking function: '{}'. Available: {:?}",
                ranking_id,
                self.handler_registry().ranking.names()
            ))
        })?;

        // Validate arity
        let (min_arity, max_arity) = handler.arity();
        if params.len() < min_arity || params.len() > max_arity {
            return Err(VectorDBError::InvalidArgument(format!(
                "Ranking function '{}' expects {}-{} parameters, got {}",
                ranking_id,
                min_arity,
                max_arity,
                params.len()
            )));
        }

        // Rank
        handler.rank(similarity, params, context)
    }

    /// Find similar embeddings using handler-based metric and ranking.
    ///
    /// Convenience method that combines similarity computation and ranking.
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding vector
    /// * `metric_id` - Similarity metric handler ID
    /// * `threshold` - Minimum similarity threshold
    /// * `ranking_id` - Ranking function handler ID
    /// * `ranking_params` - Parameters for ranking function
    /// * `context` - Function context
    ///
    /// # Returns
    ///
    /// Vector of (index, score) pairs sorted by score descending.
    fn find_similar_with_handlers(
        &self,
        query: &[f32],
        metric_id: &str,
        threshold: f32,
        ranking_id: &str,
        ranking_params: &[ResolvedArg],
        context: &FunctionContext,
    ) -> Result<Vec<(Self::Id, f32)>, VectorDBError>;
}
