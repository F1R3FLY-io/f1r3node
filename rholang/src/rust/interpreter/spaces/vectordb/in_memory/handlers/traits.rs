//! Handler traits for VectorDB similarity and ranking operations.
//!
//! This module defines the core traits that handlers must implement to provide
//! custom similarity metrics and ranking functions.

use super::super::error::VectorDBError;
use ndarray::{Array1, Array2};

use super::types::{FunctionContext, RankingResult, ResolvedArg, SimilarityResult};

// ==========================================================================
// Similarity Metric Handler Trait
// ==========================================================================

/// Handler for a similarity metric (registered by metric ID).
///
/// Implementations compute similarity scores between a query vector and all
/// stored embeddings using a specific distance/similarity metric.
///
/// # Registration
///
/// Handlers are registered in a `SimilarityMetricRegistry` by their name and
/// optional aliases:
///
/// ```ignore
/// registry.register(Arc::new(CosineMetricHandler));  // registers "cosine" + alias "cos"
/// ```
///
/// # Usage from Rholang
///
/// ```rholang
/// for (resultCh <- docs ~ sim("cos", "0.8") ~ query) { ... }
/// ```
///
/// The backend resolves "cos" to `CosineMetricHandler` and passes `["0.8"]` as params.
///
/// # Pre-filtering Optimization
///
/// Backends that support pre-filtering (like `InMemoryBackend`) filter embeddings
/// to live entries **before** calling handlers. This provides significant speedup
/// when tombstone rates are high (> 30%).
///
/// When pre-filtering is used:
/// - `embeddings`: Matrix containing **only live entries** (L × D)
/// - `live_mask`: All 1.0s (provided for backward compatibility)
/// - Result includes `index_map` to translate score indices back to original IDs
///
/// Handlers **should NOT** apply masking when receiving pre-filtered embeddings.
/// The default handlers in this crate are optimized for pre-filtering and do not
/// apply post-masking.
///
/// # Legacy Backends (No Pre-filtering)
///
/// Backends that don't pre-filter may still pass full embedding matrices with
/// tombstoned entries. In this case:
/// - `embeddings`: Full matrix (N × D, including tombstoned)
/// - `live_mask`: Array with 1.0 for live, 0.0 for tombstoned
///
/// Handlers supporting legacy backends should apply the mask to zero out
/// tombstoned entries.
pub trait SimilarityMetricHandler: Send + Sync {
    /// Primary identifier (e.g., "cosine").
    fn name(&self) -> &str;

    /// Alternative names (e.g., ["cos"] for "cosine").
    fn aliases(&self) -> &[&str] {
        &[]
    }

    /// Compute similarity scores between query and all embeddings.
    ///
    /// # Arguments
    /// - `query`: Query embedding vector (already validated for dimension match)
    /// - `embeddings`: All stored embeddings (N x D matrix)
    /// - `live_mask`: Mask of live entries (1.0 = live, 0.0 = tombstoned)
    /// - `threshold`: Minimum similarity threshold (handler may adjust)
    /// - `extra_params`: Additional parameters from Rholang `sim(...)` call
    /// - `context`: Collection configuration context
    ///
    /// # Returns
    /// `SimilarityResult` with scores array and effective threshold
    fn compute(
        &self,
        query: &[f32],
        embeddings: &Array2<f32>,
        live_mask: &Array1<f32>,
        threshold: f32,
        extra_params: &[ResolvedArg],
        context: &FunctionContext,
    ) -> Result<SimilarityResult, VectorDBError>;
}

// ==========================================================================
// Ranking Function Handler Trait
// ==========================================================================

/// Handler for a ranking function (registered by ranking ID).
///
/// Implementations select and order results from similarity scores based on
/// ranking criteria (e.g., top-k, threshold filtering, diversity).
///
/// # Usage from Rholang
///
/// ```rholang
/// for (resultCh <- docs ~ rank("topk", 5) ~ query) { ... }
/// ```
pub trait RankingFunctionHandler: Send + Sync {
    /// Primary identifier (e.g., "topk").
    fn name(&self) -> &str;

    /// Alternative names (e.g., ["top"] for "topk").
    fn aliases(&self) -> &[&str] {
        &[]
    }

    /// Expected parameter count (min, max) excluding the function identifier.
    ///
    /// Used for arity validation at runtime.
    fn arity(&self) -> (usize, usize);

    /// Select and rank results from similarity scores.
    ///
    /// # Arguments
    /// - `similarity`: Result from similarity metric computation
    /// - `params`: Parameters from Rholang `rank(...)` call (excluding function ID)
    /// - `context`: Collection configuration context
    ///
    /// # Returns
    /// `RankingResult` with selected indices and scores, sorted by score descending
    fn rank(
        &self,
        similarity: &SimilarityResult,
        params: &[ResolvedArg],
        context: &FunctionContext,
    ) -> Result<RankingResult, VectorDBError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock handler for testing trait interface
    struct MockSimilarityHandler;

    impl SimilarityMetricHandler for MockSimilarityHandler {
        fn name(&self) -> &str {
            "mock"
        }

        fn aliases(&self) -> &[&str] {
            &["m"]
        }

        fn compute(
            &self,
            _query: &[f32],
            embeddings: &Array2<f32>,
            live_mask: &Array1<f32>,
            threshold: f32,
            _extra_params: &[ResolvedArg],
            _context: &FunctionContext,
        ) -> Result<SimilarityResult, VectorDBError> {
            // Return constant score of 0.9 for all live entries
            let scores = Array1::from_elem(embeddings.nrows(), 0.9) * live_mask;
            Ok(SimilarityResult::new(scores, threshold))
        }
    }

    struct MockRankingHandler;

    impl RankingFunctionHandler for MockRankingHandler {
        fn name(&self) -> &str {
            "mock"
        }

        fn arity(&self) -> (usize, usize) {
            (0, 0)
        }

        fn rank(
            &self,
            similarity: &SimilarityResult,
            _params: &[ResolvedArg],
            _context: &FunctionContext,
        ) -> Result<RankingResult, VectorDBError> {
            let matches: Vec<(usize, f32)> = similarity
                .scores
                .iter()
                .enumerate()
                .filter(|(_, &score)| score >= similarity.threshold)
                .map(|(idx, &score)| (idx, score))
                .collect();
            Ok(RankingResult::new(matches))
        }
    }

    #[test]
    fn test_similarity_handler_interface() {
        let handler = MockSimilarityHandler;
        assert_eq!(handler.name(), "mock");
        assert_eq!(handler.aliases(), &["m"]);

        let embeddings = Array2::zeros((3, 4));
        let live_mask = Array1::from_vec(vec![1.0, 1.0, 0.0]);
        let context = FunctionContext::default();

        let result = handler
            .compute(&[0.1, 0.2, 0.3, 0.4], &embeddings, &live_mask, 0.8, &[], &context)
            .unwrap();

        assert_eq!(result.scores.len(), 3);
        assert_eq!(result.threshold, 0.8);
    }

    #[test]
    fn test_ranking_handler_interface() {
        let handler = MockRankingHandler;
        assert_eq!(handler.name(), "mock");
        assert_eq!(handler.arity(), (0, 0));

        let scores = Array1::from_vec(vec![0.9, 0.5, 0.95]);
        let similarity = SimilarityResult::new(scores, 0.8);
        let context = FunctionContext::default();

        let result = handler.rank(&similarity, &[], &context).unwrap();
        assert_eq!(result.len(), 2); // Only scores >= 0.8
    }
}
