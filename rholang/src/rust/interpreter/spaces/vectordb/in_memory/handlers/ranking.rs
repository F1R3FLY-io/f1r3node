//! Default ranking function handlers for VectorDB.
//!
//! This module provides built-in ranking function implementations:
//! - TopK: Return up to K highest-scoring results
//! - All: Return all results above threshold
//! - Threshold: Filter by custom threshold (overrides similarity threshold)
//!
//! Ranking handlers return row indices which are translated to external IDs
//! by the backend's `find_similar_with_handlers()` method.

use super::super::error::VectorDBError;

use super::traits::RankingFunctionHandler;
use super::types::{FunctionContext, RankingResult, ResolvedArg, SimilarityResult};

// ==========================================================================
// Top-K Ranking
// ==========================================================================

/// Top-K ranking function handler.
///
/// Returns up to K results sorted by similarity descending.
/// Arity: exactly 1 parameter (K)
pub struct TopKRankingHandler;

impl RankingFunctionHandler for TopKRankingHandler {
    fn name(&self) -> &str {
        "topk"
    }

    fn aliases(&self) -> &[&str] {
        &["top"]
    }

    fn arity(&self) -> (usize, usize) {
        (1, 1) // Exactly 1 param: K
    }

    fn rank(
        &self,
        similarity: &SimilarityResult,
        params: &[ResolvedArg],
        _context: &FunctionContext,
    ) -> Result<RankingResult, VectorDBError> {
        let k = params
            .first()
            .and_then(|p| p.as_integer())
            .ok_or_else(|| VectorDBError::InvalidArgument("topk requires integer K".to_string()))?
            as usize;

        let mut matches: Vec<(usize, f32)> = similarity
            .scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score >= similarity.threshold)
            .map(|(idx, &score)| (idx, score))
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(k);

        Ok(RankingResult::new(matches))
    }
}

// ==========================================================================
// All-Matches Ranking
// ==========================================================================

/// All-matches ranking function handler.
///
/// Returns all results above threshold sorted by similarity descending.
/// Arity: 0 parameters
pub struct AllRankingHandler;

impl RankingFunctionHandler for AllRankingHandler {
    fn name(&self) -> &str {
        "all"
    }

    fn arity(&self) -> (usize, usize) {
        (0, 0) // No params
    }

    fn rank(
        &self,
        similarity: &SimilarityResult,
        _params: &[ResolvedArg],
        _context: &FunctionContext,
    ) -> Result<RankingResult, VectorDBError> {
        let mut matches: Vec<(usize, f32)> = similarity
            .scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score >= similarity.threshold)
            .map(|(idx, &score)| (idx, score))
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(RankingResult::new(matches))
    }
}

// ==========================================================================
// Threshold-Filter Ranking
// ==========================================================================

/// Threshold-filter ranking function handler.
///
/// Returns results above a custom threshold (overrides sim threshold).
/// Arity: exactly 1 parameter (threshold float)
pub struct ThresholdRankingHandler;

impl RankingFunctionHandler for ThresholdRankingHandler {
    fn name(&self) -> &str {
        "threshold"
    }

    fn aliases(&self) -> &[&str] {
        &["filter"]
    }

    fn arity(&self) -> (usize, usize) {
        (1, 1) // Exactly 1 param: threshold
    }

    fn rank(
        &self,
        similarity: &SimilarityResult,
        params: &[ResolvedArg],
        _context: &FunctionContext,
    ) -> Result<RankingResult, VectorDBError> {
        let custom_threshold = params.first().and_then(|p| p.as_float()).ok_or_else(|| {
            VectorDBError::InvalidArgument("threshold filter requires float threshold".to_string())
        })? as f32;

        let mut matches: Vec<(usize, f32)> = similarity
            .scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score >= custom_threshold)
            .map(|(idx, &score)| (idx, score))
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(RankingResult::new(matches))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn setup_similarity_result() -> SimilarityResult {
        // Scores: [0.9, 0.5, 0.7, 0.3, 0.8]
        let scores = Array1::from_vec(vec![0.9, 0.5, 0.7, 0.3, 0.8]);
        SimilarityResult::new(scores, 0.6) // threshold = 0.6
    }

    #[test]
    fn test_topk_handler() {
        let handler = TopKRankingHandler;
        assert_eq!(handler.name(), "topk");
        assert!(handler.aliases().contains(&"top"));
        assert_eq!(handler.arity(), (1, 1));

        let similarity = setup_similarity_result();
        let context = FunctionContext::default();

        // Request top 2
        let params = vec![ResolvedArg::Integer(2)];
        let result = handler
            .rank(&similarity, &params, &context)
            .expect("rank should succeed");

        // Above threshold (0.6): indices 0(0.9), 2(0.7), 4(0.8)
        // Top 2 by score: 0(0.9), 4(0.8)
        assert_eq!(result.len(), 2);
        assert_eq!(result.matches[0], (0, 0.9));
        assert_eq!(result.matches[1], (4, 0.8));
    }

    #[test]
    fn test_topk_handler_larger_k() {
        let handler = TopKRankingHandler;
        let similarity = setup_similarity_result();
        let context = FunctionContext::default();

        // Request top 10, but only 3 above threshold
        let params = vec![ResolvedArg::Integer(10)];
        let result = handler
            .rank(&similarity, &params, &context)
            .expect("rank should succeed");

        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_topk_handler_missing_param() {
        let handler = TopKRankingHandler;
        let similarity = setup_similarity_result();
        let context = FunctionContext::default();

        let result = handler.rank(&similarity, &[], &context);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_handler() {
        let handler = AllRankingHandler;
        assert_eq!(handler.name(), "all");
        assert_eq!(handler.arity(), (0, 0));

        let similarity = setup_similarity_result();
        let context = FunctionContext::default();

        let result = handler
            .rank(&similarity, &[], &context)
            .expect("rank should succeed");

        // Above threshold (0.6): indices 0(0.9), 2(0.7), 4(0.8)
        // Sorted by score descending
        assert_eq!(result.len(), 3);
        assert_eq!(result.matches[0], (0, 0.9));
        assert_eq!(result.matches[1], (4, 0.8));
        assert_eq!(result.matches[2], (2, 0.7));
    }

    #[test]
    fn test_threshold_handler() {
        let handler = ThresholdRankingHandler;
        assert_eq!(handler.name(), "threshold");
        assert!(handler.aliases().contains(&"filter"));
        assert_eq!(handler.arity(), (1, 1));

        let similarity = setup_similarity_result();
        let context = FunctionContext::default();

        // Use custom threshold 0.4 (lower than similarity's 0.6)
        let params = vec![ResolvedArg::Float(0.4)];
        let result = handler
            .rank(&similarity, &params, &context)
            .expect("rank should succeed");

        // Above threshold (0.4): indices 0(0.9), 1(0.5), 2(0.7), 4(0.8)
        // Sorted by score descending
        assert_eq!(result.len(), 4);
        assert_eq!(result.matches[0], (0, 0.9));
        assert_eq!(result.matches[1], (4, 0.8));
        assert_eq!(result.matches[2], (2, 0.7));
        assert_eq!(result.matches[3], (1, 0.5));
    }

    #[test]
    fn test_threshold_handler_higher_threshold() {
        let handler = ThresholdRankingHandler;
        let similarity = setup_similarity_result();
        let context = FunctionContext::default();

        // Use custom threshold 0.85 (higher than any except index 0)
        let params = vec![ResolvedArg::Float(0.85)];
        let result = handler
            .rank(&similarity, &params, &context)
            .expect("rank should succeed");

        assert_eq!(result.len(), 1);
        assert_eq!(result.matches[0], (0, 0.9));
    }

    #[test]
    fn test_threshold_handler_missing_param() {
        let handler = ThresholdRankingHandler;
        let similarity = setup_similarity_result();
        let context = FunctionContext::default();

        let result = handler.rank(&similarity, &[], &context);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_results() {
        let handler = AllRankingHandler;
        let context = FunctionContext::default();

        // All scores below threshold
        let scores = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let similarity = SimilarityResult::new(scores, 0.5);

        let result = handler
            .rank(&similarity, &[], &context)
            .expect("rank should succeed");

        assert!(result.is_empty());
        assert_eq!(result.best(), None);
    }
}
