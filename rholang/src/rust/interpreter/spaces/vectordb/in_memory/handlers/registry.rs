//! Handler registries for VectorDB similarity and ranking operations.
//!
//! This module provides registry types that map metric/function identifiers
//! to their handler implementations. Registries support:
//! - Case-insensitive lookups
//! - Alias registration (e.g., "cos" → CosineMetricHandler)
//! - Default handler registration

use std::collections::HashMap;
use std::sync::Arc;

use super::ranking::{AllRankingHandler, ThresholdRankingHandler, TopKRankingHandler};
use super::similarity::{
    CosineMetricHandler, DotProductMetricHandler, EuclideanMetricHandler, HammingMetricHandler,
    JaccardMetricHandler, ManhattanMetricHandler,
};
use super::traits::{RankingFunctionHandler, SimilarityMetricHandler};

// ==========================================================================
// Similarity Metric Registry
// ==========================================================================

/// Registry for similarity metric handlers.
///
/// Maps metric identifiers (case-insensitive) to handler implementations.
/// Supports aliases for convenient lookups (e.g., "cos" for "cosine").
#[derive(Clone)]
pub struct SimilarityMetricRegistry {
    handlers: HashMap<String, Arc<dyn SimilarityMetricHandler>>,
}

impl Default for SimilarityMetricRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SimilarityMetricRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a handler (also registers its aliases).
    pub fn register(&mut self, handler: Arc<dyn SimilarityMetricHandler>) {
        let name = handler.name().to_lowercase();
        self.handlers.insert(name.clone(), Arc::clone(&handler));

        for alias in handler.aliases() {
            self.handlers
                .insert(alias.to_lowercase(), Arc::clone(&handler));
        }
    }

    /// Look up handler by ID (case-insensitive).
    pub fn get(&self, id: &str) -> Option<Arc<dyn SimilarityMetricHandler>> {
        self.handlers.get(&id.to_lowercase()).cloned()
    }

    /// Check if a handler is registered for the given ID.
    pub fn contains(&self, id: &str) -> bool {
        self.handlers.contains_key(&id.to_lowercase())
    }

    /// Get all registered handler names (not including aliases).
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .handlers
            .values()
            .map(|h| h.name().to_string())
            .collect();
        names.sort();
        names.dedup();
        names
    }
}

// ==========================================================================
// Ranking Function Registry
// ==========================================================================

/// Registry for ranking function handlers.
///
/// Maps ranking function identifiers (case-insensitive) to handler implementations.
/// Supports aliases for convenient lookups (e.g., "top" for "topk").
#[derive(Clone)]
pub struct RankingFunctionRegistry {
    handlers: HashMap<String, Arc<dyn RankingFunctionHandler>>,
}

impl Default for RankingFunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RankingFunctionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a handler (also registers its aliases).
    pub fn register(&mut self, handler: Arc<dyn RankingFunctionHandler>) {
        let name = handler.name().to_lowercase();
        self.handlers.insert(name.clone(), Arc::clone(&handler));

        for alias in handler.aliases() {
            self.handlers
                .insert(alias.to_lowercase(), Arc::clone(&handler));
        }
    }

    /// Look up handler by ID (case-insensitive).
    pub fn get(&self, id: &str) -> Option<Arc<dyn RankingFunctionHandler>> {
        self.handlers.get(&id.to_lowercase()).cloned()
    }

    /// Check if a handler is registered for the given ID.
    pub fn contains(&self, id: &str) -> bool {
        self.handlers.contains_key(&id.to_lowercase())
    }

    /// Get all registered handler names (not including aliases).
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .handlers
            .values()
            .map(|h| h.name().to_string())
            .collect();
        names.sort();
        names.dedup();
        names
    }
}

// ==========================================================================
// Combined Function Handler Registry
// ==========================================================================

/// Combined registry for VectorDB backend.
///
/// Provides access to both similarity metric and ranking function handlers.
/// This is the primary interface for backend handler registration and lookup.
#[derive(Clone)]
pub struct FunctionHandlerRegistry {
    /// Registry for similarity metric handlers.
    pub similarity: SimilarityMetricRegistry,
    /// Registry for ranking function handlers.
    pub ranking: RankingFunctionRegistry,
}

impl Default for FunctionHandlerRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl FunctionHandlerRegistry {
    /// Create an empty registry (no default handlers).
    pub fn new() -> Self {
        Self {
            similarity: SimilarityMetricRegistry::new(),
            ranking: RankingFunctionRegistry::new(),
        }
    }

    /// Create registry with default handlers.
    ///
    /// Default similarity metrics: cosine, dotproduct, euclidean, manhattan, hamming, jaccard
    /// Default ranking functions: topk, all, threshold
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // Register default similarity handlers
        registry.register_similarity(Arc::new(CosineMetricHandler));
        registry.register_similarity(Arc::new(DotProductMetricHandler));
        registry.register_similarity(Arc::new(EuclideanMetricHandler));
        registry.register_similarity(Arc::new(ManhattanMetricHandler));
        registry.register_similarity(Arc::new(HammingMetricHandler));
        registry.register_similarity(Arc::new(JaccardMetricHandler));

        // Register default ranking handlers
        registry.register_ranking(Arc::new(TopKRankingHandler));
        registry.register_ranking(Arc::new(AllRankingHandler));
        registry.register_ranking(Arc::new(ThresholdRankingHandler));

        registry
    }

    /// Register custom similarity metric.
    pub fn register_similarity(&mut self, handler: Arc<dyn SimilarityMetricHandler>) {
        self.similarity.register(handler);
    }

    /// Register custom ranking function.
    pub fn register_ranking(&mut self, handler: Arc<dyn RankingFunctionHandler>) {
        self.ranking.register(handler);
    }

    /// Look up similarity handler by ID.
    pub fn get_similarity(&self, id: &str) -> Option<Arc<dyn SimilarityMetricHandler>> {
        self.similarity.get(id)
    }

    /// Look up ranking handler by ID.
    pub fn get_ranking(&self, id: &str) -> Option<Arc<dyn RankingFunctionHandler>> {
        self.ranking.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_registry_lookup() {
        let mut registry = SimilarityMetricRegistry::new();
        registry.register(Arc::new(CosineMetricHandler));

        // Primary name
        assert!(registry.get("cosine").is_some());
        // Case insensitive
        assert!(registry.get("COSINE").is_some());
        assert!(registry.get("Cosine").is_some());
        // Alias
        assert!(registry.get("cos").is_some());
        // Non-existent
        assert!(registry.get("unknown").is_none());
    }

    #[test]
    fn test_ranking_registry_lookup() {
        let mut registry = RankingFunctionRegistry::new();
        registry.register(Arc::new(TopKRankingHandler));

        // Primary name
        assert!(registry.get("topk").is_some());
        // Case insensitive
        assert!(registry.get("TOPK").is_some());
        // Alias
        assert!(registry.get("top").is_some());
        // Non-existent
        assert!(registry.get("unknown").is_none());
    }

    #[test]
    fn test_combined_registry_defaults() {
        let registry = FunctionHandlerRegistry::with_defaults();

        // Similarity handlers
        assert!(registry.get_similarity("cosine").is_some());
        assert!(registry.get_similarity("cos").is_some());
        assert!(registry.get_similarity("dotproduct").is_some());
        assert!(registry.get_similarity("dot").is_some());
        assert!(registry.get_similarity("euclidean").is_some());
        assert!(registry.get_similarity("l2").is_some());
        assert!(registry.get_similarity("manhattan").is_some());
        assert!(registry.get_similarity("l1").is_some());
        assert!(registry.get_similarity("hamming").is_some());
        assert!(registry.get_similarity("jaccard").is_some());

        // Ranking handlers
        assert!(registry.get_ranking("topk").is_some());
        assert!(registry.get_ranking("top").is_some());
        assert!(registry.get_ranking("all").is_some());
        assert!(registry.get_ranking("threshold").is_some());
        assert!(registry.get_ranking("filter").is_some());
    }

    #[test]
    fn test_empty_registry() {
        let registry = FunctionHandlerRegistry::new();

        assert!(registry.get_similarity("cosine").is_none());
        assert!(registry.get_ranking("topk").is_none());
    }

    #[test]
    fn test_registry_contains() {
        let mut registry = SimilarityMetricRegistry::new();
        registry.register(Arc::new(CosineMetricHandler));

        assert!(registry.contains("cosine"));
        assert!(registry.contains("cos"));
        assert!(!registry.contains("euclidean"));
    }

    #[test]
    fn test_registry_names() {
        let mut registry = SimilarityMetricRegistry::new();
        registry.register(Arc::new(CosineMetricHandler));
        registry.register(Arc::new(EuclideanMetricHandler));

        let names = registry.names();
        assert!(names.contains(&"cosine".to_string()));
        assert!(names.contains(&"euclidean".to_string()));
        assert_eq!(names.len(), 2); // Aliases not included
    }
}
