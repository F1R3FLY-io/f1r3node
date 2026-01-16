//! Handler integration tests for rho-vectordb.
//!
//! These tests validate the handler dispatch system, registry operations,
//! and custom handler registration.

use rholang::rust::interpreter::spaces::vectordb::in_memory::{
    InMemoryBackend, VectorBackend, HandlerBackend,
    EmbeddingType, FunctionContext, ResolvedArg,
};
use rholang::rust::interpreter::spaces::vectordb::in_memory::handlers::{
    SimilarityMetricHandler, RankingFunctionHandler,
    FunctionHandlerRegistry, SimilarityResult,
    // Built-in similarity handlers
    CosineMetricHandler,
    // Built-in ranking handlers
    TopKRankingHandler, ThresholdRankingHandler, AllRankingHandler,
};
use ndarray::Array1;

// ==========================================================================
// Similarity Handler Dispatch Tests
// ==========================================================================

#[test]
fn test_cosine_handler_dispatch() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let result = backend.find_similar_with_handlers(
        &emb, "cosine", 0.0, "all", &[], &context
    );

    assert!(result.is_ok(), "cosine should dispatch correctly");
    let results = result.unwrap();
    assert!(!results.is_empty(), "should return results");
    assert!((results[0].1 - 1.0).abs() < 1e-5, "self-similarity should be 1.0");
}

#[test]
fn test_euclidean_handler_dispatch() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let result = backend.find_similar_with_handlers(
        &emb, "euclidean", 0.0, "all", &[], &context
    );

    assert!(result.is_ok(), "euclidean should dispatch correctly");
}

#[test]
fn test_dotproduct_handler_dispatch() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let result = backend.find_similar_with_handlers(
        &emb, "dotproduct", 0.0, "all", &[], &context
    );

    assert!(result.is_ok(), "dotproduct should dispatch correctly");
}

#[test]
fn test_manhattan_handler_dispatch() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let result = backend.find_similar_with_handlers(
        &emb, "manhattan", 0.0, "all", &[], &context
    );

    assert!(result.is_ok(), "manhattan should dispatch correctly");
}

#[test]
fn test_hamming_handler_dispatch() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let result = backend.find_similar_with_handlers(
        &emb, "hamming", 0.0, "all", &[], &context
    );

    assert!(result.is_ok(), "hamming should dispatch correctly");
}

#[test]
fn test_jaccard_handler_dispatch() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let result = backend.find_similar_with_handlers(
        &emb, "jaccard", 0.0, "all", &[], &context
    );

    assert!(result.is_ok(), "jaccard should dispatch correctly");
}

#[test]
fn test_all_similarity_metrics_dispatch() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let metrics = vec!["cosine", "euclidean", "dotproduct", "manhattan", "hamming", "jaccard"];

    for metric in metrics {
        let result = backend.find_similar_with_handlers(
            &emb, metric, 0.0, "all", &[], &context
        );
        assert!(result.is_ok(), "{} should dispatch correctly", metric);
    }
}

// ==========================================================================
// Ranking Handler Dispatch Tests
// ==========================================================================

#[test]
fn test_topk_handler_dispatch() {
    let mut backend = InMemoryBackend::new(8);
    for _ in 0..10 {
        let emb = vec![1.0f32; 8];
        backend.store(&emb).expect("store");
    }

    let query = vec![1.0f32; 8];
    let context = FunctionContext::default();

    let result = backend.find_similar_with_handlers(
        &query, "cosine", 0.0, "topk", &[ResolvedArg::Integer(5)], &context
    );

    assert!(result.is_ok(), "topk should dispatch correctly");
    assert_eq!(result.unwrap().len(), 5, "should return 5 results");
}

#[test]
fn test_threshold_handler_dispatch() {
    let mut backend = InMemoryBackend::new(4);

    // Store vectors with varying similarity
    let v1 = vec![1.0, 0.0, 0.0, 0.0];
    let v2 = vec![0.9, 0.4, 0.0, 0.0]; // Similar
    let v3 = vec![0.0, 1.0, 0.0, 0.0]; // Orthogonal

    // Normalize
    let normalize = |v: Vec<f32>| {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect::<Vec<f32>>()
    };

    backend.store(&v1).expect("store");
    backend.store(&normalize(v2)).expect("store");
    backend.store(&v3).expect("store");

    let context = FunctionContext::default();

    let result = backend.find_similar_with_handlers(
        &v1, "cosine", 0.5, "threshold", &[ResolvedArg::Float(0.5)], &context
    );

    assert!(result.is_ok(), "threshold should dispatch correctly");
    let results = result.unwrap();
    // Only v1 and v2 should pass threshold 0.5
    assert!(results.len() >= 1, "should return results above threshold");
}

#[test]
fn test_all_handler_dispatch() {
    let mut backend = InMemoryBackend::new(8);
    for _ in 0..20 {
        let emb = vec![1.0f32; 8];
        backend.store(&emb).expect("store");
    }

    let query = vec![1.0f32; 8];
    let context = FunctionContext::default();

    let result = backend.find_similar_with_handlers(
        &query, "cosine", 0.0, "all", &[], &context
    );

    assert!(result.is_ok(), "all should dispatch correctly");
    assert_eq!(result.unwrap().len(), 20, "should return all results");
}

// ==========================================================================
// Handler Alias Resolution Tests
// ==========================================================================

#[test]
fn test_cosine_aliases() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();

    // "cos" is an alias for "cosine"
    let cos_result = backend.find_similar_with_handlers(
        &emb, "cos", 0.0, "all", &[], &context
    );

    assert!(cos_result.is_ok(), "cos alias should work");
}

#[test]
fn test_handler_alias_same_results() {
    let mut backend = InMemoryBackend::new(8);

    let v1 = vec![1.0f32; 8];
    let v2 = vec![0.5f32; 8];
    backend.store(&v1).expect("store v1");
    backend.store(&v2).expect("store v2");

    let context = FunctionContext::default();
    let query = vec![1.0f32; 8];

    // Query with "cos"
    let cos_results = backend.find_similar_with_handlers(
        &query, "cos", 0.0, "all", &[], &context
    ).expect("cos query");

    // Query with "cosine"
    let cosine_results = backend.find_similar_with_handlers(
        &query, "cosine", 0.0, "all", &[], &context
    ).expect("cosine query");

    // Results should be identical
    assert_eq!(cos_results.len(), cosine_results.len(), "same result count");

    for i in 0..cos_results.len() {
        assert_eq!(cos_results[i].0, cosine_results[i].0, "same IDs");
        assert!((cos_results[i].1 - cosine_results[i].1).abs() < 1e-5, "same scores");
    }
}

// ==========================================================================
// Handler Registry Tests
// ==========================================================================

#[test]
fn test_similarity_registry_contains_all_metrics() {
    // Use FunctionHandlerRegistry::default() which registers all default handlers
    let registry = FunctionHandlerRegistry::default();

    assert!(registry.similarity.contains("cosine"), "should have cosine");
    assert!(registry.similarity.contains("euclidean"), "should have euclidean");
    assert!(registry.similarity.contains("dotproduct"), "should have dotproduct");
    assert!(registry.similarity.contains("manhattan"), "should have manhattan");
    assert!(registry.similarity.contains("hamming"), "should have hamming");
    assert!(registry.similarity.contains("jaccard"), "should have jaccard");
}

#[test]
fn test_ranking_registry_contains_all_functions() {
    // Use FunctionHandlerRegistry::default() which registers all default handlers
    let registry = FunctionHandlerRegistry::default();

    assert!(registry.ranking.contains("topk"), "should have topk");
    assert!(registry.ranking.contains("threshold"), "should have threshold");
    assert!(registry.ranking.contains("all"), "should have all");
}

#[test]
fn test_combined_registry_operations() {
    let registry = FunctionHandlerRegistry::default();

    // Check similarity handlers
    assert!(registry.similarity.contains("cosine"));
    assert!(registry.similarity.contains("euclidean"));

    // Check ranking handlers
    assert!(registry.ranking.contains("topk"));
    assert!(registry.ranking.contains("all"));
}

#[test]
fn test_registry_names() {
    let registry = FunctionHandlerRegistry::default();
    let names = registry.similarity.names();

    assert!(names.contains(&"cosine".to_string()) || names.contains(&"cos".to_string()),
        "should list cosine or cos");
}

// ==========================================================================
// Handler Result Validation Tests
// ==========================================================================

#[test]
fn test_topk_results_sorted_descending() {
    let mut backend = InMemoryBackend::new(8);

    // Store vectors with varying similarity to query
    let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for i in 0..10 {
        let mut v = vec![0.0f32; 8];
        v[0] = 1.0 - (i as f32 * 0.1); // Decreasing similarity
        v[1] = i as f32 * 0.1;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v: Vec<f32> = v.iter().map(|x| x / norm).collect();
        backend.store(&v).expect("store");
    }

    let context = FunctionContext::default();
    let results = backend.find_similar_with_handlers(
        &query, "cosine", 0.0, "topk", &[ResolvedArg::Integer(10)], &context
    ).expect("topk");

    // Verify sorted descending
    for window in results.windows(2) {
        assert!(window[0].1 >= window[1].1,
            "results should be sorted descending: {} >= {}", window[0].1, window[1].1);
    }
}

#[test]
fn test_threshold_results_above_cutoff() {
    let mut backend = InMemoryBackend::new(4);

    // Store vectors with known similarities
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let similar = vec![0.95, 0.31, 0.0, 0.0]; // ~cos 0.95
    let moderate = vec![0.7, 0.71, 0.0, 0.0]; // ~cos 0.7
    let dissimilar = vec![0.0, 1.0, 0.0, 0.0]; // ~cos 0.0

    let normalize = |v: Vec<f32>| {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect::<Vec<f32>>()
    };

    backend.store(&query).expect("store");
    backend.store(&normalize(similar)).expect("store");
    backend.store(&normalize(moderate)).expect("store");
    backend.store(&dissimilar).expect("store");

    let context = FunctionContext::default();
    let results = backend.find_similar_with_handlers(
        &query, "cosine", 0.6, "threshold", &[ResolvedArg::Float(0.6)], &context
    ).expect("threshold");

    // All results should be >= 0.6
    for (_, score) in &results {
        assert!(*score >= 0.6, "score {} should be >= 0.6", score);
    }
}

#[test]
fn test_all_results_complete() {
    let mut backend = InMemoryBackend::new(8);

    let count = 15;
    for i in 0..count {
        let mut v = vec![0.0f32; 8];
        v[i % 8] = 1.0;
        backend.store(&v).expect("store");
    }

    let query = vec![1.0f32; 8];
    let context = FunctionContext::default();

    let results = backend.find_similar_with_handlers(
        &query, "cosine", 0.0, "all", &[], &context
    ).expect("all");

    assert_eq!(results.len(), count, "all should return all {} results", count);
}

// ==========================================================================
// Handler with Different Contexts Tests
// ==========================================================================

#[test]
fn test_handler_respects_context_threshold() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    // Create contexts with different default thresholds
    let low_threshold_ctx = FunctionContext::new(0.1, "cosine", 8, EmbeddingType::Float);
    let high_threshold_ctx = FunctionContext::new(0.99, "cosine", 8, EmbeddingType::Float);

    // Both should work
    let low_result = backend.find_similar_with_handlers(
        &emb, "cosine", 0.0, "all", &[], &low_threshold_ctx
    );
    let high_result = backend.find_similar_with_handlers(
        &emb, "cosine", 0.0, "all", &[], &high_threshold_ctx
    );

    assert!(low_result.is_ok(), "low threshold context should work");
    assert!(high_result.is_ok(), "high threshold context should work");
}

// ==========================================================================
// Error Cases in Handler Dispatch
// ==========================================================================

#[test]
fn test_unknown_metric_error() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let result = backend.find_similar_with_handlers(
        &emb, "nonexistent", 0.0, "all", &[], &context
    );

    assert!(result.is_err(), "should reject unknown metric");
}

#[test]
fn test_unknown_ranking_error() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let result = backend.find_similar_with_handlers(
        &emb, "cosine", 0.0, "nonexistent", &[], &context
    );

    assert!(result.is_err(), "should reject unknown ranking");
}

#[test]
fn test_missing_topk_param_error() {
    let mut backend = InMemoryBackend::new(8);
    let emb = vec![1.0f32; 8];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let result = backend.find_similar_with_handlers(
        &emb, "cosine", 0.0, "topk", &[], &context // Missing k parameter
    );

    assert!(result.is_err(), "should reject topk without k parameter");
}

// ==========================================================================
// Handler Direct Invocation Tests
// ==========================================================================

#[test]
fn test_cosine_handler_direct() {
    let handler = CosineMetricHandler;

    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let embeddings = ndarray::Array2::from_shape_vec(
        (2, 4),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ).unwrap();
    let live_mask = Array1::from_vec(vec![1.0, 1.0]);

    let context = FunctionContext::default();
    let result = handler.compute(&query, &embeddings, &live_mask, 0.0, &[], &context);

    assert!(result.is_ok());
    let sim_result = result.unwrap();
    assert_eq!(sim_result.scores.len(), 2);
    // First embedding identical to query: score ~1.0
    assert!((sim_result.scores[0] - 1.0).abs() < 1e-5);
    // Second embedding orthogonal: score ~0.0
    assert!(sim_result.scores[1].abs() < 1e-5);
}

#[test]
fn test_topk_handler_direct() {
    let handler = TopKRankingHandler;

    let scores = Array1::from_vec(vec![0.9, 0.5, 0.8, 0.2, 0.7]);
    let params = vec![ResolvedArg::Integer(3)];
    let context = FunctionContext::default();

    let sim_result = SimilarityResult::new(scores, 0.0);
    let result = handler.rank(&sim_result, &params, &context);

    assert!(result.is_ok());
    let rank_result = result.unwrap();
    assert_eq!(rank_result.len(), 3);
    // Should be sorted descending: indices 0, 2, 4 with scores 0.9, 0.8, 0.7
    assert_eq!(rank_result.matches[0].0, 0);
    assert_eq!(rank_result.matches[1].0, 2);
    assert_eq!(rank_result.matches[2].0, 4);
}

#[test]
fn test_threshold_handler_direct() {
    let handler = ThresholdRankingHandler;

    let scores = Array1::from_vec(vec![0.9, 0.5, 0.8, 0.2, 0.7]);
    let params = vec![ResolvedArg::Float(0.6)];
    let context = FunctionContext::default();

    let sim_result = SimilarityResult::new(scores, 0.6);
    let result = handler.rank(&sim_result, &params, &context);

    assert!(result.is_ok());
    let rank_result = result.unwrap();
    // Only indices 0, 2, 4 have scores >= 0.6
    assert_eq!(rank_result.len(), 3);
}

#[test]
fn test_all_handler_direct() {
    let handler = AllRankingHandler;

    let scores = Array1::from_vec(vec![0.9, 0.5, 0.8, 0.2, 0.7]);
    let context = FunctionContext::default();

    let sim_result = SimilarityResult::new(scores, 0.0);
    let result = handler.rank(&sim_result, &[], &context);

    assert!(result.is_ok());
    let rank_result = result.unwrap();
    assert_eq!(rank_result.len(), 5, "all should return all scores");
}

// ==========================================================================
// Handler Composition Tests
// ==========================================================================

#[test]
fn test_different_metric_ranking_combinations() {
    let mut backend = InMemoryBackend::new(4);

    // Store diverse vectors
    let vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.9, 0.4, 0.0, 0.0],
        vec![0.5, 0.5, 0.5, 0.5],
        vec![0.0, 1.0, 0.0, 0.0],
    ];

    for v in &vectors {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = v.iter().map(|x| x / norm).collect();
        backend.store(&normalized).expect("store");
    }

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let context = FunctionContext::default();

    // Test all combinations of metrics and rankings
    let metrics = vec!["cosine", "euclidean", "manhattan"];
    let rankings = vec![
        ("topk", vec![ResolvedArg::Integer(2)]),
        ("threshold", vec![ResolvedArg::Float(0.3)]),
        ("all", vec![]),
    ];

    for metric in &metrics {
        for (ranking, params) in &rankings {
            let result = backend.find_similar_with_handlers(
                &query, metric, 0.0, ranking, params, &context
            );
            assert!(result.is_ok(),
                "combination {}/{} should work", metric, ranking);
        }
    }
}
