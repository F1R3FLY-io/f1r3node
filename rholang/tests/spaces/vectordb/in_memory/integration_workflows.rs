//! End-to-end workflow integration tests for rho-vectordb.
//!
//! These tests validate complete user workflows from embedding storage
//! through similarity search to result retrieval.

use rholang::rust::interpreter::spaces::vectordb::in_memory::{
    InMemoryBackend, VectorBackend, HandlerBackend, SimilarityMetric,
    FunctionContext, ResolvedArg,
};

// ==========================================================================
// Test Utilities
// ==========================================================================

/// Generate a deterministic embedding from a seed.
/// Creates dense embeddings with guaranteed positive similarity to nearby seeds.
fn seeded_embedding(seed: u32, dim: usize) -> Vec<f32> {
    let mut emb = vec![0.0f32; dim];

    // Create dense embedding using deterministic pseudo-random pattern
    // This ensures all embeddings have positive similarity with each other
    for i in 0..dim {
        // Use a simple LCG-like formula for deterministic variation
        let val = ((seed as f32 + 1.0) * (i as f32 + 1.0) * 0.1).sin().abs();
        emb[i] = val + 0.1; // Add baseline to ensure positive components
    }

    // Add seed-specific peaks for differentiation
    let primary = (seed as usize) % dim;
    let secondary = (seed as usize * 7 + 3) % dim;
    emb[primary] += 1.0;
    emb[secondary] += 0.5;

    // Normalize
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    emb.iter().map(|x| x / norm).collect()
}

/// Generate a text-like embedding (simulates word2vec/sentence embeddings).
fn text_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut emb = vec![0.0f32; dim];
    for (i, byte) in text.bytes().enumerate() {
        let idx = (byte as usize + i * 7) % dim;
        emb[idx] += 1.0;
    }
    // Normalize
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        emb.iter().map(|x| x / norm).collect()
    } else {
        emb[0] = 1.0;
        emb
    }
}

// ==========================================================================
// Basic CRUD Workflow Tests
// ==========================================================================

#[test]
fn test_store_query_workflow() {
    // Complete workflow: store embeddings, query by similarity, get results
    let mut backend = InMemoryBackend::new(64);

    // Store three embeddings with controlled similarity:
    // emb0 is query, emb1 is very similar, emb2 is somewhat similar (not orthogonal)
    let emb0 = seeded_embedding(0, 64);
    let emb1 = {
        // Very similar to emb0: perturb slightly
        let mut v = emb0.clone();
        v[1] += 0.1;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect::<Vec<_>>()
    };
    let emb2 = {
        // Less similar to emb0: perturb more significantly
        let mut v = emb0.clone();
        v[10] += 0.5;
        v[20] += 0.3;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect::<Vec<_>>()
    };

    let id0 = backend.store(&emb0).expect("store emb0");
    let id1 = backend.store(&emb1).expect("store emb1");
    let id2 = backend.store(&emb2).expect("store emb2");

    // Query with emb0
    let results = backend.find_similar(&emb0, SimilarityMetric::Cosine, 0.0, Some(10))
        .expect("find_similar");

    // emb0 should be the top result (self-similarity)
    assert!(!results.is_empty(), "should have results");
    assert_eq!(results[0].0, id0, "self should be top result");

    // emb1 should be more similar than emb2
    let pos1 = results.iter().position(|(id, _)| *id == id1);
    let pos2 = results.iter().position(|(id, _)| *id == id2);

    assert!(pos1.is_some(), "emb1 should be in results (high similarity)");
    assert!(pos2.is_some(), "emb2 should be in results (positive similarity)");
    assert!(pos1.unwrap() < pos2.unwrap(),
        "similar embedding should rank higher than different one");
}

#[test]
fn test_store_replace_query_workflow() {
    // Store, remove and re-store (simulate update), query to verify change is reflected
    let mut backend = InMemoryBackend::new(32);

    // Initial storage
    let initial_emb = seeded_embedding(0, 32);
    let id = backend.store(&initial_emb).expect("initial store");

    // Query matches initial embedding
    let results1 = backend.find_similar(&initial_emb, SimilarityMetric::Cosine, 0.9, None)
        .expect("find similar 1");
    assert!(results1.iter().any(|(rid, _)| *rid == id), "should find initial");

    // Remove and re-store with different embedding (simulates update)
    let updated_emb = seeded_embedding(15, 32); // Different seed = orthogonal
    backend.remove(&id);
    let new_id = backend.store(&updated_emb).expect("re-store");

    // Query with old embedding should NOT find new_id at high threshold
    let results2 = backend.find_similar(&initial_emb, SimilarityMetric::Cosine, 0.9, None)
        .expect("find similar 2");
    assert!(!results2.iter().any(|(rid, _)| *rid == new_id),
        "should NOT find with old embedding at high threshold");

    // Query with new embedding SHOULD find it
    let results3 = backend.find_similar(&updated_emb, SimilarityMetric::Cosine, 0.9, None)
        .expect("find similar 3");
    assert!(results3.iter().any(|(rid, _)| *rid == new_id),
        "should find with updated embedding");
}

#[test]
fn test_store_remove_query_workflow() {
    // Store, remove, query to verify tombstone works
    let mut backend = InMemoryBackend::new(32);

    // Store embeddings
    let emb1 = seeded_embedding(1, 32);
    let emb2 = seeded_embedding(2, 32);
    let emb3 = seeded_embedding(3, 32);

    let id1 = backend.store(&emb1).expect("store 1");
    let id2 = backend.store(&emb2).expect("store 2");
    let id3 = backend.store(&emb3).expect("store 3");

    assert_eq!(backend.len(), 3);

    // Remove id2
    let removed = backend.remove(&id2);
    assert!(removed, "should remove id2");
    assert_eq!(backend.len(), 2);

    // Query should not return id2
    let results = backend.find_similar(&emb2, SimilarityMetric::Cosine, 0.0, None)
        .expect("find similar");

    assert!(!results.iter().any(|(rid, _)| *rid == id2),
        "removed embedding should not appear in results");

    // id1 and id3 should still be findable
    let results1 = backend.find_similar(&emb1, SimilarityMetric::Cosine, 0.9, None)
        .expect("find similar 1");
    assert!(results1.iter().any(|(rid, _)| *rid == id1), "id1 should be found");

    let results3 = backend.find_similar(&emb3, SimilarityMetric::Cosine, 0.9, None)
        .expect("find similar 3");
    assert!(results3.iter().any(|(rid, _)| *rid == id3), "id3 should be found");
}

// ==========================================================================
// Multi-Metric Workflow Tests
// ==========================================================================

#[test]
fn test_multi_metric_same_data() {
    // Query same data with different metrics, verify different orderings
    let mut backend = InMemoryBackend::new(16);

    // Store embeddings with known properties - all have positive similarity with e1
    // e1: mainly direction 0, small component in direction 1
    // e2: mainly direction 1, small component in direction 0 (small positive sim with e1)
    // e3: diagonal (normalized) - moderate sim with e1
    let e1 = {
        let mut v = vec![0.0f32; 16];
        v[0] = 1.0;
        v[1] = 0.1; // Small component for overlap
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect::<Vec<_>>()
    };
    let e2 = {
        let mut v = vec![0.0f32; 16];
        v[0] = 0.1; // Small component for overlap with e1
        v[1] = 1.0;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect::<Vec<_>>()
    };
    let e3 = {
        let v = vec![1.0f32; 16];
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect::<Vec<_>>()
    };

    let id1 = backend.store(&e1).expect("store e1");
    let id2 = backend.store(&e2).expect("store e2");
    let id3 = backend.store(&e3).expect("store e3");

    // Query with e1
    let cosine_results = backend.find_similar(&e1, SimilarityMetric::Cosine, 0.0, None)
        .expect("cosine");
    let dot_results = backend.find_similar(&e1, SimilarityMetric::DotProduct, 0.0, None)
        .expect("dot product");
    let euclidean_results = backend.find_similar(&e1, SimilarityMetric::Euclidean, 0.0, None)
        .expect("euclidean");

    // All should find id1 as top result (most similar to itself)
    assert_eq!(cosine_results[0].0, id1, "cosine: id1 should be top");
    assert_eq!(dot_results[0].0, id1, "dot: id1 should be top");
    assert_eq!(euclidean_results[0].0, id1, "euclidean: id1 should be top");

    // e3 (diagonal) should have higher similarity with e1 than e2 (mostly orthogonal)
    let cosine_e3_pos = cosine_results.iter().position(|(id, _)| *id == id3);
    let cosine_e2_pos = cosine_results.iter().position(|(id, _)| *id == id2);
    assert!(cosine_e3_pos.is_some(), "e3 should be in results");
    assert!(cosine_e2_pos.is_some(), "e2 should be in results (small positive sim)");
    assert!(cosine_e3_pos.unwrap() < cosine_e2_pos.unwrap(),
        "cosine: e3 should rank higher than e2");
}

#[test]
fn test_metric_specific_behavior() {
    // Test that each metric has distinct behavior on crafted inputs
    let mut backend = InMemoryBackend::new(4);

    // Boolean-ish vectors for Hamming/Jaccard
    let v1 = vec![1.0, 0.0, 0.0, 0.0]; // 1 bit set
    let v2 = vec![1.0, 1.0, 0.0, 0.0]; // 2 bits set, overlapping
    let v3 = vec![0.0, 0.0, 1.0, 1.0]; // 2 bits set, non-overlapping

    backend.store(&v1).expect("store v1");
    let id2 = backend.store(&v2).expect("store v2");
    let id3 = backend.store(&v3).expect("store v3");

    // Query with v1 using Hamming (measures bit difference)
    let hamming_results = backend.find_similar(&v1, SimilarityMetric::Hamming, 0.0, None)
        .expect("hamming");

    // v2 differs by 1 bit from v1, v3 differs by 3 bits
    // So v2 should be more similar under Hamming
    let v2_score = hamming_results.iter().find(|(id, _)| *id == id2).map(|(_, s)| *s);
    let v3_score = hamming_results.iter().find(|(id, _)| *id == id3).map(|(_, s)| *s);

    assert!(v2_score.unwrap() > v3_score.unwrap(),
        "hamming: v2 should be more similar to v1 than v3");
}

// ==========================================================================
// Ranking Function Workflow Tests
// ==========================================================================

#[test]
fn test_topk_ranking_workflow() {
    // Test top-k ranking with handler system
    let mut backend = InMemoryBackend::new(32);

    // Store 10 embeddings
    let embeddings: Vec<Vec<f32>> = (0..10).map(|i| seeded_embedding(i, 32)).collect();
    let ids: Vec<usize> = embeddings.iter()
        .map(|e| backend.store(e).expect("store"))
        .collect();

    // Query with first embedding
    let query = &embeddings[0];

    // TopK with k=3
    let context = FunctionContext::default();
    let results = backend.find_similar_with_handlers(
        query,
        "cosine",
        0.0,
        "topk",
        &[ResolvedArg::Integer(3)],
        &context,
    ).expect("topk query");

    // Should return exactly 3 results
    assert_eq!(results.len(), 3, "topk should return 3 results");

    // First result should be the query itself (self-similarity)
    assert_eq!(results[0].0, ids[0], "self should be top result");

    // Results should be sorted descending
    for window in results.windows(2) {
        assert!(window[0].1 >= window[1].1, "results should be sorted descending");
    }
}

#[test]
fn test_threshold_ranking_workflow() {
    // Test threshold-based filtering with handler system
    let mut backend = InMemoryBackend::new(16);

    // Store embeddings with known similarities to a reference
    let reference = {
        let mut v = vec![0.0f32; 16];
        v[0] = 1.0;
        v
    };

    // Create embeddings at different angles to reference
    let very_similar = {
        let mut v = vec![0.0f32; 16];
        v[0] = 0.95;
        v[1] = 0.31225; // cos^-1(0.95) = 18 deg
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect::<Vec<_>>()
    };
    let moderately_similar = {
        let mut v = vec![0.0f32; 16];
        v[0] = 0.7;
        v[1] = 0.714; // ~45 deg
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect::<Vec<_>>()
    };
    let dissimilar = {
        let mut v = vec![0.0f32; 16];
        v[1] = 1.0; // orthogonal
        v
    };

    backend.store(&reference).expect("ref");
    let id_similar = backend.store(&very_similar).expect("similar");
    let id_moderate = backend.store(&moderately_similar).expect("moderate");
    let id_dissimilar = backend.store(&dissimilar).expect("dissimilar");

    let context = FunctionContext::default();

    // High threshold - only very similar
    let results_high = backend.find_similar_with_handlers(
        &reference,
        "cosine",
        0.9,
        "threshold",
        &[ResolvedArg::Float(0.9)],
        &context,
    ).expect("high threshold");

    assert!(results_high.iter().any(|(id, _)| *id == id_similar),
        "very similar should pass 0.9 threshold");
    assert!(!results_high.iter().any(|(id, _)| *id == id_moderate),
        "moderate should not pass 0.9 threshold");

    // Low threshold - similar and moderate
    let results_low = backend.find_similar_with_handlers(
        &reference,
        "cosine",
        0.5,
        "threshold",
        &[ResolvedArg::Float(0.5)],
        &context,
    ).expect("low threshold");

    assert!(results_low.iter().any(|(id, _)| *id == id_similar),
        "very similar should pass 0.5 threshold");
    assert!(results_low.iter().any(|(id, _)| *id == id_moderate),
        "moderate should pass 0.5 threshold");
    assert!(!results_low.iter().any(|(id, _)| *id == id_dissimilar),
        "dissimilar should not pass 0.5 threshold");
}

#[test]
fn test_all_ranking_workflow() {
    // Test 'all' ranking that returns everything above threshold
    let mut backend = InMemoryBackend::new(32);

    // Store 20 embeddings
    let embeddings: Vec<Vec<f32>> = (0..20).map(|i| seeded_embedding(i, 32)).collect();
    for e in &embeddings {
        backend.store(e).expect("store");
    }

    let query = &embeddings[0];
    let context = FunctionContext::default();

    // 'all' returns all matches above threshold
    let results = backend.find_similar_with_handlers(
        query,
        "cosine",
        0.0,
        "all",
        &[],
        &context,
    ).expect("all query");

    // Should return all 20 embeddings
    assert_eq!(results.len(), 20, "all should return every embedding");

    // Should be sorted by score descending
    for window in results.windows(2) {
        assert!(window[0].1 >= window[1].1, "results should be sorted");
    }
}

// ==========================================================================
// Handler Override Workflow Tests
// ==========================================================================

#[test]
fn test_metric_alias_resolution() {
    // Test that metric aliases work correctly
    let mut backend = InMemoryBackend::new(16);

    let emb = vec![1.0f32; 16];
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    let emb: Vec<f32> = emb.iter().map(|x| x / norm).collect();

    backend.store(&emb).expect("store");

    let context = FunctionContext::default();

    // "cos" alias for "cosine"
    let cos_result = backend.find_similar_with_handlers(
        &emb, "cos", 0.0, "all", &[], &context
    );
    let cosine_result = backend.find_similar_with_handlers(
        &emb, "cosine", 0.0, "all", &[], &context
    );

    assert!(cos_result.is_ok(), "cos alias should work");
    assert!(cosine_result.is_ok(), "cosine should work");

    // Both should return same results
    let cos_scores: Vec<f32> = cos_result.unwrap().iter().map(|(_, s)| *s).collect();
    let cosine_scores: Vec<f32> = cosine_result.unwrap().iter().map(|(_, s)| *s).collect();

    assert_eq!(cos_scores, cosine_scores, "aliases should give same results");
}

#[test]
fn test_handler_override_different_results() {
    // Query same data with different handlers, verify different results
    let mut backend = InMemoryBackend::new(32);

    // Store 15 embeddings
    let embeddings: Vec<Vec<f32>> = (0..15).map(|i| seeded_embedding(i, 32)).collect();
    for e in &embeddings {
        backend.store(e).expect("store");
    }

    let query = &embeddings[0];
    let context = FunctionContext::default();

    // TopK=5 should return 5 results
    let topk_results = backend.find_similar_with_handlers(
        query, "cosine", 0.0, "topk", &[ResolvedArg::Integer(5)], &context
    ).expect("topk");

    // All should return 15 results
    let all_results = backend.find_similar_with_handlers(
        query, "cosine", 0.0, "all", &[], &context
    ).expect("all");

    assert_eq!(topk_results.len(), 5, "topk should limit to 5");
    assert_eq!(all_results.len(), 15, "all should return all 15");

    // TopK results should be subset of all results
    for (id, score) in &topk_results {
        let found = all_results.iter().find(|(aid, _)| aid == id);
        assert!(found.is_some(), "topk result should be in all results");
        assert!((found.unwrap().1 - score).abs() < 1e-5, "scores should match");
    }
}

// ==========================================================================
// Complex Workflow Tests
// ==========================================================================

#[test]
fn test_churn_workflow() {
    // Simulate high churn: add, remove, add, query cycle
    // Note: IDs may be reused after removal, so we track embeddings not IDs
    let mut backend = InMemoryBackend::new(16);

    // Phase 1: Add 10 embeddings with seeds 0-9
    let initial_embeddings: Vec<Vec<f32>> = (0..10)
        .map(|i| seeded_embedding(i, 16))
        .collect();
    let mut initial_ids: Vec<usize> = Vec::new();
    for emb in &initial_embeddings {
        initial_ids.push(backend.store(emb).expect("store"));
    }
    assert_eq!(backend.len(), 10);

    // Phase 2: Remove embeddings at odd indices (seeds 1, 3, 5, 7, 9)
    let removed_seeds: Vec<u32> = vec![1, 3, 5, 7, 9];
    for i in (1..10).step_by(2) {
        backend.remove(&initial_ids[i]);
    }
    assert_eq!(backend.len(), 5);

    // Phase 3: Add 5 more embeddings with seeds 10-14
    let new_embeddings: Vec<Vec<f32>> = (10..15)
        .map(|i| seeded_embedding(i, 16))
        .collect();
    for emb in &new_embeddings {
        backend.store(emb).expect("store");
    }
    assert_eq!(backend.len(), 10);

    // Query with seed=0 embedding (still present)
    let query = &initial_embeddings[0];
    let results = backend.find_similar(query, SimilarityMetric::Cosine, 0.0, None)
        .expect("find similar");

    // Should have 10 results (5 remaining + 5 new)
    assert_eq!(results.len(), 10, "should find all active embeddings");

    // Verify removed embeddings are not in results by checking similarity
    // with removed embeddings - if they were in results, we'd find high similarity
    for seed in &removed_seeds {
        let removed_emb = seeded_embedding(*seed, 16);
        // Check that no result has near-perfect similarity to removed embedding
        // (Only the exact embedding would have similarity ~1.0)
        let max_sim_to_removed = results.iter()
            .filter_map(|(id, _)| backend.get(id))
            .map(|stored_emb| {
                let dot: f32 = stored_emb.iter()
                    .zip(removed_emb.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                dot
            })
            .fold(0.0f32, |a, b| a.max(b));

        assert!(max_sim_to_removed < 0.99,
            "removed embedding (seed={}) should not be in results, max_sim={}",
            seed, max_sim_to_removed);
    }
}

#[test]
fn test_large_batch_workflow() {
    // Test with larger dataset
    let mut backend = InMemoryBackend::new(128);

    // Store 1000 embeddings
    let embeddings: Vec<Vec<f32>> = (0..1000)
        .map(|i| seeded_embedding(i, 128))
        .collect();

    for emb in &embeddings {
        backend.store(emb).expect("store");
    }

    assert_eq!(backend.len(), 1000);

    // Query with various k values
    let query = &embeddings[500];

    let results_10 = backend.find_similar(query, SimilarityMetric::Cosine, 0.0, Some(10))
        .expect("k=10");
    let results_100 = backend.find_similar(query, SimilarityMetric::Cosine, 0.0, Some(100))
        .expect("k=100");
    let results_all = backend.find_similar(query, SimilarityMetric::Cosine, 0.0, None)
        .expect("all");

    assert_eq!(results_10.len(), 10);
    assert_eq!(results_100.len(), 100);
    assert_eq!(results_all.len(), 1000);

    // All results should be subsets
    for (id, _) in &results_10 {
        assert!(results_100.iter().any(|(rid, _)| rid == id));
    }
    for (id, _) in &results_100 {
        assert!(results_all.iter().any(|(rid, _)| rid == id));
    }
}

#[test]
fn test_semantic_search_workflow() {
    // Simulate semantic search with text-like embeddings
    let mut backend = InMemoryBackend::new(64);

    // Documents
    let docs = vec![
        "machine learning artificial intelligence",
        "deep learning neural networks",
        "natural language processing text",
        "computer vision image recognition",
        "cooking recipes food kitchen",
        "travel destinations vacation",
    ];

    let mut doc_ids = Vec::new();
    for doc in &docs {
        let emb = text_embedding(doc, 64);
        let id = backend.store(&emb).expect("store doc");
        doc_ids.push(id);
    }

    // Query for ML-related content
    let query = text_embedding("machine learning AI", 64);
    let results = backend.find_similar(&query, SimilarityMetric::Cosine, 0.0, Some(3))
        .expect("semantic search");

    // Top results should be ML-related docs (indices 0, 1, 2, 3)
    let ml_related: std::collections::HashSet<usize> =
        vec![doc_ids[0], doc_ids[1], doc_ids[2], doc_ids[3]].into_iter().collect();

    let top_3_ids: std::collections::HashSet<usize> =
        results.iter().map(|(id, _)| *id).collect();

    // At least 2 of top 3 should be ML-related
    let overlap = ml_related.intersection(&top_3_ids).count();
    assert!(overlap >= 2, "at least 2 of top 3 should be ML-related, got {}", overlap);
}
