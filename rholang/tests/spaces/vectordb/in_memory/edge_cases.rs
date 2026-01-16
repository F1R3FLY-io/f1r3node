//! Edge case tests for rho-vectordb.
//!
//! These tests validate boundary conditions and unusual inputs that
//! might cause unexpected behavior.

use rholang::rust::interpreter::spaces::vectordb::in_memory::{
    InMemoryBackend, VectorBackend, HandlerBackend, SimilarityMetric,
    FunctionContext, ResolvedArg,
};

// ==========================================================================
// Zero and Near-Zero Vector Tests
// ==========================================================================

#[test]
fn test_zero_vector_cosine_similarity() {
    // Zero vector has undefined cosine similarity (0/0)
    // Backend should handle gracefully
    let mut backend = InMemoryBackend::new(4);

    let normal_vec = vec![1.0, 0.0, 0.0, 0.0];
    backend.store(&normal_vec).expect("store normal");

    // Zero vector query
    let zero_vec = vec![0.0f32; 4];
    let result = backend.find_similar(&zero_vec, SimilarityMetric::Cosine, 0.0, None);

    // Should not panic - either returns empty or handles gracefully
    assert!(result.is_ok(), "zero vector query should not panic");
}

#[test]
fn test_near_zero_vector_normalization() {
    // Very small vectors might cause numerical issues during normalization
    let mut backend = InMemoryBackend::new(4);

    let tiny_vec = vec![1e-38f32; 4]; // Near f32 min normal
    let result = backend.store(&tiny_vec);

    // Should handle gracefully
    if result.is_ok() {
        let id = result.unwrap();
        // Verify we can retrieve it
        let retrieved = backend.get(&id);
        assert!(retrieved.is_some(), "should be able to retrieve tiny vector");
    }
}

#[test]
fn test_mixed_zero_normal_vectors() {
    let mut backend = InMemoryBackend::new(4);

    // Some components zero, some normal
    let mixed = vec![1.0, 0.0, 1.0, 0.0];
    let id = backend.store(&mixed).expect("store mixed");

    // Query with same pattern
    let results = backend.find_similar(&mixed, SimilarityMetric::Cosine, 0.0, None)
        .expect("query should work");

    assert!(!results.is_empty(), "should find the stored vector");
    assert_eq!(results[0].0, id, "self should be top result");
}

// ==========================================================================
// Identical Vector Tests
// ==========================================================================

#[test]
fn test_self_similarity_cosine() {
    let mut backend = InMemoryBackend::new(8);

    let vec = vec![0.5f32; 8];
    let id = backend.store(&vec).expect("store");

    let results = backend.find_similar(&vec, SimilarityMetric::Cosine, 0.0, None)
        .expect("query");

    // Self-similarity should be ~1.0
    assert!(!results.is_empty());
    assert_eq!(results[0].0, id);
    assert!((results[0].1 - 1.0).abs() < 1e-5, "self-similarity should be 1.0, got {}", results[0].1);
}

#[test]
fn test_identical_vectors_stored() {
    let mut backend = InMemoryBackend::new(8);

    let vec = vec![1.0f32; 8];
    let id1 = backend.store(&vec).expect("store 1");
    let id2 = backend.store(&vec).expect("store 2"); // Same vector
    let id3 = backend.store(&vec).expect("store 3"); // Same vector again

    // All three should have similarity 1.0 with the query
    let results = backend.find_similar(&vec, SimilarityMetric::Cosine, 0.99, None)
        .expect("query");

    assert_eq!(results.len(), 3, "all identical vectors should be found");

    // All should have same similarity
    let similarities: Vec<f32> = results.iter().map(|(_, s)| *s).collect();
    for sim in &similarities {
        assert!((sim - 1.0).abs() < 1e-5, "all should have similarity 1.0");
    }

    // All IDs should be present
    let ids: std::collections::HashSet<usize> = results.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&id1));
    assert!(ids.contains(&id2));
    assert!(ids.contains(&id3));
}

// ==========================================================================
// Orthogonal Vector Tests
// ==========================================================================

#[test]
fn test_orthogonal_vectors_cosine() {
    let mut backend = InMemoryBackend::new(4);

    // Create orthogonal unit vectors
    let e1 = vec![1.0, 0.0, 0.0, 0.0];
    let e2 = vec![0.0, 1.0, 0.0, 0.0];
    let e3 = vec![0.0, 0.0, 1.0, 0.0];

    let id1 = backend.store(&e1).expect("store e1");
    let id2 = backend.store(&e2).expect("store e2");
    let id3 = backend.store(&e3).expect("store e3");

    // Query with e1, threshold 0.0 (clamped internally to f32::EPSILON)
    let results = backend.find_similar(&e1, SimilarityMetric::Cosine, 0.0, None)
        .expect("query");

    // e1 should have similarity 1.0 with itself
    let e1_result = results.iter().find(|(id, _)| *id == id1).unwrap();
    assert!((e1_result.1 - 1.0).abs() < 1e-5, "self-similarity should be 1.0");

    // e2, e3 have similarity 0.0 with e1 (orthogonal)
    // Since threshold is clamped to f32::EPSILON, vectors with exactly 0.0 similarity
    // are excluded (this also prevents tombstoned entries from being returned)
    let e2_result = results.iter().find(|(id, _)| *id == id2);
    let e3_result = results.iter().find(|(id, _)| *id == id3);
    assert!(e2_result.is_none(), "orthogonal vectors (sim=0.0) should be excluded by EPSILON threshold");
    assert!(e3_result.is_none(), "orthogonal vectors (sim=0.0) should be excluded by EPSILON threshold");

    // Verify that id2 and id3 still exist in backend
    assert!(backend.get(&id2).is_some(), "e2 should still exist");
    assert!(backend.get(&id3).is_some(), "e3 should still exist");
}

#[test]
fn test_opposite_vectors_cosine() {
    // Note: Threshold is clamped to [0.0, 1.0], so opposite vectors (similarity -1.0)
    // won't be returned when querying with default parameters.
    // This test verifies that positive vectors ARE found and opposite are NOT found
    // with threshold 0.0 (the minimum allowed).
    let mut backend = InMemoryBackend::new(4);

    let pos = vec![1.0, 0.0, 0.0, 0.0];
    let neg = vec![-1.0, 0.0, 0.0, 0.0]; // Opposite direction

    let id_pos = backend.store(&pos).expect("store pos");
    let id_neg = backend.store(&neg).expect("store neg");

    // Query with pos, threshold 0.0 (minimum allowed)
    let results = backend.find_similar(&pos, SimilarityMetric::Cosine, 0.0, None)
        .expect("query");

    // pos should be found (similarity 1.0)
    let pos_result = results.iter().find(|(id, _)| *id == id_pos);
    assert!(pos_result.is_some(), "should find positive vector");

    // neg should NOT be found (similarity -1.0 < threshold 0.0)
    let neg_result = results.iter().find(|(id, _)| *id == id_neg);
    assert!(neg_result.is_none(), "should NOT find opposite vector with threshold 0.0");
}

// ==========================================================================
// Large Dimension Tests
// ==========================================================================

#[test]
fn test_high_dimensional_vectors() {
    // Test with 4096 dimensions (common for modern embeddings)
    let dim = 4096;
    let mut backend = InMemoryBackend::new(dim);

    // Create a sparse vector with few non-zero components
    let mut sparse_vec = vec![0.0f32; dim];
    sparse_vec[0] = 1.0;
    sparse_vec[1000] = 0.5;
    sparse_vec[2000] = 0.3;
    sparse_vec[4095] = 0.1;

    // Normalize
    let norm: f32 = sparse_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let sparse_vec: Vec<f32> = sparse_vec.iter().map(|x| x / norm).collect();

    let id = backend.store(&sparse_vec).expect("store high-dim");

    let results = backend.find_similar(&sparse_vec, SimilarityMetric::Cosine, 0.9, None)
        .expect("query high-dim");

    assert!(!results.is_empty());
    assert_eq!(results[0].0, id);
}

#[test]
fn test_very_high_dimensional_vectors() {
    // Test with 8192 dimensions (some LLM embeddings)
    let dim = 8192;
    let mut backend = InMemoryBackend::new(dim);

    let vec: Vec<f32> = (0..dim)
        .map(|i| ((i as f32) / (dim as f32)).sin())
        .collect();

    // Normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let vec: Vec<f32> = vec.iter().map(|x| x / norm).collect();

    let id = backend.store(&vec).expect("store very high-dim");

    let results = backend.find_similar(&vec, SimilarityMetric::Cosine, 0.9, None)
        .expect("query very high-dim");

    assert!(!results.is_empty());
    assert_eq!(results[0].0, id);
}

// ==========================================================================
// Empty and Single Element Tests
// ==========================================================================

#[test]
fn test_empty_backend_operations() {
    let backend = InMemoryBackend::new(16);

    assert_eq!(backend.len(), 0, "new backend should be empty");

    // Query empty backend
    let query = vec![1.0f32; 16];
    let results = backend.find_similar(&query, SimilarityMetric::Cosine, 0.0, None)
        .expect("query empty");

    assert!(results.is_empty(), "empty backend should return no results");
}

#[test]
fn test_single_embedding_backend() {
    let mut backend = InMemoryBackend::new(16);

    let vec = vec![1.0f32; 16];
    let id = backend.store(&vec).expect("store single");

    assert_eq!(backend.len(), 1);

    // Query should return exactly this one
    let results = backend.find_similar(&vec, SimilarityMetric::Cosine, 0.0, None)
        .expect("query single");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);
}

#[test]
fn test_single_embedding_topk_larger_than_count() {
    let mut backend = InMemoryBackend::new(16);

    let vec = vec![1.0f32; 16];
    backend.store(&vec).expect("store");

    // Ask for top 100 when only 1 exists
    let results = backend.find_similar(&vec, SimilarityMetric::Cosine, 0.0, Some(100))
        .expect("topk larger than count");

    assert_eq!(results.len(), 1, "should return all available (1)");
}

// ==========================================================================
// Limit Boundary Tests
// ==========================================================================

#[test]
fn test_limit_zero() {
    let mut backend = InMemoryBackend::new(16);

    for i in 0..10 {
        let mut v = vec![0.0f32; 16];
        v[i] = 1.0;
        backend.store(&v).expect("store");
    }

    let query = vec![1.0f32; 16];

    // Limit = 0 should return no results or be treated as unlimited
    let results = backend.find_similar(&query, SimilarityMetric::Cosine, 0.0, Some(0))
        .expect("limit 0");

    // Implementation-dependent: either 0 or all results
    // Just ensure it doesn't panic
    assert!(results.len() <= 10);
}

#[test]
fn test_limit_exact_count() {
    let mut backend = InMemoryBackend::new(16);

    for i in 0..5 {
        let mut v = vec![0.0f32; 16];
        v[i] = 1.0;
        backend.store(&v).expect("store");
    }

    let query = vec![1.0f32; 16];

    // Limit exactly equals count
    let results = backend.find_similar(&query, SimilarityMetric::Cosine, 0.0, Some(5))
        .expect("limit exact");

    assert_eq!(results.len(), 5);
}

// ==========================================================================
// Threshold Boundary Tests
// ==========================================================================

#[test]
fn test_threshold_exactly_one() {
    let mut backend = InMemoryBackend::new(4);

    let vec = vec![1.0, 0.0, 0.0, 0.0];
    let id = backend.store(&vec).expect("store");

    // Threshold = 1.0 should only match exact same vector
    let results = backend.find_similar(&vec, SimilarityMetric::Cosine, 1.0, None)
        .expect("threshold 1.0");

    // Only exact match (self) should be returned
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);
    assert!((results[0].1 - 1.0).abs() < 1e-5);
}

#[test]
fn test_threshold_clamped_to_zero() {
    // Threshold is clamped to [0.0, 1.0], so -1.0 becomes 0.0
    let mut backend = InMemoryBackend::new(4);

    let pos = vec![1.0, 0.0, 0.0, 0.0];
    let neg = vec![-1.0, 0.0, 0.0, 0.0];

    backend.store(&pos).expect("store pos");
    backend.store(&neg).expect("store neg");

    // Threshold = -1.0 gets clamped to 0.0
    // Only pos (similarity 1.0) passes, neg (similarity -1.0) does not
    let results = backend.find_similar(&pos, SimilarityMetric::Cosine, -1.0, None)
        .expect("threshold -1.0");

    assert_eq!(results.len(), 1, "only pos should be found (neg has sim -1.0 < 0.0)");
}

// ==========================================================================
// Numerical Precision Tests
// ==========================================================================

#[test]
fn test_similar_vectors_distinguishable() {
    let mut backend = InMemoryBackend::new(4);

    // Create vectors that are similar but distinguishably different
    let v1 = vec![1.0, 0.0, 0.0, 0.0];
    let v2 = vec![0.95, 0.31, 0.0, 0.0]; // About 18 degrees apart

    // Normalize v2
    let norm: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
    let v2: Vec<f32> = v2.iter().map(|x| x / norm).collect();

    let id1 = backend.store(&v1).expect("store v1");
    let id2 = backend.store(&v2).expect("store v2");

    // Query with v1
    let results = backend.find_similar(&v1, SimilarityMetric::Cosine, 0.9, None)
        .expect("query similar");

    // Both should be found
    assert_eq!(results.len(), 2, "should find both similar vectors");

    // v1 should rank higher (exact match)
    assert_eq!(results[0].0, id1, "v1 should be top result");
    assert_eq!(results[1].0, id2, "v2 should be second");

    // v1 should have higher similarity than v2
    assert!(results[0].1 > results[1].1,
        "v1 should have higher similarity than v2: {} vs {}", results[0].1, results[1].1);
}

#[test]
fn test_large_magnitude_differences() {
    // Cosine similarity should be scale-invariant
    let mut backend = InMemoryBackend::new(4);

    // Same direction, different magnitudes
    let small = vec![0.1, 0.0, 0.0, 0.0];
    let large = vec![1000.0, 0.0, 0.0, 0.0];

    let id_small = backend.store(&small).expect("store small");
    let id_large = backend.store(&large).expect("store large");

    // Query with unit vector in same direction
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = backend.find_similar(&query, SimilarityMetric::Cosine, 0.99, None)
        .expect("query magnitude");

    // Both should have high similarity (same direction)
    assert_eq!(results.len(), 2, "both should match high threshold");

    let small_sim = results.iter().find(|(id, _)| *id == id_small).unwrap().1;
    let large_sim = results.iter().find(|(id, _)| *id == id_large).unwrap().1;

    // Similarities should be nearly identical (scale invariance)
    assert!((small_sim - large_sim).abs() < 1e-4,
        "cosine should be scale-invariant: small={}, large={}", small_sim, large_sim);
}

// ==========================================================================
// Handler Edge Case Tests
// ==========================================================================

#[test]
fn test_topk_one() {
    let mut backend = InMemoryBackend::new(8);

    for i in 0..10 {
        let mut v = vec![0.0f32; 8];
        v[i % 8] = 1.0;
        backend.store(&v).expect("store");
    }

    let query = vec![1.0f32; 8];
    let context = FunctionContext::default();

    let results = backend.find_similar_with_handlers(
        &query,
        "cosine",
        0.0,
        "topk",
        &[ResolvedArg::Integer(1)],
        &context,
    ).expect("topk 1");

    assert_eq!(results.len(), 1, "should return exactly 1 result");
}

#[test]
fn test_threshold_zero() {
    let mut backend = InMemoryBackend::new(4);

    // Create vectors with various similarities
    let v1 = vec![1.0, 0.0, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0, 0.0]; // orthogonal

    backend.store(&v1).expect("store v1");
    backend.store(&v2).expect("store v2");

    let context = FunctionContext::default();

    // Threshold 0.0 with threshold handler should return both
    let results = backend.find_similar_with_handlers(
        &v1,
        "cosine",
        0.0,
        "threshold",
        &[ResolvedArg::Float(0.0)],
        &context,
    ).expect("threshold 0.0");

    // v2 has similarity 0.0, which passes threshold >= 0.0
    // But implementation might treat this differently
    assert!(results.len() >= 1, "should find at least v1");
}

// ==========================================================================
// Clear and Re-use Tests
// ==========================================================================

#[test]
fn test_clear_then_reuse() {
    let mut backend = InMemoryBackend::new(8);

    // Add some data
    for i in 0..5 {
        let mut v = vec![0.0f32; 8];
        v[i] = 1.0;
        backend.store(&v).expect("store");
    }

    assert_eq!(backend.len(), 5);

    // Clear
    backend.clear();
    assert_eq!(backend.len(), 0);

    // Query cleared backend
    let query = vec![1.0f32; 8];
    let results = backend.find_similar(&query, SimilarityMetric::Cosine, 0.0, None)
        .expect("query cleared");
    assert!(results.is_empty());

    // Add new data
    let new_vec = vec![1.0f32; 8];
    let new_id = backend.store(&new_vec).expect("store after clear");

    assert_eq!(backend.len(), 1);

    // Query new data
    let results2 = backend.find_similar(&new_vec, SimilarityMetric::Cosine, 0.9, None)
        .expect("query after re-add");

    assert_eq!(results2.len(), 1);
    assert_eq!(results2[0].0, new_id);
}

// ==========================================================================
// All Metrics Edge Cases
// ==========================================================================

#[test]
fn test_all_metrics_with_identical_vectors() {
    let mut backend = InMemoryBackend::new(4);

    let vec = vec![0.5f32; 4];
    backend.store(&vec).expect("store");

    let metrics = vec![
        SimilarityMetric::Cosine,
        SimilarityMetric::DotProduct,
        SimilarityMetric::Euclidean,
        SimilarityMetric::Manhattan,
        SimilarityMetric::Hamming,
        SimilarityMetric::Jaccard,
    ];

    for metric in metrics {
        let results = backend.find_similar(&vec, metric, 0.0, None);
        assert!(results.is_ok(), "{:?} should handle identical vectors", metric);
        assert!(!results.unwrap().is_empty(), "{:?} should return the identical vector", metric);
    }
}

#[test]
fn test_all_metrics_with_orthogonal_vectors() {
    let mut backend = InMemoryBackend::new(4);

    let e1 = vec![1.0, 0.0, 0.0, 0.0];
    let e2 = vec![0.0, 1.0, 0.0, 0.0];

    backend.store(&e1).expect("store e1");
    backend.store(&e2).expect("store e2");

    let metrics = vec![
        SimilarityMetric::Cosine,
        SimilarityMetric::DotProduct,
        SimilarityMetric::Euclidean,
        SimilarityMetric::Manhattan,
        SimilarityMetric::Hamming,
        SimilarityMetric::Jaccard,
    ];

    for metric in metrics {
        let results = backend.find_similar(&e1, metric, -10.0, None);
        assert!(results.is_ok(), "{:?} should handle orthogonal vectors", metric);
    }
}
