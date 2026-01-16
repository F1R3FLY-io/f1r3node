//! Property-based tests for rho-vectordb invariants.
//!
//! These tests use proptest to validate mathematical invariants and edge cases
//! that unit tests might miss.

use std::collections::HashSet;

use proptest::prelude::*;
use ndarray::Array1;

use rholang::rust::interpreter::spaces::vectordb::in_memory::{
    InMemoryBackend, VectorBackend, HandlerBackend, SimilarityMetric,
    ResolvedArg, FunctionContext,
};

// Import tensor operations from rholang
use rholang::rust::interpreter::tensor::{
    cosine_similarity, cosine_similarity_safe, euclidean_distance, dot_product,
    l2_normalize, sigmoid, softmax, temperature_sigmoid, heaviside_f32,
    bind, unbind, permute, unpermute, bundle, hamming_similarity,
};

// ============================================================================
// TEST STRATEGIES
// ============================================================================

/// Generate a random f32 embedding of fixed dimension.
fn embedding_strategy(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1.0f32..1.0f32, dim)
}

/// Generate a normalized embedding (unit vector).
fn normalized_embedding_strategy(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    embedding_strategy(dim).prop_map(|v| {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            v.iter().map(|x| x / norm).collect()
        } else {
            // Return a unit vector if the generated vector is near-zero
            let mut unit = vec![0.0f32; v.len()];
            if !unit.is_empty() {
                unit[0] = 1.0;
            }
            unit
        }
    })
}

/// Generate a batch of embeddings.
fn embedding_batch_strategy(dim: usize, min_count: usize, max_count: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
    prop::collection::vec(normalized_embedding_strategy(dim), min_count..=max_count)
}

/// Operations on a backend
#[derive(Clone, Debug)]
enum BackendOp {
    Store(Vec<f32>),
    Remove(usize),
}

/// Generate a sequence of backend operations
fn operation_sequence_strategy(dim: usize, max_ops: usize) -> impl Strategy<Value = Vec<BackendOp>> {
    prop::collection::vec(
        prop_oneof![
            3 => normalized_embedding_strategy(dim).prop_map(BackendOp::Store),
            1 => (0usize..100).prop_map(BackendOp::Remove),
        ],
        1..=max_ops
    )
}

/// Generate a binary hypervector (0s and 1s as i64)
fn binary_hypervector_strategy(dim: usize) -> impl Strategy<Value = Vec<i64>> {
    prop::collection::vec(prop::bool::ANY.prop_map(|b| if b { 1i64 } else { 0i64 }), dim)
}

// ============================================================================
// SIMILARITY METRIC INVARIANTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Cosine similarity is always in [-1, 1] for non-zero vectors
    #[test]
    fn cosine_similarity_bounded(
        v1 in normalized_embedding_strategy(64),
        v2 in normalized_embedding_strategy(64)
    ) {
        let a = Array1::from_vec(v1);
        let b = Array1::from_vec(v2);
        let sim = cosine_similarity(&a, &b);
        prop_assert!(sim >= -1.0 - f32::EPSILON && sim <= 1.0 + f32::EPSILON,
            "cosine out of range: {}", sim);
    }

    /// Cosine similarity of a vector with itself is 1.0
    #[test]
    fn cosine_self_similarity(v in normalized_embedding_strategy(64)) {
        let a = Array1::from_vec(v);
        let sim = cosine_similarity(&a, &a);
        prop_assert!((sim - 1.0).abs() < 1e-4,
            "self-similarity should be 1.0, got: {}", sim);
    }

    /// Cosine similarity is symmetric: cos(a,b) = cos(b,a)
    #[test]
    fn cosine_symmetric(
        v1 in normalized_embedding_strategy(64),
        v2 in normalized_embedding_strategy(64)
    ) {
        let a = Array1::from_vec(v1);
        let b = Array1::from_vec(v2);
        let sim1 = cosine_similarity(&a, &b);
        let sim2 = cosine_similarity(&b, &a);
        prop_assert!((sim1 - sim2).abs() < 1e-5,
            "cosine not symmetric: {} vs {}", sim1, sim2);
    }

    /// Euclidean distance is always non-negative
    #[test]
    fn euclidean_non_negative(
        v1 in embedding_strategy(64),
        v2 in embedding_strategy(64)
    ) {
        let a = Array1::from_vec(v1);
        let b = Array1::from_vec(v2);
        let dist = euclidean_distance(&a, &b);
        prop_assert!(dist >= 0.0, "euclidean negative: {}", dist);
    }

    /// Euclidean distance is zero for identical vectors
    #[test]
    fn euclidean_self_zero(v in embedding_strategy(64)) {
        let a = Array1::from_vec(v);
        let dist = euclidean_distance(&a, &a);
        prop_assert!(dist.abs() < 1e-5,
            "self-distance should be 0: {}", dist);
    }

    /// Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    #[test]
    fn euclidean_triangle_inequality(
        v1 in embedding_strategy(32),
        v2 in embedding_strategy(32),
        v3 in embedding_strategy(32)
    ) {
        let a = Array1::from_vec(v1);
        let b = Array1::from_vec(v2);
        let c = Array1::from_vec(v3);

        let d_ac = euclidean_distance(&a, &c);
        let d_ab = euclidean_distance(&a, &b);
        let d_bc = euclidean_distance(&b, &c);

        // Allow small epsilon for floating point
        prop_assert!(d_ac <= d_ab + d_bc + 1e-4,
            "triangle inequality violated: {} > {} + {}", d_ac, d_ab, d_bc);
    }

    /// Dot product is symmetric
    #[test]
    fn dot_product_symmetric(
        v1 in embedding_strategy(64),
        v2 in embedding_strategy(64)
    ) {
        let a = Array1::from_vec(v1);
        let b = Array1::from_vec(v2);
        let d1 = dot_product(&a, &b);
        let d2 = dot_product(&b, &a);
        prop_assert!((d1 - d2).abs() < 1e-4,
            "dot product not symmetric: {} vs {}", d1, d2);
    }

    /// L2 normalization produces unit vectors
    #[test]
    fn l2_normalize_produces_unit(v in embedding_strategy(64)) {
        let a = Array1::from_vec(v);
        let norm_sq: f32 = a.iter().map(|x| x * x).sum();

        // Skip zero vectors
        if norm_sq > f32::EPSILON {
            let normalized = l2_normalize(&a);
            let result_norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!((result_norm - 1.0).abs() < 1e-4,
                "normalized vector should have unit norm, got: {}", result_norm);
        }
    }

    /// Safe cosine similarity handles zero vectors gracefully
    #[test]
    fn cosine_safe_handles_zero(
        v in embedding_strategy(64)
    ) {
        let a = Array1::from_vec(v);
        let zero = Array1::zeros(64);
        let sim = cosine_similarity_safe(&a, &zero);
        // Should return 0.0 for zero vector, not NaN or Inf
        prop_assert!(sim.is_finite(), "cosine_safe should be finite: {}", sim);
        prop_assert!(sim == 0.0, "cosine with zero vector should be 0: {}", sim);
    }
}

// ============================================================================
// RANKING FUNCTION INVARIANTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// TopK never returns more than k results
    #[test]
    fn topk_cardinality(
        embeddings in embedding_batch_strategy(32, 5, 50),
        k in 1usize..20
    ) {
        let mut backend = InMemoryBackend::new(32);

        // Store embeddings
        for emb in &embeddings {
            let _ = backend.store(emb);
        }

        if !embeddings.is_empty() {
            let query = &embeddings[0];
            let results = backend.find_similar(query, SimilarityMetric::Cosine, 0.0, Some(k))
                .expect("find_similar should succeed");

            prop_assert!(results.len() <= k,
                "topk returned {} > k={}", results.len(), k);
        }
    }

    /// TopK results are sorted by score descending
    #[test]
    fn topk_sorted_descending(
        embeddings in embedding_batch_strategy(32, 5, 30),
        k in 1usize..15
    ) {
        let mut backend = InMemoryBackend::new(32);

        for emb in &embeddings {
            let _ = backend.store(emb);
        }

        if !embeddings.is_empty() {
            let query = &embeddings[0];
            let results = backend.find_similar(query, SimilarityMetric::Cosine, 0.0, Some(k))
                .expect("find_similar should succeed");

            for window in results.windows(2) {
                prop_assert!(window[0].1 >= window[1].1,
                    "results not sorted: {} < {}", window[0].1, window[1].1);
            }
        }
    }

    /// Threshold filter returns only scores >= threshold
    #[test]
    fn threshold_respects_cutoff(
        embeddings in embedding_batch_strategy(32, 5, 30),
        threshold in 0.0f32..1.0f32
    ) {
        let mut backend = InMemoryBackend::new(32);

        for emb in &embeddings {
            let _ = backend.store(emb);
        }

        if !embeddings.is_empty() {
            let query = &embeddings[0];
            // Use find_similar with threshold
            let results = backend.find_similar(query, SimilarityMetric::Cosine, threshold, None)
                .expect("find_similar should succeed");

            for (idx, score) in results {
                prop_assert!(score >= threshold - f32::EPSILON,
                    "score {} < threshold {} at idx {}", score, threshold, idx);
            }
        }
    }
}

// ============================================================================
// BACKEND STATE INVARIANTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Backend count equals stores minus removes (for valid removes)
    #[test]
    fn backend_count_invariant(
        ops in operation_sequence_strategy(32, 30)
    ) {
        let mut backend = InMemoryBackend::new(32);
        let mut valid_ids: HashSet<usize> = HashSet::new();

        for op in ops {
            match op {
                BackendOp::Store(emb) => {
                    if let Ok(id) = backend.store(&emb) {
                        valid_ids.insert(id);
                    }
                }
                BackendOp::Remove(id) => {
                    if valid_ids.remove(&id) {
                        backend.remove(&id);
                    }
                }
            }
        }

        prop_assert_eq!(backend.len(), valid_ids.len(),
            "backend len {} != tracked count {}", backend.len(), valid_ids.len());
    }

    /// Removed embeddings are not returned in queries
    #[test]
    fn removed_not_in_results(
        seeds in prop::collection::vec(0u32..1000, 5..15),
        remove_indices in prop::collection::hash_set(0usize..15, 1..5)
    ) {
        // Create unique embeddings using seed values
        let embeddings: Vec<Vec<f32>> = seeds.iter().enumerate().map(|(i, &seed)| {
            let mut emb = vec![0.0f32; 32];
            emb[i % 32] = 1.0;
            emb[(i + 1) % 32] = (seed as f32) / 1000.0;
            // Normalize
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            emb.iter().map(|x| x / norm).collect()
        }).collect();

        let mut backend = InMemoryBackend::new(32);
        let mut ids = Vec::new();

        for emb in &embeddings {
            if let Ok(id) = backend.store(emb) {
                ids.push(id);
            }
        }

        // Remove some embeddings (skip index 0 so we can query with it)
        let mut removed_ids = HashSet::new();
        for &idx in &remove_indices {
            if idx > 0 && idx < ids.len() {
                backend.remove(&ids[idx]);
                removed_ids.insert(ids[idx]);
            }
        }

        // Query with the first embedding (which we kept)
        if !embeddings.is_empty() && !ids.is_empty() {
            let query = &embeddings[0];
            let results = backend.find_similar(query, SimilarityMetric::Cosine, 0.0, None)
                .expect("find_similar should succeed");

            for (result_id, _) in results {
                prop_assert!(!removed_ids.contains(&result_id),
                    "removed id {} found in results", result_id);
            }
        }
    }

    /// Slot reuse doesn't corrupt data
    #[test]
    fn slot_reuse_correctness(
        initial in embedding_batch_strategy(16, 3, 8),
        replacements in embedding_batch_strategy(16, 3, 8)
    ) {
        let mut backend = InMemoryBackend::new(16);
        let mut ids: Vec<usize> = Vec::new();

        // Store initial embeddings
        for emb in &initial {
            if let Ok(id) = backend.store(emb) {
                ids.push(id);
            }
        }

        // Remove all
        for id in &ids {
            backend.remove(id);
        }

        // Store replacements (should reuse slots)
        ids.clear();
        for emb in &replacements {
            if let Ok(id) = backend.store(emb) {
                ids.push(id);
            }
        }

        // Backend should have exactly the number of replacements stored
        prop_assert_eq!(backend.len(), ids.len());

        // Verify query returns the replacement embeddings
        if !replacements.is_empty() {
            let query = &replacements[0];
            let results = backend.find_similar(query, SimilarityMetric::Cosine, 0.0, None)
                .expect("find_similar should succeed");

            // The best match should be the query itself (similarity ~1.0)
            if !results.is_empty() {
                prop_assert!(results[0].1 > 0.99,
                    "best match should be ~1.0, got: {}", results[0].1);
            }
        }
    }

    /// Empty backend returns empty results
    #[test]
    fn empty_backend_returns_empty(
        query in normalized_embedding_strategy(32)
    ) {
        let backend = InMemoryBackend::new(32);
        let results = backend.find_similar(&query, SimilarityMetric::Cosine, 0.0, Some(10))
            .expect("find_similar should succeed");
        prop_assert!(results.is_empty(), "empty backend should return no results");
    }

    /// Dimension mismatch is rejected
    #[test]
    fn dimension_mismatch_rejected(
        embedding in embedding_strategy(32),
        wrong_dim in 1usize..31
    ) {
        let mut backend = InMemoryBackend::new(32);

        // Create wrong-dimension embedding
        let wrong_emb: Vec<f32> = embedding.iter().take(wrong_dim).cloned().collect();

        let result = backend.store(&wrong_emb);
        prop_assert!(result.is_err(), "wrong dimension should fail");
    }
}

// ============================================================================
// HYPERVECTOR OPERATION INVARIANTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Bind is self-inverse: unbind(bind(a, b), b) = a
    #[test]
    fn bind_self_inverse(
        a in binary_hypervector_strategy(128),
        b in binary_hypervector_strategy(128)
    ) {
        let bound = bind(&a, &b);
        let unbound = unbind(&bound, &b);

        prop_assert_eq!(unbound, a, "bind should be self-inverse");
    }

    /// Permute and unpermute are inverses
    #[test]
    fn permute_inverse(
        v in binary_hypervector_strategy(128),
        shift in 0i64..128
    ) {
        let permuted = permute(&v, shift);
        let unpermuted = unpermute(&permuted, shift);

        prop_assert_eq!(unpermuted, v, "permute/unpermute should be inverses");
    }

    /// Permute by length returns original
    #[test]
    fn permute_full_rotation(
        v in binary_hypervector_strategy(64)
    ) {
        let len = v.len() as i64;
        let permuted = permute(&v, len);
        prop_assert_eq!(permuted, v, "full rotation should return original");
    }

    /// Hamming similarity of identical vectors is 100%
    #[test]
    fn hamming_identical_is_100(
        v in binary_hypervector_strategy(64)
    ) {
        let sim = hamming_similarity(&v, &v);
        prop_assert_eq!(sim, 100, "identical vectors should have 100% similarity");
    }

    /// Hamming similarity is symmetric
    #[test]
    fn hamming_symmetric(
        a in binary_hypervector_strategy(64),
        b in binary_hypervector_strategy(64)
    ) {
        let sim1 = hamming_similarity(&a, &b);
        let sim2 = hamming_similarity(&b, &a);
        prop_assert_eq!(sim1, sim2, "hamming should be symmetric");
    }

    /// Hamming similarity is in [0, 100]
    #[test]
    fn hamming_in_range(
        a in binary_hypervector_strategy(64),
        b in binary_hypervector_strategy(64)
    ) {
        let sim = hamming_similarity(&a, &b);
        prop_assert!(sim >= 0 && sim <= 100,
            "hamming similarity should be in [0, 100], got: {}", sim);
    }

    /// Bundle produces valid binary vector
    #[test]
    fn bundle_produces_binary(
        vectors in prop::collection::vec(binary_hypervector_strategy(32), 2..5)
    ) {
        let refs: Vec<&[i64]> = vectors.iter().map(|v| v.as_slice()).collect();
        let bundled = bundle(&refs);

        for &val in &bundled {
            prop_assert!(val == 0 || val == 1,
                "bundle should produce binary values, got: {}", val);
        }
    }

    /// Bundle of single vector is the vector itself
    #[test]
    fn bundle_single_is_identity(
        v in binary_hypervector_strategy(64)
    ) {
        let bundled = bundle(&[v.as_slice()]);
        prop_assert_eq!(bundled, v, "bundle of single vector should be identity");
    }
}

// ============================================================================
// HANDLER BACKEND INTEGRATION INVARIANTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Handler-based query produces valid results
    #[test]
    fn handler_query_valid(
        embeddings in embedding_batch_strategy(32, 5, 20),
        k in 1usize..10
    ) {
        let mut backend = InMemoryBackend::new(32);

        for emb in &embeddings {
            let _ = backend.store(emb);
        }

        if !embeddings.is_empty() {
            let query = &embeddings[0];
            let context = FunctionContext::default();

            // Test with different similarity functions
            for metric in &["cosine", "cos", "euclidean", "dot"] {
                let results = backend.find_similar_with_handlers(
                    query,
                    *metric,
                    0.0,
                    "topk",
                    &[ResolvedArg::Integer(k as i64)],
                    &context,
                );

                prop_assert!(results.is_ok(),
                    "handler query with {} should succeed", metric);

                let results = results.unwrap();
                prop_assert!(results.len() <= k,
                    "handler topk should return <= k results");
            }
        }
    }

    /// Unknown function returns error
    #[test]
    fn unknown_function_errors(
        embeddings in embedding_batch_strategy(32, 3, 5)
    ) {
        let mut backend = InMemoryBackend::new(32);

        for emb in &embeddings {
            let _ = backend.store(emb);
        }

        if !embeddings.is_empty() {
            let query = &embeddings[0];
            let context = FunctionContext::default();

            let result = backend.find_similar_with_handlers(
                query,
                "nonexistent_metric",
                0.8,
                "topk",
                &[ResolvedArg::Integer(5)],
                &context,
            );

            prop_assert!(result.is_err(),
                "unknown function should return error");
        }
    }
}

// ============================================================================
// TENSOR OPERATION INVARIANTS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Softmax produces probability distribution (sums to 1)
    #[test]
    fn softmax_sums_to_one(
        v in prop::collection::vec(-10.0f32..10.0f32, 2..20)
    ) {
        let a = Array1::from_vec(v);
        let result = softmax(&a);
        let sum: f32 = result.iter().sum();

        prop_assert!((sum - 1.0).abs() < 1e-4,
            "softmax should sum to 1.0, got: {}", sum);
    }

    /// Softmax produces non-negative values
    #[test]
    fn softmax_non_negative(
        v in prop::collection::vec(-10.0f32..10.0f32, 2..20)
    ) {
        let a = Array1::from_vec(v);
        let result = softmax(&a);

        for val in result.iter() {
            prop_assert!(*val >= 0.0, "softmax should be non-negative, got: {}", val);
        }
    }

    /// Sigmoid is bounded in [0, 1] (inclusive due to floating point)
    #[test]
    fn sigmoid_bounded(
        v in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        let a = Array1::from_vec(v);
        let result = sigmoid(&a);

        for val in result.iter() {
            // For very large/small inputs, sigmoid saturates to exactly 0.0 or 1.0
            prop_assert!(*val >= 0.0 && *val <= 1.0,
                "sigmoid should be in [0, 1], got: {}", val);
        }
    }

    /// Temperature sigmoid with T=1 equals regular sigmoid
    #[test]
    fn temperature_sigmoid_t1_equals_sigmoid(
        v in prop::collection::vec(-10.0f32..10.0f32, 1..20)
    ) {
        let a = Array1::from_vec(v);
        let sig = sigmoid(&a);
        let temp_sig = temperature_sigmoid(&a, 1.0);

        for (s, t) in sig.iter().zip(temp_sig.iter()) {
            prop_assert!((s - t).abs() < 1e-5,
                "T=1 temperature sigmoid should equal sigmoid");
        }
    }

    /// Heaviside produces only 0 and 1
    #[test]
    fn heaviside_binary_output(
        v in prop::collection::vec(-10.0f32..10.0f32, 1..50)
    ) {
        let a = Array1::from_vec(v);
        let result = heaviside_f32(&a);

        for val in result.iter() {
            prop_assert!(*val == 0.0 || *val == 1.0,
                "heaviside_f32 should produce 0 or 1, got: {}", val);
        }
    }
}
