//! Property-based tests for VectorDB Similarity Matching
//!
//! This module tests the similarity pattern infrastructure introduced for VectorDB
//! spaces, including:
//! - SimilarityCollection trait implementation for VectorDBDataCollection
//! - Cosine similarity calculations
//! - Threshold-based matching
//! - Integration with GenericRSpace consume_with_similarity
//!
//! # Rholang Syntax Correspondence
//!
//! ```rholang
//! // Implicit threshold (uses space default)
//! for (@doc <- vectorChannel ~ [0.8, 0.2, 0.5]) { ... }
//!
//! // Explicit threshold
//! for (@doc <- vectorChannel ~> 0.75 ~ [0.8, 0.2, 0.5]) { ... }
//! ```

use proptest::prelude::*;
use proptest::collection::vec as prop_vec;

use rholang::rust::interpreter::spaces::collections::{
    VectorDBDataCollection, DataCollection, SimilarityCollection,
};

// =============================================================================
// VectorDB Test Helpers
// =============================================================================

/// Create a VectorDB collection with specified dimensions and threshold.
fn create_vectordb(dimensions: usize, threshold: f32) -> VectorDBDataCollection<String> {
    VectorDBDataCollection::with_threshold(dimensions, threshold)
}

/// Normalize a vector to unit length for consistent cosine similarity.
#[allow(dead_code)]
fn normalize(v: &[f32]) -> Vec<f32> {
    let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        v.iter().map(|x| x / magnitude).collect()
    } else {
        v.to_vec()
    }
}

/// Calculate cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a > 0.0 && mag_b > 0.0 {
        dot / (mag_a * mag_b)
    } else {
        0.0
    }
}

// =============================================================================
// Proptest Generators for Similarity Testing
// =============================================================================

/// Generate an arbitrary embedding vector with values in [0, 1].
fn arb_embedding(dimensions: usize) -> impl Strategy<Value = Vec<f32>> {
    prop_vec(0.0f32..1.0f32, dimensions)
}

/// Generate an arbitrary threshold in [0, 1].
#[allow(dead_code)]
fn arb_threshold() -> impl Strategy<Value = f32> {
    0.0f32..1.0f32
}

/// Generate an arbitrary dimension count (2-128).
#[allow(dead_code)]
fn arb_dimensions() -> impl Strategy<Value = usize> {
    2usize..=128
}

// =============================================================================
// SimilarityCollection Trait Tests
// =============================================================================

#[cfg(test)]
mod similarity_collection_tests {
    use super::*;

    #[test]
    fn test_similarity_collection_identical_vectors() {
        let mut vdb = create_vectordb(4, 0.9);

        // Store a document with embedding
        vdb.put_with_embedding("doc1".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put should succeed");

        // Query with identical vector - should match
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[1.0, 0.0, 0.0, 0.0],
            0.9,
        );

        assert!(result.is_some(), "Identical vectors should match");
        let (data, similarity) = result.unwrap();
        assert_eq!(data, "doc1");
        assert!((similarity - 1.0).abs() < 0.001, "Identical vectors have similarity 1.0");
    }

    #[test]
    fn test_similarity_collection_orthogonal_vectors() {
        let mut vdb = create_vectordb(4, 0.5);

        // Store a document
        vdb.put_with_embedding("doc1".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put should succeed");

        // Query with orthogonal vector - should NOT match (similarity = 0)
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[0.0, 1.0, 0.0, 0.0],
            0.5,
        );

        assert!(result.is_none(), "Orthogonal vectors should not match above 0.5 threshold");
    }

    #[test]
    fn test_similarity_collection_finds_most_similar() {
        let mut vdb = create_vectordb(4, 0.5);

        // Store multiple documents
        vdb.put_with_embedding("x_axis".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put");
        vdb.put_with_embedding("y_axis".to_string(), vec![0.0, 1.0, 0.0, 0.0])
            .expect("put");
        vdb.put_with_embedding("diagonal".to_string(), vec![0.707, 0.707, 0.0, 0.0])
            .expect("put");

        // Query for something close to diagonal
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[0.6, 0.8, 0.0, 0.0],
            0.5,
        );

        assert!(result.is_some(), "Should find a match");
        let (data, _similarity) = result.unwrap();
        // The diagonal vector is closest to [0.6, 0.8, 0, 0]
        assert_eq!(data, "diagonal");
    }

    #[test]
    fn test_similarity_collection_threshold_filtering() {
        let mut vdb = create_vectordb(4, 0.95);

        // Store a document
        vdb.put_with_embedding("strict".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put");

        // Query with vector that has ~0.9 similarity (below 0.95 threshold)
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[0.9, 0.436, 0.0, 0.0], // ~0.9 cosine similarity
            0.95,
        );

        assert!(result.is_none(), "Should not match below threshold");

        // Same query with lower threshold should match
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[0.9, 0.436, 0.0, 0.0],
            0.8,
        );

        assert!(result.is_some(), "Should match with lower threshold");
    }

    #[test]
    fn test_similarity_collection_explicit_threshold_override() {
        let mut vdb = create_vectordb(4, 0.99); // Very high default threshold

        vdb.put_with_embedding("doc1".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put");

        // Query with lower explicit threshold - should match despite high default
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[0.8, 0.6, 0.0, 0.0],
            0.6, // Explicit lower threshold
        );

        assert!(result.is_some(), "Explicit threshold should override default");
    }

    #[test]
    fn test_similarity_collection_peek_does_not_remove() {
        let mut vdb = create_vectordb(4, 0.8);

        vdb.put_with_embedding("doc1".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put");

        // Peek - should find but not remove
        let peek_result = vdb.peek_most_similar_with_threshold(
            &[1.0, 0.0, 0.0, 0.0],
            0.8,
        );
        assert!(peek_result.is_some(), "Peek should find match");
        assert_eq!(vdb.len(), 1, "Peek should not remove");

        // Find and remove - should remove
        let remove_result = vdb.find_and_remove_most_similar_with_threshold(
            &[1.0, 0.0, 0.0, 0.0],
            0.8,
        );
        assert!(remove_result.is_some(), "Find and remove should find match");
        assert_eq!(vdb.len(), 0, "Find and remove should remove");
    }

    #[test]
    fn test_similarity_collection_dimension_mismatch() {
        let mut vdb = create_vectordb(4, 0.8);

        vdb.put_with_embedding("doc1".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put");

        // Query with wrong dimensions - should return None
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[1.0, 0.0, 0.0], // 3D instead of 4D
            0.5,
        );

        assert!(result.is_none(), "Dimension mismatch should return None");
    }

    #[test]
    fn test_default_threshold() {
        let vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(4, 0.75);
        assert!((vdb.default_threshold() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_embedding_dimensions() {
        let vdb: VectorDBDataCollection<String> = VectorDBDataCollection::new(16);
        assert_eq!(vdb.embedding_dimensions(), 16);
    }
}

// =============================================================================
// Proptest Property-Based Tests
// =============================================================================

#[cfg(test)]
mod proptest_similarity_tests {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property: Identical vectors always have similarity 1.0
        #[test]
        fn prop_identical_vectors_have_max_similarity(
            dim in 2usize..32,
            embedding in prop_vec(0.1f32..1.0f32, 2..32).prop_filter("non-zero", |e| e.iter().any(|&x| x > 0.0)),
        ) {
            // Resize or truncate to match dimensions
            let embedding: Vec<f32> = embedding.into_iter().take(dim).collect();
            if embedding.len() < dim {
                return Ok(());
            }

            let mut vdb: VectorDBDataCollection<i32> = VectorDBDataCollection::with_threshold(dim, 0.99);
            vdb.put_with_embedding(42, embedding.clone()).expect("put");

            let result = vdb.find_and_remove_most_similar_with_threshold(&embedding, 0.99);
            prop_assert!(result.is_some(), "Identical vectors should match");

            let (_, similarity) = result.unwrap();
            prop_assert!((similarity - 1.0).abs() < 0.01, "Similarity should be ~1.0, got {}", similarity);
        }

        /// Property: Orthogonal vectors have similarity 0.0
        #[test]
        fn prop_orthogonal_vectors_have_zero_similarity(
            idx in 0usize..8,
        ) {
            let dim = 8;
            let mut v1 = vec![0.0f32; dim];
            let mut v2 = vec![0.0f32; dim];
            v1[idx] = 1.0;
            v2[(idx + 1) % dim] = 1.0;

            let mut vdb: VectorDBDataCollection<i32> = VectorDBDataCollection::with_threshold(dim, 0.1);
            vdb.put_with_embedding(42, v1).expect("put");

            let result = vdb.find_and_remove_most_similar_with_threshold(&v2, 0.1);
            prop_assert!(result.is_none(), "Orthogonal vectors should not match above 0.1 threshold");
        }

        /// Property: Threshold is respected
        ///
        /// We test with thresholds clearly above or below the expected similarity
        /// to avoid flaky behavior around the boundary.
        #[test]
        fn prop_threshold_respected(
            threshold_choice in 0.0f32..1.0f32,
        ) {
            // Expected similarity between [1,0,0,0] and [sqrt(0.5), sqrt(0.5), 0, 0] is sqrt(0.5) ≈ 0.7071
            let _expected_similarity = (0.5f32).sqrt();

            // Avoid the boundary region by choosing thresholds clearly above or below
            // Map threshold_choice to either [0.0, 0.6] (should match) or [0.8, 1.0] (should not match)
            let (threshold, expect_match) = if threshold_choice < 0.5 {
                (threshold_choice * 0.6, true)  // [0.0, 0.3] - well below 0.707
            } else {
                (0.8 + (threshold_choice - 0.5) * 0.4, false)  // [0.8, 1.0] - well above 0.707
            };

            let mut vdb: VectorDBDataCollection<i32> = VectorDBDataCollection::with_threshold(4, threshold);

            // Store unit vector along x-axis
            vdb.put_with_embedding(1, vec![1.0, 0.0, 0.0, 0.0]).expect("put");

            // Query with normalized vector at 45 degrees (similarity = sqrt(2)/2 ≈ 0.7071)
            let sqrt_half = (0.5f32).sqrt();
            let diagonal = vec![sqrt_half, sqrt_half, 0.0, 0.0];
            let result = vdb.find_and_remove_most_similar_with_threshold(&diagonal, threshold);

            if expect_match {
                prop_assert!(result.is_some(), "Should match when threshold={} < ~0.707", threshold);
            } else {
                prop_assert!(result.is_none(), "Should not match when threshold={} > ~0.707", threshold);
            }
        }

        /// Property: find_and_remove actually removes the item
        #[test]
        fn prop_find_and_remove_decrements_length(
            n_items in 1usize..10,
        ) {
            let mut vdb: VectorDBDataCollection<i32> = VectorDBDataCollection::with_threshold(4, 0.8);

            // Add items with different embeddings
            for i in 0..n_items {
                let mut emb = vec![0.0f32; 4];
                emb[i % 4] = 1.0;
                vdb.put_with_embedding(i as i32, emb).expect("put");
            }

            let initial_len = vdb.len();

            // Remove first item
            let result = vdb.find_and_remove_most_similar_with_threshold(&[1.0, 0.0, 0.0, 0.0], 0.8);

            if result.is_some() {
                prop_assert_eq!(vdb.len(), initial_len - 1, "Length should decrement after removal");
            }
        }

        /// Property: peek does not modify length
        #[test]
        fn prop_peek_does_not_modify_length(
            n_items in 1usize..10,
        ) {
            let mut vdb: VectorDBDataCollection<i32> = VectorDBDataCollection::with_threshold(4, 0.5);

            for i in 0..n_items {
                let mut emb = vec![0.0f32; 4];
                emb[i % 4] = 1.0;
                vdb.put_with_embedding(i as i32, emb).expect("put");
            }

            let initial_len = vdb.len();

            // Peek multiple times
            for _ in 0..5 {
                let _ = vdb.peek_most_similar_with_threshold(&[1.0, 0.0, 0.0, 0.0], 0.5);
            }

            prop_assert_eq!(vdb.len(), initial_len, "Peek should not modify length");
        }

        /// Property: Cosine similarity is symmetric
        #[test]
        fn prop_cosine_similarity_symmetric(
            a in arb_embedding(4),
            b in arb_embedding(4),
        ) {
            let sim_ab = cosine_similarity(&a, &b);
            let sim_ba = cosine_similarity(&b, &a);
            prop_assert!((sim_ab - sim_ba).abs() < 0.001, "Cosine similarity should be symmetric");
        }

        /// Property: Cosine similarity of vector with itself is 1.0 (for non-zero vectors)
        #[test]
        fn prop_cosine_similarity_with_self_is_one(
            v in arb_embedding(4).prop_filter("non-zero", |e| e.iter().any(|&x| x > 0.0)),
        ) {
            let sim = cosine_similarity(&v, &v);
            prop_assert!((sim - 1.0).abs() < 0.001, "Self-similarity should be 1.0");
        }

        /// Property: Cosine similarity is in [-1, 1]
        #[test]
        fn prop_cosine_similarity_bounded(
            a in arb_embedding(4),
            b in arb_embedding(4),
        ) {
            let sim = cosine_similarity(&a, &b);
            prop_assert!(sim >= -1.001 && sim <= 1.001, "Cosine similarity should be in [-1, 1]");
        }
    }
}

// =============================================================================
// Integration Tests for consume_with_similarity Flow
// =============================================================================
// NOTE: Full GenericRSpace integration tests for similarity are located in:
// - rholang/src/rust/interpreter/spaces/generic_rspace.rs (consume_with_similarity tests)
// - rholang/src/rust/interpreter/spaces/similarity_extraction.rs (metric tests)
// - rholang/src/rust/interpreter/spaces/vectordb/in_memory/handlers/ (handler tests)
//
// The tests below exercise pattern modifiers and edge cases that don't require
// full GenericRSpace setup.

// Minimal pattern tests that don't require full GenericRSpace integration
#[cfg(test)]
mod pattern_modifier_tests {
    use models::rhoapi::{EFunction, Par, Expr, expr::ExprInstance, EList};

    #[test]
    fn test_pattern_modifier_efunction_valid() {
        // Create a valid sim modifier as EFunction
        let query_embedding = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::EListBody(EList {
                    ps: vec![
                        Par { exprs: vec![Expr { expr_instance: Some(ExprInstance::GInt(75)) }], ..Default::default() },
                        Par { exprs: vec![Expr { expr_instance: Some(ExprInstance::GInt(50)) }], ..Default::default() },
                        Par { exprs: vec![Expr { expr_instance: Some(ExprInstance::GInt(25)) }], ..Default::default() },
                    ],
                    ..Default::default()
                })),
            }],
            ..Default::default()
        };

        let sim_function = Par {
            exprs: vec![Expr { expr_instance: Some(ExprInstance::GString("cos".to_string())) }],
            ..Default::default()
        };

        let threshold_par = Par {
            exprs: vec![Expr { expr_instance: Some(ExprInstance::GString("0.8".to_string())) }],
            ..Default::default()
        };

        // sim modifier as EFunction with: function_name="sim", arguments=[query, metric, threshold]
        let sim_modifier = EFunction {
            function_name: "sim".to_string(),
            arguments: vec![query_embedding, sim_function, threshold_par],
            locally_free: vec![],
            connective_used: false,
        };

        // Verify EFunction structure
        assert_eq!(sim_modifier.function_name, "sim");
        assert_eq!(sim_modifier.arguments.len(), 3);
    }

    #[test]
    fn test_pattern_modifier_efunction_rank() {
        // Create a rank modifier as EFunction
        let rank_function = Par {
            exprs: vec![Expr { expr_instance: Some(ExprInstance::GString("topK".to_string())) }],
            ..Default::default()
        };

        let k_param = Par {
            exprs: vec![Expr { expr_instance: Some(ExprInstance::GInt(10)) }],
            ..Default::default()
        };

        // rank modifier as EFunction with: function_name="rank", arguments=[ranking_fn, k]
        let rank_modifier = EFunction {
            function_name: "rank".to_string(),
            arguments: vec![rank_function, k_param],
            locally_free: vec![],
            connective_used: false,
        };

        // Verify EFunction structure
        assert_eq!(rank_modifier.function_name, "rank");
        assert_eq!(rank_modifier.arguments.len(), 2);
    }
}

// =============================================================================
// Edge Cases and Error Handling Tests
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_collection_returns_none() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(4, 0.5);

        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[1.0, 0.0, 0.0, 0.0],
            0.5,
        );

        assert!(result.is_none(), "Empty collection should return None");
    }

    #[test]
    fn test_zero_threshold_matches_everything() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(4, 0.0);

        vdb.put_with_embedding("doc".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put");

        // With zero threshold (clamped to EPSILON internally), very low but positive
        // similarity should match. Use a nearly-orthogonal vector with tiny overlap.
        // Note: The backend clamps threshold to f32::EPSILON to exclude tombstoned entries
        // (which have exactly 0.0 similarity), so truly orthogonal vectors won't match.
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[0.01, 0.9999, 0.0, 0.0],  // Very small positive cosine similarity with [1,0,0,0]
            0.0,
        );

        assert!(result.is_some(), "Zero threshold (clamped to EPSILON) should match any positive similarity");
    }

    #[test]
    fn test_one_threshold_requires_identical() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(4, 1.0);

        vdb.put_with_embedding("doc".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put");

        // Slightly different vector should not match with threshold 1.0
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[0.999, 0.001, 0.0, 0.0],
            1.0,
        );

        assert!(result.is_none(), "Threshold 1.0 requires exact match");

        // Identical vector should match
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[1.0, 0.0, 0.0, 0.0],
            1.0,
        );

        assert!(result.is_some(), "Identical vector should match with threshold 1.0");
    }

    #[test]
    fn test_negative_values_in_embedding() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(4, 0.8);

        // Embeddings can have negative values
        vdb.put_with_embedding("doc".to_string(), vec![-1.0, 0.0, 0.0, 0.0])
            .expect("put");

        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[-1.0, 0.0, 0.0, 0.0],
            0.8,
        );

        assert!(result.is_some(), "Negative values should work");
    }

    #[test]
    fn test_very_high_dimensional_embedding() {
        let dim = 1024;
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(dim, 0.9);

        let mut embedding = vec![0.0f32; dim];
        embedding[0] = 1.0;

        vdb.put_with_embedding("doc".to_string(), embedding.clone())
            .expect("put");

        let result = vdb.find_and_remove_most_similar_with_threshold(&embedding, 0.9);

        assert!(result.is_some(), "High dimensional embedding should work");
    }

    #[test]
    fn test_multiple_items_same_similarity() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(4, 0.5);

        // Add two items with same embedding (same similarity to any query)
        vdb.put_with_embedding("doc1".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put");
        vdb.put_with_embedding("doc2".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put");

        // Should return one of them
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[1.0, 0.0, 0.0, 0.0],
            0.5,
        );

        assert!(result.is_some(), "Should find one match");
        assert_eq!(vdb.len(), 1, "Should have removed one item");
    }
}

// =============================================================================
// Embedding Type Validation Tests
// =============================================================================

#[cfg(test)]
mod embedding_type_validation_tests {
    use rholang::rust::interpreter::spaces::similarity_extraction::extract_embedding_from_map;
    use rholang::rust::interpreter::spaces::collections::EmbeddingType;
    use models::rhoapi::{Par, Expr, expr::ExprInstance, EList, EMap, KeyValuePair};

    /// Helper to create a Par containing a map with an "embedding" key.
    fn make_embedding_map(embedding_par: Par) -> Par {
        let key_par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString("embedding".to_string())),
            }],
            ..Default::default()
        };

        let kvs = vec![KeyValuePair {
            key: Some(key_par),
            value: Some(embedding_par),
        }];

        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::EMapBody(EMap {
                    kvs,
                    ..Default::default()
                })),
            }],
            ..Default::default()
        }
    }

    /// Helper to create an EList Par from integers.
    fn make_int_list_par(values: &[i64]) -> Par {
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::EListBody(EList {
                    ps: values
                        .iter()
                        .map(|&v| Par {
                            exprs: vec![Expr {
                                expr_instance: Some(ExprInstance::GInt(v)),
                            }],
                            ..Default::default()
                        })
                        .collect(),
                    ..Default::default()
                })),
            }],
            ..Default::default()
        }
    }

    /// Helper to create a GString Par.
    fn make_string_par(s: &str) -> Par {
        Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString(s.to_string())),
            }],
            ..Default::default()
        }
    }

    // -------------------------------------------------------------------------
    // Boolean Embedding Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_boolean_embedding_valid() {
        // Valid: [0, 1, 1, 0]
        let embedding = make_int_list_par(&[0, 1, 1, 0]);
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Boolean, 4);

        assert!(result.is_ok(), "Boolean [0,1,1,0] should be valid");
        let emb = result.unwrap();
        assert_eq!(emb, vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_boolean_embedding_rejects_value_2() {
        // Invalid: [0, 2, 1] - value 2 is not allowed
        let embedding = make_int_list_par(&[0, 2, 1]);
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Boolean, 3);

        assert!(result.is_err(), "Boolean embedding should reject value 2");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("0 or 1") || msg.contains("Boolean"),
            "Error should mention boolean constraint: {}",
            msg
        );
    }

    #[test]
    fn test_boolean_embedding_rejects_negative() {
        // Invalid: [-1, 0, 1]
        let embedding = make_int_list_par(&[-1, 0, 1]);
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Boolean, 3);

        assert!(result.is_err(), "Boolean embedding should reject -1");
    }

    #[test]
    fn test_boolean_embedding_dimension_mismatch() {
        // 3 values but dimensions = 4
        let embedding = make_int_list_par(&[0, 1, 1]);
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Boolean, 4);

        assert!(result.is_err(), "Should reject dimension mismatch");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("dimension") || msg.contains("mismatch") || msg.contains("expected 4"),
            "Error should mention dimension mismatch: {}",
            msg
        );
    }

    // -------------------------------------------------------------------------
    // Integer Embedding Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_integer_embedding_valid() {
        // Valid: [50, 75, 100, 0]
        let embedding = make_int_list_par(&[50, 75, 100, 0]);
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Integer, 4);

        assert!(result.is_ok(), "Integer [50,75,100,0] should be valid");
        let emb = result.unwrap();
        // Values scaled from 0-100 to 0.0-1.0
        assert!((emb[0] - 0.5).abs() < 0.001, "50 -> 0.5");
        assert!((emb[1] - 0.75).abs() < 0.001, "75 -> 0.75");
        assert!((emb[2] - 1.0).abs() < 0.001, "100 -> 1.0");
        assert!((emb[3] - 0.0).abs() < 0.001, "0 -> 0.0");
    }

    #[test]
    fn test_integer_embedding_rejects_negative() {
        // Invalid: [-1, 50, 75] - negative not allowed
        let embedding = make_int_list_par(&[-1, 50, 75]);
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Integer, 3);

        assert!(result.is_err(), "Integer embedding should reject -1");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("0-100") || msg.contains("range"),
            "Error should mention 0-100 range: {}",
            msg
        );
    }

    #[test]
    fn test_integer_embedding_rejects_over_100() {
        // Invalid: [101, 50, 75] - value over 100 not allowed
        let embedding = make_int_list_par(&[101, 50, 75]);
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Integer, 3);

        assert!(result.is_err(), "Integer embedding should reject 101");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("0-100") || msg.contains("range") || msg.contains("101"),
            "Error should mention 0-100 range: {}",
            msg
        );
    }

    #[test]
    fn test_integer_embedding_dimension_mismatch() {
        // 2 values but dimensions = 4
        let embedding = make_int_list_par(&[50, 75]);
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Integer, 4);

        assert!(result.is_err(), "Should reject dimension mismatch");
    }

    // -------------------------------------------------------------------------
    // Float String Embedding Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_float_embedding_valid() {
        // Valid: "0.5,0.75,1.0,0.0"
        let embedding = make_string_par("0.5,0.75,1.0,0.0");
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Float, 4);

        assert!(result.is_ok(), "Float string should be valid");
        let emb = result.unwrap();
        assert!((emb[0] - 0.5).abs() < 0.001);
        assert!((emb[1] - 0.75).abs() < 0.001);
        assert!((emb[2] - 1.0).abs() < 0.001);
        assert!((emb[3] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_float_embedding_with_whitespace() {
        // Valid with whitespace: " 0.5 , 0.75 , 1.0 "
        let embedding = make_string_par(" 0.5 , 0.75 , 1.0 ");
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Float, 3);

        assert!(result.is_ok(), "Float string with whitespace should be valid");
    }

    #[test]
    fn test_float_embedding_rejects_malformed() {
        // Invalid: "abc,0.5,0.3"
        let embedding = make_string_par("abc,0.5,0.3");
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Float, 3);

        assert!(result.is_err(), "Float embedding should reject 'abc'");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("parse") || msg.contains("float") || msg.contains("abc"),
            "Error should mention parse failure: {}",
            msg
        );
    }

    #[test]
    fn test_float_embedding_rejects_integer_list() {
        // Float type expects a string, not an integer list
        let embedding = make_int_list_par(&[50, 75, 100]);
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Float, 3);

        assert!(result.is_err(), "Float type should reject integer list");
    }

    #[test]
    fn test_float_embedding_dimension_mismatch() {
        // 2 values but dimensions = 4
        let embedding = make_string_par("0.5,0.75");
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Float, 4);

        assert!(result.is_err(), "Should reject dimension mismatch");
    }

    #[test]
    fn test_float_embedding_negative_values() {
        // Negative floats are allowed
        let embedding = make_string_par("-0.5,0.75,-1.0,0.0");
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Float, 4);

        assert!(result.is_ok(), "Float embedding should allow negative values");
        let emb = result.unwrap();
        assert!((emb[0] - (-0.5)).abs() < 0.001);
        assert!((emb[2] - (-1.0)).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Missing Embedding Key Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_missing_embedding_key() {
        // Map without "embedding" key
        let key_par = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GString("other_key".to_string())),
            }],
            ..Default::default()
        };
        let value_par = make_int_list_par(&[1, 2, 3]);

        let kvs = vec![KeyValuePair {
            key: Some(key_par),
            value: Some(value_par),
        }];

        let map = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::EMapBody(EMap {
                    kvs,
                    ..Default::default()
                })),
            }],
            ..Default::default()
        };

        let result = extract_embedding_from_map(&map, EmbeddingType::Integer, 3);

        assert!(result.is_err(), "Should fail when 'embedding' key is missing");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("embedding") && msg.contains("key"),
            "Error should mention missing 'embedding' key: {}",
            msg
        );
    }

    #[test]
    fn test_non_map_data() {
        // Not a map at all, just an integer
        let non_map = Par {
            exprs: vec![Expr {
                expr_instance: Some(ExprInstance::GInt(42)),
            }],
            ..Default::default()
        };

        let result = extract_embedding_from_map(&non_map, EmbeddingType::Integer, 1);

        assert!(result.is_err(), "Should fail when data is not a map");
    }

    // -------------------------------------------------------------------------
    // Type Mismatch Tests (wrong format for declared type)
    // -------------------------------------------------------------------------

    #[test]
    fn test_boolean_type_rejects_string() {
        // Boolean type expects list, not string
        let embedding = make_string_par("0,1,1,0");
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Boolean, 4);

        assert!(result.is_err(), "Boolean type should reject string format");
    }

    #[test]
    fn test_integer_type_rejects_string() {
        // Integer type expects list, not string
        let embedding = make_string_par("50,75,100");
        let map = make_embedding_map(embedding);

        let result = extract_embedding_from_map(&map, EmbeddingType::Integer, 3);

        assert!(result.is_err(), "Integer type should reject string format");
    }
}

// =============================================================================
// SimilarityMetric Tests
// =============================================================================

#[cfg(test)]
mod similarity_metric_tests {
    use super::*;
    use rholang::rust::interpreter::spaces::collections::SimilarityMetric;

    #[test]
    fn test_cosine_similarity_with_metric() {
        let mut vdb = VectorDBDataCollection::<String>::with_metric(
            4,
            0.5,
            SimilarityMetric::Cosine,
        );

        vdb.put_with_embedding("x_axis".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .expect("put");

        // Query along x_axis - should match
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[1.0, 0.0, 0.0, 0.0],
            0.9,
        );

        assert!(result.is_some(), "Cosine similarity should match identical vectors");
    }

    #[test]
    fn test_euclidean_similarity() {
        let mut vdb = VectorDBDataCollection::<String>::with_metric(
            4,
            0.5,
            SimilarityMetric::Euclidean,
        );

        vdb.put_with_embedding("origin".to_string(), vec![0.0, 0.0, 0.0, 0.0])
            .expect("put");

        // Close to origin - high similarity (1 / (1 + distance))
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[0.1, 0.1, 0.1, 0.1],
            0.5,
        );

        assert!(result.is_some(), "Euclidean should match nearby vectors");
    }

    #[test]
    fn test_manhattan_similarity() {
        let mut vdb = VectorDBDataCollection::<String>::with_metric(
            4,
            0.3,
            SimilarityMetric::Manhattan,
        );

        vdb.put_with_embedding("corner".to_string(), vec![1.0, 1.0, 1.0, 1.0])
            .expect("put");

        // Same corner - high similarity
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[1.0, 1.0, 1.0, 1.0],
            0.9,
        );

        assert!(result.is_some(), "Manhattan should match identical vectors");
    }

    #[test]
    fn test_dot_product_similarity() {
        let mut vdb = VectorDBDataCollection::<String>::with_metric(
            4,
            0.5,
            SimilarityMetric::DotProduct,
        );

        vdb.put_with_embedding("scaled".to_string(), vec![2.0, 0.0, 0.0, 0.0])
            .expect("put");

        // Dot product with aligned vector
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[1.0, 0.0, 0.0, 0.0],
            1.0, // Dot product = 2.0
        );

        assert!(result.is_some(), "Dot product should be >= 1.0 for aligned vectors");
    }

    #[test]
    fn test_hamming_similarity_identical() {
        let mut vdb = VectorDBDataCollection::<String>::with_metric(
            8,
            0.9,
            SimilarityMetric::Hamming,
        );

        // Binary vector (hypervector style)
        vdb.put_with_embedding("binary".to_string(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            .expect("put");

        // Query with identical vector
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            0.99,
        );

        assert!(result.is_some(), "Hamming should match identical binary vectors");
    }

    #[test]
    fn test_hamming_similarity_different() {
        let mut vdb = VectorDBDataCollection::<String>::with_metric(
            8,
            0.5,
            SimilarityMetric::Hamming,
        );

        vdb.put_with_embedding("binary1".to_string(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            .expect("put");

        // Query with completely different vector (all positions different)
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            0.5,
        );

        // Hamming: 8/8 bits differ -> similarity = 1 - (8/8) = 0
        assert!(result.is_none(), "Hamming should not match completely different vectors above 0.5 threshold");
    }

    #[test]
    fn test_jaccard_similarity() {
        let mut vdb = VectorDBDataCollection::<String>::with_metric(
            6,
            0.4,
            SimilarityMetric::Jaccard,
        );

        // Set {0, 2, 4} (positions with 1.0)
        vdb.put_with_embedding("set_a".to_string(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            .expect("put");

        // Set {0, 2, 3} - intersection = {0, 2} (2), union = {0, 2, 3, 4} (4)
        // Jaccard = 2/4 = 0.5
        let result = vdb.find_and_remove_most_similar_with_threshold(
            &[1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            0.4,
        );

        assert!(result.is_some(), "Jaccard should match with similarity ~0.5");
    }
}
