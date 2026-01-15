//! Error handling tests for rho-vectordb.
//!
//! These tests validate that all VectorDBError variants are correctly
//! generated and provide useful error messages.

use rholang::rust::interpreter::spaces::vectordb::in_memory::{
    InMemoryBackend, VectorBackend, HandlerBackend, SimilarityMetric,
    VectorDBError, FunctionContext, ResolvedArg,
};

// ==========================================================================
// DimensionMismatch Error Tests
// ==========================================================================

#[test]
fn test_dimension_mismatch_on_store() {
    let mut backend = InMemoryBackend::new(64);

    // Try to store embedding with wrong dimensions
    let wrong_dim_emb = vec![1.0f32; 32]; // 32 instead of 64
    let result = backend.store(&wrong_dim_emb);

    assert!(result.is_err(), "should reject wrong dimension");
    match result.unwrap_err() {
        VectorDBError::DimensionMismatch { expected, actual } => {
            assert_eq!(expected, 64);
            assert_eq!(actual, 32);
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn test_dimension_mismatch_on_query() {
    let mut backend = InMemoryBackend::new(64);

    // Store valid embedding
    let valid_emb = vec![1.0f32; 64];
    backend.store(&valid_emb).expect("store should work");

    // Query with wrong dimensions
    let wrong_query = vec![1.0f32; 128];
    let result = backend.find_similar(&wrong_query, SimilarityMetric::Cosine, 0.0, None);

    assert!(result.is_err(), "should reject wrong dimension query");
    match result.unwrap_err() {
        VectorDBError::DimensionMismatch { expected, actual } => {
            assert_eq!(expected, 64);
            assert_eq!(actual, 128);
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn test_dimension_mismatch_error_message() {
    let err = VectorDBError::DimensionMismatch {
        expected: 128,
        actual: 64,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("128"), "message should contain expected dim");
    assert!(msg.contains("64"), "message should contain actual dim");
    assert!(msg.contains("mismatch"), "message should mention mismatch");
}

// ==========================================================================
// UnknownFunction Error Tests
// ==========================================================================

#[test]
fn test_unknown_similarity_function() {
    let mut backend = InMemoryBackend::new(16);
    let emb = vec![1.0f32; 16];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let result = backend.find_similar_with_handlers(
        &emb,
        "nonexistent_metric",
        0.0,
        "all",
        &[],
        &context,
    );

    assert!(result.is_err(), "should reject unknown similarity function");
    // Unknown similarity metric returns UnsupportedMetric
    match result.unwrap_err() {
        VectorDBError::UnsupportedMetric(msg) => {
            assert!(msg.contains("nonexistent_metric"), "message should contain the metric name");
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn test_unknown_ranking_function() {
    let mut backend = InMemoryBackend::new(16);
    let emb = vec![1.0f32; 16];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();
    let result = backend.find_similar_with_handlers(
        &emb,
        "cosine",
        0.0,
        "nonexistent_ranking",
        &[],
        &context,
    );

    assert!(result.is_err(), "should reject unknown ranking function");
    // Unknown ranking function returns InvalidArgument
    match result.unwrap_err() {
        VectorDBError::InvalidArgument(msg) => {
            assert!(msg.contains("nonexistent_ranking") || msg.contains("Unknown ranking"),
                "message should mention the unknown function: {}", msg);
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn test_unknown_function_error_message() {
    let err = VectorDBError::UnknownFunction {
        kind: "similarity".to_string(),
        identifier: "fancy_metric".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("similarity"), "message should contain function kind");
    assert!(msg.contains("fancy_metric"), "message should contain identifier");
}

// ==========================================================================
// InvalidArgument Error Tests
// ==========================================================================

#[test]
fn test_topk_requires_integer() {
    let mut backend = InMemoryBackend::new(16);
    let emb = vec![1.0f32; 16];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();

    // Pass string instead of integer
    let result = backend.find_similar_with_handlers(
        &emb,
        "cosine",
        0.0,
        "topk",
        &[ResolvedArg::String("not_a_number".to_string())],
        &context,
    );

    assert!(result.is_err(), "should reject non-integer k");
    match result.unwrap_err() {
        VectorDBError::InvalidArgument(msg) => {
            assert!(msg.contains("integer") || msg.contains("K") || msg.contains("topk"),
                "message should explain the issue: {}", msg);
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn test_threshold_requires_float() {
    let mut backend = InMemoryBackend::new(16);
    let emb = vec![1.0f32; 16];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();

    // Pass string instead of float for threshold
    let result = backend.find_similar_with_handlers(
        &emb,
        "cosine",
        0.0,
        "threshold",
        &[ResolvedArg::String("not_a_number".to_string())],
        &context,
    );

    assert!(result.is_err(), "should reject non-float threshold");
    match result.unwrap_err() {
        VectorDBError::InvalidArgument(msg) => {
            assert!(msg.contains("float") || msg.contains("threshold"),
                "message should explain the issue: {}", msg);
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn test_invalid_argument_error_message() {
    let err = VectorDBError::InvalidArgument("K must be positive integer".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("K must be positive"), "message should contain description");
}

// ==========================================================================
// ArityMismatch Error Tests
// ==========================================================================

#[test]
fn test_topk_missing_argument() {
    let mut backend = InMemoryBackend::new(16);
    let emb = vec![1.0f32; 16];
    backend.store(&emb).expect("store");

    let context = FunctionContext::default();

    // TopK requires exactly 1 argument (k)
    let result = backend.find_similar_with_handlers(
        &emb,
        "cosine",
        0.0,
        "topk",
        &[], // Missing required k argument
        &context,
    );

    // This should fail with either ArityMismatch or InvalidArgument
    assert!(result.is_err(), "should reject missing argument");
}

#[test]
fn test_arity_mismatch_error_message() {
    let err = VectorDBError::ArityMismatch {
        function: "topk".to_string(),
        expected: (1, 1),
        actual: 0,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("topk"), "message should contain function name");
    assert!(msg.contains("1"), "message should contain expected arity");
    assert!(msg.contains("0"), "message should contain actual arity");
}

#[test]
fn test_arity_mismatch_range_error_message() {
    let err = VectorDBError::ArityMismatch {
        function: "fancy".to_string(),
        expected: (1, 3),
        actual: 5,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("fancy"), "message should contain function name");
    assert!(msg.contains("1-3") || msg.contains("1") && msg.contains("3"),
        "message should contain expected range: {}", msg);
}

// ==========================================================================
// Error Type Property Tests
// ==========================================================================

#[test]
fn test_errors_are_eq() {
    let err1 = VectorDBError::DimensionMismatch {
        expected: 64,
        actual: 32,
    };
    let err2 = VectorDBError::DimensionMismatch {
        expected: 64,
        actual: 32,
    };
    let err3 = VectorDBError::DimensionMismatch {
        expected: 128,
        actual: 32,
    };

    assert_eq!(err1, err2, "same errors should be equal");
    assert_ne!(err1, err3, "different errors should not be equal");
}

#[test]
fn test_errors_are_clone() {
    let err = VectorDBError::InvalidArgument("test".to_string());
    let cloned = err.clone();
    assert_eq!(err, cloned, "cloned error should equal original");
}

#[test]
fn test_errors_are_debug() {
    let err = VectorDBError::UnknownFunction {
        kind: "test".to_string(),
        identifier: "foo".to_string(),
    };
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("UnknownFunction"), "debug should include variant name");
}

#[test]
fn test_errors_implement_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(VectorDBError::InternalError {
        description: "something went wrong".to_string(),
    });
    let msg = format!("{}", err);
    assert!(msg.contains("something went wrong"));
}

// ==========================================================================
// Error Conversion Tests
// ==========================================================================

#[test]
fn test_string_converts_to_internal_error() {
    let err: VectorDBError = String::from("test error").into();
    match err {
        VectorDBError::InternalError { description } => {
            assert_eq!(description, "test error");
        }
        other => panic!("expected InternalError, got {:?}", other),
    }
}

#[test]
fn test_str_converts_to_internal_error() {
    let err: VectorDBError = "test error".into();
    match err {
        VectorDBError::InternalError { description } => {
            assert_eq!(description, "test error");
        }
        other => panic!("expected InternalError, got {:?}", other),
    }
}

// ==========================================================================
// All Error Variant Display Tests
// ==========================================================================

#[test]
fn test_embedding_extraction_error_display() {
    let err = VectorDBError::EmbeddingExtractionError {
        description: "failed to parse vector".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Embedding extraction"));
    assert!(msg.contains("failed to parse vector"));
}

#[test]
fn test_similarity_match_error_display() {
    let err = VectorDBError::SimilarityMatchError {
        reason: "vectors have NaN".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Similarity matching error"));
    assert!(msg.contains("NaN"));
}

#[test]
fn test_storage_error_display() {
    let err = VectorDBError::StorageError {
        description: "disk full".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Storage error"));
    assert!(msg.contains("disk full"));
}

#[test]
fn test_invalid_configuration_display() {
    let err = VectorDBError::InvalidConfiguration {
        description: "invalid dimensions".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Invalid configuration"));
    assert!(msg.contains("invalid dimensions"));
}

#[test]
fn test_feature_not_enabled_display() {
    let err = VectorDBError::FeatureNotEnabled {
        context: "vectordb URN".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("feature not enabled"));
    assert!(msg.contains("vectordb URN"));
}

#[test]
fn test_internal_error_display() {
    let err = VectorDBError::InternalError {
        description: "unexpected null".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("Internal VectorDB error"));
    assert!(msg.contains("unexpected null"));
}

#[test]
fn test_unsupported_metric_display() {
    let err = VectorDBError::UnsupportedMetric("quantum_similarity".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("Unsupported metric"));
    assert!(msg.contains("quantum_similarity"));
}

// ==========================================================================
// Edge Case Error Tests
// ==========================================================================

#[test]
fn test_empty_backend_query_succeeds() {
    // Empty backend query should NOT error - just return empty results
    let backend = InMemoryBackend::new(16);
    let query = vec![1.0f32; 16];

    let result = backend.find_similar(&query, SimilarityMetric::Cosine, 0.0, None);
    assert!(result.is_ok(), "empty backend query should succeed");
    assert!(result.unwrap().is_empty(), "should return empty results");
}

#[test]
fn test_zero_dimension_handling() {
    // Backend with 0 dimensions - this is an edge case
    let mut backend = InMemoryBackend::new(0);

    // Try to store an empty vector
    let empty_emb: Vec<f32> = vec![];
    let result = backend.store(&empty_emb);

    // This should either succeed or return a sensible error
    // The behavior depends on implementation
    match result {
        Ok(_) => {
            // If storing succeeds, querying should work too
            let query: Vec<f32> = vec![];
            let query_result = backend.find_similar(&query, SimilarityMetric::Cosine, 0.0, None);
            assert!(query_result.is_ok(), "0-dim query should not crash");
        }
        Err(e) => {
            // If it fails, it should be a sensible error
            let msg = format!("{}", e);
            assert!(!msg.is_empty(), "error should have a message");
        }
    }
}
