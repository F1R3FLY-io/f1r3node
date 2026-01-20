//! Error types for the rho-vectordb crate.
//!
//! These errors cover VectorDB-specific operations including similarity
//! matching, embedding extraction, and function handler errors.

use std::fmt;

/// Errors that can occur during VectorDB operations.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorDBError {
    /// Embedding dimension mismatch between data and backend configuration.
    DimensionMismatch {
        /// Expected dimensions from backend configuration
        expected: usize,
        /// Actual dimensions of the embedding
        actual: usize,
    },

    /// Error extracting embedding from data.
    EmbeddingExtractionError {
        /// Description of what went wrong
        description: String,
    },

    /// Error during similarity matching.
    SimilarityMatchError {
        /// Description of the error
        reason: String,
    },

    /// Invalid argument passed to a function handler.
    InvalidArgument(String),

    /// Unknown function identifier.
    UnknownFunction {
        /// The kind of function (similarity or ranking)
        kind: String,
        /// The unrecognized function identifier
        identifier: String,
    },

    /// Function arity mismatch.
    ArityMismatch {
        /// Function identifier
        function: String,
        /// Expected (min, max) arity
        expected: (usize, usize),
        /// Actual number of parameters
        actual: usize,
    },

    /// Backend storage error.
    StorageError {
        /// Description of the storage error
        description: String,
    },

    /// Invalid configuration.
    InvalidConfiguration {
        /// Description of the configuration error
        description: String,
    },

    /// VectorDB feature is not enabled.
    FeatureNotEnabled {
        /// The URN or operation that requires the feature
        context: String,
    },

    /// Internal error (should not occur in normal operation).
    InternalError {
        /// Description of the internal error
        description: String,
    },

    /// Unsupported similarity metric.
    UnsupportedMetric(String),
}

impl fmt::Display for VectorDBError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorDBError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Embedding dimension mismatch: expected {} dimensions, got {}",
                    expected, actual
                )
            }
            VectorDBError::EmbeddingExtractionError { description } => {
                write!(f, "Embedding extraction error: {}", description)
            }
            VectorDBError::SimilarityMatchError { reason } => {
                write!(f, "Similarity matching error: {}", reason)
            }
            VectorDBError::InvalidArgument(msg) => {
                write!(f, "Invalid argument: {}", msg)
            }
            VectorDBError::UnknownFunction { kind, identifier } => {
                write!(f, "Unknown {} function: '{}'", kind, identifier)
            }
            VectorDBError::ArityMismatch {
                function,
                expected,
                actual,
            } => {
                if expected.0 == expected.1 {
                    write!(
                        f,
                        "Function '{}' expects {} parameters, got {}",
                        function, expected.0, actual
                    )
                } else {
                    write!(
                        f,
                        "Function '{}' expects {}-{} parameters, got {}",
                        function, expected.0, expected.1, actual
                    )
                }
            }
            VectorDBError::StorageError { description } => {
                write!(f, "Storage error: {}", description)
            }
            VectorDBError::InvalidConfiguration { description } => {
                write!(f, "Invalid configuration: {}", description)
            }
            VectorDBError::FeatureNotEnabled { context } => {
                write!(
                    f,
                    "VectorDB feature not enabled. Recompile with --features vectordb. Context: {}",
                    context
                )
            }
            VectorDBError::InternalError { description } => {
                write!(f, "Internal VectorDB error: {}", description)
            }
            VectorDBError::UnsupportedMetric(msg) => {
                write!(f, "Unsupported metric: {}", msg)
            }
        }
    }
}

impl std::error::Error for VectorDBError {}

// Implement thiserror-style conversion for common error types
impl From<String> for VectorDBError {
    fn from(s: String) -> Self {
        VectorDBError::InternalError { description: s }
    }
}

impl From<&str> for VectorDBError {
    fn from(s: &str) -> Self {
        VectorDBError::InternalError {
            description: s.to_string(),
        }
    }
}
