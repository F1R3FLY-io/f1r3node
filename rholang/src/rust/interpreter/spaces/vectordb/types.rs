//! VectorDB type definitions for embeddings.
//!
//! This module defines the core types needed for VectorDB operations within rholang.
//! Similarity metrics and handler traits are defined by the backend implementations
//! (e.g., rho-vectordb) and accessed through the `VectorBackendDyn` trait.

use serde::{Deserialize, Serialize};

// =============================================================================
// Embedding Type
// =============================================================================

/// Embedding type configured at VectorDB factory construction.
///
/// This determines how Rholang sends are parsed into embedding vectors:
/// - `Boolean`: Boolean vectors like `[0, 1, 1, 0]` (best with Hamming/Jaccard metrics)
/// - `Integer`: Integer vectors on 0-100 scale like `[90, 5, 10, 20]` (scaled to 0.0-1.0)
/// - `Float`: Comma-separated float strings like `"0.9,0.05,0.1,0.2"`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum EmbeddingType {
    /// Boolean/binary vectors: [0, 1, 1, 0]
    Boolean,
    /// Integer vectors on 0-100 scale: [90, 5, 10, 20]
    Integer,
    /// Comma-separated float strings: "0.9,0.05,0.1,0.2"
    #[default]
    Float,
}

impl EmbeddingType {
    /// Parse an embedding type string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "boolean" | "bool" | "binary" => Some(Self::Boolean),
            "integer" | "int" | "scaled" => Some(Self::Integer),
            "float" | "string" => Some(Self::Float),
            _ => None,
        }
    }

    /// Get the canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Boolean => "boolean",
            Self::Integer => "integer",
            Self::Float => "float",
        }
    }
}

// =============================================================================
// Binary Packing Utilities
// =============================================================================

/// Pack a float embedding into binary representation.
///
/// Each dimension is binarized at threshold 0.5 and packed into u64 words.
/// This enables hardware-accelerated Hamming/Jaccard distance computation.
pub fn pack_to_binary(embedding: &[f32]) -> Vec<u64> {
    let num_words = (embedding.len() + 63) / 64;
    let mut packed = vec![0u64; num_words];

    for (i, &value) in embedding.iter().enumerate() {
        if value > 0.5 {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            packed[word_idx] |= 1u64 << bit_idx;
        }
    }

    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_type_from_str() {
        assert_eq!(EmbeddingType::from_str("boolean"), Some(EmbeddingType::Boolean));
        assert_eq!(EmbeddingType::from_str("BOOLEAN"), Some(EmbeddingType::Boolean));
        assert_eq!(EmbeddingType::from_str("bool"), Some(EmbeddingType::Boolean));
        assert_eq!(EmbeddingType::from_str("binary"), Some(EmbeddingType::Boolean));

        assert_eq!(EmbeddingType::from_str("integer"), Some(EmbeddingType::Integer));
        assert_eq!(EmbeddingType::from_str("int"), Some(EmbeddingType::Integer));
        assert_eq!(EmbeddingType::from_str("scaled"), Some(EmbeddingType::Integer));

        assert_eq!(EmbeddingType::from_str("float"), Some(EmbeddingType::Float));
        assert_eq!(EmbeddingType::from_str("string"), Some(EmbeddingType::Float));

        assert_eq!(EmbeddingType::from_str("unknown"), None);
    }

    #[test]
    fn test_embedding_type_as_str() {
        assert_eq!(EmbeddingType::Boolean.as_str(), "boolean");
        assert_eq!(EmbeddingType::Integer.as_str(), "integer");
        assert_eq!(EmbeddingType::Float.as_str(), "float");
    }

    #[test]
    fn test_embedding_type_default() {
        assert_eq!(EmbeddingType::default(), EmbeddingType::Float);
    }

    #[test]
    fn test_pack_to_binary() {
        let embedding = vec![0.0, 1.0, 0.0, 1.0];
        let packed = pack_to_binary(&embedding);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0b1010);
    }

    #[test]
    fn test_pack_to_binary_large() {
        // 128 dimensions - should use 2 u64 words
        let mut embedding = vec![0.0; 128];
        embedding[0] = 1.0;
        embedding[63] = 1.0;
        embedding[64] = 1.0;
        embedding[127] = 1.0;

        let packed = pack_to_binary(&embedding);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], (1u64 << 0) | (1u64 << 63));
        assert_eq!(packed[1], (1u64 << 0) | (1u64 << 63));
    }
}
