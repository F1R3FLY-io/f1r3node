//! Vector and tensor operations for Rholang computations.
//!
//! This module provides SIMD-optimized operations for:
//! - Activation functions (sigmoid, softmax, heaviside)
//! - Normalization (L2 normalize)
//! - Similarity computation (cosine, euclidean, dot product)
//! - Matrix operations (gram matrix, similarity matrix)
//! - Tensor operations (superposition, retrieval)
//! - Batch operations (batch matmul, batch cosine similarity)
//! - Hypervector operations (bind, unbind, bundle, permute)
//!
//! These operations support VectorDB pattern matching and general
//! tensor computations within the Rholang interpreter.

mod operations;

pub use operations::*;
