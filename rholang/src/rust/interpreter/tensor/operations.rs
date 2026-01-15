//! Vector and Tensor Operations for Rholang
//!
//! This module provides high-performance vector and tensor operations using ndarray
//! with BLAS acceleration and parallel processing via Rayon.
//!
//! # Tensor Logic Integration
//!
//! Based on the paper "Tensor Logic: The Language of AI", this module implements:
//! - Einstein summation (einsum) for tensor contractions
//! - Embedding superposition and retrieval operations
//! - Temperature-controlled sigmoid for soft/hard matching
//! - Gram matrix computation for similarity
//!
//! # Performance Features
//!
//! - `rayon`: Parallel element-wise operations for vectors > 1024 elements
//! - `blas`: BLAS-accelerated dot products and matrix operations
//! - `matrixmultiply-threading`: Multi-threaded matrix multiplication
//!
//! # Example Usage
//!
//! ```ignore
//! use ndarray::Array1;
//! use rholang::interpreter::tensor::*;
//!
//! let v = Array1::from_vec(vec![0.1, -0.5, 2.3]);
//!
//! // Sigmoid normalization
//! let normalized = sigmoid(&v);
//!
//! // Temperature-controlled sigmoid (T=0.5 for soft matching)
//! let soft = temperature_sigmoid(&v, 0.5);
//!
//! // Cosine similarity
//! let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
//! let b = Array1::from_vec(vec![0.707, 0.707, 0.0]);
//! let sim = cosine_similarity(&a, &b);
//! ```

use ndarray::{Array1, Array2, Array3, Axis, Zip, s, stack};
use rayon::prelude::*;

/// Threshold for switching to parallel operations.
/// Operations on vectors larger than this use Rayon parallelization.
pub const PARALLEL_THRESHOLD: usize = 1024;

// ============================================================================
// PARALLEL ELEMENT-WISE OPERATIONS
// ============================================================================

/// Elementwise sigmoid activation function: σ(x) = 1/(1+e^(-x))
///
/// Uses parallel iteration for vectors with more than PARALLEL_THRESHOLD elements.
/// Pushes elements toward 0 or 1, useful for binary-like outputs.
///
/// # Arguments
/// * `v` - Input vector
///
/// # Returns
/// Vector with sigmoid applied element-wise, values in range (0, 1)
///
/// # Example
/// ```ignore
/// let v = Array1::from_vec(vec![-2.0, 0.0, 2.0]);
/// let result = sigmoid(&v);
/// // result ≈ [0.119, 0.5, 0.881]
/// ```
pub fn sigmoid(v: &Array1<f32>) -> Array1<f32> {
    if v.len() > PARALLEL_THRESHOLD {
        let mut result = v.to_owned();
        result.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        result
    } else {
        v.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
}

/// Temperature-controlled sigmoid: σ(x,T) = 1/(1+e^(-x/T))
///
/// The temperature parameter controls the sharpness of the sigmoid:
/// - T → 0: Hard step function (deductive reasoning)
/// - T = 1: Standard sigmoid
/// - T → ∞: Uniform distribution (maximum entropy)
///
/// From Tensor Logic paper Section 4: Temperature interpolates between
/// deductive (T=0) and analogical (T>0) reasoning.
///
/// # Arguments
/// * `v` - Input vector
/// * `t` - Temperature parameter (must be > 0)
///
/// # Returns
/// Vector with temperature-scaled sigmoid applied element-wise
///
/// # Panics
/// Panics if temperature t <= 0
pub fn temperature_sigmoid(v: &Array1<f32>, t: f32) -> Array1<f32> {
    assert!(t > 0.0, "Temperature must be positive, got {}", t);

    if v.len() > PARALLEL_THRESHOLD {
        let mut result = v.to_owned();
        result.par_mapv_inplace(|x| 1.0 / (1.0 + (-x / t).exp()));
        result
    } else {
        v.mapv(|x| 1.0 / (1.0 + (-x / t).exp()))
    }
}

/// Majority voting for binary vectors.
///
/// Returns true if more than half of the elements are true.
/// Uses parallel counting for large vectors.
///
/// # Arguments
/// * `v` - Input boolean vector
///
/// # Returns
/// true if count(true) > len/2, false otherwise
///
/// # Example
/// ```ignore
/// let v = Array1::from_vec(vec![true, true, false, true]);
/// assert!(majority(&v)); // 3 > 2
/// ```
pub fn majority(v: &Array1<bool>) -> bool {
    if v.len() > PARALLEL_THRESHOLD {
        if let Some(slice) = v.as_slice() {
            let count: usize = slice.par_iter().filter(|&&b| b).count();
            count > v.len() / 2
        } else {
            // Fallback for non-contiguous arrays
            v.iter().filter(|&&b| b).count() > v.len() / 2
        }
    } else {
        v.iter().filter(|&&b| b).count() > v.len() / 2
    }
}

/// Heaviside step function: H(x) = 1 if x > 0, else 0
///
/// Converts continuous values to binary (Boolean) values.
/// Useful for hard thresholding in deductive reasoning (T=0 case).
///
/// # Arguments
/// * `v` - Input vector
///
/// # Returns
/// Boolean vector where each element is true if input > 0
pub fn heaviside(v: &Array1<f32>) -> Array1<bool> {
    if v.len() > PARALLEL_THRESHOLD {
        let slice = v.as_slice().expect("Vector must be contiguous");
        let result: Vec<bool> = slice.par_iter().map(|&x| x > 0.0).collect();
        Array1::from_vec(result)
    } else {
        v.mapv(|x| x > 0.0)
    }
}

/// Heaviside step function returning f32 instead of bool.
///
/// Returns 1.0 for positive values, 0.0 otherwise.
/// Useful when the output needs to remain a numeric tensor.
///
/// # Arguments
/// * `v` - Input vector
///
/// # Returns
/// Vector with 1.0 where input > 0, 0.0 otherwise
pub fn heaviside_f32(v: &Array1<f32>) -> Array1<f32> {
    if v.len() > PARALLEL_THRESHOLD {
        let mut result = v.to_owned();
        result.par_mapv_inplace(|x| if x > 0.0 { 1.0 } else { 0.0 });
        result
    } else {
        v.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

/// Softmax function with numerical stability.
///
/// Computes: softmax(v)[i] = e^(v[i] - max(v)) / Σ e^(v[j] - max(v))
///
/// The max subtraction prevents overflow for large input values.
/// Output is a probability distribution (sums to 1.0).
///
/// # Arguments
/// * `v` - Input vector
///
/// # Returns
/// Probability distribution vector (sums to 1.0)
pub fn softmax(v: &Array1<f32>) -> Array1<f32> {
    let max_v = v.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_v = v.mapv(|x| (x - max_v).exp());
    let sum: f32 = exp_v.sum();
    exp_v / sum
}

/// L2 normalization: v / ||v||
///
/// Normalizes vector to unit length (L2 norm = 1).
/// Uses BLAS-accelerated dot product for efficiency.
///
/// # Arguments
/// * `v` - Input vector
///
/// # Returns
/// Unit vector in same direction as input
///
/// # Panics
/// Panics if input vector has zero norm
pub fn l2_normalize(v: &Array1<f32>) -> Array1<f32> {
    let norm_sq = v.dot(v); // BLAS-accelerated
    assert!(norm_sq > 0.0, "Cannot normalize zero vector");
    let norm = norm_sq.sqrt();
    v / norm
}

/// Safe L2 normalization that returns zero vector for zero input.
///
/// # Arguments
/// * `v` - Input vector
///
/// # Returns
/// Unit vector, or zero vector if input is zero
pub fn l2_normalize_safe(v: &Array1<f32>) -> Array1<f32> {
    let norm_sq = v.dot(v);
    if norm_sq < f32::EPSILON {
        Array1::zeros(v.len())
    } else {
        v / norm_sq.sqrt()
    }
}

// ============================================================================
// BLAS-ACCELERATED SIMILARITY OPERATIONS
// ============================================================================

/// Cosine similarity using BLAS-accelerated dot product.
///
/// Computes: cos(θ) = (a · b) / (||a|| × ||b||)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity in range [-1, 1]
///
/// # Panics
/// Panics if either vector has zero norm
pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot = a.dot(b);           // BLAS-accelerated
    let norm_a = a.dot(a).sqrt(); // BLAS-accelerated
    let norm_b = b.dot(b).sqrt(); // BLAS-accelerated

    assert!(norm_a > 0.0 && norm_b > 0.0, "Cannot compute cosine similarity with zero vector");
    dot / (norm_a * norm_b)
}

/// Safe cosine similarity that returns 0.0 for zero vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity, or 0.0 if either vector is zero
pub fn cosine_similarity_safe(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot = a.dot(b);
    let norm_a_sq = a.dot(a);
    let norm_b_sq = b.dot(b);

    if norm_a_sq < f32::EPSILON || norm_b_sq < f32::EPSILON {
        0.0
    } else {
        dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt())
    }
}

/// Euclidean distance between two vectors.
///
/// Computes: ||a - b||_2
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Euclidean distance >= 0
pub fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    let diff = a - b;
    diff.dot(&diff).sqrt()
}

/// Dot product using BLAS.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Scalar dot product
pub fn dot_product(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    a.dot(b)
}

// ============================================================================
// MATRIX OPERATIONS (BLAS + Threading)
// ============================================================================

/// Gram matrix computation: Sim[i,j] = Emb[i] · Emb[j]
///
/// Computes pairwise dot products between all row vectors.
/// Uses BLAS + matrixmultiply-threading for optimal performance.
///
/// # Arguments
/// * `embeddings` - Matrix where each row is an embedding vector
///
/// # Returns
/// Symmetric similarity matrix
pub fn gram_matrix(embeddings: &Array2<f32>) -> Array2<f32> {
    embeddings.dot(&embeddings.t()) // BLAS + matrixmultiply-threading
}

/// Normalized Gram matrix (cosine similarity matrix).
///
/// Each entry is the cosine similarity between corresponding row vectors.
///
/// # Arguments
/// * `embeddings` - Matrix where each row is an embedding vector
///
/// # Returns
/// Symmetric cosine similarity matrix with diagonal entries = 1
pub fn cosine_similarity_matrix(embeddings: &Array2<f32>) -> Array2<f32> {
    let norms: Vec<f32> = embeddings
        .rows()
        .into_iter()
        .map(|row| row.dot(&row).sqrt())
        .collect();

    let gram = gram_matrix(embeddings);
    let n = gram.nrows();

    let mut result = gram;
    for i in 0..n {
        for j in 0..n {
            if norms[i] > f32::EPSILON && norms[j] > f32::EPSILON {
                result[[i, j]] /= norms[i] * norms[j];
            } else {
                result[[i, j]] = 0.0;
            }
        }
    }
    result
}

// ============================================================================
// TENSOR LOGIC OPERATIONS (from Tensor Logic paper)
// ============================================================================

/// Embedding superposition: S[d] = Σ_x V[x] * Emb[x, d]
///
/// Creates a superposition of embedding vectors weighted by values.
/// This is the "write" operation for embeddings.
///
/// From Tensor Logic: This encodes a weighted set of items into a single vector.
///
/// # Arguments
/// * `values` - Weight for each embedding (length n)
/// * `embeddings` - Embedding matrix (n × d)
///
/// # Returns
/// Superposition vector of dimension d
pub fn superposition(values: &Array1<f32>, embeddings: &Array2<f32>) -> Array1<f32> {
    assert_eq!(values.len(), embeddings.nrows(),
        "Values length must match number of embeddings");
    values.dot(embeddings) // BLAS vector-matrix multiplication
}

/// Embedding retrieval: D[A] = Σ_d S[d] * Emb[A, d]
///
/// Retrieves activation for each embedding from a superposition vector.
/// This is the "read" operation for embeddings.
///
/// From Tensor Logic: This decodes the superposition back to item activations.
///
/// # Arguments
/// * `superposition_vec` - Superposition vector (length d)
/// * `embeddings` - Embedding matrix (n × d)
///
/// # Returns
/// Activation vector for each embedding (length n)
pub fn retrieval(superposition_vec: &Array1<f32>, embeddings: &Array2<f32>) -> Array1<f32> {
    assert_eq!(superposition_vec.len(), embeddings.ncols(),
        "Superposition dimension must match embedding dimension");
    embeddings.dot(superposition_vec) // BLAS matrix-vector multiplication
}

/// Find k most similar embeddings to a query.
///
/// # Arguments
/// * `query` - Query embedding vector
/// * `embeddings` - Matrix of candidate embeddings
/// * `k` - Number of results to return
///
/// # Returns
/// Vector of (index, similarity) pairs, sorted by descending similarity
pub fn top_k_similar(query: &Array1<f32>, embeddings: &Array2<f32>, k: usize) -> Vec<(usize, f32)> {
    let query_norm = query.dot(query).sqrt();
    if query_norm < f32::EPSILON {
        return vec![];
    }

    let mut similarities: Vec<(usize, f32)> = embeddings
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            let row_norm = row.dot(&row).sqrt();
            let sim = if row_norm > f32::EPSILON {
                row.dot(query) / (row_norm * query_norm)
            } else {
                0.0
            };
            (i, sim)
        })
        .collect();

    // Sort by descending similarity
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    similarities.truncate(k);
    similarities
}

// ============================================================================
// EINSTEIN SUMMATION (EINSUM)
// ============================================================================

/// Einstein summation for 2D tensor operations.
///
/// Supports common tensor contraction patterns:
/// - `"ij,jk->ik"`: Matrix multiplication A @ B
/// - `"ij,kj->ik"`: A @ B.T
/// - `"ji,jk->ik"`: A.T @ B
/// - `"ij,ik->jk"`: Outer product-like contraction
/// - `"ik,jk->ij"`: Another outer contraction
///
/// All operations use BLAS + threaded matrix multiplication.
///
/// # Arguments
/// * `spec` - Einstein summation specification string
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
/// Result of tensor contraction
///
/// # Panics
/// Panics for unsupported einsum specifications
pub fn einsum_2d(spec: &str, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    match spec {
        "ij,jk->ik" => a.dot(b),         // C = A @ B
        "ij,kj->ik" => a.dot(&b.t()),    // C = A @ B.T
        "ji,jk->ik" => a.t().dot(b),     // C = A.T @ B
        "ij,ik->jk" => a.t().dot(b),     // Transpose first, contract on i
        "ik,jk->ij" => a.dot(&b.t()),    // Contract on k
        _ => panic!("Unsupported einsum specification: {}. Supported: ij,jk->ik, ij,kj->ik, ji,jk->ik, ij,ik->jk, ik,jk->ij", spec),
    }
}

/// Vector-matrix einsum operations.
///
/// Supports:
/// - `"i,ij->j"`: Vector-matrix product (v @ M)
/// - `"i,ji->j"`: Vector-transposed matrix product (v @ M.T)
///
/// # Arguments
/// * `spec` - Einstein summation specification
/// * `v` - Vector operand
/// * `m` - Matrix operand
///
/// # Returns
/// Result vector
pub fn einsum_vm(spec: &str, v: &Array1<f32>, m: &Array2<f32>) -> Array1<f32> {
    match spec {
        "i,ij->j" => v.dot(m),
        "i,ji->j" => v.dot(&m.t()),
        _ => panic!("Unsupported vector-matrix einsum: {}. Supported: i,ij->j, i,ji->j", spec),
    }
}

/// Matrix-vector einsum operations.
///
/// Supports:
/// - `"ij,j->i"`: Matrix-vector product (M @ v)
/// - `"ji,j->i"`: Transposed matrix-vector product (M.T @ v)
///
/// # Arguments
/// * `spec` - Einstein summation specification
/// * `m` - Matrix operand
/// * `v` - Vector operand
///
/// # Returns
/// Result vector
pub fn einsum_mv(spec: &str, m: &Array2<f32>, v: &Array1<f32>) -> Array1<f32> {
    match spec {
        "ij,j->i" => m.dot(v),
        "ji,j->i" => m.t().dot(v),
        _ => panic!("Unsupported matrix-vector einsum: {}. Supported: ij,j->i, ji,j->i", spec),
    }
}

// ============================================================================
// BATCH OPERATIONS (3D Tensors with Parallel Processing)
// ============================================================================

/// Batch matrix multiplication for 3D tensors.
///
/// Computes C[b] = A[b] @ B[b] for each batch index b.
/// Uses Rayon parallelization over the batch dimension.
///
/// # Arguments
/// * `a` - First 3D tensor (batch × m × k)
/// * `b` - Second 3D tensor (batch × k × n)
///
/// # Returns
/// Result 3D tensor (batch × m × n)
///
/// # Panics
/// Panics if batch dimensions don't match
pub fn batch_matmul(a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
    assert_eq!(a.shape()[0], b.shape()[0], "Batch dimensions must match");
    assert_eq!(a.shape()[2], b.shape()[1], "Inner dimensions must match for matmul");

    let batch_size = a.shape()[0];

    let results: Vec<Array2<f32>> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let a_slice = a.slice(s![i, .., ..]).to_owned();
            let b_slice = b.slice(s![i, .., ..]).to_owned();
            a_slice.dot(&b_slice)
        })
        .collect();

    // Stack results into 3D array
    stack(
        Axis(0),
        &results.iter().map(|x| x.view()).collect::<Vec<_>>()
    ).expect("Failed to stack batch results")
}

/// Batch cosine similarity.
///
/// Computes cosine similarity for each pair of vectors in parallel.
/// Uses broadcasting for efficient norm product computation.
///
/// # Arguments
/// * `queries` - Matrix of query vectors (n × d)
/// * `references` - Matrix of reference vectors (m × d)
///
/// # Returns
/// Similarity matrix (n × m)
pub fn batch_cosine_similarity(queries: &Array2<f32>, references: &Array2<f32>) -> Array2<f32> {
    // Compute norms using ndarray operations (BLAS-accelerated row dot products)
    let q_norms: Array1<f32> = queries.rows().into_iter()
        .map(|r| r.dot(&r).sqrt())
        .collect();
    let r_norms: Array1<f32> = references.rows().into_iter()
        .map(|r| r.dot(&r).sqrt())
        .collect();

    // Compute dot products (BLAS matmul)
    let mut result = queries.dot(&references.t());

    // Broadcasting: compute norm product matrix (n × m) without nested loops
    // q_norms.insert_axis(Axis(1)) -> (n, 1)
    // r_norms.insert_axis(Axis(0)) -> (1, m)
    // Their product broadcasts to (n, m)
    let q_norm_col = q_norms.view().insert_axis(Axis(1));
    let r_norm_row = r_norms.view().insert_axis(Axis(0));
    let norm_product = &q_norm_col * &r_norm_row;

    // Normalize using Zip for efficient element-wise operation with epsilon check
    Zip::from(&mut result)
        .and(&norm_product)
        .for_each(|r, &n| {
            if n > f32::EPSILON {
                *r /= n;
            } else {
                *r = 0.0;
            }
        });

    result
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Convert Vec<f32> to ndarray Array1<f32>.
#[inline]
pub fn vec_to_array1(v: Vec<f32>) -> Array1<f32> {
    Array1::from_vec(v)
}

/// Convert slice to ndarray Array1<f32>.
#[inline]
pub fn slice_to_array1(s: &[f32]) -> Array1<f32> {
    Array1::from_vec(s.to_vec())
}

/// Convert ndarray Array1<f32> to Vec<f32>.
#[inline]
pub fn array1_to_vec(a: Array1<f32>) -> Vec<f32> {
    a.to_vec()
}

/// Create 2D array from row vectors.
pub fn rows_to_array2(rows: &[Array1<f32>]) -> Array2<f32> {
    if rows.is_empty() {
        return Array2::zeros((0, 0));
    }

    let n = rows.len();
    let d = rows[0].len();

    let mut result = Array2::zeros((n, d));
    for (i, row) in rows.iter().enumerate() {
        result.row_mut(i).assign(row);
    }
    result
}

// ============================================================================
// HYPERVECTOR OPERATIONS (High-Dimensional Computing)
// ============================================================================

/// XOR binding for hypervectors.
///
/// In high-dimensional computing, binding creates a new hypervector that
/// represents the association of two concepts. XOR is used because:
/// - It's self-inverse: bind(bind(a, b), b) = a
/// - It preserves similarity: bind(a, c) is similar to bind(b, c) if a ≈ b
/// - It distributes over bundling
///
/// # Arguments
/// * `a` - First binary hypervector (0s and 1s as i64)
/// * `b` - Second binary hypervector (same dimensions as a)
///
/// # Returns
/// Element-wise XOR of the two vectors
///
/// # Panics
/// Panics if vectors have different lengths
pub fn bind(a: &[i64], b: &[i64]) -> Vec<i64> {
    assert_eq!(a.len(), b.len(), "Vectors must have same length for binding");

    if a.len() > PARALLEL_THRESHOLD {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| x ^ y)
            .collect()
    } else {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x ^ y)
            .collect()
    }
}

/// Majority-based bundling for hypervectors.
///
/// Bundling creates a superposition of multiple hypervectors by taking the
/// element-wise majority. This represents a "set" or "bag" of concepts.
///
/// # Arguments
/// * `vectors` - Collection of binary hypervectors to bundle
///
/// # Returns
/// Element-wise majority of all vectors (1 if more than half are 1, else 0)
/// For ties, rounds toward 1.
///
/// # Panics
/// Panics if vectors is empty or vectors have different lengths
pub fn bundle(vectors: &[&[i64]]) -> Vec<i64> {
    assert!(!vectors.is_empty(), "Bundle requires at least one vector");
    let len = vectors[0].len();
    for v in vectors.iter() {
        assert_eq!(v.len(), len, "All vectors must have same length for bundling");
    }

    let threshold = (vectors.len() + 1) / 2; // Rounds up for ties

    if len > PARALLEL_THRESHOLD {
        (0..len).into_par_iter()
            .map(|i| {
                let count: usize = vectors.iter().map(|v| v[i] as usize).sum();
                if count >= threshold { 1 } else { 0 }
            })
            .collect()
    } else {
        (0..len)
            .map(|i| {
                let count: usize = vectors.iter().map(|v| v[i] as usize).sum();
                if count >= threshold { 1 } else { 0 }
            })
            .collect()
    }
}

/// Circular shift (permutation) for hypervectors.
///
/// Permutation creates a new representation that is dissimilar to the original
/// but can be reversed. In high-dimensional computing, this is used to represent
/// sequence or order information.
///
/// # Arguments
/// * `v` - Binary hypervector to permute
/// * `shift` - Number of positions to shift (positive = right, negative = left)
///
/// # Returns
/// Circularly shifted vector
pub fn permute(v: &[i64], shift: i64) -> Vec<i64> {
    if v.is_empty() {
        return Vec::new();
    }

    let len = v.len() as i64;
    // Normalize shift to positive value in range [0, len)
    let effective_shift = ((shift % len) + len) % len;

    if effective_shift == 0 {
        return v.to_vec();
    }

    let shift_usize = effective_shift as usize;
    let mut result = vec![0i64; v.len()];

    // Copy elements with circular shift
    for (i, &x) in v.iter().enumerate() {
        let new_pos = (i + shift_usize) % v.len();
        result[new_pos] = x;
    }

    result
}

/// Hamming similarity for binary hypervectors.
///
/// Returns the percentage of matching bits (0-100 scale for integer output).
/// This is the complement of Hamming distance normalized by vector length.
///
/// # Arguments
/// * `a` - First binary hypervector
/// * `b` - Second binary hypervector
///
/// # Returns
/// Similarity as percentage (0-100): 100 means identical, 0 means completely opposite
///
/// # Panics
/// Panics if vectors have different lengths or are empty
pub fn hamming_similarity(a: &[i64], b: &[i64]) -> i64 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length for similarity");
    assert!(!a.is_empty(), "Vectors must not be empty");

    let matches = if a.len() > PARALLEL_THRESHOLD {
        a.par_iter()
            .zip(b.par_iter())
            .filter(|(&x, &y)| x == y)
            .count()
    } else {
        a.iter()
            .zip(b.iter())
            .filter(|(&x, &y)| x == y)
            .count()
    };

    // Return as percentage (0-100)
    ((matches * 100) / a.len()) as i64
}

/// Inverse permutation (undo circular shift).
///
/// This is equivalent to permute(v, -shift).
///
/// # Arguments
/// * `v` - Binary hypervector to unpermute
/// * `shift` - The original shift amount used to permute
///
/// # Returns
/// The original vector before permutation
#[inline]
pub fn unpermute(v: &[i64], shift: i64) -> Vec<i64> {
    permute(v, -shift)
}

/// Unbind operation (inverse of bind).
///
/// Since XOR is self-inverse, unbind is the same as bind.
///
/// # Arguments
/// * `bound` - The bound hypervector
/// * `key` - The key used in original binding
///
/// # Returns
/// The original value before binding
#[inline]
pub fn unbind(bound: &[i64], key: &[i64]) -> Vec<i64> {
    bind(bound, key)
}

/// Resonance (cleanup) operation for noisy hypervectors.
///
/// Given a noisy query hypervector, finds the most similar clean hypervector
/// from a known set (codebook).
///
/// # Arguments
/// * `query` - Noisy hypervector to clean up
/// * `codebook` - Set of known clean hypervectors
///
/// # Returns
/// Index of most similar hypervector in codebook
///
/// # Panics
/// Panics if codebook is empty
pub fn resonance(query: &[i64], codebook: &[&[i64]]) -> usize {
    assert!(!codebook.is_empty(), "Codebook must not be empty");

    codebook.iter()
        .enumerate()
        .map(|(i, v)| (i, hamming_similarity(query, v)))
        .max_by_key(|(_, sim)| *sim)
        .map(|(i, _)| i)
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sigmoid() {
        let v = array![-2.0, 0.0, 2.0];
        let result = sigmoid(&v);
        assert!((result[0] - 0.119).abs() < 0.01);
        assert!((result[1] - 0.5).abs() < 0.001);
        assert!((result[2] - 0.881).abs() < 0.01);
    }

    #[test]
    fn test_temperature_sigmoid() {
        let v = array![0.0, 1.0, 2.0];

        // Standard sigmoid (T=1)
        let t1 = temperature_sigmoid(&v, 1.0);
        let s = sigmoid(&v);
        for i in 0..3 {
            assert!((t1[i] - s[i]).abs() < 0.001);
        }

        // Lower temperature makes sigmoid sharper
        let t_low = temperature_sigmoid(&array![1.0], 0.1);
        let t_high = temperature_sigmoid(&array![1.0], 10.0);
        assert!(t_low[0] > t_high[0]); // Sharper sigmoid saturates faster
    }

    #[test]
    fn test_majority() {
        assert!(majority(&array![true, true, true]));
        assert!(majority(&array![true, true, false]));
        assert!(!majority(&array![true, false, false]));
        assert!(!majority(&array![false, false, false]));
    }

    #[test]
    fn test_heaviside() {
        let v = array![-1.0, 0.0, 1.0, 0.001];
        let result = heaviside(&v);
        assert_eq!(result, array![false, false, true, true]);
    }

    #[test]
    fn test_softmax() {
        let v = array![1.0, 2.0, 3.0];
        let result = softmax(&v);

        // Sum should be 1.0
        assert!((result.sum() - 1.0).abs() < 0.001);

        // Values should be in increasing order
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_l2_normalize() {
        let v = array![3.0, 4.0];
        let result = l2_normalize(&v);

        // Should have unit norm
        let norm = result.dot(&result).sqrt();
        assert!((norm - 1.0).abs() < 0.001);

        // Direction should be preserved
        assert!((result[0] - 0.6).abs() < 0.001);
        assert!((result[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        let c = array![1.0, 0.0];
        let d = array![-1.0, 0.0];

        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 0.001);   // Same direction
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);   // Orthogonal
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001); // Opposite
    }

    #[test]
    fn test_gram_matrix() {
        let embeddings = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let gram = gram_matrix(&embeddings);

        assert_eq!(gram.shape(), &[3, 3]);
        assert!((gram[[0, 0]] - 1.0).abs() < 0.001); // ||e1||^2 = 1
        assert!((gram[[0, 1]] - 0.0).abs() < 0.001); // e1 · e2 = 0
        assert!((gram[[2, 2]] - 2.0).abs() < 0.001); // ||e3||^2 = 2
    }

    #[test]
    fn test_superposition_retrieval() {
        // Create embeddings for items A, B
        let embeddings = array![[1.0, 0.0], [0.0, 1.0]];

        // Create superposition with weights [0.5, 0.5]
        let values = array![0.5, 0.5];
        let sup = superposition(&values, &embeddings);
        assert!((sup[0] - 0.5).abs() < 0.001);
        assert!((sup[1] - 0.5).abs() < 0.001);

        // Retrieve should recover weights (for orthogonal embeddings)
        let retrieved = retrieval(&sup, &embeddings);
        assert!((retrieved[0] - 0.5).abs() < 0.001);
        assert!((retrieved[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_einsum_2d() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        // Standard matrix multiplication
        let c = einsum_2d("ij,jk->ik", &a, &b);
        assert!((c[[0, 0]] - 19.0).abs() < 0.001);
        assert!((c[[0, 1]] - 22.0).abs() < 0.001);
        assert!((c[[1, 0]] - 43.0).abs() < 0.001);
        assert!((c[[1, 1]] - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_batch_matmul() {
        let a = Array3::from_shape_vec((2, 2, 2), vec![
            1.0, 2.0, 3.0, 4.0,  // Batch 0
            5.0, 6.0, 7.0, 8.0,  // Batch 1
        ]).unwrap();

        let b = Array3::from_shape_vec((2, 2, 2), vec![
            1.0, 0.0, 0.0, 1.0,  // Identity for batch 0
            2.0, 0.0, 0.0, 2.0,  // 2*Identity for batch 1
        ]).unwrap();

        let c = batch_matmul(&a, &b);

        assert_eq!(c.shape(), &[2, 2, 2]);
        // Batch 0: A * I = A
        assert!((c[[0, 0, 0]] - 1.0).abs() < 0.001);
        // Batch 1: A * 2I = 2A
        assert!((c[[1, 0, 0]] - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_top_k_similar() {
        let query = array![1.0, 0.0];
        let embeddings = array![
            [1.0, 0.0],   // Same as query
            [0.0, 1.0],   // Orthogonal
            [0.707, 0.707], // 45 degrees
        ];

        let results = top_k_similar(&query, &embeddings, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Most similar is same vector
        assert!((results[0].1 - 1.0).abs() < 0.01);
    }

    // =========================================================================
    // HYPERVECTOR TESTS
    // =========================================================================

    #[test]
    fn test_bind() {
        let a = vec![1, 0, 1, 1, 0];
        let b = vec![0, 1, 1, 0, 1];
        let result = bind(&a, &b);
        assert_eq!(result, vec![1, 1, 0, 1, 1]); // XOR
    }

    #[test]
    fn test_bind_self_inverse() {
        // bind(bind(a, b), b) = a (XOR is self-inverse)
        let a = vec![1, 0, 1, 1, 0, 0, 1, 0];
        let b = vec![0, 1, 1, 0, 1, 1, 0, 0];
        let bound = bind(&a, &b);
        let unbound = bind(&bound, &b);
        assert_eq!(unbound, a);
    }

    #[test]
    fn test_unbind() {
        let a = vec![1, 0, 1, 1, 0];
        let b = vec![0, 1, 1, 0, 1];
        let bound = bind(&a, &b);
        let recovered = unbind(&bound, &b);
        assert_eq!(recovered, a);
    }

    #[test]
    fn test_bundle_majority() {
        let v1 = vec![1, 0, 1];
        let v2 = vec![0, 1, 1];
        let v3 = vec![1, 1, 0];
        let result = bundle(&[&v1[..], &v2[..], &v3[..]]);
        // Counts: [2, 2, 2] -> threshold = 2 -> all become 1
        assert_eq!(result, vec![1, 1, 1]);
    }

    #[test]
    fn test_bundle_tie_breaking() {
        // With 2 vectors, threshold = (2+1)/2 = 1 (rounds up for ties)
        let v1 = vec![1, 0];
        let v2 = vec![0, 1];
        let result = bundle(&[&v1[..], &v2[..]]);
        // Counts: [1, 1] >= threshold 1 -> both become 1
        assert_eq!(result, vec![1, 1]);
    }

    #[test]
    fn test_bundle_single() {
        let v1 = vec![1, 0, 1];
        let result = bundle(&[&v1[..]]);
        assert_eq!(result, v1);
    }

    #[test]
    fn test_permute_right() {
        let v = vec![1, 2, 3, 4, 5];
        let result = permute(&v, 2);
        // Shift right by 2: [1,2,3,4,5] -> [4,5,1,2,3]
        assert_eq!(result, vec![4, 5, 1, 2, 3]);
    }

    #[test]
    fn test_permute_left() {
        let v = vec![1, 2, 3, 4, 5];
        let result = permute(&v, -2);
        // Shift left by 2: [1,2,3,4,5] -> [3,4,5,1,2]
        assert_eq!(result, vec![3, 4, 5, 1, 2]);
    }

    #[test]
    fn test_permute_zero() {
        let v = vec![1, 2, 3, 4, 5];
        let result = permute(&v, 0);
        assert_eq!(result, v);
    }

    #[test]
    fn test_permute_full_rotation() {
        let v = vec![1, 2, 3, 4, 5];
        let result = permute(&v, 5);
        assert_eq!(result, v); // Full rotation returns to original
    }

    #[test]
    fn test_unpermute() {
        let v = vec![1, 2, 3, 4, 5];
        let permuted = permute(&v, 2);
        let recovered = unpermute(&permuted, 2);
        assert_eq!(recovered, v);
    }

    #[test]
    fn test_hamming_similarity_identical() {
        let a = vec![1, 0, 1, 1, 0];
        let b = vec![1, 0, 1, 1, 0];
        let sim = hamming_similarity(&a, &b);
        assert_eq!(sim, 100); // 100% match
    }

    #[test]
    fn test_hamming_similarity_opposite() {
        let a = vec![1, 1, 1, 1, 1];
        let b = vec![0, 0, 0, 0, 0];
        let sim = hamming_similarity(&a, &b);
        assert_eq!(sim, 0); // 0% match
    }

    #[test]
    fn test_hamming_similarity_partial() {
        let a = vec![1, 0, 1, 1, 0];
        let b = vec![1, 1, 1, 0, 0];
        let sim = hamming_similarity(&a, &b);
        // 3 out of 5 match: 1@0, 1@2, 0@4
        assert_eq!(sim, 60); // 60%
    }

    #[test]
    fn test_resonance() {
        let codebook: Vec<Vec<i64>> = vec![
            vec![1, 0, 0, 0, 0],
            vec![0, 1, 0, 0, 0],
            vec![0, 0, 1, 0, 0],
        ];
        let codebook_refs: Vec<&[i64]> = codebook.iter().map(|v| v.as_slice()).collect();

        // Query that's close to first codebook entry
        let query = vec![1, 0, 0, 0, 1];
        let idx = resonance(&query, &codebook_refs);
        assert_eq!(idx, 0); // Most similar to first entry (80% match vs 40% for others)
    }

    #[test]
    fn test_resonance_exact_match() {
        let codebook: Vec<Vec<i64>> = vec![
            vec![1, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
        ];
        let codebook_refs: Vec<&[i64]> = codebook.iter().map(|v| v.as_slice()).collect();

        // Exact match for second entry
        let query = vec![0, 1, 0];
        let idx = resonance(&query, &codebook_refs);
        assert_eq!(idx, 1);
    }
}
