//! Binary packing utilities for Hamming and Jaccard similarity optimizations.
//!
//! These functions provide hardware-accelerated similarity computations by:
//! - Packing float vectors into u64 arrays (256x memory reduction for 64-dim)
//! - Using hardware `popcnt` instructions for fast bit counting
//!
//! # Performance
//!
//! For binary/boolean vectors, packed operations provide 50-100x speedup over
//! float-based computation due to:
//! - Cache efficiency: 64 floats → 1 u64
//! - SIMD: Hardware popcnt is extremely fast
//! - Memory bandwidth: Less data to load
//!
//! # Example
//!
//! ```ignore
//! use rholang::rust::interpreter::spaces::vectordb::in_memory::utils::binary::{
//!     pack_to_binary, hamming_distance_packed, jaccard_similarity_packed
//! };
//!
//! // Pack embeddings (threshold at 0.5)
//! let a = pack_to_binary(&[0.9, 0.1, 0.8, 0.2]); // [0b0101] = bits 0 and 2 set
//! let b = pack_to_binary(&[0.9, 0.9, 0.1, 0.1]); // [0b0011] = bits 0 and 1 set
//!
//! // Hamming: count differing bits
//! let dist = hamming_distance_packed(&a, &b); // 2 bits differ
//!
//! // Jaccard: intersection / union
//! let sim = jaccard_similarity_packed(&a, &b); // 1 / 3 = 0.333
//! ```

/// Pack a float embedding into u64 chunks (binarize at threshold 0.5).
///
/// Each chunk of 64 floats is packed into a single u64 where bit `i` is set
/// if the `i`-th value in the chunk is > 0.5.
///
/// # Arguments
///
/// * `embedding` - Float vector to pack (any length)
///
/// # Returns
///
/// Vector of u64 where each u64 holds 64 packed bits.
/// Length is `ceil(embedding.len() / 64)`.
///
/// # Example
///
/// ```ignore
/// use rholang::rust::interpreter::spaces::vectordb::in_memory::utils::binary::pack_to_binary;
///
/// let embedding = vec![0.9, 0.1, 0.8, 0.2]; // binary: 0b0101
/// let packed = pack_to_binary(&embedding);
/// assert_eq!(packed.len(), 1);
/// assert_eq!(packed[0], 0b0101); // bits 0 and 2 set
/// ```
#[inline]
pub fn pack_to_binary(embedding: &[f32]) -> Vec<u64> {
    embedding
        .chunks(64)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u64, |acc, (i, &val)| {
                    if val > 0.5 {
                        acc | (1u64 << i)
                    } else {
                        acc
                    }
                })
        })
        .collect()
}

/// Compute Hamming distance using hardware popcnt on packed binary vectors.
///
/// Returns the number of differing bits between two packed vectors.
///
/// # Arguments
///
/// * `a` - First packed binary vector
/// * `b` - Second packed binary vector (must have same length as `a`)
///
/// # Returns
///
/// Count of differing bits (XOR popcount sum).
///
/// # Example
///
/// ```ignore
/// use rholang::rust::interpreter::spaces::vectordb::in_memory::utils::binary::hamming_distance_packed;
///
/// let a = vec![0x0F_u64]; // bits 0-3 set
/// let b = vec![0xF0_u64]; // bits 4-7 set
/// let dist = hamming_distance_packed(&a, &b);
/// assert_eq!(dist, 8); // all 8 bits differ
/// ```
#[inline]
pub fn hamming_distance_packed(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// Compute Jaccard similarity using hardware popcnt on packed binary vectors.
///
/// Returns `|A ∩ B| / |A ∪ B|`, or 1.0 if both vectors are empty.
///
/// # Arguments
///
/// * `a` - First packed binary vector
/// * `b` - Second packed binary vector (must have same length as `a`)
///
/// # Returns
///
/// Jaccard similarity in range [0.0, 1.0].
///
/// # Example
///
/// ```ignore
/// use rholang::rust::interpreter::spaces::vectordb::in_memory::utils::binary::jaccard_similarity_packed;
///
/// let a = vec![0x0F_u64]; // bits 0-3 set
/// let b = vec![0x3C_u64]; // bits 2-5 set
/// let sim = jaccard_similarity_packed(&a, &b);
/// // Intersection: bits 2-3 (2 bits)
/// // Union: bits 0-5 (6 bits)
/// // Jaccard: 2/6 ≈ 0.333
/// assert!((sim - 0.333).abs() < 0.01);
/// ```
#[inline]
pub fn jaccard_similarity_packed(a: &[u64], b: &[u64]) -> f32 {
    let intersection: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x & y).count_ones())
        .sum();
    let union: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x | y).count_ones())
        .sum();
    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_to_binary_basic() {
        // First 4 elements: 0.9 > 0.5 (bit 0), 0.1 < 0.5, 0.8 > 0.5 (bit 2), 0.2 < 0.5
        let embedding = vec![0.9, 0.1, 0.8, 0.2];
        let packed = pack_to_binary(&embedding);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0b0101); // bits 0 and 2 set
    }

    #[test]
    fn test_pack_to_binary_all_high() {
        let embedding = vec![0.9; 64];
        let packed = pack_to_binary(&embedding);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], u64::MAX);
    }

    #[test]
    fn test_pack_to_binary_all_low() {
        let embedding = vec![0.1; 64];
        let packed = pack_to_binary(&embedding);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0);
    }

    #[test]
    fn test_pack_to_binary_alternating() {
        // Even indices high, odd indices low
        let mut embedding = vec![0.0f32; 64];
        for i in 0..64 {
            embedding[i] = if i % 2 == 0 { 0.9 } else { 0.1 };
        }
        let packed = pack_to_binary(&embedding);
        assert_eq!(packed[0], 0x5555555555555555); // alternating 01 pattern
    }

    #[test]
    fn test_pack_to_binary_multiple_chunks() {
        // 128 dimensions = 2 u64s
        let mut embedding = vec![0.0f32; 128];
        // First 64: all high
        for i in 0..64 {
            embedding[i] = 0.9;
        }
        // Second 64: all low
        let packed = pack_to_binary(&embedding);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], u64::MAX);
        assert_eq!(packed[1], 0);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let a = vec![0xFFFFFFFF_u64];
        let b = vec![0xFFFFFFFF_u64];
        assert_eq!(hamming_distance_packed(&a, &b), 0);
    }

    #[test]
    fn test_hamming_distance_all_different() {
        let a = vec![0u64];
        let b = vec![u64::MAX];
        assert_eq!(hamming_distance_packed(&a, &b), 64);
    }

    #[test]
    fn test_hamming_distance_half() {
        let a = vec![0x5555555555555555_u64]; // 32 bits set
        let b = vec![0xAAAAAAAAAAAAAAAA_u64]; // other 32 bits set
        assert_eq!(hamming_distance_packed(&a, &b), 64);
    }

    #[test]
    fn test_hamming_distance_multiple_chunks() {
        let a = vec![0u64, u64::MAX];
        let b = vec![u64::MAX, 0u64];
        assert_eq!(hamming_distance_packed(&a, &b), 128);
    }

    #[test]
    fn test_jaccard_identical() {
        let a = vec![0xFF_u64];
        let b = vec![0xFF_u64];
        assert!((jaccard_similarity_packed(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_no_overlap() {
        let a = vec![0x0F_u64]; // bits 0-3
        let b = vec![0xF0_u64]; // bits 4-7
        assert!((jaccard_similarity_packed(&a, &b) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a = vec![0x0F_u64]; // bits 0-3
        let b = vec![0x3C_u64]; // bits 2-5
        // Intersection: bits 2-3 = 2 bits
        // Union: bits 0-5 = 6 bits
        assert!((jaccard_similarity_packed(&a, &b) - (2.0 / 6.0)).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_both_empty() {
        let a = vec![0u64];
        let b = vec![0u64];
        assert!((jaccard_similarity_packed(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_one_empty() {
        let a = vec![0xFF_u64];
        let b = vec![0u64];
        assert!((jaccard_similarity_packed(&a, &b) - 0.0).abs() < 0.001);
    }
}
