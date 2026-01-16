//! Handler types for VectorDB similarity and ranking operations.
//!
//! This module provides core types used by similarity metric handlers and
//! ranking function handlers.

use super::super::metrics::EmbeddingType;
use ndarray::Array1;

// ==========================================================================
// Index Optimization Data
// ==========================================================================

/// Index optimization data provided to similarity handlers.
///
/// This struct carries pre-computed index data that handlers can use to
/// accelerate similarity computations. The data is populated by the backend
/// based on the configured `IndexConfig`.
///
/// # Optimization Strategies
///
/// - **Pre-normalization** (`is_pre_normalized`): When embeddings are stored
///   L2-normalized, cosine similarity becomes a simple dot product.
///
/// - **Norm caching** (`squared_norms`): Cached squared L2 norms enable
///   Euclidean distance via the identity: `||a-b||² = ||a||² + ||b||² - 2(a·b)`,
///   allowing BLAS-accelerated computation.
///
/// - **Binary packing** (`packed_binary`): Pre-binarized embeddings packed
///   into u64 enable hardware-accelerated Hamming/Jaccard via popcnt.
#[derive(Clone, Debug, Default)]
pub struct IndexOptimizationData {
    /// Whether embeddings are already L2-normalized.
    ///
    /// When `true`, cosine similarity can be computed as a simple dot product:
    /// `cos(a,b) = a' · b'` where `a'` and `b'` are L2-normalized.
    pub is_pre_normalized: bool,

    /// Cached squared L2 norms for each stored embedding.
    ///
    /// Enables Euclidean distance optimization using the identity:
    /// `||a-b||² = ||a||² + ||b||² - 2(a·b)`
    ///
    /// This avoids per-row subtraction and enables BLAS-accelerated dot products.
    /// Index `i` corresponds to row `i` in the embeddings matrix.
    pub squared_norms: Option<Vec<f32>>,

    /// Packed binary embeddings for Hamming/Jaccard optimization.
    ///
    /// Each embedding is binarized (threshold 0.5) and packed into `Vec<u64>`.
    /// This enables hardware-accelerated distance computation via popcnt:
    /// - Hamming: `popcnt(a XOR b) / bits`
    /// - Jaccard: `popcnt(a AND b) / popcnt(a OR b)`
    ///
    /// Memory reduction: 64 f32 dimensions → 1 u64 (256x compression).
    pub packed_binary: Option<Vec<Vec<u64>>>,

    /// Number of dimensions (bits) in packed binary embeddings.
    ///
    /// Used for proper normalization when computing Hamming distance.
    pub packed_bits: usize,
}

impl IndexOptimizationData {
    /// Create optimization data for pre-normalized embeddings.
    pub fn pre_normalized() -> Self {
        Self {
            is_pre_normalized: true,
            ..Default::default()
        }
    }

    /// Create optimization data with squared L2 norms.
    pub fn with_squared_norms(squared_norms: Vec<f32>) -> Self {
        Self {
            squared_norms: Some(squared_norms),
            ..Default::default()
        }
    }

    /// Create optimization data with packed binary embeddings.
    pub fn with_packed_binary(packed_binary: Vec<Vec<u64>>, packed_bits: usize) -> Self {
        Self {
            packed_binary: Some(packed_binary),
            packed_bits,
            ..Default::default()
        }
    }

    /// Check if any optimization data is available.
    pub fn has_optimizations(&self) -> bool {
        self.is_pre_normalized || self.squared_norms.is_some() || self.packed_binary.is_some()
    }
}

// ==========================================================================
// Resolved Arguments
// ==========================================================================

/// Resolved argument from Rholang expression.
///
/// Arguments to sim/rank modifiers are parsed from Par expressions into
/// concrete values that handlers can use for computation.
#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedArg {
    /// String argument (e.g., metric identifier "cos", "euclidean")
    String(String),
    /// Integer argument (e.g., top-k value)
    Integer(i64),
    /// Float argument (e.g., threshold 0.8)
    Float(f64),
    /// Boolean argument
    Boolean(bool),
    /// Nil/null value (Rholang's Nil)
    Nil,
}

impl ResolvedArg {
    /// Try to interpret as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            ResolvedArg::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to interpret as integer
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            ResolvedArg::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to interpret as float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ResolvedArg::Float(f) => Some(*f),
            ResolvedArg::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to interpret as boolean
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            ResolvedArg::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}

// ==========================================================================
// Function Context
// ==========================================================================

/// Context for function execution.
///
/// Provides access to collection configuration, defaults, and index optimization
/// data that handlers may need for computation.
///
/// # Index Optimization Data
///
/// When the backend has index optimizations enabled (via `IndexConfig`), the
/// `index_data` field carries pre-computed values that handlers can use for
/// accelerated computation:
///
/// ```ignore
/// fn compute(&self, query: &[f32], embeddings: &Array2<f32>, ..., context: &FunctionContext) {
///     // Check for pre-normalized embeddings (cosine optimization)
///     if context.is_pre_normalized() {
///         // Use dot product instead of full cosine computation
///     }
///
///     // Check for cached norms (Euclidean optimization)
///     if let Some(norms) = context.get_squared_norms() {
///         // Use ||a-b||² = ||a||² + ||b||² - 2(a·b)
///     }
///
///     // Check for packed binary (Hamming/Jaccard optimization)
///     if let Some(packed) = context.get_packed_binary() {
///         // Use hardware popcnt
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct FunctionContext {
    /// Default similarity threshold for this collection
    pub default_threshold: f32,
    /// Default metric identifier for this collection
    pub default_metric: String,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Embedding type (boolean, integer, float)
    pub embedding_type: EmbeddingType,
    /// Optional index optimization data for accelerated computation
    pub index_data: Option<IndexOptimizationData>,
}

impl Default for FunctionContext {
    fn default() -> Self {
        Self {
            default_threshold: 0.8,
            default_metric: "cosine".to_string(),
            dimensions: 0,
            embedding_type: EmbeddingType::Float,
            index_data: None,
        }
    }
}

impl FunctionContext {
    /// Create a new function context with the given parameters.
    pub fn new(
        default_threshold: f32,
        default_metric: impl Into<String>,
        dimensions: usize,
        embedding_type: EmbeddingType,
    ) -> Self {
        Self {
            default_threshold,
            default_metric: default_metric.into(),
            dimensions,
            embedding_type,
            index_data: None,
        }
    }

    /// Create a new function context with index optimization data.
    pub fn with_index_data(
        default_threshold: f32,
        default_metric: impl Into<String>,
        dimensions: usize,
        embedding_type: EmbeddingType,
        index_data: IndexOptimizationData,
    ) -> Self {
        Self {
            default_threshold,
            default_metric: default_metric.into(),
            dimensions,
            embedding_type,
            index_data: Some(index_data),
        }
    }

    /// Set index optimization data on an existing context.
    pub fn set_index_data(&mut self, index_data: IndexOptimizationData) {
        self.index_data = Some(index_data);
    }

    // =========================================================================
    // Index Optimization Getters (for Similarity Handlers)
    // =========================================================================

    /// Check if embeddings are pre-L2-normalized.
    ///
    /// When `true`, cosine similarity can be computed as a simple dot product:
    /// `cos(a,b) = a' · b'` where `a'` and `b'` are L2-normalized.
    ///
    /// # Performance Impact
    /// - Avoids per-vector norm computation
    /// - Enables BLAS-accelerated batch dot products
    /// - Expected speedup: 10-20% for cosine similarity
    #[inline]
    pub fn is_pre_normalized(&self) -> bool {
        self.index_data
            .as_ref()
            .map(|d| d.is_pre_normalized)
            .unwrap_or(false)
    }

    /// Get cached squared L2 norms for Euclidean optimization.
    ///
    /// Returns `Some(&[f32])` if norm caching is enabled, where index `i`
    /// contains `||embedding[i]||²`.
    ///
    /// Enables Euclidean distance via the identity:
    /// `||a-b||² = ||a||² + ||b||² - 2(a·b)`
    ///
    /// # Performance Impact
    /// - Converts row-by-row subtraction to single BLAS dot product
    /// - Expected speedup: 20-40% for Euclidean distance
    ///
    /// # Usage Example
    /// ```ignore
    /// if let Some(squared_norms) = context.get_squared_norms() {
    ///     let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();
    ///     let dot_products = embeddings.dot(&query);  // BLAS
    ///     // ||a-b||² = ||a||² + ||b||² - 2(a·b)
    ///     let distances_sq = squared_norms + query_norm_sq - 2.0 * dot_products;
    /// }
    /// ```
    #[inline]
    pub fn get_squared_norms(&self) -> Option<&[f32]> {
        self.index_data
            .as_ref()
            .and_then(|d| d.squared_norms.as_deref())
    }

    /// Get packed binary embeddings for Hamming/Jaccard optimization.
    ///
    /// Returns `Some(&[Vec<u64>])` if binary packing is enabled, where each
    /// `Vec<u64>` represents a binarized embedding packed into 64-bit words.
    ///
    /// Embeddings are binarized at threshold 0.5:
    /// - `value > 0.5` → bit set to 1
    /// - `value <= 0.5` → bit set to 0
    ///
    /// # Performance Impact
    /// - Memory reduction: 64 f32 → 1 u64 (256x compression)
    /// - Uses hardware popcnt for distance computation
    /// - Expected speedup: 50-100x for Hamming/Jaccard
    ///
    /// # Usage Example
    /// ```ignore
    /// if let Some(packed) = context.get_packed_binary() {
    ///     let query_packed = pack_to_binary(query);
    ///     let bits = context.get_packed_bits() as f32;
    ///     for packed_embedding in packed {
    ///         let dist = hamming_distance_packed(&query_packed, packed_embedding);
    ///         let similarity = 1.0 - (dist as f32 / bits);
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn get_packed_binary(&self) -> Option<&[Vec<u64>]> {
        self.index_data
            .as_ref()
            .and_then(|d| d.packed_binary.as_deref())
    }

    /// Get the number of dimensions (bits) in packed binary embeddings.
    ///
    /// This is the original embedding dimensionality before packing,
    /// used for proper normalization when computing Hamming distance.
    ///
    /// Returns 0 if binary packing is not enabled.
    #[inline]
    pub fn get_packed_bits(&self) -> usize {
        self.index_data
            .as_ref()
            .map(|d| d.packed_bits)
            .unwrap_or(0)
    }

    /// Check if any index optimizations are available.
    ///
    /// Returns `true` if at least one optimization (pre-normalization,
    /// norm caching, or binary packing) is enabled.
    #[inline]
    pub fn has_index_optimizations(&self) -> bool {
        self.index_data
            .as_ref()
            .map(|d| d.has_optimizations())
            .unwrap_or(false)
    }
}

// ==========================================================================
// Result Types
// ==========================================================================

/// Result of similarity computation from a metric handler.
///
/// Contains per-embedding similarity scores. With compact storage,
/// score index corresponds directly to row index in the embeddings matrix.
#[derive(Clone, Debug)]
pub struct SimilarityResult {
    /// Per-embedding similarity scores (length = num_embeddings)
    pub scores: Array1<f32>,
    /// Threshold used for filtering (may differ from query if defaulted)
    pub threshold: f32,
}

impl SimilarityResult {
    /// Create a new similarity result.
    pub fn new(scores: Array1<f32>, threshold: f32) -> Self {
        Self { scores, threshold }
    }
}

/// Result of ranking computation from a ranking handler.
#[derive(Clone, Debug)]
pub struct RankingResult {
    /// Indices of selected documents with their scores, sorted by score descending
    pub matches: Vec<(usize, f32)>,
}

impl RankingResult {
    /// Create a new ranking result.
    pub fn new(matches: Vec<(usize, f32)>) -> Self {
        Self { matches }
    }

    /// Create an empty ranking result.
    pub fn empty() -> Self {
        Self { matches: Vec::new() }
    }

    /// Get the number of matches.
    pub fn len(&self) -> usize {
        self.matches.len()
    }

    /// Check if there are no matches.
    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    /// Get the best match (highest score).
    pub fn best(&self) -> Option<(usize, f32)> {
        self.matches.first().copied()
    }

    /// Iterate over matches.
    pub fn iter(&self) -> impl Iterator<Item = &(usize, f32)> {
        self.matches.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ResolvedArg Tests
    // =========================================================================

    #[test]
    fn test_resolved_arg_as_string() {
        let arg = ResolvedArg::String("cosine".to_string());
        assert_eq!(arg.as_string(), Some("cosine"));
        assert_eq!(arg.as_integer(), None);
    }

    #[test]
    fn test_resolved_arg_as_integer() {
        let arg = ResolvedArg::Integer(42);
        assert_eq!(arg.as_integer(), Some(42));
        assert_eq!(arg.as_float(), Some(42.0));
        assert_eq!(arg.as_string(), None);
    }

    #[test]
    fn test_resolved_arg_as_float() {
        let arg = ResolvedArg::Float(0.8);
        assert_eq!(arg.as_float(), Some(0.8));
        assert_eq!(arg.as_integer(), None);
    }

    #[test]
    fn test_resolved_arg_as_boolean() {
        let arg = ResolvedArg::Boolean(true);
        assert_eq!(arg.as_boolean(), Some(true));
        assert_eq!(arg.as_string(), None);
    }

    // =========================================================================
    // IndexOptimizationData Tests
    // =========================================================================

    #[test]
    fn test_index_optimization_data_default() {
        let data = IndexOptimizationData::default();
        assert!(!data.is_pre_normalized);
        assert!(data.squared_norms.is_none());
        assert!(data.packed_binary.is_none());
        assert_eq!(data.packed_bits, 0);
        assert!(!data.has_optimizations());
    }

    #[test]
    fn test_index_optimization_data_pre_normalized() {
        let data = IndexOptimizationData::pre_normalized();
        assert!(data.is_pre_normalized);
        assert!(data.squared_norms.is_none());
        assert!(data.packed_binary.is_none());
        assert!(data.has_optimizations());
    }

    #[test]
    fn test_index_optimization_data_squared_norms() {
        let norms = vec![1.0, 2.0, 3.0];
        let data = IndexOptimizationData::with_squared_norms(norms.clone());
        assert!(!data.is_pre_normalized);
        assert_eq!(data.squared_norms, Some(norms));
        assert!(data.packed_binary.is_none());
        assert!(data.has_optimizations());
    }

    #[test]
    fn test_index_optimization_data_packed_binary() {
        let packed = vec![vec![0xFFu64], vec![0x00u64]];
        let data = IndexOptimizationData::with_packed_binary(packed.clone(), 64);
        assert!(!data.is_pre_normalized);
        assert!(data.squared_norms.is_none());
        assert_eq!(data.packed_binary, Some(packed));
        assert_eq!(data.packed_bits, 64);
        assert!(data.has_optimizations());
    }

    // =========================================================================
    // FunctionContext Tests
    // =========================================================================

    #[test]
    fn test_function_context_default() {
        let ctx = FunctionContext::default();
        assert_eq!(ctx.default_threshold, 0.8);
        assert_eq!(ctx.default_metric, "cosine");
        assert_eq!(ctx.dimensions, 0);
        assert!(matches!(ctx.embedding_type, EmbeddingType::Float));
        assert!(ctx.index_data.is_none());
        assert!(!ctx.is_pre_normalized());
        assert!(ctx.get_squared_norms().is_none());
        assert!(ctx.get_packed_binary().is_none());
        assert_eq!(ctx.get_packed_bits(), 0);
        assert!(!ctx.has_index_optimizations());
    }

    #[test]
    fn test_function_context_with_index_data() {
        let index_data = IndexOptimizationData {
            is_pre_normalized: true,
            squared_norms: Some(vec![1.0, 4.0, 9.0]),
            packed_binary: None,
            packed_bits: 0,
        };
        let ctx = FunctionContext::with_index_data(
            0.9,
            "euclidean",
            128,
            EmbeddingType::Float,
            index_data,
        );

        assert_eq!(ctx.default_threshold, 0.9);
        assert_eq!(ctx.default_metric, "euclidean");
        assert_eq!(ctx.dimensions, 128);
        assert!(ctx.is_pre_normalized());
        assert_eq!(ctx.get_squared_norms(), Some(&[1.0, 4.0, 9.0][..]));
        assert!(ctx.get_packed_binary().is_none());
        assert!(ctx.has_index_optimizations());
    }

    #[test]
    fn test_function_context_set_index_data() {
        let mut ctx = FunctionContext::default();
        assert!(!ctx.has_index_optimizations());

        let packed = vec![vec![0xABCDu64], vec![0x1234u64]];
        let index_data = IndexOptimizationData::with_packed_binary(packed.clone(), 64);
        ctx.set_index_data(index_data);

        assert!(ctx.has_index_optimizations());
        assert_eq!(ctx.get_packed_binary(), Some(&packed[..]));
        assert_eq!(ctx.get_packed_bits(), 64);
    }

    #[test]
    fn test_function_context_all_optimizations() {
        let packed = vec![vec![0xFFu64, 0x00u64]];
        let index_data = IndexOptimizationData {
            is_pre_normalized: true,
            squared_norms: Some(vec![2.5]),
            packed_binary: Some(packed.clone()),
            packed_bits: 128,
        };
        let ctx = FunctionContext::with_index_data(
            0.7,
            "cosine",
            128,
            EmbeddingType::Float,
            index_data,
        );

        assert!(ctx.is_pre_normalized());
        assert_eq!(ctx.get_squared_norms(), Some(&[2.5][..]));
        assert_eq!(ctx.get_packed_binary(), Some(&packed[..]));
        assert_eq!(ctx.get_packed_bits(), 128);
        assert!(ctx.has_index_optimizations());
    }

    // =========================================================================
    // SimilarityResult Tests
    // =========================================================================

    #[test]
    fn test_similarity_result() {
        let scores = Array1::from_vec(vec![0.9, 0.5, 0.3]);
        let result = SimilarityResult::new(scores, 0.8);
        assert_eq!(result.threshold, 0.8);
        assert_eq!(result.scores.len(), 3);
    }

    // =========================================================================
    // RankingResult Tests
    // =========================================================================

    #[test]
    fn test_ranking_result() {
        let matches = vec![(0, 0.95), (2, 0.85), (1, 0.75)];
        let result = RankingResult::new(matches);
        assert_eq!(result.len(), 3);
        assert!(!result.is_empty());
        assert_eq!(result.best(), Some((0, 0.95)));
    }

    #[test]
    fn test_ranking_result_empty() {
        let result = RankingResult::empty();
        assert!(result.is_empty());
        assert_eq!(result.best(), None);
    }
}
