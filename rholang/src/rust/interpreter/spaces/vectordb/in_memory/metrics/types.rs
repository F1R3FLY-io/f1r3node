//! Core type definitions for similarity metrics and embeddings.
//!
//! These types are always available regardless of feature flags, as they
//! define the interface that backends must implement.

use serde::{Deserialize, Serialize};

/// Similarity metric types for VectorDB operations.
///
/// Each VectorDB backend declares which metrics it supports. If a metric is
/// specified during construction that is not supported, an error is returned.
///
/// # Metric Categories
///
/// - **Dot-product based**: Cosine, DotProduct - efficient with pre-normalization
/// - **Distance-based**: Euclidean, Manhattan - converted to similarity scores
/// - **Boolean-specific**: Hamming, Jaccard - designed for hypervectors
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// Cosine similarity: dot(a_norm, b_norm)
    /// Requires pre-normalization of vectors for efficiency.
    /// Range: [-1.0, 1.0] (typically [0.0, 1.0] for positive embeddings)
    #[default]
    Cosine,

    /// Raw dot product: dot(a, b)
    /// Does not normalize vectors - magnitude matters.
    /// Range: unbounded
    DotProduct,

    /// Euclidean similarity: 1 / (1 + ||a - b||)
    /// Based on L2 distance, converted to similarity score.
    /// Range: (0.0, 1.0]
    Euclidean,

    /// Manhattan similarity: 1 / (1 + L1(a, b))
    /// Based on L1 distance, converted to similarity score.
    /// Range: (0.0, 1.0]
    Manhattan,

    /// Hamming similarity for boolean/binary vectors (hypervectors).
    /// Computed as: 1 - (count(a_i != b_i) / len)
    /// Range: [0.0, 1.0]
    Hamming,

    /// Jaccard similarity for boolean/binary vectors.
    /// Computed as: |A ∩ B| / |A ∪ B|
    /// Range: [0.0, 1.0]
    Jaccard,
}

impl SimilarityMetric {
    /// Parse a metric string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cosine" => Some(Self::Cosine),
            "dot" | "dotproduct" | "dot_product" => Some(Self::DotProduct),
            "euclidean" | "l2" => Some(Self::Euclidean),
            "manhattan" | "l1" => Some(Self::Manhattan),
            "hamming" => Some(Self::Hamming),
            "jaccard" => Some(Self::Jaccard),
            _ => None,
        }
    }

    /// Get the canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::DotProduct => "dot_product",
            Self::Euclidean => "euclidean",
            Self::Manhattan => "manhattan",
            Self::Hamming => "hamming",
            Self::Jaccard => "jaccard",
        }
    }

    /// Get all supported metrics.
    pub const fn all() -> &'static [SimilarityMetric] {
        &[
            SimilarityMetric::Cosine,
            SimilarityMetric::DotProduct,
            SimilarityMetric::Euclidean,
            SimilarityMetric::Manhattan,
            SimilarityMetric::Hamming,
            SimilarityMetric::Jaccard,
        ]
    }
}

// Note: EmbeddingType is now defined in vectordb/types.rs and re-exported
// through metrics/mod.rs to ensure a single type definition across the crate.

// =============================================================================
// Index Configuration Types
// =============================================================================

/// Configuration for embedding indexing strategies.
///
/// These indexing strategies enable metric-specific optimizations that can
/// significantly improve query performance at the cost of additional memory
/// or preprocessing time.
///
/// # Optimization Summary
///
/// | Index Type       | Memory Overhead | Query Speedup | Best For              |
/// |------------------|-----------------|---------------|-----------------------|
/// | Pre-normalization| +100% embeddings| 10-20%        | Cosine similarity     |
/// | Norm caching     | +1 f32/embedding| 20-40%        | Euclidean distance    |
/// | Binary packing   | -97% (256x)     | 50-100x       | Hamming/Jaccard       |
/// | HNSW             | +50-100%        | 10-100x       | Large datasets (>10k) |
/// | Scalar quant 8   | -75% storage    | ~0%           | Memory constraints    |
///
/// # Rholang Configuration
///
/// ```rholang
/// VectorDBFactory!({
///   "dimensions": 384,
///   "metric": "cosine",
///   "index": "pre_normalize"  // Single index
/// }, *space)
///
/// VectorDBFactory!({
///   "dimensions": 384,
///   "indices": ["pre_normalize", "cache_norms"]  // Multiple indices
/// }, *space)
///
/// VectorDBFactory!({
///   "dimensions": 768,
///   "index": {  // Complex index with parameters
///     "type": "hnsw",
///     "max_connections": 32,
///     "ef_construction": 400
///   }
/// }, *space)
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Enable pre-L2 normalization for cosine similarity optimization.
    ///
    /// When enabled, embeddings are L2-normalized at storage time. This converts
    /// cosine similarity to a simple dot product, enabling efficient BLAS operations:
    ///
    /// `cos(a,b) = (a/||a||) · (b/||b||)` → With pre-normalized: `a' · b'`
    ///
    /// Memory overhead: +100% (stores normalized copy)
    /// Speedup: 10-20% for cosine similarity
    pub pre_normalize: bool,

    /// Enable L2 norm caching for Euclidean distance optimization.
    ///
    /// When enabled, the squared L2 norm of each embedding is cached. This enables
    /// the algebraic identity:
    ///
    /// `||a-b||² = ||a||² + ||b||² - 2(a·b)`
    ///
    /// Instead of computing per-element subtraction, we use cached norms and
    /// a single BLAS dot product.
    ///
    /// Memory overhead: +1 f32 per embedding
    /// Speedup: 20-40% for Euclidean distance
    pub cache_norms: bool,

    /// Enable binary packing for Hamming/Jaccard optimization.
    ///
    /// When enabled, embeddings are pre-binarized (threshold 0.5) and packed
    /// into u64 words for efficient popcnt-based distance computation:
    ///
    /// - Hamming: `popcnt(a_bits XOR b_bits) / dimensions`
    /// - Jaccard: `popcnt(a_bits AND b_bits) / popcnt(a_bits OR b_bits)`
    ///
    /// Memory overhead: -97% (64 f32 dimensions → 1 u64, 256x compression)
    /// Speedup: 50-100x for binary metrics
    pub pack_binary: bool,

    /// Optional HNSW index configuration for approximate nearest neighbor search.
    ///
    /// HNSW (Hierarchical Navigable Small World) enables sub-linear search time
    /// for large datasets. Results are approximate but typically 95%+ accurate.
    pub hnsw: Option<HnswConfig>,

    /// Optional scalar quantization for memory reduction.
    ///
    /// Reduces embedding precision to save memory at the cost of some accuracy.
    pub scalar_quantization: Option<ScalarQuantizationConfig>,
}

impl IndexConfig {
    /// Create a new empty index configuration (no optimizations).
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable pre-normalization for cosine similarity.
    pub fn with_pre_normalize(mut self) -> Self {
        self.pre_normalize = true;
        self
    }

    /// Enable norm caching for Euclidean distance.
    pub fn with_norm_caching(mut self) -> Self {
        self.cache_norms = true;
        self
    }

    /// Enable binary packing for Hamming/Jaccard.
    pub fn with_binary_packing(mut self) -> Self {
        self.pack_binary = true;
        self
    }

    /// Enable HNSW index with default parameters.
    pub fn with_hnsw_default(mut self) -> Self {
        self.hnsw = Some(HnswConfig::default());
        self
    }

    /// Enable HNSW index with custom configuration.
    pub fn with_hnsw(mut self, config: HnswConfig) -> Self {
        self.hnsw = Some(config);
        self
    }

    /// Enable scalar quantization with 8-bit precision.
    pub fn with_scalar_quantization_8bit(mut self) -> Self {
        self.scalar_quantization = Some(ScalarQuantizationConfig::default());
        self
    }

    /// Enable scalar quantization with custom configuration.
    pub fn with_scalar_quantization(mut self, config: ScalarQuantizationConfig) -> Self {
        self.scalar_quantization = Some(config);
        self
    }

    /// Check if any index optimizations are enabled.
    pub fn has_any_optimization(&self) -> bool {
        self.pre_normalize
            || self.cache_norms
            || self.pack_binary
            || self.hnsw.is_some()
            || self.scalar_quantization.is_some()
    }

    /// Check if pre-normalization is enabled.
    pub fn is_pre_normalized(&self) -> bool {
        self.pre_normalize
    }

    /// Check if norm caching is enabled.
    pub fn is_norm_cached(&self) -> bool {
        self.cache_norms
    }

    /// Check if binary packing is enabled.
    pub fn is_binary_packed(&self) -> bool {
        self.pack_binary
    }
}

/// HNSW (Hierarchical Navigable Small World) index configuration.
///
/// HNSW provides approximate nearest neighbor search with sub-linear time
/// complexity. It's ideal for large datasets (>10k embeddings) where exact
/// search becomes too slow.
///
/// # Performance Characteristics
///
/// - Search time: O(log N) instead of O(N)
/// - Memory: +50-100% overhead for graph structure
/// - Accuracy: Typically 95%+ recall
///
/// # Parameters
///
/// - `max_connections` (M): Higher = more accurate but more memory
/// - `ef_construction`: Higher = better graph quality but slower build
/// - `ef_search`: Higher = more accurate search but slower query
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections per layer (M parameter).
    ///
    /// Controls the graph density. Higher values improve recall at the cost
    /// of more memory and longer build times.
    ///
    /// Typical values: 8-64, default: 16
    pub max_connections: usize,

    /// Size of dynamic candidate list during construction (ef_construction).
    ///
    /// Controls graph quality during index building. Higher values build
    /// a better graph but take longer.
    ///
    /// Typical values: 100-500, default: 200
    pub ef_construction: usize,

    /// Size of dynamic candidate list during search (ef_search).
    ///
    /// Controls search accuracy vs. speed trade-off. Higher values return
    /// more accurate results but take longer.
    ///
    /// Typical values: 50-200, default: 50
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

impl HnswConfig {
    /// Create an HNSW configuration optimized for high recall (>99%).
    pub fn high_recall() -> Self {
        Self {
            max_connections: 32,
            ef_construction: 400,
            ef_search: 100,
        }
    }

    /// Create an HNSW configuration optimized for fast search.
    pub fn fast_search() -> Self {
        Self {
            max_connections: 12,
            ef_construction: 100,
            ef_search: 30,
        }
    }

    /// Create an HNSW configuration for memory-constrained environments.
    pub fn low_memory() -> Self {
        Self {
            max_connections: 8,
            ef_construction: 100,
            ef_search: 50,
        }
    }
}

/// Scalar quantization configuration for memory-efficient storage.
///
/// Reduces embedding precision by quantizing floating-point values to fewer bits.
/// This trades some accuracy for significant memory savings.
///
/// # Quantization Levels
///
/// | Bits | Memory Reduction | Typical Accuracy Loss |
/// |------|------------------|----------------------|
/// | 8    | 75%              | <1%                  |
/// | 4    | 87.5%            | 1-3%                 |
/// | 2    | 93.75%           | 3-5%                 |
/// | 1    | 96.875%          | 5-10%                |
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScalarQuantizationConfig {
    /// Number of bits per dimension.
    ///
    /// Valid options: 8, 4, 2, 1
    pub bits: u8,

    /// Whether to keep original embeddings for re-ranking.
    ///
    /// When true, quantized embeddings are used for initial candidate selection,
    /// then original embeddings are used to re-rank the top candidates.
    /// This improves accuracy at the cost of keeping both versions in memory.
    pub keep_original: bool,
}

impl Default for ScalarQuantizationConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            keep_original: false,
        }
    }
}

impl ScalarQuantizationConfig {
    /// Create an 8-bit quantization config (best accuracy/memory trade-off).
    pub fn int8() -> Self {
        Self {
            bits: 8,
            keep_original: false,
        }
    }

    /// Create a 4-bit quantization config (more aggressive compression).
    pub fn int4() -> Self {
        Self {
            bits: 4,
            keep_original: false,
        }
    }

    /// Create a binary (1-bit) quantization config (maximum compression).
    pub fn binary() -> Self {
        Self {
            bits: 1,
            keep_original: false,
        }
    }

    /// Enable keeping original embeddings for re-ranking.
    pub fn with_reranking(mut self) -> Self {
        self.keep_original = true;
        self
    }
}

/// Known index type identifiers for Rholang configuration parsing.
///
/// These strings are recognized in the `"index"` or `"indices"` configuration keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexType {
    /// Pre-L2 normalization: `"pre_normalize"`, `"prenorm"`, `"l2_normalize"`
    PreNormalize,
    /// L2 norm caching: `"cache_norms"`, `"norm_cache"`
    CacheNorms,
    /// Binary packing: `"pack_binary"`, `"binary"`
    PackBinary,
    /// HNSW index: `"hnsw"`
    Hnsw,
    /// Scalar quantization: `"sq8"`, `"scalar_quantization"`
    ScalarQuantization,
}

impl IndexType {
    /// Parse an index type from a string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pre_normalize" | "prenorm" | "l2_normalize" | "prenormalize" => {
                Some(Self::PreNormalize)
            }
            "cache_norms" | "norm_cache" | "cachenorms" | "normcache" => Some(Self::CacheNorms),
            "pack_binary" | "binary" | "packbinary" => Some(Self::PackBinary),
            "hnsw" => Some(Self::Hnsw),
            "sq8" | "scalar_quantization" | "scalarquantization" | "sq" => {
                Some(Self::ScalarQuantization)
            }
            _ => None,
        }
    }

    /// Get the canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PreNormalize => "pre_normalize",
            Self::CacheNorms => "cache_norms",
            Self::PackBinary => "pack_binary",
            Self::Hnsw => "hnsw",
            Self::ScalarQuantization => "scalar_quantization",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_metric_from_str() {
        assert_eq!(SimilarityMetric::from_str("cosine"), Some(SimilarityMetric::Cosine));
        assert_eq!(SimilarityMetric::from_str("COSINE"), Some(SimilarityMetric::Cosine));
        assert_eq!(SimilarityMetric::from_str("dot"), Some(SimilarityMetric::DotProduct));
        assert_eq!(SimilarityMetric::from_str("dotproduct"), Some(SimilarityMetric::DotProduct));
        assert_eq!(SimilarityMetric::from_str("l2"), Some(SimilarityMetric::Euclidean));
        assert_eq!(SimilarityMetric::from_str("l1"), Some(SimilarityMetric::Manhattan));
        assert_eq!(SimilarityMetric::from_str("hamming"), Some(SimilarityMetric::Hamming));
        assert_eq!(SimilarityMetric::from_str("jaccard"), Some(SimilarityMetric::Jaccard));
        assert_eq!(SimilarityMetric::from_str("unknown"), None);
    }

    #[test]
    fn test_similarity_metric_as_str() {
        assert_eq!(SimilarityMetric::Cosine.as_str(), "cosine");
        assert_eq!(SimilarityMetric::DotProduct.as_str(), "dot_product");
        assert_eq!(SimilarityMetric::Euclidean.as_str(), "euclidean");
        assert_eq!(SimilarityMetric::Manhattan.as_str(), "manhattan");
        assert_eq!(SimilarityMetric::Hamming.as_str(), "hamming");
        assert_eq!(SimilarityMetric::Jaccard.as_str(), "jaccard");
    }

    // Note: EmbeddingType tests are now in vectordb/types.rs where EmbeddingType is defined.

    // =========================================================================
    // IndexConfig Tests
    // =========================================================================

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert!(!config.pre_normalize);
        assert!(!config.cache_norms);
        assert!(!config.pack_binary);
        assert!(config.hnsw.is_none());
        assert!(config.scalar_quantization.is_none());
        assert!(!config.has_any_optimization());
    }

    #[test]
    fn test_index_config_builder_pattern() {
        let config = IndexConfig::new()
            .with_pre_normalize()
            .with_norm_caching()
            .with_binary_packing();

        assert!(config.pre_normalize);
        assert!(config.cache_norms);
        assert!(config.pack_binary);
        assert!(config.has_any_optimization());
        assert!(config.is_pre_normalized());
        assert!(config.is_norm_cached());
        assert!(config.is_binary_packed());
    }

    #[test]
    fn test_index_config_with_hnsw() {
        let config = IndexConfig::new().with_hnsw(HnswConfig {
            max_connections: 32,
            ef_construction: 400,
            ef_search: 100,
        });

        assert!(config.hnsw.is_some());
        let hnsw = config.hnsw.expect("HNSW config should exist");
        assert_eq!(hnsw.max_connections, 32);
        assert_eq!(hnsw.ef_construction, 400);
        assert_eq!(hnsw.ef_search, 100);
    }

    #[test]
    fn test_index_config_with_scalar_quantization() {
        let config = IndexConfig::new()
            .with_scalar_quantization(ScalarQuantizationConfig::int4().with_reranking());

        assert!(config.scalar_quantization.is_some());
        let sq = config.scalar_quantization.expect("SQ config should exist");
        assert_eq!(sq.bits, 4);
        assert!(sq.keep_original);
    }

    #[test]
    fn test_hnsw_config_default() {
        let config = HnswConfig::default();
        assert_eq!(config.max_connections, 16);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 50);
    }

    #[test]
    fn test_hnsw_config_presets() {
        let high_recall = HnswConfig::high_recall();
        assert_eq!(high_recall.max_connections, 32);
        assert_eq!(high_recall.ef_construction, 400);
        assert_eq!(high_recall.ef_search, 100);

        let fast = HnswConfig::fast_search();
        assert_eq!(fast.max_connections, 12);
        assert_eq!(fast.ef_search, 30);

        let low_mem = HnswConfig::low_memory();
        assert_eq!(low_mem.max_connections, 8);
    }

    #[test]
    fn test_scalar_quantization_presets() {
        let int8 = ScalarQuantizationConfig::int8();
        assert_eq!(int8.bits, 8);
        assert!(!int8.keep_original);

        let int4 = ScalarQuantizationConfig::int4();
        assert_eq!(int4.bits, 4);

        let binary = ScalarQuantizationConfig::binary();
        assert_eq!(binary.bits, 1);

        let with_rerank = ScalarQuantizationConfig::int8().with_reranking();
        assert!(with_rerank.keep_original);
    }

    #[test]
    fn test_index_type_from_str() {
        // Pre-normalize variants
        assert_eq!(IndexType::from_str("pre_normalize"), Some(IndexType::PreNormalize));
        assert_eq!(IndexType::from_str("prenorm"), Some(IndexType::PreNormalize));
        assert_eq!(IndexType::from_str("l2_normalize"), Some(IndexType::PreNormalize));
        assert_eq!(IndexType::from_str("PRENORMALIZE"), Some(IndexType::PreNormalize));

        // Cache norms variants
        assert_eq!(IndexType::from_str("cache_norms"), Some(IndexType::CacheNorms));
        assert_eq!(IndexType::from_str("norm_cache"), Some(IndexType::CacheNorms));

        // Pack binary variants
        assert_eq!(IndexType::from_str("pack_binary"), Some(IndexType::PackBinary));
        assert_eq!(IndexType::from_str("binary"), Some(IndexType::PackBinary));

        // HNSW
        assert_eq!(IndexType::from_str("hnsw"), Some(IndexType::Hnsw));
        assert_eq!(IndexType::from_str("HNSW"), Some(IndexType::Hnsw));

        // Scalar quantization variants
        assert_eq!(IndexType::from_str("sq8"), Some(IndexType::ScalarQuantization));
        assert_eq!(IndexType::from_str("scalar_quantization"), Some(IndexType::ScalarQuantization));

        // Unknown
        assert_eq!(IndexType::from_str("unknown_index"), None);
    }

    #[test]
    fn test_index_type_as_str() {
        assert_eq!(IndexType::PreNormalize.as_str(), "pre_normalize");
        assert_eq!(IndexType::CacheNorms.as_str(), "cache_norms");
        assert_eq!(IndexType::PackBinary.as_str(), "pack_binary");
        assert_eq!(IndexType::Hnsw.as_str(), "hnsw");
        assert_eq!(IndexType::ScalarQuantization.as_str(), "scalar_quantization");
    }

    #[test]
    fn test_index_config_serialization() {
        let config = IndexConfig::new()
            .with_pre_normalize()
            .with_hnsw_default();

        // Serialize to JSON
        let json = serde_json::to_string(&config).expect("should serialize");
        assert!(json.contains("\"pre_normalize\":true"));
        assert!(json.contains("\"hnsw\""));

        // Deserialize back
        let parsed: IndexConfig = serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(parsed.pre_normalize, config.pre_normalize);
        assert_eq!(parsed.hnsw, config.hnsw);
    }
}
