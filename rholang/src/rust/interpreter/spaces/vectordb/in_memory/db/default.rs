//! Default VectorDB implementation.
//!
//! `DefaultVectorDB` wraps a `VectorBackend` and adds application-level
//! semantics like data association, persistence flags, and consume-on-match.
//!
//! # Builder Pattern
//!
//! For constructing `DefaultVectorDB` with `InMemoryBackend`, use the builder:
//!
//! ```ignore
//! use rho_vectordb::{VectorDBBuilder, SimilarityMetric};
//!
//! let db = VectorDBBuilder::new(384)
//!     .with_threshold(0.7)
//!     .with_metric(SimilarityMetric::Cosine)
//!     .with_pre_normalization()  // Enable cosine optimization
//!     .build();
//! ```
//!
//! Or create from a parsed configuration:
//!
//! ```ignore
//! use rho_vectordb::db::{VectorDBConfig, VectorDBBuilder};
//!
//! let config = VectorDBConfig::new(384)
//!     .with_threshold(0.8)
//!     .with_metric(SimilarityMetric::Cosine);
//!
//! let db: DefaultVectorDB<String, _> = VectorDBBuilder::from_config(config).build();
//! ```

use std::collections::HashMap;
use std::marker::PhantomData;

use super::super::backend::{InMemoryBackend, VectorBackend};
use super::super::db::config::VectorDBConfig;
use super::super::db::VectorDB;
use super::super::error::VectorDBError;
use super::super::metrics::{EmbeddingType, IndexConfig, SimilarityMetric};

// ============================================================================
// DefaultVectorDB
// ============================================================================

/// Default VectorDB implementation that wraps any VectorBackend.
///
/// This struct adds application-level semantics on top of raw vector storage:
///
/// - **Data Association**: Maps embedding IDs to user data items
/// - **Persistence Flags**: Controls whether items are removed on consume
/// - **Consume-on-Match**: Implements RSpace's produce/consume pattern
///
/// # Type Parameters
///
/// * `A` - The data type to associate with embeddings
/// * `B` - The underlying `VectorBackend` implementation
///
/// # Example
///
/// ```ignore
/// use rho_vectordb::{DefaultVectorDB, InMemoryBackend};
///
/// let backend = InMemoryBackend::new(384);
/// let mut db: DefaultVectorDB<String, _> = DefaultVectorDB::new(backend);
///
/// db.put("document1".to_string(), vec![0.1; 384]).unwrap();
///
/// if let Some((doc, score)) = db.consume_most_similar(&vec![0.1; 384], 0.8) {
///     println!("Found: {} with score {}", doc, score);
/// }
/// ```
#[derive(Clone)]
pub struct DefaultVectorDB<A, B>
where
    B: VectorBackend,
{
    /// Underlying vector storage backend.
    backend: B,
    /// Data items mapped by backend ID.
    data: HashMap<B::Id, A>,
    /// Persistence flags per ID (true = don't remove on consume).
    persist_flags: HashMap<B::Id, bool>,
    /// Default similarity threshold.
    threshold: f32,
    /// Similarity metric to use.
    metric: SimilarityMetric,
    /// Expected embedding type from input.
    embedding_type: EmbeddingType,
}

impl<A, B> DefaultVectorDB<A, B>
where
    B: VectorBackend,
{
    /// Create a new VectorDB with the given backend.
    ///
    /// Uses default settings: 0.8 threshold, Cosine metric, Float embedding type.
    pub fn new(backend: B) -> Self {
        Self::with_config(backend, 0.8, SimilarityMetric::Cosine, EmbeddingType::Float)
    }

    /// Create with a custom similarity threshold.
    pub fn with_threshold(backend: B, threshold: f32) -> Self {
        Self::with_config(
            backend,
            threshold.clamp(0.0, 1.0),
            SimilarityMetric::Cosine,
            EmbeddingType::Float,
        )
    }

    /// Create with a custom similarity metric.
    pub fn with_metric(backend: B, threshold: f32, metric: SimilarityMetric) -> Self {
        Self::with_config(backend, threshold.clamp(0.0, 1.0), metric, EmbeddingType::Float)
    }

    /// Create with full configuration.
    pub fn with_config(
        backend: B,
        threshold: f32,
        metric: SimilarityMetric,
        embedding_type: EmbeddingType,
    ) -> Self {
        DefaultVectorDB {
            backend,
            data: HashMap::new(),
            persist_flags: HashMap::new(),
            threshold: threshold.clamp(0.0, 1.0),
            metric,
            embedding_type,
        }
    }

    /// Set the similarity threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set the similarity metric.
    pub fn set_metric(&mut self, metric: SimilarityMetric) {
        self.metric = metric;
    }
}

impl<A, B> VectorDB<A> for DefaultVectorDB<A, B>
where
    A: Clone + Send + Sync,
    B: VectorBackend,
{
    type Backend = B;

    fn put(&mut self, data: A, embedding: Vec<f32>) -> Result<(), VectorDBError> {
        self.put_with_persist(data, embedding, false)
    }

    fn put_with_persist(
        &mut self,
        data: A,
        embedding: Vec<f32>,
        persist: bool,
    ) -> Result<(), VectorDBError> {
        let id = self.backend.store(&embedding)?;
        self.data.insert(id.clone(), data);
        self.persist_flags.insert(id, persist);
        Ok(())
    }

    fn consume_most_similar(&mut self, query: &[f32], threshold: f32) -> Option<(A, f32)>
    where
        A: Clone,
    {
        let threshold = threshold.clamp(0.0, 1.0);

        let results = self
            .backend
            .find_similar(query, self.metric, threshold, Some(1))
            .ok()?;

        let (id, score) = results.into_iter().next()?;

        // Get data
        let data = self.data.get(&id)?.clone();

        // Check persistence
        let persist = self.persist_flags.get(&id).copied().unwrap_or(false);

        if !persist {
            // Remove from backend and internal storage
            self.backend.remove(&id);
            self.data.remove(&id);
            self.persist_flags.remove(&id);
        }

        Some((data, score))
    }

    fn peek_most_similar(&self, query: &[f32], threshold: f32) -> Option<(&A, f32)> {
        let threshold = threshold.clamp(0.0, 1.0);

        let results = self
            .backend
            .find_similar(query, self.metric, threshold, Some(1))
            .ok()?;

        let (id, score) = results.into_iter().next()?;
        self.data.get(&id).map(|data| (data, score))
    }

    fn peek_top_k(&self, query: &[f32], threshold: f32, k: usize) -> Vec<(&A, f32)> {
        if k == 0 {
            return vec![];
        }

        let threshold = threshold.clamp(0.0, 1.0);

        let results = match self
            .backend
            .find_similar(query, self.metric, threshold, Some(k))
        {
            Ok(r) => r,
            Err(_) => return vec![],
        };

        results
            .into_iter()
            .filter_map(|(id, score)| self.data.get(&id).map(|data| (data, score)))
            .collect()
    }

    fn consume_top_k(&mut self, query: &[f32], threshold: f32, k: usize) -> Vec<(A, f32)>
    where
        A: Clone,
    {
        if k == 0 {
            return vec![];
        }

        let threshold = threshold.clamp(0.0, 1.0);

        let results = match self
            .backend
            .find_similar(query, self.metric, threshold, Some(k))
        {
            Ok(r) => r,
            Err(_) => return vec![],
        };

        results
            .into_iter()
            .filter_map(|(id, score)| {
                let data = self.data.get(&id)?.clone();
                let persist = self.persist_flags.get(&id).copied().unwrap_or(false);

                if !persist {
                    self.backend.remove(&id);
                    self.data.remove(&id);
                    self.persist_flags.remove(&id);
                }

                Some((data, score))
            })
            .collect()
    }

    fn default_threshold(&self) -> f32 {
        self.threshold
    }

    fn dimensions(&self) -> usize {
        self.backend.dimensions()
    }

    fn metric(&self) -> SimilarityMetric {
        self.metric
    }

    fn embedding_type(&self) -> EmbeddingType {
        self.embedding_type
    }

    fn len(&self) -> usize {
        self.backend.len()
    }

    fn clear(&mut self) {
        self.backend.clear();
        self.data.clear();
        self.persist_flags.clear();
    }

    fn backend(&self) -> &Self::Backend {
        &self.backend
    }

    fn backend_mut(&mut self) -> &mut Self::Backend {
        &mut self.backend
    }
}

// ============================================================================
// VectorDBBuilder
// ============================================================================

/// Builder for constructing `DefaultVectorDB` with `InMemoryBackend`.
///
/// This builder provides a fluent API for configuring VectorDB instances
/// with various options including index optimizations.
///
/// # Example
///
/// ```ignore
/// use rho_vectordb::VectorDBBuilder;
///
/// let db: DefaultVectorDB<String, InMemoryBackend> = VectorDBBuilder::new(384)
///     .with_threshold(0.7)
///     .with_capacity(10000)
///     .with_pre_normalization()
///     .build();
/// ```
pub struct VectorDBBuilder<A> {
    /// Embedding dimensions (required).
    dimensions: usize,
    /// Optional initial capacity hint.
    capacity: Option<usize>,
    /// Similarity threshold (0.0-1.0).
    threshold: f32,
    /// Similarity metric to use.
    metric: SimilarityMetric,
    /// Expected embedding type.
    embedding_type: EmbeddingType,
    /// Index optimization configuration.
    index_config: IndexConfig,
    /// Type marker for associated data type.
    _marker: PhantomData<A>,
}

impl<A> VectorDBBuilder<A> {
    /// Create a new builder with the specified embedding dimensions.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - The dimensionality of embedding vectors
    pub fn new(dimensions: usize) -> Self {
        VectorDBBuilder {
            dimensions,
            capacity: None,
            threshold: 0.8,
            metric: SimilarityMetric::Cosine,
            embedding_type: EmbeddingType::Float,
            index_config: IndexConfig::default(),
            _marker: PhantomData,
        }
    }

    /// Create a builder from a parsed `VectorDBConfig`.
    ///
    /// This is the primary entry point for constructing VectorDB from
    /// Rholang configuration.
    pub fn from_config(config: VectorDBConfig) -> Self {
        VectorDBBuilder {
            dimensions: config.dimensions,
            capacity: config.capacity,
            threshold: config.threshold,
            metric: config.metric,
            embedding_type: config.embedding_type,
            index_config: config.index_config,
            _marker: PhantomData,
        }
    }

    /// Set the initial capacity hint.
    ///
    /// Pre-allocating capacity can reduce memory allocations when the
    /// approximate number of embeddings is known in advance.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }

    /// Set the similarity threshold.
    ///
    /// Values are clamped to [0.0, 1.0].
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the similarity metric.
    pub fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the expected embedding type.
    pub fn with_embedding_type(mut self, embedding_type: EmbeddingType) -> Self {
        self.embedding_type = embedding_type;
        self
    }

    /// Set the full index configuration.
    pub fn with_index_config(mut self, index_config: IndexConfig) -> Self {
        self.index_config = index_config;
        self
    }

    /// Enable L2 pre-normalization for cosine similarity optimization.
    ///
    /// When enabled, embeddings are L2-normalized at storage time, allowing
    /// cosine similarity to be computed as a simple dot product.
    ///
    /// **Speedup**: 10-20% for cosine similarity queries.
    /// **Memory overhead**: +100% (normalized copy).
    pub fn with_pre_normalization(mut self) -> Self {
        self.index_config.pre_normalize = true;
        self
    }

    /// Enable L2 norm caching for Euclidean distance optimization.
    ///
    /// When enabled, squared L2 norms are cached for each embedding, allowing
    /// Euclidean distance to be computed using the identity:
    /// `||a-b||² = ||a||² + ||b||² - 2(a·b)`
    ///
    /// **Speedup**: 20-40% for Euclidean distance queries.
    /// **Memory overhead**: +1 f32 per embedding.
    pub fn with_norm_caching(mut self) -> Self {
        self.index_config.cache_norms = true;
        self
    }

    /// Enable binary packing for Hamming/Jaccard optimization.
    ///
    /// When enabled, embeddings are pre-binarized (threshold 0.5) and packed
    /// into u64 for efficient SIMD popcnt operations.
    ///
    /// **Speedup**: 50-100x for Hamming/Jaccard queries.
    /// **Memory reduction**: -97% (64 dimensions → 1 u64).
    pub fn with_binary_packing(mut self) -> Self {
        self.index_config.pack_binary = true;
        self
    }

    /// Build the `DefaultVectorDB` with the configured options.
    ///
    /// This creates an `InMemoryBackend` with the specified dimensions and
    /// capacity, then wraps it in a `DefaultVectorDB` with the configured
    /// threshold, metric, and embedding type.
    ///
    /// **Note**: Index optimizations (pre_normalize, cache_norms, pack_binary)
    /// are stored in the configuration but require backend support to take
    /// effect. See `InMemoryBackend::with_index_config()` (Phase 6).
    pub fn build(self) -> DefaultVectorDB<A, InMemoryBackend> {
        // Create backend with appropriate capacity
        let backend = match self.capacity {
            Some(cap) => InMemoryBackend::with_capacity(self.dimensions, cap),
            None => InMemoryBackend::new(self.dimensions),
        };

        // Create VectorDB with full configuration
        // Note: index_config will be wired to backend in Phase 6
        DefaultVectorDB::with_config(backend, self.threshold, self.metric, self.embedding_type)
    }

    /// Get the configured index configuration.
    ///
    /// This is useful for inspecting or modifying the configuration before building.
    pub fn index_config(&self) -> &IndexConfig {
        &self.index_config
    }

    /// Get the configured dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Convenience type alias for the common case.
pub type SimpleVectorDB<A> = DefaultVectorDB<A, InMemoryBackend>;

#[cfg(test)]
mod tests {
    use super::*;

    // Mock backend for testing
    #[derive(Clone)]
    struct MockBackend {
        embeddings: Vec<Vec<f32>>,
        live: Vec<bool>,
        dimensions: usize,
    }

    impl MockBackend {
        fn new(dimensions: usize) -> Self {
            MockBackend {
                embeddings: Vec::new(),
                live: Vec::new(),
                dimensions,
            }
        }
    }

    impl VectorBackend for MockBackend {
        type Id = usize;

        fn store(&mut self, embedding: &[f32]) -> Result<Self::Id, VectorDBError> {
            if embedding.len() != self.dimensions {
                return Err(VectorDBError::DimensionMismatch {
                    expected: self.dimensions,
                    actual: embedding.len(),
                });
            }
            let id = self.embeddings.len();
            self.embeddings.push(embedding.to_vec());
            self.live.push(true);
            Ok(id)
        }

        fn get(&self, id: &Self::Id) -> Option<Vec<f32>> {
            if *id < self.embeddings.len() && self.live[*id] {
                Some(self.embeddings[*id].clone())
            } else {
                None
            }
        }

        fn remove(&mut self, id: &Self::Id) -> bool {
            if *id < self.live.len() && self.live[*id] {
                self.live[*id] = false;
                true
            } else {
                false
            }
        }

        fn find_similar(
            &self,
            _query: &[f32],
            _metric: SimilarityMetric,
            _threshold: f32,
            limit: Option<usize>,
        ) -> Result<Vec<(Self::Id, f32)>, VectorDBError> {
            // Simple mock: return all live entries with score 0.9
            let results: Vec<_> = self
                .live
                .iter()
                .enumerate()
                .filter(|(_, &live)| live)
                .map(|(i, _)| (i, 0.9f32))
                .collect();

            Ok(match limit {
                Some(k) => results.into_iter().take(k).collect(),
                None => results,
            })
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn len(&self) -> usize {
            self.live.iter().filter(|&&l| l).count()
        }

        fn clear(&mut self) {
            self.embeddings.clear();
            self.live.clear();
        }
    }

    #[test]
    fn test_put_and_consume() {
        let backend = MockBackend::new(4);
        let mut db: DefaultVectorDB<String, _> = DefaultVectorDB::new(backend);

        db.put("test".to_string(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(db.len(), 1);

        let result = db.consume_most_similar(&vec![1.0, 2.0, 3.0, 4.0], 0.5);
        assert!(result.is_some());
        let (data, score) = result.unwrap();
        assert_eq!(data, "test");
        assert!((score - 0.9).abs() < 0.01);

        // Should be removed after consume
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_persist_flag() {
        let backend = MockBackend::new(4);
        let mut db: DefaultVectorDB<String, _> = DefaultVectorDB::new(backend);

        db.put_with_persist("persistent".to_string(), vec![1.0, 2.0, 3.0, 4.0], true)
            .unwrap();
        assert_eq!(db.len(), 1);

        // Consume should return data but not remove it
        let result = db.consume_most_similar(&vec![1.0, 2.0, 3.0, 4.0], 0.5);
        assert!(result.is_some());
        assert_eq!(db.len(), 1); // Still there!

        // Can consume again
        let result2 = db.consume_most_similar(&vec![1.0, 2.0, 3.0, 4.0], 0.5);
        assert!(result2.is_some());
    }

    #[test]
    fn test_peek() {
        let backend = MockBackend::new(4);
        let mut db: DefaultVectorDB<String, _> = DefaultVectorDB::new(backend);

        db.put("test".to_string(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Peek should not remove
        let result = db.peek_most_similar(&vec![1.0, 2.0, 3.0, 4.0], 0.5);
        assert!(result.is_some());
        assert_eq!(db.len(), 1);

        // Peek again
        let result2 = db.peek_most_similar(&vec![1.0, 2.0, 3.0, 4.0], 0.5);
        assert!(result2.is_some());
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_consume_top_k() {
        let backend = MockBackend::new(4);
        let mut db: DefaultVectorDB<String, _> = DefaultVectorDB::new(backend);

        db.put("one".to_string(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        db.put("two".to_string(), vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        db.put("three".to_string(), vec![0.0, 0.0, 1.0, 0.0]).unwrap();

        let results = db.consume_top_k(&vec![1.0, 1.0, 1.0, 1.0], 0.5, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(db.len(), 1); // Only 2 consumed, 1 remains
    }

    #[test]
    fn test_clear() {
        let backend = MockBackend::new(4);
        let mut db: DefaultVectorDB<String, _> = DefaultVectorDB::new(backend);

        db.put("one".to_string(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        db.put("two".to_string(), vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        db.clear();
        assert_eq!(db.len(), 0);
        assert!(db.is_empty());
    }

    #[test]
    fn test_dimension_mismatch() {
        let backend = MockBackend::new(4);
        let mut db: DefaultVectorDB<String, _> = DefaultVectorDB::new(backend);

        let result = db.put("test".to_string(), vec![1.0, 2.0, 3.0]); // Only 3 dims
        assert!(matches!(result, Err(VectorDBError::DimensionMismatch { .. })));
    }

    // ========================================================================
    // VectorDBBuilder Tests
    // ========================================================================

    #[test]
    fn test_builder_basic() {
        let db: DefaultVectorDB<String, InMemoryBackend> = VectorDBBuilder::new(4).build();

        assert_eq!(db.dimensions(), 4);
        assert!((db.default_threshold() - 0.8).abs() < 0.001); // Default threshold
        assert_eq!(db.metric(), SimilarityMetric::Cosine); // Default metric
        assert_eq!(db.embedding_type(), EmbeddingType::Float); // Default type
    }

    #[test]
    fn test_builder_with_threshold() {
        let db: DefaultVectorDB<String, InMemoryBackend> =
            VectorDBBuilder::new(4).with_threshold(0.7).build();

        assert!((db.default_threshold() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_builder_threshold_clamping() {
        // Test clamping to [0.0, 1.0]
        let db: DefaultVectorDB<String, InMemoryBackend> =
            VectorDBBuilder::new(4).with_threshold(1.5).build();
        assert!((db.default_threshold() - 1.0).abs() < 0.001);

        let db2: DefaultVectorDB<String, InMemoryBackend> =
            VectorDBBuilder::new(4).with_threshold(-0.5).build();
        assert!((db2.default_threshold()).abs() < 0.001);
    }

    #[test]
    fn test_builder_with_metric() {
        let db: DefaultVectorDB<String, InMemoryBackend> = VectorDBBuilder::new(4)
            .with_metric(SimilarityMetric::Euclidean)
            .build();

        assert_eq!(db.metric(), SimilarityMetric::Euclidean);
    }

    #[test]
    fn test_builder_with_embedding_type() {
        let db: DefaultVectorDB<String, InMemoryBackend> = VectorDBBuilder::new(4)
            .with_embedding_type(EmbeddingType::Boolean)
            .build();

        assert_eq!(db.embedding_type(), EmbeddingType::Boolean);
    }

    #[test]
    fn test_builder_with_capacity() {
        // Capacity is passed to backend - we can verify the DB works
        let mut db: DefaultVectorDB<String, InMemoryBackend> =
            VectorDBBuilder::new(4).with_capacity(100).build();

        db.put("test".to_string(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_builder_with_index_config() {
        let config = IndexConfig {
            pre_normalize: true,
            cache_norms: true,
            pack_binary: false,
            hnsw: None,
            scalar_quantization: None,
        };

        let builder: VectorDBBuilder<String> =
            VectorDBBuilder::new(4).with_index_config(config.clone());

        // Verify the config is stored
        assert!(builder.index_config().pre_normalize);
        assert!(builder.index_config().cache_norms);
        assert!(!builder.index_config().pack_binary);

        // Build should still work
        let _db = builder.build();
    }

    #[test]
    fn test_builder_index_optimization_methods() {
        let builder: VectorDBBuilder<String> = VectorDBBuilder::new(4)
            .with_pre_normalization()
            .with_norm_caching()
            .with_binary_packing();

        assert!(builder.index_config().pre_normalize);
        assert!(builder.index_config().cache_norms);
        assert!(builder.index_config().pack_binary);
    }

    #[test]
    fn test_builder_from_config() {
        let config = VectorDBConfig::new(128)
            .with_threshold(0.75)
            .with_metric(SimilarityMetric::DotProduct)
            .with_embedding_type(EmbeddingType::Integer)
            .with_capacity(500);

        let db: DefaultVectorDB<String, InMemoryBackend> =
            VectorDBBuilder::from_config(config).build();

        assert_eq!(db.dimensions(), 128);
        assert!((db.default_threshold() - 0.75).abs() < 0.001);
        assert_eq!(db.metric(), SimilarityMetric::DotProduct);
        assert_eq!(db.embedding_type(), EmbeddingType::Integer);
    }

    #[test]
    fn test_builder_chaining() {
        let mut db: DefaultVectorDB<String, InMemoryBackend> = VectorDBBuilder::new(4)
            .with_capacity(100)
            .with_threshold(0.6)
            .with_metric(SimilarityMetric::Cosine)
            .with_embedding_type(EmbeddingType::Float)
            .with_pre_normalization()
            .with_norm_caching()
            .build();

        // Verify the DB works correctly
        db.put("test".to_string(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(db.len(), 1);
        assert!((db.default_threshold() - 0.6).abs() < 0.001);
        assert_eq!(db.metric(), SimilarityMetric::Cosine);
    }

    #[test]
    fn test_simple_vectordb_alias() {
        // Verify the type alias works
        let mut db: SimpleVectorDB<String> = VectorDBBuilder::new(4).build();

        db.put("test".to_string(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(db.len(), 1);
    }
}
