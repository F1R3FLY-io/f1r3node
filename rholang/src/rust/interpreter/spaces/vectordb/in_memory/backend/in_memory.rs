//! In-memory vector backend with SIMD-optimized operations.
//!
//! This backend provides high-performance vector storage with:
//! - SIMD-accelerated similarity computations via ndarray
//! - Compact storage with swap-and-pop removal (no tombstoning)
//! - Zero-copy query path (no pre-filtering overhead)
//! - Configurable index optimizations (pre-normalization, norm caching, binary packing)
//! - Pluggable handler system for custom similarity/ranking functions
//!
//! # Index Optimizations
//!
//! The backend supports optional index configurations that can significantly
//! improve query performance:
//!
//! - **Pre-normalization**: L2-normalize embeddings at store time for fast cosine similarity
//! - **Norm caching**: Cache squared L2 norms for fast Euclidean distance
//! - **Binary packing**: Pack binarized embeddings into u64 for fast Hamming/Jaccard

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use ndarray::{s, Array1, Array2, ArrayView1, Axis};

use super::super::backend::{HandlerBackend, VectorBackend};
use super::super::error::VectorDBError;
use super::super::handlers::{FunctionContext, FunctionHandlerRegistry, ResolvedArg};
use super::super::metrics::{IndexConfig, SimilarityMetric};
use super::super::utils::binary::{hamming_distance_packed, jaccard_similarity_packed, pack_to_binary};

// ============================================================================
// Internal Types
// ============================================================================

/// Min-heap entry for top-K selection.
///
/// Ordering is reversed so that BinaryHeap gives us the entry with smallest
/// similarity at the top, enabling "kick out the worst" logic.
struct TopKEntry {
    similarity: f32,
    index: usize,
}

impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for TopKEntry {}

impl PartialOrd for TopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TopKEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order: smaller similarity compares as Greater
        // This makes BinaryHeap.peek() return the entry with smallest similarity.
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

// ============================================================================
// InMemoryBackend
// ============================================================================

/// SIMD-optimized in-memory vector storage backend.
///
/// This backend stores embeddings in a row-major matrix format for efficient
/// batch operations. Internal optimizations include:
///
/// - **Compact storage**: Embeddings are always contiguous, no gaps
/// - **Swap-and-pop removal**: O(D) removal by swapping with last row
/// - **Stable IDs**: External IDs never change after assignment
/// - **Zero-copy queries**: Matrix passed directly to handlers
/// - **Pre-normalization**: Cosine similarity precomputes L2-normalized vectors
///
/// # Example
///
/// ```ignore
/// use rho_vectordb::InMemoryBackend;
/// use rho_vectordb::VectorBackend;
/// use rho_vectordb::SimilarityMetric;
///
/// let mut backend = InMemoryBackend::new(384);
/// let id = backend.store(&vec![0.1; 384]).unwrap();
/// let results = backend.find_similar(&vec![0.1; 384], SimilarityMetric::Cosine, 0.8, Some(10)).unwrap();
/// ```
#[derive(Clone)]
pub struct InMemoryBackend {
    /// Embedding matrix: always compact, no gaps (count × dimensions)
    /// When `index_config.pre_normalize` is true, this stores L2-normalized embeddings.
    /// Otherwise, this stores raw embeddings.
    embeddings: Array2<f32>,
    /// Dimensionality of embedding vectors
    dimensions: usize,
    /// Number of stored embeddings (== valid rows in embeddings matrix)
    count: usize,
    /// Maps external ID → row index in embeddings matrix
    id_to_row: HashMap<usize, usize>,
    /// Maps row index → external ID (for result translation)
    row_to_id: Vec<usize>,
    /// Next ID to assign for new embeddings
    next_id: usize,
    /// Handler registry for similarity metrics and ranking functions
    handler_registry: FunctionHandlerRegistry,

    // =========================================================================
    // Index Configuration and Storage
    // =========================================================================

    /// Index optimization configuration.
    index_config: IndexConfig,

    /// Cached squared L2 norms: ||e_i||² (when `index_config.cache_norms` is true).
    /// Enables fast Euclidean distance: ||a-b||² = ||a||² + ||b||² - 2(a·b)
    squared_norms: Option<Vec<f32>>,

    /// Packed binary embeddings (when `index_config.pack_binary` is true).
    /// Each embedding is binarized (threshold 0.5) and packed into u64s for SIMD popcnt.
    packed_binary: Option<Vec<Vec<u64>>>,
}

impl InMemoryBackend {
    /// Create a new in-memory backend with the specified embedding dimensions.
    ///
    /// The backend is initialized with default similarity and ranking handlers,
    /// and pre-normalization enabled for backwards compatibility.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - The dimensionality of embedding vectors
    pub fn new(dimensions: usize) -> Self {
        // Default: pre_normalize=true for backwards compatibility
        let mut index_config = IndexConfig::default();
        index_config.pre_normalize = true;

        InMemoryBackend {
            embeddings: Array2::zeros((0, dimensions)),
            dimensions,
            count: 0,
            id_to_row: HashMap::new(),
            row_to_id: Vec::new(),
            next_id: 0,
            handler_registry: FunctionHandlerRegistry::with_defaults(),
            index_config,
            squared_norms: None,
            packed_binary: None,
        }
    }

    /// Create a new in-memory backend with pre-allocated capacity.
    ///
    /// This can reduce allocations when the approximate number of embeddings
    /// is known in advance. The backend is initialized with default handlers
    /// and pre-normalization enabled for backwards compatibility.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - The dimensionality of embedding vectors
    /// * `capacity` - Initial capacity for embeddings
    pub fn with_capacity(dimensions: usize, capacity: usize) -> Self {
        // Default: pre_normalize=true for backwards compatibility
        let mut index_config = IndexConfig::default();
        index_config.pre_normalize = true;

        InMemoryBackend {
            embeddings: Array2::zeros((capacity, dimensions)),
            dimensions,
            count: 0,
            id_to_row: HashMap::with_capacity(capacity),
            row_to_id: Vec::with_capacity(capacity),
            next_id: 0,
            handler_registry: FunctionHandlerRegistry::with_defaults(),
            index_config,
            squared_norms: None,
            packed_binary: None,
        }
    }

    /// Create a new in-memory backend with a custom handler registry.
    ///
    /// Use this to provide custom similarity metrics or ranking functions.
    /// Pre-normalization is enabled for backwards compatibility.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - The dimensionality of embedding vectors
    /// * `handler_registry` - Custom handler registry
    pub fn with_handlers(dimensions: usize, handler_registry: FunctionHandlerRegistry) -> Self {
        // Default: pre_normalize=true for backwards compatibility
        let mut index_config = IndexConfig::default();
        index_config.pre_normalize = true;

        InMemoryBackend {
            embeddings: Array2::zeros((0, dimensions)),
            dimensions,
            count: 0,
            id_to_row: HashMap::new(),
            row_to_id: Vec::new(),
            next_id: 0,
            handler_registry,
            index_config,
            squared_norms: None,
            packed_binary: None,
        }
    }

    /// Create a new in-memory backend with a custom index configuration.
    ///
    /// This allows configuring which index optimizations are enabled:
    /// - `pre_normalize`: L2-normalize embeddings for fast cosine similarity
    /// - `cache_norms`: Cache squared L2 norms for fast Euclidean distance
    /// - `pack_binary`: Pack binarized embeddings for fast Hamming/Jaccard
    ///
    /// # Arguments
    ///
    /// * `dimensions` - The dimensionality of embedding vectors
    /// * `index_config` - Index optimization configuration
    pub fn with_index_config(dimensions: usize, index_config: IndexConfig) -> Self {
        let squared_norms = if index_config.cache_norms {
            Some(Vec::new())
        } else {
            None
        };
        let packed_binary = if index_config.pack_binary {
            Some(Vec::new())
        } else {
            None
        };

        InMemoryBackend {
            embeddings: Array2::zeros((0, dimensions)),
            dimensions,
            count: 0,
            id_to_row: HashMap::new(),
            row_to_id: Vec::new(),
            next_id: 0,
            handler_registry: FunctionHandlerRegistry::with_defaults(),
            index_config,
            squared_norms,
            packed_binary,
        }
    }

    /// Create a new in-memory backend with capacity and index configuration.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - The dimensionality of embedding vectors
    /// * `capacity` - Initial capacity for embeddings
    /// * `index_config` - Index optimization configuration
    pub fn with_capacity_and_index_config(
        dimensions: usize,
        capacity: usize,
        index_config: IndexConfig,
    ) -> Self {
        let squared_norms = if index_config.cache_norms {
            Some(Vec::with_capacity(capacity))
        } else {
            None
        };
        let packed_binary = if index_config.pack_binary {
            Some(Vec::with_capacity(capacity))
        } else {
            None
        };

        InMemoryBackend {
            embeddings: Array2::zeros((capacity, dimensions)),
            dimensions,
            count: 0,
            id_to_row: HashMap::with_capacity(capacity),
            row_to_id: Vec::with_capacity(capacity),
            next_id: 0,
            handler_registry: FunctionHandlerRegistry::with_defaults(),
            index_config,
            squared_norms,
            packed_binary,
        }
    }

    /// Append a row to the embedding matrix.
    fn append_embedding_row(&mut self, embedding: &[f32]) {
        // Zero-copy view: avoids embedding.to_vec() allocation for push case
        let row_view = ArrayView1::from(embedding);
        if self.embeddings.nrows() == 0 {
            // First row: need owned array for insert_axis
            self.embeddings = row_view.to_owned().insert_axis(Axis(0));
        } else {
            // Subsequent rows: push directly from view (no copy until push)
            self.embeddings
                .push(Axis(0), row_view)
                .expect("Embedding dimensions should match");
        }
    }

    /// L2 normalize a vector (slice).
    fn l2_normalize(&self, v: &[f32]) -> Vec<f32> {
        let norm = self.l2_norm(v);
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    /// Compute L2 norm of a vector.
    fn l2_norm(&self, v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Compute squared L2 norm of a vector.
    fn squared_l2_norm(&self, v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum()
    }

    // =========================================================================
    // ID Management
    // =========================================================================

    /// Get all valid embedding IDs.
    ///
    /// Returns a vector of all IDs that have active embeddings.
    /// This is useful for cloning backends.
    pub fn all_ids(&self) -> Vec<usize> {
        self.id_to_row.keys().copied().collect()
    }

    // =========================================================================
    // Index Accessors
    // =========================================================================

    /// Get the index configuration.
    pub fn index_config(&self) -> &IndexConfig {
        &self.index_config
    }

    /// Get the cached squared L2 norms if available.
    ///
    /// Returns `Some` if `cache_norms` is enabled, `None` otherwise.
    pub fn squared_norms(&self) -> Option<&[f32]> {
        self.squared_norms.as_deref()
    }

    /// Get the packed binary embeddings if available.
    ///
    /// Returns `Some` if `pack_binary` is enabled, `None` otherwise.
    pub fn packed_binary(&self) -> Option<&[Vec<u64>]> {
        self.packed_binary.as_deref()
    }

    /// Check if pre-normalization is enabled.
    pub fn is_pre_normalized(&self) -> bool {
        self.index_config.pre_normalize
    }

    /// Create index optimization data for a FunctionContext.
    ///
    /// This creates an `IndexOptimizationData` struct populated with the
    /// current backend's index data. The returned struct can be used to
    /// create a `FunctionContext` that enables handlers to use optimized
    /// computation paths.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rho_vectordb::handlers::{FunctionContext, IndexOptimizationData};
    /// use rho_vectordb::metrics::EmbeddingType;
    ///
    /// let backend = InMemoryBackend::with_index_config(
    ///     128,
    ///     IndexConfig::default().with_pre_normalization().with_norm_caching(),
    /// );
    ///
    /// // Create context with index data
    /// let index_data = backend.create_index_optimization_data();
    /// let context = FunctionContext::with_index_data(
    ///     0.8,
    ///     "cosine",
    ///     128,
    ///     EmbeddingType::Float,
    ///     index_data,
    /// );
    /// ```
    pub fn create_index_optimization_data(&self) -> super::super::handlers::IndexOptimizationData {
        super::super::handlers::IndexOptimizationData {
            is_pre_normalized: self.index_config.pre_normalize,
            squared_norms: self.squared_norms.clone(),
            packed_binary: self.packed_binary.clone(),
            packed_bits: self.dimensions,
        }
    }

    /// Create a FunctionContext with this backend's index optimization data.
    ///
    /// This is a convenience method that creates a fully configured
    /// `FunctionContext` suitable for passing to similarity handlers.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Default similarity threshold
    /// * `metric` - Default metric identifier
    /// * `embedding_type` - Embedding type (boolean, integer, float)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let backend = InMemoryBackend::with_index_config(
    ///     128,
    ///     IndexConfig::default().with_pre_normalization(),
    /// );
    ///
    /// let context = backend.create_function_context(0.8, "cosine", EmbeddingType::Float);
    /// let result = handler.compute(&query, &embeddings, &live_mask, 0.8, &[], &context)?;
    /// ```
    pub fn create_function_context(
        &self,
        threshold: f32,
        metric: impl Into<String>,
        embedding_type: super::super::metrics::EmbeddingType,
    ) -> FunctionContext {
        FunctionContext::with_index_data(
            threshold,
            metric,
            self.dimensions,
            embedding_type,
            self.create_index_optimization_data(),
        )
    }

    /// Get a view of the active embeddings (rows 0..count).
    ///
    /// Since the matrix is always compact, this is just a slice view - no copy.
    fn active_embeddings(&self) -> ndarray::ArrayView2<'_, f32> {
        self.embeddings.slice(s![..self.count, ..])
    }

    /// Compute similarities between query and all stored embeddings.
    ///
    /// Returns an array of similarity scores, one per embedding.
    /// Since storage is always compact, no masking is needed.
    fn compute_similarities(&self, query: &[f32], metric: SimilarityMetric) -> Array1<f32> {
        if self.count == 0 {
            return Array1::zeros(0);
        }

        let active = self.active_embeddings();

        match metric {
            SimilarityMetric::Cosine => {
                // OPTIMIZATION: When pre_normalize is enabled, embeddings are already
                // L2-normalized, so cosine similarity is just a dot product.
                if self.index_config.pre_normalize {
                    // Just normalize the query and compute dot products
                    let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let query_normalized = if norm > 0.0 {
                        Array1::from_iter(query.iter().map(|x| x / norm))
                    } else {
                        ArrayView1::from(query).to_owned()
                    };
                    active.dot(&query_normalized)
                } else {
                    // Standard path: compute norm from slice, create normalized array
                    let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let query_normalized = if norm > 0.0 {
                        Array1::from_iter(query.iter().map(|x| x / norm))
                    } else {
                        ArrayView1::from(query).to_owned()
                    };
                    active.dot(&query_normalized)
                }
            }
            SimilarityMetric::DotProduct => {
                // Zero-copy view: dot product accepts ArrayView
                let query_view = ArrayView1::from(query);
                active.dot(&query_view)
            }
            SimilarityMetric::Euclidean => {
                // OPTIMIZATION: When squared_norms is cached, use the identity:
                // ||a-b||² = ||a||² + ||b||² - 2(a·b)
                if let Some(ref squared_norms) = self.squared_norms {
                    if squared_norms.len() >= self.count {
                        let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();
                        let query_view = ArrayView1::from(query);
                        let dot_products = active.dot(&query_view);

                        // ||a-b||² = ||a||² + ||b||² - 2(a·b)
                        let mut similarities = Array1::zeros(self.count);
                        for i in 0..self.count {
                            let dist_sq = squared_norms[i] + query_norm_sq - 2.0 * dot_products[i];
                            // Clamp to avoid negative due to floating point errors
                            let dist = dist_sq.max(0.0).sqrt();
                            similarities[i] = 1.0 / (1.0 + dist);
                        }
                        return similarities;
                    }
                }

                // Standard path: row-by-row subtraction
                let query_view = ArrayView1::from(query);
                let mut similarities = Array1::zeros(self.count);
                for (i, row) in active.rows().into_iter().enumerate() {
                    let diff = &row - &query_view;
                    let dist = diff.mapv(|x| x * x).sum().sqrt();
                    similarities[i] = 1.0 / (1.0 + dist);
                }
                similarities
            }
            SimilarityMetric::Manhattan => {
                // Zero-copy view: subtraction works with ArrayView
                let query_view = ArrayView1::from(query);
                let mut similarities = Array1::zeros(self.count);
                for (i, row) in active.rows().into_iter().enumerate() {
                    let diff = &row - &query_view;
                    let dist = diff.mapv(|x| x.abs()).sum();
                    similarities[i] = 1.0 / (1.0 + dist);
                }
                similarities
            }
            SimilarityMetric::Hamming => {
                // OPTIMIZATION: When packed_binary is available, use hardware popcnt
                if let Some(ref packed) = self.packed_binary {
                    if packed.len() >= self.count {
                        let query_packed = pack_to_binary(query);
                        let bits = self.dimensions as f32;
                        let mut similarities = Array1::zeros(self.count);
                        for i in 0..self.count {
                            let dist = hamming_distance_packed(&query_packed, &packed[i]);
                            similarities[i] = 1.0 - (dist as f32 / bits);
                        }
                        return similarities;
                    }
                }

                // Standard path: float-based computation
                let len = query.len() as f32;
                let mut similarities = Array1::zeros(self.count);
                for (i, row) in active.rows().into_iter().enumerate() {
                    let mismatches = row
                        .iter()
                        .zip(query.iter())
                        .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
                        .count() as f32;
                    similarities[i] = 1.0 - (mismatches / len);
                }
                similarities
            }
            SimilarityMetric::Jaccard => {
                // OPTIMIZATION: When packed_binary is available, use hardware popcnt
                if let Some(ref packed) = self.packed_binary {
                    if packed.len() >= self.count {
                        let query_packed = pack_to_binary(query);
                        let mut similarities = Array1::zeros(self.count);
                        for i in 0..self.count {
                            similarities[i] =
                                jaccard_similarity_packed(&query_packed, &packed[i]);
                        }
                        return similarities;
                    }
                }

                // Standard path: float-based computation
                let mut similarities = Array1::zeros(self.count);
                for (i, row) in active.rows().into_iter().enumerate() {
                    let (intersection, union) =
                        row.iter()
                            .zip(query.iter())
                            .fold((0usize, 0usize), |(int, uni), (a, b)| {
                                let a_set = *a > 0.5;
                                let b_set = *b > 0.5;
                                (
                                    int + (a_set && b_set) as usize,
                                    uni + (a_set || b_set) as usize,
                                )
                            });
                    similarities[i] = if union == 0 {
                        1.0
                    } else {
                        intersection as f32 / union as f32
                    };
                }
                similarities
            }
        }
    }

    /// Find top-K similar embeddings using a min-heap.
    ///
    /// Returns (index, similarity) pairs sorted by descending similarity.
    /// Since storage uses swap-and-pop defragmentation, all entries are live.
    fn find_top_k(&self, similarities: &Array1<f32>, threshold: f32, limit: usize) -> Vec<(usize, f32)> {
        // Min-heap top-K selection: O(n log K) time, O(K) space
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(limit);

        for (index, &similarity) in similarities.iter().enumerate() {
            // All entries are live due to swap-and-pop compaction
            if similarity >= threshold {
                if heap.len() < limit {
                    heap.push(TopKEntry { similarity, index });
                } else if let Some(min_entry) = heap.peek() {
                    if similarity > min_entry.similarity {
                        heap.pop();
                        heap.push(TopKEntry { similarity, index });
                    }
                }
            }
        }

        if heap.is_empty() {
            return vec![];
        }

        // Extract and sort by descending similarity
        let mut results: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|e| (e.index, e.similarity))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        results
    }
}

impl VectorBackend for InMemoryBackend {
    type Id = usize;

    fn store(&mut self, embedding: &[f32]) -> Result<Self::Id, VectorDBError> {
        if embedding.len() != self.dimensions {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimensions,
                actual: embedding.len(),
            });
        }

        // Determine what to store in the main embeddings matrix
        let stored_embedding = if self.index_config.pre_normalize {
            // Pre-normalize for cosine similarity optimization
            self.l2_normalize(embedding)
        } else {
            embedding.to_vec()
        };

        // Assign a new stable ID
        let id = self.next_id;
        self.next_id += 1;

        // Store at the next available row
        let row_idx = self.count;
        if row_idx < self.embeddings.nrows() {
            // Reuse existing row (matrix has capacity from previous removes)
            let row = Array1::from_vec(stored_embedding);
            self.embeddings.row_mut(row_idx).assign(&row);
        } else {
            // Append new row (matrix needs to grow)
            self.append_embedding_row(&stored_embedding);
        }

        // Update index data if enabled
        // Compute values before mutable borrow to satisfy borrow checker
        let sq_norm = if self.squared_norms.is_some() {
            Some(self.squared_l2_norm(embedding))
        } else {
            None
        };
        let binary = if self.packed_binary.is_some() {
            Some(pack_to_binary(embedding))
        } else {
            None
        };

        if let (Some(ref mut norms), Some(sq_norm)) = (&mut self.squared_norms, sq_norm) {
            // Cache the squared L2 norm of the ORIGINAL embedding (not normalized)
            if row_idx < norms.len() {
                norms[row_idx] = sq_norm;
            } else {
                norms.push(sq_norm);
            }
        }

        if let (Some(ref mut packed), Some(binary)) = (&mut self.packed_binary, binary) {
            // Pack the ORIGINAL embedding into binary format
            if row_idx < packed.len() {
                packed[row_idx] = binary;
            } else {
                packed.push(binary);
            }
        }

        // Update bidirectional mappings
        self.id_to_row.insert(id, row_idx);
        self.row_to_id.push(id);
        self.count += 1;

        Ok(id)
    }

    fn get(&self, id: &Self::Id) -> Option<Vec<f32>> {
        // Look up row index from ID
        let &row_idx = self.id_to_row.get(id)?;
        Some(self.embeddings.row(row_idx).to_vec())
    }

    fn remove(&mut self, id: &Self::Id) -> bool {
        let Some(&row_idx) = self.id_to_row.get(id) else {
            return false;
        };

        let last_row = self.count - 1;

        if row_idx != last_row {
            // Swap with last row to maintain contiguity
            // Copy last row data to the deleted position
            let last_row_data = self.embeddings.row(last_row).to_owned();
            self.embeddings.row_mut(row_idx).assign(&last_row_data);

            // Swap index data as well
            if let Some(ref mut norms) = self.squared_norms {
                if last_row < norms.len() && row_idx < norms.len() {
                    norms[row_idx] = norms[last_row];
                }
            }
            if let Some(ref mut packed) = self.packed_binary {
                if last_row < packed.len() && row_idx < packed.len() {
                    packed.swap(row_idx, last_row);
                }
            }

            // Update mappings for the swapped element
            let swapped_id = self.row_to_id[last_row];
            self.id_to_row.insert(swapped_id, row_idx);
            self.row_to_id[row_idx] = swapped_id;
        }

        // Remove the deleted entry from mappings and index data
        self.id_to_row.remove(id);
        self.row_to_id.pop();
        if let Some(ref mut norms) = self.squared_norms {
            norms.pop();
        }
        if let Some(ref mut packed) = self.packed_binary {
            packed.pop();
        }
        self.count -= 1;

        true
    }

    fn find_similar(
        &self,
        query: &[f32],
        metric: SimilarityMetric,
        threshold: f32,
        limit: Option<usize>,
    ) -> Result<Vec<(Self::Id, f32)>, VectorDBError> {
        if query.len() != self.dimensions {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimensions,
                actual: query.len(),
            });
        }

        if self.count == 0 {
            return Ok(vec![]);
        }

        // Clamp threshold to [EPSILON, 1.0]
        let threshold = threshold.clamp(f32::EPSILON, 1.0);
        let similarities = self.compute_similarities(query, metric);

        // Get row-based results
        let row_results = match limit {
            Some(k) if k > 0 => self.find_top_k(&similarities, threshold, k),
            _ => {
                // Return all matches above threshold
                let mut matches: Vec<(usize, f32)> = similarities
                    .iter()
                    .enumerate()
                    .filter(|(_, &sim)| sim >= threshold)
                    .map(|(i, &sim)| (i, sim))
                    .collect();
                matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                matches
            }
        };

        // Translate row indices to external IDs
        let results = row_results
            .into_iter()
            .map(|(row_idx, score)| (self.row_to_id[row_idx], score))
            .collect();

        Ok(results)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn len(&self) -> usize {
        self.count
    }

    fn clear(&mut self) {
        self.embeddings = Array2::zeros((0, self.dimensions));
        self.count = 0;
        self.id_to_row.clear();
        self.row_to_id.clear();
        // Clear index data as well
        if let Some(ref mut norms) = self.squared_norms {
            norms.clear();
        }
        if let Some(ref mut packed) = self.packed_binary {
            packed.clear();
        }
        // Note: next_id is NOT reset - IDs are never reused
    }

    fn supported_metrics(&self) -> Vec<SimilarityMetric> {
        vec![
            SimilarityMetric::Cosine,
            SimilarityMetric::DotProduct,
            SimilarityMetric::Euclidean,
            SimilarityMetric::Manhattan,
            SimilarityMetric::Hamming,
            SimilarityMetric::Jaccard,
        ]
    }
}

impl HandlerBackend for InMemoryBackend {
    fn handler_registry(&self) -> &FunctionHandlerRegistry {
        &self.handler_registry
    }

    fn handler_registry_mut(&mut self) -> &mut FunctionHandlerRegistry {
        &mut self.handler_registry
    }

    fn embeddings_matrix(&self) -> &Array2<f32> {
        &self.embeddings
    }

    fn live_mask(&self) -> Array1<f32> {
        // With compact storage, all entries are live - return all 1.0s
        Array1::ones(self.count)
    }

    /// Compute similarity using a registered handler.
    ///
    /// Since storage uses swap-and-pop defragmentation (always compact), we pass
    /// the embeddings directly to the handler without any filtering overhead.
    fn compute_similarity_with_handler(
        &self,
        query: &[f32],
        metric_id: &str,
        threshold: f32,
        extra_params: &[ResolvedArg],
        context: &FunctionContext,
    ) -> Result<super::super::handlers::SimilarityResult, VectorDBError> {
        // Validate dimensions
        if query.len() != self.dimensions {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimensions,
                actual: query.len(),
            });
        }

        // Look up the handler FIRST to validate metric before checking embeddings
        let handler = self.handler_registry().get_similarity(metric_id).ok_or_else(|| {
            VectorDBError::UnsupportedMetric(format!(
                "Unknown similarity metric: '{}'. Available: {:?}",
                metric_id,
                self.handler_registry().similarity.names()
            ))
        })?;

        // Handle empty case
        if self.count == 0 {
            return Ok(super::super::handlers::SimilarityResult::new(
                Array1::zeros(0),
                threshold,
            ));
        }

        // Get view of active embeddings (no copy - matrix is always compact)
        let active = self.active_embeddings();

        // Create dummy mask (all 1.0s) for backward compatibility with handlers
        let dummy_mask = Array1::ones(self.count);

        // Compute similarity directly on compact matrix
        handler.compute(
            query,
            &active.to_owned(),
            &dummy_mask,
            threshold,
            extra_params,
            context,
        )
    }

    fn find_similar_with_handlers(
        &self,
        query: &[f32],
        metric_id: &str,
        threshold: f32,
        ranking_id: &str,
        ranking_params: &[ResolvedArg],
        context: &FunctionContext,
    ) -> Result<Vec<(Self::Id, f32)>, VectorDBError> {
        // Compute similarity on compact matrix
        let similarity = self.compute_similarity_with_handler(query, metric_id, threshold, &[], context)?;

        // Rank results (returns row indices)
        let ranked = self.rank_with_handler(&similarity, ranking_id, ranking_params, context)?;

        // Translate row indices to external IDs
        let results = ranked
            .matches
            .into_iter()
            .map(|(row_idx, score)| (self.row_to_id[row_idx], score))
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_random_embedding(dimensions: usize) -> Vec<f32> {
        (0..dimensions).map(|i| (i as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn test_store_and_get() {
        let mut backend = InMemoryBackend::new(4);
        let embedding = vec![1.0, 2.0, 3.0, 4.0];

        let id = backend.store(&embedding).unwrap();
        let retrieved = backend.get(&id);

        assert!(retrieved.is_some());
        // Note: embeddings are L2-normalized on storage
        let norm = (1.0f32 + 4.0 + 9.0 + 16.0).sqrt();
        let expected: Vec<f32> = embedding.iter().map(|x| x / norm).collect();
        let retrieved = retrieved.unwrap();
        for (a, b) in retrieved.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut backend = InMemoryBackend::new(4);
        let wrong_embedding = vec![1.0, 2.0, 3.0]; // Only 3 dimensions

        let result = backend.store(&wrong_embedding);
        assert!(matches!(result, Err(VectorDBError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_remove() {
        let mut backend = InMemoryBackend::new(4);
        let embedding = vec![1.0, 2.0, 3.0, 4.0];

        let id = backend.store(&embedding).unwrap();
        assert_eq!(backend.len(), 1);

        let removed = backend.remove(&id);
        assert!(removed);
        assert_eq!(backend.len(), 0);

        // Should not be retrievable after removal
        assert!(backend.get(&id).is_none());
    }

    #[test]
    fn test_stable_ids_no_reuse() {
        let mut backend = InMemoryBackend::new(4);

        let id1 = backend.store(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        let id2 = backend.store(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);

        // Remove first embedding
        backend.remove(&id1);
        assert_eq!(backend.len(), 1);

        // New store should get a new ID (not reuse id1)
        let id3 = backend.store(&[0.0, 0.0, 1.0, 0.0]).unwrap();
        assert_eq!(id3, 2); // New unique ID, not reused
        assert_eq!(backend.len(), 2);

        // id2 should still be retrievable (it was moved to row 0 by swap-and-pop)
        assert!(backend.get(&id2).is_some());
        assert!(backend.get(&id1).is_none()); // id1 was removed
    }

    #[test]
    fn test_swap_and_pop_compaction() {
        let mut backend = InMemoryBackend::new(4);

        // Store 3 embeddings
        let id0 = backend.store(&[1.0, 0.0, 0.0, 0.0]).unwrap(); // row 0
        let id1 = backend.store(&[0.0, 1.0, 0.0, 0.0]).unwrap(); // row 1
        let id2 = backend.store(&[0.0, 0.0, 1.0, 0.0]).unwrap(); // row 2

        assert_eq!(backend.len(), 3);

        // Remove middle element (id1)
        // This should swap id2 into row 1, then pop
        backend.remove(&id1);
        assert_eq!(backend.len(), 2);

        // id0 and id2 should still be retrievable with correct data
        let v0 = backend.get(&id0).expect("id0 should exist");
        let v2 = backend.get(&id2).expect("id2 should exist");

        // Verify the normalized vectors are correct
        assert!((v0[0] - 1.0).abs() < 1e-6); // [1,0,0,0] normalized is still [1,0,0,0]
        assert!((v2[2] - 1.0).abs() < 1e-6); // [0,0,1,0] normalized is still [0,0,1,0]

        // id1 should not exist
        assert!(backend.get(&id1).is_none());
    }

    #[test]
    fn test_remove_last_element() {
        let mut backend = InMemoryBackend::new(4);

        let id0 = backend.store(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        let id1 = backend.store(&[0.0, 1.0, 0.0, 0.0]).unwrap();

        // Remove last element (no swap needed)
        backend.remove(&id1);
        assert_eq!(backend.len(), 1);
        assert!(backend.get(&id0).is_some());
        assert!(backend.get(&id1).is_none());
    }

    #[test]
    fn test_remove_only_element() {
        let mut backend = InMemoryBackend::new(4);

        let id = backend.store(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(backend.len(), 1);

        backend.remove(&id);
        assert_eq!(backend.len(), 0);
        assert!(backend.is_empty());
        assert!(backend.get(&id).is_none());
    }

    #[test]
    fn test_find_similar_cosine() {
        let mut backend = InMemoryBackend::new(4);

        // Store normalized vectors for predictable results
        backend.store(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        backend.store(&[0.9, 0.1, 0.0, 0.0]).unwrap();
        backend.store(&[0.0, 1.0, 0.0, 0.0]).unwrap();

        // Query similar to first two
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = backend
            .find_similar(&query, SimilarityMetric::Cosine, 0.5, Some(2))
            .unwrap();

        assert_eq!(results.len(), 2);
        // First result should be most similar
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn test_find_similar_with_limit() {
        let mut backend = InMemoryBackend::new(4);

        for _ in 0..10 {
            backend.store(&create_random_embedding(4)).unwrap();
        }

        let query = create_random_embedding(4);
        let results = backend
            .find_similar(&query, SimilarityMetric::Cosine, 0.0, Some(3))
            .unwrap();

        assert!(results.len() <= 3);
    }

    #[test]
    fn test_clear() {
        let mut backend = InMemoryBackend::new(4);

        for _ in 0..5 {
            backend.store(&create_random_embedding(4)).unwrap();
        }
        assert_eq!(backend.len(), 5);

        backend.clear();
        assert_eq!(backend.len(), 0);
        assert!(backend.is_empty());
    }

    #[test]
    fn test_supported_metrics() {
        let backend = InMemoryBackend::new(4);
        let metrics = backend.supported_metrics();

        assert!(metrics.contains(&SimilarityMetric::Cosine));
        assert!(metrics.contains(&SimilarityMetric::DotProduct));
        assert!(metrics.contains(&SimilarityMetric::Euclidean));
        assert!(metrics.contains(&SimilarityMetric::Manhattan));
        assert!(metrics.contains(&SimilarityMetric::Hamming));
        assert!(metrics.contains(&SimilarityMetric::Jaccard));
    }

    // ========================================================================
    // HandlerBackend Tests
    // ========================================================================

    use super::super::super::handlers::ResolvedArg;
    use super::super::super::metrics::EmbeddingType;

    #[test]
    fn test_handler_backend_compute_similarity() {
        let mut backend = InMemoryBackend::new(4);

        // Store some vectors
        backend.store(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        backend.store(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        backend.store(&[0.707, 0.707, 0.0, 0.0]).unwrap();

        let context = FunctionContext::new(0.5, "cosine", 4, EmbeddingType::Float);
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let result = backend
            .compute_similarity_with_handler(&query, "cosine", 0.5, &[], &context)
            .expect("should compute similarity");

        assert_eq!(result.scores.len(), 3);
        // First vector is identical to query (modulo normalization)
        assert!(result.scores[0] > 0.99);
        // Second vector is orthogonal
        assert!(result.scores[1] < 0.01);
        // Third vector is at 45 degrees
        assert!((result.scores[2] - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_handler_backend_rank_with_handler() {
        let mut backend = InMemoryBackend::new(4);

        backend.store(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        backend.store(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        backend.store(&[0.707, 0.707, 0.0, 0.0]).unwrap();

        let context = FunctionContext::new(0.5, "cosine", 4, EmbeddingType::Float);
        let query = vec![1.0, 0.0, 0.0, 0.0];

        // Compute similarity first
        let similarity = backend
            .compute_similarity_with_handler(&query, "cosine", 0.5, &[], &context)
            .expect("should compute similarity");

        // Rank with top-1
        let ranked = backend
            .rank_with_handler(&similarity, "topk", &[ResolvedArg::Integer(1)], &context)
            .expect("should rank");

        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked.matches[0].0, 0); // First vector is most similar
    }

    #[test]
    fn test_handler_backend_find_similar_with_handlers() {
        let mut backend = InMemoryBackend::new(4);

        backend.store(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        backend.store(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        backend.store(&[0.707, 0.707, 0.0, 0.0]).unwrap();

        let context = FunctionContext::new(0.5, "cosine", 4, EmbeddingType::Float);
        let query = vec![1.0, 0.0, 0.0, 0.0];

        // Find similar with cosine + top-2
        let results = backend
            .find_similar_with_handlers(&query, "cosine", 0.5, "topk", &[ResolvedArg::Integer(2)], &context)
            .expect("should find similar");

        assert_eq!(results.len(), 2);
        // Results should be sorted by score descending
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn test_handler_backend_unknown_metric() {
        let backend = InMemoryBackend::new(4);
        let context = FunctionContext::new(0.5, "cosine", 4, EmbeddingType::Float);
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let result = backend.compute_similarity_with_handler(&query, "unknown_metric", 0.5, &[], &context);
        assert!(result.is_err());
    }

    #[test]
    fn test_handler_backend_handler_registry_access() {
        let backend = InMemoryBackend::new(4);

        // Verify default handlers are registered
        assert!(backend.handler_registry().get_similarity("cosine").is_some());
        assert!(backend.handler_registry().get_similarity("cos").is_some());
        assert!(backend.handler_registry().get_ranking("topk").is_some());
        assert!(backend.handler_registry().get_ranking("all").is_some());
    }

    // ========================================================================
    // Index Configuration Tests
    // ========================================================================

    use super::super::super::metrics::IndexConfig;

    #[test]
    fn test_with_index_config_default() {
        let backend = InMemoryBackend::new(4);
        // Default should have pre_normalize enabled for backwards compatibility
        assert!(backend.is_pre_normalized());
        assert!(backend.squared_norms().is_none());
        assert!(backend.packed_binary().is_none());
    }

    #[test]
    fn test_with_index_config_norm_caching() {
        let config = IndexConfig {
            pre_normalize: true,
            cache_norms: true,
            pack_binary: false,
            hnsw: None,
            scalar_quantization: None,
        };
        let mut backend = InMemoryBackend::with_index_config(4, config);

        // Store some embeddings
        backend.store(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        backend.store(&[0.5, 0.5, 0.5, 0.5]).unwrap();

        // Should have norm cache
        let norms = backend.squared_norms().expect("should have norms");
        assert_eq!(norms.len(), 2);

        // First embedding: 1² + 2² + 3² + 4² = 30
        assert!((norms[0] - 30.0).abs() < 0.001);
        // Second embedding: 0.5² * 4 = 1.0
        assert!((norms[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_with_index_config_binary_packing() {
        let config = IndexConfig {
            pre_normalize: false,
            cache_norms: false,
            pack_binary: true,
            hnsw: None,
            scalar_quantization: None,
        };
        let mut backend = InMemoryBackend::with_index_config(64, config);

        // Store binary-like embedding (alternating high/low values)
        let mut embedding = vec![0.0f32; 64];
        for i in 0..64 {
            embedding[i] = if i % 2 == 0 { 0.9 } else { 0.1 };
        }
        backend.store(&embedding).unwrap();

        // Should have packed binary
        let packed = backend.packed_binary().expect("should have packed");
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].len(), 1); // 64 bits fits in one u64

        // Check the pattern: even bits set (indices 0, 2, 4, ...)
        // Binary: ...101010101010 = 0x5555555555555555
        assert_eq!(packed[0][0], 0x5555555555555555u64);
    }

    #[test]
    fn test_index_data_maintained_on_remove() {
        let config = IndexConfig {
            pre_normalize: true,
            cache_norms: true,
            pack_binary: false,
            hnsw: None,
            scalar_quantization: None,
        };
        let mut backend = InMemoryBackend::with_index_config(4, config);

        // Store 3 embeddings with distinct norms
        let id0 = backend.store(&[1.0, 0.0, 0.0, 0.0]).unwrap(); // norm² = 1
        let _id1 = backend.store(&[2.0, 0.0, 0.0, 0.0]).unwrap(); // norm² = 4
        let id2 = backend.store(&[3.0, 0.0, 0.0, 0.0]).unwrap(); // norm² = 9

        assert_eq!(backend.len(), 3);
        let norms = backend.squared_norms().expect("should have norms");
        assert_eq!(norms.len(), 3);

        // Remove middle element (triggers swap-and-pop)
        backend.remove(&_id1);

        assert_eq!(backend.len(), 2);
        let norms = backend.squared_norms().expect("should have norms");
        assert_eq!(norms.len(), 2);

        // id0 and id2 should still be correct
        assert!(backend.get(&id0).is_some());
        assert!(backend.get(&id2).is_some());
    }

    #[test]
    fn test_index_data_cleared_on_clear() {
        let config = IndexConfig {
            pre_normalize: true,
            cache_norms: true,
            pack_binary: true,
            hnsw: None,
            scalar_quantization: None,
        };
        let mut backend = InMemoryBackend::with_index_config(64, config);

        // Store some embeddings
        backend.store(&[0.5; 64]).unwrap();
        backend.store(&[0.9; 64]).unwrap();

        assert_eq!(backend.len(), 2);
        assert_eq!(backend.squared_norms().unwrap().len(), 2);
        assert_eq!(backend.packed_binary().unwrap().len(), 2);

        // Clear
        backend.clear();

        assert_eq!(backend.len(), 0);
        assert_eq!(backend.squared_norms().unwrap().len(), 0);
        assert_eq!(backend.packed_binary().unwrap().len(), 0);
    }

    #[test]
    fn test_hamming_distance_packed() {
        // All zeros vs all zeros
        let a = vec![0u64];
        let b = vec![0u64];
        assert_eq!(hamming_distance_packed(&a, &b), 0);

        // All zeros vs all ones (64 bits)
        let a = vec![0u64];
        let b = vec![u64::MAX];
        assert_eq!(hamming_distance_packed(&a, &b), 64);

        // Alternating pattern: 50% difference
        let a = vec![0x5555555555555555u64]; // 01010101...
        let b = vec![0xAAAAAAAAAAAAAAAAu64]; // 10101010...
        assert_eq!(hamming_distance_packed(&a, &b), 64);
    }

    #[test]
    fn test_jaccard_similarity_packed() {
        // Identical
        let a = vec![0xFFu64];
        let b = vec![0xFFu64];
        assert!((jaccard_similarity_packed(&a, &b) - 1.0).abs() < 0.001);

        // No overlap (but both have bits set)
        let a = vec![0x0Fu64]; // lower 4 bits
        let b = vec![0xF0u64]; // upper 4 bits of byte 0
        // Intersection: 0, Union: 8 bits
        assert!((jaccard_similarity_packed(&a, &b) - 0.0).abs() < 0.001);

        // 50% overlap
        let a = vec![0x0Fu64]; // bits 0-3
        let b = vec![0x3Cu64]; // bits 2-5
        // Intersection: bits 2-3 = 2 bits
        // Union: bits 0-5 = 6 bits
        assert!((jaccard_similarity_packed(&a, &b) - (2.0 / 6.0)).abs() < 0.001);

        // Both empty
        let a = vec![0u64];
        let b = vec![0u64];
        assert!((jaccard_similarity_packed(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_with_capacity_and_index_config() {
        let config = IndexConfig {
            pre_normalize: true,
            cache_norms: true,
            pack_binary: true,
            hnsw: None,
            scalar_quantization: None,
        };
        let mut backend = InMemoryBackend::with_capacity_and_index_config(4, 100, config);

        // Store embeddings (should work with pre-allocated capacity)
        for i in 0..50 {
            backend.store(&[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }

        assert_eq!(backend.len(), 50);
        assert_eq!(backend.squared_norms().unwrap().len(), 50);
        assert_eq!(backend.packed_binary().unwrap().len(), 50);
    }

    #[test]
    fn test_no_normalization_when_disabled() {
        let config = IndexConfig {
            pre_normalize: false,
            cache_norms: false,
            pack_binary: false,
            hnsw: None,
            scalar_quantization: None,
        };
        let mut backend = InMemoryBackend::with_index_config(4, config);

        // Store a non-unit vector
        let embedding = vec![2.0, 0.0, 0.0, 0.0];
        let id = backend.store(&embedding).unwrap();

        // Retrieved embedding should be raw (not normalized)
        let retrieved = backend.get(&id).unwrap();
        assert!((retrieved[0] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_normalization_when_enabled() {
        // Default new() has pre_normalize enabled
        let mut backend = InMemoryBackend::new(4);

        // Store a non-unit vector
        let embedding = vec![2.0, 0.0, 0.0, 0.0];
        let id = backend.store(&embedding).unwrap();

        // Retrieved embedding should be normalized (unit length)
        let retrieved = backend.get(&id).unwrap();
        assert!((retrieved[0] - 1.0).abs() < 0.001);
    }

    // ========================================================================
    // FunctionContext Creation Tests
    // ========================================================================

    #[test]
    fn test_create_index_optimization_data_default() {
        let backend = InMemoryBackend::new(4);
        let index_data = backend.create_index_optimization_data();

        // Default has pre_normalize enabled
        assert!(index_data.is_pre_normalized);
        assert!(index_data.squared_norms.is_none());
        assert!(index_data.packed_binary.is_none());
        assert_eq!(index_data.packed_bits, 4);
    }

    #[test]
    fn test_create_index_optimization_data_with_all_indices() {
        let config = IndexConfig {
            pre_normalize: true,
            cache_norms: true,
            pack_binary: true,
            hnsw: None,
            scalar_quantization: None,
        };
        let mut backend = InMemoryBackend::with_index_config(64, config);

        // Store some embeddings
        backend.store(&[0.5; 64]).unwrap();
        backend.store(&[0.9; 64]).unwrap();

        let index_data = backend.create_index_optimization_data();

        assert!(index_data.is_pre_normalized);
        assert!(index_data.squared_norms.is_some());
        assert_eq!(index_data.squared_norms.as_ref().unwrap().len(), 2);
        assert!(index_data.packed_binary.is_some());
        assert_eq!(index_data.packed_binary.as_ref().unwrap().len(), 2);
        assert_eq!(index_data.packed_bits, 64);
    }

    #[test]
    fn test_create_function_context() {
        let config = IndexConfig {
            pre_normalize: true,
            cache_norms: true,
            pack_binary: false,
            hnsw: None,
            scalar_quantization: None,
        };
        let mut backend = InMemoryBackend::with_index_config(128, config);
        backend.store(&[0.5; 128]).unwrap();

        let context = backend.create_function_context(0.75, "euclidean", EmbeddingType::Float);

        // Verify basic context fields
        assert_eq!(context.default_threshold, 0.75);
        assert_eq!(context.default_metric, "euclidean");
        assert_eq!(context.dimensions, 128);
        assert!(matches!(context.embedding_type, EmbeddingType::Float));

        // Verify index optimization data is populated
        assert!(context.is_pre_normalized());
        assert!(context.get_squared_norms().is_some());
        assert!(context.get_packed_binary().is_none());
        assert_eq!(context.get_packed_bits(), 128);
        assert!(context.has_index_optimizations());
    }

    #[test]
    fn test_create_function_context_no_indices() {
        let config = IndexConfig {
            pre_normalize: false,
            cache_norms: false,
            pack_binary: false,
            hnsw: None,
            scalar_quantization: None,
        };
        let backend = InMemoryBackend::with_index_config(4, config);

        let context = backend.create_function_context(0.8, "cosine", EmbeddingType::Float);

        // Verify no optimizations available
        assert!(!context.is_pre_normalized());
        assert!(context.get_squared_norms().is_none());
        assert!(context.get_packed_binary().is_none());
        // Note: has_index_optimizations returns false when all are disabled
        assert!(!context.has_index_optimizations());
    }
}
