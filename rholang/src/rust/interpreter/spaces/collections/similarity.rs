//! Similarity Collection Traits and Types
//!
//! This module provides types for similarity-based (VectorDB) operations:
//!
//! - `SimilarityCollection`: Trait for VectorDB-style collections
//! - `StoredSimilarityInfo`: Pre-extracted similarity requirements for continuations
//! - `ContinuationId`: Unique identifier for waiting continuations
//! - `SimilarityQueryMatrix`: SIMD-optimized matrix for batch similarity queries

use ndarray::{Array1, Array2, Axis};

use super::super::errors::SpaceError;
use super::super::vectordb::types::EmbeddingType;
use super::super::vectordb::in_memory::SimilarityMetric;
use super::core::DataCollection;

// ==========================================================================
// Similarity Collection Trait
// ==========================================================================

/// Trait for collections that support similarity-based matching.
///
/// Collections implementing this trait support VectorDB-style similarity
/// queries where pattern matching is based on vector similarity between
/// embedding vectors rather than structural matching.
///
/// # Design Rationale
///
/// This trait exists to:
/// 1. Enable pluggable VectorDB implementations with different backends
/// 2. Provide similarity-based `find_and_remove_most_similar_with_threshold` operation
/// 3. Enable `consume_with_similarity` in GenericRSpace to use VectorDB semantics
/// 4. Support configurable similarity metrics and embedding types
///
/// # Usage in Rholang
///
/// ```rholang
/// // Implicit threshold (uses space default)
/// for (@doc <- vectorChannel ~ [0.8, 0.2, 0.5]) { ... }
///
/// // Explicit threshold
/// for (@doc <- vectorChannel ~> 0.75 ~ [0.8, 0.2, 0.5]) { ... }
/// ```
///
/// # Specialization Points
///
/// Different implementations can optimize for various indexing strategies:
/// - Brute-force linear scan (default VectorDBDataCollection)
/// - HNSW (Hierarchical Navigable Small World graphs)
/// - FAISS (Facebook AI Similarity Search)
/// - External VectorDB proxies (Pinecone, Qdrant, Weaviate)
///
/// # Formal Correspondence
/// - VectorDB.v: similarity_collection_properties theorem
/// - GenericRSpace.v: consume_with_similarity_uses_trait theorem
pub trait SimilarityCollection<A>: DataCollection<A> {
    /// Store data with its embedding vector.
    ///
    /// This is the primary method for storing data in a VectorDB collection.
    /// The embedding is used for similarity-based matching during consume.
    ///
    /// # Arguments
    /// - `data`: The data to store (typically a Par with map structure)
    /// - `embedding`: The embedding vector (must match collection dimensions)
    ///
    /// # Returns
    /// - `Ok(())` if successful
    /// - `Err(SpaceError)` if embedding dimensions don't match
    ///
    /// # Note
    /// Implementations may apply preprocessing to embeddings (e.g., L2 normalization
    /// for cosine similarity) before storage for efficiency.
    fn put_with_embedding(&mut self, data: A, embedding: Vec<f32>) -> Result<(), SpaceError>;

    /// Find and remove the most similar element using a per-query threshold.
    ///
    /// This enables each consume pattern to specify its own similarity bound,
    /// as required by the Reifying RSpaces design document.
    ///
    /// # Arguments
    /// - `query_embedding`: The embedding vector to match against
    /// - `threshold`: Minimum similarity for this query (0.0 to 1.0)
    ///
    /// # Returns
    /// - `Some((data, similarity_score))` if found and removed
    /// - `None` if no match above threshold
    fn find_and_remove_most_similar_with_threshold(
        &mut self,
        query_embedding: &[f32],
        threshold: f32,
    ) -> Option<(A, f32)>;

    /// Peek at the most similar element without removing it.
    ///
    /// # Arguments
    /// - `query_embedding`: The embedding vector to match against
    /// - `threshold`: Minimum similarity for this query (0.0 to 1.0)
    ///
    /// # Returns
    /// - `Some((data_ref, similarity_score))` if found
    /// - `None` if no match above threshold
    fn peek_most_similar_with_threshold(
        &self,
        query_embedding: &[f32],
        threshold: f32,
    ) -> Option<(&A, f32)>;

    /// Get the default similarity threshold for this collection.
    ///
    /// This is used when the Rholang pattern uses implicit threshold syntax:
    /// `for (@x <- channel ~ query) { ... }`
    fn default_threshold(&self) -> f32;

    /// Get the embedding dimensions for this collection.
    fn embedding_dimensions(&self) -> usize;

    /// Get the expected embedding type for validation during produce.
    ///
    /// The embedding format in Rholang data must match this type:
    /// - `Boolean`: `[0, 1, 1, 0]` - binary vectors
    /// - `Integer`: `[90, 5, 10, 20]` - 0-100 scale
    /// - `Float`: `"0.9,0.05,0.1,0.2"` - comma-separated float string
    fn embedding_type(&self) -> EmbeddingType;

    /// Get the similarity metric used by this collection.
    fn similarity_metric(&self) -> SimilarityMetric;

    /// Get the list of similarity metrics supported by this backend.
    ///
    /// This is used for validation during factory construction.
    /// If a metric is requested that is not in this list, an error is returned.
    fn supported_metrics(&self) -> &[SimilarityMetric];

    /// Find top-K most similar elements above a threshold (peek, no removal).
    ///
    /// Uses efficient partial sort for O(n + K log K) complexity.
    ///
    /// # Arguments
    /// - `query_embedding`: The embedding vector to match against
    /// - `threshold`: Minimum similarity for this query (0.0 to 1.0)
    /// - `k`: Maximum number of results to return
    ///
    /// # Returns
    /// Vector of (data_ref, similarity) pairs, sorted by descending similarity.
    fn find_top_k_similar_with_threshold(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        k: usize,
    ) -> Vec<(&A, f32)>;

    /// Find and remove top-K most similar elements above a threshold.
    ///
    /// For non-persistent data, elements are removed (or tombstoned).
    /// For persistent data, clones are returned without removal.
    ///
    /// # Arguments
    /// - `query_embedding`: The embedding vector to match against
    /// - `threshold`: Minimum similarity for this query (0.0 to 1.0)
    /// - `k`: Maximum number of results to return
    ///
    /// # Returns
    /// Vector of (data, similarity) pairs, sorted by descending similarity.
    fn find_and_remove_top_k_similar_with_threshold(
        &mut self,
        query_embedding: &[f32],
        threshold: f32,
        k: usize,
    ) -> Vec<(A, f32)>;

    /// Validate that a metric is supported by this backend.
    ///
    /// # Returns
    /// - `Ok(())` if the metric is supported
    /// - `Err(SpaceError::InvalidConfiguration)` if not supported
    fn validate_metric(&self, metric: SimilarityMetric) -> Result<(), SpaceError> {
        if self.supported_metrics().contains(&metric) {
            Ok(())
        } else {
            Err(SpaceError::InvalidConfiguration {
                description: format!(
                    "Similarity metric {:?} not supported by this VectorDB backend. Supported: {:?}",
                    metric,
                    self.supported_metrics()
                ),
            })
        }
    }
}

// ==========================================================================
// Stored Similarity Info for Continuations
// ==========================================================================

/// Stored similarity pattern info for continuation matching.
///
/// This struct holds the pre-extracted similarity requirements for a continuation.
/// When new data arrives via `produce()`, the system checks if the data's embedding
/// meets the similarity threshold before firing the continuation.
///
/// # Fields
///
/// - `embeddings`: For each channel in the join, either `None` (no similarity required)
///   or `Some((embedding, threshold))` where embedding is the query vector and
///   threshold is the minimum similarity score (0.0 to 1.0).
///
/// # Example
///
/// For a query like `for (@doc <- docs ~ "0.9,0.1,0.1,0.15")`:
/// - `embeddings[0]` = `Some(([0.9, 0.1, 0.1, 0.15], 0.5, None, None))` (50% threshold, default metric, no top-K)
///
/// For a query like `for (@docs <- docs ~ sim("cos", "0.8") ~ rank("topk", 5) ~ "0.9,0.1,0.1,0.15")`:
/// - `embeddings[0]` = `Some(([0.9, 0.1, 0.1, 0.15], 0.8, Some(Cosine), Some(5)))` (80% threshold, cosine metric, top-5)
#[derive(Clone, Debug, Default)]
pub struct StoredSimilarityInfo {
    /// Pre-extracted embeddings, thresholds, metrics, and optional top-K for each channel.
    /// None means no similarity requirement for that channel.
    /// Tuple is (embedding, threshold, optional_metric, optional_top_k).
    pub embeddings: Vec<Option<(Vec<f32>, f32, Option<SimilarityMetric>, Option<usize>)>>,
}

impl StoredSimilarityInfo {
    /// Create a new StoredSimilarityInfo with the given embeddings.
    pub fn new(embeddings: Vec<Option<(Vec<f32>, f32, Option<SimilarityMetric>, Option<usize>)>>) -> Self {
        Self { embeddings }
    }

    /// Check if any channel has a similarity requirement.
    pub fn has_similarity(&self) -> bool {
        self.embeddings.iter().any(|e| e.is_some())
    }

    /// Get the number of channels.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

// ==========================================================================
// Similarity Query Matrix
// ==========================================================================

/// Unique identifier for a waiting continuation in the query matrix.
///
/// This is used to map from query matrix rows back to the actual continuation
/// storage (ContinuationCollection) and channel information.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ContinuationId(pub usize);

/// Matrix of normalized query embeddings for similarity-based continuations.
///
/// This structure enables SIMD-optimized batch similarity computation. When
/// `produce()` stores new data, it can compute similarities against all waiting
/// queries in a single matrix-vector multiplication instead of iterating
/// through continuations one-by-one.
///
/// # Architecture
///
/// The query matrix stores pre-normalized query embeddings as rows. When new
/// data arrives:
/// 1. The data's embedding is normalized and stored in VectorDB
/// 2. `find_matching_queries()` computes `query_matrix @ data_embedding`
/// 3. Results are compared against per-query thresholds
/// 4. Matching continuations are fired
///
/// # Performance
///
/// Matrix-vector multiplication with `ndarray` leverages SIMD instructions,
/// providing O(n×d/8) effective complexity with AVX compared to O(n×d)
/// scalar operations.
///
/// # Memory Layout
///
/// - Row-major storage for cache efficiency in matrix-vector products
/// - Compact representation: one row per waiting query
/// - Sparse removal via swap-remove to maintain dense storage
#[derive(Clone, Debug)]
pub struct SimilarityQueryMatrix {
    /// Normalized query embeddings (n_queries × dimensions).
    /// Each row is L2-normalized for cosine similarity = dot product.
    embeddings: Array2<f32>,

    /// Threshold for each query (length = n_queries).
    /// A query matches if similarity >= threshold.
    thresholds: Vec<f32>,

    /// Index mapping: row index → continuation ID.
    /// Used to fire the correct continuation when a query matches.
    continuation_ids: Vec<ContinuationId>,

    /// Channel index within the join pattern that this query applies to.
    /// For single-channel queries, this is always 0.
    channel_indices: Vec<usize>,

    /// Whether the continuation is persistent (remains after firing).
    persist_flags: Vec<bool>,

    /// Dimensionality of embedding vectors.
    dimensions: usize,
}

impl SimilarityQueryMatrix {
    /// Create a new empty query matrix with the given dimensionality.
    pub fn new(dimensions: usize) -> Self {
        Self {
            embeddings: Array2::zeros((0, dimensions)),
            thresholds: Vec::new(),
            continuation_ids: Vec::new(),
            channel_indices: Vec::new(),
            persist_flags: Vec::new(),
            dimensions,
        }
    }

    /// Add a new query embedding when `consume_with_similarity` stores a continuation.
    ///
    /// The embedding is L2-normalized before storage to enable cosine similarity
    /// computation via dot product.
    ///
    /// # Arguments
    /// - `query_embedding`: The raw query embedding (will be normalized)
    /// - `threshold`: Minimum similarity score for a match
    /// - `cont_id`: ID of the continuation to fire when matched
    /// - `channel_idx`: Index of the channel within the join pattern
    /// - `persist`: Whether the continuation persists after firing
    ///
    /// # Returns
    /// - `Ok(row_index)` on success
    /// - `Err` if embedding dimensions don't match
    pub fn add_query(
        &mut self,
        query_embedding: &[f32],
        threshold: f32,
        cont_id: ContinuationId,
        channel_idx: usize,
        persist: bool,
    ) -> Result<usize, SpaceError> {
        if query_embedding.len() != self.dimensions {
            return Err(SpaceError::InvalidConfiguration {
                description: format!(
                    "Query embedding dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    query_embedding.len()
                ),
            });
        }

        // L2-normalize the query embedding
        let normalized = self.l2_normalize(query_embedding);
        let row = Array1::from_vec(normalized);

        // Append to embedding matrix
        let row_idx = self.embeddings.nrows();
        if row_idx == 0 {
            self.embeddings = row.insert_axis(Axis(0));
        } else {
            self.embeddings
                .push(Axis(0), row.view())
                .expect("Query embedding dimensions should match");
        }

        self.thresholds.push(threshold.clamp(0.0, 1.0));
        self.continuation_ids.push(cont_id);
        self.channel_indices.push(channel_idx);
        self.persist_flags.push(persist);

        Ok(row_idx)
    }

    /// Remove a query when its continuation fires (non-persistent) or is cancelled.
    ///
    /// Uses swap-remove to maintain dense matrix storage.
    ///
    /// # Returns
    /// `true` if a query was removed, `false` if the continuation ID wasn't found.
    pub fn remove_query(&mut self, cont_id: ContinuationId) -> bool {
        if let Some(pos) = self.continuation_ids.iter().position(|&id| id == cont_id) {
            self.swap_remove_row(pos);
            true
        } else {
            false
        }
    }

    /// Find all queries that match the given data embedding.
    ///
    /// This is the core SIMD-optimized operation: computes similarities for all
    /// queries in a single matrix-vector multiplication.
    ///
    /// # Arguments
    /// - `data_embedding`: Normalized embedding of the newly stored data
    ///
    /// # Returns
    /// Vector of `(continuation_id, channel_idx, similarity_score, persist)` for all
    /// queries where `similarity >= threshold`.
    pub fn find_matching_queries(
        &self,
        data_embedding: &[f32],
    ) -> Vec<(ContinuationId, usize, f32, bool)> {
        if self.embeddings.nrows() == 0 || data_embedding.len() != self.dimensions {
            return Vec::new();
        }

        // Compute similarities = query_matrix @ data_embedding (SIMD-optimized)
        let data_vec = Array1::from_vec(data_embedding.to_vec());
        let similarities = self.embeddings.dot(&data_vec);

        // Filter by threshold
        similarities
            .iter()
            .enumerate()
            .filter(|(i, &sim)| sim >= self.thresholds[*i])
            .map(|(i, &sim)| (
                self.continuation_ids[i],
                self.channel_indices[i],
                sim,
                self.persist_flags[i],
            ))
            .collect()
    }

    /// Check if there are any waiting queries.
    pub fn is_empty(&self) -> bool {
        self.embeddings.nrows() == 0
    }

    /// Get the number of waiting queries.
    pub fn len(&self) -> usize {
        self.embeddings.nrows()
    }

    /// L2-normalize a vector for cosine similarity computation.
    fn l2_normalize(&self, v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }

    /// Swap-remove a row from all parallel structures.
    fn swap_remove_row(&mut self, index: usize) {
        let nrows = self.embeddings.nrows();
        if nrows == 0 || index >= nrows {
            return;
        }

        // Remove from parallel vectors
        self.thresholds.swap_remove(index);
        self.continuation_ids.swap_remove(index);
        self.channel_indices.swap_remove(index);
        self.persist_flags.swap_remove(index);

        // Remove from embedding matrix
        if index == nrows - 1 {
            // Just remove the last row
            self.embeddings = self.embeddings.slice(ndarray::s![..nrows - 1, ..]).to_owned();
        } else {
            // Swap with last row, then remove last
            let last_row = self.embeddings.row(nrows - 1).to_owned();
            self.embeddings.row_mut(index).assign(&last_row);
            self.embeddings = self.embeddings.slice(ndarray::s![..nrows - 1, ..]).to_owned();
        }
    }

    /// Get the dimensions of this query matrix.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl Default for SimilarityQueryMatrix {
    fn default() -> Self {
        Self::new(128) // Common embedding dimension
    }
}
