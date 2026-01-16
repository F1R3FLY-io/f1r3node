//! VectorDB Data Collection (ndarray-optimized)
//!
//! VectorDB data collection - similarity-based matching using a pluggable backend.
//!
//! This collection delegates embedding storage and similarity computation to a
//! `VectorBackendDyn` implementation. The collection manages the mapping between
//! backend embedding IDs and data items.
//!
//! # Architecture
//!
//! ```text
//! VectorDBDataCollection<A>
//! ├── backend: Box<dyn VectorBackendDyn>  ← Stores embeddings, computes similarity
//! ├── data: HashMap<usize, A>             ← Maps embedding IDs to data
//! ├── persist_flags: HashMap<usize, bool> ← Tracks persistence per ID
//! └── non_indexed_data: Vec<(A, bool)>    ← Data without embeddings (sync channels)
//! ```
//!
//! # Backends
//!
//! By default, uses the in-memory backend from `BackendRegistry::default()`.
//! Custom backends can be provided via `with_backend()`.

use std::collections::HashMap;
use std::sync::Arc;

use super::super::errors::SpaceError;
use super::super::vectordb::registry::{BackendConfig, BackendRegistry, VectorBackendDyn, ResolvedArg as BackendResolvedArg};
use super::super::vectordb::types::EmbeddingType;
use super::super::vectordb::in_memory::{
    SimilarityMetric, IndexConfig,
    FunctionHandlerRegistry,
};

use super::core::DataCollection;
use super::similarity::SimilarityCollection;

// ==========================================================================
// VectorDB Data Collection (ndarray-optimized)
// ==========================================================================

/// VectorDB data collection - similarity-based matching with matrix operations.
///
/// Data is stored with associated embedding vectors in an ndarray matrix for
/// efficient SIMD-accelerated similarity computation. Matching uses configurable
/// similarity metrics to find the most similar element above a threshold.
///
/// # Type Parameters
/// - `A`: The data type to store
///
/// # Architecture
///
/// Embeddings are stored in a row-major matrix where each row is a document
/// embedding. For cosine similarity, embeddings are pre-normalized at insertion
/// time so that similarity computation is a simple dot product (leveraging SIMD).
///
/// # Supported Similarity Metrics
///
/// - **Cosine**: `dot(a_norm, b_norm)` - pre-normalized for efficiency
/// - **DotProduct**: `dot(a, b)` - raw inner product
/// - **Euclidean**: `1 / (1 + ||a - b||)` - L2 distance converted to similarity
/// - **Manhattan**: `1 / (1 + L1(a, b))` - L1 distance converted to similarity
/// - **Hamming**: `1 - count(a != b) / len` - for boolean/hypervectors
/// - **Jaccard**: `|A ∩ B| / |A ∪ B|` - for boolean vectors
///
/// # Embedding Types (Rholang Representation)
///
/// Since Rholang has no floating point, embeddings are represented as:
/// - **Boolean**: `[0, 1, 1, 0]` - binary vectors (hypervectors)
/// - **Integer**: `[90, 5, 10, 20]` - 0-100 scale, converted to 0.0-1.0
/// - **Float**: `"0.9,0.05,0.1,0.2"` - comma-separated string, parsed to floats
///
/// # Performance
///
/// - **SIMD**: Matrix operations leverage CPU vector instructions
/// - **Cache efficiency**: Row-major storage optimizes memory access patterns
/// - **Pre-normalization**: Cosine similarity = dot product for normalized vectors
/// - **Batch operations**: Multiple queries can be vectorized
/// VectorDB data collection - similarity-based matching using a pluggable backend.
///
/// This collection delegates embedding storage and similarity computation to a
/// `VectorBackendDyn` implementation. The collection manages the mapping between
/// backend embedding IDs and data items.
///
/// # Architecture
///
/// ```text
/// VectorDBDataCollection<A>
/// ├── backend: Box<dyn VectorBackendDyn>  ← Stores embeddings, computes similarity
/// ├── data: HashMap<usize, A>             ← Maps embedding IDs to data
/// ├── persist_flags: HashMap<usize, bool> ← Tracks persistence per ID
/// └── non_indexed_data: Vec<(A, bool)>    ← Data without embeddings (sync channels)
/// ```
///
/// # Backends
///
/// By default, uses the in-memory backend from `BackendRegistry::default()`.
/// Custom backends can be provided via `with_backend()`.
///
/// # Type Parameters
/// - `A`: The data type to store
pub struct VectorDBDataCollection<A> {
    /// The underlying VectorDB backend (owns embeddings and similarity computation)
    backend: Box<dyn VectorBackendDyn>,

    /// Data items stored by embedding ID (backend returns usize IDs)
    data: HashMap<usize, A>,

    /// Persistence flags per embedding ID (true = persistent, won't be removed)
    persist_flags: HashMap<usize, bool>,

    /// Non-indexed data storage for items without embeddings.
    /// This allows sync channels like `ready!(Nil)` to work within VectorDB spaces.
    /// These items are matched via predicate-based `find_and_remove`, not similarity.
    non_indexed_data: Vec<(A, bool)>,

    /// Expected embedding format from Rholang (for validation during produce)
    embedding_type: EmbeddingType,
}

impl<A: Clone> Clone for VectorDBDataCollection<A> {
    fn clone(&self) -> Self {
        // Clone the backend using clone_boxed
        let mut cloned_backend = self.backend.clone_boxed();

        // We need to copy the embeddings manually since clone_boxed creates a new backend
        // but we need it to have the same embeddings as the original
        // Clear the cloned backend first (it starts empty from clone_boxed)
        cloned_backend.clear();

        // Re-insert all embeddings to get the same IDs
        // This works because the backend assigns IDs sequentially
        let mut id_mapping: HashMap<usize, usize> = HashMap::new();
        for &old_id in self.data.keys() {
            if let Some(embedding) = self.backend.get(old_id) {
                if let Ok(new_id) = cloned_backend.store(&embedding) {
                    id_mapping.insert(old_id, new_id);
                }
            }
        }

        // Map data to new IDs
        let data: HashMap<usize, A> = self.data.iter()
            .filter_map(|(old_id, item)| {
                id_mapping.get(old_id).map(|&new_id| (new_id, item.clone()))
            })
            .collect();

        // Map persist flags to new IDs
        let persist_flags: HashMap<usize, bool> = self.persist_flags.iter()
            .filter_map(|(old_id, &flag)| {
                id_mapping.get(old_id).map(|&new_id| (new_id, flag))
            })
            .collect();

        VectorDBDataCollection {
            backend: cloned_backend,
            data,
            persist_flags,
            non_indexed_data: self.non_indexed_data.clone(),
            embedding_type: self.embedding_type,
        }
    }
}

impl<A> VectorDBDataCollection<A> {
    /// Create a new VectorDB collection with default settings.
    ///
    /// Uses the default backend from BackendRegistry and Float embedding type.
    ///
    /// # Arguments
    /// - `dimensions`: The dimensionality of embedding vectors
    pub fn new(dimensions: usize) -> Self {
        Self::with_config("rho", dimensions, 0.8, None, EmbeddingType::Float)
    }

    /// Create with a custom similarity threshold.
    ///
    /// # Arguments
    /// - `dimensions`: The dimensionality of embedding vectors
    /// - `threshold`: Minimum similarity for a match (0.0 to 1.0)
    pub fn with_threshold(dimensions: usize, threshold: f32) -> Self {
        Self::with_config(
            "rho",
            dimensions,
            threshold.clamp(0.0, 1.0),
            None,
            EmbeddingType::Float,
        )
    }

    /// Create with a custom similarity metric.
    ///
    /// # Arguments
    /// - `dimensions`: The dimensionality of embedding vectors
    /// - `threshold`: Minimum similarity for a match (0.0 to 1.0)
    /// - `metric`: The similarity metric to use
    pub fn with_metric(dimensions: usize, threshold: f32, metric: SimilarityMetric) -> Self {
        Self::with_config("rho", dimensions, threshold.clamp(0.0, 1.0), Some(metric), EmbeddingType::Float)
    }

    /// Create with full configuration.
    ///
    /// Creates a backend from the registry with the specified settings.
    /// The backend is the single source of truth for defaults - if metric is None,
    /// the backend will derive the appropriate default based on embedding_type.
    ///
    /// # Arguments
    /// - `backend_name`: Backend to use (e.g., "rho", "default", "pinecone")
    /// - `dimensions`: The dimensionality of embedding vectors
    /// - `threshold`: Minimum similarity for a match (0.0 to 1.0)
    /// - `metric`: The similarity metric to use, or None for backend default
    /// - `embedding_type`: Expected embedding format from Rholang
    pub fn with_config(
        backend_name: &str,
        dimensions: usize,
        threshold: f32,
        metric: Option<SimilarityMetric>,
        embedding_type: EmbeddingType,
    ) -> Self {
        // Create backend config - only include metric if explicitly specified
        let mut config = BackendConfig::default();
        if let Some(m) = metric {
            config.options.insert("metric".to_string(), m.as_str().to_string());
        }
        config.options.insert("threshold".to_string(), threshold.to_string());
        config.options.insert("embedding_type".to_string(), embedding_type.as_str().to_string());

        // Create backend from default registry
        let registry = BackendRegistry::default();
        let backend = registry
            .create(backend_name, dimensions, &config)
            .expect("Failed to create VectorDB backend");

        VectorDBDataCollection {
            backend,
            data: HashMap::new(),
            persist_flags: HashMap::new(),
            non_indexed_data: Vec::new(),
            embedding_type,
        }
    }

    /// Create with full configuration including index settings.
    ///
    /// Note: Index configuration is now handled by the backend.
    #[allow(unused_variables)]
    pub fn with_index_config(
        dimensions: usize,
        threshold: f32,
        metric: SimilarityMetric,
        embedding_type: EmbeddingType,
        index_config: IndexConfig,
    ) -> Self {
        // Index config is handled by the backend internally
        Self::with_config("rho", dimensions, threshold, Some(metric), embedding_type)
    }

    /// Create a VectorDB collection with a provided backend.
    ///
    /// This allows using custom backends (Pinecone, Qdrant, etc.)
    /// or pre-configured in-memory backends.
    pub fn with_backend(
        backend: Box<dyn VectorBackendDyn>,
        embedding_type: EmbeddingType,
    ) -> Self {
        VectorDBDataCollection {
            backend,
            data: HashMap::new(),
            persist_flags: HashMap::new(),
            non_indexed_data: Vec::new(),
            embedding_type,
        }
    }

    /// Create a VectorDB collection with a custom handler registry.
    ///
    /// Note: Handler registries are now managed by the backend.
    /// This method creates a standard backend - custom handlers should be
    /// configured when creating the backend.
    #[allow(unused_variables)]
    pub fn with_registry(
        dimensions: usize,
        threshold: f32,
        metric: SimilarityMetric,
        embedding_type: EmbeddingType,
        registry: Arc<FunctionHandlerRegistry>,
    ) -> Self {
        // Handlers are managed by the backend
        Self::with_config("rho", dimensions, threshold, Some(metric), embedding_type)
    }

    /// Get the current similarity threshold.
    pub fn threshold(&self) -> f32 {
        self.backend.default_threshold()
    }

    /// Get the embedding dimensions.
    pub fn dimensions(&self) -> usize {
        self.backend.dimensions()
    }

    /// Get the default similarity metric name.
    pub fn metric_name(&self) -> &str {
        self.backend.default_similarity_fn()
    }

    /// Get the expected embedding type.
    pub fn embedding_type(&self) -> EmbeddingType {
        self.embedding_type
    }

    /// Get the number of indexed embeddings (excluding non-indexed data).
    pub fn live_count(&self) -> usize {
        self.backend.len()
    }

    /// Get the list of supported similarity metrics.
    pub fn supported_similarity_fns(&self) -> Vec<String> {
        self.backend.supported_similarity_fns()
    }

    /// Get the list of supported ranking functions.
    pub fn supported_ranking_fns(&self) -> Vec<String> {
        self.backend.supported_ranking_fns()
    }

    /// Put data with its embedding vector.
    ///
    /// The backend handles any required normalization (e.g., L2 for cosine).
    ///
    /// # Arguments
    /// - `data`: The data to store
    /// - `embedding`: The embedding vector (must match dimensions)
    ///
    /// # Returns
    /// - `Ok(())` if successful
    /// - `Err(...)` if embedding dimensions don't match
    pub fn put_with_embedding(&mut self, data: A, embedding: Vec<f32>) -> Result<(), SpaceError> {
        self.put_with_embedding_and_persist(data, embedding, false)
    }

    pub fn put_with_embedding_and_persist(&mut self, data: A, embedding: Vec<f32>, persist: bool) -> Result<(), SpaceError> {
        // Store embedding in backend - backend handles normalization and indexing
        let id = self.backend.store(&embedding)?;

        // Track data and persistence
        self.data.insert(id, data);
        self.persist_flags.insert(id, persist);

        Ok(())
    }

    /// Store data with embedding and return the index where it was stored.
    ///
    /// This is used by the store-first similarity matching approach where we:
    /// 1. Store the data first (getting its index)
    /// 2. Check the query matrix for matching queries
    /// 3. Fire matching continuations or leave data for future queries
    ///
    /// # Arguments
    /// - `data`: The data to store
    /// - `embedding`: The embedding vector (must match dimensions)
    /// - `persist`: Whether to keep the data after matching
    ///
    /// # Returns
    /// - `Ok(index)` where index is the embedding ID from the backend
    /// - `Err(...)` if embedding dimensions don't match
    pub fn put_with_embedding_returning_index(
        &mut self,
        data: A,
        embedding: Vec<f32>,
        persist: bool,
    ) -> Result<usize, SpaceError> {
        // Store embedding in backend
        let id = self.backend.store(&embedding)?;

        // Track data and persistence
        self.data.insert(id, data);
        self.persist_flags.insert(id, persist);

        Ok(id)
    }

    /// Get a copy of the embedding at the given index.
    ///
    /// # Arguments
    /// - `index`: Embedding ID from the backend
    ///
    /// # Returns
    /// - `Some(embedding)` if index is valid
    /// - `None` if index is not found
    pub fn get_normalized_embedding(&self, index: usize) -> Option<Vec<f32>> {
        self.backend.get(index)
    }

    /// Remove data and embedding at the given index.
    ///
    /// # Arguments
    /// - `index`: Embedding ID to remove
    ///
    /// # Returns
    /// - `Some((data, embedding))` if index was valid
    /// - `None` if index was not found
    pub fn remove_by_index(&mut self, index: usize) -> Option<(A, Vec<f32>)>
    where
        A: Clone,
    {
        // Get the embedding before removing
        let embedding = self.backend.get(index)?;

        // Remove from backend and data
        self.backend.remove(index);
        let data = self.data.remove(&index)?;
        self.persist_flags.remove(&index);

        Some((data, embedding))
    }

    /// Get document by index, respecting persistence semantics.
    ///
    /// - Persistent docs: returns clone without removing (can be matched again)
    /// - Non-persistent docs: removes and returns (queue-like behavior)
    ///
    /// This method should be used for lazy retrieval where we need to respect
    /// the persistence flag set during `produce` operations.
    ///
    /// # Returns
    /// - `Some((data, embedding))` if index was valid
    /// - `None` if index was not found
    pub fn get_or_remove_by_index(&mut self, index: usize) -> Option<(A, Vec<f32>)>
    where
        A: Clone,
    {
        // Get the embedding
        let embedding = self.backend.get(index)?;

        let is_persistent = self.persist_flags.get(&index).copied().unwrap_or(false);

        if is_persistent {
            // Persistent: return clone without removing
            self.data.get(&index).map(|data| (data.clone(), embedding))
        } else {
            // Non-persistent: remove and return
            self.backend.remove(index);
            let data = self.data.remove(&index)?;
            self.persist_flags.remove(&index);
            Some((data, embedding))
        }
    }

    /// Remove data and embedding at the given index.
    ///
    /// # Arguments
    /// - `index`: Embedding ID to remove
    ///
    /// # Returns
    /// - `Some(data)` if index was valid and contained data
    /// - `None` if index was not found
    fn remove_entry(&mut self, index: usize) -> Option<A> {
        // Remove from backend
        self.backend.remove(index);

        // Remove from data and flags
        self.persist_flags.remove(&index);
        self.data.remove(&index)
    }

    /// Find the most similar element to the query embedding.
    ///
    /// Uses the collection's default threshold.
    pub fn find_most_similar(&self, query_embedding: &[f32]) -> Option<(&A, f32)> {
        self.find_most_similar_with_threshold(query_embedding, self.backend.default_threshold())
    }

    /// Find the most similar element using a per-query threshold.
    pub fn find_most_similar_with_threshold(
        &self,
        query_embedding: &[f32],
        threshold: f32,
    ) -> Option<(&A, f32)> {
        if self.data.is_empty() {
            return None;
        }

        // Use backend.query with top_k = 1
        let results = self.backend.query(
            query_embedding,
            None,  // Use default similarity function
            Some(threshold),
            Some("top_k"),
            &[BackendResolvedArg::Int(1)],
        ).ok()?;

        // Get the best match
        results.into_iter().next().and_then(|(id, score)| {
            self.data.get(&id).map(|d| (d, score))
        })
    }

    /// Find and remove the most similar element to the query embedding.
    ///
    /// Uses the collection's default threshold.
    pub fn find_and_remove_most_similar(&mut self, query_embedding: &[f32]) -> Option<(A, f32)>
    where
        A: Clone,
    {
        self.find_and_remove_most_similar_impl(query_embedding, self.backend.default_threshold())
    }

    /// Find and remove the most similar element using a per-query threshold.
    fn find_and_remove_most_similar_impl(
        &mut self,
        query_embedding: &[f32],
        threshold: f32,
    ) -> Option<(A, f32)>
    where
        A: Clone,
    {
        if self.data.is_empty() {
            return None;
        }

        // Use backend.query with top_k = 1
        let results = self.backend.query(
            query_embedding,
            None,  // Use default similarity function
            Some(threshold),
            Some("top_k"),
            &[BackendResolvedArg::Int(1)],
        ).ok()?;

        // Get the best match
        results.into_iter().next().and_then(|(id, score)| {
            let is_persistent = self.persist_flags.get(&id).copied().unwrap_or(false);
            if is_persistent {
                // Persistent data: return clone without removing
                self.data.get(&id).map(|d| (d.clone(), score))
            } else {
                // Non-persistent: remove and return
                self.remove_entry(id).map(|d| (d, score))
            }
        })
    }

    /// Find all elements above a similarity threshold.
    ///
    /// # Returns
    /// Vector of (data, similarity) pairs, sorted by descending similarity
    pub fn find_all_similar(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        limit: usize,
    ) -> Vec<(&A, f32)> {
        if self.data.is_empty() {
            return vec![];
        }

        // Use backend.query with top_k limit
        let results = match self.backend.query(
            query_embedding,
            None,  // Use default similarity function
            Some(threshold),
            Some("top_k"),
            &[BackendResolvedArg::Int(limit as i64)],
        ) {
            Ok(r) => r,
            Err(_) => return vec![],
        };

        // Map results to data references
        results
            .into_iter()
            .filter_map(|(id, score)| self.data.get(&id).map(|d| (d, score)))
            .collect()
    }

    /// Find top-K most similar elements above a threshold.
    ///
    /// # Arguments
    /// - `query_embedding`: The embedding vector to match against
    /// - `threshold`: Minimum similarity for this query (0.0 to 1.0)
    /// - `k`: Maximum number of results to return
    ///
    /// # Returns
    /// Vector of (data, similarity) pairs, sorted by descending similarity.
    /// Returns at most K elements, all with similarity >= threshold.
    pub fn find_top_k_similar(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        k: usize,
    ) -> Vec<(&A, f32)> {
        if self.data.is_empty() || k == 0 {
            return vec![];
        }

        // Use backend.query with top_k
        let results = match self.backend.query(
            query_embedding,
            None,  // Use default similarity function
            Some(threshold),
            Some("top_k"),
            &[BackendResolvedArg::Int(k as i64)],
        ) {
            Ok(r) => r,
            Err(_) => return vec![],
        };

        // Map to results with data
        results
            .into_iter()
            .filter_map(|(id, score)| self.data.get(&id).map(|d| (d, score)))
            .collect()
    }

    /// Find top-K similar element indices without retrieving or removing documents.
    ///
    /// This method is designed for **lazy evaluation** - it returns only indices
    /// and scores, allowing documents to be retrieved on-demand later. This enables:
    /// - **Memory efficiency**: Only requested documents are cloned
    /// - **Early termination**: If consumer stops after 3 of 10, docs 4-10 are never retrieved
    /// - **Backpressure**: Production rate controlled by consumption
    ///
    /// # Arguments
    /// - `query_embedding`: The embedding vector to match against
    /// - `threshold`: Minimum similarity for this query (0.0 to 1.0)
    /// - `k`: Maximum number of results to return
    ///
    /// # Returns
    /// Vector of (index, similarity) pairs, sorted by descending similarity.
    /// These indices can be used with `remove_by_index()` to retrieve documents lazily.
    pub fn find_top_k_similar_indices(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        k: usize,
    ) -> Vec<(usize, f32)> {
        if self.data.is_empty() || k == 0 {
            return vec![];
        }

        // Use backend.query with top_k
        self.backend.query(
            query_embedding,
            None,  // Use default similarity function
            Some(threshold),
            Some("top_k"),
            &[BackendResolvedArg::Int(k as i64)],
        ).unwrap_or_default()
    }

    /// Execute a similarity query with optional function overrides.
    ///
    /// Delegates to the backend for similarity computation and ranking.
    ///
    /// # Arguments
    /// - `query_embedding`: The embedding vector to match against
    /// - `similarity_fn`: Optional similarity function override (e.g., "cosine", "euclidean")
    /// - `threshold`: Optional threshold override
    /// - `ranking_fn`: Optional ranking function override (e.g., "top_k", "all")
    /// - `params`: Additional parameters for the ranking function
    ///
    /// # Returns
    /// Vector of (index, score) pairs sorted by score descending.
    #[cfg(feature = "vectordb")]
    pub fn query(
        &self,
        query_embedding: &[f32],
        similarity_fn: Option<&str>,
        threshold: Option<f32>,
        ranking_fn: Option<&str>,
        params: &[BackendResolvedArg],
    ) -> Result<Vec<(usize, f32)>, SpaceError> {
        self.backend.query(query_embedding, similarity_fn, threshold, ranking_fn, params)
    }

    /// Find and remove top-K most similar elements above a threshold.
    ///
    /// For non-persistent data, elements are removed.
    /// For persistent data, clones are returned without removal.
    ///
    /// # Arguments
    /// - `query_embedding`: The embedding vector to match against
    /// - `threshold`: Minimum similarity for this query (0.0 to 1.0)
    /// - `k`: Maximum number of results to return
    ///
    /// # Returns
    /// Vector of (data, similarity) pairs, sorted by descending similarity.
    pub fn find_and_remove_top_k_similar(
        &mut self,
        query_embedding: &[f32],
        threshold: f32,
        k: usize,
    ) -> Vec<(A, f32)>
    where
        A: Clone,
    {
        if self.data.is_empty() || k == 0 {
            return vec![];
        }

        // Use backend.query with top_k
        let results = match self.backend.query(
            query_embedding,
            None,
            Some(threshold),
            Some("top_k"),
            &[BackendResolvedArg::Int(k as i64)],
        ) {
            Ok(r) => r,
            Err(_) => return vec![],
        };

        // Process results: remove non-persistent, clone persistent
        results
            .into_iter()
            .filter_map(|(id, score)| {
                let is_persistent = self.persist_flags.get(&id).copied().unwrap_or(false);
                if is_persistent {
                    // Persistent: return clone without removing
                    self.data.get(&id).map(|d| (d.clone(), score))
                } else {
                    // Non-persistent: remove and return
                    self.remove_entry(id).map(|d| (d, score))
                }
            })
            .collect()
    }

    /// Get the embedding for a data item by index.
    pub fn get_embedding(&self, index: usize) -> Option<Vec<f32>> {
        self.backend.get(index)
    }

    /// Get the similarity metric name.
    pub fn similarity_metric(&self) -> &str {
        self.backend.default_similarity_fn()
    }

    /// Get supported similarity metrics.
    pub fn supported_metrics(&self) -> Vec<String> {
        self.backend.supported_similarity_fns()
    }

    /// Query with handlers using raw string identifiers.
    ///
    /// Convenience wrapper that accepts string slices directly.
    pub fn query_with_handler_ids(
        &mut self,
        query_embedding: &[f32],
        sim_function_id: Option<&str>,
        threshold: Option<f32>,
        rank_function_id: Option<&str>,
        top_k: Option<usize>,
    ) -> Result<Vec<(A, f32)>, SpaceError>
    where
        A: Clone,
    {
        if self.data.is_empty() {
            return Ok(vec![]);
        }

        // Build params for backend query
        let params = top_k
            .map(|k| vec![BackendResolvedArg::Int(k as i64)])
            .unwrap_or_default();

        // Query via backend
        let results = self.backend.query(
            query_embedding,
            sim_function_id,
            threshold,
            rank_function_id,
            &params,
        )?;

        // Collect results, handling persistence
        let output = results
            .into_iter()
            .filter_map(|(id, score)| {
                let is_persistent = self.persist_flags.get(&id).copied().unwrap_or(false);
                if is_persistent {
                    // Persistent: return clone without removing
                    self.data.get(&id).map(|d| (d.clone(), score))
                } else {
                    // Non-persistent: remove and return
                    self.remove_entry(id).map(|d| (d, score))
                }
            })
            .collect();

        Ok(output)
    }
}

/// Debug implementation for VectorDBDataCollection.
impl<A: std::fmt::Debug> std::fmt::Debug for VectorDBDataCollection<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorDBDataCollection")
            .field("backend_len", &self.backend.len())
            .field("data_count", &self.data.len())
            .field("non_indexed_count", &self.non_indexed_data.len())
            .field("dimensions", &self.backend.dimensions())
            .field("threshold", &self.backend.default_threshold())
            .field("similarity_fn", &self.backend.default_similarity_fn())
            .field("embedding_type", &self.embedding_type)
            .finish()
    }
}

impl<A> Default for VectorDBDataCollection<A> {
    fn default() -> Self {
        Self::new(128) // Common embedding dimension
    }
}

impl<A: Clone + Send + Sync> DataCollection<A> for VectorDBDataCollection<A> {
    fn put(&mut self, data: A) -> Result<(), SpaceError> {
        // Store data without embedding in non_indexed_data.
        // This allows sync channels like `ready!(Nil)` to work within VectorDB spaces.
        // These items will be matched via predicate-based find_and_remove, not similarity.
        self.non_indexed_data.push((data, false));
        Ok(())
    }

    fn put_with_persist(&mut self, data: A, persist: bool) -> Result<(), SpaceError> {
        // Store data without embedding in non_indexed_data.
        // This allows sync channels like `ready!(Nil)` to work within VectorDB spaces.
        // These items will be matched via predicate-based find_and_remove, not similarity.
        self.non_indexed_data.push((data, persist));
        Ok(())
    }

    fn find_and_remove<F>(&mut self, predicate: F) -> Option<A>
    where
        F: Fn(&A) -> bool,
    {
        // First check indexed data (with embeddings)
        let matching_id = self.data.iter()
            .find(|(_, a)| predicate(a))
            .map(|(&id, _)| id);

        if let Some(id) = matching_id {
            let is_persistent = self.persist_flags.get(&id).copied().unwrap_or(false);
            if is_persistent {
                // Persistent: return clone without removing
                return self.data.get(&id).cloned();
            } else {
                // Non-persistent: remove and return
                return self.remove_entry(id);
            }
        }

        // Then check non-indexed data (sync channels, etc.)
        if let Some(pos) = self.non_indexed_data.iter().position(|(a, _)| predicate(a)) {
            let (ref data, is_persistent) = self.non_indexed_data[pos];
            if is_persistent {
                // Persistent: return clone without removing
                Some(data.clone())
            } else {
                // Non-persistent: remove and return
                Some(self.non_indexed_data.swap_remove(pos).0)
            }
        } else {
            None
        }
    }

    fn peek<F>(&self, predicate: F) -> Option<&A>
    where
        F: Fn(&A) -> bool,
    {
        // First check indexed data
        if let Some(a) = self.data.values().find(|a| predicate(*a)) {
            return Some(a);
        }
        // Then check non-indexed data
        self.non_indexed_data.iter().find(|(a, _)| predicate(a)).map(|(a, _)| a)
    }

    fn all_data(&self) -> Vec<&A> {
        let mut result: Vec<&A> = self.data.values().collect();
        result.extend(self.non_indexed_data.iter().map(|(a, _)| a));
        result
    }

    fn clear(&mut self) {
        self.backend.clear();
        self.data.clear();
        self.persist_flags.clear();
        self.non_indexed_data.clear();
    }

    fn is_empty(&self) -> bool {
        self.backend.is_empty() && self.non_indexed_data.is_empty()
    }

    fn len(&self) -> usize {
        self.backend.len() + self.non_indexed_data.len()
    }
}

// ==========================================================================
// SimilarityCollection Implementation for VectorDBDataCollection
// ==========================================================================

impl<A: Clone + Send + Sync> SimilarityCollection<A> for VectorDBDataCollection<A> {
    fn put_with_embedding(&mut self, data: A, embedding: Vec<f32>) -> Result<(), SpaceError> {
        // Delegate to the inherent method on VectorDBDataCollection
        VectorDBDataCollection::put_with_embedding(self, data, embedding)
    }

    fn find_and_remove_most_similar_with_threshold(
        &mut self,
        query_embedding: &[f32],
        threshold: f32,
    ) -> Option<(A, f32)> {
        // Delegate to the inherent method on VectorDBDataCollection
        self.find_and_remove_most_similar_impl(query_embedding, threshold)
    }

    fn peek_most_similar_with_threshold(
        &self,
        query_embedding: &[f32],
        threshold: f32,
    ) -> Option<(&A, f32)> {
        // Delegate to the inherent method on VectorDBDataCollection
        self.find_most_similar_with_threshold(query_embedding, threshold)
    }

    fn default_threshold(&self) -> f32 {
        self.backend.default_threshold()
    }

    fn embedding_dimensions(&self) -> usize {
        self.backend.dimensions()
    }

    fn embedding_type(&self) -> EmbeddingType {
        self.embedding_type
    }

    fn similarity_metric(&self) -> SimilarityMetric {
        SimilarityMetric::from_str(self.backend.default_similarity_fn())
            .unwrap_or_default()
    }

    fn supported_metrics(&self) -> &[SimilarityMetric] {
        // Return all metrics that the backend supports
        // Note: This returns a static slice because SimilarityMetric::all() is const
        SimilarityMetric::all()
    }

    fn find_top_k_similar_with_threshold(
        &self,
        query_embedding: &[f32],
        threshold: f32,
        k: usize,
    ) -> Vec<(&A, f32)> {
        self.find_top_k_similar(query_embedding, threshold, k)
    }

    fn find_and_remove_top_k_similar_with_threshold(
        &mut self,
        query_embedding: &[f32],
        threshold: f32,
        k: usize,
    ) -> Vec<(A, f32)> {
        self.find_and_remove_top_k_similar(query_embedding, threshold, k)
    }
}
