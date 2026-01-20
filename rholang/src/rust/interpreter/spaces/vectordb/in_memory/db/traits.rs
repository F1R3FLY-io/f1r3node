//! VectorDB trait definition.
//!
//! This trait provides a high-level interface for VectorDB operations,
//! composing a VectorBackend with application-specific semantics.

use super::super::backend::VectorBackend;
use super::super::error::VectorDBError;
use super::super::metrics::{EmbeddingType, SimilarityMetric};

/// High-level VectorDB interface for application integration.
///
/// This trait adds application-level semantics on top of a `VectorBackend`:
/// - Associates data items with their embeddings
/// - Tracks persistence flags (consume vs peek)
/// - Provides the consume-on-match pattern used by RSpace
///
/// # Type Parameters
///
/// * `A` - The data type associated with embeddings (e.g., `Par` for Rholang)
///
/// # Usage Patterns
///
/// The trait supports two primary access patterns:
///
/// 1. **Consume** - Find and remove the most similar item(s)
/// 2. **Peek** - Find the most similar item(s) without removal
///
/// Persistence flags allow items to be configured for peek-only behavior
/// even when consumed.
///
/// # Example
///
/// ```ignore
/// use rho_vectordb::db::VectorDB;
///
/// fn process_similar<A, V: VectorDB<A>>(db: &mut V, query: &[f32]) {
///     if let Some((data, score)) = db.consume_most_similar(query, 0.8) {
///         println!("Found match with score {}: {:?}", score, data);
///     }
/// }
/// ```
pub trait VectorDB<A>: Clone + Send + Sync {
    /// The underlying backend type.
    type Backend: VectorBackend;

    /// Store data with its embedding vector.
    ///
    /// This is equivalent to `put_with_persist(data, embedding, false)`.
    fn put(&mut self, data: A, embedding: Vec<f32>) -> Result<(), VectorDBError>;

    /// Store data with embedding and persistence flag.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to associate with the embedding
    /// * `embedding` - The embedding vector
    /// * `persist` - If `true`, the item will not be removed on consume
    fn put_with_persist(
        &mut self,
        data: A,
        embedding: Vec<f32>,
        persist: bool,
    ) -> Result<(), VectorDBError>;

    /// Find and remove the most similar data item (consume pattern).
    ///
    /// Returns the data and similarity score of the most similar item
    /// above the threshold. If the item is persistent, it remains in the DB.
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding vector
    /// * `threshold` - Minimum similarity threshold
    ///
    /// # Returns
    ///
    /// `Some((data, score))` if a match is found, `None` otherwise.
    fn consume_most_similar(&mut self, query: &[f32], threshold: f32) -> Option<(A, f32)>;

    /// Find the most similar data item without removal (peek pattern).
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding vector
    /// * `threshold` - Minimum similarity threshold
    ///
    /// # Returns
    ///
    /// `Some((data_ref, score))` if a match is found, `None` otherwise.
    fn peek_most_similar(&self, query: &[f32], threshold: f32) -> Option<(&A, f32)>;

    /// Find top-K most similar items (peek).
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding vector
    /// * `threshold` - Minimum similarity threshold
    /// * `k` - Maximum number of results
    ///
    /// # Returns
    ///
    /// Vector of (data_ref, score) pairs sorted by score descending.
    fn peek_top_k(&self, query: &[f32], threshold: f32, k: usize) -> Vec<(&A, f32)>;

    /// Find and remove top-K most similar items (consume).
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding vector
    /// * `threshold` - Minimum similarity threshold
    /// * `k` - Maximum number of results
    ///
    /// # Returns
    ///
    /// Vector of (data, score) pairs sorted by score descending.
    /// Persistent items are returned but not removed.
    fn consume_top_k(&mut self, query: &[f32], threshold: f32, k: usize) -> Vec<(A, f32)>;

    /// Get the default similarity threshold for this VectorDB.
    fn default_threshold(&self) -> f32;

    /// Get the embedding dimensions.
    fn dimensions(&self) -> usize;

    /// Get the similarity metric used by this VectorDB.
    fn metric(&self) -> SimilarityMetric;

    /// Get the embedding type used by this VectorDB.
    fn embedding_type(&self) -> EmbeddingType;

    /// Get the number of stored items.
    fn len(&self) -> usize;

    /// Check if the VectorDB is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all data from the VectorDB.
    fn clear(&mut self);

    /// Get a reference to the underlying backend.
    fn backend(&self) -> &Self::Backend;

    /// Get a mutable reference to the underlying backend.
    fn backend_mut(&mut self) -> &mut Self::Backend;
}
