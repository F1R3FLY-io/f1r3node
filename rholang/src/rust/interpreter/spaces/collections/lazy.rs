//! Lazy Result Producer for VectorDB Similarity Queries
//!
//! This module provides a lazy result producer for VectorDB similarity queries,
//! enabling efficient memory usage through on-demand document retrieval.

// ==========================================================================
// Lazy Result Producer for VectorDB Similarity Queries
// ==========================================================================

/// Lazy result producer for VectorDB similarity queries.
///
/// This struct enables **truly lazy** channel semantics for similarity queries:
/// 1. Stores pre-computed similarity scores (indices + scores, NOT documents)
/// 2. Returns the next index on-demand via `next_index()`
/// 3. The caller retrieves the document from the VectorDB collection
///
/// # True Laziness
///
/// Unlike eager evaluation where all matching documents are retrieved upfront,
/// this producer only returns indices. The actual document retrieval happens
/// in the consumer-side code that has access to the VectorDB collection:
///
/// - **Memory efficient**: Only requested documents are cloned
/// - **Early termination**: If consumer stops after doc 3 of 10, docs 4-10 are never retrieved
/// - **Backpressure**: Production rate controlled by consumption
///
/// # Architecture
///
/// The producer stores indices only. Document retrieval is done by the
/// `GenericRSpace.consume()` method which has access to the channel store:
///
/// ```text
/// // In consume_with_similarity:
/// let producer = LazyResultProducer::new(sorted_indices_and_scores, source_channel);
/// // Store in registry with result_channel_id as key
///
/// // In consume() when detecting lazy channel:
/// if let Some((doc_idx, score)) = producer.next_index() {
///     // Retrieve document from VectorDB collection
///     let dc = channel_store.get_data_collection_mut(&producer.source_channel());
///     let doc = dc.take_by_index(doc_idx);
///     // Produce to result channel
/// } else {
///     // Producer exhausted, produce Nil
/// }
/// ```
///
/// # Thread Safety
///
/// The producer itself is simple and doesn't hold mutable state of the collection.
/// Thread safety is managed by the calling code.
#[derive(Debug, Clone)]
pub struct LazyResultProducer<C> {
    /// Pre-computed similarity matches: (document_index, score) sorted by score descending.
    /// These are indices into the VectorDB, NOT the documents themselves.
    sorted_matches: Vec<(usize, f32)>,

    /// The source channel where the VectorDB collection lives.
    /// Used by the consumer-side code to retrieve documents.
    source_channel: C,

    /// Current position in the sorted_matches list.
    /// Incremented each time `next_index()` is called.
    next_idx: usize,

    /// Whether the producer has been exhausted (all matches consumed).
    /// Once true, `next_index()` always returns None.
    exhausted: bool,
}

impl<C> LazyResultProducer<C>
where
    C: Clone,
{
    /// Create a new lazy result producer from pre-computed similarity scores.
    ///
    /// # Arguments
    ///
    /// * `sorted_matches` - Vector of (document_index, score) pairs, sorted by score descending.
    ///                      These are indices, not documents - documents are retrieved lazily.
    /// * `source_channel` - The channel where the VectorDB collection lives.
    ///
    /// # Returns
    ///
    /// A new `LazyResultProducer` ready to provide document indices on demand.
    ///
    /// # True Laziness
    ///
    /// Documents are NOT retrieved at construction time. Only when the consumer
    /// calls `next_index()` and then retrieves from the collection does the
    /// document get accessed.
    pub fn new(sorted_matches: Vec<(usize, f32)>, source_channel: C) -> Self {
        let exhausted = sorted_matches.is_empty();
        LazyResultProducer {
            sorted_matches,
            source_channel,
            next_idx: 0,
            exhausted,
        }
    }

    /// Get the source channel where the VectorDB collection lives.
    ///
    /// The consumer-side code uses this to retrieve documents:
    /// `channel_store.get_data_collection_mut(&producer.source_channel())`
    pub fn source_channel(&self) -> &C {
        &self.source_channel
    }

    /// Get the next document index and score (TRULY LAZY).
    ///
    /// This method returns the index and score of the next document to retrieve.
    /// The caller is responsible for actually retrieving the document from
    /// the VectorDB collection using this index.
    ///
    /// # Returns
    ///
    /// - `Some((index, score))`: The index and similarity score of the next document
    /// - `None`: Producer is exhausted, no more documents available
    ///
    /// # Lazy Retrieval
    ///
    /// This method does NOT retrieve the document. The caller must:
    /// 1. Get the VectorDB collection via `channel_store.get_data_collection_mut()`
    /// 2. Retrieve the document using `collection.take_by_index(index)` or similar
    /// 3. Produce the document to the result channel
    pub fn next_index(&mut self) -> Option<(usize, f32)> {
        if self.exhausted {
            return None;
        }

        if self.next_idx >= self.sorted_matches.len() {
            self.exhausted = true;
            return None;
        }

        let result = self.sorted_matches[self.next_idx];
        self.next_idx += 1;

        // Check if we've reached the end
        if self.next_idx >= self.sorted_matches.len() {
            self.exhausted = true;
        }

        Some(result)
    }

    /// Check if the producer has been exhausted.
    ///
    /// Returns `true` if all document indices have been consumed, `false` otherwise.
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Get the number of remaining documents.
    ///
    /// This is useful for debugging and for consumers that want to know
    /// how many more documents are available without consuming them.
    pub fn remaining(&self) -> usize {
        if self.exhausted {
            0
        } else {
            self.sorted_matches.len().saturating_sub(self.next_idx)
        }
    }

    /// Get the total number of matches (regardless of how many have been consumed).
    pub fn total_matches(&self) -> usize {
        self.sorted_matches.len()
    }

    /// Peek at the next document index and score without consuming it.
    ///
    /// Returns a reference to the next (index, score) pair if available.
    pub fn peek_next(&self) -> Option<(usize, f32)> {
        if self.exhausted || self.next_idx >= self.sorted_matches.len() {
            None
        } else {
            Some(self.sorted_matches[self.next_idx])
        }
    }
}
