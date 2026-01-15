//! Core Collection Traits
//!
//! This module defines the fundamental traits for data and continuation collections.
//!
//! - `DataCollection`: Storage for data at channels
//! - `ContinuationCollection`: Storage for waiting continuations at channel patterns

use super::super::errors::SpaceError;
use super::similarity::StoredSimilarityInfo;

// ==========================================================================
// Data Collection Trait
// ==========================================================================

/// Trait for data collections at a channel.
///
/// Type parameter `A` is the data type stored in the collection.
pub trait DataCollection<A>: Clone + Send + Sync {
    /// Insert data into the collection.
    ///
    /// For Cell collections, this may return an error if the cell is already full.
    /// Data is stored as non-persistent by default.
    fn put(&mut self, data: A) -> Result<(), SpaceError>;

    /// Insert data into the collection with persistence flag.
    ///
    /// If `persist` is true, the data will remain in the collection even after
    /// being matched by `find_and_remove` (a clone is returned instead of removing).
    fn put_with_persist(&mut self, data: A, persist: bool) -> Result<(), SpaceError>;

    /// Find and remove data that matches the given predicate.
    ///
    /// Returns `Some((data, is_persistent))` if a match was found, `None` otherwise.
    /// For persistent data, a clone is returned but the data remains in the collection.
    /// For non-persistent data, the data is removed and returned.
    /// The matching behavior depends on the collection type (e.g., Queue only
    /// matches the head).
    fn find_and_remove<F>(&mut self, predicate: F) -> Option<A>
    where
        F: Fn(&A) -> bool;

    /// Peek at data that matches the given predicate without removing it.
    fn peek<F>(&self, predicate: F) -> Option<&A>
    where
        F: Fn(&A) -> bool;

    /// Get all data in the collection.
    fn all_data(&self) -> Vec<&A>;

    /// Clear all data from the collection.
    fn clear(&mut self);

    /// Check if the collection is empty.
    fn is_empty(&self) -> bool;

    /// Get the number of elements in the collection.
    fn len(&self) -> usize;
}

// ==========================================================================
// Continuation Collection Trait
// ==========================================================================

/// Trait for continuation collections at channel patterns.
///
/// Type parameters:
/// - `P`: Pattern type
/// - `K`: Continuation type
pub trait ContinuationCollection<P, K>: Clone + Send + Sync {
    /// Insert a continuation with its patterns.
    fn put(&mut self, patterns: Vec<P>, continuation: K, persist: bool);

    /// Insert a continuation with its patterns and similarity requirements.
    ///
    /// This method stores similarity information alongside the continuation so that
    /// `produce()` can properly check if incoming data meets the similarity threshold.
    ///
    /// # Arguments
    /// - `patterns`: Regular patterns for the continuation
    /// - `continuation`: The continuation to fire
    /// - `persist`: Whether to keep the continuation after firing
    /// - `similarity`: Optional similarity requirements (embeddings and thresholds)
    fn put_with_similarity(
        &mut self,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        similarity: Option<StoredSimilarityInfo>,
    );

    /// Find and remove a continuation that matches the given predicate.
    ///
    /// Returns the continuation and its patterns if found.
    fn find_and_remove<F>(&mut self, predicate: F) -> Option<(Vec<P>, K, bool)>
    where
        F: Fn(&[P], &K) -> bool;

    /// Find and remove a continuation that matches the predicate, also returning similarity info.
    ///
    /// This is the similarity-aware version of `find_and_remove`. Use this when you need
    /// to check similarity requirements before firing the continuation.
    fn find_and_remove_with_similarity<F>(
        &mut self,
        predicate: F,
    ) -> Option<(Vec<P>, K, bool, Option<StoredSimilarityInfo>)>
    where
        F: Fn(&[P], &K) -> bool;

    /// Get all continuations in the collection.
    fn all_continuations(&self) -> Vec<(&[P], &K, bool)>;

    /// Get all continuations with their similarity info.
    fn all_continuations_with_similarity(&self) -> Vec<(&[P], &K, bool, Option<&StoredSimilarityInfo>)>;

    /// Clear all continuations from the collection.
    fn clear(&mut self);

    /// Check if the collection is empty.
    fn is_empty(&self) -> bool;

    /// Get the number of continuations in the collection.
    fn len(&self) -> usize;
}
