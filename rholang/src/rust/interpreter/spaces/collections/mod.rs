//! Layer 1: Inner Collection Types
//!
//! This module defines the traits and implementations for how data and
//! continuations are stored and matched at individual channels.
//!
//! Different collection types provide different semantics:
//! - **Bag**: Multiset, any element can match (non-deterministic)
//! - **Queue**: FIFO, only head can match (deterministic)
//! - **Stack**: LIFO, only top can match
//! - **Set**: Unique elements, idempotent sends
//! - **Cell**: At most one element, error on second send
//! - **PriorityQueue**: Priority-based matching
//! - **VectorDB**: Similarity-based matching

// Submodules
pub mod semantics;
pub mod similarity;
pub mod core;
pub mod extensions;
pub mod storage;
pub mod basic;
pub mod continuations;
pub mod vectordb_coll;
pub mod lazy;

// Re-exports from submodules
pub use semantics::{SemanticEq, SemanticHash};
pub(crate) use semantics::TopKEntry;
pub use core::{DataCollection, ContinuationCollection};
pub use similarity::{SimilarityCollection, StoredSimilarityInfo, ContinuationId, SimilarityQueryMatrix};
pub use extensions::{DataCollectionExt, ContinuationCollectionExt};
pub use storage::{SmartDataStorage, SmartIterator};
pub use basic::{
    BagDataCollection, QueueDataCollection, StackDataCollection,
    SetDataCollection, CellDataCollection, PriorityQueueDataCollection,
};
pub use continuations::{
    BagContinuationCollection, QueueContinuationCollection,
    StackContinuationCollection, SetContinuationCollection,
};
pub use vectordb_coll::VectorDBDataCollection;
pub use lazy::LazyResultProducer;

// ============================================================================
// VectorDB Types
// ============================================================================
//
// Core embedding types (EmbeddingType, pack_to_binary) are defined in vectordb/types.rs.
// Similarity metrics, handlers, and registries are provided by the in-memory backend.
// This allows the backend to be the single source of truth for similarity computation.

// Core types from types.rs
pub use super::vectordb::types::{EmbeddingType, pack_to_binary};

// Similarity metrics, handlers, and registries from the in-memory backend
pub use super::vectordb::in_memory::{
    // Core metrics types
    SimilarityMetric, IndexConfig,
    // Handler types
    ResolvedArg, FunctionContext, SimilarityResult, RankingResult, IndexOptimizationData,
    // Handler traits
    SimilarityMetricHandler, RankingFunctionHandler,
    // Registries
    SimilarityMetricRegistry, RankingFunctionRegistry, FunctionHandlerRegistry,
};
pub use super::vectordb::in_memory::metrics::{HnswConfig, ScalarQuantizationConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::errors::SpaceError;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    #[test]
    fn test_bag_put_and_find() {
        let mut bag: BagDataCollection<i32> = BagDataCollection::new();
        bag.put(1).unwrap();
        bag.put(2).unwrap();
        bag.put(3).unwrap();

        assert_eq!(bag.len(), 3);
        let found = bag.find_and_remove(|&x| x == 2);
        assert_eq!(found, Some(2));
        assert_eq!(bag.len(), 2);
    }

    #[test]
    fn test_queue_fifo() {
        let mut queue: QueueDataCollection<i32> = QueueDataCollection::new();
        queue.put(1).unwrap();
        queue.put(2).unwrap();
        queue.put(3).unwrap();

        // Only front (1) can match
        let found = queue.find_and_remove(|&x| x == 2);
        assert_eq!(found, None); // 2 is not at front

        let found = queue.find_and_remove(|&x| x == 1);
        assert_eq!(found, Some(1)); // 1 is at front
    }

    #[test]
    fn test_stack_lifo() {
        let mut stack: StackDataCollection<i32> = StackDataCollection::new();
        stack.put(1).unwrap();
        stack.put(2).unwrap();
        stack.put(3).unwrap();

        // Only top (3) can match
        let found = stack.find_and_remove(|&x| x == 1);
        assert_eq!(found, None); // 1 is not at top

        let found = stack.find_and_remove(|&x| x == 3);
        assert_eq!(found, Some(3)); // 3 is at top
    }

    #[test]
    fn test_set_idempotent() {
        let mut set: SetDataCollection<i32> = SetDataCollection::new();
        set.put(1).unwrap();
        set.put(1).unwrap(); // Duplicate
        set.put(2).unwrap();

        assert_eq!(set.len(), 2); // Only 2 unique elements
    }

    #[test]
    fn test_cell_exactly_once() {
        let mut cell: CellDataCollection<i32> = CellDataCollection::new("test".to_string());
        cell.put(42).unwrap();

        let result = cell.put(43);
        assert!(matches!(result, Err(SpaceError::CellAlreadyFull { .. })));
    }

    #[test]
    fn test_cell_after_consume() {
        let mut cell: CellDataCollection<i32> = CellDataCollection::new("test".to_string());
        cell.put(42).unwrap();

        let found = cell.find_and_remove(|_| true);
        assert_eq!(found, Some(42));

        // Now we can put again
        cell.put(43).unwrap();
        assert_eq!(cell.len(), 1);
    }

    // ======================================================================
    // PriorityQueueDataCollection tests
    // ======================================================================

    #[test]
    fn test_priority_queue_basic() {
        let mut pq: PriorityQueueDataCollection<i32> = PriorityQueueDataCollection::new(3);
        assert_eq!(pq.num_priorities(), 3);
        assert!(pq.is_empty());

        pq.put(42).unwrap();
        assert_eq!(pq.len(), 1);
        assert!(!pq.is_empty());
    }

    #[test]
    fn test_priority_queue_highest_priority_first() {
        let mut pq: PriorityQueueDataCollection<&str> = PriorityQueueDataCollection::new(3);

        // Insert low priority first, then high priority
        pq.put_with_priority("low", 2).unwrap();
        pq.put_with_priority("medium", 1).unwrap();
        pq.put_with_priority("high", 0).unwrap();

        assert_eq!(pq.len(), 3);

        // Should match highest priority (0) first
        let found = pq.find_and_remove(|_| true);
        assert_eq!(found, Some("high"));

        let found = pq.find_and_remove(|_| true);
        assert_eq!(found, Some("medium"));

        let found = pq.find_and_remove(|_| true);
        assert_eq!(found, Some("low"));

        assert!(pq.is_empty());
    }

    #[test]
    fn test_priority_queue_fifo_within_priority() {
        let mut pq: PriorityQueueDataCollection<i32> = PriorityQueueDataCollection::new(3);

        // Insert multiple items at same priority
        pq.put_with_priority(1, 1).unwrap();
        pq.put_with_priority(2, 1).unwrap();
        pq.put_with_priority(3, 1).unwrap();

        // Should match in FIFO order within same priority
        assert_eq!(pq.find_and_remove(|_| true), Some(1));
        assert_eq!(pq.find_and_remove(|_| true), Some(2));
        assert_eq!(pq.find_and_remove(|_| true), Some(3));
    }

    #[test]
    fn test_priority_queue_counts() {
        let mut pq: PriorityQueueDataCollection<i32> = PriorityQueueDataCollection::new(3);

        pq.put_with_priority(1, 0).unwrap();
        pq.put_with_priority(2, 0).unwrap();
        pq.put_with_priority(3, 1).unwrap();
        pq.put_with_priority(4, 2).unwrap();
        pq.put_with_priority(5, 2).unwrap();
        pq.put_with_priority(6, 2).unwrap();

        let counts = pq.counts_by_priority();
        assert_eq!(counts, vec![2, 1, 3]);
        assert_eq!(pq.len(), 6);
    }

    #[test]
    fn test_priority_queue_default_priority() {
        let mut pq: PriorityQueueDataCollection<i32> = PriorityQueueDataCollection::new(3);

        // Default put goes to lowest priority (2)
        pq.put(42).unwrap();
        assert_eq!(pq.counts_by_priority(), vec![0, 0, 1]);

        // Change default priority
        pq.set_default_priority(0);
        pq.put(43).unwrap();
        assert_eq!(pq.counts_by_priority(), vec![1, 0, 1]);
    }

    #[test]
    fn test_priority_queue_predicate_matching() {
        let mut pq: PriorityQueueDataCollection<i32> = PriorityQueueDataCollection::new(2);

        pq.put_with_priority(10, 0).unwrap();
        pq.put_with_priority(20, 0).unwrap();
        pq.put_with_priority(15, 1).unwrap();

        // Predicate must match at front of a queue (any priority level).
        // Priority 0 front is 10 (fails), but priority 1 front is 15 (passes).
        // The implementation checks all priority levels from highest to lowest,
        // returning the first match at the front of any queue.
        let found = pq.find_and_remove(|&x| x > 12);
        assert_eq!(found, Some(15)); // 15 is at front of priority 1 queue

        // Now priority 1 is empty, priority 0 has [10, 20]
        // 10 doesn't match, so no match found
        let found = pq.find_and_remove(|&x| x > 12);
        assert_eq!(found, None);

        // Match the front of priority 0 (10)
        let found = pq.find_and_remove(|&x| x == 10);
        assert_eq!(found, Some(10));

        // Now 20 is at front of priority 0
        let found = pq.find_and_remove(|&x| x > 12);
        assert_eq!(found, Some(20));
    }

    #[test]
    fn test_priority_queue_peek() {
        let mut pq: PriorityQueueDataCollection<i32> = PriorityQueueDataCollection::new(2);

        pq.put_with_priority(100, 0).unwrap();
        pq.put_with_priority(50, 1).unwrap();

        // Peek should find highest priority
        let peeked = pq.peek(|_| true);
        assert_eq!(peeked, Some(&100));

        // Length unchanged after peek
        assert_eq!(pq.len(), 2);
    }

    #[test]
    fn test_priority_queue_clear() {
        let mut pq: PriorityQueueDataCollection<i32> = PriorityQueueDataCollection::new(3);

        pq.put_with_priority(1, 0).unwrap();
        pq.put_with_priority(2, 1).unwrap();
        pq.put_with_priority(3, 2).unwrap();

        pq.clear();
        assert!(pq.is_empty());
        assert_eq!(pq.len(), 0);
        assert_eq!(pq.counts_by_priority(), vec![0, 0, 0]);
    }

    #[test]
    fn test_priority_queue_dynamic_growth() {
        // Start with a small queue (2 priorities: 0 and 1)
        let mut pq: PriorityQueueDataCollection<&str> = PriorityQueueDataCollection::new(2);
        assert_eq!(pq.num_priorities(), 2);

        // Insert at priority 0 and 1 (within initial capacity)
        pq.put_with_priority("high", 0).unwrap();
        pq.put_with_priority("medium", 1).unwrap();
        assert_eq!(pq.num_priorities(), 2);

        // Insert at priority 5 - should grow dynamically to 6 levels (0-5)
        pq.put_with_priority("very_low", 5).unwrap();
        assert_eq!(pq.num_priorities(), 6);
        assert_eq!(pq.counts_by_priority(), vec![1, 1, 0, 0, 0, 1]);

        // Insert at priority 3 (now within capacity)
        pq.put_with_priority("low", 3).unwrap();
        assert_eq!(pq.num_priorities(), 6); // No growth needed
        assert_eq!(pq.counts_by_priority(), vec![1, 1, 0, 1, 0, 1]);

        // Verify retrieval order: highest priority (0) first
        assert_eq!(pq.find_and_remove(|_| true), Some("high"));
        assert_eq!(pq.find_and_remove(|_| true), Some("medium"));
        assert_eq!(pq.find_and_remove(|_| true), Some("low"));
        assert_eq!(pq.find_and_remove(|_| true), Some("very_low"));
        assert!(pq.is_empty());
    }

    #[test]
    fn test_priority_queue_dynamic_growth_default_priority() {
        let mut pq: PriorityQueueDataCollection<i32> = PriorityQueueDataCollection::new(2);
        assert_eq!(pq.num_priorities(), 2);

        // Set default priority beyond current capacity
        pq.set_default_priority(4);
        assert_eq!(pq.num_priorities(), 5); // Grew to accommodate priority 4

        // put() should use the new default priority
        pq.put(42).unwrap();
        assert_eq!(pq.counts_by_priority(), vec![0, 0, 0, 0, 1]);
    }

    // ======================================================================
    // VectorDBDataCollection tests
    // ======================================================================

    #[test]
    fn test_vectordb_basic() {
        let vdb: VectorDBDataCollection<String> = VectorDBDataCollection::new(4);
        assert_eq!(vdb.dimensions(), 4);
        assert!(vdb.is_empty());
        assert_eq!(vdb.threshold(), 0.8);
    }

    #[test]
    fn test_vectordb_with_threshold() {
        let vdb: VectorDBDataCollection<i32> = VectorDBDataCollection::with_threshold(8, 0.5);
        assert_eq!(vdb.dimensions(), 8);
        assert_eq!(vdb.threshold(), 0.5);
    }

    #[test]
    fn test_vectordb_put_with_embedding() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::new(3);

        let result = vdb.put_with_embedding("hello".to_string(), vec![1.0, 0.0, 0.0]);
        assert!(result.is_ok());
        assert_eq!(vdb.len(), 1);
    }

    #[test]
    fn test_vectordb_dimension_mismatch() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::new(3);

        // Wrong dimension count
        let result = vdb.put_with_embedding("hello".to_string(), vec![1.0, 0.0]);
        assert!(matches!(result, Err(SpaceError::InvalidConfiguration { .. })));
    }

    #[test]
    fn test_vectordb_cosine_similarity_identical() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.9);

        vdb.put_with_embedding("hello".to_string(), vec![1.0, 0.0, 0.0]).unwrap();

        // Query with identical vector should have similarity 1.0
        let result = vdb.find_most_similar(&[1.0, 0.0, 0.0]);
        assert!(result.is_some());
        let (data, similarity) = result.unwrap();
        assert_eq!(data, &"hello".to_string());
        assert!((similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_vectordb_cosine_similarity_orthogonal() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.5);

        vdb.put_with_embedding("x_axis".to_string(), vec![1.0, 0.0, 0.0]).unwrap();

        // Query with orthogonal vector should have similarity 0.0
        let result = vdb.find_most_similar(&[0.0, 1.0, 0.0]);
        assert!(result.is_none()); // Below threshold
    }

    #[test]
    fn test_vectordb_finds_most_similar() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.5);

        // Insert three vectors
        vdb.put_with_embedding("x".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        vdb.put_with_embedding("y".to_string(), vec![0.0, 1.0, 0.0]).unwrap();
        vdb.put_with_embedding("xy".to_string(), vec![0.707, 0.707, 0.0]).unwrap();

        // Query should find "xy" as most similar to a vector between x and y
        let result = vdb.find_most_similar(&[0.5, 0.5, 0.0]);
        assert!(result.is_some());
        let (data, _) = result.unwrap();
        assert_eq!(data, &"xy".to_string());
    }

    #[test]
    fn test_vectordb_find_and_remove_most_similar() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.9);

        vdb.put_with_embedding("target".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        vdb.put_with_embedding("other".to_string(), vec![0.0, 1.0, 0.0]).unwrap();

        assert_eq!(vdb.len(), 2);

        let result = vdb.find_and_remove_most_similar(&[1.0, 0.0, 0.0]);
        assert!(result.is_some());
        let (data, _) = result.unwrap();
        assert_eq!(data, "target".to_string());

        // Should be removed
        assert_eq!(vdb.len(), 1);
    }

    #[test]
    fn test_vectordb_threshold_filtering() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(2, 0.95);

        // Insert a vector
        vdb.put_with_embedding("strict".to_string(), vec![1.0, 0.0]).unwrap();

        // Query with slightly different vector (high but not 0.95 similarity)
        // Using default threshold of 0.95
        let result = vdb.find_most_similar(&[0.9, 0.436]); // ~0.9 similarity
        assert!(result.is_none()); // Below 0.95 threshold

        // Query again with a lower threshold using per-query threshold method
        let result = vdb.find_most_similar_with_threshold(&[0.9, 0.436], 0.8);
        assert!(result.is_some()); // Now above 0.8 threshold
    }

    // ==========================================================================
    // Top-K Similarity Tests
    // ==========================================================================

    #[test]
    fn test_vectordb_find_top_k_basic() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.5);

        // Insert 5 documents with different embeddings
        vdb.put_with_embedding("doc1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        vdb.put_with_embedding("doc2".to_string(), vec![0.9, 0.1, 0.0]).unwrap();
        vdb.put_with_embedding("doc3".to_string(), vec![0.8, 0.2, 0.0]).unwrap();
        vdb.put_with_embedding("doc4".to_string(), vec![0.0, 1.0, 0.0]).unwrap();
        vdb.put_with_embedding("doc5".to_string(), vec![0.0, 0.0, 1.0]).unwrap();

        // Query for top 3 most similar to [1.0, 0.0, 0.0]
        let results = vdb.find_top_k_similar(&[1.0, 0.0, 0.0], 0.5, 3);

        // Should return 3 results
        assert_eq!(results.len(), 3);

        // Results should be sorted by similarity (highest first)
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);

        // First result should be doc1 (exact match)
        assert_eq!(*results[0].0, "doc1");
        assert!((results[0].1 - 1.0).abs() < 0.001); // ~1.0 similarity

        // All results should have similarity >= 0.5 (threshold)
        for (_, sim) in &results {
            assert!(*sim >= 0.5);
        }
    }

    #[test]
    fn test_vectordb_find_top_k_with_threshold_filtering() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.9);

        // Insert documents
        vdb.put_with_embedding("high_sim".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        vdb.put_with_embedding("medium_sim".to_string(), vec![0.8, 0.6, 0.0]).unwrap();
        vdb.put_with_embedding("low_sim".to_string(), vec![0.5, 0.5, 0.707]).unwrap();

        // Request top 3 but with high threshold (0.9)
        let results = vdb.find_top_k_similar(&[1.0, 0.0, 0.0], 0.9, 3);

        // Only high_sim should match (similarity = 1.0)
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].0, "high_sim");
    }

    #[test]
    fn test_vectordb_find_top_k_k_larger_than_matches() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.5);

        // Insert only 2 documents that match
        vdb.put_with_embedding("doc1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        vdb.put_with_embedding("doc2".to_string(), vec![0.9, 0.1, 0.0]).unwrap();

        // Request top 5 but only 2 match
        let results = vdb.find_top_k_similar(&[1.0, 0.0, 0.0], 0.5, 5);

        // Should return only 2 (all that match)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_vectordb_find_top_k_k_equals_one() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.5);

        vdb.put_with_embedding("best".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        vdb.put_with_embedding("second".to_string(), vec![0.9, 0.1, 0.0]).unwrap();
        vdb.put_with_embedding("third".to_string(), vec![0.8, 0.2, 0.0]).unwrap();

        // Request top 1
        let results = vdb.find_top_k_similar(&[1.0, 0.0, 0.0], 0.5, 1);

        // Should return exactly 1 result (the best match)
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].0, "best");
    }

    #[test]
    fn test_vectordb_find_top_k_empty_collection() {
        let vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.5);

        let results = vdb.find_top_k_similar(&[1.0, 0.0, 0.0], 0.5, 3);

        assert!(results.is_empty());
    }

    #[test]
    fn test_vectordb_find_top_k_k_equals_zero() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.5);

        vdb.put_with_embedding("doc1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();

        // Request top 0
        let results = vdb.find_top_k_similar(&[1.0, 0.0, 0.0], 0.5, 0);

        assert!(results.is_empty());
    }

    #[test]
    fn test_vectordb_find_and_remove_top_k_basic() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.5);

        vdb.put_with_embedding("doc1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        vdb.put_with_embedding("doc2".to_string(), vec![0.9, 0.1, 0.0]).unwrap();
        vdb.put_with_embedding("doc3".to_string(), vec![0.8, 0.2, 0.0]).unwrap();
        vdb.put_with_embedding("unrelated".to_string(), vec![0.0, 0.0, 1.0]).unwrap();

        assert_eq!(vdb.len(), 4);

        // Remove top 2
        let results = vdb.find_and_remove_top_k_similar(&[1.0, 0.0, 0.0], 0.5, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "doc1");
        assert_eq!(results[1].0, "doc2");

        // Should have removed 2 documents
        assert_eq!(vdb.len(), 2);

        // The removed documents should no longer be findable
        let remaining = vdb.find_top_k_similar(&[1.0, 0.0, 0.0], 0.5, 10);
        assert_eq!(remaining.len(), 1);
        assert_eq!(*remaining[0].0, "doc3");
    }

    #[test]
    fn test_vectordb_find_and_remove_top_k_with_slot_reuse() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.5);

        // Insert initial documents
        vdb.put_with_embedding("a".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        vdb.put_with_embedding("b".to_string(), vec![0.0, 1.0, 0.0]).unwrap();

        // Remove one
        let removed = vdb.find_and_remove_top_k_similar(&[1.0, 0.0, 0.0], 0.5, 1);
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].0, "a");

        // Insert new document - should reuse the freed slot
        vdb.put_with_embedding("c".to_string(), vec![0.9, 0.1, 0.0]).unwrap();

        // len() should report live count correctly
        assert_eq!(vdb.len(), 2);

        // Find all similar - should find b and c, not a
        let results = vdb.find_top_k_similar(&[1.0, 0.0, 0.0], 0.3, 10);
        let names: Vec<&str> = results.iter().map(|(s, _)| s.as_str()).collect();
        assert!(!names.contains(&"a"));
        assert!(names.contains(&"c") || names.contains(&"b"));
    }

    #[test]
    fn test_vectordb_find_and_remove_top_k_respects_persist_flag() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.5);

        // Insert with persist=true
        vdb.put_with_embedding_and_persist("persistent".to_string(), vec![1.0, 0.0, 0.0], true).unwrap();
        // Insert with persist=false
        vdb.put_with_embedding_and_persist("consumable".to_string(), vec![0.9, 0.1, 0.0], false).unwrap();

        // Remove top 2 - should only remove the non-persistent one
        let results = vdb.find_and_remove_top_k_similar(&[1.0, 0.0, 0.0], 0.5, 2);

        // Both are returned in the results
        assert_eq!(results.len(), 2);

        // But persistent doc should still be in the collection
        assert_eq!(vdb.len(), 1);

        let remaining = vdb.find_top_k_similar(&[1.0, 0.0, 0.0], 0.5, 10);
        assert_eq!(remaining.len(), 1);
        assert_eq!(*remaining[0].0, "persistent");
    }

    #[test]
    fn test_vectordb_top_k_ordering_is_descending() {
        let mut vdb: VectorDBDataCollection<i32> = VectorDBDataCollection::with_threshold(3, 0.3);

        // Insert 10 documents with varying similarities
        for i in 0..10 {
            let angle = (i as f32) * 0.1; // Increasing angle from [1,0,0]
            let emb = vec![angle.cos(), angle.sin(), 0.0];
            vdb.put_with_embedding(i, emb).unwrap();
        }

        // Get top 5
        let results = vdb.find_top_k_similar(&[1.0, 0.0, 0.0], 0.3, 5);

        assert_eq!(results.len(), 5);

        // Verify descending order
        for i in 0..results.len() - 1 {
            assert!(
                results[i].1 >= results[i + 1].1,
                "Results not sorted: {} ({}) should be >= {} ({})",
                results[i].0, results[i].1, results[i + 1].0, results[i + 1].1
            );
        }
    }

    #[test]
    fn test_vectordb_predicate_fallback() {
        let mut vdb: VectorDBDataCollection<i32> = VectorDBDataCollection::new(2);

        vdb.put_with_embedding(10, vec![1.0, 0.0]).unwrap();
        vdb.put_with_embedding(20, vec![0.0, 1.0]).unwrap();
        vdb.put_with_embedding(30, vec![0.5, 0.5]).unwrap();

        // Use predicate-based matching (fallback)
        let found = vdb.find_and_remove(|&x| x == 20);
        assert_eq!(found, Some(20));
        assert_eq!(vdb.len(), 2);
    }

    #[test]
    fn test_vectordb_all_data() {
        let mut vdb: VectorDBDataCollection<i32> = VectorDBDataCollection::new(2);

        vdb.put_with_embedding(1, vec![1.0, 0.0]).unwrap();
        vdb.put_with_embedding(2, vec![0.0, 1.0]).unwrap();

        let all = vdb.all_data();
        assert_eq!(all.len(), 2);
        assert!(all.contains(&&1));
        assert!(all.contains(&&2));
    }

    #[test]
    fn test_vectordb_clear() {
        let mut vdb: VectorDBDataCollection<i32> = VectorDBDataCollection::new(2);

        vdb.put_with_embedding(1, vec![1.0, 0.0]).unwrap();
        vdb.put_with_embedding(2, vec![0.0, 1.0]).unwrap();

        vdb.clear();
        assert!(vdb.is_empty());
        assert_eq!(vdb.len(), 0);
    }

    #[test]
    fn test_vectordb_normalized_vectors() {
        let mut vdb: VectorDBDataCollection<String> = VectorDBDataCollection::with_threshold(3, 0.99);

        // Normalized vector
        vdb.put_with_embedding("normalized".to_string(), vec![0.577, 0.577, 0.577]).unwrap();

        // Un-normalized query (same direction, different magnitude) should still match
        let result = vdb.find_most_similar(&[2.0, 2.0, 2.0]);
        assert!(result.is_some());
        let (_, similarity) = result.unwrap();
        assert!((similarity - 1.0).abs() < 0.01); // Should be ~1.0 (same direction)
    }

    // ==========================================================================
    // Extension Trait Tests
    // ==========================================================================

    use super::super::matcher::{ExactMatch, WildcardMatch};

    #[test]
    fn test_data_collection_ext_find_with_matcher() {
        let mut bag: BagDataCollection<i32> = BagDataCollection::new();
        bag.put(10).expect("put");
        bag.put(20).expect("put");
        bag.put(30).expect("put");

        let exact_matcher: ExactMatch<i32> = ExactMatch::new();

        // Find exact match for 20
        let found = bag.find_and_remove_with_matcher(&20, &exact_matcher);
        assert_eq!(found, Some(20));
        assert_eq!(bag.len(), 2);

        // No match for 99
        let not_found = bag.find_and_remove_with_matcher(&99, &exact_matcher);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_data_collection_ext_peek_with_matcher() {
        let mut bag: BagDataCollection<i32> = BagDataCollection::new();
        bag.put(10).expect("put");
        bag.put(20).expect("put");

        let exact_matcher: ExactMatch<i32> = ExactMatch::new();

        // Peek without removing
        let found = bag.peek_with_matcher(&20, &exact_matcher);
        assert_eq!(found, Some(&20));
        assert_eq!(bag.len(), 2); // Not removed

        // Now remove it
        let removed = bag.find_and_remove_with_matcher(&20, &exact_matcher);
        assert_eq!(removed, Some(20));
        assert_eq!(bag.len(), 1);
    }

    #[test]
    fn test_data_collection_ext_with_wildcard_matcher() {
        let mut bag: BagDataCollection<i32> = BagDataCollection::new();
        bag.put(10).expect("put");
        bag.put(20).expect("put");
        bag.put(30).expect("put");

        let wildcard_matcher: WildcardMatch<i32, i32> = WildcardMatch::new();

        // Wildcard matches anything, so it returns the first element it finds
        let found = bag.find_and_remove_with_matcher(&0, &wildcard_matcher);
        assert!(found.is_some());
        assert_eq!(bag.len(), 2);
    }

    #[test]
    fn test_continuation_collection_ext_find_matching() {
        let mut conts: BagContinuationCollection<i32, String> = BagContinuationCollection::new();
        conts.put(vec![10, 20], "cont1".to_string(), false);
        conts.put(vec![30, 40], "cont2".to_string(), false);

        let exact_matcher: ExactMatch<i32> = ExactMatch::new();

        // Find continuation whose pattern matches data=10
        let found = conts.find_matching_for_data(&10, &exact_matcher);
        assert!(found.is_some());
        let (patterns, cont, persist) = found.unwrap();
        assert!(patterns.contains(&10));
        assert_eq!(cont, "cont1");
        assert!(!persist);

        // Find continuation whose pattern matches data=30
        let found2 = conts.find_matching_for_data(&30, &exact_matcher);
        assert!(found2.is_some());
        let (patterns2, cont2, _) = found2.unwrap();
        assert!(patterns2.contains(&30));
        assert_eq!(cont2, "cont2");
    }

    #[test]
    fn test_continuation_collection_ext_peek_matching() {
        let mut conts: BagContinuationCollection<i32, String> = BagContinuationCollection::new();
        conts.put(vec![10, 20], "cont1".to_string(), true);

        let exact_matcher: ExactMatch<i32> = ExactMatch::new();

        // Peek without removing
        let peeked = conts.peek_matching_for_data(&10, &exact_matcher);
        assert!(peeked.is_some());
        let (patterns, cont, persist) = peeked.unwrap();
        assert!(patterns.contains(&10));
        assert_eq!(*cont, "cont1");
        assert!(persist);

        // Collection still has the continuation
        assert_eq!(conts.len(), 1);
    }

    // ==========================================================================
    // Stack Continuation Collection Tests
    // ==========================================================================

    #[test]
    fn test_stack_continuation_basic() {
        let mut stack: StackContinuationCollection<i32, String> = StackContinuationCollection::new();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);

        stack.put(vec![1], "first".to_string(), false);
        assert_eq!(stack.len(), 1);
        assert!(!stack.is_empty());
    }

    #[test]
    fn test_stack_continuation_lifo() {
        let mut stack: StackContinuationCollection<i32, String> = StackContinuationCollection::new();
        stack.put(vec![1], "first".to_string(), false);
        stack.put(vec![2], "second".to_string(), false);
        stack.put(vec![3], "third".to_string(), false);

        assert_eq!(stack.len(), 3);

        // Only top (third) can match
        let found = stack.find_and_remove(|_, k| k == "first");
        assert!(found.is_none()); // first is not at top

        let found = stack.find_and_remove(|_, k| k == "second");
        assert!(found.is_none()); // second is not at top

        // Third is at top
        let found = stack.find_and_remove(|_, k| k == "third");
        assert!(found.is_some());
        let (patterns, cont, persist) = found.expect("third should be found");
        assert_eq!(patterns, vec![3]);
        assert_eq!(cont, "third");
        assert!(!persist);

        // Now second is at top
        assert_eq!(stack.len(), 2);
        let found = stack.find_and_remove(|_, k| k == "second");
        assert!(found.is_some());
    }

    #[test]
    fn test_stack_continuation_all_continuations() {
        let mut stack: StackContinuationCollection<i32, String> = StackContinuationCollection::new();
        stack.put(vec![1, 2], "cont1".to_string(), true);
        stack.put(vec![3, 4], "cont2".to_string(), false);

        let all = stack.all_continuations();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_stack_continuation_clear() {
        let mut stack: StackContinuationCollection<i32, String> = StackContinuationCollection::new();
        stack.put(vec![1], "a".to_string(), false);
        stack.put(vec![2], "b".to_string(), false);

        stack.clear();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
    }

    // ==========================================================================
    // Set Continuation Collection Tests
    // ==========================================================================

    #[test]
    fn test_set_continuation_basic() {
        let mut set: SetContinuationCollection<i32, String> = SetContinuationCollection::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        set.put(vec![1], "cont".to_string(), false);
        assert_eq!(set.len(), 1);
        assert!(!set.is_empty());
    }

    #[test]
    fn test_set_continuation_idempotent() {
        let mut set: SetContinuationCollection<i32, String> = SetContinuationCollection::new();
        set.put(vec![1], "cont".to_string(), false);
        set.put(vec![1], "cont".to_string(), false); // Exact duplicate
        set.put(vec![2], "other".to_string(), false);

        assert_eq!(set.len(), 2); // Only 2 unique entries
    }

    #[test]
    fn test_set_continuation_different_persist_not_duplicate() {
        let mut set: SetContinuationCollection<i32, String> = SetContinuationCollection::new();
        set.put(vec![1], "cont".to_string(), false);
        set.put(vec![1], "cont".to_string(), true); // Same patterns/cont, different persist

        assert_eq!(set.len(), 2); // Different persist flag means not duplicate
    }

    #[test]
    fn test_set_continuation_find_and_remove() {
        let mut set: SetContinuationCollection<i32, String> = SetContinuationCollection::new();
        set.put(vec![1], "first".to_string(), false);
        set.put(vec![2], "second".to_string(), false);
        set.put(vec![3], "third".to_string(), false);

        // Can find and remove any matching continuation (non-deterministic order like Bag)
        let found = set.find_and_remove(|_, k| k == "second");
        assert!(found.is_some());
        let (patterns, cont, _) = found.expect("second should be found");
        assert_eq!(patterns, vec![2]);
        assert_eq!(cont, "second");

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_set_continuation_all_continuations() {
        let mut set: SetContinuationCollection<i32, String> = SetContinuationCollection::new();
        set.put(vec![1], "a".to_string(), true);
        set.put(vec![2], "b".to_string(), false);

        let all = set.all_continuations();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_set_continuation_clear() {
        let mut set: SetContinuationCollection<i32, String> = SetContinuationCollection::new();
        set.put(vec![1], "a".to_string(), false);
        set.put(vec![2], "b".to_string(), false);

        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    // ==========================================================================
    // Additional Comprehensive Tests for Data Collections
    // ==========================================================================

    #[test]
    fn test_bag_empty_operations() {
        let mut bag: BagDataCollection<i32> = BagDataCollection::new();
        assert!(bag.find_and_remove(|_| true).is_none());
        assert!(bag.peek(|_| true).is_none());
        assert!(bag.all_data().is_empty());
    }

    #[test]
    fn test_queue_strict_fifo_ordering() {
        let mut queue: QueueDataCollection<i32> = QueueDataCollection::new();
        for i in 1..=5 {
            queue.put(i).expect("put should succeed");
        }

        // Must process in order: 1, 2, 3, 4, 5
        for i in 1..=5 {
            let found = queue.find_and_remove(|&x| x == i);
            assert_eq!(found, Some(i), "Queue should match {} at position {}", i, i);
        }
        assert!(queue.is_empty());
    }

    #[test]
    fn test_stack_strict_lifo_ordering() {
        let mut stack: StackDataCollection<i32> = StackDataCollection::new();
        for i in 1..=5 {
            stack.put(i).expect("put should succeed");
        }

        // Must process in reverse order: 5, 4, 3, 2, 1
        for i in (1..=5).rev() {
            let found = stack.find_and_remove(|&x| x == i);
            assert_eq!(found, Some(i), "Stack should match {} at position", i);
        }
        assert!(stack.is_empty());
    }

    #[test]
    fn test_set_with_complex_values() {
        #[derive(Clone, Hash, Eq, PartialEq, Debug)]
        struct ComplexValue {
            id: i32,
            name: String,
        }

        impl SemanticEq for ComplexValue {
            fn semantically_eq(&self, other: &Self) -> bool {
                self == other
            }
        }

        impl SemanticHash for ComplexValue {
            fn semantic_hash(&self) -> u64 {
                let mut hasher = DefaultHasher::new();
                self.id.hash(&mut hasher);
                self.name.hash(&mut hasher);
                hasher.finish()
            }
        }

        let mut set: SetDataCollection<ComplexValue> = SetDataCollection::new();
        let v1 = ComplexValue { id: 1, name: "first".to_string() };
        let v2 = ComplexValue { id: 1, name: "first".to_string() }; // Same as v1
        let v3 = ComplexValue { id: 2, name: "second".to_string() };

        set.put(v1.clone()).expect("put should succeed");
        set.put(v2.clone()).expect("put should succeed"); // Duplicate
        set.put(v3.clone()).expect("put should succeed");

        assert_eq!(set.len(), 2); // Only 2 unique
    }

    #[test]
    fn test_cell_capacity_invariant() {
        let mut cell: CellDataCollection<String> = CellDataCollection::new("test".to_string());

        // Can put one
        assert!(cell.put("first".to_string()).is_ok());
        assert_eq!(cell.len(), 1);

        // Can't put second
        assert!(cell.put("second".to_string()).is_err());
        assert_eq!(cell.len(), 1);

        // Consume
        let _ = cell.find_and_remove(|_| true);
        assert!(cell.is_empty());

        // Can put again after consume
        assert!(cell.put("third".to_string()).is_ok());
        assert_eq!(cell.len(), 1);
    }

    // ==========================================================================
    // Additional Comprehensive Tests for Continuation Collections
    // ==========================================================================

    #[test]
    fn test_bag_continuation_default() {
        let bag: BagContinuationCollection<i32, String> = Default::default();
        assert!(bag.is_empty());
    }

    #[test]
    fn test_queue_continuation_strict_fifo() {
        let mut queue: QueueContinuationCollection<i32, String> = QueueContinuationCollection::new();
        queue.put(vec![1], "first".to_string(), false);
        queue.put(vec![2], "second".to_string(), false);
        queue.put(vec![3], "third".to_string(), false);

        // Must process in order: first, second, third
        let found = queue.find_and_remove(|_, k| k == "first");
        assert!(found.is_some());
        assert_eq!(found.expect("first").1, "first");

        let found = queue.find_and_remove(|_, k| k == "second");
        assert!(found.is_some());
        assert_eq!(found.expect("second").1, "second");

        let found = queue.find_and_remove(|_, k| k == "third");
        assert!(found.is_some());
        assert_eq!(found.expect("third").1, "third");
    }

    #[test]
    fn test_stack_continuation_default() {
        let stack: StackContinuationCollection<i32, String> = Default::default();
        assert!(stack.is_empty());
    }

    #[test]
    fn test_stack_continuation_persist_flag() {
        let mut stack: StackContinuationCollection<i32, String> = StackContinuationCollection::new();
        stack.put(vec![1], "persistent".to_string(), true);
        stack.put(vec![2], "transient".to_string(), false);

        // Top is transient
        let found = stack.find_and_remove(|_, _| true);
        assert!(found.is_some());
        let (_, _, persist) = found.expect("top");
        assert!(!persist);

        // Now persistent is at top
        let found = stack.find_and_remove(|_, _| true);
        assert!(found.is_some());
        let (_, _, persist) = found.expect("top");
        assert!(persist);
    }

    #[test]
    fn test_set_continuation_default() {
        let set: SetContinuationCollection<i32, String> = Default::default();
        assert!(set.is_empty());
    }

    #[test]
    fn test_set_continuation_no_match_returns_none() {
        let mut set: SetContinuationCollection<i32, String> = SetContinuationCollection::new();
        set.put(vec![1], "cont".to_string(), false);

        let found = set.find_and_remove(|_, k| k == "nonexistent");
        assert!(found.is_none());
        assert_eq!(set.len(), 1); // Still has the original
    }

    #[test]
    fn test_stack_continuation_no_match_when_predicate_fails() {
        let mut stack: StackContinuationCollection<i32, String> = StackContinuationCollection::new();
        stack.put(vec![1], "first".to_string(), false);
        stack.put(vec![2], "second".to_string(), false);

        // Try to match first (not at top)
        let found = stack.find_and_remove(|_, k| k == "first");
        assert!(found.is_none()); // Can't match because not at top

        // Try with always-false predicate
        let found = stack.find_and_remove(|_, _| false);
        assert!(found.is_none()); // Predicate fails

        assert_eq!(stack.len(), 2); // Nothing removed
    }

    // ==========================================================================
    // Extension Trait Tests for New Continuation Collections
    // ==========================================================================

    #[test]
    fn test_stack_continuation_ext_find_matching() {
        let mut stack: StackContinuationCollection<i32, String> = StackContinuationCollection::new();
        stack.put(vec![10], "bottom".to_string(), false);
        stack.put(vec![20], "top".to_string(), false);

        let exact_matcher: ExactMatch<i32> = ExactMatch::new();

        // Can only find if pattern is at top
        // 10 is at bottom, but top has 20 - so matching for data=10 should fail
        // because only top can match in stack
        let found = stack.find_matching_for_data(&10, &exact_matcher);
        assert!(found.is_none()); // 10 pattern is at bottom

        // 20 is at top
        let found = stack.find_matching_for_data(&20, &exact_matcher);
        assert!(found.is_some());
        assert_eq!(found.expect("top").1, "top");
    }

    #[test]
    fn test_set_continuation_ext_find_matching() {
        let mut set: SetContinuationCollection<i32, String> = SetContinuationCollection::new();
        set.put(vec![10, 20], "cont1".to_string(), false);
        set.put(vec![30, 40], "cont2".to_string(), false);

        let exact_matcher: ExactMatch<i32> = ExactMatch::new();

        // Can find any matching pattern (like Bag)
        let found = set.find_matching_for_data(&20, &exact_matcher);
        assert!(found.is_some());

        let found = set.find_matching_for_data(&40, &exact_matcher);
        assert!(found.is_some());
    }

    #[test]
    fn test_set_continuation_ext_peek_matching() {
        let mut set: SetContinuationCollection<i32, String> = SetContinuationCollection::new();
        set.put(vec![10], "only".to_string(), true);

        let exact_matcher: ExactMatch<i32> = ExactMatch::new();

        let peeked = set.peek_matching_for_data(&10, &exact_matcher);
        assert!(peeked.is_some());
        let (_, cont, persist) = peeked.expect("peeked");
        assert_eq!(*cont, "only");
        assert!(persist);

        // Still there after peek
        assert_eq!(set.len(), 1);
    }
}
