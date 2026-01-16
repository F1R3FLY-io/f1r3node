//! Property-Based Tests for Collection Semantics
//!
//! These tests verify the semantic guarantees of each collection type:
//! - Queue: FIFO ordering
//! - Stack: LIFO ordering
//! - Set: Idempotent insertions
//! - Cell: Exactly-once writes
//! - Bag: Multiset (default)
//! - PriorityQueue: Priority-based ordering
//! - VectorDB: Similarity-based matching
//!
//! # Rholang Correspondence
//!
//! Each collection type corresponds to a Rholang space URN:
//!
//! ```rholang
//! // Queue space - FIFO ordering
//! new QueueSpace(`rho:space:queue:hashmap:default`), queue in {
//!   QueueSpace!({}, *queue) |
//!   use queue {
//!     ch!(1) | ch!(2) | ch!(3) |
//!     for (@x <- ch) { stdout!(x) }  // Prints 1 (first in)
//!   }
//! }
//!
//! // Stack space - LIFO ordering
//! new StackSpace(`rho:space:stack:hashmap:default`), stack in {
//!   StackSpace!({}, *stack) |
//!   use stack {
//!     ch!(1) | ch!(2) | ch!(3) |
//!     for (@x <- ch) { stdout!(x) }  // Prints 3 (last in)
//!   }
//! }
//!
//! // Set space - idempotent
//! new SetSpace(`rho:space:set:hashmap:default`), set in {
//!   SetSpace!({}, *set) |
//!   use set {
//!     ch!(42) | ch!(42) | ch!(42) |  // Only one 42 stored
//!     for (@x <- ch) { stdout!(x) }  // Gets 42 once
//!   }
//! }
//!
//! // Cell space - exactly once
//! new CellSpace(`rho:space:cell:hashmap:default`), cell in {
//!   CellSpace!({}, *cell) |
//!   use cell {
//!     reply!(42) |        // Succeeds
//!     // reply!(43)       // Would fail: CellAlreadyFull
//!   }
//! }
//! ```

use std::collections::HashSet;

use proptest::prelude::*;
use proptest::collection::vec as prop_vec;

use rholang::rust::interpreter::spaces::{
    SpaceError,
};
use rholang::rust::interpreter::spaces::collections::{
    BagDataCollection, QueueDataCollection, StackDataCollection,
    SetDataCollection, CellDataCollection,
    PriorityQueueDataCollection, VectorDBDataCollection,
    DataCollection,
};

use super::test_utils::*;

// =============================================================================
// Queue FIFO Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Queue always returns elements in FIFO order.
    ///
    /// **Formal Reference**: DataCollection.v lines 426-595
    /// **Rholang**: `rho:space:queue` ensures first-in-first-out
    #[test]
    fn prop_queue_fifo_order(
        elements in prop_vec(any::<i32>(), 1..100)
    ) {
        let queue = queue_with_data(elements.clone());

        // Verify FIFO: first element should match first
        if let Some(first) = queue.peek(|_| true) {
            prop_assert_eq!(*first, elements[0],
                "Queue peek should return first inserted element");
        }
    }

    /// Queue removes elements in insertion order.
    #[test]
    fn prop_queue_fifo_removal(
        elements in prop_vec(any::<i32>(), 1..50)
    ) {
        let mut queue = queue_with_data(elements.clone());
        let mut removed = Vec::new();

        // Remove all elements
        while let Some(elem) = queue.find_and_remove(|_| true) {
            removed.push(elem);
        }

        // Should be in same order as inserted
        prop_assert_eq!(removed, elements,
            "Queue should remove elements in FIFO order");
    }

    /// Queue only matches at the head.
    #[test]
    fn prop_queue_only_head_matches(
        first in any::<i32>(),
        second in any::<i32>(),
    ) {
        let mut queue: QueueDataCollection<i32> = QueueDataCollection::new();
        queue.put(first).expect("put first");
        queue.put(second).expect("put second");

        // Try to match the second element specifically
        if first != second {
            let found = queue.find_and_remove(|&x| x == second);
            // Should NOT find it (not at head)
            prop_assert!(found.is_none() || found == Some(first),
                "Queue should only match at head, not middle elements");
        }
    }
}

// =============================================================================
// Stack LIFO Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Stack always returns elements in LIFO order.
    ///
    /// **Formal Reference**: DataCollection.v lines 597-765
    /// **Rholang**: `rho:space:stack` ensures last-in-first-out
    #[test]
    fn prop_stack_lifo_order(
        elements in prop_vec(any::<i32>(), 1..100)
    ) {
        let stack = stack_with_data(elements.clone());

        // Verify LIFO: last element should match first
        if let Some(top) = stack.peek(|_| true) {
            prop_assert_eq!(*top, *elements.last().unwrap(),
                "Stack peek should return last inserted element");
        }
    }

    /// Stack removes elements in reverse insertion order.
    #[test]
    fn prop_stack_lifo_removal(
        elements in prop_vec(any::<i32>(), 1..50)
    ) {
        let mut stack = stack_with_data(elements.clone());
        let mut removed = Vec::new();

        // Remove all elements
        while let Some(elem) = stack.find_and_remove(|_| true) {
            removed.push(elem);
        }

        // Should be in reverse order
        let expected: Vec<_> = elements.into_iter().rev().collect();
        prop_assert_eq!(removed, expected,
            "Stack should remove elements in LIFO order");
    }

    /// Stack only matches at the top.
    #[test]
    fn prop_stack_only_top_matches(
        first in any::<i32>(),
        second in any::<i32>(),
    ) {
        let mut stack: StackDataCollection<i32> = StackDataCollection::new();
        stack.put(first).expect("put first");
        stack.put(second).expect("put second");

        // Try to match the first element specifically
        if first != second {
            let found = stack.find_and_remove(|&x| x == first);
            // Should NOT find it (not at top)
            prop_assert!(found.is_none() || found == Some(second),
                "Stack should only match at top, not bottom elements");
        }
    }
}

// =============================================================================
// Set Idempotent Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Set insertions are idempotent - duplicates are ignored.
    ///
    /// **Formal Reference**: DataCollection.v lines 767-855
    /// **Rholang**: `rho:space:set` deduplicates automatically
    #[test]
    fn prop_set_idempotent(
        element in any::<i32>(),
        insert_count in 1usize..100,
    ) {
        let mut set: SetDataCollection<i32> = SetDataCollection::new();

        // Insert same element multiple times
        for _ in 0..insert_count {
            set.put(element).expect("put should succeed");
        }

        // Should only have one element
        prop_assert_eq!(set.len(), 1,
            "Set should have exactly 1 element after {} insertions of same value",
            insert_count);
    }

    /// Set contains only unique elements.
    #[test]
    fn prop_set_unique_elements(
        elements in prop_vec(any::<i32>(), 1..100),
    ) {
        let set = set_with_data(elements.clone());

        // Count unique elements in input
        let unique_input: HashSet<_> = elements.iter().cloned().collect();

        prop_assert_eq!(set.len(), unique_input.len(),
            "Set length should equal number of unique elements");
    }

    /// Removing from set removes the element completely.
    #[test]
    fn prop_set_removal_complete(
        elements in prop_vec(any::<i32>(), 1..20),
    ) {
        let mut set = set_with_data(elements.clone());
        let unique: HashSet<_> = elements.iter().cloned().collect();

        // Remove all elements
        let mut removed_count = 0;
        while let Some(_) = set.find_and_remove(|_| true) {
            removed_count += 1;
        }

        prop_assert_eq!(removed_count, unique.len(),
            "Should remove exactly the number of unique elements");
        prop_assert!(set.is_empty(), "Set should be empty after removing all");
    }
}

// =============================================================================
// Cell Exactly-Once Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Cell allows exactly one write - second write fails.
    ///
    /// **Formal Reference**: DataCollection.v lines 857-1028
    /// **Rholang**: `rho:space:cell` for one-shot reply channels
    #[test]
    fn prop_cell_exactly_once(
        first in any::<i32>(),
        second in any::<i32>(),
    ) {
        let mut cell: CellDataCollection<i32> = CellDataCollection::new("test".to_string());

        // First write succeeds
        let result1 = cell.put(first);
        prop_assert!(result1.is_ok(), "First write to cell should succeed");

        // Second write fails
        let result2 = cell.put(second);
        prop_assert!(result2.is_err(), "Second write to cell should fail");

        match result2 {
            Err(SpaceError::CellAlreadyFull { .. }) => {
                // Expected error
            }
            _ => prop_assert!(false, "Expected CellAlreadyFull error"),
        }

        // Cell still contains first value
        let peeked = cell.peek(|_| true);
        prop_assert_eq!(peeked, Some(&first),
            "Cell should contain first written value");
    }

    /// Cell can be written again after consumption.
    #[test]
    fn prop_cell_reusable_after_consume(
        first in any::<i32>(),
        second in any::<i32>(),
    ) {
        let mut cell: CellDataCollection<i32> = CellDataCollection::new("test".to_string());

        // First write
        cell.put(first).expect("first write");

        // Consume
        let consumed = cell.find_and_remove(|_| true);
        prop_assert_eq!(consumed, Some(first));

        // Now we can write again
        let result = cell.put(second);
        prop_assert!(result.is_ok(), "Should be able to write to cell after consume");

        // Verify second value
        let peeked = cell.peek(|_| true);
        prop_assert_eq!(peeked, Some(&second));
    }

    /// Empty cell has length 0.
    #[test]
    fn prop_cell_empty_len(
        value in any::<i32>(),
    ) {
        let mut cell: CellDataCollection<i32> = CellDataCollection::new("test".to_string());

        prop_assert_eq!(cell.len(), 0, "Empty cell should have length 0");
        prop_assert!(cell.is_empty());

        cell.put(value).expect("put");
        prop_assert_eq!(cell.len(), 1, "Cell with value should have length 1");
        prop_assert!(!cell.is_empty());

        cell.find_and_remove(|_| true);
        prop_assert_eq!(cell.len(), 0, "Cell after consume should have length 0");
        prop_assert!(cell.is_empty());
    }
}

// =============================================================================
// Bag Multiset Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Bag allows duplicate elements (multiset semantics).
    ///
    /// **Rholang**: Default collection type for `rho:space:hashmap:bag`
    #[test]
    fn prop_bag_allows_duplicates(
        element in any::<i32>(),
        count in 1usize..100,
    ) {
        let elements = vec![element; count];
        let bag = bag_with_data(elements);

        // All duplicates should be stored
        prop_assert_eq!(bag.len(), count,
            "Bag should store all {} duplicates", count);
    }

    /// Bag removal removes one element at a time.
    #[test]
    fn prop_bag_removal_one_at_a_time(
        element in any::<i32>(),
        count in 2usize..50,
    ) {
        let elements = vec![element; count];
        let mut bag = bag_with_data(elements);

        // Remove one
        let removed = bag.find_and_remove(|_| true);
        prop_assert_eq!(removed, Some(element));

        // count - 1 should remain
        prop_assert_eq!(bag.len(), count - 1,
            "Bag should have {} elements after removing one", count - 1);
    }

    /// Bag preserves all elements regardless of duplicates.
    #[test]
    fn prop_bag_preserves_all(
        elements in prop_vec(any::<i32>(), 1..100),
    ) {
        let bag = bag_with_data(elements.clone());
        prop_assert_eq!(bag.len(), elements.len());

        let all_data: Vec<_> = bag.all_data().into_iter().cloned().collect();
        prop_assert_eq!(all_data.len(), elements.len());
    }
}

// =============================================================================
// PriorityQueue Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// PriorityQueue returns highest priority first.
    ///
    /// **Rholang**: `rho:space:priorityqueue` for task scheduling
    #[test]
    fn prop_priority_queue_highest_first(
        low_priority in prop_vec(any::<i32>(), 1..10),
        high_priority in prop_vec(any::<i32>(), 1..10),
    ) {
        let mut pq = PriorityQueueDataCollection::<i32>::new(3);

        // Add low priority (2) first
        for item in &low_priority {
            pq.put_with_priority(*item, 2).expect("put low");
        }

        // Add high priority (0) second
        for item in &high_priority {
            pq.put_with_priority(*item, 0).expect("put high");
        }

        // First removal should be from high priority
        if !high_priority.is_empty() {
            let first = pq.find_and_remove(|_| true);
            prop_assert!(first.is_some());
            prop_assert!(high_priority.contains(&first.unwrap()),
                "First removed should be high priority");
        }
    }

    /// Within same priority, PriorityQueue is FIFO.
    #[test]
    fn prop_priority_queue_fifo_within_priority(
        elements in prop_vec(any::<i32>(), 2..20),
    ) {
        let mut pq = PriorityQueueDataCollection::<i32>::new(1);

        // All same priority
        for item in &elements {
            pq.put(*item).expect("put");
        }

        // Should remove in order
        let mut removed = Vec::new();
        while let Some(item) = pq.find_and_remove(|_| true) {
            removed.push(item);
        }

        prop_assert_eq!(removed, elements,
            "Same priority should be FIFO");
    }
}

// =============================================================================
// VectorDB Similarity Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// VectorDB matches most similar vector above threshold.
    ///
    /// **Rholang**: `rho:space:vectordb` for AI/ML integration
    #[test]
    fn prop_vectordb_similarity_match(
        x in -1.0f32..1.0,
        y in -1.0f32..1.0,
    ) {
        let mut vdb = VectorDBDataCollection::<String>::with_threshold(2, 0.9);

        // Insert a normalized vector
        let norm = (x * x + y * y).sqrt().max(0.001);
        let embedding = vec![x / norm, y / norm];
        vdb.put_with_embedding("test".to_string(), embedding.clone())
            .expect("put with embedding");

        // Query with same direction should match
        let result = vdb.find_most_similar(&embedding);
        prop_assert!(result.is_some(),
            "Should find match with identical embedding");

        let (_, similarity) = result.unwrap();
        prop_assert!(similarity >= 0.99,
            "Identical embeddings should have similarity ~1.0");
    }

    /// VectorDB rejects below-threshold matches.
    #[test]
    fn prop_vectordb_threshold_rejection(
        threshold in 0.5f32..0.99,
    ) {
        let mut vdb = VectorDBDataCollection::<String>::with_threshold(2, threshold);

        // Insert x-axis vector
        vdb.put_with_embedding("x".to_string(), vec![1.0, 0.0])
            .expect("put x");

        // Query with y-axis (orthogonal, similarity = 0)
        let result = vdb.find_most_similar(&[0.0, 1.0]);
        prop_assert!(result.is_none(),
            "Orthogonal vectors should not match (similarity = 0)");
    }

    /// VectorDB dimension mismatch is rejected.
    #[test]
    fn prop_vectordb_dimension_check(
        dims in 2usize..10,
    ) {
        let mut vdb = VectorDBDataCollection::<String>::new(dims);

        // Wrong dimension should fail
        let wrong_embedding = vec![0.0; dims + 1];
        let result = vdb.put_with_embedding("test".to_string(), wrong_embedding);
        prop_assert!(result.is_err(),
            "Wrong dimension should be rejected");
    }
}

// =============================================================================
// Cross-Collection Comparison Tests
// =============================================================================

#[cfg(test)]
mod comparison_tests {
    use super::*;

    #[test]
    fn test_collection_ordering_comparison() {
        // Same elements, different ordering

        // Queue: FIFO
        let mut queue = queue_with_data(vec![1, 2, 3]);
        assert_eq!(queue.find_and_remove(|_| true), Some(1)); // First in

        // Stack: LIFO
        let mut stack = stack_with_data(vec![1, 2, 3]);
        assert_eq!(stack.find_and_remove(|_| true), Some(3)); // Last in

        // Bag: Any (implementation-dependent, but typically insertion order or optimized)
        let mut bag = bag_with_data(vec![1, 2, 3]);
        let first = bag.find_and_remove(|_| true).unwrap();
        assert!([1, 2, 3].contains(&first)); // Any element
    }

    #[test]
    fn test_duplicate_handling_comparison() {
        // Set: Deduplicates
        let set = set_with_data(vec![1, 1, 1, 2, 2, 3]);
        assert_eq!(set.len(), 3); // Unique only

        // Bag: Keeps all
        let bag = bag_with_data(vec![1, 1, 1, 2, 2, 3]);
        assert_eq!(bag.len(), 6); // All duplicates
    }

    #[test]
    fn test_write_semantics_comparison() {
        // Cell: Exactly once
        let mut cell: CellDataCollection<i32> = CellDataCollection::new("test".to_string());
        assert!(cell.put(1).is_ok());
        assert!(cell.put(2).is_err()); // Fails

        // Bag: Unlimited
        let mut bag: BagDataCollection<i32> = BagDataCollection::new();
        for _ in 0..100 {
            assert!(bag.put(42).is_ok()); // Always succeeds
        }
    }
}
