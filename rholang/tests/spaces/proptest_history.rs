//! Property-Based Tests for History Store Module
//!
//! This module contains property-based tests for the history repository abstraction,
//! including InMemoryHistoryStore, BoundedHistoryStore, NullHistoryStore, and
//! VerifyingHistoryStore.
//!
//! # Rocq Correspondence
//!
//! These tests correspond to formal proofs in:
//! - `theories/Checkpoint.v` - Checkpoint state management
//! - `theories/OuterStorage.v` - Storage property guarantees
//!
//! # Properties Tested
//!
//! 1. **Roundtrip**: Store followed by retrieve returns original data
//! 2. **Capacity Bounds**: BoundedHistoryStore never exceeds capacity
//! 3. **Eviction Order**: Oldest entries are evicted first (FIFO)
//! 4. **Contains Consistency**: contains() returns true iff retrieve() succeeds
//! 5. **Clear Empties**: clear() makes checkpoint_count() == 0
//! 6. **NullStore No-Op**: NullHistoryStore stores nothing
//! 7. **Verification**: VerifyingHistoryStore rejects mismatched hashes
//!
//! # Rholang Usage
//!
//! ```rholang
//! // Checkpoint usage in Rholang
//! new space(`rho:space:bag:hashmap:default`), cp in {
//!   space.checkpoint!(*cp) |
//!   for (@root <- cp) {
//!     // Operations here...
//!     space.reset!(root)  // Rollback to checkpoint
//!   }
//! }
//! ```

use proptest::prelude::*;
use proptest::collection::vec as prop_vec;

use rholang::rust::interpreter::spaces::{
    HistoryStore, InMemoryHistoryStore, BoundedHistoryStore, NullHistoryStore,
    BoxedHistoryStore,
};
use rholang::rust::interpreter::spaces::history::VerifyingHistoryStore;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;

// =============================================================================
// Arbitrary Generators
// =============================================================================

fn arb_state_bytes() -> impl Strategy<Value = Vec<u8>> {
    prop_vec(any::<u8>(), 1..=64)
}

fn arb_root_and_state() -> impl Strategy<Value = (Blake2b256Hash, Vec<u8>)> {
    arb_state_bytes().prop_map(|state| {
        let root = Blake2b256Hash::new(&state);
        (root, state)
    })
}

fn arb_multiple_states(count: usize) -> impl Strategy<Value = Vec<(Blake2b256Hash, Vec<u8>)>> {
    prop_vec(arb_state_bytes(), count..=count).prop_map(|states| {
        // Ensure uniqueness by prepending index
        states.into_iter().enumerate().map(|(i, mut state)| {
            state.insert(0, (i & 0xFF) as u8);
            state.insert(0, ((i >> 8) & 0xFF) as u8);
            let root = Blake2b256Hash::new(&state);
            (root, state)
        }).collect()
    })
}

fn arb_capacity() -> impl Strategy<Value = usize> {
    1usize..=20
}

fn arb_unique_states(min: usize, max: usize) -> impl Strategy<Value = Vec<(Blake2b256Hash, Vec<u8>)>> {
    prop_vec(arb_state_bytes(), min..=max).prop_map(|states| {
        // Each state is made unique by its position
        states.into_iter().enumerate().map(|(i, mut state)| {
            // Ensure uniqueness by prepending index
            state.insert(0, (i & 0xFF) as u8);
            state.insert(0, ((i >> 8) & 0xFF) as u8);
            let root = Blake2b256Hash::new(&state);
            (root, state)
        }).collect()
    })
}

// =============================================================================
// InMemoryHistoryStore Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: Store followed by retrieve returns the exact same state.
    ///
    /// ∀ state. store(hash(state), state) → retrieve(hash(state)) == state
    ///
    /// Rocq: `store_retrieve_roundtrip` in Checkpoint.v
    #[test]
    fn prop_inmemory_store_retrieve_roundtrip((root, state) in arb_root_and_state()) {
        let store = InMemoryHistoryStore::new();

        store.store(root.clone(), &state).expect("store should succeed");
        let retrieved = store.retrieve(&root).expect("retrieve should succeed");

        prop_assert_eq!(retrieved, state, "Retrieved state should match original");
    }

    /// Property: contains() returns true iff store() was called with that root.
    ///
    /// Rocq: `contains_iff_stored` in Checkpoint.v
    #[test]
    fn prop_inmemory_contains_consistency((root, state) in arb_root_and_state()) {
        let store = InMemoryHistoryStore::new();

        // Before store, should not contain
        prop_assert!(!store.contains(&root), "Should not contain before store");

        store.store(root.clone(), &state).expect("store");

        // After store, should contain
        prop_assert!(store.contains(&root), "Should contain after store");
    }

    /// Property: Storing multiple states, all can be retrieved.
    ///
    /// Rocq: `multiple_stores_retrievable` in Checkpoint.v
    #[test]
    fn prop_inmemory_multiple_stores_retrievable(
        states in arb_unique_states(1, 10)
    ) {
        let store = InMemoryHistoryStore::new();

        // Store all states
        for (root, state) in &states {
            store.store(root.clone(), state).expect("store");
        }

        // All should be retrievable
        for (root, expected_state) in &states {
            let retrieved = store.retrieve(root).expect("retrieve");
            prop_assert_eq!(retrieved, expected_state.clone());
        }

        prop_assert_eq!(store.checkpoint_count(), states.len());
    }

    /// Property: clear() results in empty store.
    ///
    /// Rocq: `clear_empties` in Checkpoint.v
    #[test]
    fn prop_inmemory_clear_empties(states in arb_unique_states(1, 5)) {
        let store = InMemoryHistoryStore::new();

        for (root, state) in &states {
            store.store(root.clone(), state).expect("store");
        }

        store.clear().expect("clear");

        prop_assert_eq!(store.checkpoint_count(), 0);

        // All roots should now fail to retrieve
        for (root, _) in &states {
            prop_assert!(!store.contains(root));
            prop_assert!(store.retrieve(root).is_err());
        }
    }

    /// Property: list_roots() returns all stored roots.
    ///
    /// Rocq: `list_roots_complete` in Checkpoint.v
    #[test]
    fn prop_inmemory_list_roots_complete(states in arb_unique_states(1, 8)) {
        let store = InMemoryHistoryStore::new();

        for (root, state) in &states {
            store.store(root.clone(), state).expect("store");
        }

        let roots = store.list_roots();

        prop_assert_eq!(roots.len(), states.len());

        for (expected_root, _) in &states {
            prop_assert!(roots.contains(expected_root), "Missing root in list_roots()");
        }
    }

    /// Property: Overwriting a root updates the state without increasing count.
    ///
    /// Rocq: `overwrite_updates` in Checkpoint.v
    #[test]
    fn prop_inmemory_overwrite_updates(
        (root, state1) in arb_root_and_state(),
        state2 in arb_state_bytes()
    ) {
        let store = InMemoryHistoryStore::new();

        store.store(root.clone(), &state1).expect("store 1");
        prop_assert_eq!(store.checkpoint_count(), 1);

        // Overwrite with different state (same root)
        store.store(root.clone(), &state2).expect("store 2");
        prop_assert_eq!(store.checkpoint_count(), 1);

        let retrieved = store.retrieve(&root).expect("retrieve");
        prop_assert_eq!(retrieved, state2);
    }
}

// =============================================================================
// BoundedHistoryStore Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: BoundedHistoryStore never exceeds its capacity.
    ///
    /// ∀ capacity, n_inserts. checkpoint_count <= capacity
    ///
    /// Rocq: `bounded_capacity_invariant` in Checkpoint.v
    #[test]
    fn prop_bounded_never_exceeds_capacity(
        capacity in 1usize..=10,
        states in arb_unique_states(1, 20)
    ) {
        let store = BoundedHistoryStore::new(capacity);

        for (root, state) in &states {
            store.store(root.clone(), state).expect("store");
            prop_assert!(
                store.checkpoint_count() <= capacity,
                "Count {} exceeds capacity {}",
                store.checkpoint_count(),
                capacity
            );
        }
    }

    /// Property: When capacity is reached, oldest entries are evicted.
    ///
    /// Rocq: `bounded_fifo_eviction` in Checkpoint.v
    #[test]
    fn prop_bounded_fifo_eviction(
        capacity in 2usize..=5,
        states in arb_unique_states(6, 15)
    ) {
        let store = BoundedHistoryStore::new(capacity);

        // Store all states
        for (root, state) in &states {
            store.store(root.clone(), state).expect("store");
        }

        // Only the last `capacity` states should be present
        let expected_present: Vec<_> = states.iter().rev().take(capacity).collect();
        let expected_absent: Vec<_> = states.iter().take(states.len().saturating_sub(capacity)).collect();

        for (root, _) in expected_present {
            prop_assert!(store.contains(root), "Expected root to be present");
        }

        for (root, _) in expected_absent {
            prop_assert!(!store.contains(root), "Expected root to be evicted");
        }
    }

    /// Property: Updating an existing root moves it to the end (LRU-like).
    ///
    /// Rocq: `bounded_update_moves_to_end` in Checkpoint.v
    #[test]
    fn prop_bounded_update_moves_to_end(
        // Generate capacity and states together to avoid prop_assume rejections
        (capacity, states) in (3usize..=5).prop_flat_map(|cap| {
            arb_multiple_states(cap).prop_map(move |s| (cap, s))
        })
    ) {

        let store = BoundedHistoryStore::new(capacity);

        // Store all states in order
        for (root, state) in &states {
            store.store(root.clone(), state).expect("store");
        }

        // Re-store the first root (should move to end)
        let (first_root, first_state) = &states[0];
        store.store(first_root.clone(), first_state).expect("re-store");

        // list_roots should show the first root at the end now
        let roots = store.list_roots();
        prop_assert_eq!(roots.last(), Some(first_root), "Re-stored root should be at end");

        // If we add a new state, the second original state should be evicted
        if states.len() >= 2 {
            let new_state = vec![0xDE, 0xAD, 0xBE, 0xEF];
            let new_root = Blake2b256Hash::new(&new_state);
            store.store(new_root.clone(), &new_state).expect("store new");

            let (second_root, _) = &states[1];
            prop_assert!(!store.contains(second_root), "Second root should be evicted");
            prop_assert!(store.contains(first_root), "First root should survive (was refreshed)");
        }
    }

    /// Property: list_roots() returns roots in insertion order.
    ///
    /// Rocq: `bounded_list_roots_ordered` in Checkpoint.v
    #[test]
    fn prop_bounded_list_roots_ordered(
        capacity in 2usize..=10,
        states in arb_unique_states(1, 10)
    ) {
        prop_assume!(states.len() <= capacity);

        let store = BoundedHistoryStore::new(capacity);

        for (root, state) in &states {
            store.store(root.clone(), state).expect("store");
        }

        let roots = store.list_roots();
        let expected_roots: Vec<_> = states.iter().map(|(r, _)| r.clone()).collect();

        prop_assert_eq!(roots, expected_roots);
    }

    /// Property: BoundedHistoryStore capacity() returns the configured capacity.
    #[test]
    fn prop_bounded_capacity_getter(capacity in arb_capacity()) {
        let store = BoundedHistoryStore::new(capacity);
        prop_assert_eq!(store.capacity(), capacity);
    }
}

// =============================================================================
// NullHistoryStore Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: NullHistoryStore.store() always succeeds but stores nothing.
    ///
    /// Rocq: `null_store_noop` in Checkpoint.v
    #[test]
    fn prop_null_store_noop((root, state) in arb_root_and_state()) {
        let store = NullHistoryStore::new();

        // Store succeeds
        let result = store.store(root.clone(), &state);
        prop_assert!(result.is_ok());

        // But nothing is stored
        prop_assert!(!store.contains(&root));
        prop_assert_eq!(store.checkpoint_count(), 0);
    }

    /// Property: NullHistoryStore.retrieve() always fails.
    ///
    /// Rocq: `null_retrieve_fails` in Checkpoint.v
    #[test]
    fn prop_null_retrieve_fails((root, state) in arb_root_and_state()) {
        let store = NullHistoryStore::new();

        store.store(root.clone(), &state).expect("store");

        let result = store.retrieve(&root);
        prop_assert!(result.is_err());
    }

    /// Property: NullHistoryStore.list_roots() is always empty.
    #[test]
    fn prop_null_list_roots_empty(states in arb_unique_states(0, 5)) {
        let store = NullHistoryStore::new();

        for (root, state) in &states {
            store.store(root.clone(), state).expect("store");
        }

        prop_assert!(store.list_roots().is_empty());
    }

    /// Property: NullHistoryStore.clear() always succeeds.
    #[test]
    fn prop_null_clear_succeeds(states in arb_unique_states(0, 3)) {
        let store = NullHistoryStore::new();

        for (root, state) in &states {
            store.store(root.clone(), state).expect("store");
        }

        prop_assert!(store.clear().is_ok());
    }
}

// =============================================================================
// VerifyingHistoryStore Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: VerifyingHistoryStore accepts correctly hashed state.
    ///
    /// Rocq: `verifying_accepts_valid` in Checkpoint.v
    #[test]
    fn prop_verifying_accepts_valid((root, state) in arb_root_and_state()) {
        let inner = InMemoryHistoryStore::new();
        let store = VerifyingHistoryStore::new(inner);

        let result = store.store(root.clone(), &state);
        prop_assert!(result.is_ok(), "Valid store should succeed");

        let retrieved = store.retrieve(&root);
        prop_assert!(retrieved.is_ok(), "Retrieve should succeed");
        prop_assert_eq!(retrieved.unwrap(), state);
    }

    /// Property: VerifyingHistoryStore rejects mismatched root.
    ///
    /// Rocq: `verifying_rejects_invalid` in Checkpoint.v
    #[test]
    fn prop_verifying_rejects_invalid(state in arb_state_bytes()) {
        let inner = InMemoryHistoryStore::new();
        let store = VerifyingHistoryStore::new(inner);

        // Create a wrong root (hash of different data)
        let wrong_data = vec![0xFF; 32];
        let wrong_root = Blake2b256Hash::new(&wrong_data);

        // This should fail if state != wrong_data
        if state != wrong_data {
            let result = store.store(wrong_root, &state);
            prop_assert!(result.is_err(), "Mismatched root should fail");
        }
    }

    /// Property: VerifyingHistoryStore delegates to inner store.
    #[test]
    fn prop_verifying_delegates((root, state) in arb_root_and_state()) {
        let inner = InMemoryHistoryStore::new();
        let store = VerifyingHistoryStore::new(inner);

        store.store(root.clone(), &state).expect("store");

        // Check that inner store has the data
        prop_assert!(store.inner().contains(&root));
        prop_assert_eq!(store.checkpoint_count(), 1);
    }
}

// =============================================================================
// BoxedHistoryStore Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Property: All BoxedHistoryStore implementations share the same interface.
    ///
    /// This verifies object safety and interface compatibility.
    #[test]
    fn prop_boxed_stores_compatible((root, state) in arb_root_and_state()) {
        use rholang::rust::interpreter::spaces::history::{boxed_in_memory, boxed_bounded, boxed_null};

        let stores: Vec<BoxedHistoryStore> = vec![
            boxed_in_memory(),
            boxed_bounded(10),
            boxed_null(),
        ];

        for store in stores {
            // All stores should accept the same operations
            let store_result = store.store(root.clone(), &state);
            prop_assert!(store_result.is_ok());

            let _ = store.contains(&root);
            let _ = store.checkpoint_count();
            let _ = store.list_roots();
        }
    }
}

// =============================================================================
// Determinism Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Property: Hash function is deterministic.
    ///
    /// ∀ state. hash(state) == hash(state)
    #[test]
    fn prop_hash_deterministic(state in arb_state_bytes()) {
        let hash1 = Blake2b256Hash::new(&state);
        let hash2 = Blake2b256Hash::new(&state);

        prop_assert_eq!(hash1, hash2);
    }

    /// Property: Different states (almost always) produce different hashes.
    ///
    /// This is a collision resistance property.
    #[test]
    fn prop_hash_collision_resistant(state1 in arb_state_bytes(), state2 in arb_state_bytes()) {
        let hash1 = Blake2b256Hash::new(&state1);
        let hash2 = Blake2b256Hash::new(&state2);

        // Different inputs should produce different outputs (except in very rare collision)
        if state1 != state2 {
            prop_assert_ne!(hash1, hash2, "Different states should have different hashes");
        }
    }
}

// =============================================================================
// Edge Case Tests (Unit Tests)
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_state() {
        let store = InMemoryHistoryStore::new();
        let state = vec![];
        let root = Blake2b256Hash::new(&state);

        store.store(root.clone(), &state).expect("store empty");
        let retrieved = store.retrieve(&root).expect("retrieve empty");

        assert_eq!(retrieved, state);
    }

    #[test]
    fn test_large_state() {
        let store = InMemoryHistoryStore::new();
        let state = vec![42u8; 10_000];
        let root = Blake2b256Hash::new(&state);

        store.store(root.clone(), &state).expect("store large");
        let retrieved = store.retrieve(&root).expect("retrieve large");

        assert_eq!(retrieved, state);
    }

    #[test]
    fn test_bounded_capacity_one() {
        let store = BoundedHistoryStore::new(1);

        let state1 = vec![1, 2, 3];
        let root1 = Blake2b256Hash::new(&state1);
        store.store(root1.clone(), &state1).expect("store 1");

        let state2 = vec![4, 5, 6];
        let root2 = Blake2b256Hash::new(&state2);
        store.store(root2.clone(), &state2).expect("store 2");

        assert!(!store.contains(&root1));
        assert!(store.contains(&root2));
        assert_eq!(store.checkpoint_count(), 1);
    }

    #[test]
    fn test_verifying_store_inner_access() {
        let inner = InMemoryHistoryStore::new();
        let store = VerifyingHistoryStore::new(inner);

        let state = vec![1, 2, 3];
        let root = Blake2b256Hash::new(&state);

        store.store(root.clone(), &state).expect("store");

        // Access inner store directly
        let inner_retrieved = store.inner().retrieve(&root).expect("inner retrieve");
        assert_eq!(inner_retrieved, state);
    }

    #[test]
    fn test_null_store_is_clone() {
        let store1 = NullHistoryStore::new();
        let store2 = store1.clone();

        // Both should behave identically
        let state = vec![1, 2, 3];
        let root = Blake2b256Hash::new(&state);

        assert!(store1.store(root.clone(), &state).is_ok());
        assert!(!store2.contains(&root));
    }

    #[test]
    fn test_default_implementations() {
        let in_memory: InMemoryHistoryStore = Default::default();
        assert_eq!(in_memory.checkpoint_count(), 0);

        let null: NullHistoryStore = Default::default();
        assert_eq!(null.checkpoint_count(), 0);
    }
}
