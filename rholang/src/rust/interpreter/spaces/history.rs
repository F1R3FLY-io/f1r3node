//! History Repository Abstraction for Reified RSpaces
//!
//! This module provides an abstract interface for storing and retrieving space state
//! at Merkle roots. It enables checkpointing and replay functionality for spaces.
//!
//! # Design
//!
//! The `HistoryStore` trait provides a simple key-value interface where:
//! - Keys are Blake2b256Hash Merkle roots
//! - Values are serialized space state
//!
//! This abstraction enables:
//! - Multiple backend implementations (in-memory, LMDB, etc.)
//! - Testable checkpointing without heavy dependencies
//! - Clean separation between space logic and persistence
//!
//! # Implementations
//!
//! - `InMemoryHistoryStore`: For testing and ephemeral spaces
//! - `RSpaceHistoryStoreAdapter`: Wraps rspace++'s HistoryRepository for production

use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;

use super::errors::SpaceError;

// =============================================================================
// Core Trait: HistoryStore
// =============================================================================

/// Abstract repository for checkpoint/replay history.
///
/// This trait provides a simple interface for storing and retrieving serialized
/// space state indexed by Merkle roots (Blake2b256Hash).
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to support concurrent access from
/// multiple space instances.
///
/// # Usage
///
/// ```ignore
/// let store = InMemoryHistoryStore::new();
///
/// // Serialize and store space state
/// let state = serialize_space(&space);
/// let root = Blake2b256Hash::new(&state);
/// store.store(root.clone(), &state)?;
///
/// // Later, retrieve and restore
/// let restored_state = store.retrieve(&root)?;
/// deserialize_space(&restored_state);
/// ```
pub trait HistoryStore: Send + Sync + Debug {
    /// Store serialized state at a Merkle root.
    ///
    /// # Arguments
    /// - `root`: The Merkle root (hash of the state)
    /// - `state`: The serialized state bytes
    ///
    /// # Errors
    /// Returns `SpaceError::CheckpointError` if storage fails.
    fn store(&self, root: Blake2b256Hash, state: &[u8]) -> Result<(), SpaceError>;

    /// Retrieve serialized state for a Merkle root.
    ///
    /// # Arguments
    /// - `root`: The Merkle root to look up
    ///
    /// # Errors
    /// Returns `SpaceError::CheckpointError` if the root is not found.
    fn retrieve(&self, root: &Blake2b256Hash) -> Result<Vec<u8>, SpaceError>;

    /// Check if a Merkle root exists in the store.
    ///
    /// # Arguments
    /// - `root`: The Merkle root to check
    ///
    /// # Returns
    /// `true` if the root exists, `false` otherwise.
    fn contains(&self, root: &Blake2b256Hash) -> bool;

    /// Clear all stored history.
    ///
    /// This removes all stored state, but does not affect the current space state.
    /// Use with caution as this is irreversible.
    ///
    /// # Errors
    /// Returns `SpaceError::CheckpointError` if clearing fails.
    fn clear(&self) -> Result<(), SpaceError>;

    /// Get the number of stored checkpoints.
    fn checkpoint_count(&self) -> usize;

    /// Get all stored roots (for debugging/enumeration).
    fn list_roots(&self) -> Vec<Blake2b256Hash>;
}

// =============================================================================
// In-Memory Implementation
// =============================================================================

/// In-memory history store for testing and ephemeral spaces.
///
/// This implementation stores state in a thread-safe HashMap. It does not
/// persist across process restarts.
///
/// # Thread Safety
///
/// Uses `RwLock` for interior mutability, allowing concurrent reads.
///
/// # Example
///
/// ```ignore
/// let store = InMemoryHistoryStore::new();
///
/// let state = vec![1, 2, 3, 4];
/// let root = Blake2b256Hash::new(&state);
///
/// store.store(root.clone(), &state)?;
/// assert!(store.contains(&root));
///
/// let retrieved = store.retrieve(&root)?;
/// assert_eq!(state, retrieved);
/// ```
#[derive(Debug)]
pub struct InMemoryHistoryStore {
    /// Map from Merkle root to serialized state
    states: RwLock<HashMap<Blake2b256Hash, Vec<u8>>>,
}

impl InMemoryHistoryStore {
    /// Create a new empty in-memory history store.
    pub fn new() -> Self {
        InMemoryHistoryStore {
            states: RwLock::new(HashMap::new()),
        }
    }

    /// Create a history store with pre-populated state.
    ///
    /// Useful for testing scenarios where initial history is needed.
    pub fn with_initial_state(states: HashMap<Blake2b256Hash, Vec<u8>>) -> Self {
        InMemoryHistoryStore {
            states: RwLock::new(states),
        }
    }
}

impl Default for InMemoryHistoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl HistoryStore for InMemoryHistoryStore {
    fn store(&self, root: Blake2b256Hash, state: &[u8]) -> Result<(), SpaceError> {
        let mut states = self.states.write().map_err(|e| SpaceError::CheckpointError {
            description: format!("Failed to acquire write lock: {}", e),
        })?;

        states.insert(root, state.to_vec());
        Ok(())
    }

    fn retrieve(&self, root: &Blake2b256Hash) -> Result<Vec<u8>, SpaceError> {
        let states = self.states.read().map_err(|e| SpaceError::CheckpointError {
            description: format!("Failed to acquire read lock: {}", e),
        })?;

        states.get(root).cloned().ok_or_else(|| SpaceError::CheckpointError {
            description: format!("Merkle root not found: {}", root),
        })
    }

    fn contains(&self, root: &Blake2b256Hash) -> bool {
        self.states
            .read()
            .map(|states| states.contains_key(root))
            .unwrap_or(false)
    }

    fn clear(&self) -> Result<(), SpaceError> {
        let mut states = self.states.write().map_err(|e| SpaceError::CheckpointError {
            description: format!("Failed to acquire write lock: {}", e),
        })?;

        states.clear();
        Ok(())
    }

    fn checkpoint_count(&self) -> usize {
        self.states.read().map(|s| s.len()).unwrap_or(0)
    }

    fn list_roots(&self) -> Vec<Blake2b256Hash> {
        self.states
            .read()
            .map(|s| s.keys().cloned().collect())
            .unwrap_or_default()
    }
}

// =============================================================================
// Bounded History Store (with capacity limit)
// =============================================================================

/// History store with a maximum capacity and O(1) eviction.
///
/// When the capacity is reached, the oldest entries are evicted (LRU-like behavior).
/// Useful for limiting memory usage in long-running processes.
///
/// # Performance
/// Uses `VecDeque` for O(1) eviction at the front, improving over the previous
/// O(n) `Vec::remove(0)` approach.
#[derive(Debug)]
pub struct BoundedHistoryStore {
    /// Map from Merkle root to serialized state
    states: RwLock<HashMap<Blake2b256Hash, Vec<u8>>>,
    /// Insertion order for LRU eviction (VecDeque for O(1) pop_front)
    insertion_order: RwLock<VecDeque<Blake2b256Hash>>,
    /// Maximum number of entries to keep
    max_capacity: usize,
}

impl BoundedHistoryStore {
    /// Create a new bounded history store with the given capacity.
    ///
    /// # Arguments
    /// - `max_capacity`: Maximum number of checkpoints to retain
    pub fn new(max_capacity: usize) -> Self {
        BoundedHistoryStore {
            states: RwLock::new(HashMap::with_capacity(max_capacity)),
            insertion_order: RwLock::new(VecDeque::with_capacity(max_capacity)),
            max_capacity,
        }
    }

    /// Get the maximum capacity of this store.
    pub fn capacity(&self) -> usize {
        self.max_capacity
    }
}

impl HistoryStore for BoundedHistoryStore {
    fn store(&self, root: Blake2b256Hash, state: &[u8]) -> Result<(), SpaceError> {
        let mut states = self.states.write().map_err(|e| SpaceError::CheckpointError {
            description: format!("Failed to acquire write lock: {}", e),
        })?;

        let mut order = self.insertion_order.write().map_err(|e| {
            SpaceError::CheckpointError {
                description: format!("Failed to acquire insertion order lock: {}", e),
            }
        })?;

        // If key already exists, update it and move to end of order (no eviction needed)
        let is_update = states.contains_key(&root);
        if is_update {
            order.retain(|r| r != &root);
        } else {
            // New key - evict oldest if at capacity (O(1) with VecDeque::pop_front)
            while states.len() >= self.max_capacity {
                if let Some(oldest_root) = order.pop_front() {
                    states.remove(&oldest_root);
                } else {
                    break;
                }
            }
        }

        states.insert(root.clone(), state.to_vec());
        order.push_back(root);

        Ok(())
    }

    fn retrieve(&self, root: &Blake2b256Hash) -> Result<Vec<u8>, SpaceError> {
        let states = self.states.read().map_err(|e| SpaceError::CheckpointError {
            description: format!("Failed to acquire read lock: {}", e),
        })?;

        states.get(root).cloned().ok_or_else(|| SpaceError::CheckpointError {
            description: format!("Merkle root not found: {}", root),
        })
    }

    fn contains(&self, root: &Blake2b256Hash) -> bool {
        self.states
            .read()
            .map(|states| states.contains_key(root))
            .unwrap_or(false)
    }

    fn clear(&self) -> Result<(), SpaceError> {
        let mut states = self.states.write().map_err(|e| SpaceError::CheckpointError {
            description: format!("Failed to acquire write lock: {}", e),
        })?;

        let mut order = self.insertion_order.write().map_err(|e| {
            SpaceError::CheckpointError {
                description: format!("Failed to acquire insertion order lock: {}", e),
            }
        })?;

        states.clear();
        order.clear();
        Ok(())
    }

    fn checkpoint_count(&self) -> usize {
        self.states.read().map(|s| s.len()).unwrap_or(0)
    }

    fn list_roots(&self) -> Vec<Blake2b256Hash> {
        // Return in insertion order (oldest first)
        // Convert VecDeque to Vec for API compatibility
        self.insertion_order
            .read()
            .map(|o| o.iter().cloned().collect())
            .unwrap_or_default()
    }
}

// =============================================================================
// Null History Store (no-op for temp spaces)
// =============================================================================

/// No-op history store for temporary spaces that don't need checkpointing.
///
/// All operations succeed but nothing is actually stored. Useful for
/// `SpaceQualifier::Temp` spaces where persistence is not needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullHistoryStore;

impl NullHistoryStore {
    /// Create a new null history store.
    pub fn new() -> Self {
        NullHistoryStore
    }
}

impl HistoryStore for NullHistoryStore {
    fn store(&self, _root: Blake2b256Hash, _state: &[u8]) -> Result<(), SpaceError> {
        // No-op: temp spaces don't persist checkpoints
        Ok(())
    }

    fn retrieve(&self, root: &Blake2b256Hash) -> Result<Vec<u8>, SpaceError> {
        // Always fails: nothing is stored
        Err(SpaceError::CheckpointError {
            description: format!(
                "NullHistoryStore does not store checkpoints. Root {} not found.",
                root
            ),
        })
    }

    fn contains(&self, _root: &Blake2b256Hash) -> bool {
        // Nothing is ever stored
        false
    }

    fn clear(&self) -> Result<(), SpaceError> {
        // No-op: nothing to clear
        Ok(())
    }

    fn checkpoint_count(&self) -> usize {
        0
    }

    fn list_roots(&self) -> Vec<Blake2b256Hash> {
        Vec::new()
    }
}

// =============================================================================
// Type-Erased Boxed History Store
// =============================================================================

/// Type-erased history store for dynamic dispatch.
///
/// This allows storing different `HistoryStore` implementations in the same
/// container, useful for the `GenericRSpace` struct.
pub type BoxedHistoryStore = Arc<dyn HistoryStore>;

/// Create a boxed in-memory history store.
pub fn boxed_in_memory() -> BoxedHistoryStore {
    Arc::new(InMemoryHistoryStore::new())
}

/// Create a boxed bounded history store.
pub fn boxed_bounded(max_capacity: usize) -> BoxedHistoryStore {
    Arc::new(BoundedHistoryStore::new(max_capacity))
}

/// Create a boxed null history store.
pub fn boxed_null() -> BoxedHistoryStore {
    Arc::new(NullHistoryStore::new())
}

// =============================================================================
// History Store with Verification
// =============================================================================

/// History store wrapper that verifies state integrity on retrieval.
///
/// This wrapper computes the hash of retrieved state and verifies it matches
/// the requested root. Useful for detecting corruption.
#[derive(Debug)]
pub struct VerifyingHistoryStore<H: HistoryStore> {
    inner: H,
}

impl<H: HistoryStore> VerifyingHistoryStore<H> {
    /// Create a new verifying history store wrapping the given store.
    pub fn new(inner: H) -> Self {
        VerifyingHistoryStore { inner }
    }

    /// Get a reference to the inner store.
    pub fn inner(&self) -> &H {
        &self.inner
    }
}

impl<H: HistoryStore> HistoryStore for VerifyingHistoryStore<H> {
    fn store(&self, root: Blake2b256Hash, state: &[u8]) -> Result<(), SpaceError> {
        // Verify the root matches the state hash
        let computed_root = Blake2b256Hash::new(state);
        if computed_root != root {
            return Err(SpaceError::CheckpointError {
                description: format!(
                    "Root mismatch on store: expected {}, computed {}",
                    root, computed_root
                ),
            });
        }

        self.inner.store(root, state)
    }

    fn retrieve(&self, root: &Blake2b256Hash) -> Result<Vec<u8>, SpaceError> {
        let state = self.inner.retrieve(root)?;

        // Verify the retrieved state matches the expected root
        let computed_root = Blake2b256Hash::new(&state);
        if &computed_root != root {
            return Err(SpaceError::CheckpointError {
                description: format!(
                    "Corrupted state: expected root {}, computed {}",
                    root, computed_root
                ),
            });
        }

        Ok(state)
    }

    fn contains(&self, root: &Blake2b256Hash) -> bool {
        self.inner.contains(root)
    }

    fn clear(&self) -> Result<(), SpaceError> {
        self.inner.clear()
    }

    fn checkpoint_count(&self) -> usize {
        self.inner.checkpoint_count()
    }

    fn list_roots(&self) -> Vec<Blake2b256Hash> {
        self.inner.list_roots()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_state(id: u8) -> (Blake2b256Hash, Vec<u8>) {
        let state = vec![id; 32];
        let root = Blake2b256Hash::new(&state);
        (root, state)
    }

    // =========================================================================
    // InMemoryHistoryStore tests
    // =========================================================================

    #[test]
    fn test_in_memory_store_roundtrip() {
        let store = InMemoryHistoryStore::new();
        let (root, state) = make_test_state(42);

        store.store(root.clone(), &state).expect("store should succeed");
        assert!(store.contains(&root));

        let retrieved = store.retrieve(&root).expect("retrieve should succeed");
        assert_eq!(state, retrieved);
    }

    #[test]
    fn test_in_memory_store_not_found() {
        let store = InMemoryHistoryStore::new();
        let (root, _) = make_test_state(99);

        let result = store.retrieve(&root);
        assert!(result.is_err());
        assert!(!store.contains(&root));
    }

    #[test]
    fn test_in_memory_store_clear() {
        let store = InMemoryHistoryStore::new();
        let (root1, state1) = make_test_state(1);
        let (root2, state2) = make_test_state(2);

        store.store(root1.clone(), &state1).expect("store 1");
        store.store(root2.clone(), &state2).expect("store 2");
        assert_eq!(store.checkpoint_count(), 2);

        store.clear().expect("clear should succeed");
        assert_eq!(store.checkpoint_count(), 0);
        assert!(!store.contains(&root1));
        assert!(!store.contains(&root2));
    }

    #[test]
    fn test_in_memory_store_overwrite() {
        let store = InMemoryHistoryStore::new();
        let (root, state1) = make_test_state(1);

        store.store(root.clone(), &state1).expect("store 1");

        // Overwrite with different data (same root for testing)
        let state2 = vec![2; 32];
        store.store(root.clone(), &state2).expect("store 2");

        let retrieved = store.retrieve(&root).expect("retrieve");
        assert_eq!(state2, retrieved);
        assert_eq!(store.checkpoint_count(), 1);
    }

    #[test]
    fn test_in_memory_store_list_roots() {
        let store = InMemoryHistoryStore::new();
        let (root1, state1) = make_test_state(1);
        let (root2, state2) = make_test_state(2);

        store.store(root1.clone(), &state1).expect("store 1");
        store.store(root2.clone(), &state2).expect("store 2");

        let roots = store.list_roots();
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&root1));
        assert!(roots.contains(&root2));
    }

    // =========================================================================
    // BoundedHistoryStore tests
    // =========================================================================

    #[test]
    fn test_bounded_store_eviction() {
        let store = BoundedHistoryStore::new(2);

        let (root1, state1) = make_test_state(1);
        let (root2, state2) = make_test_state(2);
        let (root3, state3) = make_test_state(3);

        store.store(root1.clone(), &state1).expect("store 1");
        store.store(root2.clone(), &state2).expect("store 2");
        assert_eq!(store.checkpoint_count(), 2);

        // This should evict root1
        store.store(root3.clone(), &state3).expect("store 3");
        assert_eq!(store.checkpoint_count(), 2);

        // root1 should be gone
        assert!(!store.contains(&root1));
        assert!(store.contains(&root2));
        assert!(store.contains(&root3));
    }

    #[test]
    fn test_bounded_store_insertion_order() {
        let store = BoundedHistoryStore::new(3);

        let (root1, state1) = make_test_state(1);
        let (root2, state2) = make_test_state(2);
        let (root3, state3) = make_test_state(3);

        store.store(root1.clone(), &state1).expect("store 1");
        store.store(root2.clone(), &state2).expect("store 2");
        store.store(root3.clone(), &state3).expect("store 3");

        let roots = store.list_roots();
        assert_eq!(roots, vec![root1, root2, root3]);
    }

    #[test]
    fn test_bounded_store_update_moves_to_end() {
        let store = BoundedHistoryStore::new(3);

        let (root1, state1) = make_test_state(1);
        let (root2, state2) = make_test_state(2);
        let (root3, state3) = make_test_state(3);

        store.store(root1.clone(), &state1).expect("store 1");
        store.store(root2.clone(), &state2).expect("store 2");
        store.store(root3.clone(), &state3).expect("store 3");

        // Re-store root1, should move to end
        store.store(root1.clone(), &state1).expect("re-store 1");

        let roots = store.list_roots();
        assert_eq!(roots, vec![root2.clone(), root3.clone(), root1.clone()]);

        // Now add root4, should evict root2 (oldest)
        let (root4, state4) = make_test_state(4);
        store.store(root4.clone(), &state4).expect("store 4");

        assert!(!store.contains(&root2));
        assert!(store.contains(&root3));
        assert!(store.contains(&root1));
        assert!(store.contains(&root4));
    }

    // =========================================================================
    // NullHistoryStore tests
    // =========================================================================

    #[test]
    fn test_null_store_noop() {
        let store = NullHistoryStore::new();
        let (root, state) = make_test_state(42);

        // Store succeeds
        store.store(root.clone(), &state).expect("store should succeed");

        // But nothing is stored
        assert!(!store.contains(&root));
        assert_eq!(store.checkpoint_count(), 0);

        // Retrieve fails
        let result = store.retrieve(&root);
        assert!(result.is_err());
    }

    // =========================================================================
    // VerifyingHistoryStore tests
    // =========================================================================

    #[test]
    fn test_verifying_store_valid() {
        let inner = InMemoryHistoryStore::new();
        let store = VerifyingHistoryStore::new(inner);

        let state = vec![1, 2, 3, 4, 5];
        let root = Blake2b256Hash::new(&state);

        store.store(root.clone(), &state).expect("store should succeed");

        let retrieved = store.retrieve(&root).expect("retrieve should succeed");
        assert_eq!(state, retrieved);
    }

    #[test]
    fn test_verifying_store_rejects_mismatched_root() {
        let inner = InMemoryHistoryStore::new();
        let store = VerifyingHistoryStore::new(inner);

        let state = vec![1, 2, 3, 4, 5];
        let wrong_root = Blake2b256Hash::new(&[9, 9, 9]); // Different hash

        let result = store.store(wrong_root, &state);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Root mismatch"));
    }

    // =========================================================================
    // Thread safety tests
    // =========================================================================

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let store = Arc::new(InMemoryHistoryStore::new());
        let mut handles = vec![];

        // Spawn multiple writers
        for i in 0..10u8 {
            let store_clone = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let (root, state) = make_test_state(i);
                store_clone.store(root, &state).expect("store failed");
            }));
        }

        // Wait for all writers
        for handle in handles {
            handle.join().expect("thread panicked");
        }

        assert_eq!(store.checkpoint_count(), 10);

        // Concurrent readers
        let mut read_handles = vec![];
        let roots = store.list_roots();

        for root in roots {
            let store_clone = Arc::clone(&store);
            read_handles.push(thread::spawn(move || {
                let result = store_clone.retrieve(&root);
                assert!(result.is_ok());
            }));
        }

        for handle in read_handles {
            handle.join().expect("read thread panicked");
        }
    }

    // =========================================================================
    // Boxed store tests
    // =========================================================================

    #[test]
    fn test_boxed_stores_are_compatible() {
        let stores: Vec<BoxedHistoryStore> = vec![
            boxed_in_memory(),
            boxed_bounded(10),
            boxed_null(),
        ];

        let (root, state) = make_test_state(42);

        for store in stores {
            // All stores should accept the same interface
            let _ = store.store(root.clone(), &state);
            let _ = store.contains(&root);
            let _ = store.checkpoint_count();
        }
    }
}
