//! Tombstone management for vector backend.
//!
//! This module provides centralized management of tombstone state,
//! including live masks, free lists, and allocation/deallocation.
//!
//! # Design
//!
//! Tombstoning marks entries as "dead" without physical deletion, enabling:
//! - O(1) removal (no matrix reallocation)
//! - Slot reuse via free list
//! - Vectorization-friendly filtering via mask multiplication
//!
//! # Vectorization
//!
//! The live_mask uses f32 (1.0 = live, 0.0 = dead) for element-wise multiplication
//! with score arrays using `azip!`, which provides better auto-vectorization hints
//! to LLVM than naive operator overloading.

use std::sync::RwLock;

use ndarray::{azip, s, Array1};

/// Manages tombstone state for efficient vector removal.
///
/// # Example
///
/// ```ignore
/// let mut tombstones = TombstoneManager::new();
///
/// // Allocate slots
/// tombstones.record_append(1);  // After matrix grows
/// assert!(tombstones.is_live(0));
///
/// // Tombstone an entry
/// tombstones.tombstone(0);
/// assert!(!tombstones.is_live(0));
///
/// // Reuse the slot
/// let reused = tombstones.allocate();
/// assert_eq!(reused, Some(0));
/// assert!(tombstones.is_live(0));
/// ```
#[derive(Debug)]
pub struct TombstoneManager {
    /// Live mask: 1.0 for live entries, 0.0 for tombstoned.
    /// Length matches the embedding matrix row count.
    live_mask: Array1<f32>,
    /// Free list of tombstoned indices available for reuse.
    free_list: Vec<usize>,
    /// Count of live (non-tombstoned) entries.
    live_count: usize,
    /// Cached live indices, lazily computed and invalidated on mutation.
    /// Uses RwLock for thread-safe interior mutability, allowing `live_indices()` to take `&self`
    /// while still caching the result.
    live_indices_cache: RwLock<Option<Vec<usize>>>,
}

impl Clone for TombstoneManager {
    fn clone(&self) -> Self {
        Self {
            live_mask: self.live_mask.clone(),
            free_list: self.free_list.clone(),
            live_count: self.live_count,
            // Clone the contents of the RwLock, not the RwLock itself
            live_indices_cache: RwLock::new(
                self.live_indices_cache
                    .read()
                    .expect("live_indices_cache read lock poisoned")
                    .clone(),
            ),
        }
    }
}

impl Default for TombstoneManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TombstoneManager {
    // ========================================================================
    // Construction
    // ========================================================================

    /// Create a new tombstone manager with zero capacity.
    pub fn new() -> Self {
        Self {
            live_mask: Array1::zeros(0),
            free_list: Vec::new(),
            live_count: 0,
            live_indices_cache: RwLock::new(None),
        }
    }

    /// Create a tombstone manager with pre-allocated capacity.
    ///
    /// All slots start as tombstoned (available for allocation via `allocate()`).
    /// This is useful when pre-allocating the embedding matrix.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            live_mask: Array1::zeros(capacity),
            free_list: (0..capacity).collect(),
            live_count: 0,
            live_indices_cache: RwLock::new(None),
        }
    }

    // ========================================================================
    // Query Methods
    // ========================================================================

    /// Check if an index is live (not tombstoned).
    #[inline]
    pub fn is_live(&self, index: usize) -> bool {
        index < self.live_mask.len() && self.live_mask[index] > 0.0
    }

    /// Get the number of live entries.
    #[inline]
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Get the total capacity (live + tombstoned slots).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.live_mask.len()
    }

    /// Check if there are any live entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Get the live mask for SIMD operations.
    ///
    /// Returns a reference to avoid copying for large masks.
    #[inline]
    pub fn live_mask(&self) -> &Array1<f32> {
        &self.live_mask
    }

    /// Get indices of all live entries.
    ///
    /// Lazily computes and caches the result. The cache is invalidated
    /// when entries are allocated, tombstoned, or cleared.
    ///
    /// This is useful for pre-filtering embeddings before similarity
    /// computation to avoid unnecessary work on tombstoned entries.
    ///
    /// Returns a clone of the cached indices to avoid borrowing issues with RwLock.
    pub fn live_indices(&self) -> Vec<usize> {
        // Check if cache needs population (read lock)
        let needs_compute = self
            .live_indices_cache
            .read()
            .expect("live_indices_cache read lock poisoned")
            .is_none();
        if needs_compute {
            let indices: Vec<usize> = self
                .live_mask
                .iter()
                .enumerate()
                .filter(|(_, &m)| m > 0.0)
                .map(|(i, _)| i)
                .collect();
            *self
                .live_indices_cache
                .write()
                .expect("live_indices_cache write lock poisoned") = Some(indices);
        }
        self.live_indices_cache
            .read()
            .expect("live_indices_cache read lock poisoned")
            .as_ref()
            .expect("cache just populated")
            .clone()
    }

    // ========================================================================
    // Mutation Methods
    // ========================================================================

    /// Invalidate the live indices cache.
    /// Must be called after any mutation that changes live/dead state.
    #[inline]
    fn invalidate_cache(&self) {
        *self
            .live_indices_cache
            .write()
            .expect("live_indices_cache write lock poisoned") = None;
    }

    /// Allocate a slot by reusing a tombstoned index.
    ///
    /// Returns `Some(index)` if a free slot was reused, `None` if no free slots
    /// are available (caller should append to the matrix and call `record_append`).
    ///
    /// The reused slot is automatically marked as live.
    pub fn allocate(&mut self) -> Option<usize> {
        if let Some(index) = self.free_list.pop() {
            self.live_mask[index] = 1.0;
            self.live_count += 1;
            self.invalidate_cache();
            Some(index)
        } else {
            None
        }
    }

    /// Record a new allocation at the end (after matrix growth).
    ///
    /// Call this after appending a row to the embeddings matrix.
    /// Grows the live mask and marks the new slot as live.
    pub fn record_append(&mut self, new_capacity: usize) {
        let new_index = self.live_mask.len();
        self.grow_to(new_capacity);
        self.live_mask[new_index] = 1.0;
        self.live_count += 1;
        self.invalidate_cache();
    }

    /// Tombstone an entry at the given index.
    ///
    /// Returns `true` if the entry was live and is now tombstoned,
    /// `false` if already tombstoned or out of bounds.
    pub fn tombstone(&mut self, index: usize) -> bool {
        if !self.is_live(index) {
            return false;
        }

        // Mark as dead in mask
        self.live_mask[index] = 0.0;

        // Add to free list for reuse
        self.free_list.push(index);

        // Decrement live count
        self.live_count = self.live_count.saturating_sub(1);

        self.invalidate_cache();
        true
    }

    /// Clear all entries, resetting to empty state.
    pub fn clear(&mut self) {
        self.live_mask = Array1::zeros(0);
        self.free_list.clear();
        self.live_count = 0;
        *self
            .live_indices_cache
            .write()
            .expect("live_indices_cache write lock poisoned") = None;
    }

    // ========================================================================
    // Vectorization-Friendly Operations
    // ========================================================================

    /// Apply the live mask to scores via element-wise multiplication.
    ///
    /// This zeros out scores for tombstoned entries.
    /// Uses `azip!` for better auto-vectorization hints to LLVM.
    ///
    /// Returns the original scores unchanged if dimensions don't match.
    #[inline]
    pub fn apply_mask(&self, scores: &Array1<f32>) -> Array1<f32> {
        if self.live_mask.len() == scores.len() {
            let mut result = scores.clone();
            azip!((r in &mut result, &m in &self.live_mask) *r *= m);
            result
        } else {
            scores.clone()
        }
    }

    /// Apply the live mask in-place (mutating the scores array).
    ///
    /// Uses `azip!` for better auto-vectorization hints to LLVM.
    /// No-op if dimensions don't match.
    #[inline]
    pub fn apply_mask_inplace(&self, scores: &mut Array1<f32>) {
        if self.live_mask.len() == scores.len() {
            azip!((s in scores, &m in &self.live_mask) *s *= m);
        }
    }

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /// Grow the live mask to the specified capacity.
    fn grow_to(&mut self, new_capacity: usize) {
        if new_capacity > self.live_mask.len() {
            let mut new_mask = Array1::zeros(new_capacity);
            new_mask
                .slice_mut(s![..self.live_mask.len()])
                .assign(&self.live_mask);
            self.live_mask = new_mask;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let tm = TombstoneManager::new();
        assert_eq!(tm.live_count(), 0);
        assert_eq!(tm.capacity(), 0);
        assert!(tm.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let tm = TombstoneManager::with_capacity(10);
        assert_eq!(tm.live_count(), 0);
        assert_eq!(tm.capacity(), 10);
        assert!(tm.is_empty());
        // All slots should be in free list
        assert_eq!(tm.free_list.len(), 10);
    }

    #[test]
    fn test_allocate_reuses_tombstoned() {
        let mut tm = TombstoneManager::with_capacity(5);

        // Allocate all slots
        for i in 0..5 {
            let idx = tm.allocate();
            assert!(idx.is_some());
            assert!(tm.is_live(idx.unwrap()));
        }
        assert_eq!(tm.live_count(), 5);
        assert!(tm.allocate().is_none()); // No more free slots

        // Tombstone some
        assert!(tm.tombstone(2));
        assert!(tm.tombstone(4));
        assert_eq!(tm.live_count(), 3);

        // Allocate should reuse tombstoned slots
        let reused1 = tm.allocate();
        let reused2 = tm.allocate();
        assert!(reused1.is_some());
        assert!(reused2.is_some());
        assert!(tm.is_live(reused1.unwrap()));
        assert!(tm.is_live(reused2.unwrap()));
        assert_eq!(tm.live_count(), 5);
    }

    #[test]
    fn test_record_append() {
        let mut tm = TombstoneManager::new();

        tm.record_append(1);
        assert_eq!(tm.live_count(), 1);
        assert!(tm.is_live(0));

        tm.record_append(2);
        assert_eq!(tm.live_count(), 2);
        assert!(tm.is_live(1));
    }

    #[test]
    fn test_tombstone_returns_false_for_already_dead() {
        let mut tm = TombstoneManager::with_capacity(3);
        let allocated_slot = tm.allocate().expect("should allocate"); // Pops from end (slot 2)

        assert!(tm.tombstone(allocated_slot)); // First tombstone succeeds
        assert!(!tm.tombstone(allocated_slot)); // Already dead
    }

    #[test]
    fn test_tombstone_returns_false_for_out_of_bounds() {
        let mut tm = TombstoneManager::new();
        assert!(!tm.tombstone(100)); // Out of bounds
    }

    #[test]
    fn test_clear() {
        let mut tm = TombstoneManager::with_capacity(5);
        tm.allocate();
        tm.allocate();
        assert_eq!(tm.live_count(), 2);

        tm.clear();
        assert_eq!(tm.live_count(), 0);
        assert_eq!(tm.capacity(), 0);
        assert!(tm.is_empty());
    }

    #[test]
    fn test_apply_mask() {
        let mut tm = TombstoneManager::with_capacity(5);
        // with_capacity(5) creates free_list [0, 1, 2, 3, 4]
        // allocate() pops from end, so first three allocations return 4, 3, 2
        let slot_a = tm.allocate().expect("should allocate"); // returns 4
        let slot_b = tm.allocate().expect("should allocate"); // returns 3
        let slot_c = tm.allocate().expect("should allocate"); // returns 2

        assert_eq!(slot_a, 4);
        assert_eq!(slot_b, 3);
        assert_eq!(slot_c, 2);

        // Tombstone slot 3
        tm.tombstone(slot_b);

        // Now live_mask is: [0, 0, 1, 0, 1] (indices 2 and 4 are live)
        let scores = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6, 0.5]);
        let masked = tm.apply_mask(&scores);

        assert!((masked[0] - 0.0).abs() < 1e-6); // Never allocated -> 0
        assert!((masked[1] - 0.0).abs() < 1e-6); // Never allocated -> 0
        assert!((masked[2] - 0.7).abs() < 1e-6); // Live (slot_c)
        assert!((masked[3] - 0.0).abs() < 1e-6); // Tombstoned -> 0
        assert!((masked[4] - 0.5).abs() < 1e-6); // Live (slot_a)
    }

    #[test]
    fn test_apply_mask_inplace() {
        let mut tm = TombstoneManager::with_capacity(3);
        // with_capacity(3) creates free_list [0, 1, 2]
        // allocate() pops from end: returns 2, then 1
        let slot_a = tm.allocate().expect("should allocate"); // returns 2
        let slot_b = tm.allocate().expect("should allocate"); // returns 1
        assert_eq!(slot_a, 2);
        assert_eq!(slot_b, 1);

        // Tombstone slot 2
        tm.tombstone(slot_a);

        // Now live_mask is: [0, 1, 0] (only index 1 is live)
        let mut scores = Array1::from_vec(vec![0.9, 0.8, 0.7]);
        tm.apply_mask_inplace(&mut scores);

        assert!((scores[0] - 0.0).abs() < 1e-6); // Never allocated
        assert!((scores[1] - 0.8).abs() < 1e-6); // Live
        assert!((scores[2] - 0.0).abs() < 1e-6); // Tombstoned
    }

    #[test]
    fn test_is_live() {
        let mut tm = TombstoneManager::with_capacity(3);
        // Initially all slots are in free list (tombstoned)
        assert!(!tm.is_live(0));
        assert!(!tm.is_live(1));
        assert!(!tm.is_live(2));

        // Allocate makes them live
        tm.allocate();
        assert!(tm.is_live(2)); // Free list pops from end

        // Out of bounds is not live
        assert!(!tm.is_live(100));
    }

    #[test]
    fn test_live_indices() {
        let mut tm = TombstoneManager::with_capacity(5);
        // Initially no live entries
        assert!(tm.live_indices().is_empty());

        // Allocate some slots (pops from end: 4, 3, 2)
        tm.allocate(); // 4
        tm.allocate(); // 3
        tm.allocate(); // 2

        // Should have indices 2, 3, 4 as live (sorted)
        let indices = tm.live_indices();
        assert_eq!(indices.len(), 3);
        assert!(indices.contains(&2));
        assert!(indices.contains(&3));
        assert!(indices.contains(&4));

        // Tombstone one
        tm.tombstone(3);
        let indices = tm.live_indices();
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&2));
        assert!(!indices.contains(&3));
        assert!(indices.contains(&4));

        // Clear should reset
        tm.clear();
        assert!(tm.live_indices().is_empty());
    }

    #[test]
    fn test_live_indices_cache_invalidation() {
        let mut tm = TombstoneManager::with_capacity(3);
        tm.allocate(); // 2
        tm.allocate(); // 1

        // First call computes cache
        let indices1 = tm.live_indices();
        assert_eq!(indices1.len(), 2);

        // Tombstone invalidates cache
        tm.tombstone(2);
        let indices2 = tm.live_indices();
        assert_eq!(indices2.len(), 1);
        assert!(!indices2.contains(&2));

        // Allocate invalidates cache
        tm.allocate(); // reuses 2
        let indices3 = tm.live_indices();
        assert_eq!(indices3.len(), 2);
        assert!(indices3.contains(&2));

        // record_append invalidates cache
        tm.record_append(4);
        let indices4 = tm.live_indices();
        assert_eq!(indices4.len(), 3);
        assert!(indices4.contains(&3)); // New entry at index 3
    }

    #[test]
    fn test_live_indices_with_immutable_reference() {
        // Test that live_indices works with &self (interior mutability)
        let mut tm = TombstoneManager::with_capacity(3);
        tm.allocate(); // 2
        tm.allocate(); // 1

        // Can call live_indices() through immutable reference
        let tm_ref: &TombstoneManager = &tm;
        let indices = tm_ref.live_indices();
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
    }
}
