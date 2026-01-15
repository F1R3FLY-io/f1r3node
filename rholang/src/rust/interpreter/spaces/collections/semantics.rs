//! Semantic Equality, Hashing, and Ordering Types
//!
//! This module defines traits and types for semantic comparison of data.
//!
//! - `SemanticEq`: Semantic equality (ignoring metadata like random_state)
//! - `SemanticHash`: Semantic hashing for O(1) set deduplication
//! - `TopKEntry`: Min-heap entry for efficient top-K selection

use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use models::rhoapi::ListParWithRandom;

// ==========================================================================
// Semantic Equality Trait
// ==========================================================================

/// Trait for semantic data equality (ignoring metadata like random_state).
///
/// This trait allows collections to perform deduplication based on the semantic
/// content of data rather than structural equality. For example, `ListParWithRandom`
/// contains both `pars` (the actual data) and `random_state` (per-send metadata).
/// Semantic equality compares only the `pars` field.
///
/// # Usage
/// Types that implement this trait can be used with `SetDataCollection` for
/// proper deduplication semantics.
pub trait SemanticEq {
    /// Check if two items are semantically equal.
    ///
    /// For most types, this is the same as `PartialEq::eq`. For types with
    /// metadata fields (like `ListParWithRandom`), this ignores the metadata.
    fn semantically_eq(&self, other: &Self) -> bool;
}

/// Semantic equality for ListParWithRandom compares only `pars`, ignoring `random_state`.
///
/// This is critical for Set deduplication: each send has a unique `random_state`
/// (from Blake2b512Random splitting), but the actual data in `pars` is what
/// determines semantic identity. Without this, sends like `ch!("alice")` would
/// never deduplicate because each has a different `random_state`.
impl SemanticEq for ListParWithRandom {
    fn semantically_eq(&self, other: &Self) -> bool {
        self.pars == other.pars
    }
}

// Implement SemanticEq for primitive types (uses standard equality)
impl SemanticEq for i32 {
    fn semantically_eq(&self, other: &Self) -> bool { self == other }
}

impl SemanticEq for i64 {
    fn semantically_eq(&self, other: &Self) -> bool { self == other }
}

impl SemanticEq for u32 {
    fn semantically_eq(&self, other: &Self) -> bool { self == other }
}

impl SemanticEq for u64 {
    fn semantically_eq(&self, other: &Self) -> bool { self == other }
}

impl SemanticEq for String {
    fn semantically_eq(&self, other: &Self) -> bool { self == other }
}

// ==========================================================================
// Semantic Hashing Trait
// ==========================================================================

/// Trait for semantic hashing (for O(1) deduplication in sets).
///
/// This trait complements `SemanticEq` by providing a hash value that is
/// consistent with semantic equality: if `a.semantically_eq(b)`, then
/// `a.semantic_hash() == b.semantic_hash()`.
///
/// # Usage
/// Types that implement both `SemanticEq` and `SemanticHash` can benefit from
/// O(1) deduplication in `SetDataCollection` via hash-based indexing.
///
/// # Implementation Note
/// The hash is computed using the same fields that determine semantic equality.
/// For `ListParWithRandom`, this means hashing only the `pars` field, not `random_state`.
pub trait SemanticHash: SemanticEq {
    /// Compute a semantic hash value.
    ///
    /// The hash must be consistent with `semantically_eq`: items that are
    /// semantically equal must produce the same hash value.
    fn semantic_hash(&self) -> u64;
}

/// Semantic hash for ListParWithRandom hashes only `pars`, consistent with semantic equality.
impl SemanticHash for ListParWithRandom {
    fn semantic_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        // Hash each par individually since ListParWithRandom doesn't implement Hash
        for par in &self.pars {
            // Use debug format as a proxy for par identity
            // A more sophisticated implementation could use protobuf serialization
            Hash::hash(&format!("{:?}", par), &mut hasher);
        }
        hasher.finish()
    }
}

// Implement SemanticHash for primitive types
impl SemanticHash for i32 {
    fn semantic_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        Hash::hash(self, &mut hasher);
        hasher.finish()
    }
}

impl SemanticHash for i64 {
    fn semantic_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        Hash::hash(self, &mut hasher);
        hasher.finish()
    }
}

impl SemanticHash for u32 {
    fn semantic_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        Hash::hash(self, &mut hasher);
        hasher.finish()
    }
}

impl SemanticHash for u64 {
    fn semantic_hash(&self) -> u64 {
        // For u64, the value itself is the hash
        *self
    }
}

impl SemanticHash for String {
    fn semantic_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        Hash::hash(self, &mut hasher);
        hasher.finish()
    }
}

// ==========================================================================
// Min-Heap Entry for Top-K Selection
// ==========================================================================

/// Entry for min-heap top-K selection.
///
/// Implements `Ord` in reverse order to turn `BinaryHeap` (max-heap) into a min-heap.
/// This enables O(n log K) top-K selection with O(K) space complexity.
///
/// # Algorithm
/// - Maintain a min-heap of size K (smallest of top-K at root)
/// - For each candidate: if better than root, replace root
/// - Result: heap contains exactly the K best candidates
#[derive(Clone, Copy, Debug)]
pub(crate) struct TopKEntry {
    pub similarity: f32,
    pub index: usize,
}

impl TopKEntry {
    /// Create a new TopKEntry.
    pub fn new(similarity: f32, index: usize) -> Self {
        Self { similarity, index }
    }
}

impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for TopKEntry {}

impl PartialOrd for TopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TopKEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order: smaller similarity compares as Greater
        // This makes BinaryHeap.peek() return the entry with smallest similarity,
        // enabling efficient "kick out the worst of top-K" logic.
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    #[test]
    fn test_semantic_eq_primitives() {
        assert!(42i32.semantically_eq(&42));
        assert!(!42i32.semantically_eq(&43));
        assert!("hello".to_string().semantically_eq(&"hello".to_string()));
    }

    #[test]
    fn test_semantic_hash_consistency() {
        // Same value should give same hash
        let hash1 = 42i64.semantic_hash();
        let hash2 = 42i64.semantic_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_topk_entry_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(TopKEntry::new(0.9, 0));
        heap.push(TopKEntry::new(0.5, 1));
        heap.push(TopKEntry::new(0.7, 2));

        // Min-heap: peek should return smallest similarity
        assert_eq!(heap.peek().expect("expected peek value").similarity, 0.5);
    }
}
