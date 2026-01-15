//! Allocation Mode for Channel Creation
//!
//! This module defines how `new` bindings allocate channels within different
//! space types, as specified in the "Reifying RSpaces" design document.
//!
//! # Allocation Modes
//!
//! - **Random**: Default for most spaces. Uses Blake2b512Random for cryptographic IDs.
//! - **ArrayIndex**: For Array spaces. Sequential indices up to max_size, wrapped in Unforgeable.
//! - **VectorIndex**: For Vector spaces. Growing indices, wrapped in Unforgeable.
//!
//! # Unforgeable Wrapping
//!
//! Array and Vector indices are wrapped in `GPrivate` with format:
//! `[space_id bytes (32)] ++ [index big-endian (8)]` = 40 bytes total
//!
//! This ensures:
//! - Determinism (required for blockchain consensus)
//! - Unforgeability (cannot guess the space_id prefix)
//! - Efficient lookup (extract last 8 bytes for O(1) access)

use std::fmt;

/// Allocation mode for `new` bindings within a space.
///
/// Different outer storage types use different allocation strategies for
/// creating new channel names within `use space { new x in { ... } }` blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocationMode {
    /// Random allocation using Blake2b512Random.
    /// Used by: HashMap, PathMap, HashSet (and default space).
    Random,

    /// Sequential index allocation for fixed-size arrays.
    /// Returns indices 0, 1, 2, ... up to max_size-1.
    /// In non-cyclic mode, returns OutOfNames error when exhausted.
    /// In cyclic mode, wraps around to 0.
    /// Indices are wrapped in Unforgeable with space_id prefix.
    ArrayIndex {
        /// Maximum number of channels in the array
        max_size: usize,
        /// Whether to wrap around (cyclic) or error (non-cyclic)
        cyclic: bool,
    },

    /// Sequential index allocation for unbounded vectors.
    /// Returns indices 0, 1, 2, ... growing without limit.
    /// Only fails on out-of-memory.
    /// Indices are wrapped in Unforgeable with space_id prefix.
    VectorIndex,
}

impl AllocationMode {
    /// Returns true if this mode uses index-based allocation (wrapped in Unforgeable).
    pub fn is_indexed(&self) -> bool {
        matches!(self, AllocationMode::ArrayIndex { .. } | AllocationMode::VectorIndex)
    }

    /// Returns true if this mode uses random ID allocation.
    pub fn is_random(&self) -> bool {
        matches!(self, AllocationMode::Random)
    }
}

impl Default for AllocationMode {
    fn default() -> Self {
        AllocationMode::Random
    }
}

impl fmt::Display for AllocationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocationMode::Random => write!(f, "Random"),
            AllocationMode::ArrayIndex { max_size, cyclic } => {
                write!(f, "ArrayIndex(max={}, cyclic={})", max_size, cyclic)
            }
            AllocationMode::VectorIndex => write!(f, "VectorIndex"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_mode_display() {
        assert_eq!(format!("{}", AllocationMode::Random), "Random");
        assert_eq!(
            format!("{}", AllocationMode::ArrayIndex { max_size: 10, cyclic: false }),
            "ArrayIndex(max=10, cyclic=false)"
        );
        assert_eq!(format!("{}", AllocationMode::VectorIndex), "VectorIndex");
    }

    #[test]
    fn test_is_indexed() {
        assert!(!AllocationMode::Random.is_indexed());
        assert!(AllocationMode::ArrayIndex { max_size: 10, cyclic: false }.is_indexed());
        assert!(AllocationMode::VectorIndex.is_indexed());
    }

    #[test]
    fn test_is_random() {
        assert!(AllocationMode::Random.is_random());
        assert!(!AllocationMode::ArrayIndex { max_size: 10, cyclic: false }.is_random());
        assert!(!AllocationMode::VectorIndex.is_random());
    }

    #[test]
    fn test_default() {
        assert_eq!(AllocationMode::default(), AllocationMode::Random);
    }
}
