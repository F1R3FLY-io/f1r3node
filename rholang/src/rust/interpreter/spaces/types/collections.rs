//! Inner and Outer Collection Types
//!
//! This module defines the fundamental storage types for the 6-layer trait hierarchy
//! as specified in the "Reifying RSpaces" specification.

use std::fmt;

// ==========================================================================
// LAYER 1: Inner Collection Types (at channels)
// ==========================================================================

/// Inner collection type for data at a channel.
///
/// These determine how data/continuations are stored and matched within a channel.
/// Different collection types provide different semantics for message ordering
/// and matching behavior.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InnerCollectionType {
    /// Bag (multiset) - current default, items can appear multiple times.
    /// When matching, any element in the bag can match (non-deterministic).
    Bag,

    /// Queue - FIFO semantics, only the head can match.
    /// Provides deterministic ordering for message processing.
    Queue,

    /// Stack - LIFO semantics, only the top can match.
    /// Most recently added element is matched first.
    Stack,

    /// Set - unique elements, sending the same datum is idempotent.
    /// Duplicate sends have no effect.
    Set,

    /// Cell - at most one element, error on second send.
    /// Provides exactly-once semantics with enforcement.
    Cell,

    /// PriorityQueue - pairs of queues with priority levels.
    /// Higher priority messages are matched before lower priority ones.
    PriorityQueue {
        /// Number of priority levels (e.g., 2 for high/low)
        priorities: usize,
    },

    /// VectorDB - similarity-based matching with distance bound.
    /// For embedding-based semantic matching (AI/ML integration).
    VectorDB {
        /// Dimensionality of the embedding vectors
        dimensions: usize,
        /// Backend name (e.g., "rho", "pinecone"). Looked up in BackendRegistry.
        /// Defaults to "rho" which uses the in-memory SIMD-optimized backend.
        backend: String,
    },
}

impl Default for InnerCollectionType {
    fn default() -> Self {
        InnerCollectionType::Bag
    }
}

impl fmt::Display for InnerCollectionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InnerCollectionType::Bag => write!(f, "Bag"),
            InnerCollectionType::Queue => write!(f, "Queue"),
            InnerCollectionType::Stack => write!(f, "Stack"),
            InnerCollectionType::Set => write!(f, "Set"),
            InnerCollectionType::Cell => write!(f, "Cell"),
            InnerCollectionType::PriorityQueue { priorities } => {
                write!(f, "PriorityQueue({})", priorities)
            }
            InnerCollectionType::VectorDB { dimensions, backend } => {
                write!(f, "VectorDB({}, backend={})", dimensions, backend)
            }
        }
    }
}

// ==========================================================================
// Hyperparam Validation Schema
// ==========================================================================

/// Schema describing valid hyperparameters for a collection type.
///
/// This is used to validate hyperparameters at send time, ensuring that
/// only recognized positional hyperparameters and named keys are accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperparamSchema {
    /// Maximum number of positional hyperparameters allowed.
    pub max_positional: usize,
    /// Valid named hyperparam keys (e.g., ["priority", "ttl"]).
    pub valid_keys: &'static [&'static str],
}

impl InnerCollectionType {
    /// Returns the valid hyperparam schema for this collection type.
    ///
    /// This enables validation of hyperparameters at send time:
    /// - Rejects extra positional hyperparams beyond `max_positional`
    /// - Rejects unknown named keys not in `valid_keys`
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // PriorityQueue accepts 1 positional OR named "priority"
    /// let schema = InnerCollectionType::PriorityQueue { priorities: 3 }.hyperparam_schema();
    /// assert_eq!(schema.max_positional, 1);
    /// assert!(schema.valid_keys.contains(&"priority"));
    ///
    /// // Bag doesn't accept any hyperparams
    /// let schema = InnerCollectionType::Bag.hyperparam_schema();
    /// assert_eq!(schema.max_positional, 0);
    /// assert!(schema.valid_keys.is_empty());
    /// ```
    pub fn hyperparam_schema(&self) -> HyperparamSchema {
        match self {
            // PriorityQueue accepts priority as first positional or named "priority"
            InnerCollectionType::PriorityQueue { .. } => HyperparamSchema {
                max_positional: 1,
                valid_keys: &["priority"],
            },
            // All other collection types don't accept hyperparams
            InnerCollectionType::Bag
            | InnerCollectionType::Queue
            | InnerCollectionType::Stack
            | InnerCollectionType::Set
            | InnerCollectionType::Cell
            | InnerCollectionType::VectorDB { .. } => HyperparamSchema {
                max_positional: 0,
                valid_keys: &[],
            },
        }
    }
}

// ==========================================================================
// LAYER 2: Outer Storage Structures (channel indexing)
// ==========================================================================

/// Outer storage structure type - how channels are indexed within a space.
///
/// Different storage types provide different performance characteristics and
/// capabilities for channel lookup and management.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OuterStorageType {
    /// HashMap - O(1) lookup by channel (default for most use cases).
    /// Best for random access patterns with unique channel names.
    HashMap,

    /// PathMap - hierarchical paths with prefix matching.
    /// Sending on `@[0,1,2]` also sends on `@[0,1]` and `@[0]`.
    /// Recommended for MeTTa integration and hierarchical data.
    PathMap,

    /// Array - fixed length, gensym returns indices up to max.
    /// Returns `out_of_names` error when exhausted (unless cyclic).
    Array {
        /// Maximum number of channels
        max_size: usize,
        /// Whether to wrap around when exhausted
        cyclic: bool,
    },

    /// Vector - unbounded, gensym grows vector.
    /// Returns `out_of_memory` error at system limit.
    Vector,

    /// HashSet - for sequential processes.
    /// Channels are presence-only (send adds, receive checks membership).
    /// Used with SeqSpace for restricted sequential execution.
    HashSet,
}

impl Default for OuterStorageType {
    fn default() -> Self {
        // PathMap is recommended as the default per the spec
        OuterStorageType::PathMap
    }
}

impl fmt::Display for OuterStorageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OuterStorageType::HashMap => write!(f, "HashMap"),
            OuterStorageType::PathMap => write!(f, "PathMap"),
            OuterStorageType::Array { max_size, cyclic } => {
                write!(f, "Array({}, cyclic={})", max_size, cyclic)
            }
            OuterStorageType::Vector => write!(f, "Vector"),
            OuterStorageType::HashSet => write!(f, "HashSet"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inner_collection_display() {
        assert_eq!(format!("{}", InnerCollectionType::Bag), "Bag");
        assert_eq!(
            format!("{}", InnerCollectionType::PriorityQueue { priorities: 3 }),
            "PriorityQueue(3)"
        );
        assert_eq!(
            format!("{}", InnerCollectionType::VectorDB { dimensions: 512, backend: "rho".to_string() }),
            "VectorDB(512, backend=rho)"
        );
    }

    #[test]
    fn test_outer_storage_display() {
        assert_eq!(format!("{}", OuterStorageType::HashMap), "HashMap");
        assert_eq!(
            format!(
                "{}",
                OuterStorageType::Array {
                    max_size: 100,
                    cyclic: true
                }
            ),
            "Array(100, cyclic=true)"
        );
    }

    #[test]
    fn test_defaults() {
        assert_eq!(InnerCollectionType::default(), InnerCollectionType::Bag);
        assert_eq!(OuterStorageType::default(), OuterStorageType::PathMap);
    }

    #[test]
    fn test_hyperparam_schema_priority_queue() {
        let schema = InnerCollectionType::PriorityQueue { priorities: 3 }.hyperparam_schema();
        assert_eq!(schema.max_positional, 1);
        assert_eq!(schema.valid_keys, &["priority"]);
    }

    #[test]
    fn test_hyperparam_schema_bag() {
        let schema = InnerCollectionType::Bag.hyperparam_schema();
        assert_eq!(schema.max_positional, 0);
        assert!(schema.valid_keys.is_empty());
    }

    #[test]
    fn test_hyperparam_schema_other_types() {
        // All non-PriorityQueue types should not accept hyperparams
        let types = vec![
            InnerCollectionType::Queue,
            InnerCollectionType::Stack,
            InnerCollectionType::Set,
            InnerCollectionType::Cell,
            InnerCollectionType::VectorDB {
                dimensions: 128,
                backend: "rho".to_string(),
            },
        ];

        for t in types {
            let schema = t.hyperparam_schema();
            assert_eq!(schema.max_positional, 0, "Type {} should have max_positional=0", t);
            assert!(schema.valid_keys.is_empty(), "Type {} should have no valid keys", t);
        }
    }
}
