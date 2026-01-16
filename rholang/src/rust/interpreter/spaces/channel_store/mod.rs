//! Layer 2: Outer Storage Structures (Channel Store)
//!
//! This module defines traits and implementations for how channels are indexed
//! and organized within a space. Different storage types provide different
//! performance characteristics and capabilities.
//!
//! # Storage Types
//!
//! - **HashMap**: O(1) lookup by channel key
//! - **PathMap**: Hierarchical paths with prefix matching (for MeTTa)
//! - **Array**: Fixed size, gensym returns indices
//! - **Vector**: Unbounded, gensym grows the vector
//! - **HashSet**: Presence-only for sequential processes

use std::hash::Hash;

// DataCollection and ContinuationCollection are used by consumers of this module
pub use super::errors::SpaceError;
pub use super::types::{AllocationMode, SpaceId};
pub use super::collections::{DataCollection, ContinuationCollection};

// ==========================================================================
// Channel Store Trait
// ==========================================================================

/// Trait for channel storage - how channels are indexed and accessed.
///
/// This trait uses associated types to specify the complete type configuration
/// for a channel store. This allows `GenericRSpace<CS, M>` to infer all types
/// from the store type alone.
///
/// # Associated Types
///
/// - `Channel`: The channel/name type (e.g., `Par`, `usize`, `Vec<u8>`)
/// - `Pattern`: The pattern type for continuation matching
/// - `Data`: The data type stored in channels
/// - `Continuation`: The continuation type
/// - `DataColl`: The data collection type (e.g., `BagDataCollection<A>`)
/// - `ContColl`: The continuation collection type
///
/// # Prefix Semantics
///
/// Some storage types (PathMap) support prefix semantics where data sent on a
/// child path is visible when receiving at a parent prefix. Stores can indicate
/// this capability via `supports_prefix_semantics()`.
///
/// When prefix semantics are enabled:
/// - `produce(@[0,1,2], data)` stores data at exact path `@[0,1,2]`
/// - `consume(@[0,1], pattern)` can receive data from `@[0,1,2]` with suffix `[2]`
///
/// # Formal Correspondence
/// - `PathMapStore.v`: Prefix aggregation theorems
/// - `PathMapQuantale.v`: Path concatenation properties
///
/// # Design Document Alignment
///
/// This trait aligns with the design document's `ChannelStore<C, P, A, K>` specification
/// (lines 405-441) but uses fully associated types to enable `GenericRSpace<CS, M>`
/// (2 parameters) instead of `GenericRSpace<CS, M, C, P, A, K, DC, CC>` (8 parameters).
pub trait ChannelStore: Clone + Send + Sync {
    /// The channel type (e.g., `Par`, `usize`, `Vec<u8>`).
    type Channel: Clone + Eq + Hash + Send + Sync;

    /// The pattern type for continuation matching.
    type Pattern: Clone + Send + Sync;

    /// The data type stored in channels.
    type Data: Clone + Send + Sync + std::fmt::Debug + 'static;

    /// The continuation type.
    type Continuation: Clone + Send + Sync;

    /// The data collection type for storing data at channels.
    type DataColl: DataCollection<Self::Data> + Default + Clone + Send + Sync + 'static;

    /// The continuation collection type for storing continuations.
    type ContColl: ContinuationCollection<Self::Pattern, Self::Continuation> + Default + Clone + Send + Sync;

    // =========================================================================
    // Core Channel Operations
    // =========================================================================

    /// Get or create a data collection for the given channel.
    fn get_or_create_data_collection(&mut self, channel: &Self::Channel) -> &mut Self::DataColl;

    /// Get a data collection for the given channel if it exists.
    fn get_data_collection(&self, channel: &Self::Channel) -> Option<&Self::DataColl>;

    /// Get a mutable data collection for the given channel if it exists.
    fn get_data_collection_mut(&mut self, channel: &Self::Channel) -> Option<&mut Self::DataColl>;

    /// Get or create a continuation collection for the given channel pattern.
    fn get_or_create_continuation_collection(&mut self, channels: &[Self::Channel]) -> &mut Self::ContColl;

    /// Get a continuation collection for the given channel pattern if it exists.
    fn get_continuation_collection(&self, channels: &[Self::Channel]) -> Option<&Self::ContColl>;

    /// Get a mutable continuation collection for the given channel pattern.
    fn get_continuation_collection_mut(&mut self, channels: &[Self::Channel]) -> Option<&mut Self::ContColl>;

    /// Get all channels in the store.
    fn all_channels(&self) -> Vec<&Self::Channel>;

    /// Generate a new unique channel name.
    fn gensym(&mut self, space_id: &SpaceId) -> Result<Self::Channel, SpaceError>;

    /// Get join patterns for a channel (channels that participate in joins with this channel).
    fn get_joins(&self, channel: &Self::Channel) -> Vec<Vec<Self::Channel>>;

    /// Record a join pattern.
    fn put_join(&mut self, channels: Vec<Self::Channel>);

    /// Remove a join pattern.
    fn remove_join(&mut self, channels: &[Self::Channel]);

    /// Create a snapshot of the store for checkpointing.
    fn snapshot(&self) -> Self;

    /// Clear all data and continuations from the store.
    fn clear(&mut self);

    /// Check if the store is empty.
    fn is_empty(&self) -> bool;

    /// Export all data collections for serialization.
    ///
    /// Returns a vector of (channel, data_collection) pairs.
    /// Used for checkpointing state to persistent storage.
    fn export_data(&self) -> Vec<(Self::Channel, Self::DataColl)>;

    /// Export all continuation collections for serialization.
    ///
    /// Returns a vector of (channel_pattern, continuation_collection) pairs.
    /// Used for checkpointing state to persistent storage.
    fn export_continuations(&self) -> Vec<(Vec<Self::Channel>, Self::ContColl)>;

    /// Export all join patterns for serialization.
    ///
    /// Returns a vector of (channel, join_patterns) pairs.
    /// Used for checkpointing state to persistent storage.
    fn export_joins(&self) -> Vec<(Self::Channel, Vec<Vec<Self::Channel>>)>;

    // =========================================================================
    // Zero-Copy Iteration (Callback-Based)
    // =========================================================================

    /// Iterate over all data collections without cloning.
    ///
    /// This is more efficient than `export_data()` when you only need to
    /// read the data without taking ownership. Useful for:
    /// - Streaming serialization
    /// - Computing statistics
    /// - Inspection/debugging
    ///
    /// # Arguments
    /// * `f` - Callback function called for each (channel, data_collection) pair
    fn for_each_data<F>(&self, f: F)
    where
        F: FnMut(&Self::Channel, &Self::DataColl);

    /// Iterate over all continuation collections without cloning.
    ///
    /// This is more efficient than `export_continuations()` when you only need
    /// to read the continuations without taking ownership.
    ///
    /// # Arguments
    /// * `f` - Callback function called for each (channels, continuation_collection) pair
    fn for_each_continuation<F>(&self, f: F)
    where
        F: FnMut(&[Self::Channel], &Self::ContColl);

    /// Iterate over all join patterns without cloning.
    ///
    /// This is more efficient than `export_joins()` when you only need
    /// to read the join patterns without taking ownership.
    ///
    /// # Arguments
    /// * `f` - Callback function called for each (channel, join_patterns) pair
    fn for_each_join<F>(&self, f: F)
    where
        F: FnMut(&Self::Channel, &[Vec<Self::Channel>]);

    /// Get the current gensym counter value for serialization.
    fn gensym_counter(&self) -> usize;

    /// Import data collections from deserialized state.
    ///
    /// Replaces existing data with the provided collections.
    fn import_data(&mut self, data: Vec<(Self::Channel, Self::DataColl)>);

    /// Import continuation collections from deserialized state.
    ///
    /// Replaces existing continuations with the provided collections.
    fn import_continuations(&mut self, continuations: Vec<(Vec<Self::Channel>, Self::ContColl)>);

    /// Import join patterns from deserialized state.
    ///
    /// Replaces existing joins with the provided patterns.
    fn import_joins(&mut self, joins: Vec<(Self::Channel, Vec<Vec<Self::Channel>>)>);

    /// Set the gensym counter value from deserialized state.
    fn set_gensym_counter(&mut self, counter: usize);

    // =========================================================================
    // Prefix Semantics (PathMap Support)
    // =========================================================================

    /// Check if this store supports prefix semantics.
    ///
    /// When true, data at path `@[0,1,2]` can be received by a consumer at `@[0,1]`.
    /// The suffix `[2]` is attached to the data as a key for pattern matching.
    ///
    /// # Default
    /// Returns `false` for most stores. Only PathMapChannelStore returns `true`.
    fn supports_prefix_semantics(&self) -> bool {
        false
    }

    /// Get all channels that have the given channel as a prefix.
    ///
    /// For PathMap, this returns all paths that start with `prefix`.
    /// For other stores, returns an empty vector (no prefix relationship).
    ///
    /// # Example (PathMap)
    /// If store contains `@[0,1,2]` and `@[0,1,3]`, then:
    /// `channels_with_prefix(@[0,1])` returns `[@[0,1,2], @[0,1,3]]`
    fn channels_with_prefix(&self, _prefix: &Self::Channel) -> Vec<Self::Channel> {
        Vec::new()
    }

    /// Get all prefixes of the given channel.
    ///
    /// For PathMap, returns all prefix paths from shortest to longest.
    /// For other stores, returns an empty vector (no prefix relationship).
    ///
    /// # Example (PathMap)
    /// `channel_prefixes(@[0,1,2])` returns `[@[0], @[0,1], @[0,1,2]]`
    fn channel_prefixes(&self, _channel: &Self::Channel) -> Vec<Self::Channel> {
        Vec::new()
    }

    /// Get all continuation patterns that involve the given channel as a prefix.
    ///
    /// This finds continuations waiting on prefix paths of the given channel.
    /// Used during `produce()` to find matching continuations at prefix paths.
    ///
    /// # Returns
    /// Vector of (pattern_channels, continuation_collection) pairs where
    /// at least one channel in pattern_channels is a prefix of `channel`.
    fn continuation_patterns_for_prefix(&self, _channel: &Self::Channel) -> Vec<(&Vec<Self::Channel>, &Self::ContColl)> {
        Vec::new()
    }

    /// Compute the suffix key for a descendant channel relative to a prefix channel.
    ///
    /// When a consumer at prefix path `@[0,1]` receives data from `@[0,1,2,3]`,
    /// the suffix key is `[2,3]` - the path elements that extend the prefix.
    ///
    /// # Returns
    /// - `Some(suffix_bytes)` if descendant is a proper descendant of prefix
    /// - `None` if the channels are equal (exact match, no suffix)
    /// - `None` if the channels are not in a prefix relationship
    ///
    /// # Default
    /// Returns `None` for most stores. Only PathMap stores implement this.
    fn compute_suffix_key(&self, _prefix: &Self::Channel, _descendant: &Self::Channel) -> Option<Vec<u8>> {
        None
    }

    // =========================================================================
    // Allocation Mode (Array/Vector Index Support)
    // =========================================================================

    /// Get the allocation mode for `new` bindings within this space.
    ///
    /// Different storage types use different allocation strategies:
    /// - **Random**: HashMap, PathMap, HashSet use cryptographic random IDs
    /// - **ArrayIndex**: Array stores use sequential indices up to max_size
    /// - **VectorIndex**: Vector stores use growing indices
    ///
    /// # Default
    /// Returns `AllocationMode::Random` for most stores.
    fn allocation_mode(&self) -> AllocationMode {
        AllocationMode::Random
    }
}

// Module declarations
mod hashmap_store;
mod array_store;
mod vector_store;
mod pathmap_store;
mod rholang_pathmap_store;
mod hashset_store;
mod vectordb_store;

// Re-export all store types
pub use hashmap_store::HashMapChannelStore;
pub use array_store::ArrayChannelStore;
pub use vector_store::VectorChannelStore;
pub use pathmap_store::PathMapChannelStore;
pub use rholang_pathmap_store::RholangPathMapStore;
pub use hashset_store::HashSetChannelStore;
pub use vectordb_store::VectorDBChannelStore;
