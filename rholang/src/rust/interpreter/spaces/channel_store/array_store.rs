//! ArrayChannelStore: Fixed-size indexed channel storage.

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use super::{ChannelStore, DataCollection, ContinuationCollection, SpaceId, SpaceError, AllocationMode};

/// Array-based channel store with fixed size.
///
/// Channels are created from indices via a channel factory function.
/// Gensym returns the next available index wrapped as a channel, and
/// returns `OutOfNames` when the array is full (unless cyclic mode is enabled).
///
/// # Type Parameters
///
/// - `C`: Channel type (e.g., `Par`, created from index via channel_factory)
/// - `P`: Pattern type for continuation matching
/// - `A`: Data type stored in channels
/// - `K`: Continuation type
/// - `DC`: Data collection type
/// - `CC`: Continuation collection type
///
/// # Design Document Alignment
///
/// Per the Reifying RSpaces design document (lines 194-204):
/// - Array channels are allocated sequentially via gensym (indices 0, 1, 2, ...)
/// - Indices are wrapped in Unforgeable{} so clients can't forge them
/// - Non-cyclic: returns OutOfNames error when capacity exceeded
/// - Cyclic: wraps around (ring buffer semantics)
#[derive(Debug)]
pub struct ArrayChannelStore<C, P, A, K, DC, CC>
where
    C: Clone + Eq + Hash,
    DC: Clone,
    CC: Clone,
{
    /// Data collections indexed by position (internal usize index)
    data: Vec<Option<DC>>,

    /// Continuation collections keyed by channel pattern
    continuations: HashMap<Vec<C>, CC>,

    /// Join patterns: channel -> list of join patterns it participates in
    joins: HashMap<C, Vec<Vec<C>>>,

    /// Channel to internal index mapping (for data access)
    channel_to_index: HashMap<C, usize>,

    /// Index to channel mapping (for export and iteration)
    index_to_channel: Vec<Option<C>>,

    /// Maximum size of the array
    max_size: usize,

    /// Whether to wrap around when full
    cyclic: bool,

    /// Next index to allocate
    next_index: usize,

    /// Space ID for channel creation
    space_id: SpaceId,

    /// Factory function to create channels from indices
    channel_factory: fn(&SpaceId, usize) -> C,

    /// Extractor function to get index from a channel (reverse of channel_factory)
    /// Returns Some(index) if the channel matches the expected pattern, None otherwise
    index_extractor: fn(&SpaceId, &C) -> Option<usize>,

    /// Factory function to create new data collections
    data_factory: fn() -> DC,

    /// Factory function to create new continuation collections
    cont_factory: fn() -> CC,

    /// PhantomData to track P, A, K
    _phantom: PhantomData<(P, A, K)>,
}

impl<C, P, A, K, DC, CC> Clone for ArrayChannelStore<C, P, A, K, DC, CC>
where
    C: Clone + Eq + Hash,
    DC: Clone,
    CC: Clone,
{
    fn clone(&self) -> Self {
        ArrayChannelStore {
            data: self.data.clone(),
            continuations: self.continuations.clone(),
            joins: self.joins.clone(),
            channel_to_index: self.channel_to_index.clone(),
            index_to_channel: self.index_to_channel.clone(),
            max_size: self.max_size,
            cyclic: self.cyclic,
            next_index: self.next_index,
            space_id: self.space_id.clone(),
            channel_factory: self.channel_factory,
            index_extractor: self.index_extractor,
            data_factory: self.data_factory,
            cont_factory: self.cont_factory,
            _phantom: PhantomData,
        }
    }
}

impl<C, P, A, K, DC, CC> ArrayChannelStore<C, P, A, K, DC, CC>
where
    C: Clone + Eq + Hash + Send + Sync,
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync,
    DC: Clone + Send + Sync,
    CC: Clone + Send + Sync,
{
    /// Create a new array channel store with the given channel factory.
    ///
    /// # Arguments
    ///
    /// * `max_size` - Maximum number of channels that can be allocated
    /// * `cyclic` - If true, wrap around to index 0 when full (ring buffer)
    /// * `space_id` - Space ID for channel creation (passed to channel_factory)
    /// * `channel_factory` - Function to create a channel from (space_id, index)
    /// * `index_extractor` - Function to extract index from a channel (reverse of channel_factory)
    /// * `data_factory` - Function to create empty data collections
    /// * `cont_factory` - Function to create empty continuation collections
    pub fn new(
        max_size: usize,
        cyclic: bool,
        space_id: SpaceId,
        channel_factory: fn(&SpaceId, usize) -> C,
        index_extractor: fn(&SpaceId, &C) -> Option<usize>,
        data_factory: fn() -> DC,
        cont_factory: fn() -> CC,
    ) -> Self {
        ArrayChannelStore {
            data: vec![None; max_size],
            continuations: HashMap::new(),
            joins: HashMap::new(),
            channel_to_index: HashMap::with_capacity(max_size),
            index_to_channel: vec![None; max_size],
            max_size,
            cyclic,
            next_index: 0,
            space_id,
            channel_factory,
            index_extractor,
            data_factory,
            cont_factory,
            _phantom: PhantomData,
        }
    }

    /// Get the internal index for a channel (immutable lookup only).
    fn get_index(&self, channel: &C) -> Option<usize> {
        self.channel_to_index.get(channel).copied()
    }

    /// Try to extract index from a channel and register it if valid.
    /// Returns the index if successful, None if the channel doesn't match this space.
    fn try_register_channel(&mut self, channel: &C) -> Option<usize> {
        // First check if already registered
        if let Some(index) = self.channel_to_index.get(channel) {
            return Some(*index);
        }

        // Try to extract index from the channel using the extractor
        let extracted = (self.index_extractor)(&self.space_id, channel);

        let index = match extracted {
            Some(idx) => idx,
            None => {
                // Extractor returned None - channel doesn't match this space's pattern
                return None;
            }
        };

        // Validate index is within bounds
        if index >= self.max_size {
            // For cyclic arrays, indices should have been wrapped by the reducer.
            // If we see an out-of-bounds index, it means the reducer didn't wrap.
            // This could happen if the channel was created before cyclic wrapping was added,
            // or if there's a bug in the allocation logic.
            if self.cyclic {
                // For cyclic arrays, wrap the index as a fallback
                let wrapped_index = index % self.max_size;
                // Register with wrapped index
                self.channel_to_index.insert(channel.clone(), wrapped_index);
                self.index_to_channel[wrapped_index] = Some(channel.clone());
                return Some(wrapped_index);
            } else {
                return None;
            }
        }

        // Register the channel
        self.channel_to_index.insert(channel.clone(), index);
        self.index_to_channel[index] = Some(channel.clone());

        Some(index)
    }

    /// Create a normalized key for continuation lookup.
    /// Channels in a join pattern are sorted for consistent lookup.
    fn normalize_channels(channels: &[C]) -> Vec<C>
    where
        C: Ord,
    {
        let mut sorted = channels.to_vec();
        sorted.sort();
        sorted
    }
}

impl<C, P, A, K, DC, CC> ChannelStore for ArrayChannelStore<C, P, A, K, DC, CC>
where
    C: Clone + Eq + Hash + Ord + Send + Sync,
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync,
    DC: DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: ContinuationCollection<P, K> + Default + Clone + Send + Sync,
{
    type Channel = C;
    type Pattern = P;
    type Data = A;
    type Continuation = K;
    type DataColl = DC;
    type ContColl = CC;

    fn get_or_create_data_collection(&mut self, channel: &C) -> &mut DC {
        // Try to register the channel if not already known (auto-registration for reducer-created channels)
        let index = self.try_register_channel(channel)
            .expect("Channel does not match this Array space pattern");

        if self.data[index].is_none() {
            self.data[index] = Some((self.data_factory)());
        }
        self.data[index].as_mut().expect("Data collection should exist after creation")
    }

    fn get_data_collection(&self, channel: &C) -> Option<&DC> {
        let index = self.get_index(channel)?;
        self.data.get(index)?.as_ref()
    }

    fn get_data_collection_mut(&mut self, channel: &C) -> Option<&mut DC> {
        let index = self.get_index(channel)?;
        self.data.get_mut(index)?.as_mut()
    }

    fn get_or_create_continuation_collection(&mut self, channels: &[C]) -> &mut CC {
        let key = Self::normalize_channels(channels);
        self.continuations
            .entry(key)
            .or_insert_with(|| (self.cont_factory)())
    }

    fn get_continuation_collection(&self, channels: &[C]) -> Option<&CC> {
        let key = Self::normalize_channels(channels);
        self.continuations.get(&key)
    }

    fn get_continuation_collection_mut(&mut self, channels: &[C]) -> Option<&mut CC> {
        let key = Self::normalize_channels(channels);
        self.continuations.get_mut(&key)
    }

    fn all_channels(&self) -> Vec<&C> {
        self.index_to_channel
            .iter()
            .filter_map(|opt| opt.as_ref())
            .collect()
    }

    fn gensym(&mut self, _space_id: &SpaceId) -> Result<C, SpaceError> {
        if self.next_index >= self.max_size {
            if self.cyclic {
                // Cyclic wrap-around: clear old mapping before reusing index
                self.next_index = 0;
            } else {
                return Err(SpaceError::OutOfNames {
                    space_id: self.space_id.clone(),
                    max_size: self.max_size,
                });
            }
        }

        let index = self.next_index;

        // If cyclic and reusing an index, remove old channel mapping
        if let Some(old_channel) = self.index_to_channel[index].take() {
            self.channel_to_index.remove(&old_channel);
        }

        // Create new channel from index using the factory
        let channel = (self.channel_factory)(&self.space_id, index);

        // Store bidirectional mapping
        self.channel_to_index.insert(channel.clone(), index);
        self.index_to_channel[index] = Some(channel.clone());

        self.next_index += 1;
        Ok(channel)
    }

    fn get_joins(&self, channel: &C) -> Vec<Vec<C>> {
        self.joins.get(channel).cloned().unwrap_or_default()
    }

    fn put_join(&mut self, channels: Vec<C>) {
        for channel in &channels {
            self.joins
                .entry(channel.clone())
                .or_insert_with(Vec::new)
                .push(channels.clone());
        }
    }

    fn remove_join(&mut self, channels: &[C]) {
        for channel in channels {
            if let Some(joins) = self.joins.get_mut(channel) {
                joins.retain(|j| j != channels);
            }
        }
    }

    fn snapshot(&self) -> Self {
        self.clone()
    }

    fn clear(&mut self) {
        for slot in &mut self.data {
            *slot = None;
        }
        self.continuations.clear();
        self.joins.clear();
        self.channel_to_index.clear();
        for slot in &mut self.index_to_channel {
            *slot = None;
        }
        self.next_index = 0;
    }

    fn is_empty(&self) -> bool {
        self.data.iter().all(|d| d.is_none()) && self.continuations.is_empty()
    }

    fn export_data(&self) -> Vec<(C, DC)> {
        self.index_to_channel
            .iter()
            .enumerate()
            .filter_map(|(i, opt_channel)| {
                opt_channel.as_ref().and_then(|channel| {
                    self.data.get(i)?.as_ref().map(|dc| (channel.clone(), dc.clone()))
                })
            })
            .collect()
    }

    fn export_continuations(&self) -> Vec<(Vec<C>, CC)> {
        self.continuations
            .iter()
            .map(|(cs, cc)| (cs.clone(), cc.clone()))
            .collect()
    }

    fn export_joins(&self) -> Vec<(C, Vec<Vec<C>>)> {
        self.joins
            .iter()
            .map(|(c, js)| (c.clone(), js.clone()))
            .collect()
    }

    fn for_each_data<F>(&self, mut f: F)
    where
        F: FnMut(&C, &DC),
    {
        for (i, opt_channel) in self.index_to_channel.iter().enumerate() {
            if let Some(channel) = opt_channel {
                if let Some(Some(dc)) = self.data.get(i) {
                    f(channel, dc);
                }
            }
        }
    }

    fn for_each_continuation<F>(&self, mut f: F)
    where
        F: FnMut(&[C], &CC),
    {
        for (cs, cc) in &self.continuations {
            f(cs, cc);
        }
    }

    fn for_each_join<F>(&self, mut f: F)
    where
        F: FnMut(&C, &[Vec<C>]),
    {
        for (c, js) in &self.joins {
            f(c, js);
        }
    }

    fn gensym_counter(&self) -> usize {
        self.next_index
    }

    fn import_data(&mut self, data: Vec<(C, DC)>) {
        // Clear existing data and mappings
        for slot in &mut self.data {
            *slot = None;
        }
        self.channel_to_index.clear();
        for slot in &mut self.index_to_channel {
            *slot = None;
        }

        // Import new data - we need to reconstruct the channel mappings
        // Since we don't know the original indices, we allocate sequentially
        let mut next_idx = 0;
        for (channel, dc) in data {
            if next_idx < self.max_size {
                self.data[next_idx] = Some(dc);
                self.channel_to_index.insert(channel.clone(), next_idx);
                self.index_to_channel[next_idx] = Some(channel);
                next_idx += 1;
            }
        }
        self.next_index = next_idx;
    }

    fn import_continuations(&mut self, continuations: Vec<(Vec<C>, CC)>) {
        self.continuations.clear();
        for (channels, cc) in continuations {
            self.continuations.insert(channels, cc);
        }
    }

    fn import_joins(&mut self, joins: Vec<(C, Vec<Vec<C>>)>) {
        self.joins.clear();
        for (channel, join_patterns) in joins {
            self.joins.insert(channel, join_patterns);
        }
    }

    fn set_gensym_counter(&mut self, counter: usize) {
        self.next_index = counter;
    }

    fn allocation_mode(&self) -> AllocationMode {
        AllocationMode::ArrayIndex {
            max_size: self.max_size,
            cyclic: self.cyclic,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::interpreter::spaces::collections::{BagDataCollection, BagContinuationCollection};

    /// Helper function to create a String channel from an index (for tests)
    fn string_channel_factory(_space_id: &SpaceId, index: usize) -> String {
        format!("channel_{}", index)
    }

    /// Helper function to extract index from a String channel (for tests)
    fn string_index_extractor(_space_id: &SpaceId, channel: &String) -> Option<usize> {
        channel.strip_prefix("channel_")?.parse().ok()
    }

    #[test]
    fn test_array_store_out_of_names() {
        let mut store: ArrayChannelStore<String, String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            ArrayChannelStore::new(
                3, // Only 3 slots
                false, // Not cyclic
                SpaceId::default_space(),
                string_channel_factory,
                string_index_extractor,
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        let space_id = SpaceId::default_space();
        store.gensym(&space_id).unwrap();
        store.gensym(&space_id).unwrap();
        store.gensym(&space_id).unwrap();

        // Fourth should fail
        let result = store.gensym(&space_id);
        assert!(matches!(result, Err(SpaceError::OutOfNames { .. })));
    }

    #[test]
    fn test_array_store_cyclic() {
        let mut store: ArrayChannelStore<String, String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            ArrayChannelStore::new(
                3,
                true, // Cyclic
                SpaceId::default_space(),
                string_channel_factory,
                string_index_extractor,
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        let space_id = SpaceId::default_space();
        assert_eq!(store.gensym(&space_id).unwrap(), "channel_0");
        assert_eq!(store.gensym(&space_id).unwrap(), "channel_1");
        assert_eq!(store.gensym(&space_id).unwrap(), "channel_2");
        assert_eq!(store.gensym(&space_id).unwrap(), "channel_0"); // Wraps around
    }

    #[test]
    fn test_array_allocation_mode_returns_array_index() {
        let store: ArrayChannelStore<String, String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            ArrayChannelStore::new(
                10, // max_size
                false, // cyclic
                SpaceId::default_space(),
                string_channel_factory,
                string_index_extractor,
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        assert_eq!(
            store.allocation_mode(),
            AllocationMode::ArrayIndex { max_size: 10, cyclic: false }
        );
    }

    #[test]
    fn test_array_allocation_mode_cyclic() {
        let store: ArrayChannelStore<String, String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            ArrayChannelStore::new(
                5, // max_size
                true, // cyclic
                SpaceId::default_space(),
                string_channel_factory,
                string_index_extractor,
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        assert_eq!(
            store.allocation_mode(),
            AllocationMode::ArrayIndex { max_size: 5, cyclic: true }
        );
    }
}
