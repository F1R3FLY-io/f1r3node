//! VectorChannelStore: Dynamic indexed channel storage with unbounded growth.

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use super::{ChannelStore, DataCollection, ContinuationCollection, SpaceId, SpaceError, AllocationMode};

/// Vector-based channel store with unbounded growth.
///
/// Similar to Array but grows automatically without size limits.
/// Generic over channel type C for integration with Rholang spaces.
///
/// # Type Parameters
///
/// - `C`: Channel type (must support hashing and equality)
/// - `P`: Pattern type for continuation matching
/// - `A`: Data type stored in channels
/// - `K`: Continuation type
/// - `DC`: Data collection type
/// - `CC`: Continuation collection type
///
/// # Design Document Alignment
///
/// Per the Reifying RSpaces design document:
/// - Vector channels are allocated sequentially via gensym (indices 0, 1, 2, ...)
/// - Indices are wrapped in Unforgeable{} so clients can't forge them
/// - Unlike Array, Vector grows unbounded (until OOM)
#[derive(Debug)]
pub struct VectorChannelStore<C, P, A, K, DC, CC>
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
    index_to_channel: Vec<C>,

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

impl<C, P, A, K, DC, CC> Clone for VectorChannelStore<C, P, A, K, DC, CC>
where
    C: Clone + Eq + Hash,
    DC: Clone,
    CC: Clone,
{
    fn clone(&self) -> Self {
        VectorChannelStore {
            data: self.data.clone(),
            continuations: self.continuations.clone(),
            joins: self.joins.clone(),
            channel_to_index: self.channel_to_index.clone(),
            index_to_channel: self.index_to_channel.clone(),
            space_id: self.space_id.clone(),
            channel_factory: self.channel_factory,
            index_extractor: self.index_extractor,
            data_factory: self.data_factory,
            cont_factory: self.cont_factory,
            _phantom: PhantomData,
        }
    }
}

impl<C, P, A, K, DC, CC> VectorChannelStore<C, P, A, K, DC, CC>
where
    C: Clone + Eq + Hash + Send + Sync,
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync,
    DC: Clone + Send + Sync,
    CC: Clone + Send + Sync,
{
    /// Create a new vector channel store with the given channel factory.
    ///
    /// # Arguments
    ///
    /// * `space_id` - Space ID for channel creation (passed to channel_factory)
    /// * `channel_factory` - Function to create a channel from (space_id, index)
    /// * `index_extractor` - Function to extract index from a channel (reverse of channel_factory)
    /// * `data_factory` - Function to create empty data collections
    /// * `cont_factory` - Function to create empty continuation collections
    pub fn new(
        space_id: SpaceId,
        channel_factory: fn(&SpaceId, usize) -> C,
        index_extractor: fn(&SpaceId, &C) -> Option<usize>,
        data_factory: fn() -> DC,
        cont_factory: fn() -> CC,
    ) -> Self {
        VectorChannelStore {
            data: Vec::new(),
            continuations: HashMap::new(),
            joins: HashMap::new(),
            channel_to_index: HashMap::new(),
            index_to_channel: Vec::new(),
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

        // Grow data vector if needed
        while self.data.len() <= index {
            self.data.push(None);
        }

        // Register the channel
        self.channel_to_index.insert(channel.clone(), index);
        while self.index_to_channel.len() <= index {
            // Create a placeholder channel - will be overwritten
            self.index_to_channel.push(channel.clone());
        }
        self.index_to_channel[index] = channel.clone();

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

impl<C, P, A, K, DC, CC> ChannelStore for VectorChannelStore<C, P, A, K, DC, CC>
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
            .expect("Channel does not match this Vector space pattern");

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
        self.index_to_channel.iter().collect()
    }

    fn gensym(&mut self, _space_id: &SpaceId) -> Result<C, SpaceError> {
        let index = self.data.len();

        // Create new channel from index using the factory
        let channel = (self.channel_factory)(&self.space_id, index);

        // Store bidirectional mapping
        self.channel_to_index.insert(channel.clone(), index);
        self.index_to_channel.push(channel.clone());

        // Grow data vector
        self.data.push(None);

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
        self.data.clear();
        self.continuations.clear();
        self.joins.clear();
        self.channel_to_index.clear();
        self.index_to_channel.clear();
    }

    fn is_empty(&self) -> bool {
        self.data.iter().all(|d| d.is_none()) && self.continuations.is_empty()
    }

    fn export_data(&self) -> Vec<(C, DC)> {
        self.index_to_channel
            .iter()
            .enumerate()
            .filter_map(|(i, channel)| {
                self.data.get(i)?.as_ref().map(|dc| (channel.clone(), dc.clone()))
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
        for (i, channel) in self.index_to_channel.iter().enumerate() {
            if let Some(Some(dc)) = self.data.get(i) {
                f(channel, dc);
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
        self.data.len()
    }

    fn import_data(&mut self, data: Vec<(C, DC)>) {
        // Clear existing data and mappings
        self.data.clear();
        self.channel_to_index.clear();
        self.index_to_channel.clear();

        // Import new data
        for (channel, dc) in data {
            let index = self.data.len();
            self.data.push(Some(dc));
            self.channel_to_index.insert(channel.clone(), index);
            self.index_to_channel.push(channel);
        }
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
        // For Vector, we resize the data vector to match the counter
        while self.data.len() < counter {
            self.data.push(None);
        }
    }

    fn allocation_mode(&self) -> AllocationMode {
        AllocationMode::VectorIndex
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::interpreter::spaces::collections::{BagDataCollection, BagContinuationCollection};

    /// Factory function for creating usize channels from index
    fn usize_channel_factory(_space_id: &SpaceId, index: usize) -> usize {
        index
    }

    /// Extractor function for getting index from usize channel
    fn usize_index_extractor(_space_id: &SpaceId, channel: &usize) -> Option<usize> {
        Some(*channel)
    }

    #[test]
    fn test_vector_store_growth() {
        let mut store: VectorChannelStore<usize, String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            VectorChannelStore::new(
                SpaceId::default_space(),
                usize_channel_factory,
                usize_index_extractor,
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        let space_id = SpaceId::default_space();
        for i in 0..100 {
            let index = store.gensym(&space_id).unwrap();
            assert_eq!(index, i);
        }
        assert_eq!(store.data.len(), 100);
    }

    #[test]
    fn test_vector_allocation_mode_returns_vector_index() {
        let store: VectorChannelStore<usize, String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            VectorChannelStore::new(
                SpaceId::default_space(),
                usize_channel_factory,
                usize_index_extractor,
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        assert_eq!(store.allocation_mode(), AllocationMode::VectorIndex);
    }
}
