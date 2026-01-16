//! RholangPathMapStore: PathMap-aware channel store for Rholang with Par channels.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::marker::PhantomData;

use models::rhoapi::Par;

use super::{ChannelStore, DataCollection, ContinuationCollection, SpaceId, SpaceError};
use super::super::types::{par_to_path, path_to_par, get_path_suffix, path_element_boundaries};

/// PathMap-aware channel store for Rholang that uses Par as channel type.
///
/// This store enables PathMap prefix semantics with Rholang's Par-based channels.
/// It maintains an internal mapping from path bytes to Par channels, allowing
/// prefix-based queries on channels like `@[0, 1, 2]`.
///
/// # Type Parameters
///
/// - `P`: Pattern type for continuation matching
/// - `A`: Data type stored in channels
/// - `K`: Continuation type
/// - `DC`: Data collection type
/// - `CC`: Continuation collection type
///
/// # Prefix Semantics
///
/// When a Par channel represents a path (EList of integers 0-255):
/// - `produce(@[0,1,2], data)` stores data at path `[0, 1, 2]`
/// - `consume(@[0,1], pattern)` can receive data from `@[0,1,2]` with suffix `[2]`
///
/// For non-path channels, the store falls back to exact matching (like HashMap).
///
/// # Example
/// ```ignore
/// let store = RholangPathMapStore::new(BagDataCollection::new, BagContinuationCollection::new);
///
/// // Path channel: prefix semantics enabled
/// let path_channel = path_to_par(&[0, 1, 2]);
///
/// // Non-path channel: exact matching only
/// let string_channel = create_string_par("hello");
/// ```
#[derive(Debug)]
pub struct RholangPathMapStore<P, A, K, DC, CC>
where
    DC: Clone,
    CC: Clone,
{
    /// Data collections indexed by Par channel
    data: HashMap<Par, DC>,

    /// Continuation collections indexed by channel pattern
    continuations: HashMap<Vec<Par>, CC>,

    /// Join patterns: channel -> list of join patterns it participates in
    joins: HashMap<Par, Vec<Vec<Par>>>,

    /// Path index: maps path bytes to their Par representation for prefix queries
    /// Only populated for channels that are valid paths (EList of ints)
    path_index: HashMap<Vec<u8>, Par>,

    /// Counter for generating unique channel names
    gensym_counter: AtomicUsize,

    /// Factory function to create new data collections
    data_factory: fn() -> DC,

    /// Factory function to create new continuation collections
    cont_factory: fn() -> CC,

    /// PhantomData for unused type parameters
    _phantom: PhantomData<(P, A, K)>,
}

impl<P, A, K, DC, CC> Clone for RholangPathMapStore<P, A, K, DC, CC>
where
    DC: Clone,
    CC: Clone,
{
    fn clone(&self) -> Self {
        RholangPathMapStore {
            data: self.data.clone(),
            continuations: self.continuations.clone(),
            joins: self.joins.clone(),
            path_index: self.path_index.clone(),
            gensym_counter: AtomicUsize::new(self.gensym_counter.load(Ordering::SeqCst)),
            data_factory: self.data_factory,
            cont_factory: self.cont_factory,
            _phantom: PhantomData,
        }
    }
}

impl<P, A, K, DC, CC> RholangPathMapStore<P, A, K, DC, CC>
where
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync,
    DC: Clone + Send + Sync,
    CC: Clone + Send + Sync,
{
    /// Create a new Rholang PathMap store with the given factory functions.
    pub fn new(data_factory: fn() -> DC, cont_factory: fn() -> CC) -> Self {
        RholangPathMapStore {
            data: HashMap::new(),
            continuations: HashMap::new(),
            joins: HashMap::new(),
            path_index: HashMap::new(),
            gensym_counter: AtomicUsize::new(0),
            data_factory,
            cont_factory,
            _phantom: PhantomData,
        }
    }

    /// Update the path index when adding a channel.
    fn index_channel(&mut self, channel: &Par) {
        if let Some(path) = par_to_path(channel) {
            self.path_index.insert(path, channel.clone());
        }
    }

    /// Remove from path index when removing a channel.
    #[allow(dead_code)]
    fn unindex_channel(&mut self, channel: &Par) {
        if let Some(path) = par_to_path(channel) {
            self.path_index.remove(&path);
        }
    }

    /// Get all Par channels whose paths have the given prefix.
    ///
    /// Returns channels where `par_to_path(channel)` starts with `prefix_path`.
    pub fn channels_with_path_prefix(&self, prefix_path: &[u8]) -> Vec<Par> {
        self.path_index
            .iter()
            .filter(|(path, _)| path.starts_with(prefix_path))
            .map(|(_, par)| par.clone())
            .collect()
    }
}

impl<P, A, K, DC, CC> ChannelStore for RholangPathMapStore<P, A, K, DC, CC>
where
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync,
    DC: DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: ContinuationCollection<P, K> + Default + Clone + Send + Sync,
{
    type Channel = Par;
    type Pattern = P;
    type Data = A;
    type Continuation = K;
    type DataColl = DC;
    type ContColl = CC;

    fn get_or_create_data_collection(&mut self, channel: &Par) -> &mut DC {
        // Index the channel if it's a path
        self.index_channel(channel);

        self.data
            .entry(channel.clone())
            .or_insert_with(|| (self.data_factory)())
    }

    fn get_data_collection(&self, channel: &Par) -> Option<&DC> {
        self.data.get(channel)
    }

    fn get_data_collection_mut(&mut self, channel: &Par) -> Option<&mut DC> {
        self.data.get_mut(channel)
    }

    fn get_or_create_continuation_collection(&mut self, channels: &[Par]) -> &mut CC {
        let mut sorted = channels.to_vec();
        sorted.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        self.continuations
            .entry(sorted)
            .or_insert_with(|| (self.cont_factory)())
    }

    fn get_continuation_collection(&self, channels: &[Par]) -> Option<&CC> {
        let mut sorted = channels.to_vec();
        sorted.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        self.continuations.get(&sorted)
    }

    fn get_continuation_collection_mut(&mut self, channels: &[Par]) -> Option<&mut CC> {
        let mut sorted = channels.to_vec();
        sorted.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        self.continuations.get_mut(&sorted)
    }

    fn all_channels(&self) -> Vec<&Par> {
        self.data.keys().collect()
    }

    fn gensym(&mut self, _space_id: &SpaceId) -> Result<Par, SpaceError> {
        let id = self.gensym_counter.fetch_add(1, Ordering::SeqCst);
        // Generate a unique path based on the counter
        let path = id.to_be_bytes().to_vec();
        Ok(path_to_par(&path))
    }

    fn get_joins(&self, channel: &Par) -> Vec<Vec<Par>> {
        self.joins.get(channel).cloned().unwrap_or_default()
    }

    fn put_join(&mut self, channels: Vec<Par>) {
        for channel in &channels {
            self.joins
                .entry(channel.clone())
                .or_insert_with(Vec::new)
                .push(channels.clone());
        }
    }

    fn remove_join(&mut self, channels: &[Par]) {
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
        self.path_index.clear();
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty() && self.continuations.is_empty()
    }

    fn export_data(&self) -> Vec<(Par, DC)> {
        self.data
            .iter()
            .map(|(c, dc)| (c.clone(), dc.clone()))
            .collect()
    }

    fn export_continuations(&self) -> Vec<(Vec<Par>, CC)> {
        self.continuations
            .iter()
            .map(|(cs, cc)| (cs.clone(), cc.clone()))
            .collect()
    }

    fn export_joins(&self) -> Vec<(Par, Vec<Vec<Par>>)> {
        self.joins
            .iter()
            .map(|(c, js)| (c.clone(), js.clone()))
            .collect()
    }

    fn for_each_data<F>(&self, mut f: F)
    where
        F: FnMut(&Par, &DC),
    {
        for (c, dc) in &self.data {
            f(c, dc);
        }
    }

    fn for_each_continuation<F>(&self, mut f: F)
    where
        F: FnMut(&[Par], &CC),
    {
        for (cs, cc) in &self.continuations {
            f(cs, cc);
        }
    }

    fn for_each_join<F>(&self, mut f: F)
    where
        F: FnMut(&Par, &[Vec<Par>]),
    {
        for (c, js) in &self.joins {
            f(c, js);
        }
    }

    fn gensym_counter(&self) -> usize {
        self.gensym_counter.load(Ordering::SeqCst)
    }

    fn import_data(&mut self, data: Vec<(Par, DC)>) {
        self.data.clear();
        self.path_index.clear();
        for (channel, dc) in data {
            self.index_channel(&channel);
            self.data.insert(channel, dc);
        }
    }

    fn import_continuations(&mut self, continuations: Vec<(Vec<Par>, CC)>) {
        self.continuations.clear();
        for (channels, cc) in continuations {
            self.continuations.insert(channels, cc);
        }
    }

    fn import_joins(&mut self, joins: Vec<(Par, Vec<Vec<Par>>)>) {
        self.joins.clear();
        for (channel, join_patterns) in joins {
            self.joins.insert(channel, join_patterns);
        }
    }

    fn set_gensym_counter(&mut self, counter: usize) {
        self.gensym_counter.store(counter, Ordering::SeqCst);
    }

    // =========================================================================
    // Prefix Semantics Implementation
    // =========================================================================

    fn supports_prefix_semantics(&self) -> bool {
        true
    }

    fn channels_with_prefix(&self, prefix: &Par) -> Vec<Par> {
        // Convert the prefix Par to a path
        if let Some(prefix_path) = par_to_path(prefix) {
            self.channels_with_path_prefix(&prefix_path)
        } else {
            // For non-path channels, only exact match
            if self.data.contains_key(prefix) {
                vec![prefix.clone()]
            } else {
                vec![]
            }
        }
    }

    fn channel_prefixes(&self, channel: &Par) -> Vec<Par> {
        // Convert the channel Par to a path
        if let Some(path) = par_to_path(channel) {
            // Get element boundaries in the tagged path encoding
            let boundaries = path_element_boundaries(&path);

            // Generate prefix Pars at each element boundary
            // For path @[0, 1, 2] (27 bytes), boundaries = [9, 18, 27]
            // This generates: @[0] (bytes 0..9), @[0,1] (bytes 0..18), @[0,1,2] (bytes 0..27)
            boundaries
                .iter()
                .map(|&boundary| path_to_par(&path[..boundary]))
                .collect()
        } else {
            // For non-path channels, only return the channel itself
            vec![channel.clone()]
        }
    }

    fn compute_suffix_key(&self, prefix: &Par, descendant: &Par) -> Option<Vec<u8>> {
        // Convert both channels to paths
        let prefix_path = par_to_path(prefix)?;
        let descendant_path = par_to_path(descendant)?;

        // Get the suffix - returns None if not a prefix relationship
        get_path_suffix(&prefix_path, &descendant_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::interpreter::spaces::collections::{BagDataCollection, BagContinuationCollection};

    #[test]
    fn test_rholang_pathmap_store_basic() {
        let mut store: RholangPathMapStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            RholangPathMapStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        let channel = path_to_par(&[0, 1, 2]);
        let dc = store.get_or_create_data_collection(&channel);
        dc.put(100).expect("put should succeed");

        assert_eq!(store.get_data_collection(&channel).expect("collection should exist").len(), 1);
    }

    #[test]
    fn test_rholang_pathmap_gensym() {
        let mut store: RholangPathMapStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            RholangPathMapStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        let space_id = SpaceId::default_space();
        let c1 = store.gensym(&space_id).expect("gensym should succeed");
        let c2 = store.gensym(&space_id).expect("gensym should succeed");
        let c3 = store.gensym(&space_id).expect("gensym should succeed");

        // Each gensym should produce a unique channel
        assert_ne!(c1, c2);
        assert_ne!(c2, c3);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_rholang_pathmap_supports_prefix_semantics() {
        let store: RholangPathMapStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            RholangPathMapStore::new(BagDataCollection::new, BagContinuationCollection::new);

        assert!(store.supports_prefix_semantics());
    }

    #[test]
    fn test_rholang_pathmap_path_indexing() {
        let mut store: RholangPathMapStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            RholangPathMapStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        // Create channels
        let channel_0_1 = path_to_par(&[0, 1]);
        let channel_0_1_2 = path_to_par(&[0, 1, 2]);
        let channel_0_1_3 = path_to_par(&[0, 1, 3]);
        let channel_0_2 = path_to_par(&[0, 2]);

        store.get_or_create_data_collection(&channel_0_1);
        store.get_or_create_data_collection(&channel_0_1_2);
        store.get_or_create_data_collection(&channel_0_1_3);
        store.get_or_create_data_collection(&channel_0_2);

        // Get channels with prefix @[0, 1]
        let prefix = path_to_par(&[0, 1]);
        let prefix_path = par_to_path(&prefix).expect("should be valid path");
        let mut descendants = store.channels_with_path_prefix(&prefix_path);
        descendants.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));

        // Should have @[0,1], @[0,1,2], @[0,1,3]
        assert_eq!(descendants.len(), 3);
    }
}
