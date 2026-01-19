//! HashMapChannelStore: HashMap-based channel storage with O(1) lookup.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::marker::PhantomData;

use super::{ChannelStore, DataCollection, ContinuationCollection, SpaceId, SpaceError};

/// HashMap-based channel store - O(1) lookup by channel.
///
/// This is the most common storage type, providing fast random access
/// to any channel by its key.
///
/// # Type Parameters
///
/// - `C`: Channel type
/// - `P`: Pattern type for continuation matching
/// - `A`: Data type stored in channels
/// - `K`: Continuation type
/// - `DC`: Data collection type
/// - `CC`: Continuation collection type
#[derive(Debug)]
pub struct HashMapChannelStore<C, P, A, K, DC, CC>
where
    C: Clone + Eq + Hash,
{
    /// Data collections indexed by channel
    data: HashMap<C, DC>,

    /// Continuation collections indexed by channel pattern (sorted channel vec)
    continuations: HashMap<Vec<C>, CC>,

    /// Join patterns: channel -> list of join patterns it participates in
    joins: HashMap<C, Vec<Vec<C>>>,

    /// Counter for generating unique channel names
    gensym_counter: AtomicUsize,

    /// Factory function to create new data collections
    data_factory: fn() -> DC,

    /// Factory function to create new continuation collections
    cont_factory: fn() -> CC,

    /// PhantomData to track P, A, K
    _phantom: PhantomData<(P, A, K)>,
}

impl<C, P, A, K, DC, CC> Clone for HashMapChannelStore<C, P, A, K, DC, CC>
where
    C: Clone + Eq + Hash,
    DC: Clone,
    CC: Clone,
{
    fn clone(&self) -> Self {
        HashMapChannelStore {
            data: self.data.clone(),
            continuations: self.continuations.clone(),
            joins: self.joins.clone(),
            gensym_counter: AtomicUsize::new(self.gensym_counter.load(Ordering::SeqCst)),
            data_factory: self.data_factory,
            cont_factory: self.cont_factory,
            _phantom: PhantomData,
        }
    }
}

impl<C, P, A, K, DC, CC> HashMapChannelStore<C, P, A, K, DC, CC>
where
    C: Clone + Eq + Hash + Send + Sync,
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync,
    DC: Clone + Send + Sync,
    CC: Clone + Send + Sync,
{
    /// Create a new HashMap channel store with the given factory functions.
    pub fn new(data_factory: fn() -> DC, cont_factory: fn() -> CC) -> Self {
        HashMapChannelStore {
            data: HashMap::new(),
            continuations: HashMap::new(),
            joins: HashMap::new(),
            gensym_counter: AtomicUsize::new(0),
            data_factory,
            cont_factory,
            _phantom: PhantomData,
        }
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

impl<C, P, A, K, DC, CC> ChannelStore for HashMapChannelStore<C, P, A, K, DC, CC>
where
    C: Clone + Eq + Hash + Ord + Send + Sync + From<usize>,
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
        self.data
            .entry(channel.clone())
            .or_insert_with(|| (self.data_factory)())
    }

    fn get_data_collection(&self, channel: &C) -> Option<&DC> {
        self.data.get(channel)
    }

    fn get_data_collection_mut(&mut self, channel: &C) -> Option<&mut DC> {
        self.data.get_mut(channel)
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
        self.data.keys().collect()
    }

    fn gensym(&mut self, _space_id: &SpaceId) -> Result<C, SpaceError> {
        let id = self.gensym_counter.fetch_add(1, Ordering::SeqCst);
        Ok(C::from(id))
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
        Self {
            data: self.data.clone(),
            continuations: self.continuations.clone(),
            joins: self.joins.clone(),
            gensym_counter: AtomicUsize::new(self.gensym_counter.load(Ordering::SeqCst)),
            data_factory: self.data_factory,
            cont_factory: self.cont_factory,
            _phantom: PhantomData,
        }
    }

    fn clear(&mut self) {
        self.data.clear();
        self.continuations.clear();
        self.joins.clear();
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty() && self.continuations.is_empty()
    }

    fn export_data(&self) -> Vec<(C, DC)> {
        self.data
            .iter()
            .map(|(c, dc)| (c.clone(), dc.clone()))
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
        for (c, dc) in &self.data {
            f(c, dc);
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
        self.gensym_counter.load(Ordering::SeqCst)
    }

    fn import_data(&mut self, data: Vec<(C, DC)>) {
        self.data.clear();
        for (channel, dc) in data {
            self.data.insert(channel, dc);
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
        self.gensym_counter.store(counter, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::interpreter::spaces::collections::{BagDataCollection, BagContinuationCollection};

    #[test]
    fn test_hashmap_store_basic() {
        let mut store: HashMapChannelStore<usize, String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            HashMapChannelStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        let channel = 42usize;
        let dc = store.get_or_create_data_collection(&channel);
        dc.put(100).unwrap();

        assert_eq!(store.get_data_collection(&channel).unwrap().len(), 1);
    }

    #[test]
    fn test_hashmap_gensym() {
        let mut store: HashMapChannelStore<usize, String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            HashMapChannelStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        let space_id = SpaceId::default_space();
        let c1 = store.gensym(&space_id).unwrap();
        let c2 = store.gensym(&space_id).unwrap();
        let c3 = store.gensym(&space_id).unwrap();

        assert_eq!(c1, 0);
        assert_eq!(c2, 1);
        assert_eq!(c3, 2);
    }

    #[test]
    fn test_join_patterns() {
        let mut store: HashMapChannelStore<usize, String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            HashMapChannelStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        store.put_join(vec![1, 2, 3]);
        store.put_join(vec![1, 4]);

        let joins = store.get_joins(&1);
        assert_eq!(joins.len(), 2);

        store.remove_join(&[1, 2, 3]);
        let joins = store.get_joins(&1);
        assert_eq!(joins.len(), 1);
    }

    #[test]
    fn test_hashmap_does_not_support_prefix_semantics() {
        let store: HashMapChannelStore<usize, String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            HashMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        assert!(!store.supports_prefix_semantics());
    }

    #[test]
    fn test_hashmap_allocation_mode_returns_random() {
        use crate::rust::interpreter::spaces::types::AllocationMode;

        let store: HashMapChannelStore<usize, String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            HashMapChannelStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        assert_eq!(store.allocation_mode(), AllocationMode::Random);
    }
}
