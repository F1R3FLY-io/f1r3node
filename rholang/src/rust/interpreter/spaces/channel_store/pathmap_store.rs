//! PathMapChannelStore: PathMap-based channel storage with hierarchical prefix matching.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::marker::PhantomData;

use super::{ChannelStore, DataCollection, ContinuationCollection, SpaceId, SpaceError};

/// PathMap-based channel store with hierarchical prefix matching.
///
/// Channels are represented as byte paths. Data sent on a specific path
/// is accessible to continuations listening on any prefix of that path.
/// For example, data on @[0,1,2] is accessible at @[0,1] and @[0].
///
/// This is designed for MeTTa integration where hierarchical namespacing
/// and prefix-based queries are common patterns.
///
/// # Type Parameters
///
/// - `P`: Pattern type for continuation matching
/// - `A`: Data type stored in channels
/// - `K`: Continuation type
/// - `DC`: Data collection type
/// - `CC`: Continuation collection type
#[derive(Debug)]
pub struct PathMapChannelStore<P, A, K, DC, CC>
where
    DC: Clone,
    CC: Clone,
{
    /// Data collections indexed by path (as byte vector)
    data: HashMap<Vec<u8>, DC>,

    /// Continuation collections indexed by channel pattern
    continuations: HashMap<Vec<Vec<u8>>, CC>,

    /// Join patterns: path -> list of join patterns it participates in
    joins: HashMap<Vec<u8>, Vec<Vec<Vec<u8>>>>,

    /// Counter for generating unique paths
    gensym_counter: AtomicUsize,

    /// Factory function to create new data collections
    data_factory: fn() -> DC,

    /// Factory function to create new continuation collections
    cont_factory: fn() -> CC,

    /// PhantomData for unused type parameters
    _phantom: PhantomData<(P, A, K)>,
}

impl<P, A, K, DC, CC> Clone for PathMapChannelStore<P, A, K, DC, CC>
where
    DC: Clone,
    CC: Clone,
{
    fn clone(&self) -> Self {
        PathMapChannelStore {
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

impl<P, A, K, DC, CC> PathMapChannelStore<P, A, K, DC, CC>
where
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync,
    DC: Clone + Send + Sync,
    CC: Clone + Send + Sync,
{
    /// Create a new PathMap channel store with the given factory functions.
    pub fn new(data_factory: fn() -> DC, cont_factory: fn() -> CC) -> Self {
        PathMapChannelStore {
            data: HashMap::new(),
            continuations: HashMap::new(),
            joins: HashMap::new(),
            gensym_counter: AtomicUsize::new(0),
            data_factory,
            cont_factory,
            _phantom: PhantomData,
        }
    }

    /// Get all paths that are prefixes of the given path (including the path itself).
    /// For example, for path [0, 1, 2], returns [[0], [0, 1], [0, 1, 2]].
    pub fn prefixes(path: &[u8]) -> Vec<Vec<u8>> {
        (1..=path.len())
            .map(|len| path[..len].to_vec())
            .collect()
    }

    /// Get all paths in the store that the given path is a prefix of.
    /// For example, if path is [0, 1], returns all stored paths starting with [0, 1].
    fn paths_with_prefix<'a>(&'a self, prefix: &[u8]) -> Vec<&'a Vec<u8>> {
        self.data
            .keys()
            .filter(|path| path.starts_with(prefix))
            .collect()
    }

    /// Create a normalized key for continuation lookup.
    fn normalize_channels(channels: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut sorted = channels.to_vec();
        sorted.sort();
        sorted
    }
}

impl<P, A, K, DC, CC> ChannelStore for PathMapChannelStore<P, A, K, DC, CC>
where
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync,
    DC: DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: ContinuationCollection<P, K> + Default + Clone + Send + Sync,
{
    type Channel = Vec<u8>;
    type Pattern = P;
    type Data = A;
    type Continuation = K;
    type DataColl = DC;
    type ContColl = CC;

    fn get_or_create_data_collection(&mut self, channel: &Vec<u8>) -> &mut DC {
        self.data
            .entry(channel.clone())
            .or_insert_with(|| (self.data_factory)())
    }

    fn get_data_collection(&self, channel: &Vec<u8>) -> Option<&DC> {
        self.data.get(channel)
    }

    fn get_data_collection_mut(&mut self, channel: &Vec<u8>) -> Option<&mut DC> {
        self.data.get_mut(channel)
    }

    fn get_or_create_continuation_collection(&mut self, channels: &[Vec<u8>]) -> &mut CC {
        let key = Self::normalize_channels(channels);
        self.continuations
            .entry(key)
            .or_insert_with(|| (self.cont_factory)())
    }

    fn get_continuation_collection(&self, channels: &[Vec<u8>]) -> Option<&CC> {
        let key = Self::normalize_channels(channels);
        self.continuations.get(&key)
    }

    fn get_continuation_collection_mut(&mut self, channels: &[Vec<u8>]) -> Option<&mut CC> {
        let key = Self::normalize_channels(channels);
        self.continuations.get_mut(&key)
    }

    fn all_channels(&self) -> Vec<&Vec<u8>> {
        self.data.keys().collect()
    }

    fn gensym(&mut self, _space_id: &SpaceId) -> Result<Vec<u8>, SpaceError> {
        let id = self.gensym_counter.fetch_add(1, Ordering::SeqCst);
        // Generate a unique path based on the counter
        Ok(id.to_be_bytes().to_vec())
    }

    fn get_joins(&self, channel: &Vec<u8>) -> Vec<Vec<Vec<u8>>> {
        self.joins.get(channel).cloned().unwrap_or_default()
    }

    fn put_join(&mut self, channels: Vec<Vec<u8>>) {
        for channel in &channels {
            self.joins
                .entry(channel.clone())
                .or_insert_with(Vec::new)
                .push(channels.clone());
        }
    }

    fn remove_join(&mut self, channels: &[Vec<u8>]) {
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

    fn export_data(&self) -> Vec<(Self::Channel, Self::DataColl)> {
        self.data
            .iter()
            .map(|(c, dc)| (c.clone(), dc.clone()))
            .collect()
    }

    fn export_continuations(&self) -> Vec<(Vec<Vec<u8>>, CC)> {
        self.continuations
            .iter()
            .map(|(cs, cc)| (cs.clone(), cc.clone()))
            .collect()
    }

    fn export_joins(&self) -> Vec<(Vec<u8>, Vec<Vec<Vec<u8>>>)> {
        self.joins
            .iter()
            .map(|(c, js)| (c.clone(), js.clone()))
            .collect()
    }

    fn for_each_data<F>(&self, mut f: F)
    where
        F: FnMut(&Vec<u8>, &DC),
    {
        for (c, dc) in &self.data {
            f(c, dc);
        }
    }

    fn for_each_continuation<F>(&self, mut f: F)
    where
        F: FnMut(&[Vec<u8>], &CC),
    {
        for (cs, cc) in &self.continuations {
            f(cs, cc);
        }
    }

    fn for_each_join<F>(&self, mut f: F)
    where
        F: FnMut(&Vec<u8>, &[Vec<Vec<u8>>]),
    {
        for (c, js) in &self.joins {
            f(c, js);
        }
    }

    fn gensym_counter(&self) -> usize {
        self.gensym_counter.load(Ordering::SeqCst)
    }

    fn import_data(&mut self, data: Vec<(Vec<u8>, DC)>) {
        self.data.clear();
        for (channel, dc) in data {
            self.data.insert(channel, dc);
        }
    }

    fn import_continuations(&mut self, continuations: Vec<(Vec<Vec<u8>>, CC)>) {
        self.continuations.clear();
        for (channels, cc) in continuations {
            self.continuations.insert(channels, cc);
        }
    }

    fn import_joins(&mut self, joins: Vec<(Vec<u8>, Vec<Vec<Vec<u8>>>)>) {
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

    fn channels_with_prefix(&self, prefix: &Vec<u8>) -> Vec<Vec<u8>> {
        self.data
            .keys()
            .filter(|path| path.starts_with(prefix))
            .cloned()
            .collect()
    }

    fn channel_prefixes(&self, channel: &Vec<u8>) -> Vec<Vec<u8>> {
        Self::prefixes(channel)
    }

    fn continuation_patterns_for_prefix(&self, channel: &Vec<u8>) -> Vec<(&Vec<Vec<u8>>, &CC)> {
        let channel_prefixes = Self::prefixes(channel);

        self.continuations
            .iter()
            .filter(|(pattern_channels, _cc)| {
                // Check if any channel in the pattern is a prefix of the given channel
                pattern_channels.iter().any(|pattern_ch| {
                    channel_prefixes.contains(pattern_ch)
                })
            })
            .collect()
    }
}

impl<P, A, K, DC, CC> PathMapChannelStore<P, A, K, DC, CC>
where
    P: Clone + Send + Sync,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    K: Clone + Send + Sync,
    DC: super::DataCollection<A> + Default + Clone + Send + Sync + 'static,
    CC: Clone + Send + Sync,
{
    /// Get data from a channel and all its descendants (paths that have this path as prefix).
    /// This enables hierarchical queries where a listener on @[0] can receive data from @[0,1,2].
    pub fn get_data_with_descendants(&self, prefix: &[u8]) -> Vec<(&Vec<u8>, &DC)> {
        self.paths_with_prefix(prefix)
            .into_iter()
            .filter_map(|path| self.data.get(path).map(|dc| (path, dc)))
            .collect()
    }

    /// Get data from a channel and all its ancestors (prefixes of this path).
    /// This enables hierarchical routing where data at @[0,1,2] is accessible at @[0,1].
    pub fn get_data_with_ancestors(&self, path: &[u8]) -> Vec<(Vec<u8>, &DC)> {
        Self::prefixes(path)
            .into_iter()
            .filter_map(|prefix| self.data.get(&prefix).map(|dc| (prefix, dc)))
            .collect()
    }

    // =========================================================================
    // Quantale Operations (Lattice + Path Multiplication)
    // =========================================================================
    //
    // PathMap forms a quantale: a lattice with an associative multiplication.
    // - Lattice join (∪): union of channel sets
    // - Lattice meet (∩): intersection of channel sets
    // - Subtraction (-): set difference
    // - Multiplication (*): Cartesian product path concatenation

    /// Union of channel sets (lattice join).
    ///
    /// Returns a new store containing all channels from both stores.
    /// Data from self takes precedence for channels present in both.
    pub fn union_channels(&self, other: &Self) -> Self {
        let mut result = self.clone();

        // Add all channels from other that aren't in self
        for (path, dc) in &other.data {
            result.data.entry(path.clone()).or_insert_with(|| dc.clone());
        }

        // Merge continuations
        for (key, cc) in &other.continuations {
            result.continuations.entry(key.clone()).or_insert_with(|| cc.clone());
        }

        // Merge joins
        for (path, joins) in &other.joins {
            result.joins
                .entry(path.clone())
                .or_insert_with(Vec::new)
                .extend(joins.clone());
        }

        result
    }

    /// Intersection of channel sets (lattice meet).
    ///
    /// Returns a new store containing only channels present in both stores.
    /// Data from self is used for channels in the intersection.
    pub fn intersect_channels(&self, other: &Self) -> Self {
        let mut result = PathMapChannelStore::new(self.data_factory, self.cont_factory);

        // Only include channels that are in both
        for (path, dc) in &self.data {
            if other.data.contains_key(path) {
                result.data.insert(path.clone(), dc.clone());
            }
        }

        // Only include continuation patterns where all channels are in intersection
        for (key, cc) in &self.continuations {
            let all_in_intersection = key.iter().all(|ch| other.data.contains_key(ch));
            if all_in_intersection {
                result.continuations.insert(key.clone(), cc.clone());
            }
        }

        result
    }

    /// Subtraction of channel sets (distributive lattice operation).
    ///
    /// Returns a new store containing channels in self but not in other.
    pub fn subtract_channels(&self, other: &Self) -> Self {
        let mut result = PathMapChannelStore::new(self.data_factory, self.cont_factory);

        // Include channels that are in self but not in other
        for (path, dc) in &self.data {
            if !other.data.contains_key(path) {
                result.data.insert(path.clone(), dc.clone());
            }
        }

        // Only include continuation patterns where no channel is in other
        for (key, cc) in &self.continuations {
            let none_in_other = key.iter().all(|ch| !other.data.contains_key(ch));
            if none_in_other {
                result.continuations.insert(key.clone(), cc.clone());
            }
        }

        // Preserve joins for remaining channels
        for (path, joins) in &self.joins {
            if result.data.contains_key(path) {
                result.joins.insert(path.clone(), joins.clone());
            }
        }

        result
    }

    /// Path multiplication (quantale multiplication).
    ///
    /// Returns a new store with paths formed by concatenating every path in self
    /// with every path in other: {a,b} * {c,d} = {ac, ad, bc, bd}.
    ///
    /// This operation is used for hierarchical namespace composition.
    ///
    /// # Performance Note
    ///
    /// This eagerly computes the full Cartesian product O(n₁ × n₂).
    /// For lazy evaluation, use [`concat_paths_iter`] instead.
    pub fn concat_paths(&self, other: &Self) -> Self {
        let mut result = PathMapChannelStore::new(self.data_factory, self.cont_factory);

        // Cartesian product of paths
        for (path1, dc1) in &self.data {
            for (path2, _dc2) in &other.data {
                let mut concatenated = path1.clone();
                concatenated.extend(path2);
                // Use data from self for the concatenated path
                result.data.insert(concatenated, dc1.clone());
            }
        }

        result
    }

    /// Lazy path multiplication iterator (quantale multiplication).
    ///
    /// Returns an iterator that yields concatenated paths on-demand without
    /// eagerly computing the full Cartesian product. Each iteration produces
    /// `(concatenated_path, data_collection_reference)`.
    ///
    /// # Performance
    ///
    /// - Memory: O(1) per iteration (no intermediate storage)
    /// - Time per iteration: O(|path1| + |path2|) for path cloning
    /// - Total time if fully consumed: O(n₁ × n₂ × avg_path_len)
    ///
    /// Use this when you need only a subset of the product or want to
    /// process results in a streaming fashion.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Find first matching concatenation
    /// let match = store1.concat_paths_iter(&store2)
    ///     .find(|(path, _)| path.starts_with(&prefix));
    /// ```
    pub fn concat_paths_iter<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (Vec<u8>, &'a DC)> + 'a {
        self.data.iter().flat_map(move |(path1, dc1)| {
            other.data.keys().map(move |path2| {
                let mut concatenated = path1.clone();
                concatenated.extend(path2);
                (concatenated, dc1)
            })
        })
    }

    /// Lazy path multiplication with both data collections.
    ///
    /// Similar to [`concat_paths_iter`] but yields references to both
    /// data collections, useful when you need data from both stores.
    pub fn concat_paths_iter_full<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (Vec<u8>, &'a DC, &'a DC)> + 'a {
        self.data.iter().flat_map(move |(path1, dc1)| {
            other.data.iter().map(move |(path2, dc2)| {
                let mut concatenated = path1.clone();
                concatenated.extend(path2);
                (concatenated, dc1, dc2)
            })
        })
    }

    /// Get all paths currently in the store.
    pub fn all_paths(&self) -> Vec<&Vec<u8>> {
        self.data.keys().collect()
    }

    /// Check if a path exists in the store.
    pub fn contains_path(&self, path: &[u8]) -> bool {
        self.data.contains_key(path)
    }

    /// Get the number of distinct paths in the store.
    pub fn path_count(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::interpreter::spaces::collections::{BagDataCollection, BagContinuationCollection, ContinuationCollection, DataCollection};

    #[test]
    fn test_pathmap_store_basic() {
        let mut store: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        let path = vec![0u8, 1, 2];
        let dc = store.get_or_create_data_collection(&path);
        dc.put(100).expect("put should succeed");

        assert_eq!(store.get_data_collection(&path).expect("collection should exist").len(), 1);
    }

    #[test]
    fn test_pathmap_gensym() {
        let mut store: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        let space_id = SpaceId::default_space();
        let c1 = store.gensym(&space_id).expect("gensym should succeed");
        let c2 = store.gensym(&space_id).expect("gensym should succeed");
        let c3 = store.gensym(&space_id).expect("gensym should succeed");

        // Each gensym should produce a unique path
        assert_ne!(c1, c2);
        assert_ne!(c2, c3);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_pathmap_prefixes() {
        let path = vec![0u8, 1, 2];
        let prefixes = PathMapChannelStore::<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>>::prefixes(&path);

        assert_eq!(prefixes.len(), 3);
        assert_eq!(prefixes[0], vec![0u8]);
        assert_eq!(prefixes[1], vec![0u8, 1]);
        assert_eq!(prefixes[2], vec![0u8, 1, 2]);
    }

    #[test]
    fn test_pathmap_hierarchical_queries() {
        let mut store: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        // Store data at different paths
        store.get_or_create_data_collection(&vec![0u8]).put(10).expect("put should succeed");
        store.get_or_create_data_collection(&vec![0u8, 1]).put(20).expect("put should succeed");
        store.get_or_create_data_collection(&vec![0u8, 1, 2]).put(30).expect("put should succeed");
        store.get_or_create_data_collection(&vec![1u8]).put(100).expect("put should succeed");

        // Get all descendants of [0]
        let descendants = store.get_data_with_descendants(&[0u8]);
        assert_eq!(descendants.len(), 3); // [0], [0,1], [0,1,2]

        // Get all ancestors of [0,1,2]
        let ancestors = store.get_data_with_ancestors(&[0u8, 1, 2]);
        assert_eq!(ancestors.len(), 3); // [0], [0,1], [0,1,2]

        // Get descendants of [1]
        let descendants = store.get_data_with_descendants(&[1u8]);
        assert_eq!(descendants.len(), 1); // only [1]
    }

    #[test]
    fn test_pathmap_join_patterns() {
        let mut store: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        let path1 = vec![0u8, 1];
        let path2 = vec![0u8, 2];
        let path3 = vec![1u8];

        store.put_join(vec![path1.clone(), path2.clone()]);
        store.put_join(vec![path1.clone(), path3.clone()]);

        let joins = store.get_joins(&path1);
        assert_eq!(joins.len(), 2);

        store.remove_join(&[path1.clone(), path2.clone()]);
        let joins = store.get_joins(&path1);
        assert_eq!(joins.len(), 1);
    }

    #[test]
    fn test_pathmap_union_channels() {
        let mut store1: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
        let mut store2: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        store1.get_or_create_data_collection(&vec![0u8, 1]).put(10).expect("put");
        store1.get_or_create_data_collection(&vec![0u8, 2]).put(20).expect("put");
        store2.get_or_create_data_collection(&vec![0u8, 2]).put(99).expect("put"); // overlap
        store2.get_or_create_data_collection(&vec![1u8, 0]).put(30).expect("put");

        let union = store1.union_channels(&store2);

        assert_eq!(union.path_count(), 3); // [0,1], [0,2], [1,0]
        assert!(union.contains_path(&[0u8, 1]));
        assert!(union.contains_path(&[0u8, 2]));
        assert!(union.contains_path(&[1u8, 0]));

        // [0,2] should have store1's data (10 from store1 takes precedence)
        let dc = union.get_data_collection(&vec![0u8, 2]).expect("should exist");
        assert_eq!(dc.len(), 1);
    }

    #[test]
    fn test_pathmap_intersect_channels() {
        let mut store1: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
        let mut store2: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        store1.get_or_create_data_collection(&vec![0u8, 1]).put(10).expect("put");
        store1.get_or_create_data_collection(&vec![0u8, 2]).put(20).expect("put");
        store2.get_or_create_data_collection(&vec![0u8, 2]).put(99).expect("put");
        store2.get_or_create_data_collection(&vec![1u8, 0]).put(30).expect("put");

        let intersection = store1.intersect_channels(&store2);

        assert_eq!(intersection.path_count(), 1); // only [0,2]
        assert!(!intersection.contains_path(&[0u8, 1])); // only in store1
        assert!(intersection.contains_path(&[0u8, 2]));  // in both
        assert!(!intersection.contains_path(&[1u8, 0])); // only in store2
    }

    #[test]
    fn test_pathmap_subtract_channels() {
        let mut store1: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
        let mut store2: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        store1.get_or_create_data_collection(&vec![0u8, 1]).put(10).expect("put");
        store1.get_or_create_data_collection(&vec![0u8, 2]).put(20).expect("put");
        store1.get_or_create_data_collection(&vec![0u8, 3]).put(30).expect("put");
        store2.get_or_create_data_collection(&vec![0u8, 2]).put(99).expect("put");

        let difference = store1.subtract_channels(&store2);

        assert_eq!(difference.path_count(), 2); // [0,1] and [0,3]
        assert!(difference.contains_path(&[0u8, 1]));
        assert!(!difference.contains_path(&[0u8, 2])); // subtracted out
        assert!(difference.contains_path(&[0u8, 3]));
    }

    #[test]
    fn test_pathmap_concat_paths() {
        let mut store1: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
        let mut store2: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        store1.get_or_create_data_collection(&vec![0u8]).put(1).expect("put");
        store1.get_or_create_data_collection(&vec![1u8]).put(2).expect("put");
        store2.get_or_create_data_collection(&vec![2u8]).put(3).expect("put");
        store2.get_or_create_data_collection(&vec![3u8]).put(4).expect("put");

        let product = store1.concat_paths(&store2);

        // {[0], [1]} * {[2], [3]} = {[0,2], [0,3], [1,2], [1,3]}
        assert_eq!(product.path_count(), 4);
        assert!(product.contains_path(&[0u8, 2]));
        assert!(product.contains_path(&[0u8, 3]));
        assert!(product.contains_path(&[1u8, 2]));
        assert!(product.contains_path(&[1u8, 3]));
    }

    #[test]
    fn test_pathmap_concat_paths_iter_lazy() {
        // Test that lazy iterator produces same results as eager version
        let mut store1: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
        let mut store2: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        store1.get_or_create_data_collection(&vec![0u8]).put(1).expect("put");
        store1.get_or_create_data_collection(&vec![1u8]).put(2).expect("put");
        store2.get_or_create_data_collection(&vec![2u8]).put(3).expect("put");
        store2.get_or_create_data_collection(&vec![3u8]).put(4).expect("put");

        // Collect lazy iterator results
        let lazy_results: Vec<_> = store1.concat_paths_iter(&store2)
            .map(|(path, _dc)| path)
            .collect();

        // Should produce same 4 paths as eager version
        assert_eq!(lazy_results.len(), 4);
        assert!(lazy_results.contains(&vec![0u8, 2]));
        assert!(lazy_results.contains(&vec![0u8, 3]));
        assert!(lazy_results.contains(&vec![1u8, 2]));
        assert!(lazy_results.contains(&vec![1u8, 3]));
    }

    #[test]
    fn test_pathmap_concat_paths_iter_early_termination() {
        // Test that lazy iterator allows early termination without computing all
        let mut store1: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
        let mut store2: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        // Create larger stores
        for i in 0..10u8 {
            store1.get_or_create_data_collection(&vec![i]).put(i as i32).expect("put");
            store2.get_or_create_data_collection(&vec![i + 10]).put(i as i32).expect("put");
        }

        // Find first path starting with [5] (should not compute all 100 combinations)
        let found = store1.concat_paths_iter(&store2)
            .find(|(path, _)| path.starts_with(&[5u8]));

        assert!(found.is_some());
        assert!(found.unwrap().0.starts_with(&[5u8]));
    }

    #[test]
    fn test_pathmap_quantale_associativity() {
        // Test that path multiplication is associative: (A * B) * C == A * (B * C)
        let mut a: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
        let mut b: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);
        let mut c: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        a.get_or_create_data_collection(&vec![1u8]).put(1).expect("put");
        b.get_or_create_data_collection(&vec![2u8]).put(2).expect("put");
        c.get_or_create_data_collection(&vec![3u8]).put(3).expect("put");

        let ab_c = a.concat_paths(&b).concat_paths(&c);
        let a_bc = a.concat_paths(&b.concat_paths(&c));

        // Both should produce the same paths
        let mut paths1: Vec<_> = ab_c.all_paths().into_iter().cloned().collect();
        let mut paths2: Vec<_> = a_bc.all_paths().into_iter().cloned().collect();
        paths1.sort();
        paths2.sort();
        assert_eq!(paths1, paths2);
        assert!(ab_c.contains_path(&[1u8, 2, 3]));
    }

    // ==========================================================================
    // PathMap Prefix Semantics Tests
    // ==========================================================================

    #[test]
    fn test_pathmap_supports_prefix_semantics() {
        let store: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        assert!(store.supports_prefix_semantics());
    }

    #[test]
    fn test_pathmap_channels_with_prefix() {
        let mut store: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        store.get_or_create_data_collection(&vec![0u8, 1]);
        store.get_or_create_data_collection(&vec![0u8, 1, 2]);
        store.get_or_create_data_collection(&vec![0u8, 1, 3]);
        store.get_or_create_data_collection(&vec![0u8, 2]);
        store.get_or_create_data_collection(&vec![1u8]);

        let mut descendants = store.channels_with_prefix(&vec![0u8, 1]);
        descendants.sort();

        assert_eq!(descendants.len(), 3);
        assert_eq!(descendants[0], vec![0u8, 1]);
        assert_eq!(descendants[1], vec![0u8, 1, 2]);
        assert_eq!(descendants[2], vec![0u8, 1, 3]);
    }

    #[test]
    fn test_pathmap_channel_prefixes() {
        let store: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        let prefixes = store.channel_prefixes(&vec![0u8, 1, 2]);

        assert_eq!(prefixes.len(), 3);
        assert_eq!(prefixes[0], vec![0u8]);
        assert_eq!(prefixes[1], vec![0u8, 1]);
        assert_eq!(prefixes[2], vec![0u8, 1, 2]);
    }

    #[test]
    fn test_pathmap_continuation_patterns_for_prefix() {
        let mut store: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        // Add continuations at various paths
        // Continuation at @[0, 1] - a prefix of @[0, 1, 2]
        store.get_or_create_continuation_collection(&[vec![0u8, 1]])
            .put(vec!["pattern1".to_string()], "cont1".to_string(), false);

        // Continuation at @[0] - a shorter prefix of @[0, 1, 2]
        store.get_or_create_continuation_collection(&[vec![0u8]])
            .put(vec!["pattern2".to_string()], "cont2".to_string(), false);

        // Continuation at @[1] - NOT a prefix of @[0, 1, 2]
        store.get_or_create_continuation_collection(&[vec![1u8]])
            .put(vec!["pattern3".to_string()], "cont3".to_string(), false);

        // Find continuations that are at prefixes of @[0, 1, 2]
        let matches = store.continuation_patterns_for_prefix(&vec![0u8, 1, 2]);

        assert_eq!(matches.len(), 2);

        // Verify we found the right continuations
        let patterns: Vec<_> = matches.iter().map(|(p, _)| (*p).clone()).collect();
        assert!(patterns.contains(&&vec![vec![0u8, 1]]));
        assert!(patterns.contains(&&vec![vec![0u8]]));
    }

    #[test]
    fn test_pathmap_prefix_semantics_spec_example() {
        // From spec lines 159-192:
        // @[0, 1, 2]!({|"hi"|}) | @[0, 1, 2]!({|"hello"|}) | @[0, 1, 3]!({|"there"|})
        // = @[0, 1]!({|[2, "hi"], [2, "hello"], [3, "there"]|})

        let mut store: PathMapChannelStore<String, String, String, BagDataCollection<String>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(BagDataCollection::new, BagContinuationCollection::new);

        // Store data at specific paths
        store.get_or_create_data_collection(&vec![0u8, 1, 2]).put("hi".to_string()).expect("put");
        store.get_or_create_data_collection(&vec![0u8, 1, 2]).put("hello".to_string()).expect("put");
        store.get_or_create_data_collection(&vec![0u8, 1, 3]).put("there".to_string()).expect("put");

        // Add a continuation waiting at the prefix @[0, 1]
        store.get_or_create_continuation_collection(&[vec![0u8, 1]])
            .put(vec!["*".to_string()], "consumer".to_string(), false);

        // The continuation at @[0, 1] should be found when producing on @[0, 1, 2] or @[0, 1, 3]
        let matches_for_012 = store.continuation_patterns_for_prefix(&vec![0u8, 1, 2]);
        assert_eq!(matches_for_012.len(), 1);

        let matches_for_013 = store.continuation_patterns_for_prefix(&vec![0u8, 1, 3]);
        assert_eq!(matches_for_013.len(), 1);

        // Get all descendants of @[0, 1] - should include data from @[0, 1, 2] and @[0, 1, 3]
        let descendants = store.get_data_with_descendants(&[0u8, 1]);
        assert_eq!(descendants.len(), 2); // @[0, 1, 2] and @[0, 1, 3]
    }

    #[test]
    fn test_pathmap_allocation_mode_returns_random() {
        use crate::rust::interpreter::spaces::types::AllocationMode;

        let store: PathMapChannelStore<String, i32, String, BagDataCollection<i32>, BagContinuationCollection<String, String>> =
            PathMapChannelStore::new(
                BagDataCollection::new,
                BagContinuationCollection::new,
            );

        assert_eq!(store.allocation_mode(), AllocationMode::Random);
    }
}
