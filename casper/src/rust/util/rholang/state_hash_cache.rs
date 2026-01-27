// See casper/src/main/scala/coop/rchain/casper/util/rholang/StateHashCache.scala

use indexmap::IndexMap;
use models::rust::block::state_hash::StateHash;
use std::sync::Mutex;

/// Simple LRU cache mapping pre-state hash to post-state hash.
/// Used to skip full replay when the mapping is already known.
pub struct StateHashCache {
    map: Mutex<IndexMap<StateHash, StateHash>>,
    max_entries: usize,
}

impl StateHashCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            map: Mutex::new(IndexMap::with_capacity(max_entries)),
            max_entries,
        }
    }

    /// Create with default capacity (128 entries).
    pub fn default_capacity() -> Self {
        Self::new(128)
    }

    /// Get the cached post-state hash for a given pre-state hash.
    pub fn get(&self, pre: &StateHash) -> Option<StateHash> {
        let mut map = self.map.lock().expect("StateHashCache lock poisoned");
        // Move to end on access (LRU behavior)
        if let Some(post) = map.shift_remove(pre) {
            map.insert(pre.clone(), post.clone());
            Some(post)
        } else {
            None
        }
    }

    /// Cache a pre-state to post-state mapping.
    pub fn put(&self, pre: StateHash, post: StateHash) {
        let mut map = self.map.lock().expect("StateHashCache lock poisoned");
        map.insert(pre, post);

        // Evict oldest entries if over capacity
        while map.len() > self.max_entries {
            map.shift_remove_index(0);
        }
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        let mut map = self.map.lock().expect("StateHashCache lock poisoned");
        map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hash(s: &str) -> StateHash {
        s.as_bytes().to_vec().into()
    }

    #[test]
    fn test_cache_and_retrieve() {
        let cache = StateHashCache::default_capacity();
        let pre = make_hash("A");
        let post = make_hash("B");

        cache.put(pre.clone(), post.clone());
        assert_eq!(cache.get(&pre), Some(post));
    }

    #[test]
    fn test_miss_for_unknown() {
        let cache = StateHashCache::default_capacity();
        let pre = make_hash("unknown");
        assert!(cache.get(&pre).is_none());
    }

    #[test]
    fn test_eviction() {
        let cache = StateHashCache::new(2);

        cache.put(make_hash("a"), make_hash("1"));
        cache.put(make_hash("b"), make_hash("2"));
        cache.put(make_hash("c"), make_hash("3"));

        // "a" should be evicted
        assert!(cache.get(&make_hash("a")).is_none());
        assert!(cache.get(&make_hash("b")).is_some());
        assert!(cache.get(&make_hash("c")).is_some());
    }

    #[test]
    fn test_clear() {
        let cache = StateHashCache::default_capacity();
        cache.put(make_hash("x"), make_hash("y"));
        cache.clear();
        assert!(cache.get(&make_hash("x")).is_none());
    }
}
