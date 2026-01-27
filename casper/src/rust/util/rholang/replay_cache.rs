// See casper/src/main/scala/coop/rchain/casper/util/rholang/ReplayCache.scala

use indexmap::IndexMap;
use models::rust::block::state_hash::StateHash;
use models::rust::casper::protocol::casper_message::Event;
use std::sync::Mutex;

/// Cache key: parent state + block identity (sender, seqNum).
/// Using (sender, seqNum) avoids ambiguity across forks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReplayCacheKey {
    pub parent_state: StateHash,
    pub sender_pk: Vec<u8>,
    pub seq_num: i64,
}

impl ReplayCacheKey {
    pub fn new(parent_state: StateHash, sender_pk: Vec<u8>, seq_num: i64) -> Self {
        Self {
            parent_state,
            sender_pk,
            seq_num,
        }
    }
}

/// Cached replay result containing event log and post-state hash.
#[derive(Clone, Debug)]
pub struct ReplayCacheEntry {
    pub event_log: Vec<Event>,
    pub post_state: StateHash,
}

impl ReplayCacheEntry {
    pub fn new(event_log: Vec<Event>, post_state: StateHash) -> Self {
        Self { event_log, post_state }
    }
}

/// Trait for replay caching operations.
pub trait ReplayCache: Send + Sync {
    fn get(&self, key: &ReplayCacheKey) -> Option<ReplayCacheEntry>;
    fn put(&self, key: ReplayCacheKey, entry: ReplayCacheEntry);
    fn clear(&self);
}

/// Simple in-memory LRU replay cache (thread-safe).
pub struct InMemoryReplayCache {
    map: Mutex<IndexMap<ReplayCacheKey, ReplayCacheEntry>>,
    max_entries: usize,
}

impl InMemoryReplayCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            map: Mutex::new(IndexMap::with_capacity(max_entries)),
            max_entries,
        }
    }

    /// Create with default capacity (1024 entries).
    pub fn default_capacity() -> Self {
        Self::new(1024)
    }
}

impl ReplayCache for InMemoryReplayCache {
    fn get(&self, key: &ReplayCacheKey) -> Option<ReplayCacheEntry> {
        let mut map = self.map.lock().expect("ReplayCache lock poisoned");
        // Move to end on access (LRU behavior)
        if let Some(entry) = map.shift_remove(key) {
            map.insert(key.clone(), entry.clone());
            Some(entry)
        } else {
            None
        }
    }

    fn put(&self, key: ReplayCacheKey, entry: ReplayCacheEntry) {
        let mut map = self.map.lock().expect("ReplayCache lock poisoned");
        map.insert(key, entry);

        // Evict oldest entries if over capacity
        while map.len() > self.max_entries {
            map.shift_remove_index(0);
        }
    }

    fn clear(&self) {
        let mut map = self.map.lock().expect("ReplayCache lock poisoned");
        map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(parent: &str, sender: &str, seq: i64) -> ReplayCacheKey {
        ReplayCacheKey::new(parent.as_bytes().to_vec().into(), sender.as_bytes().to_vec(), seq)
    }

    fn make_entry(post: &str) -> ReplayCacheEntry {
        ReplayCacheEntry::new(vec![], post.as_bytes().to_vec().into())
    }

    #[test]
    fn test_store_and_retrieve() {
        let cache = InMemoryReplayCache::default_capacity();
        let key = make_key("parent", "sender", 1);
        let entry = make_entry("post-state");

        cache.put(key.clone(), entry.clone());
        let result = cache.get(&key);
        assert!(result.is_some());
    }

    #[test]
    fn test_miss_for_unknown_key() {
        let cache = InMemoryReplayCache::default_capacity();
        let key = make_key("unknown", "sender", 42);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_eviction_when_over_capacity() {
        let cache = InMemoryReplayCache::new(2);

        let k1 = make_key("p1", "a", 1);
        let k2 = make_key("p2", "b", 2);
        let k3 = make_key("p3", "c", 3);
        let e = make_entry("post");

        cache.put(k1.clone(), e.clone());
        cache.put(k2.clone(), e.clone());
        cache.put(k3.clone(), e.clone());

        // k1 should be evicted
        assert!(cache.get(&k1).is_none());
        assert!(cache.get(&k2).is_some());
        assert!(cache.get(&k3).is_some());
    }

    #[test]
    fn test_clear() {
        let cache = InMemoryReplayCache::default_capacity();
        let key = make_key("p", "s", 5);
        let entry = make_entry("post");

        cache.put(key.clone(), entry);
        cache.clear();
        assert!(cache.get(&key).is_none());
    }
}
