// See shared/src/main/scala/coop/rchain/shared/RecentHashFilter.scala

use indexmap::IndexSet;
use std::sync::Mutex;

/// Simple synchronized LRU filter to suppress redundant gossip messages by hash.
///
/// This filter tracks recently seen hashes and allows callers to check if a hash
/// has been seen before. It uses an LRU eviction policy to bound memory usage.
pub struct RecentHashFilter {
    set: Mutex<IndexSet<String>>,
    max_entries: usize,
}

impl RecentHashFilter {
    /// Create a new RecentHashFilter with the specified maximum capacity.
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of entries to keep
    pub fn new(max_entries: usize) -> Self {
        Self {
            set: Mutex::new(IndexSet::with_capacity(max_entries)),
            max_entries,
        }
    }

    /// Returns true if this hash has been seen recently.
    ///
    /// If the hash is new, it will be added to the filter. If the filter is at
    /// capacity, the oldest entries will be evicted to make room.
    pub fn seen_before(&self, hash: &str) -> bool {
        let mut set = self.set.lock().expect("RecentHashFilter lock poisoned");

        let exists = set.contains(hash);
        if !exists {
            set.insert(hash.to_string());

            // Trim oldest entries if over capacity
            while set.len() > self.max_entries {
                set.shift_remove_index(0);
            }
        }

        exists
    }

    /// Returns the current number of entries in the filter.
    pub fn size(&self) -> usize {
        self.set
            .lock()
            .expect("RecentHashFilter lock poisoned")
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seen_before_returns_false_for_new_hash() {
        let filter = RecentHashFilter::new(10);
        assert!(!filter.seen_before("hash1"));
    }

    #[test]
    fn test_seen_before_returns_true_for_duplicate_hash() {
        let filter = RecentHashFilter::new(10);
        assert!(!filter.seen_before("hash1"));
        assert!(filter.seen_before("hash1"));
    }

    #[test]
    fn test_eviction_when_over_capacity() {
        let filter = RecentHashFilter::new(3);

        // Add 3 hashes
        assert!(!filter.seen_before("hash1"));
        assert!(!filter.seen_before("hash2"));
        assert!(!filter.seen_before("hash3"));
        assert_eq!(filter.size(), 3);

        // Add 4th hash, should evict oldest (hash1)
        assert!(!filter.seen_before("hash4"));
        assert_eq!(filter.size(), 3);

        // hash1 should no longer be seen
        assert!(!filter.seen_before("hash1"));

        // hash2, hash3, hash4 should still be seen (but hash2 is now oldest)
        assert!(filter.seen_before("hash3"));
        assert!(filter.seen_before("hash4"));
    }

    #[test]
    fn test_size() {
        let filter = RecentHashFilter::new(10);
        assert_eq!(filter.size(), 0);

        filter.seen_before("hash1");
        assert_eq!(filter.size(), 1);

        filter.seen_before("hash2");
        assert_eq!(filter.size(), 2);

        // Duplicate shouldn't increase size
        filter.seen_before("hash1");
        assert_eq!(filter.size(), 2);
    }
}
