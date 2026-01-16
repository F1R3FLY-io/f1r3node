//! Smart Data Storage - Hybrid Vec/im::Vector
//!
//! Provides a hybrid storage type that uses Vec for fast operations during normal
//! execution and im::Vector for efficient checkpointing via structural sharing.

use im::Vector as ImVector;

/// Hybrid storage that uses Vec for fast operations and im::Vector for checkpoints.
///
/// This enum provides the best of both worlds:
/// - **Eager (Vec)**: O(1) amortized operations for normal execution
/// - **Persistent (im::Vector)**: O(1) clone for checkpointing via structural sharing
///
/// The key insight is that most operations are produce/consume during normal execution,
/// while checkpoints are relatively infrequent. By using Vec for eager mode, we get:
/// - O(1) `swap_remove` instead of O(log n) `im::Vector::remove`
/// - Better cache locality for iteration
/// - No persistent data structure overhead
///
/// When a checkpoint is needed, we convert to Persistent mode (or clone as Persistent).
#[derive(Clone, Debug)]
pub enum SmartDataStorage<A: Clone> {
    /// Fast O(1) amortized operations for normal execution.
    /// Uses `swap_remove` for O(1) removal (order doesn't matter for Bag).
    Eager(Vec<(A, bool)>),
    /// O(log n) operations but O(1) clone for checkpoints.
    /// Uses structural sharing for efficient snapshots.
    Persistent(ImVector<(A, bool)>),
}

impl<A: Clone> Default for SmartDataStorage<A> {
    fn default() -> Self {
        SmartDataStorage::Eager(Vec::new())
    }
}

impl<A: Clone> SmartDataStorage<A> {
    /// Create new empty storage in Eager mode.
    #[inline]
    pub fn new() -> Self {
        SmartDataStorage::Eager(Vec::new())
    }

    /// Create new storage with capacity hint (only applies to Eager mode).
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        SmartDataStorage::Eager(Vec::with_capacity(capacity))
    }

    /// Push an element to the back.
    #[inline]
    pub fn push_back(&mut self, item: (A, bool)) {
        match self {
            SmartDataStorage::Eager(vec) => vec.push(item),
            SmartDataStorage::Persistent(imv) => imv.push_back(item),
        }
    }

    /// Get the last element (for Stack).
    #[inline]
    pub fn last(&self) -> Option<&(A, bool)> {
        match self {
            SmartDataStorage::Eager(vec) => vec.last(),
            SmartDataStorage::Persistent(imv) => imv.last(),
        }
    }

    /// Pop and return the last element (for Stack).
    #[inline]
    pub fn pop_back(&mut self) -> Option<(A, bool)> {
        match self {
            SmartDataStorage::Eager(vec) => vec.pop(),
            SmartDataStorage::Persistent(imv) => imv.pop_back(),
        }
    }

    /// Get element at position.
    #[inline]
    pub fn get(&self, pos: usize) -> Option<&(A, bool)> {
        match self {
            SmartDataStorage::Eager(vec) => vec.get(pos),
            SmartDataStorage::Persistent(imv) => imv.get(pos),
        }
    }

    /// Remove element at position.
    /// - Eager: O(1) with swap_remove (doesn't preserve order)
    /// - Persistent: O(log n)
    #[inline]
    pub fn remove(&mut self, pos: usize) -> (A, bool) {
        match self {
            SmartDataStorage::Eager(vec) => vec.swap_remove(pos),
            SmartDataStorage::Persistent(imv) => imv.remove(pos),
        }
    }

    /// Iterate over elements.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &(A, bool)> {
        match self {
            SmartDataStorage::Eager(vec) => SmartIterator::Vec(vec.iter()),
            SmartDataStorage::Persistent(imv) => SmartIterator::Im(imv.iter()),
        }
    }

    /// Find position of first element matching predicate.
    #[inline]
    pub fn position<F>(&self, predicate: F) -> Option<usize>
    where
        F: Fn(&(A, bool)) -> bool,
    {
        self.iter().position(predicate)
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            SmartDataStorage::Eager(vec) => vec.is_empty(),
            SmartDataStorage::Persistent(imv) => imv.is_empty(),
        }
    }

    /// Get length.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            SmartDataStorage::Eager(vec) => vec.len(),
            SmartDataStorage::Persistent(imv) => imv.len(),
        }
    }

    /// Clear all elements.
    #[inline]
    pub fn clear(&mut self) {
        match self {
            SmartDataStorage::Eager(vec) => vec.clear(),
            SmartDataStorage::Persistent(imv) => imv.clear(),
        }
    }

    /// Clone for checkpointing - always returns Persistent mode for O(1) clones.
    ///
    /// If already Persistent, clone is O(1) via structural sharing.
    /// If Eager, converts to Persistent first (O(n)), but the clone itself is O(1).
    #[inline]
    pub fn checkpoint_clone(&self) -> Self {
        match self {
            SmartDataStorage::Persistent(imv) => SmartDataStorage::Persistent(imv.clone()),
            SmartDataStorage::Eager(vec) => {
                // Convert to persistent for O(1) clone sharing
                let imv: ImVector<(A, bool)> = vec.iter().cloned().collect();
                SmartDataStorage::Persistent(imv)
            }
        }
    }

    /// Convert to persistent mode (for checkpointing).
    /// After this call, subsequent clones will be O(1).
    #[inline]
    pub fn to_persistent(&mut self) {
        if let SmartDataStorage::Eager(vec) = self {
            let imv: ImVector<(A, bool)> = vec.drain(..).collect();
            *self = SmartDataStorage::Persistent(imv);
        }
    }

    /// Convert back to eager mode (for performance after checkpoint).
    #[inline]
    pub fn to_eager(&mut self) {
        if let SmartDataStorage::Persistent(imv) = self {
            let vec: Vec<(A, bool)> = imv.iter().cloned().collect();
            *self = SmartDataStorage::Eager(vec);
        }
    }
}

/// Iterator wrapper for SmartDataStorage to handle both storage types.
pub enum SmartIterator<'a, A> {
    Vec(std::slice::Iter<'a, (A, bool)>),
    Im(im::vector::Iter<'a, (A, bool)>),
}

impl<'a, A: Clone> Iterator for SmartIterator<'a, A> {
    type Item = &'a (A, bool);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SmartIterator::Vec(iter) => iter.next(),
            SmartIterator::Im(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            SmartIterator::Vec(iter) => iter.size_hint(),
            SmartIterator::Im(iter) => iter.size_hint(),
        }
    }
}

impl<'a, A: Clone> ExactSizeIterator for SmartIterator<'a, A> {}
