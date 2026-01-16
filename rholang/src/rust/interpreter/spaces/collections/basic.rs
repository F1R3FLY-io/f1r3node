//! Basic Data Collection Implementations
//!
//! This module provides the standard data collection implementations for channels:
//! - **Bag**: Multiset, any element can match (non-deterministic)
//! - **Queue**: FIFO, only head can match (deterministic)
//! - **Stack**: LIFO, only top can match
//! - **Set**: Unique elements, idempotent sends
//! - **Cell**: At most one element, error on second send
//! - **PriorityQueue**: Priority-based matching

use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

use super::core::DataCollection;
use super::semantics::SemanticHash;
use super::storage::SmartDataStorage;
use crate::rust::interpreter::spaces::errors::SpaceError;

// ==========================================================================
// Bag Data Collection (Default)
// ==========================================================================

/// Bag (multiset) data collection - the default.
///
/// Any element in the bag can match a pattern. Elements can appear multiple times.
/// This provides non-deterministic matching behavior.
/// Data is stored as (value, is_persistent) tuples to support persistent produces.
///
/// Uses `SmartDataStorage` for optimal performance:
/// - O(1) amortized operations during normal execution (Eager mode)
/// - O(1) clone for checkpointing (Persistent mode)
#[derive(Clone, Debug)]
pub struct BagDataCollection<A: Clone> {
    /// Data stored as (value, is_persistent) tuples using smart hybrid storage.
    data: SmartDataStorage<A>,
}

impl<A: Clone> Default for BagDataCollection<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Clone> BagDataCollection<A> {
    pub fn new() -> Self {
        BagDataCollection {
            data: SmartDataStorage::new(),
        }
    }

    /// Creates a new BagDataCollection with pre-allocated capacity.
    /// In Eager mode, uses Vec's capacity; in Persistent mode, capacity is ignored.
    pub fn with_capacity(capacity: usize) -> Self {
        BagDataCollection {
            data: SmartDataStorage::with_capacity(capacity),
        }
    }
}

impl<A: Clone + Send + Sync> DataCollection<A> for BagDataCollection<A> {
    fn put(&mut self, data: A) -> Result<(), SpaceError> {
        self.data.push_back((data, false)); // Non-persistent by default
        Ok(())
    }

    fn put_with_persist(&mut self, data: A, persist: bool) -> Result<(), SpaceError> {
        self.data.push_back((data, persist));
        Ok(())
    }

    fn find_and_remove<F>(&mut self, predicate: F) -> Option<A>
    where
        F: Fn(&A) -> bool,
    {
        // First pass: find position and persistence flag
        let found = self.data.iter().enumerate().find_map(|(pos, (a, persist))| {
            if predicate(a) {
                Some((pos, a.clone(), *persist))
            } else {
                None
            }
        });

        // Second pass: remove if needed (iterator dropped, so mutation is safe)
        if let Some((pos, data, is_persistent)) = found {
            if is_persistent {
                // Persistent data: return clone without removing
                Some(data)
            } else {
                // Non-persistent data: remove and return
                // SmartDataStorage::remove uses O(1) swap_remove in Eager mode,
                // O(log n) in Persistent mode
                Some(self.data.remove(pos).0)
            }
        } else {
            None
        }
    }

    fn peek<F>(&self, predicate: F) -> Option<&A>
    where
        F: Fn(&A) -> bool,
    {
        self.data.iter().find(|(a, _)| predicate(a)).map(|(a, _)| a)
    }

    fn all_data(&self) -> Vec<&A> {
        self.data.iter().map(|(a, _)| a).collect()
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

// ==========================================================================
// Queue Data Collection (FIFO)
// ==========================================================================

/// Queue data collection - FIFO semantics.
///
/// Only the head (front) of the queue can match. This provides deterministic
/// ordering for message processing.
/// Data is stored as (value, is_persistent) tuples to support persistent produces.
#[derive(Clone, Debug)]
pub struct QueueDataCollection<A> {
    data: VecDeque<(A, bool)>,
}

impl<A> Default for QueueDataCollection<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> QueueDataCollection<A> {
    pub fn new() -> Self {
        QueueDataCollection {
            data: VecDeque::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        QueueDataCollection {
            data: VecDeque::with_capacity(capacity),
        }
    }
}

impl<A: Clone + Send + Sync> DataCollection<A> for QueueDataCollection<A> {
    fn put(&mut self, data: A) -> Result<(), SpaceError> {
        self.data.push_back((data, false));
        Ok(())
    }

    fn put_with_persist(&mut self, data: A, persist: bool) -> Result<(), SpaceError> {
        self.data.push_back((data, persist));
        Ok(())
    }

    fn find_and_remove<F>(&mut self, predicate: F) -> Option<A>
    where
        F: Fn(&A) -> bool,
    {
        // Only the front can match in a queue
        if let Some((front, is_persistent)) = self.data.front() {
            if predicate(front) {
                if *is_persistent {
                    return Some(front.clone());
                }
                return self.data.pop_front().map(|(a, _)| a);
            }
        }
        None
    }

    fn peek<F>(&self, predicate: F) -> Option<&A>
    where
        F: Fn(&A) -> bool,
    {
        // Only the front can be peeked in a queue
        self.data.front().filter(|(a, _)| predicate(a)).map(|(a, _)| a)
    }

    fn all_data(&self) -> Vec<&A> {
        self.data.iter().map(|(a, _)| a).collect()
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

// ==========================================================================
// Stack Data Collection (LIFO)
// ==========================================================================

/// Stack data collection - LIFO semantics.
///
/// Only the top of the stack can match. Most recently added element is matched first.
/// Data is stored as (value, is_persistent) tuples to support persistent produces.
///
/// Uses `SmartDataStorage` for optimal performance:
/// - O(1) push/pop during normal execution (Eager mode)
/// - O(1) clone for checkpointing (Persistent mode)
#[derive(Clone, Debug)]
pub struct StackDataCollection<A: Clone> {
    data: SmartDataStorage<A>,
}

impl<A: Clone> Default for StackDataCollection<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Clone> StackDataCollection<A> {
    pub fn new() -> Self {
        StackDataCollection {
            data: SmartDataStorage::new(),
        }
    }

    /// Creates a new StackDataCollection with pre-allocated capacity.
    /// In Eager mode, uses Vec's capacity; in Persistent mode, capacity is ignored.
    pub fn with_capacity(capacity: usize) -> Self {
        StackDataCollection {
            data: SmartDataStorage::with_capacity(capacity),
        }
    }
}

impl<A: Clone + Send + Sync> DataCollection<A> for StackDataCollection<A> {
    fn put(&mut self, data: A) -> Result<(), SpaceError> {
        self.data.push_back((data, false));
        Ok(())
    }

    fn put_with_persist(&mut self, data: A, persist: bool) -> Result<(), SpaceError> {
        self.data.push_back((data, persist));
        Ok(())
    }

    fn find_and_remove<F>(&mut self, predicate: F) -> Option<A>
    where
        F: Fn(&A) -> bool,
    {
        // Only the top can match in a stack
        if let Some((top, is_persistent)) = self.data.last() {
            if predicate(top) {
                if *is_persistent {
                    return Some(top.clone());
                }
                return self.data.pop_back().map(|(a, _)| a);
            }
        }
        None
    }

    fn peek<F>(&self, predicate: F) -> Option<&A>
    where
        F: Fn(&A) -> bool,
    {
        // Only the top can be peeked in a stack
        self.data.last().filter(|(a, _)| predicate(a)).map(|(a, _)| a)
    }

    fn all_data(&self) -> Vec<&A> {
        self.data.iter().map(|(a, _)| a).collect()
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

// ==========================================================================
// Set Data Collection (Idempotent) - O(1) Deduplication
// ==========================================================================

/// Set data collection - unique elements only with O(1) deduplication.
///
/// Sending the same datum multiple times has no effect (idempotent).
/// Useful for ensuring message deduplication.
/// Data is stored as (value, is_persistent) tuples to support persistent produces.
///
/// # Performance
/// Uses a hash index for O(1) duplicate detection, improving over the previous
/// O(n) linear scan. The hash index stores semantic hashes, consistent with
/// the `SemanticHash` trait which ignores metadata like `random_state`.
#[derive(Clone, Debug)]
pub struct SetDataCollection<A: Hash + Eq + SemanticHash> {
    /// Data stored as (value, is_persistent) tuples
    data: Vec<(A, bool)>,
    /// Hash index for O(1) deduplication (stores semantic hashes)
    hash_index: HashSet<u64>,
}

impl<A: Hash + Eq + SemanticHash> Default for SetDataCollection<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Hash + Eq + SemanticHash> SetDataCollection<A> {
    pub fn new() -> Self {
        SetDataCollection {
            data: Vec::new(),
            hash_index: HashSet::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        SetDataCollection {
            data: Vec::with_capacity(capacity),
            hash_index: HashSet::with_capacity(capacity),
        }
    }
}

impl<A: Clone + Send + Sync + Hash + Eq + SemanticHash> DataCollection<A> for SetDataCollection<A> {
    fn put(&mut self, data: A) -> Result<(), SpaceError> {
        // O(1) deduplication using semantic hash
        let hash = data.semantic_hash();
        if !self.hash_index.contains(&hash) {
            self.hash_index.insert(hash);
            self.data.push((data, false));
        }
        Ok(())
    }

    fn put_with_persist(&mut self, data: A, persist: bool) -> Result<(), SpaceError> {
        // O(1) deduplication using semantic hash
        let hash = data.semantic_hash();
        if !self.hash_index.contains(&hash) {
            self.hash_index.insert(hash);
            self.data.push((data, persist));
        }
        Ok(())
    }

    fn find_and_remove<F>(&mut self, predicate: F) -> Option<A>
    where
        F: Fn(&A) -> bool,
    {
        if let Some(pos) = self.data.iter().position(|(a, _)| predicate(a)) {
            let (data, is_persistent) = &self.data[pos];
            if *is_persistent {
                Some(data.clone())
            } else {
                // Remove from hash index when removing data
                let hash = data.semantic_hash();
                self.hash_index.remove(&hash);
                Some(self.data.swap_remove(pos).0)
            }
        } else {
            None
        }
    }

    fn peek<F>(&self, predicate: F) -> Option<&A>
    where
        F: Fn(&A) -> bool,
    {
        self.data.iter().find(|(a, _)| predicate(a)).map(|(a, _)| a)
    }

    fn all_data(&self) -> Vec<&A> {
        self.data.iter().map(|(a, _)| a).collect()
    }

    fn clear(&mut self) {
        self.data.clear();
        self.hash_index.clear();
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

// ==========================================================================
// Cell Data Collection (Exactly Once)
// ==========================================================================

/// Cell data collection - at most one element.
///
/// Attempting to send to a cell that already contains data returns an error.
/// Provides exactly-once semantics with enforcement.
/// Data is stored as (value, is_persistent) tuple to support persistent produces.
#[derive(Clone, Debug)]
pub struct CellDataCollection<A> {
    /// Data stored as (value, is_persistent) tuple
    data: Option<(A, bool)>,
    channel_desc: String,
}

impl<A> Default for CellDataCollection<A> {
    fn default() -> Self {
        Self::new("unknown".to_string())
    }
}

impl<A> CellDataCollection<A> {
    pub fn new(channel_desc: String) -> Self {
        CellDataCollection {
            data: None,
            channel_desc,
        }
    }
}

impl<A: Clone + Send + Sync> DataCollection<A> for CellDataCollection<A> {
    fn put(&mut self, data: A) -> Result<(), SpaceError> {
        if self.data.is_some() {
            return Err(SpaceError::CellAlreadyFull {
                channel_desc: self.channel_desc.clone(),
            });
        }
        self.data = Some((data, false));
        Ok(())
    }

    fn put_with_persist(&mut self, data: A, persist: bool) -> Result<(), SpaceError> {
        if self.data.is_some() {
            return Err(SpaceError::CellAlreadyFull {
                channel_desc: self.channel_desc.clone(),
            });
        }
        self.data = Some((data, persist));
        Ok(())
    }

    fn find_and_remove<F>(&mut self, predicate: F) -> Option<A>
    where
        F: Fn(&A) -> bool,
    {
        if let Some((ref data, is_persistent)) = self.data {
            if predicate(data) {
                if is_persistent {
                    return Some(data.clone());
                }
                return self.data.take().map(|(a, _)| a);
            }
        }
        None
    }

    fn peek<F>(&self, predicate: F) -> Option<&A>
    where
        F: Fn(&A) -> bool,
    {
        self.data.as_ref().filter(|(a, _)| predicate(a)).map(|(a, _)| a)
    }

    fn all_data(&self) -> Vec<&A> {
        self.data.iter().map(|(a, _)| a).collect()
    }

    fn clear(&mut self) {
        self.data = None;
    }

    fn is_empty(&self) -> bool {
        self.data.is_none()
    }

    fn len(&self) -> usize {
        if self.data.is_some() { 1 } else { 0 }
    }
}

// ==========================================================================
// Priority Queue Data Collection
// ==========================================================================

/// Priority queue data collection - priority-based matching.
///
/// Data is organized into priority levels (0 = highest). Matching always
/// attempts the highest priority level first. Within a priority level,
/// elements are matched in FIFO order.
///
/// This is useful for implementing priority-based message handling where
/// urgent messages should be processed before normal messages.
/// Data is stored as (value, is_persistent) tuples to support persistent produces.
#[derive(Clone, Debug)]
pub struct PriorityQueueDataCollection<A> {
    /// Queues for each priority level, index 0 = highest priority
    /// Data stored as (value, is_persistent) tuples
    queues: Vec<VecDeque<(A, bool)>>,
    /// Number of priority levels
    num_priorities: usize,
    /// Default priority for put() without explicit priority
    default_priority: usize,
}

impl<A> PriorityQueueDataCollection<A> {
    /// Create a new priority queue with the given number of priority levels.
    ///
    /// # Arguments
    /// - `num_priorities`: Number of priority levels (must be >= 1)
    pub fn new(num_priorities: usize) -> Self {
        let num_priorities = num_priorities.max(1);
        PriorityQueueDataCollection {
            queues: (0..num_priorities).map(|_| VecDeque::new()).collect(),
            num_priorities,
            default_priority: num_priorities - 1, // Lowest priority is default
        }
    }

    /// Create a priority queue with default 3 priority levels.
    pub fn new_default() -> Self {
        Self::new(3)
    }

    /// Ensure the priority queue has capacity for the given priority level.
    ///
    /// If the requested priority exceeds current capacity, new empty queues
    /// are added to accommodate it. This enables dynamic growth of priority
    /// levels without requiring upfront specification.
    ///
    /// # Arguments
    /// - `priority`: The priority level that must be accommodated
    fn ensure_capacity(&mut self, priority: usize) {
        while self.queues.len() <= priority {
            self.queues.push(VecDeque::new());
        }
        self.num_priorities = self.queues.len();
    }

    /// Put data with a specific priority level (non-persistent).
    ///
    /// # Arguments
    /// - `data`: The data to insert
    /// - `priority`: Priority level (0 = highest). If the priority exceeds current
    ///   capacity, the queue dynamically grows to accommodate it.
    pub fn put_with_priority(&mut self, data: A, priority: usize) -> Result<(), SpaceError> {
        self.ensure_capacity(priority);
        self.queues[priority].push_back((data, false));
        Ok(())
    }

    /// Put data with a specific priority level and persistence flag.
    ///
    /// # Arguments
    /// - `data`: The data to insert
    /// - `priority`: Priority level (0 = highest). If the priority exceeds current
    ///   capacity, the queue dynamically grows to accommodate it.
    /// - `persist`: If true, the data will not be removed when matched
    pub fn put_with_priority_and_persist(&mut self, data: A, priority: usize, persist: bool) -> Result<(), SpaceError> {
        self.ensure_capacity(priority);
        self.queues[priority].push_back((data, persist));
        Ok(())
    }

    /// Set the default priority for put() calls.
    ///
    /// If the requested priority exceeds current capacity, the queue
    /// dynamically grows to accommodate it.
    pub fn set_default_priority(&mut self, priority: usize) {
        self.ensure_capacity(priority);
        self.default_priority = priority;
    }

    /// Get the number of priority levels.
    pub fn num_priorities(&self) -> usize {
        self.num_priorities
    }

    /// Get the count at each priority level.
    pub fn counts_by_priority(&self) -> Vec<usize> {
        self.queues.iter().map(|q| q.len()).collect()
    }
}

impl<A> Default for PriorityQueueDataCollection<A> {
    fn default() -> Self {
        Self::new_default()
    }
}

impl<A: Clone + Send + Sync> DataCollection<A> for PriorityQueueDataCollection<A> {
    fn put(&mut self, data: A) -> Result<(), SpaceError> {
        self.put_with_priority(data, self.default_priority)
    }

    fn put_with_persist(&mut self, data: A, persist: bool) -> Result<(), SpaceError> {
        self.put_with_priority_and_persist(data, self.default_priority, persist)
    }

    fn find_and_remove<F>(&mut self, predicate: F) -> Option<A>
    where
        F: Fn(&A) -> bool,
    {
        // Search from highest priority (0) to lowest
        for queue in &mut self.queues {
            if let Some((front, is_persistent)) = queue.front() {
                if predicate(front) {
                    if *is_persistent {
                        return Some(front.clone());
                    }
                    return queue.pop_front().map(|(a, _)| a);
                }
            }
        }
        None
    }

    fn peek<F>(&self, predicate: F) -> Option<&A>
    where
        F: Fn(&A) -> bool,
    {
        // Search from highest priority (0) to lowest
        for queue in &self.queues {
            if let Some((front, _)) = queue.front() {
                if predicate(front) {
                    return Some(front);
                }
            }
        }
        None
    }

    fn all_data(&self) -> Vec<&A> {
        self.queues.iter().flat_map(|q| q.iter().map(|(a, _)| a)).collect()
    }

    fn clear(&mut self) {
        for queue in &mut self.queues {
            queue.clear();
        }
    }

    fn is_empty(&self) -> bool {
        self.queues.iter().all(|q| q.is_empty())
    }

    fn len(&self) -> usize {
        self.queues.iter().map(|q| q.len()).sum()
    }
}
