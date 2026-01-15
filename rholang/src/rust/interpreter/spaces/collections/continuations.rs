//! Continuation Collection Implementations
//!
//! This module provides the continuation collection implementations for channels:
//! - **Bag**: Any continuation can match (non-deterministic)
//! - **Queue**: FIFO, only front can match (deterministic)
//! - **Stack**: LIFO, only top can match
//! - **Set**: Unique continuations, idempotent registration

use std::collections::VecDeque;

use im::Vector as ImVector;

use super::core::ContinuationCollection;
use super::similarity::StoredSimilarityInfo;

// ==========================================================================
// Bag Continuation Collection
// ==========================================================================

/// Bag continuation collection - default for continuations.
///
/// Stores tuples of (patterns, continuation, persist, similarity_info).
///
/// Uses `im::Vector` for O(1) clone/snapshot via structural sharing - optimal for checkpointing.
#[derive(Clone, Debug)]
pub struct BagContinuationCollection<P: Clone, K: Clone> {
    /// Storage format: (patterns, continuation, persist, optional_similarity)
    continuations: ImVector<(Vec<P>, K, bool, Option<StoredSimilarityInfo>)>,
}

impl<P: Clone, K: Clone> Default for BagContinuationCollection<P, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Clone, K: Clone> BagContinuationCollection<P, K> {
    pub fn new() -> Self {
        BagContinuationCollection {
            continuations: ImVector::new(),
        }
    }
}

impl<P: Clone + Send + Sync, K: Clone + Send + Sync> ContinuationCollection<P, K>
    for BagContinuationCollection<P, K>
{
    fn put(&mut self, patterns: Vec<P>, continuation: K, persist: bool) {
        self.continuations.push_back((patterns, continuation, persist, None));
    }

    fn put_with_similarity(
        &mut self,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        similarity: Option<StoredSimilarityInfo>,
    ) {
        self.continuations.push_back((patterns, continuation, persist, similarity));
    }

    fn find_and_remove<F>(&mut self, predicate: F) -> Option<(Vec<P>, K, bool)>
    where
        F: Fn(&[P], &K) -> bool,
    {
        if let Some(pos) = self
            .continuations
            .iter()
            .position(|(p, k, _, _)| predicate(p, k))
        {
            let (patterns, cont, persist, _similarity) = self.continuations.remove(pos);
            Some((patterns, cont, persist))
        } else {
            None
        }
    }

    fn find_and_remove_with_similarity<F>(
        &mut self,
        predicate: F,
    ) -> Option<(Vec<P>, K, bool, Option<StoredSimilarityInfo>)>
    where
        F: Fn(&[P], &K) -> bool,
    {
        if let Some(pos) = self
            .continuations
            .iter()
            .position(|(p, k, _, _)| predicate(p, k))
        {
            Some(self.continuations.remove(pos))
        } else {
            None
        }
    }

    fn all_continuations(&self) -> Vec<(&[P], &K, bool)> {
        self.continuations
            .iter()
            .map(|(p, k, persist, _)| (p.as_slice(), k, *persist))
            .collect()
    }

    fn all_continuations_with_similarity(&self) -> Vec<(&[P], &K, bool, Option<&StoredSimilarityInfo>)> {
        self.continuations
            .iter()
            .map(|(p, k, persist, sim)| (p.as_slice(), k, *persist, sim.as_ref()))
            .collect()
    }

    fn clear(&mut self) {
        self.continuations.clear();
    }

    fn is_empty(&self) -> bool {
        self.continuations.is_empty()
    }

    fn len(&self) -> usize {
        self.continuations.len()
    }
}

// ==========================================================================
// Queue Continuation Collection
// ==========================================================================

/// Queue continuation collection - FIFO for continuations.
#[derive(Clone, Debug)]
pub struct QueueContinuationCollection<P, K> {
    continuations: VecDeque<(Vec<P>, K, bool, Option<StoredSimilarityInfo>)>,
}

impl<P, K> Default for QueueContinuationCollection<P, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P, K> QueueContinuationCollection<P, K> {
    pub fn new() -> Self {
        QueueContinuationCollection {
            continuations: VecDeque::new(),
        }
    }
}

impl<P: Clone + Send + Sync, K: Clone + Send + Sync> ContinuationCollection<P, K>
    for QueueContinuationCollection<P, K>
{
    fn put(&mut self, patterns: Vec<P>, continuation: K, persist: bool) {
        self.continuations.push_back((patterns, continuation, persist, None));
    }

    fn put_with_similarity(
        &mut self,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        similarity: Option<StoredSimilarityInfo>,
    ) {
        self.continuations.push_back((patterns, continuation, persist, similarity));
    }

    fn find_and_remove<F>(&mut self, predicate: F) -> Option<(Vec<P>, K, bool)>
    where
        F: Fn(&[P], &K) -> bool,
    {
        // Only the front can match in a queue
        if let Some((patterns, k, _persist, _)) = self.continuations.front() {
            if predicate(patterns, k) {
                return self.continuations.pop_front().map(|(p, k, persist, _)| (p, k, persist));
            }
        }
        None
    }

    fn find_and_remove_with_similarity<F>(
        &mut self,
        predicate: F,
    ) -> Option<(Vec<P>, K, bool, Option<StoredSimilarityInfo>)>
    where
        F: Fn(&[P], &K) -> bool,
    {
        if let Some((patterns, k, _persist, _)) = self.continuations.front() {
            if predicate(patterns, k) {
                return self.continuations.pop_front();
            }
        }
        None
    }

    fn all_continuations(&self) -> Vec<(&[P], &K, bool)> {
        self.continuations
            .iter()
            .map(|(p, k, persist, _)| (p.as_slice(), k, *persist))
            .collect()
    }

    fn all_continuations_with_similarity(&self) -> Vec<(&[P], &K, bool, Option<&StoredSimilarityInfo>)> {
        self.continuations
            .iter()
            .map(|(p, k, persist, sim)| (p.as_slice(), k, *persist, sim.as_ref()))
            .collect()
    }

    fn clear(&mut self) {
        self.continuations.clear();
    }

    fn is_empty(&self) -> bool {
        self.continuations.is_empty()
    }

    fn len(&self) -> usize {
        self.continuations.len()
    }
}

// ==========================================================================
// Stack Continuation Collection (LIFO)
// ==========================================================================

/// Stack continuation collection - LIFO for continuations.
///
/// Only the top (most recently added) continuation can match. This provides
/// deterministic LIFO ordering for continuation processing.
///
/// Stores tuples of (patterns, continuation, persist, similarity_info).
#[derive(Clone, Debug)]
pub struct StackContinuationCollection<P, K> {
    /// Storage format: (patterns, continuation, persist, optional_similarity)
    continuations: Vec<(Vec<P>, K, bool, Option<StoredSimilarityInfo>)>,
}

impl<P, K> Default for StackContinuationCollection<P, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P, K> StackContinuationCollection<P, K> {
    pub fn new() -> Self {
        StackContinuationCollection {
            continuations: Vec::new(),
        }
    }
}

impl<P: Clone + Send + Sync, K: Clone + Send + Sync> ContinuationCollection<P, K>
    for StackContinuationCollection<P, K>
{
    fn put(&mut self, patterns: Vec<P>, continuation: K, persist: bool) {
        self.continuations.push((patterns, continuation, persist, None));
    }

    fn put_with_similarity(
        &mut self,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        similarity: Option<StoredSimilarityInfo>,
    ) {
        self.continuations.push((patterns, continuation, persist, similarity));
    }

    fn find_and_remove<F>(&mut self, predicate: F) -> Option<(Vec<P>, K, bool)>
    where
        F: Fn(&[P], &K) -> bool,
    {
        // Only the top (last) can match in a stack
        if let Some((patterns, k, _persist, _)) = self.continuations.last() {
            if predicate(patterns, k) {
                let (patterns, cont, persist, _similarity) = self.continuations.pop().expect("just checked");
                return Some((patterns, cont, persist));
            }
        }
        None
    }

    fn find_and_remove_with_similarity<F>(
        &mut self,
        predicate: F,
    ) -> Option<(Vec<P>, K, bool, Option<StoredSimilarityInfo>)>
    where
        F: Fn(&[P], &K) -> bool,
    {
        // Only the top (last) can match in a stack
        if let Some((patterns, k, _persist, _)) = self.continuations.last() {
            if predicate(patterns, k) {
                return self.continuations.pop();
            }
        }
        None
    }

    fn all_continuations(&self) -> Vec<(&[P], &K, bool)> {
        self.continuations
            .iter()
            .map(|(p, k, persist, _)| (p.as_slice(), k, *persist))
            .collect()
    }

    fn all_continuations_with_similarity(&self) -> Vec<(&[P], &K, bool, Option<&StoredSimilarityInfo>)> {
        self.continuations
            .iter()
            .map(|(p, k, persist, sim)| (p.as_slice(), k, *persist, sim.as_ref()))
            .collect()
    }

    fn clear(&mut self) {
        self.continuations.clear();
    }

    fn is_empty(&self) -> bool {
        self.continuations.is_empty()
    }

    fn len(&self) -> usize {
        self.continuations.len()
    }
}

// ==========================================================================
// Set Continuation Collection (Idempotent)
// ==========================================================================

/// Set continuation collection - unique continuations only.
///
/// Registering the same continuation (same patterns, continuation, persist)
/// multiple times has no effect (idempotent). Useful for deduplication.
///
/// Note: Unlike SetDataCollection, this uses a Vec internally and checks
/// for duplicates on insert, because (Vec<P>, K, bool) tuples don't
/// naturally implement Hash + Eq for all P, K types.
///
/// Stores tuples of (patterns, continuation, persist, similarity_info).
#[derive(Clone, Debug)]
pub struct SetContinuationCollection<P, K> {
    /// Storage format: (patterns, continuation, persist, optional_similarity)
    continuations: Vec<(Vec<P>, K, bool, Option<StoredSimilarityInfo>)>,
}

impl<P, K> Default for SetContinuationCollection<P, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P, K> SetContinuationCollection<P, K> {
    pub fn new() -> Self {
        SetContinuationCollection {
            continuations: Vec::new(),
        }
    }
}

impl<P: Clone + Send + Sync + PartialEq, K: Clone + Send + Sync + PartialEq> ContinuationCollection<P, K>
    for SetContinuationCollection<P, K>
{
    fn put(&mut self, patterns: Vec<P>, continuation: K, persist: bool) {
        // Only add if not already present (idempotent)
        // For backwards compatibility, insert with no similarity info
        let entry = (patterns.clone(), continuation.clone(), persist, None);
        // Check equality on (patterns, continuation, persist), ignoring similarity
        if !self.continuations.iter().any(|(p, k, per, _)| *p == patterns && *k == continuation && *per == persist) {
            self.continuations.push(entry);
        }
    }

    fn put_with_similarity(
        &mut self,
        patterns: Vec<P>,
        continuation: K,
        persist: bool,
        similarity: Option<StoredSimilarityInfo>,
    ) {
        // Only add if not already present (idempotent)
        // Check equality on (patterns, continuation, persist), ignoring similarity
        if !self.continuations.iter().any(|(p, k, per, _)| *p == patterns && *k == continuation && *per == persist) {
            self.continuations.push((patterns, continuation, persist, similarity));
        }
    }

    fn find_and_remove<F>(&mut self, predicate: F) -> Option<(Vec<P>, K, bool)>
    where
        F: Fn(&[P], &K) -> bool,
    {
        // Any matching continuation can be removed (non-deterministic like Bag)
        if let Some(pos) = self
            .continuations
            .iter()
            .position(|(p, k, _, _)| predicate(p, k))
        {
            let (patterns, cont, persist, _similarity) = self.continuations.swap_remove(pos);
            Some((patterns, cont, persist))
        } else {
            None
        }
    }

    fn find_and_remove_with_similarity<F>(
        &mut self,
        predicate: F,
    ) -> Option<(Vec<P>, K, bool, Option<StoredSimilarityInfo>)>
    where
        F: Fn(&[P], &K) -> bool,
    {
        // Any matching continuation can be removed (non-deterministic like Bag)
        if let Some(pos) = self
            .continuations
            .iter()
            .position(|(p, k, _, _)| predicate(p, k))
        {
            Some(self.continuations.swap_remove(pos))
        } else {
            None
        }
    }

    fn all_continuations(&self) -> Vec<(&[P], &K, bool)> {
        self.continuations
            .iter()
            .map(|(p, k, persist, _)| (p.as_slice(), k, *persist))
            .collect()
    }

    fn all_continuations_with_similarity(&self) -> Vec<(&[P], &K, bool, Option<&StoredSimilarityInfo>)> {
        self.continuations
            .iter()
            .map(|(p, k, persist, sim)| (p.as_slice(), k, *persist, sim.as_ref()))
            .collect()
    }

    fn clear(&mut self) {
        self.continuations.clear();
    }

    fn is_empty(&self) -> bool {
        self.continuations.is_empty()
    }

    fn len(&self) -> usize {
        self.continuations.len()
    }
}
