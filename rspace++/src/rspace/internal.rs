// See rspace/src/main/scala/coop/rchain/rspace/internal.scala

use std::collections::BTreeSet;
use std::hash::Hash;

use dashmap::DashMap;
use smallvec::SmallVec;
use proptest_derive::Arbitrary;
use serde::{Deserialize, Serialize};

use super::trace::event::{Consume, Produce};

// The 'Arbitrary' macro is needed here for proptest in hot_store_spec.rs
// The 'Default' macro is needed here for hot_store_spec.rs
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash, Arbitrary, Default)]
pub struct Datum<A: Clone> {
    pub a: A,
    pub persist: bool,
    pub source: Produce,
}

impl<A> Datum<A>
where A: Clone + Serialize
{
    pub fn create<C: Serialize>(channel: &C, a: A, persist: bool) -> Datum<A> {
        let source = Produce::create(channel, &a, persist);
        Datum { a, persist, source }
    }
}

// The 'Arbitrary' macro is needed here for proptest in hot_store_spec.rs
// The 'Default' macro is needed here for hot_store_spec.rs
#[derive(Clone, Debug, Arbitrary, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct WaitingContinuation<P: Clone, K: Clone> {
    pub patterns: Vec<P>,
    pub continuation: K,
    pub persist: bool,
    pub peeks: BTreeSet<i32>,
    pub source: Consume,
}

impl<P, K> WaitingContinuation<P, K>
where
    P: Clone + Serialize,
    K: Clone + Serialize,
{
    pub fn create<C: Clone + Serialize>(
        channels: &Vec<C>,
        patterns: &Vec<P>,
        continuation: &K,
        persist: bool,
        peeks: BTreeSet<i32>,
    ) -> WaitingContinuation<P, K> {
        let source = Consume::create(&channels, &patterns, &continuation, persist);
        WaitingContinuation {
            patterns: patterns.to_vec(),
            continuation: continuation.clone(),
            persist,
            peeks,
            source,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConsumeCandidate<C, A: Clone> {
    pub channel: C,
    pub datum: Datum<A>,
    pub removed_datum: A,
    pub datum_index: i32,
}

#[derive(Debug)]
pub struct ProduceCandidate<C, P: Clone, A: Clone, K: Clone> {
    pub channels: Vec<C>,
    pub continuation: WaitingContinuation<P, K>,
    pub continuation_index: i32,
    pub data_candidates: Vec<ConsumeCandidate<C, A>>,
}

// Eq and PartialEq is needed here for reduce_spec tests
#[derive(Debug, Eq, PartialEq)]
pub struct Row<P: Clone, A: Clone, K: Clone> {
    pub data: Vec<Datum<A>>,
    pub wks: Vec<WaitingContinuation<P, K>>,
}

#[derive(Clone, Debug)]
pub struct Install<P, K> {
    pub patterns: Vec<P>,
    pub continuation: K,
}

/// Multiset multi-map that preserves insertion order per key.
///
/// Uses `SmallVec<[V; 4]>` instead of `Counter<V>` (HashMap) so that
/// iteration order is deterministic — matching the event log order from
/// `rig()`.  This is critical for replay correctness: when multiple COMMs
/// exist for the same IOEvent (e.g. a persistent consume firing multiple
/// times), `get_comm_or_candidate` tries them in iteration order and
/// returns the FIRST match.  Non-deterministic HashMap ordering caused
/// different COMMs to fire first on observer vs validator, producing
/// different execution paths and cost mismatches.
#[derive(Clone, Debug)]
pub struct MultisetMultiMap<K: Hash + Eq, V: PartialEq> {
    pub map: DashMap<K, SmallVec<[V; 4]>>,
}

impl<K, V> MultisetMultiMap<K, V>
where
    K: Eq + Hash,
    V: PartialEq,
{
    pub fn empty() -> Self {
        MultisetMultiMap {
            map: DashMap::new(),
        }
    }

    pub fn add_binding(&self, k: K, v: V) {
        match self.map.get_mut(&k) {
            Some(mut current) => {
                current.push(v);
            }
            None => {
                let mut sv = SmallVec::new();
                sv.push(v);
                self.map.insert(k, sv);
            }
        }
    }

    pub fn clear(&self) { self.map.clear(); }

    pub fn is_empty(&self) -> bool { self.map.is_empty() }

    // In-place removal to avoid moving the whole map.
    // Removes the first occurrence of `v` from the vec at key `k`.
    pub fn remove_binding_in_place(&self, k: &K, v: &V) {
        let mut should_remove_key = false;

        if let Some(mut current) = self.map.get_mut(k) {
            if let Some(pos) = current.iter().position(|x| x == v) {
                current.remove(pos);
            }

            if current.is_empty() {
                should_remove_key = true;
            }
        }

        if should_remove_key {
            self.map.remove(k);
        }
    }
}

// This function remains for compatibility but delegates to in-place version and
// returns the same map
pub fn remove_binding<K: Hash + Eq, V: PartialEq>(
    ms: MultisetMultiMap<K, V>,
    k: K,
    v: V,
) -> MultisetMultiMap<K, V> {
    ms.remove_binding_in_place(&k, &v);
    ms
}

#[cfg(test)]
mod tests {
    use super::MultisetMultiMap;

    #[test]
    fn multiset_multimap_add_binding_preserves_insertion_order() {
        let ms = MultisetMultiMap::empty();
        ms.add_binding("k", "v1");
        ms.add_binding("k", "v2");
        ms.add_binding("k", "v1");

        let vec: Vec<_> = ms
            .map
            .get(&"k")
            .map(|v| v.to_vec())
            .unwrap_or_default();
        assert_eq!(vec, vec!["v1", "v2", "v1"]);
    }

    #[test]
    fn multiset_multimap_remove_binding_removes_first_occurrence() {
        let ms = MultisetMultiMap::empty();
        ms.add_binding("k", "v1");
        ms.add_binding("k", "v2");
        ms.add_binding("k", "v1");

        ms.remove_binding_in_place(&"k", &"v1");
        let vec_after_one_remove: Vec<_> = ms
            .map
            .get(&"k")
            .map(|v| v.to_vec())
            .unwrap_or_default();
        // First "v1" removed, leaving ["v2", "v1"]
        assert_eq!(vec_after_one_remove, vec!["v2", "v1"]);

        ms.remove_binding_in_place(&"k", &"v2");
        let vec_after_two_removes: Vec<_> = ms
            .map
            .get(&"k")
            .map(|v| v.to_vec())
            .unwrap_or_default();
        assert_eq!(vec_after_two_removes, vec!["v1"]);

        ms.remove_binding_in_place(&"k", &"v1");
        assert!(ms.map.get(&"k").is_none());
    }
}
