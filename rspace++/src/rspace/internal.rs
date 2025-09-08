// See rspace/src/main/scala/coop/rchain/rspace/internal.scala

use counter::Counter;
use dashmap::DashMap;
use proptest_derive::Arbitrary;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::hash::Hash;

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
where
    A: Clone + Serialize,
{
    pub fn create<C: Serialize>(channel: &C, a: A, persist: bool) -> Datum<A> {
        Datum {
            a: a.clone(),
            persist,
            source: Produce::create(channel, &a, persist),
        }
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
        WaitingContinuation {
            patterns: patterns.clone(),
            continuation: continuation.clone(),
            persist,
            peeks,
            source: Consume::create(channels, patterns, continuation, persist),
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

#[derive(Clone, Debug)]
pub struct MultisetMultiMap<K: Hash + Eq, V: Hash + Eq> {
    pub map: DashMap<K, Counter<V>>,
}

impl<K, V> MultisetMultiMap<K, V>
where
    K: Eq + Hash,
    V: Eq + Hash,
{
    pub fn empty() -> Self {
        MultisetMultiMap {
            map: DashMap::new(),
        }
    }

    pub fn add_binding(&self, k: K, v: V) {
        match self.map.get_mut(&k) {
            Some(mut current) => {
                current.insert(v, 1);
            }
            None => {
                let mut ms = Counter::new();
                ms.insert(v, 1);
                self.map.insert(k, ms);
            }
        }
    }

    pub fn clear(&self) {
        self.map.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl<K: Hash + Eq, V: Hash + Eq> MultisetMultiMap<K, V> {
    // In-place removal to avoid moving the whole map
    pub fn remove_binding_in_place(&self, k: &K, v: &V) {
        let mut should_remove_key = false;

        if let Some(mut current) = self.map.get_mut(k) {
            current.remove(v);
            if current.is_empty() {
                should_remove_key = true;
            }
        }

        if should_remove_key {
            self.map.remove(k);
        }
    }
}

// This function remains for compatibility but delegates to in-place version and returns the same map
pub fn remove_binding<K: Hash + Eq, V: Hash + Eq>(
    ms: MultisetMultiMap<K, V>,
    k: K,
    v: V,
) -> MultisetMultiMap<K, V> {
    ms.remove_binding_in_place(&k, &v);
    ms
}
