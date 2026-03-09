// See rspace/src/main/scala/coop/rchain/rspace/merger/EventLogIndex.scala

use rayon::prelude::*;
use shared::rust::hashable_set::HashableSet;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::rspace::trace::event::{Consume, Event, IOEvent, Produce};

use super::merging_logic::{combine_produces_copied_by_peek, NumberChannelsDiff};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EventLogIndex {
    pub produces_linear: HashableSet<Produce>,
    pub produces_persistent: HashableSet<Produce>,
    pub produces_consumed: HashableSet<Produce>,
    pub produces_peeked: HashableSet<Produce>,
    pub produces_copied_by_peek: HashableSet<Produce>,
    pub produces_touching_base_joins: HashableSet<Produce>,
    pub consumes_linear_and_peeks: HashableSet<Consume>,
    pub consumes_persistent: HashableSet<Consume>,
    pub consumes_produced: HashableSet<Consume>,
    pub produces_mergeable: HashableSet<Produce>,
    pub consumes_mergeable: HashableSet<Consume>,
    pub number_channels_data: NumberChannelsDiff,
}

// Ordering for deterministic processing in merge operations.
// Compares by numberChannelsData entries (key and value) in sorted key order,
// with fallback to produce/consume counts to distinguish structurally different indices.
// This replaces a previous derived Ord that was susceptible to non-deterministic
// HashSet iteration order, where two different EventLogIndex instances with different
// numberChannelsData could compare inconsistently.
impl PartialOrd for EventLogIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EventLogIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // numberChannelsData is BTreeMap<Blake2b256Hash, i64>, already sorted by key
        let a_entries: Vec<_> = self.number_channels_data.iter().collect();
        let b_entries: Vec<_> = other.number_channels_data.iter().collect();

        let len_cmp = a_entries.len().cmp(&b_entries.len());
        if len_cmp != std::cmp::Ordering::Equal {
            return len_cmp;
        }

        // Compare entries lexicographically: first by key (Blake2b256Hash), then by value (i64)
        for ((ak, av), (bk, bv)) in a_entries.iter().zip(b_entries.iter()) {
            let key_cmp = ak.cmp(bk);
            if key_cmp != std::cmp::Ordering::Equal {
                return key_cmp;
            }
            let val_cmp = av.cmp(bv);
            if val_cmp != std::cmp::Ordering::Equal {
                return val_cmp;
            }
        }

        // If numberChannelsData are identical, distinguish by event counts
        let a_prod = self.produces_linear.0.len()
            + self.produces_persistent.0.len()
            + self.produces_consumed.0.len();
        let b_prod = other.produces_linear.0.len()
            + other.produces_persistent.0.len()
            + other.produces_consumed.0.len();
        let prod_cmp = a_prod.cmp(&b_prod);
        if prod_cmp != std::cmp::Ordering::Equal {
            return prod_cmp;
        }

        let a_cons = self.consumes_linear_and_peeks.0.len()
            + self.consumes_persistent.0.len()
            + self.consumes_produced.0.len();
        let b_cons = other.consumes_linear_and_peeks.0.len()
            + other.consumes_persistent.0.len()
            + other.consumes_produced.0.len();
        a_cons.cmp(&b_cons)
    }
}

impl EventLogIndex {
    pub fn new(
        event_log: Vec<Event>,
        produce_exists_in_pre_state: impl Fn(&Produce) -> bool,
        produce_touch_pre_state_join: impl Fn(&Produce) -> bool,
        mergeable_chs: NumberChannelsDiff,
    ) -> Self {
        // Use Arc<Mutex<>> for thread-safe collections that will be updated in parallel
        let produces_linear = Arc::new(Mutex::new(HashSet::new()));
        let produces_persistent = Arc::new(Mutex::new(HashSet::new()));
        let produces_consumed = Arc::new(Mutex::new(HashSet::new()));
        let produces_peeked = Arc::new(Mutex::new(HashSet::new()));
        let produces_copied_by_peek = Arc::new(Mutex::new(HashSet::new()));
        let produces_touching_base_joins = Arc::new(Mutex::new(HashSet::new()));
        let consumes_linear_and_peeks = Arc::new(Mutex::new(HashSet::new()));
        let consumes_persistent = Arc::new(Mutex::new(HashSet::new()));
        let consumes_produced = Arc::new(Mutex::new(HashSet::new()));

        // Pre-process events to collect all produces and consumes
        // This allows us to clone each event at most once
        let mut all_produces: Vec<(Produce, bool, bool)> = Vec::new(); // (produce, exists_in_pre_state, touches_join)
        let mut all_consumes: Vec<Consume> = Vec::new();
        let mut all_comms: Vec<(Consume, Vec<Produce>, bool)> = Vec::new(); // (consume, produces, has_peeks)

        // First, gather all events to avoid locking overhead in the parallel section
        for event in &event_log {
            match event {
                Event::IoEvent(IOEvent::Produce(p)) => {
                    let exists = produce_exists_in_pre_state(p);
                    let touches_join = produce_touch_pre_state_join(p);
                    all_produces.push((p.clone(), exists, touches_join));
                }
                Event::IoEvent(IOEvent::Consume(c)) => {
                    all_consumes.push(c.clone());
                }
                Event::Comm(comm) => {
                    let has_peeks = !comm.peeks.is_empty();
                    all_comms.push((comm.consume.clone(), comm.produces.clone(), has_peeks));
                }
            }
        }

        // Process the collected events in parallel
        // Produces
        all_produces
            .par_iter()
            .for_each(|(p, exists, touches_join)| {
                if *exists {
                    if let Ok(mut set) = produces_copied_by_peek.lock() {
                        set.insert(p.clone());
                    }
                }

                if *touches_join {
                    if let Ok(mut set) = produces_touching_base_joins.lock() {
                        set.insert(p.clone());
                    }
                }

                if let Ok(mut set) = if p.persistent {
                    produces_persistent.lock()
                } else {
                    produces_linear.lock()
                } {
                    set.insert(p.clone());
                }
            });

        // Consumes
        all_consumes.par_iter().for_each(|c| {
            if let Ok(mut set) = if c.persistent {
                consumes_persistent.lock()
            } else {
                consumes_linear_and_peeks.lock()
            } {
                set.insert(c.clone());
            }
        });

        // COMM events
        all_comms
            .par_iter()
            .for_each(|(consume, produces, has_peeks)| {
                if let Ok(mut set) = consumes_produced.lock() {
                    set.insert(consume.clone());
                }

                let target_set = if *has_peeks {
                    &produces_peeked
                } else {
                    &produces_consumed
                };

                for p in produces {
                    if let Ok(mut set) = target_set.lock() {
                        set.insert(p.clone());
                    }
                }
            });

        // Helper function to safely unwrap Arc<Mutex<HashSet<T>>> with minimal cloning
        fn unwrap_arc_mutex<T>(arc_mutex: Arc<Mutex<HashSet<T>>>) -> HashableSet<T>
        where
            T: Eq + std::hash::Hash + Clone,
        {
            // Try to get exclusive ownership of the Arc
            match Arc::try_unwrap(arc_mutex) {
                // Success case: we have exclusive ownership, just unwrap the mutex
                Ok(mutex) => HashableSet(mutex.into_inner().unwrap_or_default()),

                // Can't get exclusive ownership - we need to construct a new set
                // with minimal cloning by draining items instead of cloning the whole set
                Err(arc) => {
                    let mut result = HashSet::new();
                    if let Ok(guard) = arc.lock() {
                        // Insert each item individually - still requires cloning elements
                        // but avoids cloning the entire collection structure
                        for item in guard.iter() {
                            result.insert(item.clone());
                        }
                    }
                    HashableSet(result)
                }
            }
        }

        // Unwrap the Arc<Mutex<>> to get the final collections
        let produces_linear = unwrap_arc_mutex(produces_linear);
        let produces_persistent = unwrap_arc_mutex(produces_persistent);
        let produces_consumed = unwrap_arc_mutex(produces_consumed);
        let produces_peeked = unwrap_arc_mutex(produces_peeked);
        let produces_copied_by_peek = unwrap_arc_mutex(produces_copied_by_peek);
        let produces_touching_base_joins = unwrap_arc_mutex(produces_touching_base_joins);
        let consumes_linear_and_peeks = unwrap_arc_mutex(consumes_linear_and_peeks);
        let consumes_persistent = unwrap_arc_mutex(consumes_persistent);
        let consumes_produced = unwrap_arc_mutex(consumes_produced);

        // Calculate mergeable channels more efficiently
        // First, create a HashSet for efficient lookups
        let all_produces: HashSet<Produce> = produces_linear
            .0
            .iter()
            .chain(produces_persistent.0.iter())
            .chain(produces_consumed.0.iter())
            .chain(produces_peeked.0.iter())
            .cloned()
            .collect();

        // Then filter and clone only once for the final set
        let produces_mergeable = HashableSet(
            all_produces
                .into_iter()
                .filter(|p| mergeable_chs.contains_key(&p.channel_hash))
                .collect(),
        );

        // Same approach for consumes
        let all_consumes: HashSet<Consume> = consumes_linear_and_peeks
            .0
            .iter()
            .chain(consumes_persistent.0.iter())
            .chain(consumes_produced.0.iter())
            .cloned()
            .collect();

        let consumes_mergeable = HashableSet(
            all_consumes
                .into_iter()
                .filter(|c| {
                    c.channel_hashes
                        .iter()
                        .any(|hash| mergeable_chs.contains_key(hash))
                })
                .collect(),
        );

        EventLogIndex {
            produces_linear,
            produces_persistent,
            produces_consumed,
            produces_peeked,
            produces_copied_by_peek,
            produces_touching_base_joins,
            consumes_linear_and_peeks,
            consumes_persistent,
            consumes_produced,
            produces_mergeable,
            consumes_mergeable,
            number_channels_data: mergeable_chs,
        }
    }

    pub fn empty() -> Self {
        EventLogIndex {
            produces_linear: HashableSet(HashSet::new()),
            produces_persistent: HashableSet(HashSet::new()),
            produces_consumed: HashableSet(HashSet::new()),
            produces_peeked: HashableSet(HashSet::new()),
            produces_copied_by_peek: HashableSet(HashSet::new()),
            produces_touching_base_joins: HashableSet(HashSet::new()),
            consumes_linear_and_peeks: HashableSet(HashSet::new()),
            consumes_persistent: HashableSet(HashSet::new()),
            consumes_produced: HashableSet(HashSet::new()),
            produces_mergeable: HashableSet(HashSet::new()),
            consumes_mergeable: HashableSet(HashSet::new()),
            number_channels_data: NumberChannelsDiff::new(),
        }
    }

    pub fn combine(x: &Self, y: &Self) -> Self {
        EventLogIndex {
            produces_linear: HashableSet(
                x.produces_linear
                    .0
                    .union(&y.produces_linear.0)
                    .cloned()
                    .collect(),
            ),
            produces_persistent: HashableSet(
                x.produces_persistent
                    .0
                    .union(&y.produces_persistent.0)
                    .cloned()
                    .collect(),
            ),
            produces_consumed: HashableSet(
                x.produces_consumed
                    .0
                    .union(&y.produces_consumed.0)
                    .cloned()
                    .collect(),
            ),
            produces_peeked: HashableSet(
                x.produces_peeked
                    .0
                    .union(&y.produces_peeked.0)
                    .cloned()
                    .collect(),
            ),
            produces_copied_by_peek: combine_produces_copied_by_peek(&x, &y),
            //TODO this joins combination is very restrictive. Join might be originated inside aggregated event log - OLD
            produces_touching_base_joins: HashableSet(
                x.produces_touching_base_joins
                    .0
                    .union(&y.produces_touching_base_joins.0)
                    .cloned()
                    .collect(),
            ),
            consumes_linear_and_peeks: HashableSet(
                x.consumes_linear_and_peeks
                    .0
                    .union(&y.consumes_linear_and_peeks.0)
                    .cloned()
                    .collect(),
            ),
            consumes_persistent: HashableSet(
                x.consumes_persistent
                    .0
                    .union(&y.consumes_persistent.0)
                    .cloned()
                    .collect(),
            ),
            consumes_produced: HashableSet(
                x.consumes_produced
                    .0
                    .union(&y.consumes_produced.0)
                    .cloned()
                    .collect(),
            ),
            // Combine mergeable produces and consumes
            produces_mergeable: HashableSet(
                x.produces_mergeable
                    .0
                    .union(&y.produces_mergeable.0)
                    .cloned()
                    .collect(),
            ),
            consumes_mergeable: HashableSet(
                x.consumes_mergeable
                    .0
                    .union(&y.consumes_mergeable.0)
                    .cloned()
                    .collect(),
            ),
            // Merge number channels (add differences)
            number_channels_data: x
                .number_channels_data
                .iter()
                .chain(y.number_channels_data.iter())
                .fold(NumberChannelsDiff::new(), |mut acc, (key, value)| {
                    // Sum the values if key exists, otherwise just insert
                    acc.entry(key.clone())
                        .and_modify(|existing| *existing += value)
                        .or_insert(*value);
                    acc
                }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
    use std::collections::BTreeMap;

    /// Create a 32-byte Blake2b256Hash filled with the given byte value.
    fn mk_hash(byte: u8) -> Blake2b256Hash {
        Blake2b256Hash::from_bytes(vec![byte; 32])
    }

    /// Helper: create an empty EventLogIndex with a specific number_channels_data map.
    fn empty_with_channels(data: BTreeMap<Blake2b256Hash, i64>) -> EventLogIndex {
        let mut eli = EventLogIndex::empty();
        eli.number_channels_data = data;
        eli
    }

    #[test]
    fn ordering_distinguishes_different_number_channels_data_keys() {
        let a = empty_with_channels(BTreeMap::from([(mk_hash(1), 100i64)]));
        let b = empty_with_channels(BTreeMap::from([(mk_hash(2), 100i64)]));
        assert_ne!(a.cmp(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn ordering_distinguishes_same_keys_different_values() {
        let a = empty_with_channels(BTreeMap::from([(mk_hash(1), 100i64)]));
        let b = empty_with_channels(BTreeMap::from([(mk_hash(1), 200i64)]));
        assert_ne!(a.cmp(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn ordering_compares_equal_for_identical_instances() {
        let a = empty_with_channels(BTreeMap::from([(mk_hash(1), 100i64)]));
        let b = empty_with_channels(BTreeMap::from([(mk_hash(1), 100i64)]));
        assert_eq!(a.cmp(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn ordering_distinguishes_by_entry_count() {
        let a = empty_with_channels(BTreeMap::from([(mk_hash(1), 100i64)]));
        let b = empty_with_channels(BTreeMap::from([(mk_hash(1), 100i64), (mk_hash(2), 200i64)]));
        assert_ne!(a.cmp(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn ordering_is_consistent_and_antisymmetric() {
        let a = empty_with_channels(BTreeMap::from([(mk_hash(1), 100i64)]));
        let b = empty_with_channels(BTreeMap::from([(mk_hash(2), 50i64)]));

        let result1 = a.cmp(&b);
        let result2 = a.cmp(&b);
        assert_eq!(result1, result2, "ordering must be consistent across calls");

        // Antisymmetry: compare(a,b) == reverse of compare(b,a)
        assert_eq!(b.cmp(&a), result1.reverse(), "ordering must be antisymmetric");
    }
}
