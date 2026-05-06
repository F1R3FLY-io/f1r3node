// See rspace/src/main/scala/coop/rchain/rspace/merger/EventLogIndex.scala

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use shared::rust::hashable_set::HashableSet;

use super::merging_logic::{
    NumberChannelsDiff, combine_mergeable_value, combine_produces_copied_by_peek,
};
use crate::rspace::errors::HistoryError;
use crate::rspace::trace::event::{Consume, Event, IOEvent, Produce};

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
// with fallback to produce/consume counts to distinguish structurally different
// indices. This replaces a previous derived Ord that was susceptible to
// non-deterministic HashSet iteration order, where two different EventLogIndex
// instances with different numberChannelsData could compare inconsistently.
impl PartialOrd for EventLogIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
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

        // Compare entries lexicographically: first by key (Blake2b256Hash), then by
        // value (i64)
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
        let a_prod = self.produces_linear.0.len() +
            self.produces_persistent.0.len() +
            self.produces_consumed.0.len();
        let b_prod = other.produces_linear.0.len() +
            other.produces_persistent.0.len() +
            other.produces_consumed.0.len();
        let prod_cmp = a_prod.cmp(&b_prod);
        if prod_cmp != std::cmp::Ordering::Equal {
            return prod_cmp;
        }

        let a_cons = self.consumes_linear_and_peeks.0.len() +
            self.consumes_persistent.0.len() +
            self.consumes_produced.0.len();
        let b_cons = other.consumes_linear_and_peeks.0.len() +
            other.consumes_persistent.0.len() +
            other.consumes_produced.0.len();
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
        where T: Eq + std::hash::Hash + Clone {
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

    pub fn combine(x: &Self, y: &Self) -> Result<Self, HistoryError> {
        // Merge number channels (combine differences according to per-channel
        // merge strategy: IntegerAdd uses wrapping addition, BitmaskOr uses
        // bitwise OR through u64). Both branches must agree on merge_type for
        // a given channel; disagreement yields a tagged error so callers can
        // reject the merge instead of crashing the validator.
        let mut number_channels_data = NumberChannelsDiff::new();
        for (key, value) in x
            .number_channels_data
            .iter()
            .chain(y.number_channels_data.iter())
        {
            let (incoming_diff, incoming_mt) = *value;
            match number_channels_data.get_mut(key) {
                Some(existing) => {
                    if existing.1 != incoming_mt {
                        return Err(HistoryError::MergeError(format!(
                            "MergeType mismatch on channel {:?}: {:?} vs {:?}",
                            key, existing.1, incoming_mt,
                        )));
                    }
                    existing.0 = combine_mergeable_value(existing.0, incoming_diff, incoming_mt);
                }
                None => {
                    number_channels_data.insert(key.clone(), (incoming_diff, incoming_mt));
                }
            }
        }

        Ok(EventLogIndex {
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
            //TODO this joins combination is very restrictive. Join might be originated inside
            // aggregated event log - OLD
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
            number_channels_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;

    /// Create a 32-byte Blake2b256Hash filled with the given byte value.
    fn mk_hash(byte: u8) -> Blake2b256Hash { Blake2b256Hash::from_bytes(vec![byte; 32]) }

    /// Helper: create an empty EventLogIndex with a specific
    /// number_channels_data map. All entries default to IntegerAdd semantics.
    fn empty_with_channels(data: BTreeMap<Blake2b256Hash, i64>) -> EventLogIndex {
        let mut eli = EventLogIndex::empty();
        eli.number_channels_data = data
            .into_iter()
            .map(|(k, v)| (k, (v, super::super::merging_logic::MergeType::IntegerAdd)))
            .collect();
        eli
    }

    /// Helper: same as `empty_with_channels` but tags every entry with
    /// `BitmaskOr` semantics — used by the bitmap-monotonicity tests.
    fn empty_with_bitmask_channels(data: BTreeMap<Blake2b256Hash, i64>) -> EventLogIndex {
        let mut eli = EventLogIndex::empty();
        eli.number_channels_data = data
            .into_iter()
            .map(|(k, v)| (k, (v, super::super::merging_logic::MergeType::BitmaskOr)))
            .collect();
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

    // --- BitmaskOr merger property tests ----------------------------------
    //
    // When two event-log indices touch the same `BitmaskOr` channel, the
    // combined diff must be the bitwise OR of both inputs. This is what
    // allows concurrent registry inserts at different keys that share an
    // interior node to merge without rejection. Registry.rho only adds
    // bits at interior nodes; these tests verify the merger faithfully
    // aggregates those additions.

    #[test]
    fn combine_or_folds_two_bitmask_diffs_on_same_channel() {
        // Two diffs setting different bits on the same interior channel —
        // the merger must produce the bitwise OR.
        let ch = mk_hash(42);
        let a = empty_with_bitmask_channels(BTreeMap::from([(ch.clone(), 0b00010001)]));
        let b = empty_with_bitmask_channels(BTreeMap::from([(ch.clone(), 0b00100010)]));
        let combined = EventLogIndex::combine(&a, &b).expect("combine must not fail");
        let (val, mt) = combined.number_channels_data[&ch];
        assert_eq!(val, 0b00110011, "BitmaskOr must produce OR of both diffs, not max");
        assert_eq!(
            mt,
            super::super::merging_logic::MergeType::BitmaskOr,
            "merge type preserved through combine",
        );
    }

    #[test]
    fn combine_bitmask_is_commutative() {
        let ch = mk_hash(7);
        let a = empty_with_bitmask_channels(BTreeMap::from([(ch.clone(), 0b1010_1010)]));
        let b = empty_with_bitmask_channels(BTreeMap::from([(ch.clone(), 0b0101_0101)]));
        let ab = EventLogIndex::combine(&a, &b).unwrap();
        let ba = EventLogIndex::combine(&b, &a).unwrap();
        assert_eq!(
            ab.number_channels_data[&ch].0, ba.number_channels_data[&ch].0,
            "BitmaskOr combine must be commutative",
        );
    }

    #[test]
    fn combine_bitmask_is_associative_and_monotonic() {
        // Drive a sequence of N diffs (each setting a distinct bit on the
        // same channel) through combine. Assert: (1) the running combined
        // value is bitwise non-decreasing across the sequence — no bit ever
        // gets cleared; (2) reordering produces the same final value.
        let ch = mk_hash(99);
        let diffs: Vec<i64> = (0..16).map(|i| 1i64 << i).collect();

        let indices: Vec<EventLogIndex> = diffs
            .iter()
            .map(|d| empty_with_bitmask_channels(BTreeMap::from([(ch.clone(), *d)])))
            .collect();

        // Monotonicity: each successive combine never clears a previously-set bit.
        let mut acc = EventLogIndex::empty();
        let mut prev_val: u64 = 0;
        for index in &indices {
            acc = EventLogIndex::combine(&acc, index).unwrap();
            let cur_val = acc
                .number_channels_data
                .get(&ch)
                .map(|(v, _)| *v as u64)
                .unwrap_or(0);
            assert_eq!(
                cur_val & prev_val,
                prev_val,
                "monotonicity violated: a previously-set bit was cleared",
            );
            prev_val = cur_val;
        }
        // After all 16 diffs, every low bit must be set.
        assert_eq!(prev_val, 0xFFFF, "all 16 bits must be present after combining 16 diffs");

        // Reordering should produce the same final value (commutativity at scale).
        let mut reversed = indices.clone();
        reversed.reverse();
        let folded_reverse = reversed
            .iter()
            .fold(EventLogIndex::empty(), |a, b| EventLogIndex::combine(&a, b).unwrap());
        assert_eq!(
            acc.number_channels_data[&ch].0, folded_reverse.number_channels_data[&ch].0,
            "BitmaskOr combine fold must be order-independent",
        );
    }

    #[test]
    fn combine_bitmask_is_idempotent_on_same_index() {
        // Combining an index with itself must not change its bitmap value.
        let ch = mk_hash(33);
        let a = empty_with_bitmask_channels(BTreeMap::from([(ch.clone(), 0b1100_1100)]));
        let aa = EventLogIndex::combine(&a, &a).unwrap();
        assert_eq!(
            aa.number_channels_data[&ch].0, a.number_channels_data[&ch].0,
            "BitmaskOr combine must be idempotent",
        );
    }

    #[test]
    fn combine_returns_err_on_mergetype_mismatch() {
        // Same channel hash with disagreeing MergeType in two indices must
        // yield Err, not panic, not silent pick-one.
        let ch = mk_hash(55);
        let a = empty_with_channels(BTreeMap::from([(ch.clone(), 5i64)])); // IntegerAdd
        let b = empty_with_bitmask_channels(BTreeMap::from([(ch.clone(), 5i64)])); // BitmaskOr
        let result = EventLogIndex::combine(&a, &b);
        assert!(result.is_err(), "MergeType mismatch on the same channel must produce Err",);
    }
}
