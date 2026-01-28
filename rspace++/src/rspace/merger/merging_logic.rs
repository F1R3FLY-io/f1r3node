// See rspace/src/main/scala/coop/rchain/rspace/merger/MergingLogic.scala
// See rspace/src/test/scala/coop/rchain/rspace/merging/MergingLogicSpec.scala

use shared::rust::hashable_set::HashableSet;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    trace::event::{Consume, Produce},
};

use super::event_log_index::EventLogIndex;

pub type NumberChannelsEndVal = BTreeMap<Blake2b256Hash, i64>;

pub type NumberChannelsDiff = BTreeMap<Blake2b256Hash, i64>;

/// If target depends on source.
pub fn depends(target: &EventLogIndex, source: &EventLogIndex) -> bool {
    let produces_source: HashableSet<Produce> = HashableSet(
        produces_created_and_not_destroyed(source)
            .0
            .difference(&source.produces_mergeable.0)
            .cloned()
            .collect(),
    );

    let produces_target: HashableSet<Produce> = HashableSet(
        target
            .produces_consumed
            .0
            .difference(&source.produces_mergeable.0)
            .cloned()
            .collect(),
    );

    let consumes_source = consumes_created_and_not_destroyed(source);
    let consumes_target = &target.consumes_produced;

    let produces_depends: HashableSet<Produce> = HashableSet(
        produces_source
            .0
            .intersection(&produces_target.0)
            .cloned()
            .collect(),
    );

    let consumes_depends: HashableSet<Consume> = HashableSet(
        consumes_source
            .0
            .intersection(&consumes_target.0)
            .cloned()
            .collect(),
    );

    !produces_depends.0.is_empty() || !consumes_depends.0.is_empty()
}

/// If two event logs are conflicting.
pub fn are_conflicting(a: &EventLogIndex, b: &EventLogIndex) -> bool {
    conflicts(a, b).0.is_empty() == false
}

/// Debug version that returns the reason for conflict.
/// Returns None if no conflict, or Some(reason_string) if there is a conflict.
pub fn conflict_reason(a: &EventLogIndex, b: &EventLogIndex) -> Option<String> {
    // Check #1: Races for same IO event
    let races_for_same_io_event = {
        let shared_consumes: HashableSet<Consume> = HashableSet(
            a.consumes_produced
                .0
                .intersection(&b.consumes_produced.0)
                .cloned()
                .collect(),
        );
        let mergeable_consumes: HashableSet<Consume> = HashableSet(
            a.consumes_mergeable
                .0
                .intersection(&b.consumes_mergeable.0)
                .cloned()
                .collect(),
        );
        let consume_races: HashSet<Consume> = shared_consumes
            .0
            .difference(&mergeable_consumes.0)
            .filter(|c| !c.persistent)
            .cloned()
            .collect();

        let shared_produces: HashableSet<Produce> = HashableSet(
            a.produces_consumed
                .0
                .intersection(&b.produces_consumed.0)
                .cloned()
                .collect(),
        );
        let mergeable_produces: HashableSet<Produce> = HashableSet(
            a.produces_mergeable
                .0
                .intersection(&b.produces_mergeable.0)
                .cloned()
                .collect(),
        );
        let produce_races: HashSet<Produce> = shared_produces
            .0
            .difference(&mergeable_produces.0)
            .filter(|p| !p.persistent)
            .cloned()
            .collect();

        match (consume_races.is_empty(), produce_races.is_empty()) {
            (false, false) => Some(format!(
                "racesForSameIOEvent: consumeRaces={}, produceRaces={}",
                consume_races.len(),
                produce_races.len()
            )),
            (false, true) => Some(format!("racesForSameIOEvent: consumeRaces={}", consume_races.len())),
            (true, false) => Some(format!("racesForSameIOEvent: produceRaces={}", produce_races.len())),
            (true, true) => None,
        }
    };

    // Check #2: Potential COMMs
    let potential_comms = || {
        fn match_found(consume: &Consume, produce: &Produce) -> bool {
            consume.channel_hashes.contains(&produce.channel_hash)
        }

        fn check(left: &EventLogIndex, right: &EventLogIndex) -> usize {
            let p = produces_created_and_not_destroyed(left);
            let c = consumes_created_and_not_destroyed(right);
            let mut count = 0;
            for produce in &p.0 {
                for consume in &c.0 {
                    if match_found(consume, produce) {
                        count += 1;
                    }
                }
            }
            count
        }

        let a_to_b = check(a, b);
        let b_to_a = check(b, a);
        if a_to_b > 0 || b_to_a > 0 {
            Some(format!("potentialCOMMs: a->b={}, b->a={}", a_to_b, b_to_a))
        } else {
            None
        }
    };

    // Check #3: Produce touch base join
    let produce_touch_base_join = || {
        let count = a.produces_touching_base_joins.0.len() + b.produces_touching_base_joins.0.len();
        if count > 0 {
            Some(format!("produceTouchBaseJoin: count={}", count))
        } else {
            None
        }
    };

    races_for_same_io_event
        .or_else(potential_comms)
        .or_else(produce_touch_base_join)
}

/// Channels conflicting between a pair of event logs.
pub fn conflicts(a: &EventLogIndex, b: &EventLogIndex) -> HashableSet<Blake2b256Hash> {
    // Check #1
    // If the same produce or consume is destroyed in COMM in both branches, this might be a race.
    // All events created in event logs are unique, this match can be identified by comparing case classes.
    //
    // Produce is considered destroyed in COMM if it is not persistent and been consumed without peek.
    // Consume is considered destroyed in COMM when it is not persistent.
    //
    // If produces/consumes are mergeable in both indices, they are not considered as conflicts.
    let races_for_same_io_event = {
        let shared_consumes: HashableSet<Consume> = HashableSet(
            a.consumes_produced
                .0
                .intersection(&b.consumes_produced.0)
                .cloned()
                .collect(),
        );
        let mergeable_consumes: HashableSet<Consume> = HashableSet(
            a.consumes_mergeable
                .0
                .intersection(&b.consumes_mergeable.0)
                .cloned()
                .collect(),
        );
        let consume_races: HashSet<Consume> = shared_consumes
            .0
            .difference(&mergeable_consumes.0)
            .filter(|c| !c.persistent)
            .cloned()
            .collect();

        let shared_produces: HashableSet<Produce> = HashableSet(
            a.produces_consumed
                .0
                .intersection(&b.produces_consumed.0)
                .cloned()
                .collect(),
        );
        let mergeable_produces: HashableSet<Produce> = HashableSet(
            a.produces_mergeable
                .0
                .intersection(&b.produces_mergeable.0)
                .cloned()
                .collect(),
        );
        let produce_races: HashSet<Produce> = shared_produces
            .0
            .difference(&mergeable_produces.0)
            .filter(|p| !p.persistent)
            .cloned()
            .collect();

        let mut result = HashSet::new();
        for consume in consume_races {
            result.extend(consume.channel_hashes.iter().cloned());
        }
        for produce in produce_races {
            result.insert(produce.channel_hash.clone());
        }
        result
    };

    // Check #2
    // Events that are created inside branch and has not been destroyed in branch's COMMs
    // can lead to potential COMM during merge.
    let potential_comms = {
        // TODO analyze joins to make less conflicts. Now plain channel intersection treated as a conflict - OLD
        fn match_found(consume: &Consume, produce: &Produce) -> bool {
            consume.channel_hashes.contains(&produce.channel_hash)
        }

        // Search for match in both directions
        fn check(left: &EventLogIndex, right: &EventLogIndex) -> HashSet<Blake2b256Hash> {
            let p = produces_created_and_not_destroyed(left);
            let c = consumes_created_and_not_destroyed(right);
            let mut result = HashSet::new();
            for produce in &p.0 {
                for consume in &c.0 {
                    if match_found(consume, produce) {
                        result.insert(produce.channel_hash.clone());
                    }
                }
            }
            result
        }

        let mut result = check(a, b);
        result.extend(check(b, a));
        result
    };

    // Now we don't analyze joins and declare conflicting cases when produce touch join because applying
    // produces from both event logs might trigger continuation of some join, so COMM event
    let produce_touch_base_join = {
        let mut result = HashSet::new();
        for produce in a
            .produces_touching_base_joins
            .0
            .iter()
            .chain(b.produces_touching_base_joins.0.iter())
        {
            result.insert(produce.channel_hash.clone());
        }
        result
    };

    // Combine all conflicts
    let mut all_conflicts = HashSet::new();
    all_conflicts.extend(races_for_same_io_event);
    all_conflicts.extend(potential_comms);
    all_conflicts.extend(produce_touch_base_join);
    HashableSet(all_conflicts)
}

/// Produce created inside event log.
pub fn produces_created(e: &EventLogIndex) -> HashableSet<Produce> {
    let mut result: HashSet<Produce> = e
        .produces_linear
        .0
        .union(&e.produces_persistent.0)
        .cloned()
        .collect();

    for produce in &e.produces_copied_by_peek.0 {
        result.remove(produce);
    }
    HashableSet(result)
}

/// Consume created inside event log.
pub fn consumes_created(e: &EventLogIndex) -> HashableSet<Consume> {
    HashableSet(
        e.consumes_linear_and_peeks
            .0
            .union(&e.consumes_persistent.0)
            .cloned()
            .collect(),
    )
}

/// Produces that are created inside event log and not destroyed via COMM inside event log.
pub fn produces_created_and_not_destroyed(e: &EventLogIndex) -> HashableSet<Produce> {
    let linear_not_consumed: HashSet<Produce> = e
        .produces_linear
        .0
        .difference(&e.produces_consumed.0)
        .cloned()
        .collect();
    let combined: HashSet<Produce> = linear_not_consumed
        .union(&e.produces_persistent.0)
        .cloned()
        .collect();

    HashableSet(
        combined
            .difference(&e.produces_copied_by_peek.0)
            .cloned()
            .collect(),
    )
}

/// Consumes that are created inside event log and not destroyed via COMM inside event log.
pub fn consumes_created_and_not_destroyed(e: &EventLogIndex) -> HashableSet<Consume> {
    let linear_not_produced: HashSet<Consume> = e
        .consumes_linear_and_peeks
        .0
        .difference(&e.consumes_produced.0)
        .cloned()
        .collect();

    HashableSet(
        linear_not_produced
            .union(&e.consumes_persistent.0)
            .cloned()
            .collect(),
    )
}

/// Produces that are affected by event log - locally created + external destroyed.
pub fn produces_affected(e: &EventLogIndex) -> HashableSet<Produce> {
    let created = produces_created(e);
    let external_produces_destroyed: HashableSet<Produce> = HashableSet(
        e.produces_consumed
            .0
            .difference(&created.0)
            .filter(|p| !p.persistent)
            .cloned()
            .collect(),
    );

    HashableSet(
        produces_created_and_not_destroyed(e)
            .0
            .union(&external_produces_destroyed.0)
            .cloned()
            .collect(),
    )
}

/// Consumes that are affected by event log - locally created + external destroyed.
pub fn consumes_affected(e: &EventLogIndex) -> HashableSet<Consume> {
    let created = consumes_created(e);
    let external_consumes_destroyed: HashableSet<Consume> = HashableSet(
        e.consumes_produced
            .0
            .difference(&created.0)
            .filter(|c| !c.persistent)
            .cloned()
            .collect(),
    );

    HashableSet(
        consumes_created_and_not_destroyed(e)
            .0
            .union(&external_consumes_destroyed.0)
            .cloned()
            .collect(),
    )
}

/// If produce is copied by peek in one index and originated in another - it is considered as created in aggregate.
pub fn combine_produces_copied_by_peek(
    x: &EventLogIndex,
    y: &EventLogIndex,
) -> HashableSet<Produce> {
    let combined_copied_by_peek: HashableSet<Produce> = HashableSet(
        x.produces_copied_by_peek
            .0
            .union(&y.produces_copied_by_peek.0)
            .cloned()
            .collect(),
    );

    let combined_created: HashableSet<Produce> = HashableSet(
        produces_created(x)
            .0
            .union(&produces_created(y).0)
            .cloned()
            .collect(),
    );

    HashableSet(
        combined_copied_by_peek
            .0
            .difference(&combined_created.0)
            .cloned()
            .collect(),
    )
}

/// Arrange list[v] into map v -> Vec[v] for items that match predicate.
/// NOTE: predicate here is forced to be non directional.
/// If either (a,b) or (b,a) is true, both relations are recorded as true.
/// TODO: adjust once dependency graph is implemented for branch computing - OLD
pub fn compute_relation_map<A: Eq + std::hash::Hash + Clone + PartialOrd>(
    items: &HashableSet<A>,
    relation: impl Fn(&A, &A) -> bool,
) -> HashMap<A, HashableSet<A>> {
    let mut init: HashMap<A, HashableSet<A>> = items
        .0
        .iter()
        .map(|item| (item.clone(), HashableSet(HashSet::new())))
        .collect();

    for item1 in items.0.iter() {
        for item2 in items.0.iter() {
            // Skip self-comparisons and duplicated comparisons
            if std::ptr::eq(item1, item2) || item1 > item2 {
                continue;
            }

            if relation(item1, item2) || relation(item2, item1) {
                if let Some(set1) = init.get_mut(item1) {
                    set1.0.insert(item2.clone());
                }
                if let Some(set2) = init.get_mut(item2) {
                    set2.0.insert(item1.clone());
                }
            }
        }
    }

    init
}

/// Given relation map, return sets of related items.
pub fn gather_related_sets<A: Eq + std::hash::Hash + Clone>(
    relation_map: &HashMap<A, HashableSet<A>>,
) -> HashableSet<HashableSet<A>> {
    fn add_relations<A: Eq + std::hash::Hash + Clone>(
        to_add: &HashableSet<A>,
        acc: HashSet<A>, // Take ownership instead of reference
        relation_map: &HashMap<A, HashableSet<A>>,
    ) -> HashSet<A> {
        // Add all items to accumulator
        let mut next = acc;
        let initial_size = next.len();

        for item in &to_add.0 {
            next.insert(item.clone());
        }

        // Stop if no new items were added
        if next.len() == initial_size {
            return next;
        }

        // Find new related items
        let mut n = HashSet::new();
        for v in to_add.0.iter() {
            if let Some(related) = relation_map.get(v) {
                for r in &related.0 {
                    if !next.contains(r) {
                        // Only collect items not already in next
                        n.insert(r.clone());
                    }
                }
            }
        }

        if n.is_empty() {
            return next;
        }

        // Continue with new items
        add_relations(&HashableSet(n), next, relation_map)
    }

    // Use a more efficient way to track processed nodes
    let mut processed = HashSet::new();
    let mut result = HashSet::new();

    for k in relation_map.keys() {
        if processed.contains(k) {
            continue; // Skip already processed nodes
        }

        let mut start = HashSet::new();
        start.insert(k.clone());
        let component = add_relations(relation_map.get(k).unwrap(), start, relation_map);

        // Mark all nodes in this component as processed
        for item in &component {
            processed.insert(item.clone());
        }

        result.insert(HashableSet(component));
    }

    HashableSet(result)
}

/// Compute related sets directly from items and relation
pub fn compute_related_sets<A: Eq + std::hash::Hash + Clone + PartialOrd>(
    items: &HashableSet<A>,
    relation: impl Fn(&A, &A) -> bool,
) -> HashableSet<HashableSet<A>> {
    let relation_map = compute_relation_map(items, relation);
    gather_related_sets(&relation_map)
}

/// Given conflicts map, output possible rejection options.
pub fn compute_rejection_options<A: Eq + std::hash::Hash + Clone>(
    conflict_map: &HashMap<A, HashableSet<A>>,
) -> HashableSet<HashableSet<A>> {
    // Set of rejection paths with corresponding remaining conflicts map
    #[derive(Clone)]
    struct RejectionOption<A: Eq + std::hash::Hash + Clone> {
        rejected_so_far: HashableSet<A>,
        remaining_conflicts_map: HashMap<A, HashableSet<A>>,
    }

    impl<A: Eq + std::hash::Hash + Clone> PartialEq for RejectionOption<A> {
        fn eq(&self, other: &Self) -> bool {
            // Only compare rejected_so_far for equality
            self.rejected_so_far == other.rejected_so_far
        }
    }

    impl<A: Eq + std::hash::Hash + Clone> Eq for RejectionOption<A> {}

    impl<A: Eq + std::hash::Hash + Clone> std::hash::Hash for RejectionOption<A> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            // Only hash rejected_so_far
            self.rejected_so_far.hash(state);
        }
    }

    fn gather_rej_options<A: Eq + std::hash::Hash + Clone>(
        conflicts_map: &HashMap<A, HashableSet<A>>,
    ) -> HashSet<RejectionOption<A>> {
        let mut result = HashSet::new();

        for (_, to_reject) in conflicts_map {
            // keeping each key - reject conflicting values
            let mut remaining_conflicts_map = HashMap::new();

            for (key, conflicts) in conflicts_map {
                if !to_reject.0.contains(key) {
                    // Filter out rejected items from conflicts
                    let updated_conflicts: HashSet<A> = conflicts
                        .0
                        .iter()
                        .filter(|c| !to_reject.0.contains(c))
                        .cloned()
                        .collect();

                    // Only include keys that still have conflicts
                    if !updated_conflicts.is_empty() {
                        remaining_conflicts_map.insert(key.clone(), HashableSet(updated_conflicts));
                    }
                }
            }

            let option = RejectionOption {
                rejected_so_far: to_reject.clone(),
                remaining_conflicts_map,
            };

            result.insert(option);
        }

        result
    }

    // Only keys that have conflicts associated should be examined
    let mut conflicts_only_map = HashMap::new();
    for (key, conflicts) in conflict_map {
        if !conflicts.0.is_empty() {
            conflicts_only_map.insert(key.clone(), conflicts.clone());
        }
    }

    // Start with rejecting nothing and full conflicts map
    let start = RejectionOption {
        rejected_so_far: HashableSet(HashSet::new()),
        remaining_conflicts_map: conflicts_only_map,
    };

    let mut result = HashSet::new();
    let mut current = {
        let mut set = HashSet::new();
        set.insert(start);
        set
    };

    while !current.is_empty() {
        let mut next = HashSet::new();

        for option in current {
            if option.remaining_conflicts_map.is_empty() {
                // No more conflicts, this is a valid rejection option
                // Only add if not already in result
                let already_exists = result.iter().any(|existing_set: &HashableSet<A>| {
                    existing_set.0.len() == option.rejected_so_far.0.len()
                        && existing_set
                            .0
                            .iter()
                            .all(|item| option.rejected_so_far.0.contains(item))
                });

                if !already_exists {
                    result.insert(option.rejected_so_far);
                }
            } else {
                // Continue resolving conflicts
                for mut new_option in gather_rej_options(&option.remaining_conflicts_map) {
                    // Add previously rejected items
                    for item in &option.rejected_so_far.0 {
                        new_option.rejected_so_far.0.insert(item.clone());
                    }
                    next.insert(new_option);
                }
            }
        }

        current = next;
    }

    HashableSet(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;

    #[test]
    fn test_compute_rejection_options() {
        // Test 1
        let mut map1: HashMap<i32, HashableSet<i32>> = HashMap::new();
        map1.insert(1, HashableSet(HashSet::from_iter(vec![2, 3, 4])));
        map1.insert(2, HashableSet(HashSet::from_iter(vec![1])));
        map1.insert(3, HashableSet(HashSet::from_iter(vec![1, 2])));
        map1.insert(4, HashableSet(HashSet::from_iter(vec![1])));

        let result1 = compute_rejection_options(&map1);
        assert_eq!(result1.0.len(), 2);
        assert!(
            result1
                .0
                .iter()
                .any(|set| set.0.len() == 2 && set.0.contains(&1) && set.0.contains(&2))
        );
        assert!(result1.0.iter().any(|set| set.0.len() == 3
            && set.0.contains(&2)
            && set.0.contains(&3)
            && set.0.contains(&4)));

        // Test 2
        let mut map2: HashMap<i32, HashableSet<i32>> = HashMap::new();
        map2.insert(1, HashableSet(HashSet::from_iter(vec![2, 3, 4])));
        map2.insert(2, HashableSet(HashSet::from_iter(vec![1, 3, 4])));
        map2.insert(3, HashableSet(HashSet::from_iter(vec![1, 2, 4])));
        map2.insert(4, HashableSet(HashSet::from_iter(vec![1, 2, 3])));

        let result2 = compute_rejection_options(&map2);
        assert_eq!(result2.0.len(), 4);
        assert!(result2.0.iter().any(|set| set.0.len() == 3
            && set.0.contains(&2)
            && set.0.contains(&3)
            && set.0.contains(&4)));
        assert!(result2.0.iter().any(|set| set.0.len() == 3
            && set.0.contains(&1)
            && set.0.contains(&3)
            && set.0.contains(&4)));
        assert!(result2.0.iter().any(|set| set.0.len() == 3
            && set.0.contains(&1)
            && set.0.contains(&2)
            && set.0.contains(&4)));
        assert!(result2.0.iter().any(|set| set.0.len() == 3
            && set.0.contains(&1)
            && set.0.contains(&2)
            && set.0.contains(&3)));

        // Test 3
        let mut map3: HashMap<i32, HashableSet<i32>> = HashMap::new();
        map3.insert(1, HashableSet(HashSet::from_iter(vec![2, 3, 4])));
        map3.insert(2, HashableSet(HashSet::from_iter(vec![1])));
        map3.insert(3, HashableSet(HashSet::from_iter(vec![1, 4])));
        map3.insert(4, HashableSet(HashSet::from_iter(vec![1, 3])));

        let result3 = compute_rejection_options(&map3);
        assert_eq!(result3.0.len(), 3);
        assert!(result3.0.iter().any(|set| set.0.len() == 3
            && set.0.contains(&2)
            && set.0.contains(&3)
            && set.0.contains(&4)));
        assert!(
            result3
                .0
                .iter()
                .any(|set| set.0.len() == 2 && set.0.contains(&1) && set.0.contains(&3))
        );
        assert!(
            result3
                .0
                .iter()
                .any(|set| set.0.len() == 2 && set.0.contains(&1) && set.0.contains(&4))
        );

        // Test 4
        let mut map4: HashMap<i32, HashableSet<i32>> = HashMap::new();
        map4.insert(1, HashableSet(HashSet::new()));
        map4.insert(2, HashableSet(HashSet::from_iter(vec![3])));
        map4.insert(3, HashableSet(HashSet::from_iter(vec![2, 4])));
        map4.insert(4, HashableSet(HashSet::from_iter(vec![3])));

        let result4 = compute_rejection_options(&map4);
        assert_eq!(result4.0.len(), 2);
        assert!(
            result4
                .0
                .iter()
                .any(|set| set.0.len() == 1 && set.0.contains(&3))
        );
        assert!(
            result4
                .0
                .iter()
                .any(|set| set.0.len() == 2 && set.0.contains(&2) && set.0.contains(&4))
        );

        let all: HashSet<i32> = (1..=1000).collect();
        let mut map5: HashMap<i32, HashableSet<i32>> = HashMap::new();
        for i in 1..=1000 {
            let mut conflicts = all.clone();
            conflicts.remove(&i);
            map5.insert(i, HashableSet(conflicts));
        }

        let result5 = compute_rejection_options(&map5);
        assert_eq!(result5.0.len(), 1000);
        for i in 1..=1000 {
            let mut expected = all.clone();
            expected.remove(&i);
            assert!(result5.0.iter().any(|set| {
                set.0.len() == 999
                    && !set.0.contains(&i)
                    && (1..=1000).filter(|j| *j != i).all(|j| set.0.contains(&j))
            }));
        }
    }

    #[test]
    fn test_compute_related_sets() {
        // Test relation: numbers with the same parity (both odd or both even)
        let items: HashableSet<i32> = HashableSet(HashSet::from_iter(vec![1, 2, 3, 4, 5]));
        let same_parity = |a: &i32, b: &i32| a % 2 == b % 2;

        let result = compute_related_sets(&items, same_parity);
        assert_eq!(result.0.len(), 2);

        // Should have one set with odd numbers and one with even numbers
        let mut found_odd = false;
        let mut found_even = false;

        for set in &result.0 {
            if set.0.len() == 3 && set.0.contains(&1) && set.0.contains(&3) && set.0.contains(&5) {
                found_odd = true;
            }
            if set.0.len() == 2 && set.0.contains(&2) && set.0.contains(&4) {
                found_even = true;
            }
        }

        assert!(found_odd);
        assert!(found_even);
    }

    #[test]
    fn test_relation_map() {
        // Test creating relation map for divisibility relationship
        let items: HashableSet<i32> = HashableSet(HashSet::from_iter(vec![2, 3, 4, 6, 12]));
        let is_divisible = |a: &i32, b: &i32| b % a == 0;

        let relation_map = compute_relation_map(&items, is_divisible);

        // Check a few key relationships
        assert!(relation_map.get(&2).unwrap().0.contains(&4));
        assert!(relation_map.get(&2).unwrap().0.contains(&6));
        assert!(relation_map.get(&2).unwrap().0.contains(&12));

        assert!(relation_map.get(&3).unwrap().0.contains(&6));
        assert!(relation_map.get(&3).unwrap().0.contains(&12));

        assert!(!relation_map.get(&3).unwrap().0.contains(&4));
        assert!(!relation_map.get(&4).unwrap().0.contains(&6));
    }

    #[test]
    fn test_depends() {
        // Setup basic event indices
        let mut source = EventLogIndex::empty();
        let mut target = EventLogIndex::empty();

        // Create test data
        let channel_hash = Blake2b256Hash::from_bytes(vec![1]);

        let produce1 = Produce {
            channel_hash: channel_hash.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![10]),
            is_deterministic: true,
            output_value: vec![],
        };

        let consume1 = Consume {
            channel_hashes: vec![channel_hash.clone()].into_iter().collect(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![20]),
        };

        // Test 1: No dependency
        assert!(!depends(&target, &source));

        // Test 2: Dependency via produces
        source.produces_linear.0.insert(produce1.clone());
        target.produces_consumed.0.insert(produce1.clone());
        assert!(depends(&target, &source));

        // Test 3: No dependency when produce is mergeable
        let mut source2 = source.clone();
        source2.produces_mergeable.0.insert(produce1.clone());
        assert!(!depends(&target, &source2));

        // Test 4: Dependency via consumes
        let mut source3 = EventLogIndex::empty();
        let mut target3 = EventLogIndex::empty();
        source3.consumes_linear_and_peeks.0.insert(consume1.clone());
        target3.consumes_produced.0.insert(consume1.clone());
        assert!(depends(&target3, &source3));
    }

    #[test]
    fn test_conflicts_and_are_conflicting() {
        // Setup basic event indices
        let mut a = EventLogIndex::empty();
        let mut b = EventLogIndex::empty();

        // Create test channel hashes
        let ch1 = Blake2b256Hash::from_bytes(vec![1]);
        let ch2 = Blake2b256Hash::from_bytes(vec![2]);

        // Create test data
        let produce1 = Produce {
            channel_hash: ch1.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![10]),
            is_deterministic: true,
            output_value: vec![],
        };

        let produce2 = Produce {
            channel_hash: ch2.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![11]),
            is_deterministic: true,
            output_value: vec![],
        };

        // Test 1: No conflicts initially
        assert!(!are_conflicting(&a, &b));
        assert!(conflicts(&a, &b).0.is_empty());

        // Test 2: Race conflict (same produce consumed in both)
        a.produces_consumed.0.insert(produce1.clone());
        b.produces_consumed.0.insert(produce1.clone());
        assert!(are_conflicting(&a, &b));
        assert!(conflicts(&a, &b).0.contains(&ch1));

        // Test 3: No conflict when produce is mergeable
        a.produces_mergeable.0.insert(produce1.clone());
        b.produces_mergeable.0.insert(produce1.clone());
        assert!(!are_conflicting(&a, &b));

        // Test 4: Potential COMM conflict
        let mut a2 = EventLogIndex::empty();
        let mut b2 = EventLogIndex::empty();
        a2.produces_linear.0.insert(produce2.clone());

        // Create a consume that includes ch2 in its channel_hashes
        let consume_for_comm = Consume {
            channel_hashes: vec![ch1.clone(), ch2.clone()].into_iter().collect(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![22]),
        };
        b2.consumes_linear_and_peeks.0.insert(consume_for_comm);
        assert!(are_conflicting(&a2, &b2));

        // Test 5: Conflict with produce touching base join
        let mut a3 = EventLogIndex::empty();
        let b3 = EventLogIndex::empty();
        a3.produces_touching_base_joins.0.insert(produce1.clone());
        assert!(are_conflicting(&a3, &b3));
    }

    #[test]
    fn test_produces_and_consumes_created() {
        let mut e = EventLogIndex::empty();

        // Create test data
        let ch = Blake2b256Hash::from_bytes(vec![1]);

        let produce_linear = Produce {
            channel_hash: ch.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![10]),
            is_deterministic: true,
            output_value: vec![],
        };

        let produce_persistent = Produce {
            channel_hash: ch.clone(),
            persistent: true,
            hash: Blake2b256Hash::from_bytes(vec![11]),
            is_deterministic: true,
            output_value: vec![],
        };

        let produce_peek = Produce {
            channel_hash: ch.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![12]),
            is_deterministic: true,
            output_value: vec![],
        };

        let consume = Consume {
            channel_hashes: vec![ch.clone()].into_iter().collect(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![20]),
        };

        // Test empty case
        assert!(produces_created(&e).0.is_empty());
        assert!(consumes_created(&e).0.is_empty());

        // Add data and test
        e.produces_linear.0.insert(produce_linear.clone());
        e.produces_persistent.0.insert(produce_persistent.clone());
        e.produces_copied_by_peek.0.insert(produce_peek.clone());
        e.consumes_linear_and_peeks.0.insert(consume.clone());

        // Check produces_created
        let created = produces_created(&e);
        assert_eq!(created.0.len(), 2);
        assert!(created.0.contains(&produce_linear));
        assert!(created.0.contains(&produce_persistent));
        assert!(!created.0.contains(&produce_peek));

        // Check consumes_created
        let created_consumes = consumes_created(&e);
        assert_eq!(created_consumes.0.len(), 1);
        assert!(created_consumes.0.contains(&consume));
    }

    #[test]
    fn test_produces_and_consumes_created_and_not_destroyed() {
        let mut e = EventLogIndex::empty();

        // Create test data
        let ch = Blake2b256Hash::from_bytes(vec![1]);

        let produce_linear = Produce {
            channel_hash: ch.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![10]),
            is_deterministic: true,
            output_value: vec![],
        };

        let produce_linear2 = Produce {
            channel_hash: ch.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![11]),
            is_deterministic: true,
            output_value: vec![],
        };

        let produce_persistent = Produce {
            channel_hash: ch.clone(),
            persistent: true,
            hash: Blake2b256Hash::from_bytes(vec![12]),
            is_deterministic: true,
            output_value: vec![],
        };

        let consume = Consume {
            channel_hashes: vec![ch.clone()].into_iter().collect(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![20]),
        };

        let consume2 = Consume {
            channel_hashes: vec![ch.clone()].into_iter().collect(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![21]),
        };

        // Add data
        e.produces_linear.0.insert(produce_linear.clone());
        e.produces_linear.0.insert(produce_linear2.clone());
        e.produces_consumed.0.insert(produce_linear2.clone());
        e.produces_persistent.0.insert(produce_persistent.clone());
        e.consumes_linear_and_peeks.0.insert(consume.clone());
        e.consumes_linear_and_peeks.0.insert(consume2.clone());
        e.consumes_produced.0.insert(consume2.clone());

        // Test produces_created_and_not_destroyed
        let not_destroyed = produces_created_and_not_destroyed(&e);
        assert_eq!(not_destroyed.0.len(), 2);
        assert!(not_destroyed.0.contains(&produce_linear));
        assert!(!not_destroyed.0.contains(&produce_linear2)); // consumed
        assert!(not_destroyed.0.contains(&produce_persistent));

        // Test consumes_created_and_not_destroyed
        let consumes_not_destroyed = consumes_created_and_not_destroyed(&e);
        assert_eq!(consumes_not_destroyed.0.len(), 1);
        assert!(consumes_not_destroyed.0.contains(&consume));
        assert!(!consumes_not_destroyed.0.contains(&consume2)); // produced
    }

    #[test]
    fn test_produces_and_consumes_affected() {
        let mut e = EventLogIndex::empty();

        // Create test data
        let ch = Blake2b256Hash::from_bytes(vec![1]);

        // Local produce
        let local_produce = Produce {
            channel_hash: ch.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![10]),
            is_deterministic: true,
            output_value: vec![],
        };

        // External produce that is consumed
        let external_produce = Produce {
            channel_hash: ch.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![11]),
            is_deterministic: true,
            output_value: vec![],
        };

        // Persistent external - shouldn't count as "affected"
        let persistent_external = Produce {
            channel_hash: ch.clone(),
            persistent: true,
            hash: Blake2b256Hash::from_bytes(vec![12]),
            is_deterministic: true,
            output_value: vec![],
        };

        // Set up similar consumes
        let local_consume = Consume {
            channel_hashes: vec![ch.clone()].into_iter().collect(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![20]),
        };

        let external_consume = Consume {
            channel_hashes: vec![ch.clone()].into_iter().collect(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![21]),
        };

        let persistent_ext_consume = Consume {
            channel_hashes: vec![ch.clone()].into_iter().collect(),
            persistent: true,
            hash: Blake2b256Hash::from_bytes(vec![22]),
        };

        // Set up index
        e.produces_linear.0.insert(local_produce.clone());
        e.produces_consumed.0.insert(external_produce.clone());
        e.produces_consumed.0.insert(persistent_external.clone());

        e.consumes_linear_and_peeks.0.insert(local_consume.clone());
        e.consumes_produced.0.insert(external_consume.clone());
        e.consumes_produced.0.insert(persistent_ext_consume.clone());

        // Test produces_affected
        let affected_produces = produces_affected(&e);
        assert_eq!(affected_produces.0.len(), 2);
        assert!(affected_produces.0.contains(&local_produce));
        assert!(affected_produces.0.contains(&external_produce));
        assert!(!affected_produces.0.contains(&persistent_external));

        // Test consumes_affected
        let affected_consumes = consumes_affected(&e);
        assert_eq!(affected_consumes.0.len(), 2);
        assert!(affected_consumes.0.contains(&local_consume));
        assert!(affected_consumes.0.contains(&external_consume));
        assert!(!affected_consumes.0.contains(&persistent_ext_consume));
    }

    #[test]
    fn test_combine_produces_copied_by_peek() {
        // Set up test indices
        let mut x = EventLogIndex::empty();
        let mut y = EventLogIndex::empty();

        let ch1 = Blake2b256Hash::from_bytes(vec![1]);
        let ch2 = Blake2b256Hash::from_bytes(vec![2]);

        // Produce that's copied by peek in x and created in y
        let p1 = Produce {
            channel_hash: ch1.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![10]),
            is_deterministic: true,
            output_value: vec![],
        };

        // Produce that's copied by peek in both but not created in either
        let p2 = Produce {
            channel_hash: ch2.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![11]),
            is_deterministic: true,
            output_value: vec![],
        };

        // Set up data
        x.produces_copied_by_peek.0.insert(p1.clone());
        x.produces_copied_by_peek.0.insert(p2.clone());

        y.produces_copied_by_peek.0.insert(p2.clone());
        y.produces_linear.0.insert(p1.clone());

        // Test combine_produces_copied_by_peek
        let combined = combine_produces_copied_by_peek(&x, &y);

        // p1 is created in y, so it shouldn't be in the result
        // p2 is copied by peek in both but not created in either, so it should be in the result
        assert_eq!(combined.0.len(), 1);
        assert!(combined.0.contains(&p2));
        assert!(!combined.0.contains(&p1));

        // Test empty case
        let empty_x = EventLogIndex::empty();
        let empty_y = EventLogIndex::empty();
        assert!(
            combine_produces_copied_by_peek(&empty_x, &empty_y)
                .0
                .is_empty()
        );
    }

    #[test]
    fn test_edge_cases() {
        // Test case with empty event logs
        let empty = EventLogIndex::empty();
        assert!(!depends(&empty, &empty));
        assert!(!are_conflicting(&empty, &empty));
        assert!(conflicts(&empty, &empty).0.is_empty());

        // Test case with single produce that is consumed and created
        let mut e = EventLogIndex::empty();
        let ch = Blake2b256Hash::from_bytes(vec![1]);
        let p = Produce {
            channel_hash: ch.clone(),
            persistent: false,
            hash: Blake2b256Hash::from_bytes(vec![1]),
            is_deterministic: true,
            output_value: vec![],
        };

        e.produces_linear.0.insert(p.clone());
        e.produces_consumed.0.insert(p.clone());

        // Should be empty since the produce is both created and consumed
        assert!(produces_created_and_not_destroyed(&e).0.is_empty());

        // The produce is still "created" even if consumed
        assert_eq!(produces_created(&e).0.len(), 1);

        // Not affected since it's consumed but also created in the same log
        assert!(produces_affected(&e).0.is_empty());
    }
}
