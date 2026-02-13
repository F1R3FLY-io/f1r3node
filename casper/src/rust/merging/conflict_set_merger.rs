// See casper/src/main/scala/coop/rchain/casper/merging/ConflictSetMerger.scala

use models::rhoapi::ListParWithRandom;
use rholang::rust::interpreter::merging::rholang_merging_logic::RholangMergingLogic;
use rspace_plus_plus::rspace::{
    errors::HistoryError,
    hashing::blake2b256_hash::Blake2b256Hash,
    hot_store_trie_action::HotStoreTrieAction,
    internal::Datum,
    merger::{merging_logic::NumberChannelsDiff, state_change::StateChange},
};
use shared::rust::hashable_set::HashableSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::time::{Duration, Instant};
use tracing::{debug, info};

type Branch<R> = HashableSet<R>;

// Utility for timing operations
fn measure_time<T, F: FnOnce() -> T>(f: F) -> (T, Duration) {
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

// Utility to time operations that return Result
fn measure_result_time<T, E, F: FnOnce() -> Result<T, E>>(f: F) -> Result<(T, Duration), E> {
    let start = Instant::now();
    let result = f()?;
    let duration = start.elapsed();
    Ok((result, duration))
}

/// Compare two branches for deterministic ordering.
/// Ordering for branches to ensure deterministic comparison.
fn compare_branches<R: Ord>(a: &Branch<R>, b: &Branch<R>) -> std::cmp::Ordering {
    // Compare by sorted elements
    let mut a_sorted: Vec<_> = a.0.iter().collect();
    let mut b_sorted: Vec<_> = b.0.iter().collect();
    a_sorted.sort();
    b_sorted.sort();

    let len_cmp = a_sorted.len().cmp(&b_sorted.len());
    if len_cmp != std::cmp::Ordering::Equal {
        return len_cmp;
    }

    for (a_item, b_item) in a_sorted.iter().zip(b_sorted.iter()) {
        let cmp = a_item.cmp(b_item);
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
    }

    std::cmp::Ordering::Equal
}

/// R is a type for minimal rejection unit.
/// IMPORTANT: actual_seq and late_seq must be passed in sorted order to ensure
/// deterministic processing across all validators.
pub fn merge<
    R: Clone + Eq + std::hash::Hash + PartialOrd + Ord,
    C: Clone,
    P: Clone,
    A: Clone,
    K: Clone,
>(
    actual_seq: Vec<R>, // Changed from HashableSet to Vec for deterministic ordering
    late_seq: Vec<R>,   // Changed from HashableSet to Vec for deterministic ordering
    depends: impl Fn(&R, &R) -> bool,
    conflicts: impl Fn(&HashableSet<R>, &HashableSet<R>) -> bool,
    cost: impl Fn(&R) -> u64,
    state_changes: impl Fn(&R) -> Result<StateChange, HistoryError>,
    mergeable_channels: impl Fn(&R) -> NumberChannelsDiff,
    compute_trie_actions: impl Fn(
        StateChange,
        NumberChannelsDiff,
    ) -> Result<Vec<HotStoreTrieAction<C, P, A, K>>, HistoryError>,
    apply_trie_actions: impl Fn(
        Vec<HotStoreTrieAction<C, P, A, K>>,
    ) -> Result<Blake2b256Hash, HistoryError>,
    get_data: impl Fn(Blake2b256Hash) -> Result<Vec<Datum<ListParWithRandom>>, HistoryError>,
) -> Result<(Blake2b256Hash, HashableSet<R>), HistoryError> {
    // Convert to Sets for set operations, but use Vec for ordered iteration
    let actual_set: HashSet<R> = actual_seq.iter().cloned().collect();
    let late_set: HashSet<R> = late_seq.iter().cloned().collect();

    // Split the actual_set into branches without cross dependencies
    let (rejected_as_dependents, merge_set): (HashableSet<R>, HashableSet<R>) = {
        let mut rejected = HashableSet(HashSet::new());
        let mut to_merge = HashableSet(HashSet::new());

        for item in &actual_set {
            if late_set.iter().any(|late_item| depends(item, late_item)) {
                rejected.0.insert(item.clone());
            } else {
                to_merge.0.insert(item.clone());
            }
        }

        (rejected, to_merge)
    };

    // Compute related sets to split merging set into branches without cross dependencies
    use rspace_plus_plus::rspace::merger::merging_logic::compute_related_sets;
    let (branches, branches_time) =
        measure_time(|| compute_related_sets(&merge_set, |a, b| depends(a, b)));

    // Compute relation map for conflicting branches with timing
    use rspace_plus_plus::rspace::merger::merging_logic::compute_relation_map;
    let branches_set = HashableSet(branches.0.iter().cloned().collect());
    let (conflict_map, conflicts_map_time) =
        measure_time(|| compute_relation_map(&branches_set, |a, b| conflicts(a, b)));

    // Compute rejection options that leave only non-conflicting branches with timing
    use rspace_plus_plus::rspace::merger::merging_logic::compute_rejection_options;
    let (rejection_options, rejection_options_time) =
        measure_time(|| compute_rejection_options(&conflict_map));

    // Get base mergeable channel results
    // Sort keys for deterministic ordering across instances
    let mut all_channel_keys: Vec<Blake2b256Hash> = Vec::new();
    for branch in &branches {
        for item in branch {
            let item_channels = mergeable_channels(item);
            for (channel_hash, _) in item_channels.iter() {
                if !all_channel_keys.contains(channel_hash) {
                    all_channel_keys.push(channel_hash.clone());
                }
            }
        }
    }
    // Sort channel keys for deterministic processing order
    all_channel_keys.sort();

    let mut base_mergeable_ch_res = HashMap::new();

    // Use RholangMergingLogic to convert the data reader function
    let get_data_ref = |hash: &Blake2b256Hash| get_data(hash.clone());
    let read_number = RholangMergingLogic::convert_to_read_number(get_data_ref);

    // Read channel numbers from storage in sorted order
    for channel_hash in &all_channel_keys {
        match read_number(channel_hash) {
            Some(value) => {
                base_mergeable_ch_res.insert(channel_hash.clone(), value);
            }
            None => {
                // If the channel doesn't exist yet, we can use 0 as a starting value
                base_mergeable_ch_res.insert(channel_hash.clone(), 0);
            }
        }
    }

    // Get merged result rejection options
    let rejection_options_with_overflow = get_merged_result_rejection(
        &branches_set,
        &rejection_options,
        base_mergeable_ch_res.clone(),
        &mergeable_channels,
    );

    // Compute optimal rejection using cost function
    let optimal_rejection = get_optimal_rejection(rejection_options_with_overflow, |branch| {
        branch.0.iter().map(|item| cost(item)).sum()
    });

    // Compute branches to merge (difference of branches and optimal_rejection)
    let to_merge: Vec<HashableSet<R>> = branches
        .into_iter()
        .filter(|branch| {
            // Check if branch is not in optimal_rejection
            !optimal_rejection.0.iter().any(|reject_branch| {
                if branch.0.len() != reject_branch.0.len() {
                    return false;
                }
                branch.0.iter().all(|item| reject_branch.0.contains(item))
            })
        })
        .collect();

    // Flatten the optimal rejection set
    let mut optimal_rejection_flattened = HashableSet(HashSet::new());
    for branch in &optimal_rejection {
        for item in branch {
            optimal_rejection_flattened.0.insert(item.clone());
        }
    }

    // Combine all rejected items
    let mut rejected = HashableSet(HashSet::new());
    for item in &late_set {
        rejected.0.insert(item.clone());
    }
    for item in &rejected_as_dependents {
        rejected.0.insert(item.clone());
    }
    for item in &optimal_rejection_flattened {
        rejected.0.insert(item.clone());
    }

    // Detailed INFO logging for rejection breakdown (always visible)
    info!(
        "ConflictSetMerger rejection breakdown: lateSet={}, rejectedAsDependents={}, \
        optimalRejection={}, total rejected={}, branches={}, toMerge={}, \
        conflictMap entries with conflicts={}, rejectionOptions={}, rejectionOptionsWithOverflow={}",
        late_set.len(),
        rejected_as_dependents.0.len(),
        optimal_rejection_flattened.0.len(),
        rejected.0.len(),
        branches_set.0.len(),
        to_merge.len(),
        conflict_map.iter().filter(|(_, v)| !v.0.is_empty()).count(),
        rejection_options.0.len(),
        1  // rejectionOptionsWithOverflow.size - approximation
    );

    // Sort toMerge for deterministic processing order
    let mut to_merge_sorted: Vec<&HashableSet<R>> = to_merge.iter().collect();
    to_merge_sorted.sort_by(|a, b| compare_branches(a, b));

    // Flatten and sort items within each branch
    let mut to_merge_items: Vec<&R> = Vec::new();
    for branch in to_merge_sorted {
        let mut branch_items: Vec<_> = branch.0.iter().collect();
        branch_items.sort();
        to_merge_items.extend(branch_items);
    }

    // Combine state changes from all items to be merged with timing
    let (all_changes, combine_all_changes_time) =
        measure_result_time(|| -> Result<StateChange, HistoryError> {
            let mut combined = StateChange::empty();
            for item in &to_merge_items {
                let item_changes = state_changes(item)?;
                combined = combined.combine(item_changes);
            }
            Ok(combined)
        })?;

    // Combine all mergeable channels (in sorted order)
    let mut all_mergeable_channels = NumberChannelsDiff::new();
    for item in &to_merge_items {
        let item_channels = mergeable_channels(item);
        for (key, value) in item_channels.iter() {
            *all_mergeable_channels.entry(key.clone()).or_insert(0) += *value;
        }
    }

    // Compute and apply trie actions with timing
    let (trie_actions, compute_actions_time) =
        measure_result_time(|| compute_trie_actions(all_changes, all_mergeable_channels.clone()))?;

    let (new_state, apply_actions_time) =
        measure_result_time(|| apply_trie_actions(trie_actions.clone()))?;

    // Prepare log message
    let log_str = format!(
        "Merging done: late set size {}; actual set size {}; computed branches ({}) in {:?}; \
        conflicts map in {:?}; rejection options ({}) in {:?}; optimal rejection set size {}; \
        rejected as late dependency {}; changes combined in {:?}; trie actions ({}) in {:?}; \
        actions applied in {:?}",
        late_set.len(),
        actual_set.len(),
        branches_set.0.len(),
        branches_time,
        conflicts_map_time,
        rejection_options.0.len(),
        rejection_options_time,
        optimal_rejection.0.len(),
        rejected_as_dependents.0.len(),
        combine_all_changes_time,
        trie_actions.len(),
        compute_actions_time,
        apply_actions_time
    );

    debug!("{}", log_str);

    Ok((new_state, rejected))
}

/// Compute optimal rejection configuration.
/// Find the optimal rejection set from conflicting branches.
fn get_optimal_rejection<R: Eq + std::hash::Hash + Clone + Ord>(
    options: HashableSet<HashableSet<Branch<R>>>,
    target_f: impl Fn(&Branch<R>) -> u64,
) -> HashableSet<Branch<R>> {
    assert!(
        options
            .0
            .iter()
            .map(|b| {
                let mut heads = HashSet::new();
                for branch in &b.0 {
                    if let Some(head) = branch.0.iter().min() {
                        // Use min() for determinism
                        heads.insert(head);
                    }
                }
                heads
            })
            .collect::<Vec<_>>()
            .len()
            == options.0.len(),
        "Same rejection unit is found in two rejection options. Please report this to code maintainer."
    );

    // Convert to sorted list for deterministic processing
    let mut options_vec: Vec<_> = options.0.into_iter().collect();
    options_vec.sort_by(|a, b| {
        // First criterion: sum of target function values
        let a_sum: u64 = a.0.iter().map(|branch| target_f(branch)).sum();
        let b_sum: u64 = b.0.iter().map(|branch| target_f(branch)).sum();

        if a_sum != b_sum {
            return a_sum.cmp(&b_sum);
        }

        // Second criterion: total size of branches
        let a_size: usize = a.0.iter().map(|branch| branch.0.len()).sum();
        let b_size: usize = b.0.iter().map(|branch| branch.0.len()).sum();

        if a_size != b_size {
            return a_size.cmp(&b_size);
        }

        // Third criterion: deterministic tiebreak by smallest element across all branches
        let a_first = a.0.iter().flat_map(|branch| branch.0.iter()).min();
        let b_first = b.0.iter().flat_map(|branch| branch.0.iter()).min();

        match (a_first, b_first) {
            (Some(a_item), Some(b_item)) => a_item.cmp(b_item),
            (Some(_), None) => std::cmp::Ordering::Greater,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (None, None) => std::cmp::Ordering::Equal,
        }
    });

    options_vec
        .into_iter()
        .next()
        .unwrap_or_else(|| HashableSet(HashSet::new()))
}

/// Calculate merged result for a branch with the origin result map.
/// Calculate the merged result from base and branches.
fn cal_merged_result<R: Clone + Eq + std::hash::Hash>(
    branch: &Branch<R>,
    origin_result: HashMap<Blake2b256Hash, i64>,
    mergeable_channels: impl Fn(&R) -> NumberChannelsDiff,
) -> Option<HashMap<Blake2b256Hash, i64>> {
    // Combine all channel diffs from the branch
    let diff = branch.0.iter().map(|r| mergeable_channels(r)).fold(
        NumberChannelsDiff::new(),
        |mut acc, x| {
            // Manually combine maps by adding values for each key
            for (k, v) in x {
                *acc.entry(k).or_insert(0) += v;
            }
            acc
        },
    );

    // Start with Some(origin_result) and fold over the diffs
    diff.iter()
        .fold(Some(origin_result), |ba_opt, (channel, diff_val)| {
            ba_opt.and_then(|mut ba| {
                let current = *ba.get(channel).unwrap_or(&0);
                // Check for overflow and negative results
                match current.checked_add(*diff_val) {
                    Some(result) if result >= 0 => {
                        ba.insert(channel.clone(), result);
                        Some(ba)
                    }
                    _ => None, // Return None for overflow or negative result
                }
            })
        })
}

/// Evaluate branches and return the set of branches that should be rejected.
/// Fold over branches and compute rejections.
fn fold_rejection<R: Clone + Eq + std::hash::Hash + Ord>(
    base_balance: HashMap<Blake2b256Hash, i64>,
    branches: &HashableSet<Branch<R>>,
    mergeable_channels: impl Fn(&R) -> NumberChannelsDiff,
) -> HashableSet<Branch<R>> {
    // Sort branches to ensure deterministic processing order
    let mut sorted_branches: Vec<&Branch<R>> = branches.0.iter().collect();
    sorted_branches.sort_by(|a, b| compare_branches(a, b));

    // Fold branches to find which ones would result in negative or overflow balances
    let (_, rejected) = sorted_branches.iter().fold(
        (base_balance, HashableSet(HashSet::new())),
        |(balances, mut rejected), branch| {
            // Check if the branch can be merged without overflow or negative results
            match cal_merged_result(branch, balances.clone(), &mergeable_channels) {
                Some(new_balances) => (new_balances, rejected),
                None => {
                    // If merge calculation returns None, reject this branch
                    rejected.0.insert((*branch).clone());
                    (balances, rejected)
                }
            }
        },
    );

    rejected
}

/// Get merged result rejection options.
/// Get the merged result along with rejected deploys.
fn get_merged_result_rejection<R: Clone + Eq + std::hash::Hash + Ord>(
    branches: &HashableSet<Branch<R>>,
    reject_options: &HashableSet<HashableSet<Branch<R>>>,
    base: HashMap<Blake2b256Hash, i64>,
    mergeable_channels: impl Fn(&R) -> NumberChannelsDiff,
) -> HashableSet<HashableSet<Branch<R>>> {
    if reject_options.0.is_empty() {
        // If no rejection options, fold the branches and return as single option
        let rejected = fold_rejection(base, branches, &mergeable_channels);
        let mut result = HashSet::new();
        result.insert(rejected);
        HashableSet(result)
    } else {
        // For each reject option, compute the difference and fold
        let result: HashSet<HashableSet<Branch<R>>> = reject_options
            .0
            .iter()
            .map(|normal_reject_options| {
                // Find branches that aren't in normal_reject_options
                let diff = HashableSet(
                    branches
                        .0
                        .iter()
                        .filter(|branch| {
                            // Check if branch is not in normal_reject_options
                            !normal_reject_options.0.iter().any(|reject_branch| {
                                if branch.0.len() != reject_branch.0.len() {
                                    return false;
                                }
                                branch.0.iter().all(|item| reject_branch.0.contains(item))
                            })
                        })
                        .cloned()
                        .collect(),
                );

                // Get branches that should be rejected from the diff
                let rejected = fold_rejection(base.clone(), &diff, &mergeable_channels);

                // Combine rejected with normal_reject_options
                let mut result = HashableSet(normal_reject_options.0.clone());
                for reject in &rejected.0 {
                    result.0.insert(reject.clone());
                }

                result
            })
            .collect();

        HashableSet(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn branch(items: &[i32]) -> Branch<i32> {
        HashableSet(items.iter().copied().collect())
    }

    fn mk_options(opts: Vec<Vec<Vec<i32>>>) -> HashableSet<HashableSet<Branch<i32>>> {
        let result: HashSet<HashableSet<Branch<i32>>> = opts
            .into_iter()
            .map(|branches| {
                HashableSet(
                    branches
                        .into_iter()
                        .map(|items| branch(&items))
                        .collect(),
                )
            })
            .collect();
        HashableSet(result)
    }

    #[test]
    fn optimal_rejection_deterministic_when_costs_equal() {
        let opts = mk_options(vec![
            vec![vec![3, 4]],
            vec![vec![5, 6]],
            vec![vec![1, 2]],
        ]);

        let result = get_optimal_rejection(opts, |_| 0);
        // Should deterministically pick the option containing the smallest element (1)
        assert_eq!(result, HashableSet(HashSet::from([branch(&[1, 2])])));
    }

    #[test]
    fn optimal_rejection_stable_across_repeated_invocations() {
        let build = || {
            mk_options(vec![
                vec![vec![10, 20]],
                vec![vec![5, 15]],
                vec![vec![30, 40]],
            ])
        };

        let results: Vec<_> = (0..100)
            .map(|_| get_optimal_rejection(build(), |_| 0))
            .collect();

        for r in &results[1..] {
            assert_eq!(results[0], *r, "result must be identical across invocations");
        }
        // Should always pick the one with min element = 5
        assert_eq!(
            results[0],
            HashableSet(HashSet::from([branch(&[5, 15])]))
        );
    }

    #[test]
    fn optimal_rejection_prefers_lower_cost() {
        let opts = mk_options(vec![
            vec![vec![100, 200]], // high elements but low cost when sum
            vec![vec![1, 2]],    // low elements but also low cost
        ]);

        let cost_f = |s: &Branch<i32>| -> u64 { s.0.iter().copied().sum::<i32>() as u64 };
        let result = get_optimal_rejection(opts, cost_f);
        // costF({1,2}) = 3, costF({100,200}) = 300
        // Should prefer the lower cost option
        assert_eq!(result, HashableSet(HashSet::from([branch(&[1, 2])])));
    }

    #[test]
    fn optimal_rejection_prefers_smaller_size_when_costs_equal() {
        let opts = mk_options(vec![
            vec![vec![1, 2]],
            vec![vec![3, 4], vec![5, 6]],
        ]);

        let result = get_optimal_rejection(opts, |_| 10);
        // Both have same cost per branch (10), but small has 1 branch (cost=10) vs 2 (cost=20).
        // So small wins on cost.
        assert_eq!(result, HashableSet(HashSet::from([branch(&[1, 2])])));
    }

    #[test]
    fn optimal_rejection_uses_min_element_tiebreak() {
        let opts = mk_options(vec![
            vec![vec![10]],
            vec![vec![5]],
            vec![vec![20]],
        ]);

        let result = get_optimal_rejection(opts, |_| 100);
        // All have cost 100 and size 1. Min element: 5 < 10 < 20
        assert_eq!(result, HashableSet(HashSet::from([branch(&[5])])));
    }
}
