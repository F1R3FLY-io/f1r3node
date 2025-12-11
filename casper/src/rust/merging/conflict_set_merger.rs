// See casper/src/main/scala/coop/rchain/casper/merging/ConflictSetMerger.scala

use shared::rust::hashable_set::HashableSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::time::{Duration, Instant};

use tracing::debug;
use models::rhoapi::ListParWithRandom;
use rholang::rust::interpreter::merging::rholang_merging_logic::RholangMergingLogic;
use rspace_plus_plus::rspace::{
    errors::HistoryError,
    hashing::blake2b256_hash::Blake2b256Hash,
    hot_store_trie_action::HotStoreTrieAction,
    internal::Datum,
    merger::{merging_logic::NumberChannelsDiff, state_change::StateChange},
};

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

pub fn merge<
    R: Clone + Eq + std::hash::Hash + PartialOrd + Ord,
    C: Clone,
    P: Clone,
    A: Clone,
    K: Clone,
>(
    actual_set: HashableSet<R>,
    late_set: HashableSet<R>,
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
    // Split the actual_set into branches without cross dependencies
    let (rejected_as_dependents, merge_set): (HashableSet<R>, HashableSet<R>) = {
        let mut rejected = HashableSet(HashSet::new());
        let mut to_merge = HashableSet(HashSet::new());

        for item in &actual_set {
            if late_set.0.iter().any(|late_item| depends(item, late_item)) {
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
    let mut base_mergeable_ch_res = HashMap::new();

    // Use RholangMergingLogic to convert the data reader function
    let get_data_ref = |hash: &Blake2b256Hash| get_data(hash.clone());
    let read_number = RholangMergingLogic::convert_to_read_number(get_data_ref);

    // Read channel numbers from storage with proper conversion
    for branch in &branches {
        for item in branch {
            let item_channels = mergeable_channels(item);
            for (channel_hash, _) in item_channels.iter() {
                if !base_mergeable_ch_res.contains_key(channel_hash) {
                    // Use proper conversion to read number and handle Option correctly
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

    // Compute branches to merge and rejected items
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

    // Combine state changes from all items to be merged with timing
    let (all_changes, combine_all_changes_time) =
        measure_result_time(|| -> Result<StateChange, HistoryError> {
            let mut combined = StateChange::empty();
            for branch in &to_merge {
                for item in branch {
                    let item_changes = state_changes(item)?;
                    combined = combined.combine(item_changes);
                }
            }
            Ok(combined)
        })?;

    // Combine all mergeable channels
    let mut all_mergeable_channels = NumberChannelsDiff::new();
    for branch in &to_merge {
        for item in branch {
            let item_channels = mergeable_channels(item);
            for (key, value) in item_channels.iter() {
                *all_mergeable_channels.entry(key.clone()).or_insert(0) += *value;
            }
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
        late_set.0.len(),
        actual_set.0.len(),
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

/** compute optimal rejection configuration */
fn get_optimal_rejection<R: Eq + std::hash::Hash + Clone + Ord>(
    options: HashableSet<HashableSet<Branch<R>>>,
    target_f: impl Fn(&Branch<R>) -> u64,
) -> HashableSet<Branch<R>> {
    assert!(
        options.0.iter().map(|b| {
            let mut heads = HashSet::new();
            for branch in &b.0 {
                if let Some(head) = branch.0.iter().next() {
                    heads.insert(head);
                }
            }
            heads
        }).collect::<Vec<_>>().len() == options.0.len(),
        "Same rejection unit is found in two rejection options. Please report this to code maintainer."
    );

    // reject set with min sum of target function output,
    // if equal value - min size of a branch
    // if equal size - use a deterministic tie-breaker based on the first element of first branch (like Scala)
    options
        .0
        .into_iter()
        .min_by(|a, b| {
            // First criterion: sum of target function values
            let a_sum = a.0.iter().map(|branch| target_f(branch)).sum::<u64>();
            let b_sum = b.0.iter().map(|branch| target_f(branch)).sum::<u64>();

            if a_sum != b_sum {
                return a_sum.cmp(&b_sum);
            }

            // Second criterion: total size of branches
            let a_size = a.0.iter().map(|branch| branch.0.len()).sum::<usize>();
            let b_size = b.0.iter().map(|branch| branch.0.len()).sum::<usize>();

            if a_size != b_size {
                return a_size.cmp(&b_size);
            }

            // Third criterion: For tie-breaking, compare the first element of the first branch (as Scala does)
            // This requires the R type to implement Ord
            let a_first_branch = a.0.iter().next();
            let b_first_branch = b.0.iter().next();

            match (a_first_branch, b_first_branch) {
                (Some(a_branch), Some(b_branch)) => {
                    let a_first_item = a_branch.0.iter().next();
                    let b_first_item = b_branch.0.iter().next();

                    match (a_first_item, b_first_item) {
                        (Some(a_item), Some(b_item)) => a_item.cmp(b_item),
                        (Some(_), None) => std::cmp::Ordering::Greater,
                        (None, Some(_)) => std::cmp::Ordering::Less,
                        (None, None) => std::cmp::Ordering::Equal,
                    }
                }
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (None, None) => std::cmp::Ordering::Equal,
            }
        })
        .unwrap_or_else(|| HashableSet(HashSet::new()))
}

/** Calculate merged result for a branch with the origin result map */
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

/** Evaluate branches and return the set of branches that should be rejected */
fn fold_rejection<R: Clone + Eq + std::hash::Hash>(
    base_balance: HashMap<Blake2b256Hash, i64>,
    branches: &HashableSet<Branch<R>>,
    mergeable_channels: impl Fn(&R) -> NumberChannelsDiff,
) -> HashableSet<Branch<R>>
where
    R: Ord,
{
    // Sort branches for deterministic processing order
    // For DeployChainIndex, we'll sort by the smallest deploy signature in each branch
    let mut sorted_branches: Vec<&Branch<R>> = branches.0.iter().collect();
    sorted_branches.sort_by(|a, b| {
        // For deterministic ordering, find the lexicographically smallest element in each branch
        let a_min = a.0.iter().min();
        let b_min = b.0.iter().min();
        a_min.cmp(&b_min)
    });

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

/** Get merged result rejection options */
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
