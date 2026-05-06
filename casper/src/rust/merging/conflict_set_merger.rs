// See casper/src/main/scala/coop/rchain/casper/merging/ConflictSetMerger.scala

use models::rhoapi::ListParWithRandom;
use rholang::rust::interpreter::merging::rholang_merging_logic::RholangMergingLogic;
use rspace_plus_plus::rspace::{
    errors::HistoryError,
    hashing::blake2b256_hash::Blake2b256Hash,
    hot_store_trie_action::HotStoreTrieAction,
    internal::Datum,
    merger::{
        merging_logic::{combine_mergeable_value, MergeType, NumberChannelsDiff},
        state_change::StateChange,
    },
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

/// Result of conflict resolution. Callers that need to adjust the rejection
/// set before diffs are applied — for example, to add DAG-descendants of
/// rejected blocks whose diffs would be stale — can do so before invoking
/// `compute_merged_state`.
pub struct ResolvedConflicts<R: Clone + Eq + std::hash::Hash> {
    /// Branches surviving conflict resolution; their diffs will be applied.
    pub to_merge: Vec<HashableSet<R>>,
    /// Rejected items (late set + dependents + optimal rejection).
    pub rejected: HashableSet<R>,
    // Diagnostic counters used in the summary log.
    pub late_set_size: usize,
    pub actual_set_size: usize,
    pub branches_count: usize,
    pub rejected_as_dependents_count: usize,
    pub optimal_rejection_count: usize,
    pub conflict_map_conflicts_count: usize,
    pub rejection_options_count: usize,
    // Timings.
    pub branches_time: Duration,
    pub conflicts_map_time: Duration,
    pub rejection_options_time: Duration,
}

/// Conflict detection and optimal rejection selection. Returns the set of
/// chains to merge along with those rejected. Callers can adjust the result
/// before calling `compute_merged_state`.
///
/// `conflicts` is fallible: a `MergeType` mismatch (or any other invariant
/// violation surfaced by event-log combination) is propagated as a hard error
/// so the merge is rejected rather than silently absorbed.
pub fn resolve_conflicts<R: Clone + Eq + std::hash::Hash + PartialOrd + Ord>(
    actual_seq: Vec<R>,
    late_seq: Vec<R>,
    depends: &impl Fn(&R, &R) -> bool,
    conflicts: &impl Fn(&HashableSet<R>, &HashableSet<R>) -> Result<bool, HistoryError>,
    cost: &impl Fn(&R) -> u64,
    mergeable_channels: &impl Fn(&R) -> NumberChannelsDiff,
    get_data: &impl Fn(Blake2b256Hash) -> Result<Vec<Datum<ListParWithRandom>>, HistoryError>,
) -> Result<ResolvedConflicts<R>, HistoryError> {
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
    metrics::histogram!(
        crate::rust::metrics_constants::DAG_MERGE_BRANCHES_TIME_METRIC,
        "source" => crate::rust::metrics_constants::MERGING_METRICS_SOURCE
    )
    .record(branches_time.as_secs_f64());

    // Compute relation map for conflicting branches with timing.
    // `compute_relation_map` requires an infallible predicate, but our
    // `conflicts` closure is fallible (see EventLogIndex::combine). We capture
    // the first error via RefCell and short-circuit the predicate; after the
    // call returns, propagate any captured error so the merge is rejected.
    use rspace_plus_plus::rspace::merger::merging_logic::compute_relation_map;
    let branches_set = HashableSet(branches.0.iter().cloned().collect());
    let conflict_err: std::cell::RefCell<Option<HistoryError>> = std::cell::RefCell::new(None);
    let (conflict_map, conflicts_map_time) = measure_time(|| {
        compute_relation_map(&branches_set, |a, b| {
            if conflict_err.borrow().is_some() {
                // Already failed; short-circuit. Returning false here doesn't affect
                // correctness because we will propagate the captured Err below.
                return false;
            }
            match conflicts(a, b) {
                Ok(c) => c,
                Err(e) => {
                    *conflict_err.borrow_mut() = Some(e);
                    false
                }
            }
        })
    });
    if let Some(err) = conflict_err.into_inner() {
        return Err(err);
    }
    metrics::histogram!(
        crate::rust::metrics_constants::DAG_MERGE_CONFLICTS_MAP_TIME_METRIC,
        "source" => crate::rust::metrics_constants::MERGING_METRICS_SOURCE
    )
    .record(conflicts_map_time.as_secs_f64());

    // Compute rejection options that leave only non-conflicting branches with timing
    use rspace_plus_plus::rspace::merger::merging_logic::compute_rejection_options;
    let (rejection_options, rejection_options_time) =
        measure_time(|| compute_rejection_options(&conflict_map));
    metrics::histogram!(
        crate::rust::metrics_constants::DAG_MERGE_REJECTION_OPTIONS_TIME_METRIC,
        "source" => crate::rust::metrics_constants::MERGING_METRICS_SOURCE
    )
    .record(rejection_options_time.as_secs_f64());

    // Get base mergeable channel results
    let channel_reads_start = Instant::now();
    // Sort keys for deterministic ordering across instances
    let mut all_channel_keys_set: std::collections::HashSet<Blake2b256Hash> =
        std::collections::HashSet::new();
    for branch in &branches {
        for item in branch {
            let item_channels = mergeable_channels(item);
            for (channel_hash, _) in item_channels.iter() {
                all_channel_keys_set.insert(channel_hash.clone());
            }
        }
    }
    let mut all_channel_keys: Vec<Blake2b256Hash> = all_channel_keys_set.into_iter().collect();
    // Sort channel keys for deterministic processing order
    all_channel_keys.sort();

    let mut base_mergeable_ch_res = HashMap::new();

    // Use RholangMergingLogic to convert the data reader function
    let get_data_ref = |hash: &Blake2b256Hash| get_data(hash.clone());
    let read_number = RholangMergingLogic::convert_to_read_number(get_data_ref);

    // Read channel numbers from storage in sorted order. `read_number` distinguishes
    // three outcomes: Ok(Some(n)) = numeric value present; Ok(None) = channel doesn't
    // exist (legitimate, start from 0); Err(_) = invariant violation or I/O error
    // (propagate to reject the merge rather than silently substituting 0).
    for channel_hash in &all_channel_keys {
        let value = read_number(channel_hash)?.unwrap_or(0);
        base_mergeable_ch_res.insert(channel_hash.clone(), value);
    }

    metrics::histogram!(
        "dag.merge.channel-reads.time",
        "source" => crate::rust::metrics_constants::MERGING_METRICS_SOURCE
    )
    .record(channel_reads_start.elapsed().as_secs_f64());

    // Get merged result rejection options
    let rejection_options_with_overflow = get_merged_result_rejection(
        &branches_set,
        &rejection_options,
        base_mergeable_ch_res.clone(),
        mergeable_channels,
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
    let conflict_map_conflicts_count = conflict_map.iter().filter(|(_, v)| !v.0.is_empty()).count();
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
        conflict_map_conflicts_count,
        rejection_options.0.len(),
        1  // rejectionOptionsWithOverflow.size - approximation
    );

    Ok(ResolvedConflicts {
        to_merge,
        rejected,
        late_set_size: late_set.len(),
        actual_set_size: actual_set.len(),
        branches_count: branches_set.0.len(),
        rejected_as_dependents_count: rejected_as_dependents.0.len(),
        optimal_rejection_count: optimal_rejection.0.len(),
        conflict_map_conflicts_count,
        rejection_options_count: rejection_options.0.len(),
        branches_time,
        conflicts_map_time,
        rejection_options_time,
    })
}

/// Combine the surviving chains' diffs into trie actions and apply them to
/// the merged base state. Reads `resolved.to_merge` and returns the new state
/// root; `resolved.rejected` is not read or modified.
pub fn compute_merged_state<R, C, P, A, K>(
    resolved: &ResolvedConflicts<R>,
    state_changes: &impl Fn(&R) -> Result<StateChange, HistoryError>,
    mergeable_channels: &impl Fn(&R) -> NumberChannelsDiff,
    compute_trie_actions: &impl Fn(
        StateChange,
        NumberChannelsDiff,
    ) -> Result<Vec<HotStoreTrieAction<C, P, A, K>>, HistoryError>,
    apply_trie_actions: &impl Fn(
        Vec<HotStoreTrieAction<C, P, A, K>>,
    ) -> Result<Blake2b256Hash, HistoryError>,
) -> Result<Blake2b256Hash, HistoryError>
where
    R: Clone + Eq + std::hash::Hash + PartialOrd + Ord,
    C: Clone,
    P: Clone,
    A: Clone,
    K: Clone,
{
    // Sort toMerge for deterministic processing order
    let mut to_merge_sorted: Vec<&HashableSet<R>> = resolved.to_merge.iter().collect();
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

    metrics::histogram!(
        "dag.merge.combine-changes.time",
        "source" => crate::rust::metrics_constants::MERGING_METRICS_SOURCE
    )
    .record(combine_all_changes_time.as_secs_f64());

    let combined_datums_count = all_changes.datums_changes.len();
    let combined_conts_count = all_changes.cont_changes.len();
    let combined_joins_count = all_changes.consume_channels_to_join_serialized_map.len();

    // Combine all mergeable channels (in sorted order). Per-channel `MergeType`
    // determines how diffs combine: integer-add uses wrapping addition; bitmask-OR
    // uses bitwise OR through u64. Branches must agree on merge_type for a given
    // channel; disagreement yields a tagged error so callers reject the merge
    // rather than crashing the validator.
    let mut all_mergeable_channels = NumberChannelsDiff::new();
    for item in &to_merge_items {
        let item_channels = mergeable_channels(item);
        for (key, value) in item_channels.iter() {
            let (incoming_diff, incoming_mt) = *value;
            match all_mergeable_channels.get_mut(key) {
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
                    all_mergeable_channels.insert(key.clone(), (incoming_diff, incoming_mt));
                }
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
        rejected as late dependency {}; changes combined (datums={}, conts={}, joins={}) in {:?}; \
        trie actions ({}) in {:?}; actions applied in {:?}",
        resolved.late_set_size,
        resolved.actual_set_size,
        resolved.branches_count,
        resolved.branches_time,
        resolved.conflicts_map_time,
        resolved.rejection_options_count,
        resolved.rejection_options_time,
        resolved.optimal_rejection_count,
        resolved.rejected_as_dependents_count,
        combined_datums_count,
        combined_conts_count,
        combined_joins_count,
        combine_all_changes_time,
        trie_actions.len(),
        compute_actions_time,
        apply_actions_time
    );

    debug!("{}", log_str);

    Ok(new_state)
}

/// R is a type for minimal rejection unit.
/// IMPORTANT: actual_seq and late_seq must be passed in sorted order to ensure
/// deterministic processing across all validators.
///
/// Convenience wrapper that runs `resolve_conflicts` followed by
/// `compute_merged_state`. Callers that need to inspect or adjust the rejection
/// set between the two steps should call them directly instead.
pub fn merge<
    R: Clone + Eq + std::hash::Hash + PartialOrd + Ord,
    C: Clone,
    P: Clone,
    A: Clone,
    K: Clone,
>(
    actual_seq: Vec<R>,
    late_seq: Vec<R>,
    depends: impl Fn(&R, &R) -> bool,
    conflicts: impl Fn(&HashableSet<R>, &HashableSet<R>) -> Result<bool, HistoryError>,
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
    let resolved = resolve_conflicts(
        actual_seq,
        late_seq,
        &depends,
        &conflicts,
        &cost,
        &mergeable_channels,
        &get_data,
    )?;
    let new_state = compute_merged_state(
        &resolved,
        &state_changes,
        &mergeable_channels,
        &compute_trie_actions,
        &apply_trie_actions,
    )?;
    Ok((new_state, resolved.rejected))
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

        // Third criterion: For tie-breaking, compare the first element of the first branch
        // Use sorted branches and min element for deterministic tie-breaking
        let mut a_branches: Vec<_> = a.0.iter().collect();
        let mut b_branches: Vec<_> = b.0.iter().collect();
        a_branches.sort_by(|x, y| compare_branches(x, y));
        b_branches.sort_by(|x, y| compare_branches(x, y));

        let a_first = a_branches.first().and_then(|branch| branch.0.iter().min());
        let b_first = b_branches.first().and_then(|branch| branch.0.iter().min());

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
///
/// Note: the non-negative-result check applies only to `IntegerAdd` channels
/// (vault balances). `BitmaskOr` channels are bitmaps, where any value is
/// representable; we OR the diff into the existing value without overflow
/// concerns.
fn cal_merged_result<R: Clone + Eq + std::hash::Hash>(
    branch: &Branch<R>,
    origin_result: HashMap<Blake2b256Hash, i64>,
    mergeable_channels: impl Fn(&R) -> NumberChannelsDiff,
) -> Option<HashMap<Blake2b256Hash, i64>> {
    // Combine all channel diffs from the branch using per-channel merge strategy.
    let diff = branch.0.iter().map(|r| mergeable_channels(r)).fold(
        NumberChannelsDiff::new(),
        |mut acc, x| {
            for (k, v) in x {
                let (incoming_diff, incoming_mt) = v;
                acc.entry(k)
                    .and_modify(|existing| {
                        existing.0 =
                            combine_mergeable_value(existing.0, incoming_diff, incoming_mt);
                    })
                    .or_insert((incoming_diff, incoming_mt));
            }
            acc
        },
    );

    // Start with Some(origin_result) and fold over the diffs
    diff.iter()
        .fold(Some(origin_result), |ba_opt, (channel, value)| {
            ba_opt.and_then(|mut ba| {
                let (diff_val, merge_type) = *value;
                let current = *ba.get(channel).unwrap_or(&0);
                match merge_type {
                    MergeType::IntegerAdd => {
                        // Vault balance: overflow or negative result rejects the branch
                        match current.checked_add(diff_val) {
                            Some(result) if result >= 0 => {
                                ba.insert(channel.clone(), result);
                                Some(ba)
                            }
                            _ => None,
                        }
                    }
                    MergeType::BitmaskOr => {
                        // Bitmap: OR the new bits in; no overflow concern
                        let result = ((current as u64) | (diff_val as u64)) as i64;
                        ba.insert(channel.clone(), result);
                        Some(ba)
                    }
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
    use std::collections::{BTreeMap, HashSet};

    fn branch(items: &[i32]) -> Branch<i32> {
        HashableSet(items.iter().copied().collect::<HashSet<i32>>())
    }

    fn rejection_option(branches: &[Branch<i32>]) -> HashableSet<Branch<i32>> {
        HashableSet(branches.iter().cloned().collect::<HashSet<Branch<i32>>>())
    }

    #[test]
    fn compare_branches_is_deterministic() {
        let a = branch(&[1, 2]);
        let b = branch(&[2, 1]);
        let c = branch(&[1, 3]);
        let d = branch(&[1, 4]);
        let short = branch(&[1]);

        assert_eq!(compare_branches(&a, &b), std::cmp::Ordering::Equal);
        assert_eq!(compare_branches(&short, &a), std::cmp::Ordering::Less);
        assert_eq!(compare_branches(&c, &d), std::cmp::Ordering::Less);
    }

    #[test]
    fn optimal_rejection_tie_break_is_stable() {
        // Both options have equal target sum (5) and equal branch count.
        // Deterministic tie-break should pick option_a because its first branch
        // starts with lower element (1 < 2).
        let option_a = rejection_option(&[branch(&[1]), branch(&[4])]);
        let option_b = rejection_option(&[branch(&[2]), branch(&[3])]);
        let options = HashableSet(HashSet::from([option_b.clone(), option_a.clone()]));

        let chosen = get_optimal_rejection(options, |branch| {
            branch.0.iter().map(|value| *value as u64).sum()
        });

        assert_eq!(chosen, option_a);
    }

    #[test]
    fn merge_rejects_negative_channel_balance() {
        let actual_seq = vec![1, 2];
        let late_seq = Vec::<i32>::new();
        let base_channel = Blake2b256Hash::from_bytes(vec![7u8; 32]);

        let result = merge(
            actual_seq,
            late_seq,
            |_a, _b| false,     // depends
            |_a, _b| Ok(false), // conflicts
            |_r| 1,             // cost
            |_r| Ok(StateChange::empty()),
            |r| {
                let mut diff = BTreeMap::new();
                // item 1 decrements channel, item 2 increments channel
                let delta = if *r == 1 { -1 } else { 1 };
                diff.insert(base_channel.clone(), (delta, MergeType::IntegerAdd));
                diff
            },
            |_state_change, _channels| Ok(Vec::<HotStoreTrieAction<i32, i32, i32, i32>>::new()),
            |_actions: Vec<HotStoreTrieAction<i32, i32, i32, i32>>| {
                Ok(Blake2b256Hash::from_bytes(vec![9u8; 32]))
            },
            |_hash| Ok(Vec::new()),
        );

        assert!(result.is_ok());
        let (_new_state, rejected) = result.unwrap();
        assert!(!rejected.0.is_empty());
    }
}
