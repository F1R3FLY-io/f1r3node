// See casper/src/main/scala/coop/rchain/casper/merging/DagMerger.scala

use prost::bytes::Bytes;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use models::rust::block_hash::BlockHash;
use rholang::rust::interpreter::{
    merging::rholang_merging_logic::RholangMergingLogic, rho_runtime::RhoHistoryRepository,
};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    merger::{
        merging_logic::{self, NumberChannelsDiff},
        state_change_merger,
    },
};
use shared::rust::hashable_set::HashableSet;

use super::{conflict_set_merger, deploy_chain_index::DeployChainIndex};
use crate::rust::{
    errors::CasperError,
    system_deploy::{is_slash_deploy_id, is_system_deploy_id},
};

pub fn cost_optimal_rejection_alg() -> impl Fn(&DeployChainIndex) -> u64 {
    |deploy_chain_index: &DeployChainIndex| {
        deploy_chain_index
            .deploys_with_cost
            .0
            .iter()
            .map(|deploy| deploy.cost)
            .sum()
    }
}

/// BFS walk of DAG descendants of `start_blocks`, restricted to `scope`.
///
/// When the merge rejects the deploy chains of a block, any descendant block
/// in scope has diffs that were computed against the rejected block's
/// post-state and are therefore stale. This walk identifies the affected
/// descendants so their chains can be rejected as well.
///
/// Returns the strict descendants; the start blocks themselves are not included.
fn descendants_within_scope(
    dag: &KeyValueDagRepresentation,
    start_blocks: &HashSet<BlockHash>,
    scope: &HashSet<BlockHash>,
) -> HashSet<BlockHash> {
    let mut result = HashSet::new();
    let mut queue: Vec<BlockHash> = start_blocks.iter().cloned().collect();
    let mut visited: HashSet<BlockHash> = start_blocks.clone();

    while let Some(current) = queue.pop() {
        if let Some(children) = dag.children(&current) {
            for child in children {
                if scope.contains(&child) && visited.insert(child.clone()) {
                    result.insert(child.clone());
                    queue.push(child);
                }
            }
        }
    }

    result
}

pub fn merge(
    dag: &KeyValueDagRepresentation,
    lfb: &BlockHash,
    lfb_post_state: &Blake2b256Hash,
    index: impl Fn(&BlockHash) -> Result<Vec<DeployChainIndex>, CasperError>,
    history_repository: &RhoHistoryRepository,
    rejection_cost_f: impl Fn(&DeployChainIndex) -> u64,
    scope: Option<HashSet<BlockHash>>,
    disable_late_block_filtering: bool,
) -> Result<
    (
        Blake2b256Hash,
        Vec<(Bytes, BlockHash)>,
        Vec<(Bytes, BlockHash)>,
    ),
    CasperError,
> {
    // Blocks to merge are all blocks in scope that are NOT the LFB or its ancestors.
    // This includes:
    // 1. Descendants of LFB (blocks built on top of LFB)
    // 2. Siblings of LFB (blocks at same height but different branch) that are ancestors of the tips
    // Previously we only included descendants, which missed deploy effects from sibling branches.
    let actual_blocks: HashSet<BlockHash> = match &scope {
        Some(scope_blocks) => {
            // Avoid unbounded full-DAG ancestor scans. Check each scope block against LFB directly.
            let mut result = HashSet::new();
            for candidate in scope_blocks {
                if !dag.is_in_main_chain(candidate, lfb)? {
                    result.insert(candidate.clone());
                }
            }
            result
        }
        None => {
            // Legacy behavior: use descendants of LFB
            dag.descendants(lfb)?
        }
    };

    // Late blocks: With the new actualBlocks definition that includes sibling branches,
    // there are no "late" blocks when scope is provided - all non-ancestor blocks are in actualBlocks.
    // Late block filtering is now only relevant for legacy code paths without scope.
    let late_blocks: HashSet<BlockHash> = if disable_late_block_filtering || scope.is_some() {
        // No late blocks when scope is provided (all relevant blocks are in actualBlocks)
        HashSet::new()
    } else {
        // Legacy: query nonFinalizedBlocks (non-deterministic, but no scope means
        // this is not a multi-parent merge validation)
        let non_finalized_blocks = dag.non_finalized_blocks()?;
        non_finalized_blocks
            .difference(&actual_blocks)
            .cloned()
            .collect()
    };

    // Log the block sets for debugging
    tracing::info!(
        "DagMerger.merge: LFB={}, scope={}, actualBlocks (above LFB)={}, lateBlocks={}",
        hex::encode(&lfb[..std::cmp::min(8, lfb.len())]),
        scope
            .as_ref()
            .map_or("ALL".to_string(), |s| format!("{} blocks", s.len())),
        actual_blocks.len(),
        late_blocks.len()
    );

    // Get indices for actual and late blocks, converting to sorted vectors for determinism
    let mut actual_set_vec = Vec::new();
    let mut late_set_vec = Vec::new();

    // Process actual blocks (sorted for determinism)
    let mut actual_blocks_sorted: Vec<_> = actual_blocks.iter().collect();
    actual_blocks_sorted.sort();
    for block_hash in actual_blocks_sorted {
        let indices = index(block_hash)?;
        actual_set_vec.extend(indices);
    }

    // Process late blocks (sorted for determinism)
    let mut late_blocks_sorted: Vec<_> = late_blocks.iter().collect();
    late_blocks_sorted.sort();
    for block_hash in late_blocks_sorted {
        let indices = index(block_hash)?;
        late_set_vec.extend(indices);
    }

    // Accumulator for deploys that lose their chain via dedup but have no
    // fresher copy elsewhere. These are treated the same as conflict-rejected
    // deploys downstream — added to the rejected-deploy buffer so the
    // recovery path can re-propose them in a subsequent block.
    let mut collateral_lost_pairs: Vec<(Bytes, BlockHash)> = Vec::new();

    // Deploy de-duplication. When the same deploy ID appears in chains from
    // multiple blocks in scope — for example, because a previously-rejected
    // deploy was re-proposed in a later block — keep the copy from the freshest
    // source: higher block number first, then lexicographically-smaller block
    // hash as a deterministic tiebreak. A chain containing any deploy whose
    // freshest source is a different chain is dropped; its diffs were computed
    // against a pre-state that the fresh execution replaces.
    if !actual_set_vec.is_empty() {
        // Find the freshest source for each deploy_id across all chains.
        let mut latest_for_deploy: HashMap<Bytes, (i64, BlockHash)> = HashMap::new();
        for chain in &actual_set_vec {
            for deploy in &chain.deploys_with_cost.0 {
                let candidate = (chain.source_block_number, chain.source_block_hash.clone());
                match latest_for_deploy.get(&deploy.deploy_id) {
                    Some((best_num, best_hash)) => {
                        // Fresher = higher block number, or byte-lex smaller hash at tie.
                        let is_fresher = candidate.0 > *best_num
                            || (candidate.0 == *best_num && candidate.1 < *best_hash);
                        if is_fresher {
                            latest_for_deploy.insert(deploy.deploy_id.clone(), candidate);
                        }
                    }
                    None => {
                        latest_for_deploy.insert(deploy.deploy_id.clone(), candidate);
                    }
                }
            }
        }

        // Retain chains only if every deploy in the chain points back to THIS chain
        // as the freshest source. A chain with even one stale deploy is discarded —
        // its diffs are against a pre-state that includes the stale deploy's effects,
        // which are being dropped.
        //
        // Dropping a chain with multiple deploys can cost "collateral": deploys in
        // the dropped chain whose IDs have no fresher copy elsewhere are effectively
        // lost. Collect those sigs so the rejected-deploy buffer can re-propose
        // them in a later block, mirroring how conflict-rejected deploys recover.
        let pre_dedup_count = actual_set_vec.len();
        let (retained, dropped): (Vec<_>, Vec<_>) = std::mem::take(&mut actual_set_vec)
            .into_iter()
            .partition(|chain| {
                chain.deploys_with_cost.0.iter().all(|deploy| {
                    match latest_for_deploy.get(&deploy.deploy_id) {
                        Some((best_num, best_hash)) => {
                            chain.source_block_number == *best_num
                                && chain.source_block_hash == *best_hash
                        }
                        None => true,
                    }
                })
            });
        actual_set_vec = retained;
        let post_dedup_count = actual_set_vec.len();

        for chain in &dropped {
            for deploy in chain.deploys_with_cost.0.iter() {
                if is_system_deploy_id(&deploy.deploy_id) {
                    continue;
                }
                let best = latest_for_deploy.get(&deploy.deploy_id);
                let is_collateral = match best {
                    Some((best_num, best_hash)) => {
                        chain.source_block_number == *best_num
                            && chain.source_block_hash == *best_hash
                    }
                    None => true,
                };
                if is_collateral {
                    collateral_lost_pairs
                        .push((deploy.deploy_id.clone(), chain.source_block_hash.clone()));
                }
            }
        }

        if post_dedup_count < pre_dedup_count {
            tracing::info!(
                "DagMerger dedup: dropped {} stale chain(s) ({} -> {}), collateral deploys={}",
                pre_dedup_count - post_dedup_count,
                pre_dedup_count,
                post_dedup_count,
                collateral_lost_pairs.len(),
            );
        }
    }

    // Sort the deploy chain indices for deterministic iteration order
    actual_set_vec.sort();
    late_set_vec.sort();

    // Log state change details for debugging merge issues
    for (i, chain) in actual_set_vec.iter().enumerate() {
        tracing::debug!(
            target: "f1r3fly.dag_merger.state_changes",
            "deploy_chain[{}]: datums={}, conts={}, joins={}, deploys={}, cost={}",
            i,
            chain.state_changes.datums_changes.len(),
            chain.state_changes.cont_changes.len(),
            chain.state_changes.consume_channels_to_join_serialized_map.len(),
            chain.deploys_with_cost.0.len(),
            chain.deploys_with_cost.0.iter().map(|d| d.cost).sum::<u64>(),
        );
    }

    // Keep as Vec for deterministic processing (ConflictSetMerger expects sorted Vecs)
    let actual_seq = actual_set_vec;
    let late_seq = late_set_vec;

    // Pre-computed data for a single DeployChainIndex, cached by pointer address
    // to avoid recomputing on every O(D²) depends() call.
    struct ChainDerived {
        produces_created: HashableSet<rspace_plus_plus::rspace::trace::event::Produce>,
        consumes_created: HashableSet<rspace_plus_plus::rspace::trace::event::Consume>,
    }

    // Pre-computed data for a branch (HashableSet<DeployChainIndex>), cached by
    // pointer address to avoid recomputing on every O(B²) conflicts() call.
    struct BranchDerived {
        user_deploy_ids: HashSet<Bytes>,
        combined_event_log: rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex,
    }

    fn compute_branch_derived(branch: &HashableSet<DeployChainIndex>) -> BranchDerived {
        let user_deploy_ids: HashSet<_> = branch
            .0
            .iter()
            .flat_map(|chain| chain.deploys_with_cost.0.iter())
            .filter(|deploy| !is_system_deploy_id(&deploy.deploy_id))
            .map(|deploy| deploy.deploy_id.clone())
            .collect();

        let combined_event_log = branch
            .0
            .iter()
            .filter(|idx| {
                idx.deploys_with_cost
                    .0
                    .iter()
                    .all(|d| !is_system_deploy_id(&d.deploy_id))
            })
            .map(|chain| &chain.event_log_index)
            .fold(
                rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex::empty(),
                |acc, index| {
                    rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex::combine(
                        &acc, index,
                    )
                },
            );

        BranchDerived {
            user_deploy_ids,
            combined_event_log,
        }
    }

    // Lazy caches keyed by pointer address. Safe because:
    // - References come from HashSet iteration, addresses stable during iteration
    // - DerivedSets/BranchDerived are pure functions of the item
    let chain_cache: RefCell<HashMap<usize, ChainDerived>> = RefCell::new(HashMap::new());
    let branch_cache: RefCell<HashMap<usize, BranchDerived>> = RefCell::new(HashMap::new());

    let get_chain_derived = |chain: &DeployChainIndex| -> usize {
        let addr = std::ptr::addr_of!(*chain) as usize;
        let mut cache = chain_cache.borrow_mut();
        cache.entry(addr).or_insert_with(|| ChainDerived {
            produces_created: merging_logic::produces_created_and_not_destroyed(
                &chain.event_log_index,
            ),
            consumes_created: merging_logic::consumes_created_and_not_destroyed(
                &chain.event_log_index,
            ),
        });
        addr
    };

    let get_branch_derived = |branch: &HashableSet<DeployChainIndex>| -> usize {
        let addr = std::ptr::addr_of!(*branch) as usize;
        let mut cache = branch_cache.borrow_mut();
        cache
            .entry(addr)
            .or_insert_with(|| compute_branch_derived(branch));
        addr
    };

    // Create history reader for base state
    let history_reader = std::sync::Arc::new(
        history_repository
            .get_history_reader(lfb_post_state)
            .map_err(|e| CasperError::HistoryError(e))?,
    );

    // Bind merge-logic closures to named variables so both resolve_conflicts
    // and compute_merged_state can take them by reference, with the rejection
    // expansion step interposed between the two calls.
    let depends_fn = |target: &DeployChainIndex, source: &DeployChainIndex| -> bool {
        // Cached depends: pre-computes source's derived sets on first access
        let source_addr = get_chain_derived(source);
        let cache = chain_cache.borrow();
        let derived = cache.get(&source_addr).unwrap();

        let produces_source: HashSet<_> = derived
            .produces_created
            .0
            .difference(&source.event_log_index.produces_mergeable.0)
            .collect();
        let produces_target: HashSet<_> = target
            .event_log_index
            .produces_consumed
            .0
            .difference(&source.event_log_index.produces_mergeable.0)
            .collect();

        if produces_source
            .intersection(&produces_target)
            .next()
            .is_some()
        {
            return true;
        }

        derived
            .consumes_created
            .0
            .intersection(&target.event_log_index.consumes_produced.0)
            .next()
            .is_some()
    };

    let conflicts_fn = |as_set: &HashableSet<DeployChainIndex>,
                        bs_set: &HashableSet<DeployChainIndex>|
     -> bool {
        // Cached branch conflicts: pre-computes deploy IDs and combined event log
        let a_addr = get_branch_derived(as_set);
        let b_addr = get_branch_derived(bs_set);
        let cache = branch_cache.borrow();
        let a_derived = cache.get(&a_addr).unwrap();
        let b_derived = cache.get(&b_addr).unwrap();

        let same_deploy = a_derived
            .user_deploy_ids
            .intersection(&b_derived.user_deploy_ids)
            .next()
            .is_some();

        if same_deploy {
            return true;
        }

        merging_logic::are_conflicting(&a_derived.combined_event_log, &b_derived.combined_event_log)
    };

    let state_changes_fn = |chain: &DeployChainIndex| Ok(chain.state_changes.clone());

    let mergeable_channels_fn =
        |chain: &DeployChainIndex| chain.event_log_index.number_channels_data.clone();

    let compute_trie_actions_fn = {
        let reader = Arc::clone(&history_reader);
        move |changes, mergeable_chs| {
            state_change_merger::compute_trie_actions(
                &changes,
                &*reader,
                &mergeable_chs,
                |hash: &Blake2b256Hash, channel_changes, number_chs: &NumberChannelsDiff| {
                    if let Some(number_ch_val) = number_chs.get(hash) {
                        let base_get_data = |h: &Blake2b256Hash| reader.get_data(h);
                        Ok(Some(RholangMergingLogic::calculate_number_channel_merge(
                            hash,
                            *number_ch_val,
                            channel_changes,
                            base_get_data,
                        )))
                    } else {
                        Ok(None)
                    }
                },
            )
        }
    };

    let apply_trie_actions_fn = |actions| {
        history_repository
            .reset(lfb_post_state)
            .and_then(|reset_repo| Ok(reset_repo.do_checkpoint(actions)))
            .map(|checkpoint| checkpoint.root())
            .map_err(|e| e.into())
    };

    let get_data_fn = |hash| history_reader.get_data(&hash).map_err(|e| e.into());

    // Resolve conflicts: detect conflicts and select the cost-optimal rejection set.
    let mut resolved = conflict_set_merger::resolve_conflicts(
        actual_seq,
        late_seq,
        &depends_fn,
        &conflicts_fn,
        &rejection_cost_f,
        &mergeable_channels_fn,
        &get_data_fn,
    )
    .map_err(|e| CasperError::HistoryError(e))?;

    // Rejection expansion. Any chain whose source block is a DAG descendant
    // of a rejected chain's source block (within merge scope) has pre-computed
    // diffs against a pre-state that the merge is about to drop. Expand the
    // rejection set to include those chains before applying diffs.
    //
    // All descendant chains are rejected unconditionally; an event-log
    // read/write analysis could narrow this, but event logs miss indirect
    // dependencies through system contracts (the very condition the expansion
    // exists to catch), so we prefer conservative correctness.
    let rejected_source_blocks: HashSet<BlockHash> = resolved
        .rejected
        .0
        .iter()
        .map(|chain| chain.source_block_hash.clone())
        .collect();

    let pre_expansion_rejected = resolved.rejected.0.len();

    if !rejected_source_blocks.is_empty() {
        let descendant_blocks =
            descendants_within_scope(dag, &rejected_source_blocks, &actual_blocks);

        if !descendant_blocks.is_empty() {
            // Rebuild to_merge: any branch containing a chain from a descendant
            // block gets moved whole into rejected. Branches are dependency
            // clusters — rejecting partial branches would leave internally
            // inconsistent diffs.
            let mut new_to_merge: Vec<HashableSet<DeployChainIndex>> = Vec::new();
            for branch in resolved.to_merge.drain(..) {
                let has_stale = branch
                    .0
                    .iter()
                    .any(|chain| descendant_blocks.contains(&chain.source_block_hash));
                if has_stale {
                    for chain in branch.0 {
                        resolved.rejected.0.insert(chain);
                    }
                } else {
                    new_to_merge.push(branch);
                }
            }
            resolved.to_merge = new_to_merge;

            tracing::info!(
                "DagMerger rejection expansion: {} descendant blocks, rejected grew from {} to {} chains",
                descendant_blocks.len(),
                pre_expansion_rejected,
                resolved.rejected.0.len()
            );
        }
    }

    // Combine surviving diffs and apply to the LFB post-state.
    let new_state = conflict_set_merger::compute_merged_state(
        &resolved,
        &state_changes_fn,
        &mergeable_channels_fn,
        &compute_trie_actions_fn,
        &apply_trie_actions_fn,
    )
    .map_err(|e| CasperError::HistoryError(e))?;

    let rejected = resolved.rejected;

    // Extract (rejected deploy ID, source block hash) pairs, split by kind.
    // User deploys feed the rejected-deploy buffer for re-proposal. Slash
    // deploys feed the block creator's dedup step so that the slash effect
    // persists in the merge block's body regardless of cost-optimal rejection
    // of the source chain. Non-slash system deploys (close block, heartbeat)
    // are intentionally dropped here — they are atomic with their containing
    // block and have no recovery semantics.
    let all_pairs: Vec<(Bytes, BlockHash)> = rejected
        .0
        .iter()
        .flat_map(|chain| {
            let src = chain.source_block_hash.clone();
            chain
                .deploys_with_cost
                .0
                .iter()
                .map(move |deploy| (deploy.deploy_id.clone(), src.clone()))
        })
        .collect();

    let mut rejected_user_deploys: Vec<(Bytes, BlockHash)> = all_pairs
        .iter()
        .filter(|(id, _)| !is_system_deploy_id(id))
        .cloned()
        .collect();
    let mut rejected_slashes: Vec<(Bytes, BlockHash)> = all_pairs
        .into_iter()
        .filter(|(id, _)| is_slash_deploy_id(id))
        .collect();

    // Fold dedup collateral into the rejected-user list so the buffer can
    // recover deploys whose chain was dropped for reasons other than
    // cost-optimal rejection. Keep the list unique per deploy_id — a deploy
    // already present from conflict rejection takes precedence.
    if !collateral_lost_pairs.is_empty() {
        let existing_ids: HashSet<Bytes> = rejected_user_deploys
            .iter()
            .map(|(id, _)| id.clone())
            .collect();
        for pair in collateral_lost_pairs {
            if !existing_ids.contains(&pair.0) {
                rejected_user_deploys.push(pair);
            }
        }
    }

    // Deterministic ordering across validators.
    rejected_user_deploys.sort();
    rejected_slashes.sort();

    tracing::debug!(
        "DagMerger.merge: LFB={}, scope={}, actual={}, late={}, rejected_user={}, rejected_slash={}",
        hex::encode(&lfb[..std::cmp::min(8, lfb.len())]),
        scope
            .as_ref()
            .map_or("ALL".to_string(), |s| s.len().to_string()),
        actual_blocks.len(),
        late_blocks.len(),
        rejected_user_deploys.len(),
        rejected_slashes.len(),
    );

    if !rejected_user_deploys.is_empty() {
        let rejected_str: Vec<_> = rejected_user_deploys
            .iter()
            .map(|(sig, _)| hex::encode(&sig[..std::cmp::min(8, sig.len())]))
            .collect();
        tracing::info!(
            "DagMerger rejected {} user deploys: {}",
            rejected_user_deploys.len(),
            rejected_str.join(", ")
        );
    }
    if !rejected_slashes.is_empty() {
        let rejected_str: Vec<_> = rejected_slashes
            .iter()
            .map(|(sig, _)| hex::encode(&sig[..std::cmp::min(8, sig.len())]))
            .collect();
        tracing::info!(
            "DagMerger rejected {} slashes: {}",
            rejected_slashes.len(),
            rejected_str.join(", ")
        );
    }

    Ok((new_state, rejected_user_deploys, rejected_slashes))
}
