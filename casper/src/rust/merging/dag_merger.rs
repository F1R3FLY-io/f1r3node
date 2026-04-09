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
use crate::rust::{errors::CasperError, system_deploy::is_system_deploy_id};

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

pub fn merge(
    dag: &KeyValueDagRepresentation,
    lfb: &BlockHash,
    lfb_post_state: &Blake2b256Hash,
    index: impl Fn(&BlockHash) -> Result<Vec<DeployChainIndex>, CasperError>,
    history_repository: &RhoHistoryRepository,
    rejection_cost_f: impl Fn(&DeployChainIndex) -> u64,
    scope: Option<HashSet<BlockHash>>,
    disable_late_block_filtering: bool,
) -> Result<(Blake2b256Hash, Vec<Bytes>), CasperError> {
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

    // Sort the deploy chain indices for deterministic iteration order
    actual_set_vec.sort();
    late_set_vec.sort();

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

        BranchDerived { user_deploy_ids, combined_event_log }
    }

    // Lazy caches keyed by pointer address. Safe because:
    // - References come from HashSet iteration, addresses stable during iteration
    // - DerivedSets/BranchDerived are pure functions of the item
    let chain_cache: RefCell<HashMap<usize, ChainDerived>> = RefCell::new(HashMap::new());
    let branch_cache: RefCell<HashMap<usize, BranchDerived>> = RefCell::new(HashMap::new());

    let get_chain_derived = |chain: &DeployChainIndex| -> usize {
        let addr = std::ptr::addr_of!(*chain) as usize;
        let mut cache = chain_cache.borrow_mut();
        cache.entry(addr).or_insert_with(|| {
            ChainDerived {
                produces_created: merging_logic::produces_created_and_not_destroyed(&chain.event_log_index),
                consumes_created: merging_logic::consumes_created_and_not_destroyed(&chain.event_log_index),
            }
        });
        addr
    };

    let get_branch_derived = |branch: &HashableSet<DeployChainIndex>| -> usize {
        let addr = std::ptr::addr_of!(*branch) as usize;
        let mut cache = branch_cache.borrow_mut();
        cache.entry(addr).or_insert_with(|| compute_branch_derived(branch));
        addr
    };

    // Create history reader for base state
    let history_reader = std::sync::Arc::new(
        history_repository
            .get_history_reader(lfb_post_state)
            .map_err(|e| CasperError::HistoryError(e))?,
    );

    // Use ConflictSetMerger to perform the actual merge
    let result = conflict_set_merger::merge(
        actual_seq,
        late_seq,
        |target: &DeployChainIndex, source: &DeployChainIndex| {
            // Cached depends: pre-computes source's derived sets on first access
            let source_addr = get_chain_derived(source);
            let cache = chain_cache.borrow();
            let derived = cache.get(&source_addr).unwrap();

            let produces_source: HashSet<_> = derived.produces_created
                .0
                .difference(&source.event_log_index.produces_mergeable.0)
                .collect();
            let produces_target: HashSet<_> = target.event_log_index
                .produces_consumed
                .0
                .difference(&source.event_log_index.produces_mergeable.0)
                .collect();

            if produces_source.intersection(&produces_target).next().is_some() {
                return true;
            }

            derived.consumes_created
                .0
                .intersection(&target.event_log_index.consumes_produced.0)
                .next()
                .is_some()
        },
        |as_set: &HashableSet<DeployChainIndex>, bs_set: &HashableSet<DeployChainIndex>| {
            // Cached branch conflicts: pre-computes deploy IDs and combined event log
            let a_addr = get_branch_derived(as_set);
            let b_addr = get_branch_derived(bs_set);
            let cache = branch_cache.borrow();
            let a_derived = cache.get(&a_addr).unwrap();
            let b_derived = cache.get(&b_addr).unwrap();

            let same_deploy = a_derived.user_deploy_ids
                .intersection(&b_derived.user_deploy_ids)
                .next()
                .is_some();

            if same_deploy {
                return true;
            }

            merging_logic::are_conflicting(&a_derived.combined_event_log, &b_derived.combined_event_log)
        },
        rejection_cost_f,
        |chain: &DeployChainIndex| Ok(chain.state_changes.clone()),
        |chain: &DeployChainIndex| chain.event_log_index.number_channels_data.clone(),
        {
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
        },
        |actions| {
            history_repository
                .reset(lfb_post_state)
                .and_then(|reset_repo| Ok(reset_repo.do_checkpoint(actions)))
                .map(|checkpoint| checkpoint.root())
                .map_err(|e| e.into())
        },
        |hash| history_reader.get_data(&hash).map_err(|e| e.into()),
    )
    .map_err(|e| CasperError::HistoryError(e))?;

    let (new_state, rejected) = result;

    // Extract rejected deploy IDs, filtering out system deploy IDs
    // System deploys are deterministic and excluded from conflict detection
    let mut rejected_deploys: Vec<Bytes> = rejected
        .0
        .iter()
        .flat_map(|chain| chain.deploys_with_cost.0.iter())
        .map(|deploy| deploy.deploy_id.clone())
        .filter(|id| !is_system_deploy_id(id))
        .collect();

    // Sort rejected deploys to ensure deterministic ordering
    rejected_deploys.sort();

    // Log merge summary at debug level
    tracing::debug!(
        "DagMerger.merge: LFB={}, scope={}, actual={}, late={}, rejected={}",
        hex::encode(&lfb[..std::cmp::min(8, lfb.len())]),
        scope
            .as_ref()
            .map_or("ALL".to_string(), |s| s.len().to_string()),
        actual_blocks.len(),
        late_blocks.len(),
        rejected_deploys.len()
    );

    // Log rejected deploys at info level only if there are any
    if !rejected_deploys.is_empty() {
        let rejected_str: Vec<_> = rejected_deploys
            .iter()
            .map(|bs| hex::encode(&bs[..std::cmp::min(8, bs.len())]))
            .collect();
        tracing::info!(
            "DagMerger rejected {} deploys: {}",
            rejected_deploys.len(),
            rejected_str.join(", ")
        );
    }

    Ok((new_state, rejected_deploys))
}
