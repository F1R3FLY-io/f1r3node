// See casper/src/main/scala/coop/rchain/casper/merging/DagMerger.scala

use prost::bytes::Bytes;
use std::collections::HashSet;
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
use crate::rust::errors::CasperError;

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

/// Helper function to detect system deploy IDs.
/// System deploy IDs are 33 bytes: [32-byte blockHash][1-byte marker]
/// Markers: 0x01 (slash), 0x02 (close block), 0x03 (empty/heartbeat)
fn is_system_deploy_id(id: &Bytes) -> bool {
    id.len() == 33 && {
        let last_byte = id[32];
        last_byte == 1 || last_byte == 2 || last_byte == 3
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
    // Get ancestors of LFB (blocks whose state is already included in LFB's post-state)
    let lfb_ancestors = dag.with_ancestors(lfb.clone(), |_| true)?;

    // Blocks to merge are all blocks in scope that are NOT the LFB or its ancestors.
    // This includes:
    // 1. Descendants of LFB (blocks built on top of LFB)
    // 2. Siblings of LFB (blocks at same height but different branch) that are ancestors of the tips
    // Previously we only included descendants, which missed deploy effects from sibling branches.
    // Note: lfb_ancestors includes the LFB itself (via with_ancestors)
    let actual_blocks: HashSet<BlockHash> = match &scope {
        Some(scope_blocks) => {
            // Include all scope blocks except LFB and its ancestors
            scope_blocks.difference(&lfb_ancestors).cloned().collect()
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
        "DagMerger.merge: LFB={}, scope={}, actualBlocks (above LFB)={}, lfbAncestors={}, lateBlocks={}",
        hex::encode(&lfb[..std::cmp::min(8, lfb.len())]),
        scope.as_ref().map_or("ALL".to_string(), |s| format!("{} blocks", s.len())),
        actual_blocks.len(),
        lfb_ancestors.len(),
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

    // Check if branches are conflicting
    let branches_are_conflicting =
        |as_set: &HashableSet<DeployChainIndex>, bs_set: &HashableSet<DeployChainIndex>| {
            // Filter out system deploy IDs - they should never be treated as conflicts
            // System deploys are deterministic and execute the same way in all branches
            let as_user_deploys: HashSet<_> = as_set
                .0
                .iter()
                .flat_map(|chain| chain.deploys_with_cost.0.iter())
                .filter(|deploy| !is_system_deploy_id(&deploy.deploy_id))
                .map(|deploy| &deploy.deploy_id)
                .collect();

            let bs_user_deploys: HashSet<_> = bs_set
                .0
                .iter()
                .flat_map(|chain| chain.deploys_with_cost.0.iter())
                .filter(|deploy| !is_system_deploy_id(&deploy.deploy_id))
                .map(|deploy| &deploy.deploy_id)
                .collect();

            // Check if user deploy IDs intersect
            let same_deploy_in_both = as_user_deploys.intersection(&bs_user_deploys).count() > 0;

            // Also filter system deploys from event log comparison
            let as_user_indices: Vec<_> = as_set
                .0
                .iter()
                .filter(|idx| {
                    idx.deploys_with_cost
                        .0
                        .iter()
                        .all(|d| !is_system_deploy_id(&d.deploy_id))
                })
                .collect();

            let bs_user_indices: Vec<_> = bs_set
                .0
                .iter()
                .filter(|idx| {
                    idx.deploys_with_cost
                        .0
                        .iter()
                        .all(|d| !is_system_deploy_id(&d.deploy_id))
                })
                .collect();

            // Check if event logs are conflicting (using only user deploy indices)
            let as_combined = as_user_indices
                .iter()
                .map(|chain| &chain.event_log_index)
                .fold(
                    rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex::empty(),
                    |acc, index| {
                        rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex::combine(
                            &acc, index,
                        )
                    },
                );
            let bs_combined = bs_user_indices
                .iter()
                .map(|chain| &chain.event_log_index)
                .fold(
                    rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex::empty(),
                    |acc, index| {
                        rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex::combine(
                            &acc, index,
                        )
                    },
                );

            let event_log_conflict = merging_logic::are_conflicting(&as_combined, &bs_combined);

            // Debug: log conflict reason when conflict is detected
            if same_deploy_in_both || event_log_conflict {
                let as_deploy_ids: Vec<_> = as_user_deploys
                    .iter()
                    .map(|id| hex::encode(&id[..std::cmp::min(8, id.len())]))
                    .collect();
                let bs_deploy_ids: Vec<_> = bs_user_deploys
                    .iter()
                    .map(|id| hex::encode(&id[..std::cmp::min(8, id.len())]))
                    .collect();

                let conflict_reason_str = if same_deploy_in_both {
                    let intersection: Vec<_> = as_user_deploys
                        .intersection(&bs_user_deploys)
                        .map(|id| hex::encode(&id[..std::cmp::min(8, id.len())]))
                        .collect();
                    format!("sameDeployInBoth: {}", intersection.join(","))
                } else {
                    merging_logic::conflict_reason(&as_combined, &bs_combined)
                        .unwrap_or_else(|| "unknown".to_string())
                };

                tracing::debug!(
                    "[DEBUG] CONFLICT DETECTED: [{}] vs [{}] - reason: {}",
                    as_deploy_ids.join(","),
                    bs_deploy_ids.join(","),
                    conflict_reason_str
                );
            }

            same_deploy_in_both || event_log_conflict
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
            merging_logic::depends(&target.event_log_index, &source.event_log_index)
        },
        branches_are_conflicting,
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
