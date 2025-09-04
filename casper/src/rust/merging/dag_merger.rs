// See casper/src/main/scala/coop/rchain/casper/merging/DagMerger.scala

use prost::bytes::Bytes;
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

pub fn merge(
    dag: &KeyValueDagRepresentation,
    lfb: &BlockHash,
    lfb_post_state: &Blake2b256Hash,
    index: impl Fn(&BlockHash) -> Result<Vec<DeployChainIndex>, CasperError>,
    history_repository: &RhoHistoryRepository,
    rejection_cost_f: impl Fn(&DeployChainIndex) -> u64,
) -> Result<(Blake2b256Hash, Vec<Bytes>), CasperError> {
    // All not finalized blocks (conflict set)
    let non_finalized_blocks = dag.non_finalized_blocks()?;

    // Blocks that see last finalized state
    let actual_blocks = dag.descendants(lfb)?;

    // Get indices for actual and late blocks efficiently
    let mut actual_set_vec = Vec::new();
    let mut late_set_vec = Vec::new();

    // Process actual blocks
    for block_hash in &actual_blocks {
        let indices = index(block_hash)?;
        actual_set_vec.extend(indices);
    }

    // Process late blocks (blocks that do not see last finalized state)
    // TODO: reject only late units conflicting with finalised body - OLD
    for block_hash in non_finalized_blocks.difference(&actual_blocks) {
        let indices = index(block_hash)?;
        late_set_vec.extend(indices);
    }

    let actual_set = HashableSet(actual_set_vec.into_iter().collect());
    let late_set = HashableSet(late_set_vec.into_iter().collect());

    // Check if branches are conflicting
    let branches_are_conflicting =
        |as_set: &HashableSet<DeployChainIndex>, bs_set: &HashableSet<DeployChainIndex>| {
            // Check if deploy IDs intersect
            let deploy_intersection = as_set.0.iter().any(|as_chain| {
                as_chain.deploys_with_cost.0.iter().any(|as_deploy| {
                    bs_set.0.iter().any(|bs_chain| {
                        bs_chain
                            .deploys_with_cost
                            .0
                            .iter()
                            .any(|bs_deploy| as_deploy.deploy_id == bs_deploy.deploy_id)
                    })
                })
            });

            if deploy_intersection {
                return true;
            }

            // Check if event logs are conflicting
            let as_combined = as_set.0.iter().map(|chain| &chain.event_log_index).fold(
                rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex::empty(),
                |acc, index| {
                    rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex::combine(
                        &acc, index,
                    )
                },
            );
            let bs_combined = bs_set.0.iter().map(|chain| &chain.event_log_index).fold(
                rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex::empty(),
                |acc, index| {
                    rspace_plus_plus::rspace::merger::event_log_index::EventLogIndex::combine(
                        &acc, index,
                    )
                },
            );

            merging_logic::are_conflicting(&as_combined, &bs_combined)
        };

    // Create history reader for base state
    let history_reader = std::sync::Arc::new(
        history_repository
            .get_history_reader(lfb_post_state)
            .map_err(|e| CasperError::HistoryError(e))?,
    );

    // Use ConflictSetMerger to perform the actual merge
    let result = conflict_set_merger::merge(
        actual_set,
        late_set,
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

    // Extract rejected deploy IDs
    let rejected_deploys: Vec<Bytes> = rejected
        .0
        .iter()
        .flat_map(|chain| chain.deploys_with_cost.0.iter())
        .map(|deploy| deploy.deploy_id.clone())
        .collect();

    Ok((new_state, rejected_deploys))
}
