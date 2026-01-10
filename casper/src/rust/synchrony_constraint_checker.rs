// See casper/src/main/scala/coop/rchain/casper/SynchronyConstraintChecker.scala

use std::collections::{HashMap, HashSet};

use block_storage::rust::{
    dag::block_dag_key_value_storage::KeyValueDagRepresentation,
    key_value_block_store::KeyValueBlockStore,
};
use models::rust::{block_metadata::BlockMetadata, validator::Validator};

use crate::rust::util::proto_util;

use super::{
    blocks::proposer::propose_result::CheckProposeConstraintsResult, casper::CasperSnapshot,
    errors::CasperError, util::rholang::runtime_manager::RuntimeManager,
    validator_identity::ValidatorIdentity,
};

pub async fn check(
    snapshot: &CasperSnapshot,
    runtime_manager: &RuntimeManager,
    block_store: &KeyValueBlockStore,
    validator_identity: &ValidatorIdentity,
) -> Result<CheckProposeConstraintsResult, CasperError> {
    let validator = validator_identity.public_key.bytes.clone();
    let main_parent_opt = snapshot.parents.first();
    let synchrony_constraint_threshold = snapshot
        .on_chain_state
        .shard_conf
        .synchrony_constraint_threshold;

    match snapshot.dag.latest_message_hash(&validator) {
        Some(last_proposed_block_hash) => {
            let last_proposed_block_meta = snapshot.dag.lookup_unsafe(&last_proposed_block_hash)?;

            // If validator's latest block is genesis, it's not proposed any block yet and hence allowed to propose once.
            let latest_block_is_genesis = last_proposed_block_meta.block_number == 0;
            if latest_block_is_genesis {
                Ok(CheckProposeConstraintsResult::success())
            } else {
                let main_parent = main_parent_opt.ok_or(CasperError::Other(
                    "Synchrony constraint checker: Parent blocks not found".to_string(),
                ))?;

                let main_parent_meta = snapshot.dag.lookup_unsafe(&main_parent.block_hash)?;

                // Loading the whole block is only needed to get post-state hash
                let main_parent_block = block_store.get_unsafe(&main_parent_meta.block_hash);
                let main_parent_state_hash = proto_util::post_state_hash(&main_parent_block);

                // Get bonds map from PoS
                // NOTE: It would be useful to have active validators cached in the block in the same way as bonds.
                let active_validators = runtime_manager
                    .get_active_validators(&main_parent_state_hash)
                    .await?;

                // Validators weight map filtered by active validators only.
                let validator_weight_map: HashMap<Validator, i64> = main_parent_meta
                    .weight_map
                    .into_iter()
                    .filter(|(validator, _)| active_validators.contains(validator))
                    .collect();

                // Guaranteed to be present since last proposed block was present
                let seen_senders =
                    calculate_seen_senders_since(last_proposed_block_meta, snapshot.dag.clone());

                let seen_senders_weight: i64 = seen_senders
                    .iter()
                    .map(|validator| validator_weight_map.get(validator).unwrap_or(&0))
                    .sum();

                // This method can be called on readonly node or not active validator.
                // So map validator -> stake might not have key associated with the node,
                // that's why we need `getOrElse`
                let validator_own_stake = validator_weight_map.get(&validator).unwrap_or(&0);
                let other_validators_weight =
                    validator_weight_map.values().sum::<i64>() - validator_own_stake;

                // If there is no other active validators, do not put any constraint (value = 1)
                let synchrony_constraint_value = if other_validators_weight == 0 {
                    1.0
                } else {
                    seen_senders_weight as f32 / other_validators_weight as f32
                };

                tracing::warn!(
                    "Seen {} senders with weight {} out of total {} ({:.2} out of {:.2} needed)",
                    seen_senders.len(),
                    seen_senders_weight,
                    other_validators_weight,
                    synchrony_constraint_value,
                    synchrony_constraint_threshold
                );

                if synchrony_constraint_value >= synchrony_constraint_threshold {
                    Ok(CheckProposeConstraintsResult::success())
                } else {
                    Ok(CheckProposeConstraintsResult::not_enough_new_block())
                }
            }
        }
        None => Err(CasperError::Other(
            "Synchrony constraint checker: Validator does not have a latest message".to_string(),
        )),
    }
}

fn calculate_seen_senders_since(
    last_proposed: BlockMetadata,
    dag: KeyValueDagRepresentation,
) -> HashSet<Validator> {
    let latest_messages = dag.latest_message_hashes();

    last_proposed
        .justifications
        .iter()
        .filter_map(|justification| {
            let latest_block_for_validator = latest_messages.get(&justification.validator);

            if justification.validator != last_proposed.sender
                && latest_block_for_validator
                    .map(|value| *value != justification.latest_block_hash)
                    .unwrap_or(false)
            {
                // Since we would have fetched missing justifications initially, it can only mean
                // that we have received at least one new block since then
                Some(justification.validator.clone())
            } else {
                None
            }
        })
        .collect()
}
