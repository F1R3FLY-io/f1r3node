// See casper/src/main/scala/coop/rchain/casper/blocks/proposer/BlockCreator.scala

use dashmap::DashSet;
use log;
use prost::bytes::Bytes;
use std::sync::{Arc, Mutex};
use std::{collections::HashSet, time::SystemTime};

use block_storage::rust::{
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    key_value_block_store::KeyValueBlockStore,
};
use crypto::rust::{private_key::PrivateKey, signatures::signed::Signed};
use models::rust::casper::pretty_printer;
use models::rust::casper::protocol::casper_message::{
    BlockMessage, Body, Bond, DeployData, F1r3flyState, Header, Justification, ProcessedDeploy,
    ProcessedSystemDeploy, RejectedDeploy,
};

use rholang::rust::interpreter::system_processes::BlockData;

use crate::rust::util::construct_deploy;
use crate::rust::util::rholang::{
    costacc::{close_block_deploy::CloseBlockDeploy, slash_deploy::SlashDeploy},
    interpreter_util, system_deploy_util,
};
use crate::rust::{
    blocks::proposer::propose_result::BlockCreatorResult,
    casper::CasperSnapshot,
    errors::CasperError,
    util::{proto_util, rholang::runtime_manager::RuntimeManager},
    validator_identity::ValidatorIdentity,
};

/*
 * Overview of createBlock
 *
 *  1. Rank each of the block cs's latest messages (blocks) via the LMD GHOST estimator.
 *  2. Let each latest message have a score of 2^(-i) where i is the index of that latest message in the ranking.
 *     Take a subset S of the latest messages such that the sum of scores is the greatest and
 *     none of the blocks in S conflicts with each other. S will become the parents of the
 *     about-to-be-created block.
 *  3. Extract all valid deploys that aren't already in all ancestors of S (the parents).
 *  4. Create a new block that contains the deploys from the previous step.
 */
async fn prepare_user_deploys(
    casper_snapshot: &CasperSnapshot,
    block_number: i64,
    deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
) -> Result<HashSet<Signed<DeployData>>, CasperError> {
    let deploy_storage_guard = deploy_storage
        .lock()
        .map_err(|e| CasperError::LockError(e.to_string()))?;

    // Read all unfinalized deploys from storage
    let unfinalized = deploy_storage_guard.read_all()?;

    let earliest_block_number =
        block_number - casper_snapshot.on_chain_state.shard_conf.deploy_lifespan;

    // Filter valid deploys (not expired and not future)
    let valid: DashSet<Signed<DeployData>> = unfinalized
        .into_iter()
        .filter(|deploy| {
            not_future_deploy(block_number, &deploy.data)
                && not_expired_deploy(earliest_block_number, &deploy.data)
        })
        .collect();

    // Remove deploys that are already in scope to prevent resending
    let valid_unique: HashSet<Signed<DeployData>> = valid
        .into_iter()
        .filter(|deploy| !casper_snapshot.deploys_in_scope.contains(deploy))
        .collect();

    Ok(valid_unique)
}

async fn prepare_slashing_deploys(
    casper_snapshot: &CasperSnapshot,
    validator_identity: &ValidatorIdentity,
    seq_num: i32,
) -> Result<Vec<SlashDeploy>, CasperError> {
    let self_id = Bytes::copy_from_slice(&validator_identity.public_key.bytes);

    // Get invalid latest messages from DAG
    let invalid_latest_messages = casper_snapshot.dag.invalid_latest_messages()?;

    // Filter to only include bonded validators
    let bonded_invalid_messages: Vec<_> = invalid_latest_messages
        .into_iter()
        .filter(|(validator, _)| {
            casper_snapshot
                .on_chain_state
                .bonds_map
                .get(validator)
                .map(|stake| *stake > 0)
                .unwrap_or(false)
        })
        .collect();

    // TODO: Add `slashingDeploys` to DeployStorage - OLD
    // Create SlashDeploy objects
    let mut slashing_deploys = Vec::new();
    for (_, invalid_block_hash) in bonded_invalid_messages {
        let slash_deploy = SlashDeploy {
            invalid_block_hash: invalid_block_hash.clone(),
            pk: validator_identity.public_key.clone(),
            initial_rand: system_deploy_util::generate_slash_deploy_random_seed(
                self_id.clone(),
                seq_num,
            ),
        };

        log::info!(
            "Issuing slashing deploy justified by block {}",
            pretty_printer::PrettyPrinter::build_string_bytes(&invalid_block_hash)
        );

        slashing_deploys.push(slash_deploy);
    }

    Ok(slashing_deploys)
}

fn prepare_dummy_deploy(
    block_number: i64,
    shard_id: String,
    dummy_deploy_opt: Option<(PrivateKey, String)>,
) -> Result<Vec<Signed<DeployData>>, CasperError> {
    match dummy_deploy_opt {
        Some((private_key, term)) => {
            let deploy = construct_deploy::source_deploy_now(
                term,
                Some(private_key),
                Some(block_number - 1),
                Some(shard_id),
            )
            .map_err(|e| {
                CasperError::RuntimeError(format!("Failed to create dummy deploy: {}", e))
            })?;
            Ok(vec![deploy])
        }
        None => Ok(Vec::new()),
    }
}

pub async fn create(
    casper_snapshot: &CasperSnapshot,
    validator_identity: &ValidatorIdentity,
    dummy_deploy_opt: Option<(PrivateKey, String)>,
    deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    runtime_manager: &mut RuntimeManager,
    block_store: &mut KeyValueBlockStore,
    allow_empty_blocks: bool,
) -> Result<BlockCreatorResult, CasperError> {
    let next_seq_num = casper_snapshot
        .max_seq_nums
        .get(&validator_identity.public_key.bytes)
        .map(|seq| *seq + 1)
        .unwrap_or(1) as i32;
    let next_block_num = casper_snapshot.max_block_num + 1;
    let parents = &casper_snapshot.parents;
    let justifications = &casper_snapshot.justifications;

    log::info!(
        "Creating block #{} (seqNum {})",
        next_block_num,
        next_seq_num
    );

    let shard_id = casper_snapshot.on_chain_state.shard_conf.shard_name.clone();

    // Prepare deploys
    let user_deploys =
        prepare_user_deploys(casper_snapshot, next_block_num, deploy_storage).await?;
    let dummy_deploys = prepare_dummy_deploy(next_block_num, shard_id.clone(), dummy_deploy_opt)?;
    let slashing_deploys =
        prepare_slashing_deploys(casper_snapshot, validator_identity, next_seq_num).await?;

    // Combine all deploys, removing those already in scope
    let mut all_deploys: HashSet<Signed<DeployData>> = user_deploys
        .into_iter()
        .filter(|deploy| !casper_snapshot.deploys_in_scope.contains(deploy))
        .collect();

    // Add dummy deploys
    all_deploys.extend(dummy_deploys);

    // Check if we have any deploys to process
    // Scala: if (deploys.nonEmpty || slashingDeploys.nonEmpty)
    // Note: system_deploys always contains CloseBlockDeploy, but that doesn't count
    // as "new deploys" for the purpose of creating a block
    let has_slashing_deploys = !slashing_deploys.is_empty();
    if !allow_empty_blocks && all_deploys.is_empty() && !has_slashing_deploys {
        return Ok(BlockCreatorResult::NoNewDeploys);
    }

    // Make sure closeBlock is the last system Deploy
    let mut system_deploys_converted = Vec::new();

    // TODO why we push CloseBlockDeploy instead of SlashDeploy?
    // Add slashing deploys (converted to CloseBlockDeploy for now - this needs proper system deploy handling)
    for slash_deploy in slashing_deploys {
        system_deploys_converted.push(CloseBlockDeploy {
            initial_rand: slash_deploy.initial_rand,
        });
    }

    // Add the actual close block deploy
    system_deploys_converted.push(CloseBlockDeploy {
        initial_rand: system_deploy_util::generate_close_deploy_random_seed_from_pk(
            validator_identity.public_key.clone(),
            next_seq_num,
        ),
    });

    // Get current time
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|e| CasperError::RuntimeError(format!("Failed to get current time: {}", e)))?
        .as_millis() as i64;

    let invalid_blocks = casper_snapshot.invalid_blocks.clone();
    let block_data = BlockData {
        time_stamp: now,
        block_number: next_block_num,
        sender: validator_identity.public_key.clone(),
        seq_num: next_seq_num,
    };

    // Compute checkpoint data
    let checkpoint_data = interpreter_util::compute_deploys_checkpoint(
        block_store,
        parents.clone(),
        all_deploys.into_iter().collect(),
        system_deploys_converted,
        casper_snapshot,
        runtime_manager,
        block_data.clone(),
        invalid_blocks,
    )
    .await?;

    let (
        pre_state_hash,
        post_state_hash,
        processed_deploys,
        rejected_deploys,
        processed_system_deploys,
    ) = checkpoint_data;

    // Compute new bonds
    let new_bonds = runtime_manager.compute_bonds(&post_state_hash).await?;

    let casper_version = casper_snapshot.on_chain_state.shard_conf.casper_version;

    // Create unsigned block
    let unsigned_block = package_block(
        &block_data,
        parents.iter().map(|p| p.block_hash.clone()).collect(),
        justifications.iter().map(|j| j.clone()).collect(),
        pre_state_hash,
        post_state_hash,
        processed_deploys,
        rejected_deploys,
        processed_system_deploys,
        new_bonds,
        shard_id,
        casper_version,
    );

    // Sign the block
    let signed_block = validator_identity.sign_block(&unsigned_block);

    let block_info = pretty_printer::PrettyPrinter::build_string_block_message(&signed_block, true);
    let deploy_count = signed_block.body.deploys.len();
    log::info!("Block created: {} ({}d)", block_info, deploy_count);

    Ok(BlockCreatorResult::Created(signed_block))
}

fn package_block(
    block_data: &BlockData,
    parents: Vec<Bytes>,
    justifications: Vec<Justification>,
    pre_state_hash: Bytes,
    post_state_hash: Bytes,
    deploys: Vec<ProcessedDeploy>,
    rejected_deploys: Vec<Bytes>,
    system_deploys: Vec<ProcessedSystemDeploy>,
    bonds_map: Vec<Bond>,
    shard_id: String,
    version: i64,
) -> BlockMessage {
    let state = F1r3flyState {
        pre_state_hash,
        post_state_hash,
        bonds: bonds_map,
        block_number: block_data.block_number,
    };

    let rejected_deploys_wrapped: Vec<RejectedDeploy> = rejected_deploys
        .into_iter()
        .map(|r| RejectedDeploy { sig: r })
        .collect();

    let body = Body {
        state,
        deploys,
        rejected_deploys: rejected_deploys_wrapped,
        system_deploys,
        extra_bytes: Bytes::new(),
    };

    let header = Header {
        parents_hash_list: parents,
        timestamp: block_data.time_stamp,
        version,
        extra_bytes: Bytes::new(),
    };

    proto_util::unsigned_block_proto(
        body,
        header,
        justifications,
        shard_id,
        Some(block_data.seq_num as i32),
    )
}

fn not_expired_deploy(earliest_block_number: i64, deploy_data: &DeployData) -> bool {
    deploy_data.valid_after_block_number > earliest_block_number
}

fn not_future_deploy(current_block_number: i64, deploy_data: &DeployData) -> bool {
    deploy_data.valid_after_block_number < current_block_number
}
