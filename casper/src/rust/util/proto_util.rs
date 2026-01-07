// See casper/src/main/scala/coop/rchain/casper/util/ProtoUtil.scala

use std::collections::HashSet;

use block_storage::rust::{
    dag::block_dag_key_value_storage::KeyValueDagRepresentation,
    key_value_block_store::KeyValueBlockStore,
};
use crypto::rust::{hash::blake2b256::Blake2b256, signatures::signed::Signed};
use models::{
    casper::{BondInfo, JustificationInfo},
    rhoapi::{expr::ExprInstance, Expr, Par},
    rust::{
        block_hash::BlockHash,
        block_metadata::BlockMetadata,
        casper::{
            pretty_printer::PrettyPrinter,
            protocol::casper_message::{
                BlockMessage, Body, Bond, DeployData, Header, Justification, ProcessedDeploy,
                ProcessedSystemDeploy,
            },
        },
        validator::Validator,
    },
};
use rholang::rust::interpreter::deploy_parameters::DeployParameters;
use shared::rust::{store::key_value_store::KvStoreError, ByteString};

pub fn get_main_chain_until_depth(
    block_store: &KeyValueBlockStore,
    estimate: BlockMessage,
    mut acc: Vec<BlockMessage>,
    depth: i32,
) -> Result<Vec<BlockMessage>, KvStoreError> {
    let parents_hashes = parent_hashes(&estimate);
    let maybe_main_parent_hash = parents_hashes.first();
    match maybe_main_parent_hash {
        Some(main_parent_hash) => {
            let updated_estimate = block_store.get_unsafe(main_parent_hash);
            let depth_delta = block_number(&updated_estimate) - block_number(&estimate);
            let new_depth = depth + depth_delta as i32;
            if new_depth <= 0 {
                acc.push(estimate);
                Ok(acc)
            } else {
                acc.push(estimate);
                get_main_chain_until_depth(block_store, updated_estimate, acc, new_depth)
            }
        }
        None => {
            acc.push(estimate);
            Ok(acc)
        }
    }
}

pub fn creator_justification_block_message(block: &BlockMessage) -> Option<Justification> {
    block
        .justifications
        .iter()
        .find(|j| j.validator == block.sender)
        .cloned()
}

pub fn creator_justification_block_metadata(block: &BlockMetadata) -> Option<Justification> {
    block
        .justifications
        .iter()
        .find(|j| j.validator == block.sender)
        .cloned()
}

/// Get creator justification as list until goal in memory
/// Since the creator justification is unique, we don't need to return a list.
/// However, the bfTraverseF requires a list to be returned.
/// When we reach the goalFunc, we return an empty list.
pub fn get_creator_justification_as_list_until_goal_in_memory(
    dag: &KeyValueDagRepresentation,
    block_hash: &BlockHash,
    goal_func: impl Fn(&BlockHash) -> bool,
) -> Result<Vec<BlockHash>, KvStoreError> {
    match dag.lookup(block_hash)? {
        Some(block) => {
            // Find creator justification hash
            let creator_justification = block
                .justifications
                .iter()
                .find(|j| j.validator == block.sender)
                .map(|j| &j.latest_block_hash);

            match creator_justification {
                Some(creator_justification_hash) => {
                    // Look up creator justification metadata
                    match dag.lookup(creator_justification_hash)? {
                        Some(creator_justification) => {
                            // Check if goal is reached
                            if goal_func(&creator_justification.block_hash) {
                                Ok(Vec::new())
                            } else {
                                Ok(vec![creator_justification.block_hash.clone()])
                            }
                        }
                        None => Ok(Vec::new()),
                    }
                }
                None => Ok(Vec::new()),
            }
        }
        None => Ok(Vec::new()),
    }
}

/// Get weight map from a block message
pub fn weight_map(
    block_message: &BlockMessage,
) -> std::collections::HashMap<prost::bytes::Bytes, i64> {
    weight_map_from_state(&block_message.body.state)
}

/// Get weight map from a state
fn weight_map_from_state(
    state: &models::rust::casper::protocol::casper_message::F1r3flyState,
) -> std::collections::HashMap<prost::bytes::Bytes, i64> {
    state
        .bonds
        .iter()
        .map(|bond| (bond.validator.clone(), bond.stake))
        .collect()
}

/// Get total weight from a weight map
pub fn weight_map_total(weights: &std::collections::HashMap<ByteString, i64>) -> i64 {
    weights.values().sum()
}

/// Get minimum total validator weight
pub fn min_total_validator_weight(
    dag: &mut KeyValueDagRepresentation,
    block_hash: &BlockHash,
    max_clique_min_size: i32,
) -> Result<i64, KvStoreError> {
    dag.lookup(block_hash).map(|block_metadata_opt| {
        let block_metadata = block_metadata_opt.expect("Block metadata should exist");
        let mut sorted_weights: Vec<i64> = block_metadata.weight_map.values().cloned().collect();
        sorted_weights.sort();
        sorted_weights
            .iter()
            .take(max_clique_min_size as usize)
            .sum()
    })
}

/// Get main parent of a block
pub fn main_parent(
    block_store: &mut KeyValueBlockStore,
    block_message: &BlockMessage,
) -> Result<Option<BlockMessage>, KvStoreError> {
    match block_message.header.parents_hash_list.first() {
        Some(parent_hash) => block_store.get(parent_hash),
        None => Ok(None),
    }
}

/// Get weight from validator by dag
pub fn weight_from_validator_by_dag(
    dag: &mut KeyValueDagRepresentation,
    block_hash: &BlockHash,
    validator: &Validator,
) -> Result<i64, KvStoreError> {
    // Get block metadata
    let block_metadata = dag
        .lookup(block_hash)?
        .expect("Block metadata should exist");

    // Try to get parent's weight for this validator
    match block_metadata.parents.first() {
        Some(parent_hash) => {
            // Look up parent
            let parent_metadata = dag
                .lookup(parent_hash)?
                .expect("Parent metadata should exist");
            // Return validator's weight from parent or 0 if not found
            Ok(parent_metadata
                .weight_map
                .get(validator)
                .cloned()
                .unwrap_or(0))
        }
        None => {
            // No parents (genesis) - use current block's weight map
            Ok(block_metadata
                .weight_map
                .get(validator)
                .cloned()
                .unwrap_or(0))
        }
    }
}

/// Get weight from validator
pub fn weight_from_validator(
    block_store: &mut KeyValueBlockStore,
    b: &BlockMessage,
    validator: &prost::bytes::Bytes,
) -> Result<i64, KvStoreError> {
    // Get main parent
    let maybe_main_parent = main_parent(block_store, b)?;

    // Get weight from validator (from parent or current block)
    let weight = match maybe_main_parent {
        Some(parent) => weight_map(&parent).get(validator).cloned().unwrap_or(0),
        None => weight_map(b).get(validator).cloned().unwrap_or(0), // No parents means genesis - use itself
    };

    Ok(weight)
}

/// Get weight from sender
pub fn weight_from_sender(
    block_store: &mut KeyValueBlockStore,
    b: &BlockMessage,
) -> Result<i64, KvStoreError> {
    weight_from_validator(block_store, b, &b.sender)
}

pub fn parent_hashes(block: &BlockMessage) -> Vec<prost::bytes::Bytes> {
    block
        .header
        .parents_hash_list
        .iter()
        .map(|bytes| bytes.clone())
        .collect()
}

pub fn get_parents(block_store: &KeyValueBlockStore, block: &BlockMessage) -> Vec<BlockMessage> {
    parent_hashes(block)
        .iter()
        .map(|bytes| block_store.get_unsafe(bytes))
        .collect()
}

pub fn get_parents_metadata(
    dag: &KeyValueDagRepresentation,
    block: &BlockMetadata,
) -> Result<Vec<BlockMetadata>, KvStoreError> {
    // Use lookup() instead of lookup_unsafe() to handle race conditions gracefully
    // In parallel test execution, parent metadata might not be persisted yet
    block
        .parents
        .iter()
        .filter_map(|parent| match dag.lookup(parent) {
            Ok(Some(metadata)) => Some(Ok(metadata)),
            Ok(None) => None, // Skip parents not yet available
            Err(e) => Some(Err(e)),
        })
        .collect()
}

pub fn get_parent_metadatas_above_block_number(
    block: &BlockMetadata,
    block_number: i64,
    dag: &KeyValueDagRepresentation,
) -> Result<Vec<BlockMetadata>, KvStoreError> {
    get_parents_metadata(dag, block).map(|parents| {
        parents
            .into_iter()
            .filter(|p| p.block_number >= block_number)
            .collect()
    })
}

pub fn deploys(block: &BlockMessage) -> Vec<ProcessedDeploy> {
    block.body.deploys.clone()
}

pub fn system_deploys(block: &BlockMessage) -> Vec<ProcessedSystemDeploy> {
    block.body.system_deploys.clone()
}

pub fn post_state_hash(block: &BlockMessage) -> prost::bytes::Bytes {
    block.body.state.post_state_hash.clone()
}

pub fn pre_state_hash(block: &BlockMessage) -> prost::bytes::Bytes {
    block.body.state.pre_state_hash.clone()
}

pub fn bonds(block: &BlockMessage) -> Vec<Bond> {
    block.body.state.bonds.clone()
}

pub fn block_number(block: &BlockMessage) -> i64 {
    block.body.state.block_number
}

pub fn bond_to_bond_info(bond: &Bond) -> BondInfo {
    BondInfo {
        validator: PrettyPrinter::build_string_no_limit(&bond.validator),
        stake: bond.stake,
    }
}

pub fn max_block_number_metadata(blocks: &Vec<BlockMetadata>) -> i64 {
    blocks
        .iter()
        .fold(-1, |acc, block| std::cmp::max(acc, block.block_number))
}

pub fn justifications_to_justification_infos(justification: &Justification) -> JustificationInfo {
    JustificationInfo {
        validator: PrettyPrinter::build_string_no_limit(&justification.validator),
        latest_block_hash: PrettyPrinter::build_string_no_limit(&justification.latest_block_hash),
    }
}

pub fn to_justification(
    latest_messages: std::collections::HashMap<Validator, BlockMetadata>,
) -> Vec<Justification> {
    latest_messages
        .into_iter()
        .map(|(validator, block_metadata)| Justification {
            validator,
            latest_block_hash: block_metadata.block_hash,
        })
        .collect()
}

pub fn to_latest_message_hashes(
    justifications: Vec<Justification>,
) -> std::collections::HashMap<Validator, BlockHash> {
    justifications
        .into_iter()
        .map(|justification| (justification.validator, justification.latest_block_hash))
        .collect()
}

pub fn to_latest_message(
    justifications: &Vec<Justification>,
    dag: &mut KeyValueDagRepresentation,
) -> Result<std::collections::HashMap<Validator, BlockMetadata>, KvStoreError> {
    let mut latest_messages = std::collections::HashMap::new();
    for justification in justifications {
        let block_metadata = dag.lookup(&justification.latest_block_hash)?;
        match block_metadata {
            Some(block_metadata) => {
                latest_messages.insert(justification.validator.clone(), block_metadata);
            }
            None => {
                return Err(KvStoreError::KeyNotFound(format!(
                    "Could not find a block for {} in the DAG storage",
                    PrettyPrinter::build_string_bytes(&justification.latest_block_hash)
                )));
            }
        }
    }
    Ok(latest_messages)
}

pub fn block_header(parent_hashes: Vec<ByteString>, version: i64, timestamp: i64) -> Header {
    Header {
        parents_hash_list: parent_hashes
            .into_iter()
            .map(|bytes| bytes.into())
            .collect(),
        timestamp,
        version,
        extra_bytes: prost::bytes::Bytes::new(),
    }
}

pub fn unsigned_block_proto(
    body: Body,
    header: Header,
    justifications: Vec<Justification>,
    shard_id: String,
    seq_num: Option<i32>,
) -> BlockMessage {
    let seq_num = seq_num.unwrap_or(0);
    let mut block = BlockMessage {
        block_hash: prost::bytes::Bytes::new(),
        header,
        body,
        justifications,
        sender: prost::bytes::Bytes::new(),
        seq_num,
        sig: prost::bytes::Bytes::new(),
        sig_algorithm: "".to_string(),
        shard_id,
        extra_bytes: prost::bytes::Bytes::new(),
    };

    let hash = hash_block(&block);
    block.block_hash = hash.into();
    block
}

pub fn hash_block(block: &BlockMessage) -> BlockHash {
    use prost::Message;

    let bytes: Vec<u8> = block
        .header
        .to_proto()
        .encode_to_vec()
        .into_iter()
        .chain(block.body.to_proto().encode_to_vec().into_iter())
        .chain(block.sender.clone().into_iter())
        .chain(block.sig_algorithm.as_bytes().to_vec().into_iter())
        .chain(block.seq_num.to_le_bytes().into_iter())
        .chain(block.shard_id.as_bytes().to_vec().into_iter())
        .chain(block.extra_bytes.clone().into_iter())
        .collect();

    Blake2b256::hash(bytes).into()
}

pub fn hash_string(b: &BlockMessage) -> BlockHash {
    use prost::Message;
    hex::encode(b.block_hash.encode_to_vec()).into()
}

pub fn compute_code_hash(dd: &DeployData) -> Par {
    let term = dd.term.as_bytes();
    let hash = Blake2b256::hash(term.to_vec());
    Par {
        exprs: vec![Expr {
            expr_instance: Some(ExprInstance::GByteArray(hash)),
        }],
        ..Default::default()
    }
}

pub fn get_rholang_deploy_params(dd: &Signed<DeployData>) -> DeployParameters {
    let user_id: Par = Par {
        exprs: vec![Expr {
            expr_instance: Some(ExprInstance::GByteArray(dd.pk.bytes.to_vec())),
        }],
        ..Default::default()
    };

    DeployParameters { user_id }
}

pub fn dependencies_hashes_of(b: &BlockMessage) -> Vec<BlockHash> {
    let missing_parents: HashSet<BlockHash> = parent_hashes(b).into_iter().collect();
    let missing_justifications: HashSet<BlockHash> = b
        .justifications
        .iter()
        .map(|j| j.latest_block_hash.clone())
        .collect();

    (missing_parents.union(&missing_justifications))
        .cloned()
        .collect()
}

// Return hashes of all blocks that are yet to be seen by the passed in block
pub fn unseen_block_hashes(
    dag: &mut KeyValueDagRepresentation,
    block: &BlockMessage,
) -> Result<HashSet<BlockHash>, KvStoreError> {
    let dags_latest_messages = dag.latest_messages()?;
    let blocks_latest_messages = to_latest_message(&block.justifications, dag)?;

    // From input block perspective we want to find what latest messages are not seen
    //  that are in the DAG latest messages.
    // - if validator is not in the justification of the block
    // - if justification contains validator's newer latest message
    let unseen_latest_messages =
        dags_latest_messages
            .into_iter()
            .filter(|(validator, dag_latest_message)| {
                let validator_in_justification = blocks_latest_messages.contains_key(validator);
                let block_has_newer_latest_message =
                    blocks_latest_messages
                        .get(validator)
                        .map(|block_latest_message| {
                            dag_latest_message.sequence_number
                                > block_latest_message.sequence_number
                        });

                !validator_in_justification
                    || (validator_in_justification
                        && block_has_newer_latest_message.unwrap_or(false))
            });

    // Collect all unseen block hashes
    let mut all_unseen_blocks = HashSet::new();
    for (validator, unseen_latest_message) in unseen_latest_messages {
        let validator_latest_message = blocks_latest_messages.get(&validator);
        let creator_blocks =
            get_creator_blocks_between(dag, &unseen_latest_message, validator_latest_message)?;
        all_unseen_blocks.extend(creator_blocks);
    }

    // Remove blocks that are already in the block's justifications
    for block_metadata in blocks_latest_messages.values() {
        all_unseen_blocks.remove(&block_metadata.block_hash);
    }

    // Remove the current block hash
    all_unseen_blocks.remove(&block.block_hash);

    Ok(all_unseen_blocks)
}

fn get_creator_blocks_between(
    dag: &mut KeyValueDagRepresentation,
    top_block: &BlockMetadata,
    bottom_block: Option<&BlockMetadata>,
) -> Result<HashSet<BlockHash>, KvStoreError> {
    match bottom_block {
        Some(bottom_block) => {
            // Use the bf_traverse function from dag_ops for breadth-first traversal
            let neighbor_fn = |block: &BlockMetadata| -> Vec<BlockMetadata> {
                get_creator_justification_unless_goal(dag, block, bottom_block).unwrap_or_default()
            };

            // Start traversal from top_block
            let traversal_result =
                shared::rust::dag::dag_ops::bf_traverse(vec![top_block.clone()], neighbor_fn);

            // Collect all block hashes into a HashSet
            let blocks_set: HashSet<BlockHash> = traversal_result
                .into_iter()
                .map(|block| block.block_hash.clone())
                .collect();

            Ok(blocks_set)
        }

        None => Ok(HashSet::from([top_block.block_hash.clone()])),
    }
}

fn get_creator_justification_unless_goal(
    dag: &mut KeyValueDagRepresentation,
    block: &BlockMetadata,
    goal: &BlockMetadata,
) -> Result<Vec<BlockMetadata>, KvStoreError> {
    match creator_justification_block_metadata(block) {
        Some(Justification {
            validator: _,
            latest_block_hash,
        }) => match dag.lookup(&latest_block_hash) {
            Ok(Some(creator_justification)) => {
                if creator_justification == *goal {
                    Ok(vec![])
                } else {
                    Ok(vec![creator_justification])
                }
            }

            _ => Err(KvStoreError::KeyNotFound(format!(
                "BlockDAG is missing justification {} for {}",
                PrettyPrinter::build_string_bytes(&latest_block_hash),
                PrettyPrinter::build_string_bytes(&block.block_hash)
            ))),
        },

        None => Ok(vec![]),
    }
}

pub fn justification_to_justification_info(justification: &Justification) -> JustificationInfo {
    JustificationInfo {
        validator: PrettyPrinter::build_string_no_limit(&justification.validator),
        latest_block_hash: PrettyPrinter::build_string_no_limit(&justification.latest_block_hash),
    }
}
