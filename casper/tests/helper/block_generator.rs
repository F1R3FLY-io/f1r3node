// See casper/src/test/scala/coop/rchain/casper/helper/BlockGenerator.scala

use std::{
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};

use block_storage::rust::{
    dag::block_dag_key_value_storage::KeyValueDagRepresentation,
    key_value_block_store::KeyValueBlockStore,
    test::indexed_block_dag_storage::IndexedBlockDagStorage,
};
use casper::rust::{
    casper::CasperSnapshot,
    errors::CasperError,
    util::{
        construct_deploy, proto_util,
        rholang::{
            costacc::close_block_deploy::CloseBlockDeploy,
            interpreter_util::compute_deploys_checkpoint, runtime_manager::RuntimeManager,
        },
    },
};
use models::rust::{
    block::state_hash::StateHash,
    block_hash::BlockHash,
    block_implicits,
    casper::protocol::casper_message::{BlockMessage, Bond, Justification, ProcessedDeploy},
    validator::Validator,
};
use rholang::rust::interpreter::system_processes::BlockData;

pub fn mk_casper_snapshot(dag: KeyValueDagRepresentation) -> CasperSnapshot {
    CasperSnapshot::new(dag)
}

pub async fn step(
    block_dag_storage: &mut IndexedBlockDagStorage,
    block_store: &mut KeyValueBlockStore,
    runtime_manager: &mut RuntimeManager,
    block: &BlockMessage,
) -> Result<(), CasperError> {
    let dag = block_dag_storage.get_representation();
    let (post_b1_state_hash, post_b1_processed_deploys) = compute_block_checkpoint(
        block_store,
        block,
        &mk_casper_snapshot(dag),
        runtime_manager,
    )
    .await?;

    inject_post_state_hash(
        block_store,
        block_dag_storage,
        block,
        post_b1_state_hash,
        post_b1_processed_deploys,
    )
}

async fn compute_block_checkpoint(
    block_store: &mut KeyValueBlockStore,
    block: &BlockMessage,
    casper_snapshot: &CasperSnapshot,
    runtime_manager: &mut RuntimeManager,
) -> Result<(StateHash, Vec<ProcessedDeploy>), CasperError> {
    let parents = proto_util::get_parents(block_store, block);
    let deploys = proto_util::deploys(block)
        .into_iter()
        .map(|d| d.deploy)
        .collect();

    let (_, post_state_hash, processed_deploys, _, _) = compute_deploys_checkpoint(
        block_store,
        parents,
        deploys,
        Vec::new(), // No system deploys
        casper_snapshot,
        runtime_manager,
        BlockData::from_block(block),
        HashMap::new(),
    )
    .await?;

    Ok((post_state_hash, processed_deploys))
}

fn inject_post_state_hash(
    block_store: &mut KeyValueBlockStore,
    block_dag_storage: &mut IndexedBlockDagStorage,
    block: &BlockMessage,
    post_state_hash: StateHash,
    processed_deploys: Vec<ProcessedDeploy>,
) -> Result<(), CasperError> {
    let mut updated_block = block.clone();
    updated_block.body.state.post_state_hash = post_state_hash;
    updated_block.body.deploys = processed_deploys;
    block_store.put(block.block_hash.clone(), &updated_block)?;
    block_dag_storage.insert(&updated_block, false, false)?;
    Ok(())
}

pub fn build_block(
    parents_hash_list: Vec<BlockHash>,
    creator: Option<Validator>,
    now: i64,
    bonds: Option<Vec<Bond>>,
    justifications: Option<Vec<Justification>>,
    deploys: Option<Vec<ProcessedDeploy>>,
    post_state_hash: Option<StateHash>,
    shard_id: Option<String>,
    pre_state_hash: Option<StateHash>,
    seq_num: Option<i32>,
) -> BlockMessage {
    let creator = creator.unwrap_or_default();
    let bonds = bonds.unwrap_or_default();
    let justifications = justifications.unwrap_or_default();
    let deploys = deploys.unwrap_or_default();
    let post_state_hash = post_state_hash.unwrap_or_default();
    let shard_id = shard_id.unwrap_or("root".to_string());
    let pre_state_hash = pre_state_hash.unwrap_or_default();
    let seq_num = seq_num.unwrap_or(0);

    block_implicits::get_random_block(
        None,
        Some(seq_num),
        Some(pre_state_hash),
        Some(post_state_hash),
        Some(creator),
        None,
        Some(now),
        Some(parents_hash_list),
        Some(justifications),
        Some(deploys),
        None,
        Some(bonds),
        Some(shard_id),
        None,
    )
}

pub fn create_genesis_block(
    block_store: &mut KeyValueBlockStore,
    indexed_block_dag_storage: &mut IndexedBlockDagStorage,
    creator: Option<Validator>,
    bonds: Option<Vec<Bond>>,
    justifications: Option<Vec<Justification>>,
    deploys: Option<Vec<ProcessedDeploy>>,
    ts_hash: Option<StateHash>,
    shard_id: Option<String>,
    pre_state_hash: Option<StateHash>,
    seq_num: Option<i32>,
) -> BlockMessage {
    let creator = creator.unwrap_or_default();
    let bonds = bonds.unwrap_or_default();
    let justifications = justifications.unwrap_or_default();
    let deploys = deploys.unwrap_or_default();
    let ts_hash = ts_hash.unwrap_or_default();
    let shard_id = shard_id.unwrap_or("root".to_string());
    let pre_state_hash = pre_state_hash.unwrap_or_default();
    let seq_num = seq_num.unwrap_or(0);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    let genesis = build_block(
        vec![],
        Some(creator),
        now,
        Some(bonds),
        Some(justifications),
        Some(deploys),
        Some(ts_hash),
        Some(shard_id),
        Some(pre_state_hash),
        Some(seq_num),
    );

    let modified_block = indexed_block_dag_storage
        .insert_indexed(&genesis, &genesis, false)
        .unwrap();

    block_store
        .put(genesis.block_hash.clone(), &modified_block)
        .unwrap();

    genesis
}

pub fn create_block(
    block_store: &mut KeyValueBlockStore,
    indexed_block_dag_storage: &mut IndexedBlockDagStorage,
    parents_hash_list: Vec<BlockHash>,
    genesis: &BlockMessage,
    creator: Option<Validator>,
    bonds: Option<Vec<Bond>>,
    justifications: Option<std::collections::HashMap<Validator, BlockHash>>,
    deploys: Option<Vec<ProcessedDeploy>>,
    post_state_hash: Option<StateHash>,
    shard_id: Option<String>,
    pre_state_hash: Option<StateHash>,
    seq_num: Option<i32>,
    invalid: Option<bool>,
) -> BlockMessage {
    let creator = creator.unwrap_or_default();
    let bonds = bonds.unwrap_or_default();
    let justifications = justifications
        .unwrap_or_default()
        .into_iter()
        .map(|(validator, block_hash)| Justification {
            validator,
            latest_block_hash: block_hash,
        })
        .collect();
    let deploys = deploys.unwrap_or_default();
    let post_state_hash = post_state_hash.unwrap_or_default();
    let shard_id = shard_id.unwrap_or("root".to_string());
    let pre_state_hash = pre_state_hash.unwrap_or_default();
    let seq_num = seq_num.unwrap_or(0);
    let invalid = invalid.unwrap_or(false);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    let block = build_block(
        parents_hash_list,
        Some(creator),
        now,
        Some(bonds),
        Some(justifications),
        Some(deploys),
        Some(post_state_hash),
        Some(shard_id),
        Some(pre_state_hash),
        Some(seq_num),
    );

    let modified_block = indexed_block_dag_storage
        .insert_indexed(&block, genesis, invalid)
        .unwrap();

    block_store
        .put(block.block_hash.clone(), &modified_block)
        .unwrap();

    modified_block
}

pub fn create_block_fast(
    block_store: &mut KeyValueBlockStore,
    block_dag_storage: &mut IndexedBlockDagStorage,
    parents: Vec<BlockHash>,
    genesis: &BlockMessage,
) -> BlockMessage {
    create_block(
        block_store,
        block_dag_storage,
        parents,
        genesis,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

pub fn create_block_fast_with_creator(
    block_store: &mut KeyValueBlockStore,
    block_dag_storage: &mut IndexedBlockDagStorage,
    parents: Vec<BlockHash>,
    genesis: &BlockMessage,
    creator: Validator,
) -> BlockMessage {
    create_block(
        block_store,
        block_dag_storage,
        parents,
        genesis,
        Some(creator),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

pub fn create_validator_block(
    block_store: &mut KeyValueBlockStore,
    indexed_block_dag_storage: &mut IndexedBlockDagStorage,
    parents: Vec<BlockMessage>,
    genesis: &BlockMessage,
    justifications: Vec<BlockMessage>,
    validator: Validator,
    bonds: Vec<Bond>,
    seq_num: Option<i32>,
    invalid: Option<bool>,
    shard_id: String,
) -> BlockMessage {
    let deploy = construct_deploy::basic_processed_deploy(0, Some(shard_id.clone())).unwrap();

    let justifications_map: std::collections::HashMap<Validator, BlockHash> = justifications
        .into_iter()
        .map(|b| (b.sender, b.block_hash))
        .collect();

    create_block(
        block_store,
        indexed_block_dag_storage,
        parents.into_iter().map(|p| p.block_hash).collect(),
        genesis,
        Some(validator),
        Some(bonds),
        Some(justifications_map),
        Some(vec![deploy]),
        None,
        Some(shard_id),
        None,
        seq_num,
        invalid,
    )
}
