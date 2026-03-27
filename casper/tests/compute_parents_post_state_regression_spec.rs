use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use block_storage::rust::{
    dag::block_dag_key_value_storage::{BlockDagKeyValueStorage, KeyValueDagRepresentation},
    key_value_block_store::KeyValueBlockStore,
};
use casper::rust::{
    casper::{CasperShardConf, CasperSnapshot, OnChainCasperState},
    errors::CasperError,
    genesis::contracts::{proof_of_stake::ProofOfStake, validator::Validator as GenesisValidator},
    genesis::genesis::Genesis,
    util::{
        proto_util,
        rholang::{
            interpreter_util::{compute_deploys_checkpoint, compute_parents_post_state},
            runtime_manager::RuntimeManager,
            system_deploy_enum::SystemDeployEnum,
        },
    },
};
use crypto::rust::signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg};
use dashmap::{DashMap, DashSet};
use models::rust::{
    block::state_hash::StateHash,
    block_hash::BlockHash,
    block_implicits,
    casper::protocol::casper_message::{BlockMessage, Bond},
    validator::Validator,
};
use prost::bytes::Bytes;
use rholang::rust::interpreter::{
    external_services::ExternalServices, system_processes::BlockData,
};
use rspace_plus_plus::rspace::shared::{
    in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
};

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn mk_snapshot(
    dag: KeyValueDagRepresentation,
    validator: Validator,
    shard_name: String,
    last_finalized_block: BlockHash,
) -> CasperSnapshot {
    let mut snapshot = CasperSnapshot::new(dag);
    snapshot.last_finalized_block = last_finalized_block;

    let max_seq_nums: DashMap<Validator, u64> = DashMap::new();
    max_seq_nums.insert(validator.clone(), 0);
    snapshot.max_seq_nums = max_seq_nums;

    let mut shard_conf = CasperShardConf::new();
    shard_conf.shard_name = shard_name;
    shard_conf.max_parent_depth = 0;
    shard_conf.disable_late_block_filtering = false;
    shard_conf.disable_validator_progress_check = false;

    let mut bonds_map = HashMap::new();
    bonds_map.insert(validator.clone(), 100);
    snapshot.on_chain_state = OnChainCasperState {
        shard_conf,
        bonds_map,
        active_validators: vec![validator],
    };
    snapshot.deploys_in_scope = std::sync::Arc::new(DashSet::new());
    snapshot
}

fn build_empty_block(
    block_number: i64,
    seq_num: i32,
    creator: Validator,
    parent_hashes: Vec<BlockHash>,
    pre_state_hash: StateHash,
    bonds: Vec<Bond>,
    shard_id: String,
) -> BlockMessage {
    block_implicits::get_random_block(
        Some(block_number),
        Some(seq_num),
        Some(pre_state_hash),
        Some(StateHash::default()),
        Some(creator),
        Some(1),
        Some(now_millis()),
        Some(parent_hashes),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(bonds),
        Some(shard_id),
        None,
    )
}

async fn step_block(
    block_store: &mut KeyValueBlockStore,
    dag_storage: &BlockDagKeyValueStorage,
    runtime_manager: &mut RuntimeManager,
    block: &BlockMessage,
    validator: Validator,
    shard_name: String,
    last_finalized_block: BlockHash,
) -> Result<BlockMessage, CasperError> {
    let snapshot = mk_snapshot(
        dag_storage.get_representation(),
        validator,
        shard_name,
        last_finalized_block,
    );

    let parents = proto_util::get_parents(block_store, block);
    let deploys = proto_util::deploys(block)
        .into_iter()
        .map(|d| d.deploy)
        .collect::<Vec<_>>();

    let (_, post_state_hash, processed_deploys, _, processed_system_deploys, bonds) =
        compute_deploys_checkpoint(
            block_store,
            parents,
            deploys,
            Vec::<SystemDeployEnum>::new(),
            &snapshot,
            runtime_manager,
            BlockData::from_block(block),
            HashMap::new(),
        )
        .await?;

    let mut updated = block.clone();
    updated.body.state.post_state_hash = post_state_hash;
    updated.body.deploys = processed_deploys;
    updated.body.system_deploys = processed_system_deploys;
    updated.body.state.bonds = bonds;

    block_store
        .put_block_message(&updated)
        .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
    dag_storage
        .insert(&updated, false, false)
        .map_err(|e| CasperError::RuntimeError(e.to_string()))?;

    Ok(updated)
}

#[test]
fn compute_parents_post_state_should_not_depend_on_local_finalized_set() {
    let stack_bytes = std::env::var("F1R3_COMPUTE_PARENTS_REGRESSION_STACK_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64 * 1024 * 1024);

    let handle = std::thread::Builder::new()
        .name("compute-parents-post-state-regression".to_string())
        .stack_size(stack_bytes)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build Tokio runtime");
            runtime.block_on(run_compute_parents_post_state_finalized_skew_regression());
        })
        .expect("Failed to spawn regression test thread");

    handle
        .join()
        .expect("Regression test thread panicked before completing");
}

async fn run_compute_parents_post_state_finalized_skew_regression() {
    let secp = Secp256k1;
    let (_validator_sk, validator_pk) = secp.new_key_pair();
    let validator: Bytes = validator_pk.bytes.clone().into();
    let shard_name = "test-shard".to_string();

    let mut kvm = InMemoryStoreManager::new();
    let mut block_store = KeyValueBlockStore::create_from_kvm(&mut kvm)
        .await
        .expect("Failed to create block store");
    let dag_storage = BlockDagKeyValueStorage::new(&mut kvm)
        .await
        .expect("Failed to create DAG storage");

    let rspace_store = kvm
        .r_space_stores()
        .await
        .expect("Failed to get rspace stores");
    let mergeable_store = RuntimeManager::mergeable_store(&mut kvm)
        .await
        .expect("Failed to create mergeable store");
    let (mut runtime_manager, _) = RuntimeManager::create_with_history(
        rspace_store,
        mergeable_store,
        Genesis::non_negative_mergeable_tag_name(),
        ExternalServices::noop(),
    );

    let genesis = Genesis {
        shard_id: shard_name.clone(),
        timestamp: 0,
        block_number: 0,
        proof_of_stake: ProofOfStake {
            minimum_bond: 1,
            maximum_bond: i64::MAX,
            validators: vec![GenesisValidator {
                pk: validator_pk.clone(),
                stake: 100,
            }],
            epoch_length: 1000,
            quarantine_length: 50000,
            number_of_active_validators: 1,
            pos_multi_sig_public_keys: vec![
                "04db91a53a2b72fcdcb201031772da86edad1e4979eb6742928d27731b1771e0bc40c9e9c9fa6554bdec041a87cee423d6f2e09e9dfb408b78e85a4aa611aad20c".to_string(),
                "042a736b30fffcc7d5a58bb9416f7e46180818c82b15542d0a7819d1a437aa7f4b6940c50db73a67bfc5f5ec5b5fa555d24ef8339b03edaa09c096de4ded6eae14".to_string(),
                "047f0f0f5bbe1d6d1a8dac4d88a3957851940f39a57cd89d55fe25b536ab67e6d76fd3f365c83e5bfe11fe7117e549b1ae3dd39bfc867d1c725a4177692c4e7754".to_string(),
            ],
            pos_multi_sig_quorum: 2,
        },
        vaults: Vec::new(),
        supply: i64::MAX,
        version: 1,
    };

    let genesis_block = Genesis::create_genesis_block(&mut runtime_manager, &genesis)
        .await
        .expect("Failed to create genesis block");
    block_store
        .put_block_message(&genesis_block)
        .expect("Failed to store genesis block");
    dag_storage
        .insert(&genesis_block, false, true)
        .expect("Failed to insert genesis in DAG");

    let b1_raw = build_empty_block(
        1,
        1,
        validator.clone(),
        vec![genesis_block.block_hash.clone()],
        proto_util::post_state_hash(&genesis_block),
        genesis_block.body.state.bonds.clone(),
        shard_name.clone(),
    );
    let b1 = step_block(
        &mut block_store,
        &dag_storage,
        &mut runtime_manager,
        &b1_raw,
        validator.clone(),
        shard_name.clone(),
        genesis_block.block_hash.clone(),
    )
    .await
    .expect("Failed to step b1");

    let b2_raw = build_empty_block(
        2,
        2,
        validator.clone(),
        vec![b1.block_hash.clone()],
        proto_util::post_state_hash(&b1),
        b1.body.state.bonds.clone(),
        shard_name.clone(),
    );
    let b2 = step_block(
        &mut block_store,
        &dag_storage,
        &mut runtime_manager,
        &b2_raw,
        validator.clone(),
        shard_name.clone(),
        genesis_block.block_hash.clone(),
    )
    .await
    .expect("Failed to step b2");

    let b3_raw = build_empty_block(
        2,
        3,
        validator.clone(),
        vec![b1.block_hash.clone()],
        proto_util::post_state_hash(&b1),
        b1.body.state.bonds.clone(),
        shard_name.clone(),
    );
    let b3 = step_block(
        &mut block_store,
        &dag_storage,
        &mut runtime_manager,
        &b3_raw,
        validator.clone(),
        shard_name.clone(),
        genesis_block.block_hash.clone(),
    )
    .await
    .expect("Failed to step b3");

    let parents = vec![b2.clone(), b3.clone()];

    let mut snapshot_without_skew = mk_snapshot(
        dag_storage.get_representation(),
        validator.clone(),
        shard_name.clone(),
        genesis_block.block_hash.clone(),
    );
    snapshot_without_skew.dag.last_finalized_block_hash = genesis_block.block_hash.clone();

    let (state_without_skew, rejected_without_skew) = compute_parents_post_state(
        &block_store,
        parents.clone(),
        &snapshot_without_skew,
        &runtime_manager,
        None,
        None,
    )
    .expect("Failed to compute parents post-state without finalized skew");

    runtime_manager.parents_post_state_cache.clear();
    runtime_manager.block_index_cache.clear();

    let mut snapshot_with_skew = mk_snapshot(
        dag_storage.get_representation(),
        validator,
        shard_name,
        genesis_block.block_hash.clone(),
    );
    snapshot_with_skew.dag.last_finalized_block_hash = genesis_block.block_hash.clone();
    snapshot_with_skew
        .dag
        .finalized_blocks_set
        .insert(b1.block_hash.clone());

    let (state_with_skew, rejected_with_skew) = compute_parents_post_state(
        &block_store,
        parents,
        &snapshot_with_skew,
        &runtime_manager,
        None,
        None,
    )
    .expect("Failed to compute parents post-state with finalized skew");

    assert_eq!(
        state_without_skew, state_with_skew,
        "Parents post-state should be invariant to finalized-set skew for the same parent set."
    );
    assert_eq!(
        rejected_without_skew, rejected_with_skew,
        "Rejected deploy set should be invariant to finalized-set skew for the same parent set."
    );
}

#[test]
fn compute_parents_post_state_should_fail_when_required_mergeable_is_missing() {
    let stack_bytes = std::env::var("F1R3_COMPUTE_PARENTS_REGRESSION_STACK_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64 * 1024 * 1024);

    let handle = std::thread::Builder::new()
        .name("compute-parents-post-state-missing-mergeable".to_string())
        .stack_size(stack_bytes)
        .spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build Tokio runtime");
            runtime.block_on(run_compute_parents_post_state_missing_mergeable_regression());
        })
        .expect("Failed to spawn regression test thread");

    handle
        .join()
        .expect("Regression test thread panicked before completing");
}

async fn run_compute_parents_post_state_missing_mergeable_regression() {
    let secp = Secp256k1;
    let (_validator_sk, validator_pk) = secp.new_key_pair();
    let validator: Bytes = validator_pk.bytes.clone().into();
    let shard_name = "test-shard".to_string();

    let mut kvm = InMemoryStoreManager::new();
    let mut block_store = KeyValueBlockStore::create_from_kvm(&mut kvm)
        .await
        .expect("Failed to create block store");
    let dag_storage = BlockDagKeyValueStorage::new(&mut kvm)
        .await
        .expect("Failed to create DAG storage");

    let rspace_store = kvm
        .r_space_stores()
        .await
        .expect("Failed to get rspace stores");
    let mergeable_store = RuntimeManager::mergeable_store(&mut kvm)
        .await
        .expect("Failed to create mergeable store");
    let (mut runtime_manager, _) = RuntimeManager::create_with_history(
        rspace_store,
        mergeable_store,
        Genesis::non_negative_mergeable_tag_name(),
        ExternalServices::noop(),
    );

    let genesis = Genesis {
        shard_id: shard_name.clone(),
        timestamp: 0,
        block_number: 0,
        proof_of_stake: ProofOfStake {
            minimum_bond: 1,
            maximum_bond: i64::MAX,
            validators: vec![GenesisValidator {
                pk: validator_pk.clone(),
                stake: 100,
            }],
            epoch_length: 1000,
            quarantine_length: 50000,
            number_of_active_validators: 1,
            pos_multi_sig_public_keys: vec![
                "04db91a53a2b72fcdcb201031772da86edad1e4979eb6742928d27731b1771e0bc40c9e9c9fa6554bdec041a87cee423d6f2e09e9dfb408b78e85a4aa611aad20c".to_string(),
                "042a736b30fffcc7d5a58bb9416f7e46180818c82b15542d0a7819d1a437aa7f4b6940c50db73a67bfc5f5ec5b5fa555d24ef8339b03edaa09c096de4ded6eae14".to_string(),
                "047f0f0f5bbe1d6d1a8dac4d88a3957851940f39a57cd89d55fe25b536ab67e6d76fd3f365c83e5bfe11fe7117e549b1ae3dd39bfc867d1c725a4177692c4e7754".to_string(),
            ],
            pos_multi_sig_quorum: 2,
        },
        vaults: Vec::new(),
        supply: i64::MAX,
        version: 1,
    };

    let genesis_block = Genesis::create_genesis_block(&mut runtime_manager, &genesis)
        .await
        .expect("Failed to create genesis block");
    block_store
        .put_block_message(&genesis_block)
        .expect("Failed to store genesis block");
    dag_storage
        .insert(&genesis_block, false, true)
        .expect("Failed to insert genesis in DAG");

    let b1_raw = build_empty_block(
        1,
        1,
        validator.clone(),
        vec![genesis_block.block_hash.clone()],
        proto_util::post_state_hash(&genesis_block),
        genesis_block.body.state.bonds.clone(),
        shard_name.clone(),
    );
    let b1 = step_block(
        &mut block_store,
        &dag_storage,
        &mut runtime_manager,
        &b1_raw,
        validator.clone(),
        shard_name.clone(),
        genesis_block.block_hash.clone(),
    )
    .await
    .expect("Failed to step b1");

    let b2_raw = build_empty_block(
        2,
        2,
        validator.clone(),
        vec![b1.block_hash.clone()],
        proto_util::post_state_hash(&b1),
        b1.body.state.bonds.clone(),
        shard_name.clone(),
    );
    let b2 = step_block(
        &mut block_store,
        &dag_storage,
        &mut runtime_manager,
        &b2_raw,
        validator.clone(),
        shard_name.clone(),
        genesis_block.block_hash.clone(),
    )
    .await
    .expect("Failed to step b2");

    let b3_raw = build_empty_block(
        2,
        3,
        validator.clone(),
        vec![b1.block_hash.clone()],
        proto_util::post_state_hash(&b1),
        b1.body.state.bonds.clone(),
        shard_name.clone(),
    );
    let b3 = step_block(
        &mut block_store,
        &dag_storage,
        &mut runtime_manager,
        &b3_raw,
        validator.clone(),
        shard_name.clone(),
        genesis_block.block_hash.clone(),
    )
    .await
    .expect("Failed to step b3");

    let deleted = runtime_manager
        .delete_mergeable_channels(
            &b2.body.state.post_state_hash,
            b2.sender.clone(),
            b2.seq_num,
        )
        .expect("Failed to delete mergeable entry");
    assert!(
        deleted,
        "Expected parent mergeable entry to exist before deletion."
    );

    let mut snapshot = mk_snapshot(
        dag_storage.get_representation(),
        validator,
        shard_name,
        genesis_block.block_hash.clone(),
    );
    snapshot.dag.last_finalized_block_hash = genesis_block.block_hash;

    let result = compute_parents_post_state(
        &block_store,
        vec![b2, b3],
        &snapshot,
        &runtime_manager,
        None,
        None,
    );

    assert!(
        matches!(result, Err(CasperError::KvStoreError(_))),
        "Expected compute_parents_post_state to fail when a required mergeable entry is missing; got {result:?}"
    );
}
