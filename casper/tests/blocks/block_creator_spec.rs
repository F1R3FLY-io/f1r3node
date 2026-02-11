// See casper/src/test/scala/coop/rchain/casper/blocks/proposer/BlockCreatorSpec.scala
//
// Unit tests for BlockCreator.
// Tests the deploy preparation and cleanup logic.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use block_storage::rust::{
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    key_value_block_store::KeyValueBlockStore,
};
use casper::rust::{
    blocks::proposer::block_creator,
    casper::{CasperShardConf, CasperSnapshot, OnChainCasperState},
    util::rholang::runtime_manager::RuntimeManager,
    validator_identity::ValidatorIdentity,
};
use crypto::rust::{
    private_key::PrivateKey,
    signatures::{secp256k1::Secp256k1, signed::Signed},
};
use dashmap::{DashMap, DashSet};
use models::rust::casper::protocol::casper_message::DeployData;
use models::ByteString;
use prost::bytes::Bytes;
use rspace_plus_plus::rspace::shared::{
    in_mem_store_manager::InMemoryStoreManager,
    key_value_store_manager::KeyValueStoreManager,
};

use crate::util::genesis_builder::DEFAULT_VALIDATOR_SKS;
use crate::util::rholang::resources;

const DEPLOY_LIFESPAN: i64 = 50;

/// Creates a signed deploy with the given parameters
fn create_deploy(
    valid_after_block_number: i64,
    expiration_timestamp: Option<i64>,
    validator_sk: &PrivateKey,
) -> Signed<DeployData> {
    let deploy_data = DeployData {
        term: format!("new x in {{ x!({}) }}", valid_after_block_number),
        time_stamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0),
        phlo_price: 1,
        phlo_limit: 1000,
        valid_after_block_number,
        shard_id: "test-shard".to_string(),
        expiration_timestamp,
    };

    Signed::create(
        deploy_data,
        Box::new(Secp256k1),
        validator_sk.clone(),
    )
    .expect("Failed to create signed deploy")
}

/// Creates a CasperSnapshot for testing with the given parameters.
/// Uses an in-memory DAG representation (matching Scala's TestBlockDagRepresentation).
fn create_snapshot(
    max_block_num: i64,
    validator_id: Bytes,
) -> CasperSnapshot {
    let shard_conf = CasperShardConf {
        fault_tolerance_threshold: 0.0,
        shard_name: "test-shard".to_string(),
        parent_shard_id: "".to_string(),
        finalization_rate: 0,
        max_number_of_parents: 10,
        max_parent_depth: 0,
        synchrony_constraint_threshold: 0.0,
        height_constraint_threshold: 0,
        deploy_lifespan: DEPLOY_LIFESPAN,
        casper_version: 1,
        config_version: 1,
        bond_minimum: 0,
        bond_maximum: i64::MAX,
        epoch_length: 0,
        quarantine_length: 0,
        min_phlo_price: 0,
        enable_mergeable_channel_gc: false,
        mergeable_channels_gc_depth_buffer: 10,
        disable_late_block_filtering: false,
        disable_validator_progress_check: false,
    };

    let mut bonds_map: HashMap<ByteString, i64> = HashMap::new();
    bonds_map.insert(validator_id.clone(), 100);

    // Set maxSeqNums like Scala does: Map(validatorId -> 0)
    let max_seq_nums: DashMap<ByteString, u64> = DashMap::new();
    max_seq_nums.insert(validator_id.clone(), 0);

    let on_chain_state = OnChainCasperState {
        shard_conf,
        bonds_map,
        active_validators: vec![validator_id],
    };

    // Use in-memory DAG representation (like Scala's TestBlockDagRepresentation)
    let dag = resources::new_key_value_dag_representation();

    CasperSnapshot {
        dag,
        last_finalized_block: Bytes::new(),
        lca: Bytes::new(),
        tips: vec![],
        parents: vec![],
        justifications: DashSet::new(),
        invalid_blocks: HashMap::new(),
        deploys_in_scope: DashSet::new(),
        max_block_num,
        max_seq_nums,
        on_chain_state,
    }
}

/// Test: "remove block-expired deploys while keeping valid ones in storage"
///
/// With deployLifespan = 50 and currentBlock = 101 (maxBlockNum = 100),
/// earliestBlockNumber = 101 - 50 = 51
///
/// Expired deploy: validAfterBlockNumber = 0 (<= 51, expired)
/// Valid deploy: validAfterBlockNumber = 60 (> 51 and < 101, valid)
#[tokio::test]
async fn should_remove_block_expired_deploys_while_keeping_valid_ones() {
    crate::init_logger();

    let validator_sk = DEFAULT_VALIDATOR_SKS[0].clone();
    let validator_identity = ValidatorIdentity::new(&validator_sk);
    let validator_id: Bytes = validator_identity.public_key.bytes.clone().into();

    // Create all stores from a single InMemoryStoreManager (like Scala's kvm pattern)
    let mut kvm = InMemoryStoreManager::new();

    let deploy_storage = Arc::new(Mutex::new(
        KeyValueDeployStorage::new(&mut kvm)
            .await
            .expect("Failed to create deploy storage"),
    ));

    let block_store = KeyValueBlockStore::create_from_kvm(&mut kvm)
        .await
        .expect("Failed to create block store");

    let rspace_store = kvm.r_space_stores().await.expect("Failed to get rspace store");
    let mergeable_store = resources::mergeable_store_from_dyn(&mut kvm)
        .await
        .expect("Failed to create mergeable store");

    let (mut runtime_manager, _) = RuntimeManager::create_with_history(
        rspace_store,
        mergeable_store,
        casper::rust::genesis::genesis::Genesis::non_negative_mergeable_tag_name(),
        rholang::rust::interpreter::external_services::ExternalServices::noop(),
    );

    // Create deploys:
    // - Expired deploy: validAfterBlockNumber = 0 (<= 51, expired)
    // - Valid deploy: validAfterBlockNumber = 60 (> 51 and < 101, valid)
    let expired_deploy = create_deploy(0, None, &validator_sk);
    let valid_deploy = create_deploy(60, None, &validator_sk);

    // Add both deploys to storage
    {
        let mut ds = deploy_storage.lock().unwrap();
        ds.add(vec![expired_deploy.clone(), valid_deploy.clone()])
            .expect("Failed to add deploys");

        // Verify both deploys are in storage
        let deploys_before = ds.read_all().expect("Failed to read deploys");
        assert_eq!(deploys_before.len(), 2, "Expected 2 deploys before create");
    }

    // Create snapshot with maxBlockNum = 100
    let snapshot = create_snapshot(100, validator_id);

    // Call BlockCreator.create
    // The cleanup happens in prepareUserDeploys before block creation
    // Block creation may fail due to empty parents, but that's after cleanup
    let _ = block_creator::create(
        &snapshot,
        &validator_identity,
        None,
        deploy_storage.clone(),
        &mut runtime_manager,
        &mut block_store.clone(),
        false,
    )
    .await;

    // Verify: expired deploy removed, valid deploy kept
    {
        let ds = deploy_storage.lock().unwrap();
        let deploys_after = ds.read_all().expect("Failed to read deploys");
        assert_eq!(deploys_after.len(), 1, "Expected 1 deploy after create (expired should be removed)");

        let remaining_deploy = deploys_after.iter().next().unwrap();
        assert_eq!(
            remaining_deploy.sig, valid_deploy.sig,
            "Expected valid deploy to remain"
        );
    }
}

/// Test: "remove both block-expired and time-expired deploys while keeping valid ones"
///
/// - Block-expired deploy (validAfterBlockNumber = 0 is expired)
/// - Time-expired deploy (validAfterBlockNumber = 60 is valid, but expirationTimestamp is past)
/// - Valid deploy (validAfterBlockNumber = 60 is valid, no expiration timestamp)
#[tokio::test]
async fn should_remove_both_block_expired_and_time_expired_deploys() {
    crate::init_logger();

    let validator_sk = DEFAULT_VALIDATOR_SKS[0].clone();
    let validator_identity = ValidatorIdentity::new(&validator_sk);
    let validator_id: Bytes = validator_identity.public_key.bytes.clone().into();

    // Create all stores from a single InMemoryStoreManager (like Scala's kvm pattern)
    let mut kvm = InMemoryStoreManager::new();

    let deploy_storage = Arc::new(Mutex::new(
        KeyValueDeployStorage::new(&mut kvm)
            .await
            .expect("Failed to create deploy storage"),
    ));

    let block_store = KeyValueBlockStore::create_from_kvm(&mut kvm)
        .await
        .expect("Failed to create block store");

    let rspace_store = kvm.r_space_stores().await.expect("Failed to get rspace store");
    let mergeable_store = resources::mergeable_store_from_dyn(&mut kvm)
        .await
        .expect("Failed to create mergeable store");

    let (mut runtime_manager, _) = RuntimeManager::create_with_history(
        rspace_store,
        mergeable_store,
        casper::rust::genesis::genesis::Genesis::non_negative_mergeable_tag_name(),
        rholang::rust::interpreter::external_services::ExternalServices::noop(),
    );

    // 1 minute ago (past timestamp for time-expired deploy)
    let past_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
        - 60000;

    // Create deploys:
    // - Block-expired deploy (validAfterBlockNumber = 0 is expired)
    // - Time-expired deploy (validAfterBlockNumber = 60 is valid, but expirationTimestamp is past)
    // - Valid deploy (validAfterBlockNumber = 60 is valid, no expiration timestamp)
    let block_expired_deploy = create_deploy(0, None, &validator_sk);
    let time_expired_deploy = create_deploy(60, Some(past_timestamp), &validator_sk);
    let valid_deploy = create_deploy(60, None, &validator_sk);

    // Add all deploys to storage
    {
        let mut ds = deploy_storage.lock().unwrap();
        ds.add(vec![
            block_expired_deploy.clone(),
            time_expired_deploy.clone(),
            valid_deploy.clone(),
        ])
        .expect("Failed to add deploys");

        // Verify all deploys are in storage
        let deploys_before = ds.read_all().expect("Failed to read deploys");
        assert_eq!(deploys_before.len(), 3, "Expected 3 deploys before create");
    }

    // Create snapshot with maxBlockNum = 100
    let snapshot = create_snapshot(100, validator_id);

    // Call BlockCreator.create
    let _ = block_creator::create(
        &snapshot,
        &validator_identity,
        None,
        deploy_storage.clone(),
        &mut runtime_manager,
        &mut block_store.clone(),
        false,
    )
    .await;

    // Verify: both expired deploys removed, valid deploy kept
    {
        let ds = deploy_storage.lock().unwrap();
        let deploys_after = ds.read_all().expect("Failed to read deploys");
        assert_eq!(
            deploys_after.len(),
            1,
            "Expected 1 deploy after create (both expired should be removed)"
        );

        let remaining_deploy = deploys_after.iter().next().unwrap();
        assert_eq!(
            remaining_deploy.sig, valid_deploy.sig,
            "Expected valid deploy to remain"
        );
    }
}
