// See casper/src/test/scala/coop/rchain/casper/api/BlockQueryResponseAPITest.scala

use crate::helper::no_ops_casper_effect::NoOpsCasperEffect;
use crate::util::rholang::resources::{mk_runtime_manager_at, mk_test_rnode_store_manager};
use crate::util::test_mocks::MockKeyValueStore;
use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use block_storage::rust::test::indexed_block_dag_storage::IndexedBlockDagStorage;
use casper::rust::api::block_api::BlockAPI;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::engine::engine_with_casper::EngineWithCasper;
use casper::rust::safety_oracle::{CliqueOracleImpl, MIN_FAULT_TOLERANCE};
use casper::rust::util::construct_deploy;
use casper::rust::util::proto_util::{bond_to_bond_info, justifications_to_justification_infos};
use casper::rust::util::rholang::runtime_manager::RuntimeManager;
use models::casper::{BlockInfo, BondInfo, JustificationInfo, LightBlockInfo};
use models::rust::block_implicits::{get_random_block, get_random_block_default};
use models::rust::casper::protocol::casper_message::{
    BlockMessage, Bond, Justification, ProcessedDeploy,
};
use prost::bytes::Bytes;
use prost::Message;
use rholang::rust::interpreter::test_utils::resources::mk_temp_dir;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const TOO_SHORT_QUERY: &str = "12345";
const BAD_TEST_HASH_QUERY: &str = "1234acd";
const INVALID_HEX_QUERY: &str = "No such a hash";
const DEPLOY_COUNT: usize = 10;
const FAULT_TOLERANCE: f32 = MIN_FAULT_TOLERANCE;

struct TestContext {
    shared_kvm_data: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
    block_store: KeyValueBlockStore,
    dag_storage: IndexedBlockDagStorage,
    runtime_manager: RuntimeManager,
    _temp_dir: PathBuf,
}

impl TestContext {
    async fn new(prefix: &str) -> Self {
        let shared_kvm_data = Arc::new(Mutex::new(HashMap::<Vec<u8>, Vec<u8>>::new()));

        let block_store = KeyValueBlockStore::new(
            Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
            Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
        );

        let temp_dir = mk_temp_dir(prefix);
        let mut kvm = mk_test_rnode_store_manager(temp_dir.clone());
        let dag = BlockDagKeyValueStorage::new(&mut kvm).await.unwrap();
        let dag_storage = IndexedBlockDagStorage::new(dag);

        let runtime_manager =
            mk_runtime_manager_at(mk_test_rnode_store_manager(temp_dir.clone()), None).await;

        Self {
            shared_kvm_data,
            block_store,
            dag_storage,
            runtime_manager,
            _temp_dir: temp_dir,
        }
    }
}

fn create_sender_and_validator() -> (String, Bytes, Bond) {
    let sender_string = "3456789101112131415161718192345678910111213141516171819261718192113456789101112131415161718192345678910111213141516171819261718192";
    let sender = hex::decode(sender_string).unwrap();
    let sender = Bytes::from(sender);
    let bonds_validator = Bond {
        validator: sender.clone(),
        stake: 1,
    };
    (sender_string.to_string(), sender, bonds_validator)
}

fn create_test_blocks() -> (BlockMessage, BlockMessage, Vec<ProcessedDeploy>) {
    let genesis_block = get_random_block_default();

    let random_deploys: Vec<ProcessedDeploy> = (0..DEPLOY_COUNT)
        .map(|i| construct_deploy::basic_processed_deploy(i as i32, None).unwrap())
        .collect();

    let (_sender_string, sender, bonds_validator) = create_sender_and_validator();

    let second_block = get_random_block(
        None,
        None,
        None,
        None,
        Some(sender.clone()),
        None,
        None,
        Some(vec![genesis_block.block_hash.clone()]),
        Some(vec![Justification {
            validator: bonds_validator.validator.clone(),
            latest_block_hash: genesis_block.block_hash.clone(),
        }]),
        Some(random_deploys.clone()),
        None,
        Some(vec![bonds_validator]),
        None,
        None,
    );

    (genesis_block, second_block, random_deploys)
}

async fn effects_for_simple_casper_setup(
    block_store: KeyValueBlockStore,
    block_dag_storage: &mut IndexedBlockDagStorage,
    runtime_manager: RuntimeManager,
    genesis_block: &BlockMessage,
    second_block: &BlockMessage,
    shared_kvm_data: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
) -> (EngineCell, CliqueOracleImpl) {
    block_dag_storage
        .insert(&genesis_block, false, true)
        .unwrap();

    block_dag_storage
        .insert(&second_block, false, false)
        .unwrap();

    let casper_effect = NoOpsCasperEffect::new_with_shared_kvm(
        None,
        Arc::new(tokio::sync::Mutex::new(runtime_manager)),
        block_store.clone(),
        block_dag_storage.get_representation(),
        shared_kvm_data,
    );

    let engine = EngineWithCasper::new(Arc::new(casper_effect));

    let engine_cell = EngineCell::init();
    engine_cell.set(Arc::new(engine)).await;

    let clique_oracle_effect = CliqueOracleImpl;

    (engine_cell, clique_oracle_effect)
}

async fn empty_effects(
    block_store: KeyValueBlockStore,
    block_dag_storage: &mut IndexedBlockDagStorage,
    runtime_manager: RuntimeManager,
    genesis_block: &BlockMessage,
    _second_block: &BlockMessage,
    shared_kvm_data: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
) -> (EngineCell, CliqueOracleImpl) {
    let casper_effect = NoOpsCasperEffect::new_with_shared_kvm(
        None,
        Arc::new(tokio::sync::Mutex::new(runtime_manager)),
        block_store.clone(),
        block_dag_storage.get_representation(),
        shared_kvm_data,
    );

    block_dag_storage
        .insert(&genesis_block, false, true)
        .unwrap();

    let engine = EngineWithCasper::new(Arc::new(casper_effect));

    let engine_cell = EngineCell::init();
    engine_cell.set(Arc::new(engine)).await;

    let clique_oracle_effect = CliqueOracleImpl;

    (engine_cell, clique_oracle_effect)
}

#[tokio::test]
async fn get_block_should_return_successful_block_info_response() {
    let (genesis_block, second_block, random_deploys) = create_test_blocks();

    let mut ctx = TestContext::new("block-query-api-test-").await;

    ctx.block_store
        .put(genesis_block.block_hash.clone(), &genesis_block)
        .unwrap();
    ctx.block_store
        .put(second_block.block_hash.clone(), &second_block)
        .unwrap();

    let (engine_cell, _clique_oracle_effect) = effects_for_simple_casper_setup(
        ctx.block_store.clone(),
        &mut ctx.dag_storage,
        ctx.runtime_manager,
        &genesis_block,
        &second_block,
        ctx.shared_kvm_data.clone(),
    )
    .await;

    let hash = hex::encode(&second_block.block_hash);

    let block_query_response = BlockAPI::get_block(&engine_cell, &hash).await;

    assert!(
        block_query_response.is_ok(),
        "Expected successful response but got error: {:?}",
        block_query_response.err()
    );

    let block_info: BlockInfo = block_query_response.unwrap();

    assert_eq!(
        block_info.deploys.len(),
        random_deploys.len(),
        "Deploy count mismatch"
    );

    let b: LightBlockInfo = block_info.block_info.expect("block_info should be present");

    assert_eq!(
        b.block_hash,
        hex::encode(&second_block.block_hash),
        "Block hash mismatch"
    );

    assert_eq!(
        b.sender,
        hex::encode(&second_block.sender),
        "Sender mismatch"
    );

    let expected_block_size = Message::encode_to_vec(&second_block.to_proto()).len();
    assert_eq!(
        b.block_size,
        expected_block_size.to_string(),
        "Block size mismatch"
    );

    assert_eq!(
        b.seq_num, second_block.seq_num as i64,
        "Sequence number mismatch"
    );

    assert_eq!(b.sig, hex::encode(&second_block.sig), "Signature mismatch");

    assert_eq!(
        b.sig_algorithm, second_block.sig_algorithm,
        "Signature algorithm mismatch"
    );

    assert_eq!(b.shard_id, second_block.shard_id, "Shard ID mismatch");

    assert_eq!(
        b.extra_bytes, second_block.extra_bytes,
        "Extra bytes mismatch"
    );

    assert_eq!(b.version, second_block.header.version, "Version mismatch");

    assert_eq!(
        b.timestamp, second_block.header.timestamp,
        "Timestamp mismatch"
    );

    assert_eq!(
        b.header_extra_bytes, second_block.header.extra_bytes,
        "Header extra bytes mismatch"
    );

    let expected_parents: Vec<String> = second_block
        .header
        .parents_hash_list
        .iter()
        .map(|h| hex::encode(h))
        .collect();
    assert_eq!(
        b.parents_hash_list, expected_parents,
        "Parents hash list mismatch"
    );

    assert_eq!(
        b.block_number, second_block.body.state.block_number,
        "Block number mismatch"
    );

    assert_eq!(
        b.pre_state_hash,
        hex::encode(&second_block.body.state.pre_state_hash),
        "Pre-state hash mismatch"
    );

    assert_eq!(
        b.post_state_hash,
        hex::encode(&second_block.body.state.post_state_hash),
        "Post-state hash mismatch"
    );

    assert_eq!(
        b.body_extra_bytes, second_block.body.extra_bytes,
        "Body extra bytes mismatch"
    );

    assert_eq!(
        b.bonds.len(),
        second_block.body.state.bonds.len(),
        "Bonds count mismatch"
    );

    assert_eq!(
        b.block_size,
        expected_block_size.to_string(),
        "Block size mismatch (second check)"
    );

    assert_eq!(
        b.deploy_count,
        second_block.body.deploys.len() as i32,
        "Deploy count mismatch"
    );

    // Scala line 120: b.faultTolerance should be(faultTolerance)
    assert_eq!(
        b.fault_tolerance, FAULT_TOLERANCE,
        "Fault tolerance mismatch"
    );

    let expected_justifications: Vec<JustificationInfo> = second_block
        .justifications
        .iter()
        .map(|j| justifications_to_justification_infos(j))
        .collect();
    assert_eq!(
        b.justifications, expected_justifications,
        "Justifications mismatch"
    );
}

#[tokio::test]
async fn get_block_should_return_error_when_no_block_exists() {
    let (genesis_block, second_block, _random_deploys) = create_test_blocks();

    let mut ctx = TestContext::new("block-query-api-error-test-").await;

    let (engine_cell, _clique_oracle_effect) = empty_effects(
        ctx.block_store.clone(),
        &mut ctx.dag_storage,
        ctx.runtime_manager,
        &genesis_block,
        &second_block,
        ctx.shared_kvm_data.clone(),
    )
    .await;

    let hash = BAD_TEST_HASH_QUERY;

    let block_query_response = BlockAPI::get_block(&engine_cell, hash).await;
    assert!(
        block_query_response.is_err(),
        "Expected error response but got success"
    );

    let error_msg = block_query_response.unwrap_err().to_string();
    let expected_msg = format!(
        "Error: Failure to find block with hash: {}",
        BAD_TEST_HASH_QUERY
    );
    assert_eq!(error_msg, expected_msg, "Error message mismatch");
}

#[tokio::test]
async fn get_block_should_return_error_when_hash_is_invalid_hex_string() {
    let (genesis_block, second_block, _random_deploys) = create_test_blocks();

    let mut ctx = TestContext::new("block-query-api-invalid-hex-test-").await;

    let (engine_cell, _clique_oracle_effect) = empty_effects(
        ctx.block_store.clone(),
        &mut ctx.dag_storage,
        ctx.runtime_manager,
        &genesis_block,
        &second_block,
        ctx.shared_kvm_data.clone(),
    )
    .await;

    let hash = INVALID_HEX_QUERY;

    let block_query_response = BlockAPI::get_block(&engine_cell, hash).await;

    assert!(
        block_query_response.is_err(),
        "Expected error response but got success"
    );

    let error_msg = block_query_response.unwrap_err().to_string();
    let expected_msg = format!(
        "Input hash value is not valid hex string: {}",
        INVALID_HEX_QUERY
    );
    assert_eq!(error_msg, expected_msg, "Error message mismatch");
}

#[tokio::test]
async fn get_block_should_return_error_when_hash_is_too_short() {
    let (genesis_block, second_block, _random_deploys) = create_test_blocks();

    let mut ctx = TestContext::new("block-query-api-short-hash-test-").await;

    let (engine_cell, _clique_oracle_effect) = empty_effects(
        ctx.block_store.clone(),
        &mut ctx.dag_storage,
        ctx.runtime_manager,
        &genesis_block,
        &second_block,
        ctx.shared_kvm_data.clone(),
    )
    .await;

    let hash = TOO_SHORT_QUERY;

    let block_query_response = BlockAPI::get_block(&engine_cell, hash).await;

    assert!(
        block_query_response.is_err(),
        "Expected error response but got success"
    );

    let error_msg = block_query_response.unwrap_err().to_string();
    let expected_msg = format!(
        "Input hash value must be at least 6 characters: {}",
        TOO_SHORT_QUERY
    );
    assert_eq!(error_msg, expected_msg, "Error message mismatch");
}

#[tokio::test]
async fn find_deploy_should_return_successful_block_info_response_when_block_contains_deploy_with_given_signature(
) {
    let (genesis_block, second_block, random_deploys) = create_test_blocks();

    let mut ctx = TestContext::new("block-query-api-find-deploy-test-").await;

    ctx.block_store
        .put(genesis_block.block_hash.clone(), &genesis_block)
        .unwrap();
    ctx.block_store
        .put(second_block.block_hash.clone(), &second_block)
        .unwrap();

    let (engine_cell, _clique_oracle_effect) = effects_for_simple_casper_setup(
        ctx.block_store.clone(),
        &mut ctx.dag_storage,
        ctx.runtime_manager,
        &genesis_block,
        &second_block,
        ctx.shared_kvm_data.clone(),
    )
    .await;

    let deploy_id = random_deploys[0].deploy.sig.to_vec();

    let block_query_response = BlockAPI::find_deploy(&engine_cell, &deploy_id).await;

    assert!(
        block_query_response.is_ok(),
        "Expected successful response but got error: {:?}",
        block_query_response.err()
    );

    let block_info: LightBlockInfo = block_query_response.unwrap();

    assert_eq!(
        block_info.block_hash,
        hex::encode(&second_block.block_hash),
        "Block hash mismatch"
    );

    assert_eq!(
        block_info.sender,
        hex::encode(&second_block.sender),
        "Sender mismatch"
    );

    let expected_block_size = Message::encode_to_vec(&second_block.to_proto()).len();
    assert_eq!(
        block_info.block_size,
        expected_block_size.to_string(),
        "Block size mismatch"
    );

    assert_eq!(
        block_info.seq_num, second_block.seq_num as i64,
        "Sequence number mismatch"
    );

    assert_eq!(
        block_info.sig,
        hex::encode(&second_block.sig),
        "Signature mismatch"
    );

    assert_eq!(
        block_info.sig_algorithm, second_block.sig_algorithm,
        "Signature algorithm mismatch"
    );

    assert_eq!(
        block_info.shard_id, second_block.shard_id,
        "Shard ID mismatch"
    );

    assert_eq!(
        block_info.extra_bytes, second_block.extra_bytes,
        "Extra bytes mismatch"
    );

    assert_eq!(
        block_info.version, second_block.header.version,
        "Version mismatch"
    );

    assert_eq!(
        block_info.timestamp, second_block.header.timestamp,
        "Timestamp mismatch"
    );

    assert_eq!(
        block_info.header_extra_bytes, second_block.header.extra_bytes,
        "Header extra bytes mismatch"
    );

    let expected_parents: Vec<String> = second_block
        .header
        .parents_hash_list
        .iter()
        .map(|h| hex::encode(h))
        .collect();
    assert_eq!(
        block_info.parents_hash_list, expected_parents,
        "Parents hash list mismatch"
    );

    assert_eq!(
        block_info.block_number, second_block.body.state.block_number,
        "Block number mismatch"
    );

    assert_eq!(
        block_info.pre_state_hash,
        hex::encode(&second_block.body.state.pre_state_hash),
        "Pre-state hash mismatch"
    );

    assert_eq!(
        block_info.post_state_hash,
        hex::encode(&second_block.body.state.post_state_hash),
        "Post-state hash mismatch"
    );

    assert_eq!(
        block_info.body_extra_bytes, second_block.body.extra_bytes,
        "Body extra bytes mismatch"
    );

    let expected_bonds: Vec<BondInfo> = second_block
        .body
        .state
        .bonds
        .iter()
        .map(|b| bond_to_bond_info(b))
        .collect();
    assert_eq!(
        block_info.bonds.len(),
        expected_bonds.len(),
        "Bonds count mismatch"
    );

    assert_eq!(
        block_info.block_size,
        expected_block_size.to_string(),
        "Block size mismatch (second check)"
    );

    assert_eq!(
        block_info.deploy_count,
        second_block.body.deploys.len() as i32,
        "Deploy count mismatch"
    );

    let fault_tolerance = MIN_FAULT_TOLERANCE;
    assert_eq!(
        block_info.fault_tolerance, fault_tolerance,
        "Fault tolerance mismatch"
    );

    let expected_justifications: Vec<JustificationInfo> = second_block
        .justifications
        .iter()
        .map(|j| justifications_to_justification_infos(j))
        .collect();
    assert_eq!(
        block_info.justifications, expected_justifications,
        "Justifications mismatch"
    );
}

#[tokio::test]
async fn find_deploy_should_return_error_when_no_block_contains_deploy_with_given_signature() {
    let (genesis_block, second_block, _random_deploys) = create_test_blocks();

    let mut ctx = TestContext::new("block-query-api-find-deploy-error-test-").await;

    let (engine_cell, _clique_oracle_effect) = empty_effects(
        ctx.block_store.clone(),
        &mut ctx.dag_storage,
        ctx.runtime_manager,
        &genesis_block,
        &second_block,
        ctx.shared_kvm_data.clone(),
    )
    .await;

    let deploy_id = b"asdfQwertyUiopxyzcbv".to_vec();

    let block_query_response = BlockAPI::find_deploy(&engine_cell, &deploy_id).await;

    assert!(
        block_query_response.is_err(),
        "Expected error response but got success"
    );

    let error_msg = block_query_response.unwrap_err().to_string();
    let expected_msg = format!(
        "Couldn't find block containing deploy with id: {}",
        hex::encode(&deploy_id)
    );
    assert_eq!(error_msg, expected_msg, "Error message mismatch");
}
