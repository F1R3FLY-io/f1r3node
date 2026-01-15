// See casper/src/test/scala/coop/rchain/casper/api/BlocksResponseAPITest.scala
// See [[/docs/casper/images/no_finalizable_block_mistake_with_no_disagreement_check.png]]

use casper::rust::test_utils::helper::block_generator;
use casper::rust::test_utils::helper::block_util;
use casper::rust::test_utils::helper::no_ops_casper_effect::NoOpsCasperEffect;
use casper::rust::test_utils::helper::unlimited_parents_estimator_fixture::UnlimitedParentsEstimatorFixture;
use casper::rust::test_utils::util::rholang::resources::{mk_runtime_manager_at, mk_test_rnode_store_manager_shared, generate_scope_id};
use casper::rust::test_utils::util::test_mocks::MockKeyValueStore;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use block_storage::rust::test::indexed_block_dag_storage::IndexedBlockDagStorage;
use casper::rust::api::block_api::BlockAPI;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::engine::engine_with_casper::EngineWithCasper;
use models::rust::block_hash::BlockHash;
use models::rust::casper::protocol::casper_message::{BlockMessage, Bond};
use models::rust::validator::Validator;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

fn create_validators_and_bonds() -> (Validator, Validator, Validator, Bond, Bond, Bond, Vec<Bond>) {
    let v1 = block_util::generate_validator(Some("Validator One"));
    let v2 = block_util::generate_validator(Some("Validator Two"));
    let v3 = block_util::generate_validator(Some("Validator Three"));

    let v1_bond = Bond {
        validator: v1.clone(),
        stake: 25,
    };
    let v2_bond = Bond {
        validator: v2.clone(),
        stake: 20,
    };
    let v3_bond = Bond {
        validator: v3.clone(),
        stake: 15,
    };

    let bonds = vec![v1_bond.clone(), v2_bond.clone(), v3_bond.clone()];

    (v1, v2, v3, v1_bond, v2_bond, v3_bond, bonds)
}

// Helper function to create storage components (similar to Scala's BlockDagStorageFixture)
async fn create_storage(
    _prefix: &str,
) -> IndexedBlockDagStorage {
    let scope_id = generate_scope_id();
    let mut kvm = mk_test_rnode_store_manager_shared(scope_id);
    let dag = casper::rust::test_utils::util::rholang::resources::block_dag_storage_from_dyn(&mut *kvm).await.unwrap();
    IndexedBlockDagStorage::new(dag)
}

const MAX_BLOCK_LIMIT: i32 = 50;

fn create_dag_with_8_blocks(
    block_store: &mut KeyValueBlockStore,
    dag_storage: &mut IndexedBlockDagStorage,
) -> BlockMessage {
    let (v1, v2, v3, _v1_bond, _v2_bond, _v3_bond, bonds) = create_validators_and_bonds();

    let genesis = block_generator::create_genesis_block(
        block_store,
        dag_storage,
        None,
        Some(bonds.clone()),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let genesis_hash = genesis.block_hash.clone();

    let justifications_b2: HashMap<Validator, BlockHash> = [
        (v1.clone(), genesis_hash.clone()),
        (v2.clone(), genesis_hash.clone()),
        (v3.clone(), genesis_hash.clone()),
    ]
    .into_iter()
    .collect();

    let b2 = block_generator::create_block(
        block_store,
        dag_storage,
        vec![genesis_hash.clone()],
        &genesis,
        Some(v2.clone()),
        Some(bonds.clone()),
        Some(justifications_b2),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let justifications_b3: HashMap<Validator, BlockHash> = [
        (v1.clone(), genesis_hash.clone()),
        (v2.clone(), genesis_hash.clone()),
        (v3.clone(), genesis_hash.clone()),
    ]
    .into_iter()
    .collect();

    let b3 = block_generator::create_block(
        block_store,
        dag_storage,
        vec![genesis_hash.clone()],
        &genesis,
        Some(v1.clone()),
        Some(bonds.clone()),
        Some(justifications_b3),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let b2_hash = b2.block_hash.clone();
    let b3_hash = b3.block_hash.clone();

    let justifications_b4: HashMap<Validator, BlockHash> = [
        (v1.clone(), genesis_hash.clone()),
        (v2.clone(), b2_hash.clone()),
        (v3.clone(), b2_hash.clone()),
    ]
    .into_iter()
    .collect();

    let b4 = block_generator::create_block(
        block_store,
        dag_storage,
        vec![b2_hash.clone()],
        &genesis,
        Some(v3.clone()),
        Some(bonds.clone()),
        Some(justifications_b4),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let justifications_b5: HashMap<Validator, BlockHash> = [
        (v1.clone(), b3_hash.clone()),
        (v2.clone(), b2_hash.clone()),
        (v3.clone(), genesis_hash.clone()),
    ]
    .into_iter()
    .collect();

    let b5 = block_generator::create_block(
        block_store,
        dag_storage,
        vec![b3_hash.clone()],
        &genesis,
        Some(v2.clone()),
        Some(bonds.clone()),
        Some(justifications_b5),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let b4_hash = b4.block_hash.clone();
    let b5_hash = b5.block_hash.clone();

    let justifications_b6: HashMap<Validator, BlockHash> = [
        (v1.clone(), b3_hash.clone()),
        (v2.clone(), b2_hash.clone()),
        (v3.clone(), b4_hash.clone()),
    ]
    .into_iter()
    .collect();

    let b6 = block_generator::create_block(
        block_store,
        dag_storage,
        vec![b4_hash.clone()],
        &genesis,
        Some(v1.clone()),
        Some(bonds.clone()),
        Some(justifications_b6),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let justifications_b7: HashMap<Validator, BlockHash> = [
        (v1.clone(), b3_hash.clone()),
        (v2.clone(), b5_hash.clone()),
        (v3.clone(), b4_hash.clone()),
    ]
    .into_iter()
    .collect();

    let _b7 = block_generator::create_block(
        block_store,
        dag_storage,
        vec![b5_hash.clone()],
        &genesis,
        Some(v3.clone()),
        Some(bonds.clone()),
        Some(justifications_b7),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let b6_hash = b6.block_hash.clone();

    let justifications_b8: HashMap<Validator, BlockHash> = [
        (v1.clone(), b6_hash.clone()),
        (v2.clone(), b5_hash.clone()),
        (v3.clone(), b4_hash.clone()),
    ]
    .into_iter()
    .collect();

    let _b8 = block_generator::create_block(
        block_store,
        dag_storage,
        vec![b6_hash],
        &genesis,
        Some(v2.clone()),
        Some(bonds.clone()),
        Some(justifications_b8),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    genesis
}

#[tokio::test]
async fn show_main_chain_should_return_only_blocks_in_the_main_chain() {
    let shared_kvm_data = Arc::new(Mutex::new(HashMap::<Vec<u8>, Vec<u8>>::new()));

    // Create block_store with shared kvm
    let mut block_store = KeyValueBlockStore::new(
        Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
        Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
    );

    let mut block_dag_storage = create_storage("show-main-chain-test-").await;

    let genesis = create_dag_with_8_blocks(&mut block_store, &mut block_dag_storage);

    let mut dag = block_dag_storage.get_representation();

    let tips = UnlimitedParentsEstimatorFixture::create_estimator()
        .tips(&mut dag, &genesis)
        .await
        .unwrap();

    let scope_id = generate_scope_id();
    let mut kvm = mk_test_rnode_store_manager_shared(scope_id);
    let runtime_manager = mk_runtime_manager_at(&mut *kvm, None).await;

    let casper_effect = NoOpsCasperEffect::new_with_shared_kvm(
        Some(tips.tips),
        Arc::new(tokio::sync::Mutex::new(runtime_manager)),
        block_store.clone(),
        dag.clone(),
        shared_kvm_data.clone(), // shared kvm!
    );

    let engine = EngineWithCasper::new(Arc::new(casper_effect));

    let engine_cell = EngineCell::init();
    engine_cell.set(Arc::new(engine)).await;

    let blocks_response = BlockAPI::show_main_chain(&engine_cell, 10, MAX_BLOCK_LIMIT).await;

    assert_eq!(
        blocks_response.len(),
        5,
        "Main chain should contain 5 blocks"
    );
}

#[tokio::test]
async fn get_blocks_should_return_all_blocks() {
    let shared_kvm_data = Arc::new(Mutex::new(HashMap::<Vec<u8>, Vec<u8>>::new()));

    // Create block_store with shared kvm
    let mut block_store = KeyValueBlockStore::new(
        Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
        Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
    );

    let mut dag_storage = create_storage("get-blocks-test-").await;

    let genesis = create_dag_with_8_blocks(&mut block_store, &mut dag_storage);

    let mut dag = dag_storage.get_representation();

    let tips = UnlimitedParentsEstimatorFixture::create_estimator()
        .tips(&mut dag, &genesis)
        .await
        .unwrap();

    let scope_id = generate_scope_id();
    let mut kvm = mk_test_rnode_store_manager_shared(scope_id);
    let runtime_manager = mk_runtime_manager_at(&mut *kvm, None).await;

    let casper_effect = NoOpsCasperEffect::new_with_shared_kvm(
        Some(tips.tips),
        Arc::new(tokio::sync::Mutex::new(runtime_manager)),
        block_store.clone(),
        dag.clone(),
        shared_kvm_data.clone(),
    );

    let engine = EngineWithCasper::new(Arc::new(casper_effect));

    let engine_cell = EngineCell::init();
    engine_cell.set(Arc::new(engine)).await;

    let blocks_response = BlockAPI::get_blocks(&engine_cell, 10, MAX_BLOCK_LIMIT)
        .await
        .unwrap();
    // TODO: from Scala -> Switch to 4 when we implement block height correctly
    assert_eq!(blocks_response.len(), 8, "Should return all 8 blocks");
}

#[tokio::test]
async fn get_blocks_should_return_until_depth() {
    let shared_kvm_data = Arc::new(Mutex::new(HashMap::<Vec<u8>, Vec<u8>>::new()));

    // Create block_store with shared kvm
    let mut block_store = KeyValueBlockStore::new(
        Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
        Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
    );

    let mut dag_storage = create_storage("get-blocks-depth-test-").await;

    let genesis = create_dag_with_8_blocks(&mut block_store, &mut dag_storage);

    let mut dag = dag_storage.get_representation();

    let tips = UnlimitedParentsEstimatorFixture::create_estimator()
        .tips(&mut dag, &genesis)
        .await
        .unwrap();

    let scope_id = generate_scope_id();
    let mut kvm = mk_test_rnode_store_manager_shared(scope_id);
    let runtime_manager = mk_runtime_manager_at(&mut *kvm, None).await;

    let casper_effect = NoOpsCasperEffect::new_with_shared_kvm(
        Some(tips.tips),
        Arc::new(tokio::sync::Mutex::new(runtime_manager)),
        block_store.clone(),
        dag.clone(),
        shared_kvm_data.clone(),
    );

    let engine = EngineWithCasper::new(Arc::new(casper_effect));

    let engine_cell = EngineCell::init();
    engine_cell.set(Arc::new(engine)).await;

    let blocks_response = BlockAPI::get_blocks(&engine_cell, 2, MAX_BLOCK_LIMIT)
        .await
        .unwrap();

    // TODO: from Scala -> Switch to 3 when we implement block height correctly
    assert_eq!(
        blocks_response.len(),
        2,
        "Should return 2 blocks until depth"
    );
}

#[tokio::test]
async fn get_blocks_by_heights_should_return_blocks_between_start_and_end() {
    let shared_kvm_data = Arc::new(Mutex::new(HashMap::<Vec<u8>, Vec<u8>>::new()));

    // Create block_store with shared kvm
    let mut block_store = KeyValueBlockStore::new(
        Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
        Arc::new(MockKeyValueStore::with_shared_data(shared_kvm_data.clone())),
    );

    let mut dag_storage = create_storage("get-blocks-by-heights-test-").await;

    let genesis = create_dag_with_8_blocks(&mut block_store, &mut dag_storage);

    let mut dag = dag_storage.get_representation();

    let tips = UnlimitedParentsEstimatorFixture::create_estimator()
        .tips(&mut dag, &genesis)
        .await
        .unwrap();

    let scope_id = generate_scope_id();
    let mut kvm = mk_test_rnode_store_manager_shared(scope_id);
    let runtime_manager = mk_runtime_manager_at(&mut *kvm, None).await;

    let casper_effect = NoOpsCasperEffect::new_with_shared_kvm(
        Some(tips.tips),
        Arc::new(tokio::sync::Mutex::new(runtime_manager)),
        block_store.clone(),
        dag.clone(),
        shared_kvm_data.clone(),
    );

    let engine = EngineWithCasper::new(Arc::new(casper_effect));

    let engine_cell = EngineCell::init();
    engine_cell.set(Arc::new(engine)).await;

    let blocks_response = BlockAPI::get_blocks_by_heights(&engine_cell, 2, 5, MAX_BLOCK_LIMIT)
        .await
        .unwrap();

    assert_eq!(blocks_response.len(), 4, "Should return 4 blocks");
    assert_eq!(
        blocks_response.first().unwrap().block_number,
        2,
        "First block should be at height 2"
    );
    assert_eq!(
        blocks_response.last().unwrap().block_number,
        5,
        "Last block should be at height 5"
    );
}
