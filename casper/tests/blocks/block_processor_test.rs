use crate::engine::setup;
use crate::helper::{
    block_dag_storage_fixture::with_storage, block_generator::create_genesis_block,
    block_util::generate_validator,
};
use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
};
use casper::rust::{
    blocks::block_processor::BlockProcessorDependencies, engine::block_retriever::BlockRetriever,
};
use comm::rust::{
    peer_node::PeerNode,
    rp::connect::{Connections, ConnectionsCell},
    test_instances::{create_rp_conf_ask, TransportLayerStub},
};
use models::rust::{
    block_hash::BlockHash,
    casper::protocol::casper_message::{BlockMessage, Bond, Header},
};
use rspace_plus_plus::rspace::shared::{
    in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
};
use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

struct TestFixture {
    local_peer: PeerNode,
    dependencies: BlockProcessorDependencies<TransportLayerStub>,
    genesis: BlockMessage,
    test_block: BlockMessage,
}

impl TestFixture {
    async fn new() -> Self {
        let local_peer = setup::peer_node("test-peer", 40400);
        let connections_cell = ConnectionsCell {
            peers: Arc::new(Mutex::new(Connections::from_vec(vec![local_peer.clone()]))),
        };
        let rp_conf = create_rp_conf_ask(local_peer.clone(), None, None);
        let transport_layer = Arc::new(TransportLayerStub::new());

        // Create a new ConnectionsCell for BlockRetriever instead of cloning
        let connections_cell_for_retriever = ConnectionsCell {
            peers: Arc::new(Mutex::new(Connections::from_vec(vec![local_peer.clone()]))),
        };

        let requested_blocks = Arc::new(Mutex::new(HashMap::new()));
        let block_retriever = BlockRetriever::new(
            requested_blocks,
            transport_layer.clone(),
            connections_cell_for_retriever,
            rp_conf.clone(),
        );

        let (mut block_store, mut indexed_dag_storage, casper_buffer) =
            with_storage(|bs, ids| async move {
                // Create CasperBuffer from in-memory store
                let mut kvm = InMemoryStoreManager::new();
                let store = kvm.store("parents-map".to_string()).await.unwrap();
                let typed_store = KeyValueTypedStoreImpl::new(store);
                let cb = CasperBufferKeyValueStorage::new_from_kv_store(typed_store)
                    .await
                    .unwrap();

                (bs, ids, cb)
            })
            .await;

        // Get underlying BlockDagKeyValueStorage for CasperDependencyAnalyzer
        let block_dag_storage = {
            // We need to extract underlying storage for the dependency analyzer
            // For now, create a separate one since IndexedBlockDagStorage doesn't expose underlying
            let mut dag_kvm = InMemoryStoreManager::new();
            BlockDagKeyValueStorage::new(&mut dag_kvm).await.unwrap()
        };

        let v1 = generate_validator(Some("Test Validator"));
        let bonds = vec![Bond {
            validator: v1.clone(),
            stake: 100,
        }];

        let genesis = create_genesis_block(
            &mut block_store,
            &mut indexed_dag_storage,
            None,
            Some(bonds.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        // Create test block
        let test_block = crate::helper::block_generator::create_block(
            &mut block_store,
            &mut indexed_dag_storage,
            vec![genesis.block_hash.clone()],
            &genesis,
            Some(v1),
            Some(bonds),
            Some(HashMap::new()),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        // Create unified dependencies
        let dependencies = BlockProcessorDependencies::new(
            block_store,
            casper_buffer,
            block_dag_storage,
            block_retriever,
            transport_layer,
            connections_cell,
            rp_conf,
        );

        Self {
            local_peer,
            dependencies,
            genesis,
            test_block,
        }
    }

    fn reset_transport(&self) {
        self.dependencies.transport().reset();
    }
}

#[tokio::test]
async fn request_missing_dependencies_should_call_admit_hash_for_each_dependency() {
    let fixture = TestFixture::new().await;
    fixture.reset_transport();

    // Create test dependencies
    let dep1 = BlockHash::from(b"dependency1".to_vec());
    let dep2 = BlockHash::from(b"dependency2".to_vec());
    let deps = HashSet::from([dep1.clone(), dep2.clone()]);

    // Call request_missing_dependencies using new architecture
    let result = fixture
        .dependencies
        .request_missing_dependencies(&deps)
        .await;
    assert!(result.is_ok());

    // Verify that both dependencies were requested
    let request_count = fixture.dependencies.transport().request_count();
    assert_eq!(
        request_count, 2,
        "Should have made 2 requests for 2 dependencies"
    );
}

#[tokio::test]
async fn request_missing_dependencies_should_handle_empty_set() {
    let fixture = TestFixture::new().await;
    fixture.reset_transport();

    let empty_deps = HashSet::new();

    let result = fixture
        .dependencies
        .request_missing_dependencies(&empty_deps)
        .await;
    assert!(result.is_ok());

    // No requests should be made for empty dependency set
    let request_count = fixture.dependencies.transport().request_count();
    assert_eq!(
        request_count, 0,
        "Should not make any requests for empty dependency set"
    );
}

#[tokio::test]
async fn commit_to_buffer_should_add_pendant_when_no_dependencies() {
    let fixture = TestFixture::new().await;

    // Commit block without dependencies (should become pendant)
    let result = fixture
        .dependencies
        .commit_to_buffer(&fixture.test_block, None)
        .await;
    assert!(result.is_ok());

    // Verify block was added as pendant
    let buffer = fixture.dependencies.casper_buffer();
    let pendants = buffer.get_pendants();
    assert!(
        pendants
            .iter()
            .any(|p| p.0 == fixture.test_block.block_hash),
        "Block should be added as pendant when no dependencies provided"
    );
}

#[tokio::test]
async fn commit_to_buffer_should_add_relations_when_dependencies_provided() {
    let fixture = TestFixture::new().await;

    // Create dependency set
    let deps = HashSet::from([fixture.genesis.block_hash.clone()]);

    // Commit block with dependencies
    let result = fixture
        .dependencies
        .commit_to_buffer(&fixture.test_block, Some(deps))
        .await;
    assert!(result.is_ok());

    // Verify block was added with relations
    let buffer = fixture.dependencies.casper_buffer();
    let block_hash_serde =
        models::rust::block_hash::BlockHashSerde(fixture.test_block.block_hash.clone());
    assert!(
        buffer.contains(&block_hash_serde),
        "Block should be added with relations when dependencies provided"
    );
}

#[tokio::test]
async fn remove_from_buffer_should_remove_block() {
    let fixture = TestFixture::new().await;

    // First add block to buffer
    let result = fixture
        .dependencies
        .commit_to_buffer(&fixture.test_block, None)
        .await;
    assert!(result.is_ok());

    // Verify block is in buffer
    let buffer = fixture.dependencies.casper_buffer();
    let pendants = buffer.get_pendants();
    assert!(
        pendants
            .iter()
            .any(|p| p.0 == fixture.test_block.block_hash),
        "Block should be in buffer before removal"
    );

    // Remove block from buffer
    let result = fixture
        .dependencies
        .remove_from_buffer(&fixture.test_block)
        .await;
    assert!(result.is_ok());

    // Verify block was removed
    let buffer = fixture.dependencies.casper_buffer();
    let pendants = buffer.get_pendants();
    assert!(
        !pendants
            .iter()
            .any(|p| p.0 == fixture.test_block.block_hash),
        "Block should be removed from buffer"
    );
}

#[tokio::test]
async fn buffer_manager_should_handle_concurrent_operations() {
    use tokio::task;

    let fixture = TestFixture::new().await;

    // Create multiple tasks that operate on the buffer concurrently
    let mut tasks = Vec::new();
    for _i in 0..10 {
        let casper_buffer = fixture.dependencies.casper_buffer().clone();

        let task = task::spawn(async move {
            let buffer = casper_buffer;
            // Simulate concurrent buffer operations
            let pendants = buffer.get_pendants();
            pendants.len() // Return some value to verify task completed
        });

        tasks.push(task);
    }

    // Wait for all tasks to complete
    let results = futures::future::join_all(tasks).await;

    // Verify all tasks completed successfully
    assert_eq!(results.len(), 10);
    for result in results {
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn block_processor_components_should_work_together() {
    let fixture = TestFixture::new().await;

    // Test CasperBuffer logic correctly:
    // 1. Add block as pendant (no dependencies)
    // 2. Add another block that depends on the first one
    // 3. Remove the first block (which is now a parent)

    // 1. Add test_block as pendant (no dependencies)
    let result = fixture
        .dependencies
        .commit_to_buffer(&fixture.test_block, None)
        .await;
    assert!(result.is_ok());

    // Verify test_block is pendant
    let buffer = fixture.dependencies.casper_buffer();
    let block_hash_serde =
        models::rust::block_hash::BlockHashSerde(fixture.test_block.block_hash.clone());
    assert!(buffer.is_pendant(&block_hash_serde));

    // 2. Create another block that depends on test_block
    let dependent_block = BlockMessage {
        block_hash: prost::bytes::Bytes::from(b"dependent_block".to_vec()),
        header: Header {
            parents_hash_list: vec![fixture.test_block.block_hash.clone()],
            timestamp: 0,
            version: 1,
            extra_bytes: prost::bytes::Bytes::new(),
        },
        body: fixture.test_block.body.clone(), // Use same body as test block
        justifications: vec![],
        sender: prost::bytes::Bytes::new(),
        seq_num: 0,
        sig: prost::bytes::Bytes::new(),
        sig_algorithm: String::new(),
        shard_id: String::new(),
        extra_bytes: prost::bytes::Bytes::new(),
    };

    let deps = HashSet::from([fixture.test_block.block_hash.clone()]);
    let result = fixture
        .dependencies
        .commit_to_buffer(&dependent_block, Some(deps))
        .await;
    assert!(result.is_ok());

    // 3. Now test_block is a parent, so we can remove it
    let result = fixture
        .dependencies
        .remove_from_buffer(&fixture.test_block)
        .await;
    assert!(result.is_ok());

    // 4. Test other operations
    let deps = HashSet::from([fixture.genesis.block_hash.clone()]);
    let result = fixture
        .dependencies
        .request_missing_dependencies(&deps)
        .await;
    assert!(result.is_ok());

    // 5. Acknowledge processing
    let result = fixture
        .dependencies
        .ack_processed(&fixture.test_block)
        .await;
    assert!(result.is_ok());

    // All operations should complete successfully
    assert!(true, "All components should work together");
}
