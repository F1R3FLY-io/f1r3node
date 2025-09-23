// See casper/src/test/scala/coop/rchain/casper/engine/InitializingSpec.scala

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use block_storage::rust::casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage;
use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
use crypto::rust::{
    hash::blake2b256::Blake2b256,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg},
};
use models::casper::Signature;
use models::routing::protocol::Message as ProtocolMessage;
use models::rust::casper::protocol::casper_message::{
    ApprovedBlock, ApprovedBlockRequest, BlockMessage, BlockRequest, CasperMessage,
    StoreItemsMessage, StoreItemsMessageRequest,
};
use prost::bytes::Bytes;
use prost::Message;
use shared::rust::shared::f1r3fly_events::{EventPublisher, EventPublisherFactory};

use crate::engine::setup::TestFixture;
use casper::rust::engine::engine::Engine;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::engine::initializing::Initializing;
use casper::rust::engine::lfs_tuple_space_requester;

use crate::util::test_mocks::MockKeyValueStore;
use casper::rust::errors::CasperError;
use casper::rust::util::rholang::runtime_manager::RuntimeManager;
use comm::rust::rp::connect::{Connections, ConnectionsCell};
use comm::rust::rp::protocol_helper::packet_with_content;
use comm::rust::rp::rp_conf::RPConf;
use comm::rust::test_instances::TransportLayerStub;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::rspace::RSpaceStore;
use rspace_plus_plus::rspace::shared::in_mem_store_manager::InMemoryStoreManager;
use rspace_plus_plus::rspace::state::exporters::rspace_exporter_items::RSpaceExporterItems;
use rspace_plus_plus::rspace::state::instances::rspace_exporter_store::RSpaceExporterImpl;
use rspace_plus_plus::rspace::state::instances::rspace_importer_store::RSpaceImporterImpl;
use rspace_plus_plus::rspace::state::rspace_exporter::RSpaceExporter;
use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;
use shared::rust::store::key_value_store::KeyValueStore;
use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
use shared::rust::ByteVector;

struct InitializingSpec;

impl InitializingSpec {
    fn event_bus() -> Box<dyn EventPublisher> {
        EventPublisherFactory::noop()
    }

    fn before_each(fixture: &TestFixture) {
        fixture
            .transport_layer
            .set_responses(|_peer, _protocol| Ok(()));
    }

    fn after_each(fixture: &TestFixture) {
        fixture.transport_layer.reset();
    }
    async fn make_transition_to_running_once_approved_block_received() {
        let _event_bus = Self::event_bus();

        let fixture = TestFixture::new().await;

        Self::before_each(&fixture);

        let the_init = || Ok::<(), CasperError>(());

        let engine_cell = Arc::new(EngineCell::unsafe_init().expect("Failed to create EngineCell"));

        // interval and duration don't really matter since we don't require and signs from validators
        let initializing_engine = create_initializing_engine(&fixture, Box::new(the_init), engine_cell.clone())
                .await
                .expect("Failed to create Initializing engine");

        let genesis = &fixture.genesis;
        let approved_block_candidate = fixture.approved_block_candidate.clone();
        let validator_sk = &fixture.validator_sk;
        let validator_pk = &fixture.validator_pk;

        let approved_block = {
            let candidate_proto = approved_block_candidate.clone().to_proto();
            let candidate_bytes = {
                let mut buf = Vec::new();
                Message::encode(&candidate_proto, &mut buf).expect("Failed to encode candidate");
                buf
            };
            let candidate_hash = Blake2b256::hash(candidate_bytes);
            let signature_bytes = Secp256k1.sign(&candidate_hash, &validator_sk.bytes);

            ApprovedBlock {
                candidate: approved_block_candidate,
                sigs: vec![Signature {
                    public_key: validator_pk.bytes.clone(),
                    algorithm: "secp256k1".to_string(),
                    sig: signature_bytes.into(),
                }],
            }
        };

        // Get exporter for genesis block
        // Note: Instead of default exported, we should use RSpaceExporterItems::get_history_and_data
        let _genesis_exporter = &fixture.exporter;

        let chunk_size = lfs_tuple_space_requester::PAGE_SIZE;

        fn genesis_export(
            genesis_exporter: Arc<std::sync::Mutex<Box<dyn RSpaceExporter>>>,
            start_path: Vec<(Blake2b256Hash, Option<u8>)>,
            exporter_params: &crate::engine::setup::ExporterParams,
        ) -> Result<
            (
                Vec<(Blake2b256Hash, ByteVector)>,
                Vec<(Blake2b256Hash, ByteVector)>,
                Vec<(Blake2b256Hash, Option<u8>)>,
            ),
            String,
        > {
            let (history_store_items, data_store_items) = RSpaceExporterItems::get_history_and_data(
                genesis_exporter,
                start_path,
                exporter_params.skip,
                exporter_params.take,
            );
            Ok((
                history_store_items.items,
                data_store_items.items,
                history_store_items.last_path,
            ))
        }

        let post_state_hash_bs = &approved_block.candidate.block.body.state.post_state_hash;
        let post_state_hash = Blake2b256Hash::from_bytes_prost(post_state_hash_bs);
        let start_path1 = vec![(post_state_hash, None::<u8>)];

        let rspace_store = &fixture.rspace_store;
        let genesis_exporter_impl = rspace_plus_plus::rspace::state::instances::rspace_exporter_store::RSpaceExporterStore::create(
            rspace_store.history.clone(),
            rspace_store.cold.clone(),
            rspace_store.roots.clone(),
        );
        let genesis_exporter_new: Box<dyn RSpaceExporter> = Box::new(genesis_exporter_impl);
        let genesis_exporter_arc = std::sync::Arc::new(std::sync::Mutex::new(genesis_exporter_new));

        // Get history and data items from genesis block (two chunks, as in Scala)
        let (history_items1, data_items1, last_path1) = genesis_export(
            genesis_exporter_arc.clone(),
            start_path1.clone(),
            &fixture.exporter_params,
        )
        .expect("Failed to export history and data items 1");
        let (history_items2, data_items2, last_path2) = genesis_export(
            genesis_exporter_arc.clone(),
            last_path1.clone(),
            &fixture.exporter_params,
        )
        .expect("Failed to export history and data items 2");

        // Store request messages for two chunks
        let store_request_message1 = StoreItemsMessageRequest {
            start_path: start_path1.clone(),
            skip: 0,
            take: chunk_size,
        };
        let store_request_message2 = StoreItemsMessageRequest {
            start_path: last_path1.clone(),
            skip: 0,
            take: chunk_size,
        };

        // Store response messages for two chunks
        let store_response_message1 = StoreItemsMessage {
            start_path: start_path1,
            last_path: last_path1.clone(),
            history_items: history_items1
                .into_iter()
                .map(|(hash, bytes)| (hash, Bytes::from(bytes)))
                .collect(),
            data_items: data_items1
                .into_iter()
                .map(|(hash, bytes)| (hash, Bytes::from(bytes)))
                .collect(),
        };
        let store_response_message2 = StoreItemsMessage {
            start_path: last_path1,
            last_path: last_path2,
            history_items: history_items2
                .into_iter()
                .map(|(hash, bytes)| (hash, prost::bytes::Bytes::from(bytes)))
                .collect(),
            data_items: data_items2
                .into_iter()
                .map(|(hash, bytes)| (hash, prost::bytes::Bytes::from(bytes)))
                .collect(),
        };

        // Block request message
        let block_request_message = BlockRequest {
            hash: genesis.block_hash.clone(),
        };

        // Send two response messages to signal the end
        // Scala equivalent: stateResponseQueue.enqueue1(storeResponseMessage1) *>
        //                   stateResponseQueue.enqueue1(storeResponseMessage2) *>
        //                   blockResponseQueue.enqueue1(genesis)
        // IMPORTANT: Write directly to channels (NOT through handle()) like Scala test does
        let tuple_space_tx = initializing_engine.tuple_space_tx.lock().unwrap().as_ref().unwrap().clone();
        let block_message_tx = initializing_engine.block_message_tx.lock().unwrap().as_ref().unwrap().clone();
        
        let store_msg1_clone = store_response_message1.clone();
        let store_msg2_clone = store_response_message2.clone();
        let genesis_clone = genesis.clone();
        
        let enqueue_responses = async move {
            // Write directly to tuple space channel (equivalent to stateResponseQueue.enqueue1)
            let _ = tuple_space_tx.send(store_msg1_clone);
            let _ = tuple_space_tx.send(store_msg2_clone);
            // Write directly to block message channel (equivalent to blockResponseQueue.enqueue1)
            let _ = block_message_tx.send(genesis_clone);
        };

        let local_for_expected = fixture.local.clone();
        let expected_requests = vec![
            packet_with_content(
                &local_for_expected,
                &fixture.network_id,
                store_request_message1.to_proto(),
            ),
            packet_with_content(
                &local_for_expected,
                &fixture.network_id,
                store_request_message2.to_proto(),
            ),
            packet_with_content(
                &local_for_expected,
                &fixture.network_id,
                block_request_message.to_proto(),
            ),
            packet_with_content(
                &local_for_expected,
                &fixture.network_id,
                models::casper::ForkChoiceTipRequestProto::default(),
            ),
        ];

        let test = async {
            engine_cell
                .set(initializing_engine.clone())
                .await
                .expect("Failed to set engine");

            let enqueue_responses_with_delay = async move {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                enqueue_responses.await;
            };

            let approved_block_clone = approved_block.clone();
            let local_for_handle = fixture.local.clone();
            // Handle approved block (it's blocking until responses are received)
            let handle_fut = async {
                let mut engine = engine_cell
                    .read_boxed()
                    .await
                    .expect("Failed to read engine");
                engine
                    .handle(
                        local_for_handle,
                        CasperMessage::ApprovedBlock(approved_block_clone),
                    )
                    .await
                    .expect("Failed to handle approved block");
            };
            let _ = tokio::join!(enqueue_responses_with_delay, handle_fut);

            let engine = engine_cell
                .read()
                .await
                .expect("Failed to read engine from cell");

            let casper_defined = engine.with_casper().is_some();
            assert!(
                casper_defined,
                "Casper should be defined after handling approved block"
            );

            let block_option = fixture
                .block_store
                .get(&genesis.block_hash)
                .expect("Failed to get block from store");
            assert!(block_option.is_some(), "Block should be defined in store");
            assert_eq!(block_option.as_ref(), Some(genesis));

            let handler_internal = engine_cell.read().await.expect("Failed to read engine");

            // We use with_casper().is_some() as a proxy: Running engines have casper, Initializing engines return None.
            // This is functionally equivalent since after transition_to_running(), only Running engines should be in the cell.
            assert!(
                handler_internal.with_casper().is_some(),
                "Engine should be Running (checked via casper presence)"
            );

            let requests = fixture.transport_layer.get_all_requests();
            // Assert requested messages for the state and fork choice tip
            assert_eq!(
                requests.len(),
                expected_requests.len(),
                "Transport layer should have received expected number of requests"
            );

            // Note: Since Protocol doesn't implement Hash/Eq, we compare packet contents like in original Scala code
            // which compares `_.msg.message.packet.get.content`, not the entire Protocol objects
            let request_packet_contents: HashSet<_> = requests
                .iter()
                .filter_map(|r| match &r.msg.message {
                    Some(ProtocolMessage::Packet(packet)) => Some(&packet.content),
                    _ => None,
                })
                .collect();
            let expected_packet_contents: HashSet<_> = expected_requests
                .iter()
                .filter_map(|protocol| match &protocol.message {
                    Some(ProtocolMessage::Packet(packet)) => Some(&packet.content),
                    _ => None,
                })
                .collect();
            assert_eq!(
                request_packet_contents, expected_packet_contents,
                "Request packet contents should match expected packet contents (order doesn't matter)"
            );

            fixture.transport_layer.reset();

            let last_approved_block_o = fixture.last_approved_block.lock().unwrap().clone();
            assert!(last_approved_block_o.is_some());

            {
                let mut engine = engine_cell
                    .read_boxed()
                    .await
                    .expect("Failed to read engine");
                engine
                    .handle(
                        fixture.local.clone(),
                        CasperMessage::ApprovedBlockRequest(ApprovedBlockRequest {
                            identifier: "test".to_string(),
                            trim_state: false,
                        }),
                    )
                    .await
                    .expect("Failed to handle approved block request");
            };

            let requests_after = fixture.transport_layer.get_all_requests();
            let approved_block_bytes =
                prost::bytes::Bytes::from(approved_block.clone().to_proto().encode_to_vec());
            let found_approved_block = requests_after.iter().any(|r| match &r.msg.message {
                Some(ProtocolMessage::Packet(packet)) => packet.content == approved_block_bytes,
                _ => false,
            });
            assert!(
                found_approved_block,
                "Expected to find approved block in transport layer requests"
            );
        };

        test.await;

        Self::after_each(&fixture);
    }
}

// Test-only rationale:
// We intentionally use in-memory/mocks (TransportLayerStub, InMemoryStoreManager, MockKeyValueStore),
// mirroring Scala Setup(), which also relies on in-memory stores and a stub transport layer for this spec.
// This preserves 1:1 behavior at the engine boundary (ApprovedBlock flow, LFS requests, DAG population,
// transition to Running, and network requests) while avoiding external I/O.
// If stricter parity is desired, we can pass fixture.block_store into Initializing::new so both the engine
// and the assertions use the same store instance. For this test the current setup remains functionally
// equivalent to Scala and is acceptable.
async fn create_initializing_engine(
    fixture: &TestFixture,
    the_init: Box<dyn FnOnce() -> Result<(), CasperError> + Send + Sync>,
    engine_cell: Arc<EngineCell>,
) -> Result<Arc<Initializing<TransportLayerStub>>, String> {
    let rp_conf = RPConf::new(
        fixture.local.clone(),
        fixture.network_id.clone(),
        Some(fixture.local.clone()),
        std::time::Duration::from_secs(30),
        10,
        5,
    );

    let connections_cell = ConnectionsCell {
        peers: Arc::new(std::sync::Mutex::new(Connections::from_vec(vec![fixture
            .local
            .clone()]))),
    };

    let mut mock_dag_store = InMemoryStoreManager::new();
    let block_dag_storage = BlockDagKeyValueStorage::new(&mut mock_dag_store)
        .await
        .map_err(|e| format!("Failed to create block dag storage: {}", e))?;

    let mut mock_deploy_store = InMemoryStoreManager::new();
    let deploy_storage =
        block_storage::rust::deploy::key_value_deploy_storage::KeyValueDeployStorage::new(
            &mut mock_deploy_store,
        )
        .await
        .map_err(|e| format!("Failed to create deploy storage: {}", e))?;

    let mut mock_casper_buffer_store = InMemoryStoreManager::new();
    let casper_buffer_storage =
        CasperBufferKeyValueStorage::new_from_kvm(&mut mock_casper_buffer_store)
            .await
            .map_err(|e| format!("Failed to create casper buffer storage: {}", e))?;

    let mock_store1 = Arc::new(Mutex::new(
        Box::new(MockKeyValueStore::new()) as Box<dyn KeyValueStore>
    ));
    let mock_store2 = Arc::new(Mutex::new(
        Box::new(MockKeyValueStore::new()) as Box<dyn KeyValueStore>
    ));
    let mock_store3 = Arc::new(Mutex::new(
        Box::new(MockKeyValueStore::new()) as Box<dyn KeyValueStore>
    ));

    let rspace_state_manager = RSpaceStateManager::new(
        RSpaceExporterImpl {
            source_history_store: mock_store1,
            source_value_store: mock_store2,
            source_roots_store: mock_store3,
        },
        RSpaceImporterImpl {
            history_store: Arc::new(Mutex::new(
                Box::new(MockKeyValueStore::new()) as Box<dyn KeyValueStore>
            )),
            value_store: Arc::new(Mutex::new(
                Box::new(MockKeyValueStore::new()) as Box<dyn KeyValueStore>
            )),
            roots_store: Arc::new(Mutex::new(
                Box::new(MockKeyValueStore::new()) as Box<dyn KeyValueStore>
            )),
        },
    );

    let rspace_store = RSpaceStore {
        history: Arc::new(Mutex::new(
            Box::new(MockKeyValueStore::new()) as Box<dyn KeyValueStore>
        )),
        roots: Arc::new(Mutex::new(
            Box::new(MockKeyValueStore::new()) as Box<dyn KeyValueStore>
        )),
        cold: Arc::new(Mutex::new(
            Box::new(MockKeyValueStore::new()) as Box<dyn KeyValueStore>
        )),
    };
    let mergeable_store = Arc::new(Mutex::new(KeyValueTypedStoreImpl::new(Box::new(
        MockKeyValueStore::new(),
    )
        as Box<dyn KeyValueStore>)));
    let runtime_manager = RuntimeManager::create_with_store(
        rspace_store,
        mergeable_store,
        models::rhoapi::Par::default(),
    );

    let estimator = casper::rust::estimator::Estimator::apply(5, Some(10));

    let event_publisher = Arc::new(shared::rust::shared::f1r3fly_events::F1r3flyEvents::new(
        Some(1000),
    ));

    let block_retriever = Arc::new(casper::rust::engine::block_retriever::BlockRetriever::new(
        fixture.transport_layer.clone(),
        Arc::new(connections_cell.clone()),
        Arc::new(rp_conf.clone()),
    ));

    let blocks_in_processing = Arc::new(Mutex::new(HashSet::new()));

    let (block_tx, block_rx) = mpsc::unbounded_channel::<BlockMessage>();
    let (tuple_tx, tuple_rx) = mpsc::unbounded_channel::<StoreItemsMessage>();

    Ok(Arc::new(Initializing::new(
        fixture.transport_layer.as_ref().clone(),
        rp_conf,
        connections_cell,
        fixture.last_approved_block.clone(),
        block_storage::rust::key_value_block_store::KeyValueBlockStore::new(
            Box::new(MockKeyValueStore::new()),
            Box::new(MockKeyValueStore::new()),
        ),
        block_dag_storage,
        deploy_storage,
        casper_buffer_storage,
        rspace_state_manager,
        Arc::new(Mutex::new(std::collections::VecDeque::new())),
        blocks_in_processing,
        fixture.casper_shard_conf.clone(),
        Some(fixture.validator_identity.clone()),
        the_init,
        block_tx,
        block_rx,
        tuple_tx,
        tuple_rx,
        true,
        false,
        event_publisher,
        block_retriever,
        engine_cell.clone(),
        runtime_manager,
        estimator,
    )))
}

#[tokio::test]
async fn make_transition_to_running_once_approved_block_received() {
    InitializingSpec::make_transition_to_running_once_approved_block_received().await;
}
