// See casper/src/test/scala/coop/rchain/casper/engine/InitializingSpec.scala

use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::mpsc;

use crypto::rust::{
    hash::blake2b256::Blake2b256,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg},
};
use models::casper::Signature;
use models::routing::protocol::Message as ProtocolMessage;
use models::rust::casper::protocol::casper_message::{
    ApprovedBlock, ApprovedBlockRequest, BlockMessage, BlockRequest,
    CasperMessage, StoreItemsMessage, StoreItemsMessageRequest,
};
use prost::bytes::Bytes;
use prost::Message;
use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
use shared::rust::shared::f1r3fly_events::{EventPublisher, EventPublisherFactory};

use crate::engine::setup::{TestFixture};
use casper::rust::engine::engine::Engine;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::engine::initializing::Initializing;
use casper::rust::engine::lfs_tuple_space_requester;

use casper::rust::errors::CasperError;
use comm::rust::rp::connect::{Connections, ConnectionsCell};
use comm::rust::rp::protocol_helper::packet_with_content;
use comm::rust::rp::rp_conf::RPConf;
use comm::rust::test_instances::TransportLayerStub;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::shared::in_mem_store_manager::InMemoryStoreManager;
use rspace_plus_plus::rspace::state::rspace_exporter::RSpaceExporter;
use rspace_plus_plus::rspace::state::exporters::rspace_exporter_items::RSpaceExporterItems;
use shared::rust::ByteVector;

/// **Scala equivalent**: `class InitializingSpec extends WordSpec with BeforeAndAfterEach`
struct InitializingSpec;

impl InitializingSpec {
    /// **Scala equivalent**: `implicit val eventBus = EventPublisher.noop[Task]`
    fn event_bus() -> Box<dyn EventPublisher> {
        EventPublisherFactory::noop()
    }
    /// **Scala equivalent**: `"Initializing state" should { "make a transition to Running once ApprovedBlock has been received" in { ... } }`

    /// Extracted beforeEach logic from Scala: sets transport layer responses
    fn before_each(fixture: &TestFixture) {
        fixture
            .transport_layer
            .set_responses(|_peer, _protocol| Ok(()));
    }

    /// Extracted afterEach logic from Scala: resets transport layer
    fn after_each(fixture: &TestFixture) {
        fixture.transport_layer.reset();
    }
    async fn make_transition_to_running_once_approved_block_received() {
        // **Scala equivalent**: `implicit val eventBus = EventPublisher.noop[Task]`
        let _event_bus = Self::event_bus();

        // **Scala equivalent**: `val fixture = Setup()`
        println!("Creating TestFixture...");
        let fixture = TestFixture::new().await;
        println!("TestFixture created successfully");

        // **Scala equivalent**: `override def beforeEach(): Unit = transportLayer.setResponses(_ => p => Right(()))`
        Self::before_each(&fixture);

        // **Scala equivalent**: `val theInit = Task.unit`
        let the_init = || Ok::<(), casper::rust::errors::CasperError>(());

        // **Scala equivalent**: `val blockResponseQueue = Queue.unbounded[Task, BlockMessage].runSyncUnsafe()`
        let (block_response_tx, block_response_rx) = mpsc::unbounded_channel::<BlockMessage>();

        // **Scala equivalent**: `val stateResponseQueue = Queue.unbounded[Task, StoreItemsMessage].runSyncUnsafe()`
        let (state_response_tx, state_response_rx) =
            mpsc::unbounded_channel::<StoreItemsMessage>();

        // **Scala equivalent**: `implicit val engineCell = Cell.unsafe[Task, Engine[Task]](Engine.noop)`
        let engine_cell = Arc::new(EngineCell::unsafe_init().expect("Failed to create EngineCell"));

        // **Scala equivalent**: `new Initializing[Task](...)`
        let initializing_engine = Arc::new(
            create_initializing_engine(
                &fixture,
                Box::new(the_init),
                (state_response_tx.clone(), block_response_rx, state_response_rx),
                engine_cell.clone(),
            )
            .await
            .expect("Failed to create Initializing engine"),
        );

        // **Scala equivalent**: `val approvedBlockCandidate = ApprovedBlockCandidate(block = genesis, requiredSigs = 0)`
        // **Scala equivalent**: `val (validatorSk, validatorPk) = context.validatorKeyPairs.head`
        // All these values are now available from fixture (like Scala's `import fixture._`)
        let genesis = &fixture.genesis;
        let approved_block_candidate = fixture.approved_block_candidate.clone();
        let validator_sk = &fixture.validator_sk;
        let validator_pk = &fixture.validator_pk;

        // **Scala equivalent**: `val approvedBlock: ApprovedBlock = ApprovedBlock(...)`
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

        // **Scala equivalent**: `val genesisExporter = exporter`
        let genesis_exporter = &fixture.exporter;

        // **Scala equivalent**: `val chunkSize = LfsTupleSpaceRequester.pageSize`
        let chunk_size = lfs_tuple_space_requester::PAGE_SIZE;

        // **Scala equivalent**: `def genesisExport(startPath: Seq[(Blake2b256Hash, Option[Byte])]) = ...`
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
            // **Scala equivalent**: `val history_and_data_items = RSpaceExporterItems::get_history_and_data(...)`
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

        // Get history and data items from genesis block
        // Convert Box<dyn RSpaceExporter> to Arc<Mutex<Box<dyn RSpaceExporter>>> for RSpaceExporterItems
        // Since we can't clone trait objects, we'll create a new exporter from the same RSpaceStore
        let rspace_store = &fixture.rspace_store;
        let genesis_exporter_impl = rspace_plus_plus::rspace::state::instances::rspace_exporter_store::RSpaceExporterStore::create(
            rspace_store.history.clone(),
            rspace_store.cold.clone(),
            rspace_store.roots.clone(),
        );
        let genesis_exporter_new: Box<dyn RSpaceExporter> = Box::new(genesis_exporter_impl);
        let genesis_exporter_arc = std::sync::Arc::new(std::sync::Mutex::new(genesis_exporter_new));
        
        let (history_items1, data_items1, last_path1) =
            genesis_export(genesis_exporter_arc.clone(), start_path1.clone(), &fixture.exporter_params)
                .expect("Failed to export history and data items 1");
        let (history_items2, data_items2, last_path2) =
            genesis_export(genesis_exporter_arc.clone(), last_path1.clone(), &fixture.exporter_params)
                .expect("Failed to export history and data items 2");

        // Store request message
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

        // Store response message
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

        let enqueue_responses = {
            let state_tx_clone = state_response_tx.clone();
            let block_tx_clone = block_response_tx.clone();
            let store_msg1_clone = store_response_message1.clone();
            let store_msg2_clone = store_response_message2.clone();
            let genesis_clone = genesis.clone();

            async move {
                state_tx_clone.send(store_msg1_clone).unwrap();
                state_tx_clone.send(store_msg2_clone).unwrap();
                block_tx_clone.send(genesis_clone).unwrap();
            }
        };

        let expected_requests = vec![
            packet_with_content(
                &fixture.local,
                &fixture.network_id,
                store_request_message1.to_proto(),
            ),
            packet_with_content(
                &fixture.local,
                &fixture.network_id,
                store_request_message2.to_proto(),
            ),
            packet_with_content(
                &fixture.local,
                &fixture.network_id,
                block_request_message.to_proto(),
            ),
            packet_with_content(
                &fixture.local,
                &fixture.network_id,
                models::casper::ForkChoiceTipRequestProto::default(),
            ),
        ];

        let test = async {
            // **Scala equivalent**: `_ <- EngineCell[Task].set(initializingEngine)`
            // TODO: CRITICAL - This step is essential for the test to work correctly!
            //
            // PROBLEM: We need to set the Initializing engine in EngineCell, but we have Arc<Mutex<Initializing>>
            // which doesn't implement Engine trait directly.
            //
            // In Scala, this works simply:
            // ```scala
            // val initializingEngine: Initializing[Task] = new Initializing[Task](...)
            // _ <- EngineCell[Task].set(initializingEngine)  // Works directly
            // _ <- initializingEngine.handle(local, approvedBlock)  // Mutable access works
            // ```
            //
            // In Rust, we have ownership/mutability conflicts:
            // ```rust
            // let initializing_engine = Arc::new(Mutex::new(create_initializing_engine(...)));
            engine_cell.set(initializing_engine.clone()).await.expect("Failed to set engine");  // ERROR: Mutex<Initializing> doesn't implement Engine
            // let mut guard = initializing_engine.lock().await;
            // guard.handle(...).await  // Mutable access through guard
            // ```
            //
            // TODO: How to fix it?

            tokio::spawn(async move {
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                enqueue_responses.await;
            });

            // Handle approved block (it's blocking until responses are received)
						let mut engine = engine_cell.read_boxed().await.expect("Failed to read engine");
            engine
                .handle(fixture.local.clone(), CasperMessage::ApprovedBlock(approved_block.clone()))
                .await
                .expect("Failed to handle approved block");

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

            // Note: In Rust we can't directly check type like Scala's isInstanceOf.
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

            // **Scala equivalent**: `messages.toSet == expectedRequests.toSet`
            // Note: Since Protocol doesn't implement Hash/Eq, we compare packet contents like in original Scala code
            // which compares `_.msg.message.packet.get.content`, not the entire Protocol objects
            let request_packet_contents: HashSet<_> = requests.iter()
                .filter_map(|r| match &r.msg.message {
                    Some(ProtocolMessage::Packet(packet)) => Some(&packet.content),
                    _ => None,
                })
                .collect();
            let expected_packet_contents: HashSet<_> = expected_requests.iter()
                .filter_map(|protocol| match &protocol.message {
                    Some(ProtocolMessage::Packet(packet)) => Some(&packet.content),
                    _ => None,
                })
                .collect();
            assert_eq!(
                request_packet_contents, expected_packet_contents,
                "Request packet contents should match expected packet contents (order doesn't matter)"
            );

            // **Scala equivalent**: `_ = transportLayer.reset()`
            fixture.transport_layer.reset();

            // **Scala equivalent**: `lastApprovedBlockO <- LastApprovedBlock[Task].get`
            let last_approved_block_o = fixture.last_approved_block.lock().unwrap().clone();
            // **Scala equivalent**: `_ = assert(lastApprovedBlockO.isDefined)`
            assert!(last_approved_block_o.is_some());

            // **Scala equivalent**: `_ <- EngineCell[Task].read >>= (_.handle(local, ApprovedBlockRequest("test", trimState = false)))`
            {
                let mut engine = engine_cell.read_boxed().await.expect("Failed to read engine");
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

            // **Scala equivalent**: `_ = assert(transportLayer.requests.exists(_.msg.message.packet.get.content == approvedBlock.toProto.toByteString))`
            let requests_after = fixture.transport_layer.get_all_requests();
            let approved_block_bytes = prost::bytes::Bytes::from(approved_block.clone().to_proto().encode_to_vec());
            let found_approved_block = requests_after.iter().any(|r| match &r.msg.message {
                Some(ProtocolMessage::Packet(packet)) => packet.content == approved_block_bytes,
                _ => false,
            });
            assert!(found_approved_block, "Expected to find approved block in transport layer requests");
        };

        // **Scala equivalent**: `test.unsafeRunSync`
        test.await;

        // **Scala equivalent**: `override def afterEach(): Unit = transportLayer.reset()`
        Self::after_each(&fixture);
    }
}

//TODO should we use mocks from setup.rs or real impls?
async fn create_initializing_engine(
    fixture: &TestFixture,
    the_init: Box<dyn FnOnce() -> Result<(),CasperError> + Send + Sync>,
    _tuple: (mpsc::UnboundedSender<StoreItemsMessage>, mpsc::UnboundedReceiver<BlockMessage>, mpsc::UnboundedReceiver<StoreItemsMessage>),
    engine_cell: Arc<EngineCell>,
) -> Result<Initializing<TransportLayerStub>, String> {
    let (_state_response_tx, block_response_rx, state_response_rx) = _tuple;
    let rp_conf = RPConf::new(
        fixture.local.clone(),
        fixture.network_id.clone(),
        Some(fixture.local.clone()),
        std::time::Duration::from_secs(30),
        10,
        5,
    );

    let connections_cell = ConnectionsCell {
        peers: Arc::new(std::sync::Mutex::new(
            Connections::from_vec(vec![fixture.local.clone()]),
        )),
    };

    let mut mock_dag_store =
        InMemoryStoreManager::new();
    let block_dag_storage =
        BlockDagKeyValueStorage::new(
            &mut mock_dag_store,
        )
        .await
        .map_err(|e| format!("Failed to create block dag storage: {}", e))?;

    let mut mock_deploy_store =
        rspace_plus_plus::rspace::shared::in_mem_store_manager::InMemoryStoreManager::new();
    let deploy_storage =
        block_storage::rust::deploy::key_value_deploy_storage::KeyValueDeployStorage::new(
            &mut mock_deploy_store,
        )
        .await
        .map_err(|e| format!("Failed to create deploy storage: {}", e))?;

    let mut mock_casper_buffer_store =
        rspace_plus_plus::rspace::shared::in_mem_store_manager::InMemoryStoreManager::new();
    let casper_buffer_storage = block_storage::rust::casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage::new_from_kvm(&mut mock_casper_buffer_store).await
        .map_err(|e| format!("Failed to create casper buffer storage: {}", e))?;

    let mock_store1 = Arc::new(std::sync::Mutex::new(Box::new(
        crate::util::test_mocks::MockKeyValueStore::new(),
    )
        as Box<dyn shared::rust::store::key_value_store::KeyValueStore>));
    let mock_store2 = Arc::new(std::sync::Mutex::new(Box::new(
        crate::util::test_mocks::MockKeyValueStore::new(),
    )
        as Box<dyn shared::rust::store::key_value_store::KeyValueStore>));
    let mock_store3 = Arc::new(std::sync::Mutex::new(Box::new(
        crate::util::test_mocks::MockKeyValueStore::new(),
    )
        as Box<dyn shared::rust::store::key_value_store::KeyValueStore>));

    let rspace_state_manager =
        rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager::new(
            rspace_plus_plus::rspace::state::instances::rspace_exporter_store::RSpaceExporterImpl {
                source_history_store: mock_store1,
                source_value_store: mock_store2,
                source_roots_store: mock_store3,
            },
            rspace_plus_plus::rspace::state::instances::rspace_importer_store::RSpaceImporterImpl {
                history_store: Arc::new(std::sync::Mutex::new(Box::new(
                    crate::util::test_mocks::MockKeyValueStore::new(),
                )
                    as Box<dyn shared::rust::store::key_value_store::KeyValueStore>)),
                value_store: Arc::new(std::sync::Mutex::new(Box::new(
                    crate::util::test_mocks::MockKeyValueStore::new(),
                )
                    as Box<dyn shared::rust::store::key_value_store::KeyValueStore>)),
                roots_store: Arc::new(std::sync::Mutex::new(Box::new(
                    crate::util::test_mocks::MockKeyValueStore::new(),
                )
                    as Box<dyn shared::rust::store::key_value_store::KeyValueStore>)),
            },
        );

    let rspace_store = rspace_plus_plus::rspace::rspace::RSpaceStore {
        history: Arc::new(std::sync::Mutex::new(Box::new(
            crate::util::test_mocks::MockKeyValueStore::new(),
        )
            as Box<dyn shared::rust::store::key_value_store::KeyValueStore>)),
        roots: Arc::new(std::sync::Mutex::new(Box::new(
            crate::util::test_mocks::MockKeyValueStore::new(),
        )
            as Box<dyn shared::rust::store::key_value_store::KeyValueStore>)),
        cold: Arc::new(std::sync::Mutex::new(Box::new(
            crate::util::test_mocks::MockKeyValueStore::new(),
        )
            as Box<dyn shared::rust::store::key_value_store::KeyValueStore>)),
    };
    let mergeable_store = Arc::new(std::sync::Mutex::new(
        shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl::new(Box::new(
            crate::util::test_mocks::MockKeyValueStore::new(),
        )
            as Box<dyn shared::rust::store::key_value_store::KeyValueStore>),
    ));
    let runtime_manager =
        casper::rust::util::rholang::runtime_manager::RuntimeManager::create_with_store(
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

    let blocks_in_processing = Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));

    Ok(Initializing::new(
        fixture.transport_layer.as_ref().clone(),
        rp_conf,
        connections_cell,
        fixture.last_approved_block.clone(),
        block_storage::rust::key_value_block_store::KeyValueBlockStore::new(
            Box::new(crate::util::test_mocks::MockKeyValueStore::new()),
            Box::new(crate::util::test_mocks::MockKeyValueStore::new()),
        ),
        block_dag_storage,
        deploy_storage,
        casper_buffer_storage,
        rspace_state_manager,
        Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new())),
        blocks_in_processing,
        fixture.casper_shard_conf.clone(),
        Some(fixture.validator_identity.clone()),
        the_init,
        block_response_rx,
        state_response_rx,
        true,
        false,
        event_publisher,
        block_retriever,
        engine_cell.clone(),
        runtime_manager,
        estimator,
    ))
}

#[tokio::test]
async fn make_transition_to_running_once_approved_block_received() {
    InitializingSpec::make_transition_to_running_once_approved_block_received().await;
}
