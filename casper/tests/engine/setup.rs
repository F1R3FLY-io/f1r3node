// See casper/src/test/scala/coop/rchain/casper/engine/Setup.scala

use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::{
        block_dag_key_value_storage::{BlockDagKeyValueStorage, DeployId},
        block_metadata_store::BlockMetadataStore,
        equivocation_tracker_store::EquivocationTrackerStore,
    },
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    key_value_block_store::KeyValueBlockStore,
};
use casper::rust::{
    engine::{
        block_approver_protocol::BlockApproverProtocol, block_retriever, engine_cell::EngineCell,
        running::Running,
    },
    validator_identity::ValidatorIdentity,
};
use comm::rust::{
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
    rp::connect::{Connections, ConnectionsCell},
    rp::rp_conf::RPConf,
    test_instances::{create_rp_conf_ask, TransportLayerStub},
};
use crypto::rust::{private_key::PrivateKey, public_key::PublicKey};
use models::{
    routing::Protocol,
    rust::{
        block_hash::{BlockHash, BlockHashSerde},
        block_metadata::BlockMetadata,
        casper::protocol::casper_message::{
            ApprovedBlock, ApprovedBlockCandidate, BlockMessage, CasperMessage, HasBlock,
        },
        equivocation_record::SequenceNumber,
        validator::ValidatorSerde,
    },
};
use prost::bytes::Bytes;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;
use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::util::rholang::resources::mk_test_rnode_store_manager;
use crate::{
    helper::no_ops_casper_effect::NoOpsCasperEffect,
    util::{genesis_builder::GenesisBuilder, test_mocks::MockKeyValueStore},
};
use casper::rust::casper::{CasperShardConf, MultiParentCasper};
use casper::rust::errors::CasperError;
use casper::rust::estimator::Estimator;
use casper::rust::genesis::genesis::Genesis;
use casper::rust::util::rholang::runtime_manager::RuntimeManager;
use crypto::rust::signatures::signed::Signed;
use models::rust::casper::protocol::casper_message::DeployData;
use prost::Message;
use rspace_plus_plus::rspace::shared::rspace_store_manager::get_or_create_rspace_store;
use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;
use shared::rust::ByteString;

/// Test fixture struct to hold all test dependencies
pub struct TestFixture {
    // Scala: implicit val transportLayer = new TransportLayerStub[Task]
    pub transport_layer: Arc<TransportLayerStub>,
    // Scala: val local: PeerNode = peerNode("src", 40400)
    pub local: PeerNode,
    // Scala: val networkId = "test"
    pub network_id: String,
    // TODO NOT in Scala Setup - created locally in each Scala test as: implicit val casper = NoOpsCasperEffect[Task]().unsafeRunSync
    // In Rust TestFixture for convenience to avoid recreating in each test
    pub casper: NoOpsCasperEffect,
    // TODO NOT in Scala Setup - created locally in each test as: new Running(..., casper, approvedBlock, ...)
    // In Rust TestFixture for convenience to avoid recreating in each test
    // NOTE: Running now uses Arc<dyn MultiParentCasper> instead of generic M parameter
    pub engine: Running<TransportLayerStub>,
    // Scala: implicit val blockProcessingQueue = Queue.unbounded[Task, (Casper[Task], BlockMessage)]
    // NOTE: Changed to trait object to match Running changes
    pub block_processing_queue:
        Arc<Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>>,
    // Scala Step 4: implicit val rspaceStateManager = RSpacePlusPlusStateManagerImpl(exporter, importer)
    pub rspace_state_manager: Arc<Mutex<Option<RSpaceStateManager>>>,
    // Scala: implicit val runtimeManager = RuntimeManager[Task](rspace, replay, historyRepo, mStore, Genesis.NonNegativeMergeableTagName)
    pub runtime_manager: Arc<Mutex<RuntimeManager>>,
    // Scala: implicit val estimator = Estimator[Task](Estimator.UnlimitedParents, None)
    pub estimator: Arc<Mutex<Option<Estimator>>>,
    pub rspace_store: rspace_plus_plus::rspace::rspace::RSpaceStore,
    // Scala: implicit val blockStore = KeyValueBlockStore[Task](kvm).unsafeRunSync(...)
    pub block_store: Arc<Mutex<Option<KeyValueBlockStore>>>,
    // Scala: implicit val lab = LastApprovedBlock.of[Task].unsafeRunSync(...)
    pub last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
    // Scala: implicit val casperShardConf = CasperShardConf(-1, shardId, "", finalizationRate, ...)
    pub casper_shard_conf: CasperShardConf,
    // Scala: val genesis: BlockMessage = context.genesisBlock
    pub genesis: BlockMessage,
    // Scala: val (validatorSk, validatorPk) = context.validatorKeyPairs.head
    pub validator_sk: PrivateKey,
    // Scala: val (validatorSk, validatorPk) = context.validatorKeyPairs.head
    pub validator_pk: PublicKey,
    // TODO NOT in Scala Setup - created locally in each test as: ApprovedBlockCandidate(block = genesis, requiredSigs = 0)
    // In Rust TestFixture for convenience (used in initializing_spec.rs)
    pub approved_block_candidate: ApprovedBlockCandidate,
    // TODO NOT in Scala Setup - Scala passes inline: getHistoryAndData(startPath, skip = 0, take = chunkSize)
    // Rust extracts to struct for reusability (used multiple times in initializing_spec.rs)
    pub exporter_params: ExporterParams,
    // Scala: val requiredSigs = 1
    pub required_sigs: i32,
    // Scala: val params @ (_, _, genesisParams) = GenesisBuilder.buildGenesisParameters()
    pub genesis_parameters: casper::rust::genesis::genesis::Genesis,
    // Scala: val validatorId = ValidatorIdentity(validatorPk, validatorSk, "secp256k1")
    pub validator_id: ValidatorIdentity,
    // Scala: val bap = BlockApproverProtocol.of[Task](validatorId, deployTimestamp, ...)
    pub bap: BlockApproverProtocol<TransportLayerStub>,
    // Scala: implicit val blockProcessingState = Ref.of[Task, Set[BlockHash]](Set.empty)
    pub blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
    // Scala: implicit val rpConf = createRPConfAsk[Task](local)
    pub rp_conf_ask: RPConf,
    // Scala: implicit val connectionsCell: ConnectionsCell[Task] = Cell.unsafe[Task, Connections](List(local))
    pub connections_cell: ConnectionsCell,
    // TODO NOT in Scala Setup - created locally in each test as: implicit val eventBus = EventPublisher.noop[Task]
    // In Rust TestFixture for convenience to avoid recreating in each test
    pub event_publisher: Arc<F1r3flyEvents>,
    // Scala: implicit val blockRetriever = BlockRetriever.of[Task]
    pub block_retriever: Arc<block_retriever::BlockRetriever<TransportLayerStub>>,
    // TODO NOT in Scala Setup - created locally in each test as: implicit val engineCell = Cell.unsafe[Task, Engine[Task]](Engine.noop)
    // In Rust TestFixture for convenience to avoid recreating in each test
    pub engine_cell: Arc<EngineCell>,
    // Scala: implicit val blockDagStorage = BlockDagKeyValueStorage.create(kvm).unsafeRunSync(...)
    pub block_dag_storage: Arc<Mutex<Option<BlockDagKeyValueStorage>>>,
    // Scala: implicit val deployStorage = KeyValueDeployStorage[Task](kvm).unsafeRunSync(...)
    pub deploy_storage: Arc<Mutex<Option<KeyValueDeployStorage>>>,
    // Scala: implicit val casperBuffer = CasperBufferKeyValueStorage.create[Task](spaceKVManager).unsafeRunSync(...)
    pub casper_buffer_storage: Arc<Mutex<Option<CasperBufferKeyValueStorage>>>,
}

impl TestFixture {
    pub async fn new() -> Self {
        // Scala: val params @ (_, _, genesisParams) = GenesisBuilder.buildGenesisParameters()
        let mut genesis_builder = GenesisBuilder::new();
        let genesis_parameters_tuple =
            GenesisBuilder::build_genesis_parameters_with_defaults(None, None);
        let (validator_key_pairs_params, genesis_vaults_params, genesis_params) =
            genesis_parameters_tuple.clone();

        // Scala: val context = GenesisBuilder.buildGenesis(params)
        let context = genesis_builder
            .build_genesis_with_parameters(Some(genesis_parameters_tuple.clone()))
            .await
            .expect("Failed to build genesis context");

        // Scala: val genesis: BlockMessage = context.genesisBlock
        let genesis = context.genesis_block.clone();

        // Scala: val (validatorSk, validatorPk) = context.validatorKeyPairs.head
        let (validator_sk, validator_pk) = context
            .validator_key_pairs
            .first()
            .expect("No validator key pairs available")
            .clone();

        // Scala: val networkId = "test"
        let network_id = "test".to_string();

        // Scala: val spaceKVManager = mkTestRNodeStoreManager[Task](context.storageDirectory).runSyncUnsafe()
        // IMPORTANT: Use context.storage_directory (where genesis was created) to ensure same RSpace state!
        // This matches Scala Setup which uses context.storageDirectory for both genesis creation and tests
        let mut space_kv_manager = mk_test_rnode_store_manager(context.storage_directory.clone());

        // Scala Step 1-2: val spaces = RSpacePlusPlus_RhoTypes.createWithReplay[Task, ...](context.storageDirectory.toString())
        // Scala's createWithReplay calls Rust RSpace++ code which uses the real Matcher (not DummyMatcher)
        // In Rust, we must use RuntimeManager::create_with_history to match this behavior
        // Now using create_with_history (like GenesisBuilder does) which uses the real Matcher from RSpace++
        let rspace_store_path = context
            .storage_directory
            .to_str()
            .expect("Invalid storage directory path");

        let rspace_store_for_runtime =
            rspace_plus_plus::rspace::shared::rspace_store_manager::get_or_create_rspace_store(
                rspace_store_path,
                1024 * 1024 * 100, // 100MB map size
            )
            .expect("Failed to create RSpace store for RuntimeManager");

        // Scala: val mStore = RuntimeManager.mergeableStore(spaceKVManager).unsafeRunSync(scheduler)
        let m_store = RuntimeManager::mergeable_store(&mut space_kv_manager)
            .await
            .expect("Failed to create mergeable store");

        // Scala: implicit val runtimeManager = RuntimeManager[Task](rspace, replay, historyRepo, mStore, ...)
        // Rust equivalent: Use create_with_history instead of manual RSpace creation with DummyMatcher
        let (runtime_manager, history_repo) = RuntimeManager::create_with_history(
            rspace_store_for_runtime,
            m_store,
            Genesis::non_negative_mergeable_tag_name(),
        );

        // Create separate rspace_store for TestFixture field (RSpaceStore is not Clone)
        let rspace_store = get_or_create_rspace_store(
            rspace_store_path,
            1024 * 1024 * 100, // 100MB map size
        )
        .expect("Failed to create RSpace store for TestFixture");

        // Scala Step 3: val (exporter, importer) = { (historyRepo.exporter.unsafeRunSync, historyRepo.importer.unsafeRunSync) }
        let exporter_trait = history_repo.exporter();
        let importer_trait = history_repo.importer();
        let rspace_state_manager_unwrapped =
            RSpaceStateManager::new(exporter_trait, importer_trait);

        // Scala: val kvm = InMemoryStoreManager[Task]()
        // In Scala, InMemoryStoreManager creates separate stores for each name via kvm.store("name")
        // We simulate this by creating separate shared HashMaps for each "database name"
        let kvm_blockstorage = Arc::new(Mutex::new(HashMap::new()));
        let kvm_approved_block = Arc::new(Mutex::new(HashMap::new()));
        let kvm_dagstorage_metadata = Arc::new(Mutex::new(HashMap::new()));
        let kvm_dagstorage_deploy_index = Arc::new(Mutex::new(HashMap::new()));
        let kvm_dagstorage_latest_messages = Arc::new(Mutex::new(HashMap::new()));
        let kvm_dagstorage_invalid_blocks = Arc::new(Mutex::new(HashMap::new()));
        let kvm_dagstorage_equivocation_tracker = Arc::new(Mutex::new(HashMap::new()));
        let kvm_deploystorage = Arc::new(Mutex::new(HashMap::new()));

        // Scala: implicit val blockStore = KeyValueBlockStore[Task](kvm).unsafeRunSync(...)
        // Each storage gets its own "database" from kvm, equivalent to kvm.store("blockstorage")
        let store = Arc::new(MockKeyValueStore::with_shared_data(
            kvm_blockstorage.clone(),
        ));
        let store_approved_block = Arc::new(MockKeyValueStore::with_shared_data(
            kvm_approved_block.clone(),
        ));
        let mut block_store_unwrapped = KeyValueBlockStore::new(store, store_approved_block);
        block_store_unwrapped
            .put(genesis.block_hash.clone(), &genesis)
            .expect("Failed to store genesis block");

        // Scala: implicit val blockDagStorage = BlockDagKeyValueStorage.create(kvm).unsafeRunSync(...)
        // NOTE: Changed from KeyValueDagRepresentation to BlockDagKeyValueStorage because:
        // - Scala uses BlockDagStorage trait with insert() method
        // - In Rust, insert() is only on BlockDagKeyValueStorage (persistent storage)
        // - KeyValueDagRepresentation is just an in-memory snapshot
        // - GenesisValidator and Initializing need insert() to record blocks in DAG
        // - This matches Scala Setup.scala which creates BlockDagKeyValueStorage.create(kvm)
        let metadata_store = Arc::new(MockKeyValueStore::with_shared_data(
            kvm_dagstorage_metadata.clone(),
        ));
        let metadata_typed_store =
            KeyValueTypedStoreImpl::<BlockHashSerde, BlockMetadata>::new(metadata_store);
        let block_metadata_store = BlockMetadataStore::new(metadata_typed_store);

        let deploy_index_store = Arc::new(MockKeyValueStore::with_shared_data(
            kvm_dagstorage_deploy_index.clone(),
        ));
        let deploy_index_typed_store =
            KeyValueTypedStoreImpl::<DeployId, BlockHashSerde>::new(deploy_index_store);

        let latest_messages_store = Arc::new(MockKeyValueStore::with_shared_data(
            kvm_dagstorage_latest_messages.clone(),
        ));
        let latest_messages_typed_store =
            KeyValueTypedStoreImpl::<ValidatorSerde, BlockHashSerde>::new(latest_messages_store);

        let invalid_blocks_store = Arc::new(MockKeyValueStore::with_shared_data(
            kvm_dagstorage_invalid_blocks.clone(),
        ));
        let invalid_blocks_typed_store =
            KeyValueTypedStoreImpl::<BlockHashSerde, BlockMetadata>::new(invalid_blocks_store);

        let equivocation_tracker_store = Arc::new(MockKeyValueStore::with_shared_data(
            kvm_dagstorage_equivocation_tracker.clone(),
        ));
        let equivocation_tracker_typed_store = KeyValueTypedStoreImpl::<
            (ValidatorSerde, SequenceNumber),
            BTreeSet<BlockHashSerde>,
        >::new(equivocation_tracker_store);
        let equivocation_tracker = EquivocationTrackerStore::new(equivocation_tracker_typed_store);

        let mut block_dag_storage_unwrapped = BlockDagKeyValueStorage {
            latest_messages_index: latest_messages_typed_store,
            block_metadata_index: Arc::new(std::sync::RwLock::new(block_metadata_store)),
            deploy_index: Arc::new(std::sync::RwLock::new(deploy_index_typed_store)),
            invalid_blocks_index: invalid_blocks_typed_store,
            equivocation_tracker_index: equivocation_tracker,
        };

        // Insert genesis block into DAG storage (approved = true, invalid = false)
        block_dag_storage_unwrapped
            .insert(&genesis, false, true)
            .expect("Failed to insert genesis into BlockDagStorage");

        // OLD CODE (kept for reference, replaced with BlockDagKeyValueStorage):
        // let block_dag_storage = KeyValueDagRepresentation {
        //     dag_set: Default::default(),
        //     latest_messages_map: Default::default(),
        //     child_map: Default::default(),
        //     height_map: Default::default(),
        //     invalid_blocks_set: Default::default(),
        //     last_finalized_block_hash: genesis.block_hash.clone(),
        //     finalized_blocks_set: Default::default(),
        //     block_metadata_index: Arc::new(std::sync::RwLock::new(block_metadata_store)),
        //     deploy_index: Arc::new(std::sync::RwLock::new(deploy_typed_store)),
        // };
        // block_dag_storage.dag_set.insert(genesis.block_hash.clone());
        // block_dag_storage.finalized_blocks_set.insert(genesis.block_hash.clone());

        // Scala: implicit val deployStorage = KeyValueDeployStorage[Task](kvm).unsafeRunSync(...)
        // Equivalent to kvm.store("deploystorage")
        let deploy_storage_store = Arc::new(MockKeyValueStore::with_shared_data(
            kvm_deploystorage.clone(),
        ));
        let deploy_storage_typed_store =
            KeyValueTypedStoreImpl::<ByteString, Signed<DeployData>>::new(deploy_storage_store);
        let deploy_storage_unwrapped = KeyValueDeployStorage {
            store: deploy_storage_typed_store,
        };

        // Scala: implicit val estimator = Estimator[Task](Estimator.UnlimitedParents, None)
        let estimator_unwrapped = Estimator::apply(Estimator::UNLIMITED_PARENTS, None);

        // Create NoOpsCasperEffect with comprehensive dependencies from genesis context
        // NoOpsCasperEffect will use the same kvm_blockstorage for its internal block store
        // This ensures consistency with the external block_store
        // NOTE: NoOpsCasperEffect requires KeyValueDagRepresentation, so we get it from BlockDagKeyValueStorage
        let block_dag_representation = block_dag_storage_unwrapped.get_representation();

        // Wrap RuntimeManager in Arc<Mutex<>> for shared mutable access
        let runtime_manager_shared = Arc::new(Mutex::new(runtime_manager));

        let mut casper = NoOpsCasperEffect::new_with_shared_kvm(
            None, // estimator_func
            runtime_manager_shared.clone(),
            block_store_unwrapped.clone(),
            block_dag_representation,
            kvm_blockstorage.clone(),
        );

        // Add the genesis block using the new test-friendly methods
        casper.add_block_to_store(genesis.clone());
        casper.add_to_dag(genesis.block_hash.clone());

        // NOTE: Changed to trait object to match Running changes
        let block_processing_queue: Arc<
            Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>,
        > = Arc::new(Mutex::new(VecDeque::new()));

        let approved_block = ApprovedBlock {
            candidate: ApprovedBlockCandidate {
                block: genesis.clone(),
                required_sigs: 0,
            },
            sigs: Vec::new(),
        };

        let approved_block_candidate = ApprovedBlockCandidate {
            block: genesis.clone(),
            required_sigs: 0,
        };

        // Scala: implicit val casperBuffer = CasperBufferKeyValueStorage.create[Task](spaceKVManager).unsafeRunSync(...)
        let casper_buffer_storage_unwrapped =
            CasperBufferKeyValueStorage::new_from_kvm(&mut space_kv_manager)
                .await
                .expect("Failed to create CasperBufferKeyValueStorage");

        let last_approved_block = Arc::new(Mutex::new(None::<ApprovedBlock>));

        // Scala: val shardId = genesisParams.shardId
        // Scala: val finalizationRate = 1
        let shard_id = genesis_params.shard_id.clone();
        let finalization_rate = 1i32;

        // Scala: implicit val casperShardConf = CasperShardConf(-1, shardId, "", finalizationRate, ...)
        let mut casper_shard_conf = CasperShardConf::new();
        casper_shard_conf.fault_tolerance_threshold = -1.0;
        casper_shard_conf.shard_name = shard_id;
        casper_shard_conf.parent_shard_id = "".to_string();
        casper_shard_conf.finalization_rate = finalization_rate;
        casper_shard_conf.max_number_of_parents = i32::MAX;
        casper_shard_conf.max_parent_depth = i32::MAX;
        casper_shard_conf.synchrony_constraint_threshold = 0.0;
        casper_shard_conf.height_constraint_threshold = i64::MAX;
        casper_shard_conf.deploy_lifespan = 50;
        casper_shard_conf.casper_version = 1;
        casper_shard_conf.config_version = 1;
        casper_shard_conf.bond_minimum = genesis_params.proof_of_stake.minimum_bond;
        casper_shard_conf.bond_maximum = genesis_params.proof_of_stake.maximum_bond;
        casper_shard_conf.epoch_length = genesis_params.proof_of_stake.epoch_length;
        casper_shard_conf.quarantine_length = genesis_params.proof_of_stake.quarantine_length;
        casper_shard_conf.min_phlo_price = 1;

        // **Scala equivalent**: exporter params with skip and take
        let exporter_params = ExporterParams {
            skip: 0,
            take: casper::rust::engine::lfs_tuple_space_requester::PAGE_SIZE,
        };

        // Scala: val requiredSigs = 1 from Setup.scala
        let required_sigs = 1i32;

        // Scala: val validatorId = ValidatorIdentity(validatorPk, validatorSk, "secp256k1")
        let validator_id = ValidatorIdentity {
            public_key: validator_pk.clone(),
            private_key: validator_sk.clone(),
            signature_algorithm: "secp256k1".to_string(),
        };

        // Scala: val deployTimestamp = 0L (from Setup.scala, used for bap)
        let deploy_timestamp = 0i64;

        // Extract bonds from genesis validators for bap
        let bonds: HashMap<PublicKey, i64> = genesis_params
            .proof_of_stake
            .validators
            .iter()
            .map(|v| (v.pk.clone(), v.stake))
            .collect();

        // Scala: val local: PeerNode = peerNode("src", 40400)
        let local = peer_node("src", 40400);

        // Scala: implicit val transportLayer = new TransportLayerStub[Task]
        let transport_layer = Arc::new(TransportLayerStub::new());

        // Scala: implicit val rpConf = createRPConfAsk[Task](local)
        let rp_conf = create_rp_conf_ask(local.clone(), None, None);

        // Scala: val bap = BlockApproverProtocol.of[Task](...)
        let bap = BlockApproverProtocol::new(
            validator_id.clone(),
            deploy_timestamp,
            genesis_params.vaults.clone(),
            bonds,
            genesis_params.proof_of_stake.minimum_bond,
            genesis_params.proof_of_stake.maximum_bond,
            genesis_params.proof_of_stake.epoch_length,
            genesis_params.proof_of_stake.quarantine_length,
            genesis_params.proof_of_stake.number_of_active_validators,
            required_sigs,
            genesis_params
                .proof_of_stake
                .pos_multi_sig_public_keys
                .clone(),
            genesis_params.proof_of_stake.pos_multi_sig_quorum,
            transport_layer.clone(),
            Arc::new(rp_conf.clone()),
        )
        .expect("Failed to create BlockApproverProtocol");

        // Scala: implicit val connectionsCell: ConnectionsCell[Task] = Cell.unsafe[Task, Connections](List(local))
        let connections = Connections::from_vec(vec![local.clone()]);
        let connections_cell = ConnectionsCell {
            peers: Arc::new(Mutex::new(connections.clone())),
        };
        let connections_cell_for_retriever = ConnectionsCell {
            peers: Arc::new(Mutex::new(connections)),
        };

        // Scala: implicit val blockProcessingState = Ref.of[Task, Set[BlockHash]](Set.empty)
        let blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>> =
            Arc::new(Mutex::new(HashSet::new()));

        // NOT in Scala Setup - created locally in each test as: implicit val eventBus = EventPublisher.noop[Task]
        // Rust: Create F1r3flyEvents with default capacity (equivalent to noop for tests)
        let event_publisher = Arc::new(F1r3flyEvents::default());

        // TODO NOT in Scala Setup - created locally in each test as: implicit val engineCell = Cell.unsafe[Task, Engine[Task]](Engine.noop)
        // Rust: Create EngineCell with Engine::noop (equivalent to Scala)
        let engine_cell = Arc::new(EngineCell::init());

        let block_retriever = Arc::new(block_retriever::BlockRetriever::new(
            transport_layer.clone(),
            Arc::new(connections_cell_for_retriever),
            Arc::new(rp_conf.clone()),
        ));

        // NOTE: Cast Arc<NoOpsCasperEffect> to Arc<dyn MultiParentCasper + Send + Sync>
        let casper_trait_object: Arc<dyn MultiParentCasper + Send + Sync> =
            Arc::new(casper.clone());

        let engine = Running::new(
            block_processing_queue.clone(),
            Arc::new(Mutex::new(Default::default())),
            casper_trait_object,
            approved_block,
            Arc::new(|| {
                Box::pin(async { Ok(()) })
                    as Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>>
            }),
            false,
            connections_cell.clone(),
            transport_layer.clone(),
            rp_conf.clone(),
            block_retriever.clone(),
        );

        Self {
            transport_layer,
            local,
            network_id,
            casper,
            engine,
            block_processing_queue,
            rspace_state_manager: Arc::new(Mutex::new(Some(rspace_state_manager_unwrapped))),
            runtime_manager: runtime_manager_shared,
            estimator: Arc::new(Mutex::new(Some(estimator_unwrapped))),
            rspace_store,
            block_store: Arc::new(Mutex::new(Some(block_store_unwrapped))),
            last_approved_block,
            casper_shard_conf,
            genesis,
            validator_sk,
            validator_pk,
            approved_block_candidate,
            exporter_params,
            required_sigs,
            genesis_parameters: genesis_params,
            validator_id,
            bap,
            blocks_in_processing,
            rp_conf_ask: rp_conf,
            connections_cell,
            event_publisher,
            block_retriever,
            engine_cell,
            block_dag_storage: Arc::new(Mutex::new(Some(block_dag_storage_unwrapped))),
            deploy_storage: Arc::new(Mutex::new(Some(deploy_storage_unwrapped))),
            casper_buffer_storage: Arc::new(Mutex::new(Some(casper_buffer_storage_unwrapped))),
        }
    }

    /// Get the current length of the block processing queue (for testing)
    #[allow(dead_code)]
    pub fn block_processing_queue_len(&self) -> usize {
        match self.block_processing_queue.lock() {
            Ok(queue) => queue.len(),
            Err(_) => 0,
        }
    }

    /// Check if a block with the given hash is in the processing queue (for testing)
    pub fn is_block_in_processing_queue(&self, hash: &BlockHash) -> bool {
        match self.block_processing_queue.lock() {
            Ok(queue) => queue.iter().any(|(_, block)| &block.block_hash == hash),
            Err(_) => false,
        }
    }
}

pub fn to_casper_message(p: Protocol) -> CasperMessage {
    if let Some(packet) = p.message {
        if let models::routing::protocol::Message::Packet(packet_data) = packet {
            // This is a simplified stand-in for the full conversion logic,
            // which would involve looking at the typeId of the packet.
            // For these tests, we can make assumptions about the message type.
            if let Ok(bm) = models::casper::BlockMessageProto::decode(packet_data.content.as_ref())
            {
                if let Ok(block_message) = BlockMessage::from_proto(bm) {
                    return CasperMessage::BlockMessage(block_message);
                }
            }
            if let Ok(ab) = models::casper::ApprovedBlockProto::decode(packet_data.content.as_ref())
            {
                if let Ok(approved_block) = ApprovedBlock::from_proto(ab) {
                    return CasperMessage::ApprovedBlock(approved_block);
                }
            }
            if let Ok(hb) = models::casper::HasBlockProto::decode(packet_data.content.as_ref()) {
                let has_block = HasBlock::from_proto(hb);
                return CasperMessage::HasBlock(has_block);
            }
        }
    }
    panic!("Could not convert protocol to casper message");
}

/// Exporter parameters for RSpaceExporterItems
#[derive(Clone, Debug)]
pub struct ExporterParams {
    pub skip: i32,
    pub take: i32,
}

fn endpoint(port: u32) -> Endpoint {
    Endpoint {
        host: "host".to_string(),
        tcp_port: port,
        udp_port: port,
    }
}

pub fn peer_node(name: &str, port: u32) -> PeerNode {
    PeerNode {
        id: NodeIdentifier {
            key: Bytes::from(name.as_bytes().to_vec()),
        },
        endpoint: endpoint(port),
    }
}
