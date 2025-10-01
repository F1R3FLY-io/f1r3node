// See casper/src/test/scala/coop/rchain/casper/engine/Setup.scala

use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::{
        block_dag_key_value_storage::{DeployId, KeyValueDagRepresentation},
        block_metadata_store::BlockMetadataStore,
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
    },
};
use prost::bytes::Bytes;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;
use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use crate::util::rholang::resources::mk_test_rnode_store_manager;
use crate::{
    helper::no_ops_casper_effect::NoOpsCasperEffect,
    util::{
        genesis_builder::GenesisBuilder, rholang::resources::mk_runtime_manager,
        test_mocks::MockKeyValueStore,
    },
};
use casper::rust::casper::CasperShardConf;
use casper::rust::estimator::Estimator;
use casper::rust::genesis::genesis::Genesis;
use casper::rust::util::rholang::runtime_manager::RuntimeManager;
use crypto::rust::signatures::signed::Signed;
use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use models::rust::casper::protocol::casper_message::DeployData;
use prost::Message;
use rspace_plus_plus::rspace::r#match::Match;
use rspace_plus_plus::rspace::rspace::RSpace;
use rspace_plus_plus::rspace::state::instances::rspace_exporter_store::RSpaceExporterImpl;
use rspace_plus_plus::rspace::state::instances::rspace_importer_store::RSpaceImporterImpl;
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
    pub engine: Running<NoOpsCasperEffect, TransportLayerStub>,
    // Scala: implicit val blockProcessingQueue = Queue.unbounded[Task, (Casper[Task], BlockMessage)]
    pub block_processing_queue: Arc<Mutex<VecDeque<(Arc<NoOpsCasperEffect>, BlockMessage)>>>,
    // Scala Step 4: implicit val rspaceStateManager = RSpacePlusPlusStateManagerImpl(exporter, importer)
    pub rspace_state_manager: RSpaceStateManager,
    // Scala: implicit val runtimeManager = RuntimeManager[Task](rspace, replay, historyRepo, mStore, Genesis.NonNegativeMergeableTagName)
    // Note: Stored as Arc because NoOpsCasperEffect takes ownership and wraps in Arc
    pub runtime_manager: Arc<RuntimeManager>,
    // Scala: implicit val estimator = Estimator[Task](Estimator.UnlimitedParents, None)
    pub estimator: Estimator,
    pub rspace_store: rspace_plus_plus::rspace::rspace::RSpaceStore,
    // Scala: implicit val blockStore = KeyValueBlockStore[Task](kvm).unsafeRunSync(...)
    pub block_store: KeyValueBlockStore,
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
    pub block_dag_storage: KeyValueDagRepresentation,
    // Scala: implicit val deployStorage = KeyValueDeployStorage[Task](kvm).unsafeRunSync(...)
    pub deploy_storage: KeyValueDeployStorage,
    // Scala: implicit val casperBuffer = CasperBufferKeyValueStorage.create[Task](spaceKVManager).unsafeRunSync(...)
    pub casper_buffer_storage: CasperBufferKeyValueStorage,
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
        let mut space_kv_manager = mk_test_rnode_store_manager(context.storage_directory.clone());

        // Scala Step 1: val spaces = RSpacePlusPlus_RhoTypes.createWithReplay[Task, ...](context.storageDirectory.toString())
        // where spaces = createWithReplay[Task, Par, BindPattern, ListParWithRandom, TaggedContinuation](...)
        let rspace_store =
            rspace_plus_plus::rspace::shared::rspace_store_manager::get_or_create_rspace_store(
                context
                    .storage_directory
                    .to_str()
                    .expect("Invalid storage directory path"),
                1024 * 1024 * 100, // 100MB map size
            )
            .expect("Failed to create RSpace store");

        // Create matcher for RSpace
        let matcher: Arc<Box<dyn Match<BindPattern, ListParWithRandom>>> =
            Arc::new(Box::new(DummyMatcher));

        // Scala Step 2: val (rspace, replay) = spaces
        let (rspace, replay) =
            RSpace::<Par, BindPattern, ListParWithRandom, TaggedContinuation>::create_with_replay(
                rspace_store.clone(),
                matcher,
            )
            .expect("Failed to create RSpace with replay");

        // Scala: val historyRepo = rspace.historyRepo
        let history_repo = rspace.history_repository.clone();

        //TODO  ask Steven about it (зараз код не відповідає Scala коду де ми діставали exported/importer з historyRepo,
        //TODO бо у нас повертаються трейт об*єкти, а не конкретні типи, тому я створюю руками конкретні типи)

        // Scala Step 3: val (exporter, importer) = { (historyRepo.exporter.unsafeRunSync, historyRepo.importer.unsafeRunSync) }
        // Note: In Rust we get trait objects from history_repo, but they're not used for RSpaceStateManager
        let _exporter_trait = history_repo.exporter();
        let _importer_trait = history_repo.importer();

        // Scala Step 4: implicit val rspaceStateManager = RSpacePlusPlusStateManagerImpl(exporter, importer)
        // In Rust, RSpaceStateManager needs concrete types RSpaceExporterImpl/RSpaceImporterImpl
        let exporter_concrete = RSpaceExporterImpl {
            source_history_store: rspace_store.history.clone(),
            source_value_store: rspace_store.cold.clone(),
            source_roots_store: rspace_store.roots.clone(),
        };
        let importer_concrete = RSpaceImporterImpl {
            history_store: rspace_store.history.clone(),
            value_store: rspace_store.cold.clone(),
            roots_store: rspace_store.roots.clone(),
        };
        let rspace_state_manager = RSpaceStateManager::new(exporter_concrete, importer_concrete);

        // Scala: val mStore = RuntimeManager.mergeableStore(spaceKVManager).unsafeRunSync(scheduler)
        let m_store = RuntimeManager::mergeable_store(&mut space_kv_manager)
            .await
            .expect("Failed to create mergeable store");

        // Scala: implicit val runtimeManager = RuntimeManager[Task](rspace, replay, historyRepo, mStore, Genesis.NonNegativeMergeableTagName)
        let runtime_manager = RuntimeManager::create_with_space(
            rspace,
            replay,
            history_repo,
            m_store,
            Genesis::non_negative_mergeable_tag_name(),
        );

        // Scala: val kvm = InMemoryStoreManager[Task]()
        // In Scala, InMemoryStoreManager creates separate stores for each name via kvm.store("name")
        // We simulate this by creating separate shared HashMaps for each "database name"
        let kvm_blockstorage = Arc::new(Mutex::new(HashMap::new()));
        let kvm_approved_block = Arc::new(Mutex::new(HashMap::new()));
        let kvm_dagstorage_metadata = Arc::new(Mutex::new(HashMap::new()));
        let kvm_dagstorage_deploy_index = Arc::new(Mutex::new(HashMap::new()));
        let kvm_deploystorage = Arc::new(Mutex::new(HashMap::new()));

        // Scala: implicit val blockStore = KeyValueBlockStore[Task](kvm).unsafeRunSync(...)
        // Each storage gets its own "database" from kvm, equivalent to kvm.store("blockstorage")
        let store = Box::new(MockKeyValueStore::with_shared_data(kvm_blockstorage.clone()));
        let store_approved_block = Box::new(MockKeyValueStore::with_shared_data(kvm_approved_block.clone()));
        let mut block_store = KeyValueBlockStore::new(store, store_approved_block);
        block_store
            .put(genesis.block_hash.clone(), &genesis)
            .expect("Failed to store genesis block");

        // Scala: implicit val blockDagStorage = BlockDagKeyValueStorage.create(kvm).unsafeRunSync(...)
        // Equivalent to kvm.store("dagstorage-metadata") and kvm.store("dagstorage-deploy")
        let metadata_store = Box::new(MockKeyValueStore::with_shared_data(kvm_dagstorage_metadata.clone()));
        let metadata_typed_store =
            KeyValueTypedStoreImpl::<BlockHashSerde, BlockMetadata>::new(metadata_store);
        let block_metadata_store = BlockMetadataStore::new(metadata_typed_store);

        let deploy_store = Box::new(MockKeyValueStore::with_shared_data(kvm_dagstorage_deploy_index.clone()));
        let deploy_typed_store =
            KeyValueTypedStoreImpl::<DeployId, BlockHashSerde>::new(deploy_store);

        let block_dag_storage = KeyValueDagRepresentation {
            dag_set: Default::default(),
            latest_messages_map: Default::default(),
            child_map: Default::default(),
            height_map: Default::default(),
            invalid_blocks_set: Default::default(),
            last_finalized_block_hash: genesis.block_hash.clone(),
            finalized_blocks_set: Default::default(),
            block_metadata_index: Arc::new(std::sync::RwLock::new(block_metadata_store)),
            deploy_index: Arc::new(std::sync::RwLock::new(deploy_typed_store)),
        };

        block_dag_storage.dag_set.insert(genesis.block_hash.clone());
        block_dag_storage
            .finalized_blocks_set
            .insert(genesis.block_hash.clone());

        // Scala: implicit val deployStorage = KeyValueDeployStorage[Task](kvm).unsafeRunSync(...)
        // Equivalent to kvm.store("deploystorage")
        let deploy_storage_store = Box::new(MockKeyValueStore::with_shared_data(kvm_deploystorage.clone()));
        let deploy_storage_typed_store =
            KeyValueTypedStoreImpl::<ByteString, Signed<DeployData>>::new(deploy_storage_store);
        let deploy_storage = KeyValueDeployStorage {
            store: deploy_storage_typed_store,
        };

        // Scala: implicit val estimator = Estimator[Task](Estimator.UnlimitedParents, None)
        let estimator = Estimator::apply(Estimator::UNLIMITED_PARENTS, None);

        // Create NoOpsCasperEffect with comprehensive dependencies from genesis context
        // NoOpsCasperEffect will use the same kvm_blockstorage for its internal block store
        // This ensures consistency with the external block_store
        let mut casper = NoOpsCasperEffect::new_with_shared_kvm(
            None, // estimator_func
            runtime_manager,
            block_store.clone(),
            block_dag_storage.clone(),
            kvm_blockstorage.clone(),
        );

        // Add the genesis block using the new test-friendly methods
        casper.add_block_to_store(genesis.clone());
        casper.add_to_dag(genesis.block_hash.clone());

        let block_processing_queue: Arc<Mutex<VecDeque<(Arc<NoOpsCasperEffect>, BlockMessage)>>> =
            Arc::new(Mutex::new(VecDeque::new()));

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
        let casper_buffer_storage =
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
        let engine_cell = Arc::new(EngineCell::unsafe_init().expect("Failed to create EngineCell"));

        let block_retriever = Arc::new(block_retriever::BlockRetriever::new(
            transport_layer.clone(),
            Arc::new(connections_cell_for_retriever),
            Arc::new(rp_conf.clone()),
        ));

        let engine = Running::new(
            block_processing_queue.clone(),
            Arc::new(Mutex::new(Default::default())),
            Arc::new(casper.clone()),
            approved_block,
            Arc::new(|| Ok(())),
            false,
            connections_cell.clone(),
            transport_layer.clone(),
            rp_conf.clone(),
            block_retriever.clone(),
        );

        // Get runtime_manager Arc from casper (it was transferred to NoOpsCasperEffect)
        let runtime_manager_arc = casper.runtime_manager.clone();

        Self {
            transport_layer,
            local,
            network_id,
            casper,
            engine,
            block_processing_queue,
            rspace_state_manager,
            runtime_manager: runtime_manager_arc,
            estimator,
            rspace_store,
            block_store,
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
            block_dag_storage,
            deploy_storage,
            casper_buffer_storage,
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

/// Dummy matcher for tests - spatial matching is handled by RSpace++ internally
struct DummyMatcher;

impl Match<BindPattern, ListParWithRandom> for DummyMatcher {
  fn get(&self, _pattern: BindPattern, data: ListParWithRandom) -> Option<ListParWithRandom> {
    // For tests, we just return the data as-is (always matches)
    // Real implementation would use spatial matching via rholang
    Some(data)
  }
}
