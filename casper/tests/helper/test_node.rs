// See casper/src/test/scala/coop/rchain/casper/helper/TestNode.scala

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex, RwLock},
};
use tokio::sync::{mpsc, oneshot};

use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    key_value_block_store::KeyValueBlockStore,
};
use casper::rust::{
    block_status::BlockStatus,
    blocks::{
        block_processor::{BlockProcessor, BlockProcessorDependencies},
        proposer::proposer::{new_proposer, ProductionProposer, ProposerResult},
    },
    casper::{CasperShardConf, MultiParentCasper},
    engine::block_retriever::{BlockRetriever, RequestState, RequestedBlocks},
    errors::CasperError,
    estimator::Estimator,
    genesis::genesis::Genesis,
    multi_parent_casper_impl::MultiParentCasperImpl,
    safety_oracle::{CliqueOracleImpl, SafetyOracle},
    util::rholang::runtime_manager::RuntimeManager,
    validator_identity::ValidatorIdentity,
    ValidBlockProcessing,
};
use comm::rust::{
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    test_instances::create_rp_conf_ask,
};
use crypto::rust::private_key::PrivateKey;
use models::rust::{
    block_hash::BlockHash,
    casper::protocol::casper_message::{ApprovedBlock, ApprovedBlockCandidate, BlockMessage},
};
use rholang::rust::interpreter::rho_runtime::RhoHistoryRepository;
use rspace_plus_plus::rspace::history::Either;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;

use crate::util::{
    comm::transport_layer_test_impl::{
        test_network::TestNetwork, TransportLayerServerTestImpl, TransportLayerTestImpl,
    },
    genesis_builder::GenesisContext,
    rholang::resources,
};

pub struct TestNode {
    pub name: String,
    pub local: PeerNode,
    pub tle: Arc<TransportLayerTestImpl>,
    pub tls: TransportLayerServerTestImpl,
    pub genesis: BlockMessage,
    pub validator_id_opt: Option<ValidatorIdentity>,
    // TODO: pub logical_time: LogicalTime,
    pub synchrony_constraint_threshold: f64,
    pub data_dir: PathBuf,
    pub max_number_of_parents: i32,
    pub max_parent_depth: Option<i32>,
    pub shard_id: String,
    pub finalization_rate: i32,
    pub is_read_only: bool,
    // Note: trigger_propose_f_opt is implemented as method trigger_propose
    pub proposer_opt: Option<ProductionProposer<TransportLayerTestImpl>>,
    pub block_processor_queue: (
        mpsc::UnboundedSender<(Arc<dyn MultiParentCasper>, BlockMessage)>,
        Arc<Mutex<mpsc::UnboundedReceiver<(Arc<dyn MultiParentCasper>, BlockMessage)>>>,
    ),
    pub block_processor_state: Arc<RwLock<HashSet<BlockHash>>>,
    // Note: blockProcessingPipe implemented as method process_block_through_pipe
    pub block_processor: BlockProcessor<TransportLayerTestImpl>,
    pub block_store: Arc<Mutex<KeyValueBlockStore>>,
    pub block_dag_storage: Arc<Mutex<BlockDagKeyValueStorage>>,
    pub deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    // Note: Removed comm_util field, will use transport_layer directly
    pub block_retriever: BlockRetriever<TransportLayerTestImpl>,
    // TODO: pub metrics: Metrics,
    // TODO: pub span: Span,
    pub casper_buffer_storage: Arc<Mutex<CasperBufferKeyValueStorage>>,
    pub runtime_manager: Arc<Mutex<RuntimeManager>>,
    pub rho_history_repository: RhoHistoryRepository,
    // Note: no log field, logging will come from log crate
    pub requested_blocks: RequestedBlocks,
    // Note: no need for SynchronyConstraintChecker struct, will use 'check' method directly
    // Note: no need for LastFinalizedHeightConstraintChecker struct, will use 'check' method directly
    pub estimator: Estimator,
    pub safety_oracle: Box<dyn SafetyOracle>,
    // TODO: pub time: Time,
    // Note: no need for duplicate transport_layer field, will use tls field directly
    pub connections_cell: ConnectionsCell,
    pub rp_conf: RPConf,
    pub event_publisher: F1r3flyEvents,
}

impl TestNode {
    fn create_casper(&self) -> MultiParentCasperImpl<TransportLayerTestImpl> {
        let approved_block = ApprovedBlock {
            candidate: ApprovedBlockCandidate {
                block: self.genesis.clone(),
                required_sigs: 0,
            },
            sigs: vec![],
        };

        let shard_conf = CasperShardConf {
            fault_tolerance_threshold: 0.0,
            shard_name: self.shard_id.clone(),
            parent_shard_id: "".to_string(),
            finalization_rate: self.finalization_rate,
            max_number_of_parents: self.max_number_of_parents,
            max_parent_depth: self.max_parent_depth.unwrap_or(i32::MAX),
            synchrony_constraint_threshold: self.synchrony_constraint_threshold as f32,
            height_constraint_threshold: i64::MAX,
            // Validators will try to put deploy in a block only for next `deployLifespan` blocks.
            // Required to enable protection from re-submitting duplicate deploys
            deploy_lifespan: 50,
            casper_version: 1,
            config_version: 1,
            bond_minimum: 0,
            bond_maximum: i64::MAX,
            epoch_length: 10000,
            quarantine_length: 20000,
            min_phlo_price: 1,
        };

        MultiParentCasperImpl {
            block_retriever: self.block_retriever.clone(),
            event_publisher: self.event_publisher.clone(),
            runtime_manager: self.runtime_manager.clone(),
            estimator: self.estimator.clone(),
            block_store: self.block_store.clone(),
            block_dag_storage: self.block_dag_storage.clone(),
            deploy_storage: self.deploy_storage.clone(),
            casper_buffer_storage: self.casper_buffer_storage.clone(),
            validator_id: self.validator_id_opt.clone(),
            casper_shard_conf: shard_conf,
            approved_block: self.genesis.clone(),
        }
    }

    pub async fn trigger_propose<C: MultiParentCasper>(
        &mut self,
        casper: &mut C,
    ) -> Result<BlockHash, CasperError> {
        match &mut self.proposer_opt {
            Some(proposer) => {
                let (sender, receiver) = oneshot::channel();
                let _result = proposer.propose(casper, false, sender).await?;
                let proposer_result = receiver
                    .await
                    .map_err(|_| CasperError::RuntimeError("Channel closed".to_string()))?;

                match proposer_result {
                    ProposerResult::Success(_, block) => Ok(block.block_hash),
                    _ => Err(CasperError::RuntimeError(
                        "Propose failed or another in progress".to_string(),
                    )),
                }
            }
            None => Err(CasperError::RuntimeError(
                "Propose is called in read-only mode".to_string(),
            )),
        }
    }

    /// Processes a block through the validation pipeline
    pub async fn process_block_through_pipe<C: MultiParentCasper>(
        &mut self,
        casper: &mut C,
        block: BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError> {
        // Check if block is of interest
        let is_of_interest = self
            .block_processor
            .check_if_of_interest(casper, &block)
            .await?;

        if !is_of_interest {
            return Ok(Either::Left(BlockStatus::not_of_interest()));
        }

        // Check if well-formed and store
        let is_well_formed = self
            .block_processor
            .check_if_well_formed_and_store(&block)
            .await?;

        if !is_well_formed {
            return Ok(Either::Left(BlockStatus::invalid_format()));
        }

        // Check dependencies
        let dependencies_ready = self
            .block_processor
            .check_dependencies_with_effects(casper, &block)
            .await?;

        if !dependencies_ready {
            return Ok(Either::Left(BlockStatus::missing_blocks()));
        }

        // Validate with effects
        self.block_processor
            .validate_with_effects(casper, &block, None)
            .await
    }

    /// Creates a standalone TestNode (single node network)
    pub async fn standalone(genesis: GenesisContext) -> Result<TestNode, CasperError> {
        let nodes = Self::create_network(genesis, 1, None, None, None, None).await?;

        Ok(nodes.into_iter().next().unwrap())
    }

    /// Creates a network of TestNodes
    pub async fn create_network(
        genesis: GenesisContext,
        network_size: usize,
        synchrony_constraint_threshold: Option<f64>,
        max_number_of_parents: Option<i32>,
        max_parent_depth: Option<i32>,
        with_read_only_size: Option<usize>,
    ) -> Result<Vec<TestNode>, CasperError> {
        let test_network = TestNetwork::empty();

        // Take the required number of validator keys
        let sks_to_use: Vec<PrivateKey> = genesis
            .validator_sks()
            .into_iter()
            .take(network_size + with_read_only_size.unwrap_or(0))
            .collect();

        Self::network(
            sks_to_use,
            genesis.genesis_block,
            genesis.storage_directory,
            synchrony_constraint_threshold.unwrap_or(0.0),
            max_number_of_parents.unwrap_or(Estimator::UNLIMITED_PARENTS),
            max_parent_depth,
            with_read_only_size.unwrap_or(0),
            test_network,
        )
        .await
    }

    /// Creates a network of TestNodes
    async fn network(
        sks: Vec<PrivateKey>,
        genesis: BlockMessage,
        storage_matrix_path: PathBuf,
        synchrony_constraint_threshold: f64,
        max_number_of_parents: i32,
        max_parent_depth: Option<i32>,
        with_read_only_size: usize,
        test_network: TestNetwork,
    ) -> Result<Vec<TestNode>, CasperError> {
        let n = sks.len();

        // Generate node names: "node-1", "node-2", ..., "readOnly-{i}" for read-only nodes
        let names: Vec<String> = (1..=n)
            .map(|i| {
                if i <= (n - with_read_only_size) {
                    format!("node-{}", i)
                } else {
                    format!("readOnly-{}", i)
                }
            })
            .collect();

        // Generate is_read_only flags
        let is_read_only: Vec<bool> = (1..=n).map(|i| i > (n - with_read_only_size)).collect();

        // Generate peers using port 40400
        let peers: Vec<PeerNode> = names
            .iter()
            .map(|name| Self::peer_node(name, 40400))
            .collect();

        // Create nodes
        let mut nodes = Vec::new();
        for (((name, peer), sk), is_readonly) in names
            .into_iter()
            .zip(peers.into_iter())
            .zip(sks.into_iter())
            .zip(is_read_only.into_iter())
        {
            let node = Self::create_node(
                name,
                peer,
                genesis.clone(),
                sk,
                storage_matrix_path.clone(),
                synchrony_constraint_threshold,
                max_number_of_parents,
                max_parent_depth,
                is_readonly,
                test_network.clone(),
            )
            .await;
            nodes.push(node);
        }

        // Set up connections between all nodes
        for node_a in &nodes {
            for node_b in &nodes {
                if node_a.local != node_b.local {
                    // Add connection from node_a to node_b
                    node_a
                        .connections_cell
                        .flat_modify(|connections| connections.add_conn(node_b.local.clone()))
                        .map_err(|e| {
                            CasperError::RuntimeError(format!("Connection setup failed: {}", e))
                        })?;
                }
            }
        }

        Ok(nodes)
    }

    async fn create_node(
        name: String,
        current_peer_node: PeerNode,
        genesis: BlockMessage,
        sk: PrivateKey,
        storage_dir: PathBuf,
        // TODO: logical_time: LogicalTime,
        synchrony_constraint_threshold: f64,
        max_number_of_parents: i32,
        max_parent_depth: Option<i32>,
        is_read_only: bool,
        test_network: TestNetwork,
    ) -> TestNode {
        let tle = Arc::new(TransportLayerTestImpl::new(test_network.clone()));
        let tls =
            TransportLayerServerTestImpl::new(current_peer_node.clone(), test_network.clone());

        let new_storage_dir = resources::copy_storage(storage_dir);
        let mut kvm = resources::mk_test_rnode_store_manager(new_storage_dir.clone());

        let block_store_base = KeyValueBlockStore::create_from_kvm(&mut kvm).await.unwrap();
        let block_store = Arc::new(Mutex::new(block_store_base));

        let block_dag_storage = Arc::new(Mutex::new(
            BlockDagKeyValueStorage::new(&mut kvm).await.unwrap(),
        ));

        let deploy_storage = Arc::new(Mutex::new(
            KeyValueDeployStorage::new(&mut kvm).await.unwrap(),
        ));

        let casper_buffer_storage = Arc::new(Mutex::new(
            CasperBufferKeyValueStorage::new_from_kvm(&mut kvm)
                .await
                .unwrap(),
        ));

        let rspace_store = kvm.r_space_stores().await.unwrap();
        let mergeable_store = RuntimeManager::mergeable_store(&mut kvm).await.unwrap();
        let runtime_manager = Arc::new(Mutex::new(RuntimeManager::create_with_store(
            rspace_store,
            mergeable_store,
            Genesis::non_negative_mergeable_tag_name(),
        )));

        let runtime_manager_guard = runtime_manager.lock().unwrap();
        let rho_history_repository = runtime_manager_guard.get_history_repo();
        drop(runtime_manager_guard);

        let connections_cell = ConnectionsCell::new();
        let clique_oracle = CliqueOracleImpl;
        let estimator = Estimator::apply(max_number_of_parents, max_parent_depth);
        let rp_conf = create_rp_conf_ask(current_peer_node.clone(), None, None);
        let event_publisher = F1r3flyEvents::new(None);
        let requested_blocks = Arc::new(RwLock::new(HashMap::<BlockHash, RequestState>::new()));
        let block_retriever =
            BlockRetriever::new(tle.clone(), connections_cell.clone(), rp_conf.clone());

        let _ = test_network.add_peer(&current_peer_node);

        // Proposer
        let validator_id_opt = if is_read_only {
            None
        } else {
            Some(ValidatorIdentity::new(&sk))
        };

        let proposer_opt = match validator_id_opt {
            Some(ref vi) => Some(new_proposer(
                vi.clone(),
                None,
                runtime_manager.clone(),
                block_store.clone(),
                deploy_storage,
                block_retriever.clone(),
                tle.clone(),
                connections_cell.clone(),
                rp_conf.clone(),
                event_publisher.clone(),
            )),
            None => None,
        };

        let bp_dependencies = BlockProcessorDependencies::new(
            block_store.clone(),
            casper_buffer_storage.clone(),
            block_dag_storage.clone(),
            block_retriever.clone(),
            tle.clone(),
            connections_cell.clone(),
            rp_conf.clone(),
        );

        let block_processor = BlockProcessor::new(bp_dependencies);

        // Creates an unbounded tokio channel for processing (Casper, BlockMessage) tuples
        // - Sender: Non-blocking, cloneable, used to enqueue blocks for processing
        // - Receiver: Thread-safe (Arc<Mutex>), used to dequeue blocks from processing pipeline
        let (block_processor_queue_tx, block_processor_queue_rx) =
            mpsc::unbounded_channel::<(Arc<dyn MultiParentCasper>, BlockMessage)>();
        let block_processor_queue = (
            block_processor_queue_tx,
            Arc::new(Mutex::new(block_processor_queue_rx)),
        );

        let block_processor_state = Arc::new(RwLock::new(HashSet::<BlockHash>::new()));

        TestNode {
            name,
            local: current_peer_node,
            tle,
            tls,
            genesis,
            validator_id_opt,
            synchrony_constraint_threshold,
            data_dir: new_storage_dir,
            max_number_of_parents: max_number_of_parents,
            max_parent_depth,
            shard_id: "root".to_string(),
            finalization_rate: 1,
            is_read_only: is_read_only,
            proposer_opt,
            block_processor_queue,
            block_processor_state,
            block_processor,
            block_store,
            block_dag_storage,
            deploy_storage,
            block_retriever,
            casper_buffer_storage,
            runtime_manager,
            rho_history_repository,
            requested_blocks,
            estimator,
            safety_oracle: Box::new(clique_oracle),
            connections_cell,
            rp_conf,
            event_publisher,
        }
    }

    /// Creates a PeerNode with the given name and port
    fn peer_node(name: &str, port: u32) -> PeerNode {
        // Convert name bytes to hex string for NodeIdentifier
        let name_hex = hex::encode(name.as_bytes());
        let node_id = NodeIdentifier::new(name_hex);
        let endpoint = Self::endpoint(port);

        PeerNode {
            id: node_id,
            endpoint,
        }
    }

    /// Creates an endpoint with the given port for both TCP and UDP
    fn endpoint(port: u32) -> Endpoint {
        Endpoint::new("host".to_string(), port, port)
    }
}
