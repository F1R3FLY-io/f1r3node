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
        proposer::{
            block_creator,
            propose_result::BlockCreatorResult,
            proposer::{new_proposer, ProductionProposer, ProposerResult},
        },
    },
    casper::{Casper, CasperShardConf, MultiParentCasper},
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
    errors::CommError,
    p2p::packet_handler::{NOPPacketHandler, PacketHandler},
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
    rp::{connect::ConnectionsCell, handle_messages, rp_conf::RPConf},
    test_instances::create_rp_conf_ask,
    transport::{
        communication_response::CommunicationResponse, grpc_transport_server::TransportLayerServer,
        transport_layer::Blob,
    },
};
use crypto::rust::{private_key::PrivateKey, signatures::signed::Signed};
use models::{
    routing::Protocol,
    rust::{
        block_hash::BlockHash,
        casper::protocol::casper_message::{
            ApprovedBlock, ApprovedBlockCandidate, BlockMessage, DeployData,
        },
    },
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

use casper::rust::{
    engine::{engine_cell::EngineCell, engine_with_casper::EngineWithCasper},
    util::comm::casper_packet_handler::CasperPacketHandler,
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
    pub block_store: KeyValueBlockStore,
    pub block_dag_storage: BlockDagKeyValueStorage,
    pub deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    // Note: Removed comm_util field, will use transport_layer directly
    pub block_retriever: BlockRetriever<TransportLayerTestImpl>,
    // TODO: pub metrics: Metrics,
    // TODO: pub span: Span,
    pub casper_buffer_storage: CasperBufferKeyValueStorage,
    pub runtime_manager: RuntimeManager,
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
    // Casper instance (Arc for shared ownership)
    pub casper: Arc<MultiParentCasperImpl<TransportLayerTestImpl>>,
    // Engine cell for packet handling (matches Scala line 177)
    pub engine_cell: EngineCell,
    // Packet handler for receiving messages (matches Scala line 178)
    pub packet_handler: CasperPacketHandler,
}

impl TestNode {
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

    /// Creates a block with the given deploys (equivalent to Scala createBlock, line 233-239).
    ///
    /// This method:
    /// 1. Deploys each datum to casper
    /// 2. Gets a snapshot from casper
    /// 3. Gets validator identity
    /// 4. Calls BlockCreator.create to produce the block
    ///
    /// Returns BlockCreatorResult which may be Created, NoNewDeploys, or ReadOnlyMode.
    pub async fn create_block(
        &mut self,
        deploy_datums: &[Signed<DeployData>],
    ) -> Result<BlockCreatorResult, CasperError> {
        // Get mutable access to casper
        let casper_ref = Arc::get_mut(&mut self.casper).ok_or_else(|| {
            CasperError::RuntimeError(
                "Cannot get mutable casper reference - Arc has multiple owners".to_string(),
            )
        })?;

        // Deploy all datums
        for deploy_datum in deploy_datums {
            casper_ref.deploy(deploy_datum.clone())?;
        }

        // Get snapshot
        let snapshot = casper_ref.get_snapshot().await?;

        // Get validator
        let validator = casper_ref.get_validator().ok_or_else(|| {
            CasperError::RuntimeError("No validator identity available".to_string())
        })?;

        // Create block using block_creator
        block_creator::create(
            &snapshot,
            &validator,
            None, // dummy_deploy_opt
            self.deploy_storage.clone(),
            &mut self.runtime_manager.clone(),
            &mut self.block_store.clone(),
        )
        .await
    }

    /// Creates a block with the given deploys, assuming success (equivalent to Scala createBlockUnsafe, line 242-255).
    ///
    /// Unlike create_block, this method:
    /// - Returns the BlockMessage directly (not BlockCreatorResult)
    /// - Errors if block creation fails for any reason
    ///
    /// This is useful for tests that expect block creation to succeed.
    pub async fn create_block_unsafe(
        &mut self,
        deploy_datums: &[Signed<DeployData>],
    ) -> Result<BlockMessage, CasperError> {
        let result = self.create_block(deploy_datums).await?;

        match result {
            BlockCreatorResult::Created(block) => Ok(block),
            _ => Err(CasperError::RuntimeError(format!(
                "Failed creating block: {:?}",
                result
            ))),
        }
    }

    /// Processes a block through the validation pipeline (equivalent to Scala processBlock, line 257-260).
    ///
    /// This is the wrapper method that processes an existing block through the full validation pipeline.
    pub async fn process_block(
        &mut self,
        block: BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError> {
        self.process_block_through_pipe(block).await
    }

    /// Processes a block through the validation pipeline (internal implementation).
    ///
    /// This method:
    /// 1. Checks if block is of interest
    /// 2. Checks if well-formed and stores
    /// 3. Checks dependencies
    /// 4. Validates with effects
    pub async fn process_block_through_pipe(
        &mut self,
        block: BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError> {
        // Get mutable access to casper through Arc
        // Note: In tests, we typically have exclusive access to TestNode, so this is safe
        let casper_ref = Arc::get_mut(&mut self.casper).ok_or_else(|| {
            CasperError::RuntimeError(
                "Cannot get mutable casper reference - Arc has multiple owners".to_string(),
            )
        })?;

        // Check if block is of interest
        let is_of_interest = self
            .block_processor
            .check_if_of_interest(casper_ref, &block)
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
            .check_dependencies_with_effects(casper_ref, &block)
            .await?;

        if !dependencies_ready {
            return Ok(Either::Left(BlockStatus::missing_blocks()));
        }

        // Validate with effects
        self.block_processor
            .validate_with_effects(casper_ref, &block, None)
            .await
    }

    /// Adds and processes a block (equivalent to Scala addBlock(block), line 198-199).
    ///
    /// Takes an existing block and processes it through the validation pipeline.
    pub async fn add_block(
        &mut self,
        block: BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError> {
        self.process_block_through_pipe(block).await
    }

    /// Creates and adds a block from deploys (equivalent to Scala addBlock(deploys), line 201-202).
    ///
    /// This is a convenience method that:
    /// 1. Creates a block from the given deploys
    /// 2. Processes it through the validation pipeline
    /// 3. Returns the block (assuming Valid status)
    pub async fn add_block_from_deploys(
        &mut self,
        deploy_datums: &[Signed<DeployData>],
    ) -> Result<BlockMessage, CasperError> {
        self.add_block_status(deploy_datums, |status| matches!(status, Either::Right(_)))
            .await
    }

    /// Creates and adds a block with expected status validation (equivalent to Scala addBlockStatus, line 223-231).
    ///
    /// This method:
    /// 1. Creates a block from deploys
    /// 2. Processes it through the validation pipeline
    /// 3. Validates the status matches the expected predicate
    /// 4. Returns the block on success
    ///
    /// # Parameters
    /// * `deploy_datums` - Deploys to include in the block
    /// * `expected_status` - Predicate to validate the processing status
    pub async fn add_block_status<F>(
        &mut self,
        deploy_datums: &[Signed<DeployData>],
        expected_status: F,
    ) -> Result<BlockMessage, CasperError>
    where
        F: FnOnce(&ValidBlockProcessing) -> bool,
    {
        // Create block
        let result = self.create_block(deploy_datums).await?;

        // Extract block
        let block = match result {
            BlockCreatorResult::Created(b) => b,
            other => {
                return Err(CasperError::RuntimeError(format!(
                    "Expected Created block, got: {:?}",
                    other
                )))
            }
        };

        // Process block
        let status = self.process_block(block.clone()).await?;

        // Validate status
        if !expected_status(&status) {
            return Err(CasperError::RuntimeError(format!(
                "Block status did not match expected: {:?}",
                status
            )));
        }

        Ok(block)
    }

    /// Publishes a block to other nodes (equivalent to Scala publishBlock, line 204-208).
    ///
    /// This method:
    /// 1. Creates a block from deploys
    /// 2. Triggers handleReceive on all other nodes
    /// 3. Returns the created block
    ///
    /// # Parameters
    /// * `deploy_datums` - Deploys to include in the block
    /// * `nodes` - Other nodes to publish to
    pub async fn publish_block(
        &mut self,
        deploy_datums: &[Signed<DeployData>],
        nodes: &mut [&mut TestNode],
    ) -> Result<BlockMessage, CasperError> {
        // Create and add block
        let block = self.add_block_from_deploys(deploy_datums).await?;

        // Trigger handleReceive on all other nodes (excluding self)
        for node in nodes.iter_mut() {
            if node.local != self.local {
                node.handle_receive().await?;
            }
        }

        Ok(block)
    }

    /// Propagates a block to target nodes (equivalent to Scala propagateBlock, line 210-221).
    ///
    /// This method:
    /// 1. Logs block creation
    /// 2. Creates a block from deploys
    /// 3. Logs propagation targets
    /// 4. Calls processBlock on each target node
    /// 5. Returns the created block
    ///
    /// # Parameters
    /// * `deploy_datums` - Deploys to include in the block
    /// * `nodes` - Target nodes to propagate to
    pub async fn propagate_block(
        &mut self,
        deploy_datums: &[Signed<DeployData>],
        nodes: &mut [&mut TestNode],
    ) -> Result<BlockMessage, CasperError> {
        // Log block creation
        log::debug!("\n{} creating block", self.name);

        // Create and add block
        let block = self.add_block_from_deploys(deploy_datums).await?;

        // Filter targets (exclude self)
        let targets: Vec<&mut &mut TestNode> = nodes
            .iter_mut()
            .filter(|node| node.local != self.local)
            .collect();

        // Log propagation
        let target_names: Vec<String> = targets.iter().map(|node| node.name.clone()).collect();
        log::debug!(
            "{} ! [{}] => {}",
            self.name,
            models::rust::casper::pretty_printer::PrettyPrinter::build_string_block_message(
                &block, true
            ),
            target_names.join(" ; ")
        );

        // Process block on each target
        for node in targets {
            node.process_block(block.clone()).await?;
        }

        Ok(block)
    }

    /// Synchronizes this node with other nodes (equivalent to Scala syncWith, line 293-344).
    ///
    /// This method implements iterative synchronization:
    /// 1. Drains message queues from requested block peers
    /// 2. Handles receive on this node
    /// 3. Repeats until all blocks are received or max attempts reached
    ///
    /// # Parameters
    /// * `nodes` - Nodes to synchronize with
    pub async fn sync_with(&mut self, nodes: &mut [&mut TestNode]) -> Result<(), CasperError> {
        const MAX_SYNC_ATTEMPTS: usize = 10;

        // Build network map (peer -> node index)
        let network_map: std::collections::HashMap<PeerNode, usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.local != self.local)
            .map(|(idx, node)| (node.local.clone(), idx))
            .collect();

        // Initial handleReceive
        self.handle_receive().await?;

        // Check if all synced
        let mut done = {
            let requested = self.requested_blocks.read().unwrap();
            !requested.values().any(|req| !req.received)
        };

        let mut cnt = 0;

        // Synchronization loop
        while cnt < MAX_SYNC_ATTEMPTS && !done {
            // Get list of peers we're waiting for
            let asked_peers: Vec<PeerNode> = {
                let requested = self.requested_blocks.read().unwrap();
                requested
                    .values()
                    .flat_map(|req| {
                        if req.peers.is_empty() {
                            // Empty peers means broadcast - check everyone
                            network_map.keys().cloned().collect()
                        } else {
                            req.peers.clone()
                        }
                    })
                    .collect()
            };

            // Drain queues of asked peers
            for peer in asked_peers {
                if let Some(&idx) = network_map.get(&peer) {
                    nodes[idx].handle_receive().await?;
                }
            }

            // Handle receive on this node
            self.handle_receive().await?;

            // Check if we're done
            done = {
                let requested = self.requested_blocks.read().unwrap();
                !requested.values().any(|req| !req.received)
            };
            cnt += 1;
        }

        // Log results
        if !done {
            let requested = self.requested_blocks.read().unwrap();
            let pending: Vec<String> = requested
                .iter()
                .filter(|(_, req)| !req.received)
                .map(|(hash, req)| {
                    format!(
                        "{} -> {:?}",
                        models::rust::casper::pretty_printer::PrettyPrinter::build_string_no_limit(
                            hash
                        ),
                        req
                    )
                })
                .collect();

            log::warn!(
                "Node {} still pending requests for blocks (after {} attempts): {:?}",
                self.local,
                MAX_SYNC_ATTEMPTS,
                pending
            );
        } else {
            let peer_names: Vec<String> = network_map.keys().map(|p| p.to_string()).collect();
            log::info!(
                "Node {} has exchanged all the requested blocks with [{}] after {} round(s)",
                self.local,
                peer_names.join("; "),
                cnt
            );
        }

        Ok(())
    }

    /// Synchronizes with a single node.
    pub async fn sync_with_one(&mut self, node: &mut TestNode) -> Result<(), CasperError> {
        self.sync_with(&mut [node]).await
    }

    /// Synchronizes with two nodes.
    pub async fn sync_with_two(
        &mut self,
        node1: &mut TestNode,
        node2: &mut TestNode,
    ) -> Result<(), CasperError> {
        self.sync_with(&mut [node1, node2]).await
    }

    /// Synchronizes with multiple nodes (variadic version).
    pub async fn sync_with_many(&mut self, nodes: &mut [&mut TestNode]) -> Result<(), CasperError> {
        self.sync_with(nodes).await
    }

    /// Checks if this node contains a block (equivalent to Scala contains, line 346).
    pub fn contains(&self, block_hash: &BlockHash) -> bool {
        self.casper.contains(block_hash)
    }

    /// Checks if this node knows about a block (in storage or requested) (equivalent to Scala knowsAbout, line 347-348).
    pub fn knows_about(&self, block_hash: &BlockHash) -> bool {
        // Check if in storage
        let in_storage = self.contains(block_hash);

        // Check if in requested blocks
        let in_requested = {
            let requested = self.requested_blocks.read().unwrap();
            requested.contains_key(block_hash)
        };

        in_storage || in_requested
    }

    /// Shuts off this node by clearing its transport layer queue (equivalent to Scala shutoff, line 350).
    ///
    /// This is useful for simulating network partitions or node failures in tests.
    pub fn shutoff(&self) -> Result<(), CommError> {
        self.tle.test_network().clear(&self.local)
    }

    /// Visualizes the DAG starting from a block number (equivalent to Scala visualizeDag, line 352-369).
    ///
    /// This method:
    /// 1. Creates a StringSerializer for capturing the graph
    /// 2. Calls BlockAPI::visualize_dag with depth=Int::MAX and max_depth_limit=50
    /// 3. Uses GraphzGenerator::dag_as_cluster to generate the DOT format graph
    /// 4. Returns the graph as a String
    ///
    /// # Parameters
    /// * `start_block_number` - Starting block number for visualization
    pub async fn visualize_dag(&self, start_block_number: i64) -> Result<String, CasperError> {
        use casper::rust::api::{
            block_api::BlockAPI,
            graph_generator::{GraphConfig, GraphzGenerator},
        };
        use graphz::rust::graphz::StringSerializer;
        use std::sync::Arc;

        const API_MAX_BLOCKS_LIMIT: i32 = 50;

        // Create a StringSerializer to capture the graph output
        let serializer = Arc::new(StringSerializer::new());

        // Clone casper to avoid capturing self in the closure
        let casper = self.casper.clone();

        // Create a oneshot channel for sending the result
        let (sender, receiver) = tokio::sync::oneshot::channel::<String>();

        // Create the visualizer closure that calls GraphzGenerator::dag_as_cluster
        let visualizer = move |topo_sort: Vec<Vec<models::rust::block_hash::BlockHash>>,
                               lfb: String| {
            let serializer = serializer.clone();
            let casper = casper.clone();

            async move {
                // Clone the block_store (cheap since it's Arc-based) to get a mutable reference
                let mut block_store = casper.block_store().clone();
                GraphzGenerator::dag_as_cluster(
                    topo_sort,
                    lfb,
                    GraphConfig {
                        show_justification_lines: true,
                    },
                    serializer.clone(),
                    &mut block_store,
                )
                .await
                .map(|_| ())?;

                // After visualization is complete, get the content and send it
                let content = serializer.get_content().await;
                let _ = sender.send(content);

                Ok(())
            }
        };

        // Call BlockAPI::visualize_dag
        let result = BlockAPI::visualize_dag(
            &self.engine_cell,
            i32::MAX,
            start_block_number as i32,
            visualizer,
            receiver,
        )
        .await;

        match result {
            Ok(dot_string) => Ok(dot_string),
            Err(e) => Err(CasperError::RuntimeError(format!(
                "Failed to visualize DAG: {}",
                e
            ))),
        }
    }

    /// Prints a URL for visualizing the DAG (equivalent to Scala printVisualizeDagUrl, line 375-383).
    ///
    /// This method:
    /// 1. Calls visualize_dag to get the DOT format graph
    /// 2. URL-encodes the graph string
    /// 3. Prints a URL to https://dreampuf.github.io/GraphvizOnline/
    ///
    /// # Parameters
    /// * `start_block_number` - Starting block number for visualization
    pub async fn print_visualize_dag_url(
        &self,
        start_block_number: i64,
    ) -> Result<(), CasperError> {
        let dot = self.visualize_dag(start_block_number).await?;

        // URL encoding: encode special characters (similar to Java's URLEncoder.encode)
        let url_encoded = dot
            .chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                ' ' => "%20".to_string(),
                _ => format!("%{:02X}", c as u8),
            })
            .collect::<String>();

        println!(
            "DAG @ {}: https://dreampuf.github.io/GraphvizOnline/#{}",
            self.name, url_encoded
        );
        Ok(())
    }

    pub async fn handle_receive(&self) -> Result<(), CasperError> {
        let tle = self.tle.clone();
        let connections_cell = self.connections_cell.clone();
        let rp_conf = self.rp_conf.clone();
        let packet_handler = self.packet_handler.clone();

        let dispatch = Arc::new(
            move |protocol: Protocol| -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = Result<CommunicationResponse, CommError>>
                        + Send,
                >,
            > {
                let tle = tle.clone();
                let connections_cell = connections_cell.clone();
                let rp_conf = rp_conf.clone();
                let packet_handler = packet_handler.clone();

                Box::pin(async move {
                    match protocol.message {
                        Some(models::routing::protocol::Message::Packet(ref packet)) => {
                            // Extract peer from protocol header
                            let header = protocol.header.as_ref().ok_or_else(|| {
                                CommError::UnexpectedMessage("No header in protocol".to_string())
                            })?;

                            let sender_node = header.sender.as_ref().ok_or_else(|| {
                                CommError::UnexpectedMessage("No sender in header".to_string())
                            })?;

                            // Convert Node to PeerNode
                            let peer = PeerNode {
                                id: NodeIdentifier::new(hex::encode(&sender_node.id)),
                                endpoint: Endpoint::new(
                                    String::from_utf8_lossy(&sender_node.host).to_string(),
                                    sender_node.tcp_port,
                                    sender_node.udp_port,
                                ),
                            };

                            // Use CasperPacketHandler for packet processing
                            packet_handler.handle_packet(&peer, packet).await?;
                            Ok(CommunicationResponse::handled_without_message())
                        }
                        _ => {
                            handle_messages::handle(
                                &protocol,
                                tle.as_ref(),
                                &NOPPacketHandler::new(),
                                &connections_cell,
                                &rp_conf,
                            )
                            .await
                        }
                    }
                })
            },
        );

        let handle_streamed = Arc::new(
            |_blob: Blob| -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<(), CommError>> + Send>,
            > { Box::pin(async move { Ok(()) }) },
        );

        let _ = self.tls.handle_receive(dispatch, handle_streamed).await?;

        Ok(())
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
        let block_store = block_store_base;

        let block_dag_storage = BlockDagKeyValueStorage::new(&mut kvm).await.unwrap();
        let deploy_storage = Arc::new(Mutex::new(
            KeyValueDeployStorage::new(&mut kvm).await.unwrap(),
        ));

        let casper_buffer_storage = CasperBufferKeyValueStorage::new_from_kvm(&mut kvm)
            .await
            .unwrap();

        let rspace_store = kvm.r_space_stores().await.unwrap();
        let mergeable_store = RuntimeManager::mergeable_store(&mut kvm).await.unwrap();
        let runtime_manager = RuntimeManager::create_with_store(
            rspace_store,
            mergeable_store,
            Genesis::non_negative_mergeable_tag_name(),
        );

        let rho_history_repository = runtime_manager.get_history_repo();

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
                deploy_storage.clone(),
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

        let shard_id = "root".to_string();
        let finalization_rate = 1;

        let _approved_block = ApprovedBlock {
            candidate: ApprovedBlockCandidate {
                block: genesis.clone(),
                required_sigs: 0,
            },
            sigs: vec![],
        };

        let casper = {
            let shard_conf = CasperShardConf {
                fault_tolerance_threshold: 0.0,
                shard_name: shard_id.clone(),
                parent_shard_id: "".to_string(),
                finalization_rate: finalization_rate,
                max_number_of_parents: max_number_of_parents,
                max_parent_depth: max_parent_depth.unwrap_or(i32::MAX),
                synchrony_constraint_threshold: synchrony_constraint_threshold as f32,
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

            Arc::new(MultiParentCasperImpl {
                block_retriever: block_retriever.clone(),
                event_publisher: event_publisher.clone(),
                runtime_manager: Arc::new(Mutex::new(runtime_manager.clone())),
                estimator: estimator.clone(),
                block_store: block_store.clone(),
                block_dag_storage: block_dag_storage.clone(),
                deploy_storage: deploy_storage.clone(),
                casper_buffer_storage: casper_buffer_storage.clone(),
                validator_id: validator_id_opt.clone(),
                casper_shard_conf: shard_conf,
                approved_block: genesis.clone(),
            })
        };

        // Create EngineWithCasper (matches Scala line 167-177)
        let engine_with_casper = EngineWithCasper::new(casper.clone());

        // Create EngineCell (matches Scala line 177)
        let engine_cell = EngineCell::init();
        engine_cell.set(Arc::new(engine_with_casper)).await;

        // Create CasperPacketHandler (matches Scala line 178)
        let packet_handler = CasperPacketHandler::new(engine_cell.clone());

        TestNode {
            name,
            local: current_peer_node,
            tle,
            tls,
            genesis,
            validator_id_opt,
            synchrony_constraint_threshold,
            data_dir: new_storage_dir,
            max_number_of_parents,
            max_parent_depth,
            shard_id,
            finalization_rate,
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
            casper,
            engine_cell,
            packet_handler,
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

    /// Propagates messages across all nodes until all queues are empty (equivalent to Scala propagate, line 640-649).
    ///
    /// This static method:
    /// 1. Repeatedly calls handleReceive on all nodes
    /// 2. Checks if all message queues are empty after each round
    /// 3. Continues until all queues are empty (heat death) or max iterations
    ///
    /// This is useful for simulating complete message propagation in tests.
    ///
    /// # Parameters
    /// * `nodes` - All nodes in the network to propagate messages between
    pub async fn propagate(nodes: &mut [&mut TestNode]) -> Result<(), CasperError> {
        if nodes.is_empty() {
            return Ok(());
        }

        const MAX_PROPAGATION_ROUNDS: usize = 100;
        let mut rounds = 0;

        // Keep propagating until queues are empty or max rounds
        loop {
            if rounds >= MAX_PROPAGATION_ROUNDS {
                log::warn!(
                    "Propagation stopped after {} rounds - queues may not be empty",
                    MAX_PROPAGATION_ROUNDS
                );
                break;
            }

            // Call handleReceive on all nodes
            let mut any_messages = false;
            for node in nodes.iter() {
                // Check if this node's queue has messages
                let queue_size = node
                    .tle
                    .test_network()
                    .peer_queue(&node.local)
                    .unwrap_or_else(|_| std::collections::VecDeque::new())
                    .len();

                if queue_size > 0 {
                    any_messages = true;
                    node.handle_receive().await?;
                }
            }

            // If no messages were processed, we've reached heat death
            if !any_messages {
                break;
            }

            rounds += 1;
        }

        log::debug!("Propagation completed after {} rounds", rounds);
        Ok(())
    }

    /// Propagates messages between two nodes (equivalent to Scala propagate overload, line 651-652).
    ///
    /// Convenience method for two-node propagation.
    pub async fn propagate_two(
        node1: &mut TestNode,
        node2: &mut TestNode,
    ) -> Result<(), CasperError> {
        Self::propagate(&mut [node1, node2]).await
    }
}
