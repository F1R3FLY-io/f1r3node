// See casper/src/test/scala/coop/rchain/casper/helper/TestNode.scala

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, RwLock},
};
use tokio::sync::mpsc;

use block_storage::rust::{
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
            proposer::new_proposer,
        },
    },
    casper::{Casper, CasperShardConf, MultiParentCasper},
    engine::block_retriever::{BlockRetriever, RequestState, RequestedBlocks},
    errors::CasperError,
    estimator::Estimator,
    genesis::genesis::Genesis,
    multi_parent_casper_impl::MultiParentCasperImpl,
    safety_oracle::CliqueOracleImpl,
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
use rspace_plus_plus::rspace::history::Either;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;

use crate::util::{
    comm::transport_layer_test_impl::{
        test_network::TestNetwork, TransportLayerServerTestImpl, TransportLayerTestImpl,
    },
    genesis_builder::GenesisContext,
    rholang::resources,
};

use casper::rust::{
    engine::{engine_cell::EngineCell, running::Running},
    util::comm::casper_packet_handler::CasperPacketHandler,
};

pub struct TestNode {
    pub name: String,
    pub local: PeerNode,
    pub tle: Arc<TransportLayerTestImpl>,
    pub tls: TransportLayerServerTestImpl,
    pub genesis: BlockMessage,
    pub validator_id_opt: Option<ValidatorIdentity>,
    // Note: blockProcessingPipe implemented as method process_block_through_pipe
    pub block_processor: BlockProcessor<TransportLayerTestImpl>,
    pub block_store: KeyValueBlockStore,
    pub block_dag_storage: BlockDagKeyValueStorage,
    pub deploy_storage: Arc<Mutex<KeyValueDeployStorage>>,
    pub runtime_manager: RuntimeManager,
    // Note: no log field, logging will come from log crate
    pub requested_blocks: RequestedBlocks,
    pub connections_cell: ConnectionsCell,
    pub rp_conf: RPConf,
    // Casper instance (Arc<Mutex> for shared ownership with interior mutability)
    pub casper: Arc<MultiParentCasperImpl<TransportLayerTestImpl>>,
    // Engine cell for packet handling (matches Scala line 177)
    pub engine_cell: EngineCell,
    // Packet handler for receiving messages (matches Scala line 178)
    pub packet_handler: CasperPacketHandler,
}

impl TestNode {
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
        // Deploy all datums
        for deploy_datum in deploy_datums {
            self.casper.deploy(deploy_datum.clone())?;
        }

        // Get snapshot
        let snapshot = self.casper.get_snapshot().await?;

        // Get validator
        let validator = self.casper.get_validator().ok_or_else(|| {
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
            false,
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
        Self::process_block_through_pipe(self.casper.clone(), &self.block_processor, block).await
    }

    /// Processes a block through the validation pipeline.
    ///
    /// This method:
    /// 1. Checks if block is of interest
    /// 2. Checks if well-formed and stores
    /// 3. Checks dependencies
    /// 4. Validates with effects
    pub async fn process_block_through_pipe(
        casper: Arc<dyn Casper + Send + Sync + 'static>,
        block_processor: &BlockProcessor<TransportLayerTestImpl>,
        block: BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError> {
        // Check if block is of interest
        let is_of_interest = block_processor.check_if_of_interest(casper.clone(), &block)?;

        if !is_of_interest {
            return Ok(Either::Left(BlockStatus::not_of_interest()));
        }

        // Check if well-formed and store
        let is_well_formed = block_processor
            .check_if_well_formed_and_store(&block)
            .await?;

        if !is_well_formed {
            return Ok(Either::Left(BlockStatus::invalid_format()));
        }

        // Check dependencies
        let dependencies_ready = block_processor
            .check_dependencies_with_effects(casper.clone(), &block)
            .await?;

        if !dependencies_ready {
            return Ok(Either::Left(BlockStatus::missing_blocks()));
        }

        // Validate with effects
        block_processor
            .validate_with_effects(casper.clone(), &block, None)
            .await
    }

    /// Adds and processes a block (equivalent to Scala addBlock(block), line 198-199).
    ///
    /// Takes an existing block and processes it through the validation pipeline.
    pub async fn add_block(
        &mut self,
        block: BlockMessage,
    ) -> Result<ValidBlockProcessing, CasperError> {
        Self::process_block_through_pipe(self.casper.clone(), &self.block_processor, block).await
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

    /// Helper method to propagate a block from a node at a specific index in a nodes array.
    ///
    /// This method works around Rust's borrow checker limitation where we cannot do:
    /// ```ignore
    /// nodes[0].propagate_block(&deploys, &mut nodes)
    /// ```
    /// because it would require borrowing `nodes` mutably twice:
    /// - First borrow: `nodes[0]` (mutable access to call the method)
    /// - Second borrow: `&mut nodes` (mutable parameter to pass all nodes)
    ///
    /// This helper uses `split_at_mut` to split the array into non-overlapping parts,
    /// allowing the borrow checker to verify that we're accessing different memory regions.
    ///
    /// # Scala equivalent
    /// In Scala this is simply: `nodes(index).propagateBlock(deploys)(nodes: _*)`
    ///
    /// # Parameters
    /// * `nodes` - All nodes in the network
    /// * `index` - Index of the node that should create and propagate the block
    /// * `deploy_datums` - Deploys to include in the block
    pub async fn propagate_block_at_index(
        nodes: &mut [TestNode],
        index: usize,
        deploy_datums: &[Signed<DeployData>],
    ) -> Result<BlockMessage, CasperError> {
        let (before, rest) = nodes.split_at_mut(index);
        let (current, after) = rest.split_at_mut(1);
        let mut all_others: Vec<&mut TestNode> =
            before.iter_mut().chain(after.iter_mut()).collect();
        current[0]
            .propagate_block(deploy_datums, &mut all_others)
            .await
    }

    /// Helper method to propagate a block from one node to another specific node.
    ///
    /// This method works around Rust's borrow checker limitation where we cannot do:
    /// ```ignore
    /// nodes[from_index].propagate_block(&deploys, &mut [&mut nodes[to_index]])
    /// ```
    /// because it would require borrowing from `nodes` mutably twice.
    ///
    /// This helper uses `split_at_mut` to split the array into non-overlapping parts,
    /// allowing the borrow checker to verify that we're accessing different memory regions.
    ///
    /// # Scala equivalent
    /// In Scala this is simply: `nodes(from_index).propagateBlock(deploys)(nodes(to_index))`
    ///
    /// # Parameters
    /// * `nodes` - All nodes in the network
    /// * `from_index` - Index of the node that should create and propagate the block
    /// * `to_index` - Index of the node that should receive the block
    /// * `deploy_datums` - Deploys to include in the block
    pub async fn propagate_block_to_one(
        nodes: &mut [TestNode],
        from_index: usize,
        to_index: usize,
        deploy_datums: &[Signed<DeployData>],
    ) -> Result<BlockMessage, CasperError> {
        assert_ne!(
            from_index, to_index,
            "from_index and to_index must be different"
        );

        // Split to get mutable references to both nodes without overlapping borrows
        if from_index < to_index {
            let (left, right) = nodes.split_at_mut(to_index);
            let from_node = &mut left[from_index];
            let to_node = &mut right[0];
            from_node
                .propagate_block(deploy_datums, &mut [to_node])
                .await
        } else {
            let (left, right) = nodes.split_at_mut(from_index);
            let to_node = &mut left[to_index];
            let from_node = &mut right[0];
            from_node
                .propagate_block(deploy_datums, &mut [to_node])
                .await
        }
    }

    /// Helper method to publish a block from a node at a specific index to all other nodes.
    ///
    /// This method works around Rust's borrow checker limitation similar to `propagate_block_at_index`.
    ///
    /// # Scala equivalent
    /// In Scala this is simply: `nodes(index).publishBlock(deploys)(nodes: _*)`
    ///
    /// # Parameters
    /// * `nodes` - All nodes in the network
    /// * `index` - Index of the node that should create and publish the block
    /// * `deploy_datums` - Deploys to include in the block
    pub async fn publish_block_at_index(
        nodes: &mut [TestNode],
        index: usize,
        deploy_datums: &[Signed<DeployData>],
    ) -> Result<BlockMessage, CasperError> {
        let (before, rest) = nodes.split_at_mut(index);
        let (current, after) = rest.split_at_mut(1);
        let mut all_others: Vec<&mut TestNode> =
            before.iter_mut().chain(after.iter_mut()).collect();
        current[0]
            .publish_block(deploy_datums, &mut all_others)
            .await
    }

    /// Helper method to publish a block from one node to another specific node.
    ///
    /// # Scala equivalent
    /// In Scala this is simply: `nodes(from_index).publishBlock(deploys)(nodes(to_index))`
    ///
    /// # Parameters
    /// * `nodes` - All nodes in the network
    /// * `from_index` - Index of the node that should create and publish the block
    /// * `to_index` - Index of the node that should receive the block
    /// * `deploy_datums` - Deploys to include in the block
    pub async fn publish_block_to_one(
        nodes: &mut [TestNode],
        from_index: usize,
        to_index: usize,
        deploy_datums: &[Signed<DeployData>],
    ) -> Result<BlockMessage, CasperError> {
        assert_ne!(
            from_index, to_index,
            "from_index and to_index must be different"
        );

        if from_index < to_index {
            let (left, right) = nodes.split_at_mut(to_index);
            let from_node = &mut left[from_index];
            let to_node = &mut right[0];
            from_node.publish_block(deploy_datums, &mut [to_node]).await
        } else {
            let (left, right) = nodes.split_at_mut(from_index);
            let to_node = &mut left[to_index];
            let from_node = &mut right[0];
            from_node.publish_block(deploy_datums, &mut [to_node]).await
        }
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
        tracing::debug!("\n{} creating block", self.name);

        // Create and add block
        let block = self.add_block_from_deploys(deploy_datums).await?;

        // Filter targets (exclude self)
        let targets: Vec<&mut &mut TestNode> = nodes
            .iter_mut()
            .filter(|node| node.local != self.local)
            .collect();

        // Log propagation
        let target_names: Vec<String> = targets.iter().map(|node| node.name.clone()).collect();
        tracing::debug!(
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
            let requested = self.requested_blocks.lock().unwrap();
            !requested.values().any(|req| !req.received)
        };

        let mut cnt = 0;

        // Synchronization loop
        while cnt < MAX_SYNC_ATTEMPTS && !done {
            // Get list of peers we're waiting for
            let asked_peers: Vec<PeerNode> = {
                let requested = self.requested_blocks.lock().unwrap();
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
                let requested = self.requested_blocks.lock().unwrap();
                !requested.values().any(|req| !req.received)
            };
            cnt += 1;
        }

        // Log results
        if !done {
            let requested = self.requested_blocks.lock().unwrap();
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

            tracing::warn!(
                "Node {} still pending requests for blocks (after {} attempts): {:?}",
                self.local,
                MAX_SYNC_ATTEMPTS,
                pending
            );
        } else {
            let peer_names: Vec<String> = network_map.keys().map(|p| p.to_string()).collect();
            tracing::info!(
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
            let requested = self.requested_blocks.lock().unwrap();
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

    pub async fn handle_receive(&self) -> Result<(), CasperError> {
        let tle = self.tle.clone();
        let connections_cell = self.connections_cell.clone();
        let rp_conf = self.rp_conf.clone();
        let packet_handler = self.packet_handler.clone();

        // Clone casper and block_processor for direct BlockMessage processing
        let casper = self.casper.clone();
        let block_processor = self.block_processor.clone();

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
                let casper = casper.clone();
                let block_processor = block_processor.clone(); // Clone Arc for this invocation

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

                            // Parse CasperMessage to check if it's a BlockMessage
                            use casper::rust::protocol::{
                                casper_message_from_proto, to_casper_message_proto,
                            };
                            use models::rust::casper::protocol::casper_message::CasperMessage;

                            let parse_result = to_casper_message_proto(packet).get();
                            if let Ok(proto) = parse_result {
                                if let Ok(casper_msg) = casper_message_from_proto(proto) {
                                    match casper_msg {
                                        CasperMessage::BlockMessage(block) => {
                                            // Call process_block_through_pipe (static method)
                                            let _result = TestNode::process_block_through_pipe(
                                                casper.clone(),
                                                &block_processor,
                                                block,
                                            )
                                            .await
                                            .map_err(|e| CommError::CasperError(e.to_string()))?;

                                            return Ok(
                                                CommunicationResponse::handled_without_message(),
                                            );
                                        }
                                        _ => {
                                            // All other messages: use engine as before
                                            packet_handler.handle_packet(&peer, packet).await?;
                                            return Ok(
                                                CommunicationResponse::handled_without_message(),
                                            );
                                        }
                                    }
                                }
                            }

                            // Fallback: if parsing failed, use packet handler
                            packet_handler.handle_packet(&peer, packet).await?;
                            Ok(CommunicationResponse::handled_without_message())
                        }
                        _ => {
                            handle_messages::handle(
                                &protocol,
                                tle.clone(),
                                Arc::new(NOPPacketHandler::new()),
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
            genesis.clone(),
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
        genesis_context: GenesisContext,
        synchrony_constraint_threshold: f64,
        max_number_of_parents: i32,
        max_parent_depth: Option<i32>,
        with_read_only_size: usize,
        test_network: TestNetwork,
    ) -> Result<Vec<TestNode>, CasperError> {
        let genesis = genesis_context.genesis_block.clone();
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
                synchrony_constraint_threshold,
                max_number_of_parents,
                max_parent_depth,
                is_readonly,
                test_network.clone(),
                &genesis_context,
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
        // TODO: logical_time: LogicalTime,
        synchrony_constraint_threshold: f64,
        max_number_of_parents: i32,
        max_parent_depth: Option<i32>,
        is_read_only: bool,
        test_network: TestNetwork,
        genesis_context: &GenesisContext,
    ) -> TestNode {
        let tle = Arc::new(TransportLayerTestImpl::new(test_network.clone()));
        let tls =
            TransportLayerServerTestImpl::new(current_peer_node.clone(), test_network.clone());

        // With shared LMDB, we don't need to copy storage directories.
        // Use the shared LMDB path for data_dir (for logging/debugging purposes only).
        let _new_storage_dir = resources::get_shared_lmdb_path();
        // Use mk_test_rnode_store_manager_with_shared_rspace to get a new scope with genesis data copied
        // This ensures test isolation for blocks/DAG (each TestNode has its own scope)
        // while sharing RSpace scope so all nodes in this test can see each other's state
        let mut kvm = resources::mk_test_rnode_store_manager_with_shared_rspace(
            genesis_context,
            &genesis_context.rspace_scope_id
        ).await
            .expect("Failed to create store manager with shared RSpace");

        let block_store_base = KeyValueBlockStore::create_from_kvm(&mut *kvm).await.unwrap();
        let block_store = block_store_base;

        // Initialize block store with genesis block
        block_store.put(genesis.block_hash.clone(), &genesis)
            .expect("Failed to store genesis block in TestNode");

        let block_dag_storage = resources::block_dag_storage_from_dyn(&mut *kvm).await.unwrap();
        
        // Initialize DAG storage with genesis block metadata
        block_dag_storage.insert(&genesis, false, true)
            .expect("Failed to insert genesis into DAG storage in TestNode");
        let deploy_storage = Arc::new(Mutex::new(
            resources::key_value_deploy_storage_from_dyn(&mut *kvm).await.unwrap(),
        ));

        let casper_buffer_storage = resources::casper_buffer_storage_from_dyn(&mut *kvm)
            .await
            .unwrap();

        let rspace_store = (&mut *kvm).r_space_stores().await.unwrap();
        let mergeable_store = resources::mergeable_store_from_dyn(&mut *kvm).await.unwrap();
        // Use create_with_history to ensure tests can reset to genesis state root hash
        let (runtime_manager, _rho_history_repository) = RuntimeManager::create_with_history(
            rspace_store,
            mergeable_store,
            Genesis::non_negative_mergeable_tag_name(),
        );

        let connections_cell = ConnectionsCell::new();
        let _clique_oracle = CliqueOracleImpl;
        let estimator = Estimator::apply(max_number_of_parents, max_parent_depth);
        let rp_conf = create_rp_conf_ask(current_peer_node.clone(), None, None);
        let event_publisher = F1r3flyEvents::new(None);
        // Scala: implicit val requestedBlocks: RequestedBlocks[F] = Ref.unsafe[F, Map[BlockHash, RequestState]](Map.empty)
        let requested_blocks = Arc::new(Mutex::new(HashMap::<BlockHash, RequestState>::new()));
        // Scala: implicit val blockRetriever: BlockRetriever[F] = BlockRetriever.of[F]
        let block_retriever = BlockRetriever::new(
            requested_blocks.clone(),
            tle.clone(),
            connections_cell.clone(),
            rp_conf.clone(),
        );

        let _ = test_network.add_peer(&current_peer_node);

        // Proposer
        let validator_id_opt = if is_read_only {
            None
        } else {
            Some(ValidatorIdentity::new(&sk))
        };

        let _proposer_opt = match validator_id_opt {
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
            mpsc::unbounded_channel::<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>();
        let block_processor_queue = (
            block_processor_queue_tx,
            Arc::new(Mutex::new(block_processor_queue_rx)),
        );

        let _block_processor_state = Arc::new(RwLock::new(HashSet::<BlockHash>::new()));

        let shard_id = "root".to_string();
        let finalization_rate = 1;

        let _approved_block = ApprovedBlock {
            candidate: ApprovedBlockCandidate {
                block: genesis.clone(),
                required_sigs: 0,
            },
            sigs: vec![],
        };

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

        let casper_impl = MultiParentCasperImpl {
            block_retriever: block_retriever.clone(),
            event_publisher: event_publisher.clone(),
            runtime_manager: Arc::new(tokio::sync::Mutex::new(runtime_manager.clone())),
            estimator: estimator.clone(),
            block_store: block_store.clone(),
            block_dag_storage: block_dag_storage.clone(),
            deploy_storage: deploy_storage.clone(),
            casper_buffer_storage: casper_buffer_storage.clone(),
            validator_id: validator_id_opt.clone(),
            casper_shard_conf: shard_conf,
            approved_block: genesis.clone(),
        };

        let casper = Arc::new(casper_impl);

        // Create Running engine

        // Create the_init as a no-op async function
        let the_init: Arc<
            dyn Fn() -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<(), CasperError>> + Send>,
                > + Send
                + Sync,
        > = Arc::new(|| Box::pin(async { Ok(()) }));

        let running_engine = Running::new(
            block_processor_queue.0.clone(),      // block_processing_queue_tx
            Arc::new(Mutex::new(HashSet::new())), // blocks_in_processing (converted from block_processor_state)
            casper.clone() as Arc<dyn MultiParentCasper + Send + Sync>, // casper
            _approved_block.clone(),              // approved_block
            the_init,                             // the_init
            true,                                 // disable_state_exporter
            tle.clone(),                          // transport
            rp_conf.clone(),                      // conf
            block_retriever.clone(),              // block_retriever
        );

        // Create EngineCell
        let engine_cell = EngineCell::init();
        engine_cell.set(Arc::new(running_engine)).await;

        // Create CasperPacketHandler
        let packet_handler = CasperPacketHandler::new(engine_cell.clone());

        TestNode {
            name,
            local: current_peer_node,
            tle,
            tls,
            genesis,
            validator_id_opt,
            block_processor,
            block_store,
            block_dag_storage,
            deploy_storage,
            runtime_manager,
            requested_blocks,
            connections_cell,
            rp_conf,
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
                tracing::warn!(
                    "Propagation stopped after {} rounds - queues may not be empty",
                    MAX_PROPAGATION_ROUNDS
                );
                break;
            }

            // Call handleReceive on all nodes (matching Scala's traverse_)
            for node in nodes.iter() {
                node.handle_receive().await?;
            }

            // Check heat death: all queues empty
            let mut any_messages = false;
            for node in nodes.iter() {
                let queue_size = node
                    .tle
                    .test_network()
                    .peer_queue(&node.local)
                    .unwrap_or_else(|_| std::collections::VecDeque::new())
                    .len();
                if queue_size > 0 {
                    any_messages = true;
                    break;
                }
            }

            // If no messages remain, we've reached heat death
            if !any_messages {
                break;
            }

            rounds += 1;
        }

        tracing::debug!("Propagation completed after {} rounds", rounds);
        Ok(())
    }
}
