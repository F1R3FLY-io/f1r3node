// See node/src/main/scala/coop/rchain/node/runtime/NodeRuntime.scala

use casper::rust::errors::CasperError;
use comm::rust::peer_node::NodeIdentifier;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::info;

use crate::rust::{
    configuration::{KamonConf, NodeConf},
    effects::node_discover,
    node_environment,
};

// Type aliases for repeatable async operations
pub type CasperLoop =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync>;
pub type EngineInit =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync>;

/// Wrapper for task results that includes the task name for identification
#[derive(Debug)]
struct NamedTaskResult {
    name: String,
    result: eyre::Result<()>,
}

/// Spawn a named task in a JoinSet
///
/// Wraps the task result with its name so we can identify which task completed.
fn spawn_named_task(
    join_set: &mut JoinSet<NamedTaskResult>,
    name: impl Into<String>,
    task: impl Future<Output = eyre::Result<()>> + Send + 'static,
) {
    let name = name.into();
    join_set.spawn(async move {
        let result = task.await;
        NamedTaskResult { name, result }
    });
}

/// NodeRuntime - Main entry point for running an F1r3node
///
/// This struct manages the entire lifecycle of a node, including:
/// - Initialization of all subsystems
/// - Orchestration of concurrent tasks (Casper, block processing, networking)
/// - Server management (gRPC, HTTP, Kademlia)
/// - Graceful shutdown and cleanup
pub struct NodeRuntime {
    node_conf: NodeConf,
    #[allow(dead_code)] // Will be used in Phase 3 for metrics configuration
    kamon_conf: KamonConf, // should be double-checked with Yurii
    id: NodeIdentifier,
}

impl NodeRuntime {
    /// Create a new NodeRuntime instance
    ///
    /// # Arguments
    /// * `node_conf` - Node configuration
    /// * `kamon_conf` - Kamon metrics configuration (deprecated, for compatibility)
    /// * `id` - Node identifier derived from TLS certificate
    pub fn new(node_conf: NodeConf, kamon_conf: KamonConf, id: NodeIdentifier) -> Self {
        Self {
            node_conf,
            kamon_conf,
            id,
        }
    }

    /// Main node entry point
    ///
    /// Initializes all node subsystems and starts the main runtime loop.
    ///
    /// Returns `Ok(())` on successful node shutdown, or an error if initialization fails
    pub async fn main(&self) -> eyre::Result<()> {
        info!("NodeRuntime.main() called");

        // Fetch local peer node
        let local = comm::rust::who_am_i::fetch_local_peer_node(
            self.node_conf.protocol_server.host.clone(),
            self.node_conf.protocol_server.port,
            self.node_conf.peers_discovery.port,
            self.node_conf.protocol_server.no_upnp,
            self.id.clone(),
        )
        .await
        .map_err(|e| eyre::eyre!("Failed to fetch local peer node: {}", e))?;

        info!("Local peer node: {}", local.to_address());

        // Metrics and time effects stubbed for now (coming in separate PR)
        let _metrics = (); // Placeholder
        let _time = (); // Placeholder

        // Create transport client
        let transport = {
            use comm::rust::transport::grpc_transport_client::GrpcTransportClient;
            use std::collections::HashMap;

            let channels_map = Arc::new(tokio::sync::Mutex::new(HashMap::new()));

            // Read certificate and key file contents
            let cert = tokio::fs::read_to_string(&self.node_conf.tls.certificate_path)
                .await
                .map_err(|e| eyre::eyre!("Failed to read certificate file: {}", e))?;
            let key = tokio::fs::read_to_string(&self.node_conf.tls.key_path)
                .await
                .map_err(|e| eyre::eyre!("Failed to read key file: {}", e))?;

            const CLIENT_QUEUE_SIZE: i32 = 100;

            GrpcTransportClient::new(
                self.node_conf.protocol_client.network_id.clone(),
                cert,
                key,
                self.node_conf.protocol_client.grpc_max_recv_message_size as i32,
                self.node_conf.protocol_client.grpc_stream_chunk_size as i32,
                CLIENT_QUEUE_SIZE,
                channels_map,
                self.node_conf.protocol_client.network_timeout,
            )
            .map_err(|e| eyre::eyre!("Failed to create transport client: {}", e))?
        };

        info!("Transport client created successfully");

        // Create RP connections
        let rp_connections = comm::rust::rp::connect::ConnectionsCell::new();

        // Determine initial peer for bootstrapping
        let init_peer = if self.node_conf.standalone {
            None
        } else {
            Some(
                comm::rust::peer_node::PeerNode::from_address(
                    &self.node_conf.protocol_client.bootstrap,
                )
                .map_err(|e| eyre::eyre!("Failed to parse bootstrap peer address: {}", e))?,
            )
        };

        // Create RPConf
        let rp_conf = comm::rust::rp::rp_conf::RPConf::new(
            local.clone(),
            self.node_conf.protocol_client.network_id.clone(),
            init_peer.clone(),
            self.node_conf.protocol_client.network_timeout,
            self.node_conf.protocol_client.batch_max_connections as usize,
            self.node_conf.peers_discovery.heartbeat_batch_size as usize,
        );

        // Wrap RPConf in RPConfCell for shared mutable access (allows dynamic IP updates)
        let rp_conf_cell = comm::rust::rp::rp_conf::RPConfCell::new(rp_conf.clone());

        // Create requested blocks tracking
        let requested_blocks = Arc::new(std::sync::Mutex::new(std::collections::HashMap::<
            models::rust::block_hash::BlockHash,
            casper::rust::engine::block_retriever::RequestState,
        >::new()));

        info!("RP connections and configuration initialized");

        // Create BlockRetriever
        let block_retriever = {
            use casper::rust::engine::block_retriever::BlockRetriever;

            BlockRetriever::new(
                requested_blocks.clone(),
                Arc::new(transport.clone()),
                rp_connections.clone(), // ConnectionsCell is Clone and already wraps Arc
                rp_conf.clone(),        // BlockRetriever uses RPConf by value
            )
        };

        info!("BlockRetriever initialized");

        // Create KademliaRPC
        let kademlia_rpc = {
            use comm::rust::discovery::grpc_kademlia_rpc::GrpcKademliaRPC;

            Arc::new(GrpcKademliaRPC::new(
                self.node_conf.protocol_server.network_id.clone(),
                self.node_conf.protocol_client.network_timeout,
                self.node_conf.protocol_server.allow_private_addresses,
                local.clone(),
            ))
        };

        info!("KademliaRPC initialized");

        // Create KademliaStore
        let kademlia_store = {
            use comm::rust::discovery::kademlia_store::KademliaStore;

            Arc::new(KademliaStore::new(self.id.clone(), kademlia_rpc.clone()))
        };

        info!("KademliaStore initialized");

        // Update bootstrap peer's last seen timestamp
        if let Some(ref peer) = init_peer {
            kademlia_store
                .update_last_seen(peer)
                .await
                .map_err(|e| eyre::eyre!("Failed to update last seen for bootstrap peer: {}", e))?;
            info!(
                "Updated last seen timestamp for bootstrap peer: {}",
                peer.to_address()
            );
        }

        // Create NodeDiscovery
        let node_discovery = node_discover(
            self.id.clone(),
            kademlia_rpc.clone(),
            kademlia_store.clone(),
        )
        .await?;

        info!("NodeDiscovery initialized");

        // Create event bus with circular buffer (capacity: 1)
        let event_bus = {
            use shared::rust::shared::f1r3fly_events::F1r3flyEvents;

            F1r3flyEvents::new(Some(1))
        };

        info!("Event bus (F1r3flyEvents) initialized");

        // Create last approved block storage
        let last_approved_block = Arc::new(std::sync::Mutex::new(None));

        // Call setup_node_program to initialize core components
        info!("Calling setup_node_program...");

        let result = crate::rust::runtime::setup::setup_node_program(
            rp_connections.clone(),
            rp_conf_cell.clone(),
            Arc::new(transport.clone()),
            block_retriever,
            self.node_conf.clone(),
            event_bus.clone(),
            node_discovery.clone(),
            last_approved_block.clone(),
        )
        .await?;

        // Destructure the result
        let (
            packet_handler,
            api_servers,
            casper_loop,
            update_fork_choice_loop,
            engine_init,
            casper_launch,
            reporting_http_routes,
            web_api,
            admin_web_api,
            proposer_opt,
            proposer_queue_rx,
            proposer_queue_tx,
            proposer_state_ref_opt,
            block_processor,
            block_processor_state,
            block_processor_queue_tx,
            block_processor_queue_rx,
            trigger_propose_f,
            block_report_api,
            _block_store, // Kept in scope to ensure LMDB cleanup happens on drop
        ) = result;

        info!("setup_node_program completed successfully");

        // Launch Casper
        info!("Launching Casper...");
        casper_launch.launch().await?;
        info!("Casper launched successfully");

        // Run the node program - orchestrates all concurrent tasks
        info!("Starting node program...");
        let program = self.node_program(
            api_servers,
            casper_loop,
            update_fork_choice_loop,
            engine_init,
            reporting_http_routes,
            web_api,
            admin_web_api,
            proposer_opt,
            proposer_queue_rx,
            proposer_queue_tx,
            trigger_propose_f,
            proposer_state_ref_opt,
            block_processor,
            block_processor_state,
            block_processor_queue_tx,
            block_processor_queue_rx,
            transport,
            rp_conf_cell,
            rp_connections,
            kademlia_store,
            node_discovery,
            packet_handler,
            event_bus,
            block_report_api,
        );

        // Wrap with error handling
        handle_unrecoverable_errors(program).await
    }

    /// Node program - orchestrates all concurrent tasks
    ///
    /// Coordinates all long-running tasks that make up the node:
    /// - API servers (gRPC and HTTP)
    /// - Casper consensus loops
    /// - Block processing
    /// - Proposer (if validator)
    /// - Network discovery and connection management
    #[allow(clippy::too_many_arguments)]
    async fn node_program<
        T: comm::rust::transport::transport_layer::TransportLayer + Send + Sync + Clone + 'static,
    >(
        &self,
        api_servers: crate::rust::runtime::api_servers::APIServers,
        casper_loop: CasperLoop,
        update_fork_choice_loop: CasperLoop,
        engine_init: EngineInit,
        reporting_http_routes: crate::rust::web::reporting_routes::ReportingHttpRoutes,
        web_api: Arc<dyn crate::rust::api::web_api::WebApi + Send + Sync + 'static>,
        admin_web_api: Arc<
            dyn crate::rust::api::admin_web_api::AdminWebApi + Send + Sync + 'static,
        >,
        proposer_opt: Option<casper::rust::blocks::proposer::proposer::ProductionProposer<T>>,
        proposer_queue_rx: tokio::sync::mpsc::UnboundedReceiver<(
            Arc<dyn casper::rust::casper::Casper + Send + Sync>,
            bool,
            tokio::sync::oneshot::Sender<casper::rust::blocks::proposer::proposer::ProposerResult>,
        )>,
        proposer_queue_tx: tokio::sync::mpsc::UnboundedSender<(
            Arc<dyn casper::rust::casper::Casper + Send + Sync>,
            bool,
            tokio::sync::oneshot::Sender<casper::rust::blocks::proposer::proposer::ProposerResult>,
        )>,
        trigger_propose_f: Option<Arc<casper::rust::ProposeFunction>>,
        proposer_state_ref_opt: Option<
            Arc<tokio::sync::RwLock<casper::rust::state::instances::ProposerState>>,
        >,
        block_processor: casper::rust::blocks::block_processor::BlockProcessor<T>,
        block_processor_state: Arc<
            std::sync::Mutex<std::collections::HashSet<models::rust::block_hash::BlockHash>>,
        >,
        block_processor_queue_tx: tokio::sync::mpsc::UnboundedSender<(
            Arc<dyn casper::rust::casper::MultiParentCasper + Send + Sync>,
            models::rust::casper::protocol::casper_message::BlockMessage,
        )>,
        block_processor_queue_rx: tokio::sync::mpsc::UnboundedReceiver<(
            Arc<dyn casper::rust::casper::MultiParentCasper + Send + Sync>,
            models::rust::casper::protocol::casper_message::BlockMessage,
        )>,
        transport: comm::rust::transport::grpc_transport_client::GrpcTransportClient,
        rp_conf_cell: comm::rust::rp::rp_conf::RPConfCell,
        rp_connections: comm::rust::rp::connect::ConnectionsCell,
        kademlia_store: Arc<
            comm::rust::discovery::kademlia_store::KademliaStore<
                comm::rust::discovery::grpc_kademlia_rpc::GrpcKademliaRPC,
            >,
        >,
        node_discovery: Arc<dyn comm::rust::discovery::node_discovery::NodeDiscovery + Send + Sync>,
        packet_handler: Arc<
            dyn comm::rust::p2p::packet_handler::PacketHandler + Send + Sync + 'static,
        >,
        event_bus: shared::rust::shared::f1r3fly_events::F1r3flyEvents,
        block_report_api: Arc<casper::rust::api::block_report_api::BlockReportAPI>,
    ) -> eyre::Result<()> {
        // Display node startup info
        if self.node_conf.standalone {
            info!("Starting stand-alone node.");
        } else {
            info!(
                "Starting node that will bootstrap from {}",
                self.node_conf.protocol_client.bootstrap
            );
        }

        // Get local peer configuration
        let rp_conf = rp_conf_cell
            .read()
            .map_err(|e| eyre::eyre!("Failed to read RPConf: {}", e))?;
        let local = rp_conf.local.clone();

        let address = local.to_address();
        let host = local.endpoint.host.clone();

        info!("Node address: {}, host: {}", address, host);

        // Build server instances
        info!("Building server instances...");

        let event_stream = event_bus.consume();

        // Create gRPC packet handler closure
        let grpc_packet_handler = {
            let transport_arc = Arc::new(transport.clone());
            let packet_handler_arc = packet_handler.clone();
            let rp_connections_clone = rp_connections.clone();
            let rp_conf_clone = rp_conf.clone();

            Arc::new(move |protocol: models::routing::Protocol| {
                let transport = transport_arc.clone();
                let handler = packet_handler_arc.clone();
                let connections = rp_connections_clone.clone();
                let conf = rp_conf_clone.clone();

                Box::pin(async move {
                    use comm::rust::rp::handle_messages;
                    handle_messages::handle(&protocol, transport, handler, &connections, &conf).await
                })
                    as Pin<
                        Box<
                            dyn Future<
                                    Output = Result<
                                        comm::rust::transport::communication_response::CommunicationResponse,
                                        comm::rust::errors::CommError,
                                    >,
                                > + Send,
                        >,
                    >
            })
        };

        // Create gRPC blob handler closure
        let grpc_blob_handler = {
            let packet_handler_arc = packet_handler.clone();

            Arc::new(move |blob: comm::rust::transport::transport_layer::Blob| {
                let handler = packet_handler_arc.clone();

                Box::pin(async move { handler.handle_packet(&blob.sender, &blob.packet).await })
                    as Pin<
                        Box<dyn Future<Output = Result<(), comm::rust::errors::CommError>> + Send>,
                    >
            })
        };

        // Build all server instances
        let servers = crate::rust::runtime::servers_instances::ServersInstances::build(
            api_servers,
            web_api,
            admin_web_api,
            grpc_packet_handler,
            grpc_blob_handler,
            &host,
            &address,
            self.node_conf.clone(),
            self.kamon_conf.clone(),
            rp_conf_cell.clone(),
            rp_connections.clone(),
            node_discovery.clone(),
            block_report_api,
            event_stream,
            kademlia_store.clone(),
        )
        .await?;

        info!("All servers started successfully");

        // Publish NodeStarted event
        let node_started_event =
            shared::rust::shared::f1r3fly_event::F1r3flyEvent::node_started(address.clone());
        event_bus
            .publish(node_started_event)
            .map_err(|e| eyre::eyre!("Failed to publish NodeStarted event: {}", e))?;

        info!("NodeStarted event published: {}", address);

        // Start all concurrent tasks with categorized failure handling
        // Critical tasks: Failure triggers immediate shutdown
        // Supportive tasks: Failure is logged as warning, node continues
        let mut critical_tasks: JoinSet<NamedTaskResult> = JoinSet::new();

        // === CRITICAL TASKS: Tier 1 - Server Infrastructure ===
        // All server failures are critical - if a server dies, the node cannot function properly

        info!("Starting critical server tasks...");

        // Transport server
        spawn_named_task(
            &mut critical_tasks,
            "Transport Server",
            await_http_server_task(servers.transport_server_handle, "Transport"),
        );

        // Kademlia server
        spawn_named_task(
            &mut critical_tasks,
            "Kademlia Server",
            await_server_task(servers.kademlia_server_handle, "Kademlia"),
        );

        // Node discovery loop (Tier 3: Supportive)
        // Start BEFORE wait_for_first_connection so it can help establish the first connection
        let nd_clone = node_discovery.clone();
        let rp_conn_clone = rp_connections.clone();
        let rp_conf_cell_clone = rp_conf_cell.clone();
        let transport_clone = transport.clone();

        spawn_named_task(&mut critical_tasks, "Node Discovery Loop", async move {
            node_discovery_loop(nd_clone, rp_conn_clone, rp_conf_cell_clone, transport_clone).await
        });

        // Clear connections loop (Tier 3: Supportive)
        // Start BEFORE wait_for_first_connection to match Scala connectivityStream behavior
        let rp_conn_clone2 = rp_connections.clone();
        let rp_conf_cell_clone2 = rp_conf_cell.clone();
        let transport_clone2 = transport.clone();
        let node_conf_clone2 = self.node_conf.clone();

        spawn_named_task(&mut critical_tasks, "Clear Connections Loop", async move {
            clear_connections_loop(
                rp_conn_clone2,
                rp_conf_cell_clone2,
                transport_clone2,
                node_conf_clone2,
            )
            .await
        });

        // Wait for first connection (unless standalone)
        // This runs in parallel with the connectivity tasks above (Transport Server, Kademlia Server,
        // Node Discovery Loop, Clear Connections Loop), matching Scala's connectivityStream behavior
        if !self.node_conf.standalone {
            info!("Waiting for first connection...");
            wait_for_first_connection(rp_connections.clone()).await?;
            info!("First connection established, starting engine tasks");
        } else {
            info!("Running in standalone mode, starting engine tasks immediately");
        }

        // Engine initialization (Tier 2: Critical - runs once)
        // started it as a separate task because it is not a long-running task and we want to keep the critical tasks separate.
        // Also running it as a separate task avoids warning log that critical task should run forever.
        let mut engine_init_handler = Some(tokio::spawn(async move {
            info!("Running engine initialization...");
            match engine_init().await {
                Ok(_) => {
                    info!("Engine initialization completed successfully");
                    Ok(())
                }
                Err(e) => {
                    tracing::error!("Engine initialization failed: {}", e);
                    Err(eyre::eyre!("Engine init failed: {}", e))
                }
            }
        }));

        // === CRITICAL TASKS: Tier 2 - Core Consensus Logic ===
        // Casper loop (Tier 2: Critical - runs indefinitely)
        spawn_named_task(&mut critical_tasks, "Casper Loop", async move {
            run_casper_loop(casper_loop).await
        });

        // Update fork choice loop (Tier 2: Critical - runs indefinitely)
        spawn_named_task(&mut critical_tasks, "Update Fork Choice Loop", async move {
            run_update_fork_choice_loop(update_fork_choice_loop).await
        });

        // Block processor instance (Tier 2: Critical)
        let trigger_propose_opt = if self.node_conf.autopropose {
            trigger_propose_f
        } else {
            None
        };

        let bpi_block_queue_tx = block_processor_queue_tx.clone();

        spawn_named_task(
            &mut critical_tasks,
            "Block Processor Instance",
            async move {
                use crate::rust::instances::block_processor_instance::BlockProcessorInstance;
                use dashmap::DashSet;

                info!("Starting block processor instance...");

                // Convert Arc<Mutex<HashSet>> to Arc<DashSet> for BlockProcessorInstance
                let blocks_in_processing = {
                    let hash_set = block_processor_state.lock().unwrap().clone();
                    let dash_set = DashSet::new();
                    for item in hash_set {
                        dash_set.insert(item);
                    }
                    Arc::new(dash_set)
                };

                let instance = BlockProcessorInstance::new(
                    (block_processor_queue_rx, bpi_block_queue_tx),
                    Arc::new(block_processor),
                    blocks_in_processing,
                    trigger_propose_opt,
                    10, // max_parallel_blocks - reasonable default
                );

                // BlockProcessorInstance::create spawns the processing task and returns a result receiver
                match instance.create() {
                    Ok(mut result_rx) => {
                        // Drain results (we're just logging for now)
                        while let Some(_result) = result_rx.recv().await {
                            // Results are logged inside block_processor_instance
                        }
                        info!("Block processor instance completed");
                        Ok(())
                    }
                    Err(e) => {
                        tracing::error!("Block processor instance failed: {}", e);
                        Err(eyre::eyre!("Block processor failed: {}", e))
                    }
                }
            },
        );

        // Proposer instance (Tier 2: Critical - if configured as validator)
        if let (Some(proposer), Some(proposer_state_ref)) = (proposer_opt, proposer_state_ref_opt) {
            spawn_named_task(&mut critical_tasks, "Proposer Instance", async move {
                use crate::rust::instances::proposer_instance::ProposerInstance;

                info!("Starting proposer instance...");

                // Wrap proposer in Arc<Mutex> for shared mutable access
                let proposer_arc = Arc::new(tokio::sync::Mutex::new(proposer));

                // Create proposer instance with state tracking for API observability
                // The state allows the API to check:
                // - Is a propose currently in progress? (curr_propose_result.is_some())
                // - What was the last propose result? (latest_propose_result)
                // Pass both receiver and sender as tuple
                let instance = ProposerInstance::new(
                    (proposer_queue_rx, proposer_queue_tx),
                    proposer_arc,
                    proposer_state_ref, // State for API observability
                );

                // Start the proposer stream - it will process propose requests as they arrive
                match instance.create() {
                    Ok(mut result_rx) => {
                        // Drain results (logged inside proposer_instance)
                        while let Some(_result) = result_rx.recv().await {
                            // Results are already logged inside ProposerInstance
                        }
                        info!("Proposer instance completed");
                        Ok(())
                    }
                    Err(e) => {
                        tracing::error!("Proposer instance failed to start: {}", e);
                        Err(eyre::eyre!("Proposer instance failed: {}", e))
                    }
                }
            });
        } else {
            info!("Node not configured as validator - proposer instance will not start");
        }

        // === CRITICAL TASKS: Tier 3 - API Servers ===
        // These are critical for the node to function properly

        info!("Starting API server tasks...");

        // External API server
        spawn_named_task(
            &mut critical_tasks,
            "External API Server",
            await_server_task(servers.external_api_server_handle, "External API"),
        );

        // Internal API server
        spawn_named_task(
            &mut critical_tasks,
            "Internal API Server",
            await_server_task(servers.internal_api_server_handle, "Internal API"),
        );

        // HTTP server
        spawn_named_task(
            &mut critical_tasks,
            "HTTP Server",
            await_http_server_task(servers.http_server_handle, "HTTP"),
        );

        // Admin HTTP server
        spawn_named_task(
            &mut critical_tasks,
            "Admin HTTP Server",
            await_http_server_task(servers.admin_http_server_handle, "Admin HTTP"),
        );

        info!("All server tasks started successfully");

        // Keep variables in scope to avoid unused warnings
        let _ = reporting_http_routes;

        // === Monitor both JoinSets for failures ===
        // Critical tasks: Any failure triggers immediate shutdown
        // Supportive tasks: Failures are logged as warnings, node continues
        info!("All tasks started. Node is now running.");

        loop {
            tokio::select! {
                // Monitor critical tasks - any failure is fatal
                Some(result) = critical_tasks.join_next() => {
                    match result {
                        Ok(named_result) => {
                            let task_name = named_result.name;
                            match named_result.result {
                                Ok(()) => {
                                    tracing::warn!(
                                        "Critical task '{}' completed unexpectedly (they should run forever)",
                                        task_name
                                    );
                                }
                                Err(e) => {
                                    tracing::error!("Critical task '{}' failed: {}", task_name, e);
                                    // Trigger shutdown
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Critical task panicked: {}", e);
                            // Trigger shutdown
                            break;
                        }
                    }
                }

                result = async {
                    match engine_init_handler.take() {
                        Some(handle) => Some(handle.await),
                        None => None,
                    }
                }, if engine_init_handler.is_some() => {
                    match result {
                        Some(Ok(Ok(_))) => {
                            continue;
                        }
                        Some(Ok(Err(e))) => {
                            tracing::error!("Engine initialization failed: {}", e);
                            // Engine init failure is critical - trigger shutdown
                            break;
                        }
                        Some(Err(e)) => {
                            tracing::error!("Engine initialization task panicked: {}", e);
                            // Task panic is critical - trigger shutdown
                            break;
                        }
                        None => {
                            // This shouldn't happen due to the guard, but handle it anyway
                            continue;
                        }
                    }
                }

                // Graceful shutdown signal (CTRL+C or SIGTERM)
                _ = shutdown_signal() => {
                    info!("Received shutdown signal, initiating graceful shutdown");
                    break;
                }

                // If all tasks complete (shouldn't happen), exit
                else => {
                    tracing::error!("All tasks completed - this should never happen");
                    break;
                }
            }
        }

        // === SHUTDOWN SEQUENCE ===
        info!("Shutting down all tasks...");

        // Step 1: Abort all tasks with 30-second timeout
        match tokio::time::timeout(std::time::Duration::from_secs(30), async {
            critical_tasks.shutdown().await;
        })
        .await
        {
            Ok(_) => info!("All tasks shut down gracefully"),
            Err(_) => {
                tracing::warn!("Shutdown timeout reached after 30 seconds, forcing termination");
            }
        }

        // Step 2: Perform resource cleanup (metrics, logging, etc.)
        // Block store cleanup happens automatically via Drop when block_store goes out of scope
        match Self::perform_shutdown_cleanup().await {
            Ok(_) => info!("Resource cleanup completed successfully"),
            Err(e) => {
                tracing::error!("Resource cleanup failed: {}", e);
                // Continue anyway - we've already stopped the tasks
            }
        }

        info!("Node shutdown complete");
        Ok(())
    }

    /// Perform shutdown cleanup
    ///
    /// Cleanup operations:
    /// - Stops metrics reporters (stubbed - coming in separate PR)
    /// - Logs shutdown messages
    ///
    /// Note: Block store cleanup is handled automatically by Rust's Drop implementation
    /// when variables go out of scope. This matches Scala's approach which relies on
    /// Resource finalizers rather than explicit close() calls.
    async fn perform_shutdown_cleanup() -> eyre::Result<()> {
        info!("Starting shutdown cleanup...");

        // Stop metrics reporters (TODO: coming in separate PR)
        info!("Metrics cleanup skipped (coming in separate PR)");

        // Block store cleanup is handled automatically by Drop implementation
        // when the block_store variable goes out of scope
        info!("Bringing BlockStore down ...");
        info!("Block store will be closed automatically via Drop implementation");

        info!("Goodbye.");

        Ok(())
    }
}

/// Wait for shutdown signal (CTRL+C or SIGTERM)
///
/// Handles:
/// - CTRL+C (cross-platform)
/// - SIGTERM (Unix only)
///
/// Returns when either signal is received
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received CTRL+C signal");
        },
        _ = terminate => {
            info!("Received SIGTERM signal");
        },
    }
}

/// Node discovery loop - runs indefinitely
///
/// Periodically discovers new peers and attempts to connect to them.
/// Runs every 20 seconds.
async fn node_discovery_loop(
    node_discovery: Arc<dyn comm::rust::discovery::node_discovery::NodeDiscovery + Send + Sync>,
    connections: comm::rust::rp::connect::ConnectionsCell,
    rp_conf_cell: comm::rust::rp::rp_conf::RPConfCell,
    transport: comm::rust::transport::grpc_transport_client::GrpcTransportClient,
) -> eyre::Result<()> {
    use tokio::time::{sleep, Duration};

    loop {
        // Discover new peers
        if let Err(e) = node_discovery.discover().await {
            tracing::warn!("Node discovery failed: {}", e);
        }

        // Read current RPConf (in case IP has changed)
        let rp_conf = match rp_conf_cell.read() {
            Ok(conf) => conf,
            Err(e) => {
                tracing::warn!("Failed to read RPConf: {}", e);
                sleep(Duration::from_secs(20)).await;
                continue;
            }
        };

        // Find and connect to new peers using the proper find_and_connect function
        use comm::rust::rp::connect;

        // Create the connect closure
        let connect_fn = |peer: &comm::rust::peer_node::PeerNode| {
            let conf = rp_conf.clone();
            let transport = transport.clone();
            let peer = peer.clone();
            async move { connect::connect(&peer, &conf, &transport).await }
        };

        // Use find_and_connect with trait object
        match connect::find_and_connect(&connections, node_discovery.as_ref(), connect_fn).await {
            Ok(new_connections) => {
                if !new_connections.is_empty() {
                    info!("Connected to {} new peer(s)", new_connections.len());
                }
            }
            Err(e) => {
                tracing::warn!("Find and connect failed: {}", e);
            }
        }

        // Sleep for 20 seconds before next iteration
        sleep(Duration::from_secs(20)).await;
    }
}

/// Clear connections loop - runs indefinitely
///
/// Periodically checks connection health by sending heartbeats and removes
/// failed peers. Also handles dynamic IP changes. Runs every 10 minutes.
async fn clear_connections_loop(
    connections: comm::rust::rp::connect::ConnectionsCell,
    rp_conf_cell: comm::rust::rp::rp_conf::RPConfCell,
    transport: comm::rust::transport::grpc_transport_client::GrpcTransportClient,
    node_conf: NodeConf,
) -> eyre::Result<()> {
    use tokio::time::{sleep, Duration};

    loop {
        // Read current RPConf
        let rp_conf = match rp_conf_cell.read() {
            Ok(conf) => conf,
            Err(e) => {
                tracing::warn!("Failed to read RPConf: {}", e);
                sleep(Duration::from_secs(600)).await;
                continue;
            }
        };

        // Dynamic IP check - detect and handle external IP address changes
        if node_conf.protocol_server.dynamic_ip {
            match comm::rust::who_am_i::check_local_peer_node(
                node_conf.protocol_server.port as u16,
                node_conf.peers_discovery.port as u16,
                &rp_conf.local,
            )
            .await
            {
                Ok(Some(new_local)) => {
                    info!(
                        "External IP address has changed to {}, updating RPConf",
                        new_local.to_address()
                    );

                    // Update RPConf with new local peer
                    if let Err(e) = rp_conf_cell.update_local(new_local.clone()) {
                        tracing::error!("Failed to update RPConf with new local peer: {}", e);
                    } else {
                        info!("Successfully updated RPConf with new local peer");

                        // Reset all connections since our address changed
                        if let Err(e) = comm::rust::rp::connect::reset_connections(&connections) {
                            tracing::warn!("Failed to reset connections: {}", e);
                        } else {
                            info!("Reset all connections after IP change");
                        }
                    }
                }
                Ok(None) => {
                    // IP hasn't changed, continue normally
                }
                Err(e) => {
                    tracing::warn!("Failed to check for IP change: {}", e);
                }
            }
        }

        // Re-read RPConf in case it was updated
        let rp_conf = match rp_conf_cell.read() {
            Ok(conf) => conf,
            Err(e) => {
                tracing::warn!("Failed to read RPConf after IP check: {}", e);
                sleep(Duration::from_secs(600)).await;
                continue;
            }
        };

        // Clear connections (send heartbeats, remove failed peers)
        match comm::rust::rp::connect::clear_connections(&connections, &rp_conf, &transport).await {
            Ok(cleared_count) => {
                if cleared_count > 0 {
                    info!("Cleared {} failed connection(s)", cleared_count);
                }
            }
            Err(e) => {
                tracing::warn!("Clear connections failed: {}", e);
            }
        }

        // Sleep for 10 minutes before next iteration
        sleep(Duration::from_secs(600)).await;
    }
}

/// Wait for the first peer connection
///
/// Polls the connections cell every second until at least one connection exists.
/// Only called when not in standalone mode.
async fn wait_for_first_connection(
    connections: comm::rust::rp::connect::ConnectionsCell,
) -> eyre::Result<()> {
    use tokio::time::{sleep, Duration};

    loop {
        sleep(Duration::from_secs(1)).await;

        let conns = connections
            .read()
            .map_err(|e| eyre::eyre!("Failed to read connections: {}", e))?;

        if !conns.is_empty() {
            return Ok(());
        }
    }
}

/// Run the Casper loop indefinitely
///
/// Periodically fetches dependencies and maintains requested blocks.
/// Errors are logged but don't stop the loop.
async fn run_casper_loop(casper_loop: CasperLoop) -> eyre::Result<()> {
    loop {
        match casper_loop().await {
            Ok(_) => {
                // Casper loop iteration completed successfully
            }
            Err(e) => {
                tracing::error!("Casper loop iteration failed: {}", e);
                // Sleep a bit before retrying to avoid tight error loops
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        }
    }
}

/// Run the update fork choice loop indefinitely
///
/// Periodically checks if the fork choice is stale and broadcasts a request
/// for updated tips if needed. Errors are logged but don't stop the loop.
async fn run_update_fork_choice_loop(update_fork_choice_loop: CasperLoop) -> eyre::Result<()> {
    loop {
        match update_fork_choice_loop().await {
            Ok(_) => {
                // Fork choice update iteration completed successfully
            }
            Err(e) => {
                tracing::error!("Update fork choice loop iteration failed: {}", e);
                // Sleep a bit before retrying to avoid tight error loops
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        }
    }
}

/// Helper function to await gRPC server task completion with proper error wrapping
///
/// Wraps a gRPC server `JoinHandle` to provide:
/// - Named server identification in logs
/// - Error type conversion (tonic::transport::Error -> eyre::Error)
/// - Graceful handling of task panics
///
/// # Arguments
/// * `handle` - The server's JoinHandle
/// * `server_name` - Human-readable name for logging
///
/// # Returns
/// Returns `Ok(())` if server completes successfully, or an error if it fails/panics
async fn await_server_task(
    handle: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
    server_name: &str,
) -> eyre::Result<()> {
    match handle.await {
        Ok(Ok(())) => {
            tracing::warn!(
                "{} server completed unexpectedly (should run forever)",
                server_name
            );
            Ok(())
        }
        Ok(Err(e)) => {
            let err = eyre::eyre!("{} server failed: {}", server_name, e);
            tracing::error!("{}", err);
            Err(err)
        }
        Err(e) => {
            let err = eyre::eyre!("{} server panicked: {}", server_name, e);
            tracing::error!("{}", err);
            Err(err)
        }
    }
}

/// Helper function to await HTTP server task completion with proper error wrapping
///
/// Wraps an HTTP server `JoinHandle` to provide:
/// - Named server identification in logs
/// - Error handling for eyre::Error
/// - Graceful handling of task panics
///
/// # Arguments
/// * `handle` - The server's JoinHandle
/// * `server_name` - Human-readable name for logging
///
/// # Returns
/// Returns `Ok(())` if server completes successfully, or an error if it fails/panics
async fn await_http_server_task(
    handle: tokio::task::JoinHandle<Result<(), eyre::Error>>,
    server_name: &str,
) -> eyre::Result<()> {
    match handle.await {
        Ok(Ok(())) => {
            tracing::warn!(
                "{} server completed unexpectedly (should run forever)",
                server_name
            );
            Ok(())
        }
        Ok(Err(e)) => {
            let err = eyre::eyre!("{} server failed: {}", server_name, e);
            tracing::error!("{}", err);
            Err(err)
        }
        Err(e) => {
            let err = eyre::eyre!("{} server panicked: {}", server_name, e);
            tracing::error!("{}", err);
            Err(err)
        }
    }
}

/// Start the node runtime
///
/// This is the primary entry point for starting an RChain node. It:
/// 1. Creates the node identifier from the TLS certificate
/// 2. Instantiates the NodeRuntime
/// 3. Runs the main node program
/// 4. Handles any unrecoverable errors
///
/// # Arguments
/// * `node_conf` - Node configuration
/// * `kamon_conf` - Kamon metrics configuration
///
/// # Returns
/// Returns `Ok(())` on successful node shutdown, or an error if initialization fails
pub async fn start(node_conf: NodeConf, kamon_conf: KamonConf) -> eyre::Result<()> {
    info!("Starting RChain node runtime...");

    // Create node identifier from certificate
    let id = node_environment::create(&node_conf).await?;

    info!("Node initialized with ID: {}", hex::encode(&id.key));

    // Create NodeRuntime instance
    let runtime = NodeRuntime::new(node_conf, kamon_conf, id);

    // Run the main node program with error handling
    handle_unrecoverable_errors(runtime.main()).await
}

/// Handle unrecoverable errors in the node program
///
/// Catches any errors that bubble up from the main program and logs them
/// before exiting. These are errors that should not happen in a properly
/// configured environment and mean immediate termination.
///
/// # Arguments
/// * `program` - The async program to run
///
/// # Returns
/// Returns `Ok(())` on successful completion, or logs the error and exits with code 1
async fn handle_unrecoverable_errors<F>(program: F) -> eyre::Result<()>
where
    F: Future<Output = eyre::Result<()>>,
{
    match program.await {
        Ok(_) => {
            info!("Node program completed successfully. Exiting.");
            std::process::exit(0);
        }
        Err(e) => {
            tracing::error!("Caught unhandable error. Exiting. Error: {:?}", e);
            std::process::exit(1);
        }
    }
}
