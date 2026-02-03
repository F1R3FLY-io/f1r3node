// See node/src/main/scala/coop/rchain/node/runtime/ServersInstances.scala

use std::net::IpAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::{future::Future, net::SocketAddr};

use crate::rust::{
    api::{
        admin_web_api::AdminWebApi,
        grpc_package::{acquire_external_server, acquire_internal_server},
        web_api::WebApi,
    },
    configuration::NodeConf,
    runtime::api_servers::APIServers,
    web::{routes::Routes, shared_handlers::AppState},
};
use comm::rust::discovery::kademlia_handle_rpc::{handle_lookup, handle_ping};
use comm::rust::discovery::kademlia_rpc::KademliaRPC;
use comm::rust::discovery::kademlia_store::KademliaStore;
use comm::rust::peer_node::PeerNode;
use comm::rust::{
    discovery::{node_discovery::NodeDiscovery, utils::acquire_kademlia_rpc_server},
    rp::connect::ConnectionsCell,
    transport::grpc_transport_server::{DispatchFn, HandleStreamedFn, TransportServer},
};
use futures::FutureExt;
use shared::rust::grpc::grpc_server::GrpcServer;
use shared::rust::shared::f1r3fly_events::EventStream;
use std::str::FromStr;
use tokio::task::JoinHandle;
use tracing::{error, info};

/// Container for all servers Node provides
pub struct ServersInstances {
    // Server instances for control/inspection (backward compatible)
    pub transport_server: Arc<TransportServer>,
    pub kademlia_server: GrpcServer,
    pub external_api_server: GrpcServer,
    pub internal_api_server: GrpcServer,

    // Lifecycle management handles for monitoring server health
    pub transport_server_handle: JoinHandle<Result<(), eyre::Error>>,
    pub kademlia_server_handle: JoinHandle<Result<(), tonic::transport::Error>>,
    pub external_api_server_handle: JoinHandle<Result<(), tonic::transport::Error>>,
    pub internal_api_server_handle: JoinHandle<Result<(), tonic::transport::Error>>,
    pub http_server_handle: JoinHandle<Result<(), eyre::Error>>,
    pub admin_http_server_handle: JoinHandle<Result<(), eyre::Error>>,
}

impl ServersInstances {
    /// Build all server instances
    ///
    /// # Arguments
    /// * `api_servers` - API service implementations
    /// * `web_api` - Web API implementation
    /// * `admin_web_api` - Admin Web API implementation
    /// * `grpc_packet_handler` - Handler for gRPC protocol messages
    /// * `grpc_stream_handler` - Handler for gRPC streamed blob messages
    /// * `host` - Host address for servers
    /// * `address` - Address string for logging
    /// * `node_conf` - Node configuration
    /// * `rp_conf` - RChain Protocol configuration (needed for transport server)
    /// * `rp_connections` - Connections cell (needed for AppState)
    /// * `node_discovery` - Node discovery service (needed for AppState)
    /// * `block_report_api` - Block report API (needed for AppState)
    /// * `event_stream` - Event stream (needed for AppState)
    /// * `kademlia_store` - Kademlia store (needed for Kademlia server)
    pub async fn build<T: KademliaRPC + Send + Sync + 'static>(
        api_servers: APIServers,
        web_api: Arc<dyn WebApi + Send + Sync + 'static>,
        admin_web_api: Arc<dyn AdminWebApi + Send + Sync + 'static>,
        grpc_packet_handler: DispatchFn,
        grpc_stream_handler: HandleStreamedFn,
        host: &str,
        address: &str,
        node_conf: NodeConf,
        rp_conf_cell: comm::rust::rp::rp_conf::RPConfCell,
        rp_connections: ConnectionsCell,
        node_discovery: Arc<dyn NodeDiscovery + Send + Sync>,
        block_report_api: Arc<casper::rust::api::block_report_api::BlockReportAPI>,
        event_stream: EventStream,
        kademlia_store: Arc<KademliaStore<T>>,
    ) -> eyre::Result<Self> {
        // Read current RPConf
        let rp_conf = rp_conf_cell
            .read()
            .map_err(|e| eyre::eyre!("Failed to read RPConf: {}", e))?;

        // Acquire and start transport server
        let transport_server =
            comm::rust::transport::grpc_transport_server::GrpcTransportServer::acquire_server(
                node_conf.protocol_server.network_id.clone(),
                node_conf.protocol_server.port,
                &node_conf.tls.certificate_path,
                &node_conf.tls.key_path,
                node_conf.protocol_server.grpc_max_recv_message_size as i32,
                node_conf.protocol_server.grpc_max_recv_stream_message_size as u64,
                node_conf.protocol_server.max_message_consumers as usize,
                rp_conf.clone(),
            )
            .await
            .map_err(|e| eyre::eyre!("Failed to acquire transport server: {}", e))?;

        // Start transport server
        transport_server
            .start(grpc_packet_handler.clone(), grpc_stream_handler.clone())
            .await
            .map_err(|e| eyre::eyre!("Failed to start transport server: {}", e))?;

        info!("Listening for traffic on {}.", address);

        let kademlia_store_clone = kademlia_store.clone();

        // Acquire and start Kademlia server
        let mut kademlia_server = acquire_kademlia_rpc_server(
            node_conf.protocol_server.network_id.clone(),
            node_conf.peers_discovery.port,
            Box::new(move |sender_peer: PeerNode| {
                let kademlia_store = kademlia_store_clone.clone();
                Box::pin(async move {
                    match handle_ping(sender_peer, kademlia_store.clone(), None).await {
                        Ok(_) => (),
                        Err(e) => error!("Kademlia ping error: {}", e),
                    }
                }) as Pin<Box<dyn Future<Output = ()> + Send>>
            }),
            Box::new(move |peer_node: PeerNode, key: Vec<u8>| {
                let kademlia_store = kademlia_store.clone();
                async move {
                    match handle_lookup(peer_node, key, kademlia_store.clone(), None).await {
                        Ok(peers) => peers,
                        Err(e) => {
                            error!("Kademlia lookup error: {}", e);
                            Vec::new()
                        }
                    }
                }
                .boxed()
            }),
        )
        .await
        .map_err(|e| eyre::eyre!("Failed to acquire Kademlia RPC server: {}", e))?;

        info!(
            "Kademlia RPC server started at {}:{}",
            host,
            kademlia_server.port()
        );

        // Acquire external API server router
        let external_api_router = acquire_external_server(
            api_servers.deploy.clone(),
            node_conf.api_server.grpc_max_recv_message_size as usize,
            node_conf.api_server.keep_alive_time,
            node_conf.api_server.keep_alive_timeout,
            node_conf.api_server.permit_keep_alive_time,
            node_conf.api_server.max_connection_idle,
            node_conf.api_server.max_connection_age,
            node_conf.api_server.max_connection_age_grace,
        )
        .map_err(|e| eyre::eyre!("Failed to acquire external API server: {}", e))?;

        // Create and start external API server
        let mut external_api_server = GrpcServer::new(node_conf.api_server.port_grpc_external);
        external_api_server
            .start_with_router(external_api_router)
            .await
            .map_err(|e| eyre::eyre!("Failed to start external API server: {}", e))?;

        info!(
            "External API server started at {}:{}",
            node_conf.api_server.host,
            external_api_server.port()
        );

        // Acquire internal API server router
        let internal_api_router = acquire_internal_server(
            api_servers.repl.clone(),
            api_servers.deploy.clone(),
            api_servers.propose.clone(),
            api_servers.lsp.clone(),
            node_conf.api_server.grpc_max_recv_message_size as usize,
            node_conf.api_server.keep_alive_time,
            node_conf.api_server.keep_alive_timeout,
            node_conf.api_server.permit_keep_alive_time,
            node_conf.api_server.max_connection_idle,
            node_conf.api_server.max_connection_age,
            node_conf.api_server.max_connection_age_grace,
        )
        .await
        .map_err(|e| eyre::eyre!("Failed to acquire internal API server: {}", e))?;

        // Create and start internal API server
        let mut internal_api_server = GrpcServer::new(node_conf.api_server.port_grpc_internal);
        internal_api_server
            .start_with_router(internal_api_router)
            .await
            .map_err(|e| eyre::eyre!("Failed to start internal API server: {}", e))?;

        info!(
            "Internal API server started at {}:{}",
            host,
            internal_api_server.port()
        );

        // Create AppState for HTTP servers
        let app_state = AppState::new(
            admin_web_api.clone(),
            web_api.clone(),
            block_report_api.clone(),
            rp_conf_cell.clone(),
            Arc::new(rp_connections),
            node_discovery.clone(),
            Arc::new(event_stream.new_subscribe()),
        );

        // Create HTTP server router
        let http_router = Routes::create_main_routes(node_conf.api_server.enable_reporting)
            .with_state(app_state.clone());

        // Start HTTP server

        let ip_http_addr = IpAddr::from_str(&node_conf.api_server.host)
            .map_err(|e| eyre::eyre!("Invalid HTTP server address: {}", e))?;

        let http_addr = SocketAddr::from((ip_http_addr, node_conf.api_server.port_http));

        let http_server_handle = tokio::spawn(async move {
            let listener = tokio::net::TcpListener::bind(&http_addr)
                .await
                .map_err(|e| eyre::eyre!("Failed to bind HTTP server: {}", e))?;

            axum::serve(listener, http_router)
                .await
                .map_err(|e| eyre::eyre!("HTTP server error: {}", e))?;

            Ok(())
        });

        info!(
            "HTTP API server started at {}:{}",
            node_conf.api_server.host, node_conf.api_server.port_http
        );

        // Create admin HTTP server router
        let admin_http_router = Routes::create_admin_routes().with_state(app_state);

        let ip_admin_http = IpAddr::from_str(&node_conf.api_server.host)
            .map_err(|e| eyre::eyre!("Invalid HTTP server address: {}", e))?;
        // Start admin HTTP server
        let admin_http_addr =
            SocketAddr::from((ip_admin_http, node_conf.api_server.port_admin_http));

        let admin_http_server_handle = tokio::spawn(async move {
            let listener = tokio::net::TcpListener::bind(&admin_http_addr)
                .await
                .map_err(|e| eyre::eyre!("Failed to bind admin HTTP server: {}", e))?;

            axum::serve(listener, admin_http_router)
                .await
                .map_err(|e| eyre::eyre!("Admin HTTP server error: {}", e))?;

            Ok(())
        });

        info!(
            "Admin HTTP API server started at {}:{}",
            node_conf.api_server.host, node_conf.api_server.port_admin_http
        );

        // TODO: Configure Kamon metrics
        // In Scala: Kamon.reconfigure(kamonConf.withFallback(Kamon.config()))
        // In Scala: Kamon.addReporter(...) for various reporters
        // For now, we'll skip Kamon setup as it's Java/Scala specific
        // Metrics can be configured separately if needed

        // Extract lifecycle handles from gRPC servers
        // Note: After taking the handle, the GrpcServer will no longer manage its own lifecycle

        let transport_server_arc = Arc::new(transport_server);

        // Create transport server monitor handle
        let transport_server_monitor = transport_server_arc.clone();
        let transport_server_handle = tokio::spawn(async move {
            // Monitor the transport server's running state
            match transport_server_monitor.get_monitor_handle().await {
                Some(handle) => handle
                    .await
                    .map_err(|e| eyre::eyre!("Transport server monitor failed: {}", e)),
                None => Err(eyre::eyre!("Transport server not running")),
            }
        });

        let kademlia_server_handle = kademlia_server
            .take_handle()
            .ok_or_else(|| eyre::eyre!("Kademlia server not running"))?;

        let external_api_server_handle = external_api_server
            .take_handle()
            .ok_or_else(|| eyre::eyre!("External API server not running"))?;

        let internal_api_server_handle = internal_api_server
            .take_handle()
            .ok_or_else(|| eyre::eyre!("Internal API server not running"))?;

        Ok(Self {
            // Server instances for control/inspection
            transport_server: transport_server_arc,
            kademlia_server,
            external_api_server,
            internal_api_server,

            // Lifecycle handles for monitoring
            transport_server_handle,
            kademlia_server_handle,
            external_api_server_handle,
            internal_api_server_handle,
            http_server_handle,
            admin_http_server_handle,
        })
    }
}
