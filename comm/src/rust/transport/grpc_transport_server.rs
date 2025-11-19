// See comm/src/main/scala/coop/rchain/comm/transport/GrpcTransportServer.scala

use async_trait::async_trait;
use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, OnceCell};
use tokio::task::JoinHandle;

use models::routing::Protocol;

use crate::rust::{
    errors::CommError,
    metrics_constants::{DISPATCHED_MESSAGES_METRIC, DISPATCHED_PACKETS_METRIC, TRANSPORT_METRICS_SOURCE},
    peer_node::PeerNode,
    rp::rp_conf::RPConf,
    transport::{
        communication_response::CommunicationResponse,
        grpc_transport_receiver::{GrpcTransportReceiver, MessageBuffers, MessageHandlers},
        packet_ops::StreamCache,
        stream_handler::StreamHandler,
        transport_layer::Blob,
    },
};

/// Cancelable resource that can be cancelled/stopped
pub type Cancelable = JoinHandle<()>;

/// Type alias for dispatch function that handles Protocol messages
pub type DispatchFn = Arc<
    dyn Fn(
            Protocol,
        )
            -> Pin<Box<dyn Future<Output = Result<CommunicationResponse, CommError>> + Send>>
        + Send
        + Sync,
>;

/// Type alias for stream handler function that processes Blob messages  
pub type HandleStreamedFn =
    Arc<dyn Fn(Blob) -> Pin<Box<dyn Future<Output = Result<(), CommError>> + Send>> + Send + Sync>;

/// Transport Layer Server trait for handling incoming gRPC connections
#[async_trait]
pub trait TransportLayerServer {
    /// Set up message handlers and start receiving connections
    ///
    /// This method configures the server with handlers for different types of messages
    /// and starts listening for incoming gRPC connections.
    async fn handle_receive(
        &self,
        dispatch: DispatchFn,
        handle_streamed: HandleStreamedFn,
    ) -> Result<Cancelable, CommError>;
}

/// GrpcTransportServer - gRPC server implementation for handling transport layer messages
#[derive(Debug, Clone)]
pub struct GrpcTransportServer {
    /// Local peer node
    pub local_peer: RPConf,
    /// Network identifier for this node
    pub network_id: String,
    /// Port number to bind the gRPC server to
    pub port: u16,
    /// TLS certificate in PEM format
    pub cert: String,
    /// TLS private key in PEM format
    pub key: String,
    /// Maximum message size for regular gRPC messages in bytes
    pub max_message_size: i32,
    /// Maximum message size for streamed messages in bytes
    pub max_stream_message_size: u64,
    /// Number of parallel message processing tasks
    pub parallelism: usize,
    /// Cache to store received partial data (streaming packets)
    pub cache: StreamCache,
}

impl GrpcTransportServer {
    /// Create a new GrpcTransportServer instance
    pub fn new(
        local_peer: RPConf,
        network_id: String,
        port: u16,
        cert: String,
        key: String,
        max_message_size: i32,
        max_stream_message_size: u64,
        parallelism: usize,
    ) -> Self {
        Self {
            local_peer,
            network_id,
            port,
            cert,
            key,
            max_message_size,
            max_stream_message_size,
            parallelism,
            // Create cache for storing received partial data (streaming packets)
            cache: Arc::new(dashmap::DashMap::new()),
        }
    }

    /// Create a GrpcTransportServer by reading certificate and key from files
    pub async fn acquire_server(
        network_id: String,
        port: u16,
        cert_path: &Path,
        key_path: &Path,
        max_message_size: i32,
        max_stream_message_size: u64,
        parallelism: usize,
        local_peer: RPConf,
    ) -> Result<TransportServer, CommError> {
        // Read certificate file
        let cert = tokio::fs::read_to_string(cert_path).await.map_err(|e| {
            CommError::ConfigError(format!("Failed to read cert file {:?}: {}", cert_path, e))
        })?;

        // Read key file
        let key = tokio::fs::read_to_string(key_path).await.map_err(|e| {
            CommError::ConfigError(format!("Failed to read key file {:?}: {}", key_path, e))
        })?;

        let server = Self::new(
            local_peer,
            network_id,
            port,
            cert,
            key,
            max_message_size,
            max_stream_message_size,
            parallelism,
        );

        Ok(TransportServer::new(server))
    }
}

/// TransportServer manages the lifecycle of a GrpcTransportServer
///
/// Provides atomic start/stop operations to ensure only one server instance
/// runs at a time, matching the Scala implementation's behavior.
pub struct TransportServer {
    server: GrpcTransportServer,
    server_handle: Arc<Mutex<Option<Cancelable>>>,
    is_running: Arc<AtomicBool>,
}

impl TransportServer {
    /// Create a new TransportServer
    pub fn new(server: GrpcTransportServer) -> Self {
        Self {
            server,
            server_handle: Arc::new(Mutex::new(None)),
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the server with the given message handlers
    ///
    /// This method is idempotent - if the server is already running, it returns immediately.
    /// If a previous server instance exists, it will be stopped before starting the new one.
    pub async fn start(
        &self,
        dispatch: DispatchFn,
        handle_streamed: HandleStreamedFn,
    ) -> Result<(), CommError> {
        // Check if already running
        if self.is_running.load(Ordering::Acquire) {
            return Ok(());
        }

        // Get exclusive access to server handle
        let mut handle_guard = self.server_handle.lock().await;

        // Double-check pattern: verify still not running after acquiring lock
        if self.is_running.load(Ordering::Acquire) {
            return Ok(());
        }

        // Stop any existing server
        if let Some(old_handle) = handle_guard.take() {
            old_handle.abort();
        }

        // Start new server
        let new_handle = self
            .server
            .handle_receive(dispatch, handle_streamed)
            .await?;

        // Update state atomically
        *handle_guard = Some(new_handle);
        self.is_running.store(true, Ordering::Release);

        Ok(())
    }

    /// Stop the server
    ///
    /// This method is idempotent - if the server is not running, it returns immediately.
    pub async fn stop(&self) -> Result<(), CommError> {
        // Check if not running
        if !self.is_running.load(Ordering::Acquire) {
            return Ok(());
        }

        // Get exclusive access to server handle
        let mut handle_guard = self.server_handle.lock().await;

        // Double-check pattern: verify still running after acquiring lock
        if !self.is_running.load(Ordering::Acquire) {
            return Ok(());
        }

        // Stop server if running
        if let Some(handle) = handle_guard.take() {
            handle.abort();
        }

        // Update state atomically
        self.is_running.store(false, Ordering::Release);

        Ok(())
    }

    /// Check if the server is currently running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Get a JoinHandle for monitoring the server's lifecycle
    ///
    /// Returns a new JoinHandle that will complete when the server stops.
    /// This allows external code to monitor the server without taking ownership
    /// of the internal handle, preserving the server's ability to manage itself.
    ///
    /// Returns None if the server is not currently running.
    pub async fn get_monitor_handle(&self) -> Option<JoinHandle<()>> {
        let handle_guard = self.server_handle.lock().await;
        if handle_guard.is_some() {
            // Clone the Arc to the handle and create a monitoring task
            let is_running = self.is_running.clone();
            let monitor = tokio::spawn(async move {
                // Wait for the server to stop
                while is_running.load(Ordering::Acquire) {
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            });
            Some(monitor)
        } else {
            None
        }
    }
}

#[async_trait]
impl TransportLayerServer for GrpcTransportServer {
    /// Set up message handlers and start receiving connections
    ///
    /// This method configures the server with handlers for different types of messages
    /// and starts listening for incoming gRPC connections.
    async fn handle_receive(
        &self,
        dispatch: DispatchFn,
        handle_streamed: HandleStreamedFn,
    ) -> Result<Cancelable, CommError> {
        // Create buffers map for per-peer message queues
        let buffers_map: Arc<Mutex<HashMap<PeerNode, Arc<OnceCell<MessageBuffers>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Create message handlers that bridge to the provided dispatch functions
        let message_handlers: MessageHandlers = (
            // Handler for Send messages (Protocol messages)
            Arc::new({
                let dispatch = dispatch.clone();
                move |send_msg| {
                    let protocol = send_msg.msg;
                    let dispatch = dispatch.clone();

                    Box::pin(async move {
                        // Execute the dispatch function
                        match dispatch(protocol).await {
                            Ok(_communication_response) => {
                                metrics::counter!(DISPATCHED_MESSAGES_METRIC, "source" => TRANSPORT_METRICS_SOURCE).increment(1);
                                Ok(())
                            }
                            Err(e) => {
                                log::error!("Sending gRPC message failed: {}", e);
                                Err(e)
                            }
                        }
                    })
                }
            }),
            // Handler for StreamMessage (Blob streaming)
            Arc::new({
                let handle_streamed = handle_streamed.clone();
                let cache = self.cache.clone();
                move |stream_msg| {
                    let cache = cache.clone();
                    let handle_streamed = handle_streamed.clone();

                    Box::pin(async move {
                        match StreamHandler::restore(&stream_msg, &cache).await {
                            Ok(blob) => {
                                // Execute the stream handler function
                                match handle_streamed(blob).await {
                                    Ok(()) => {
                                        metrics::counter!(DISPATCHED_PACKETS_METRIC, "source" => TRANSPORT_METRICS_SOURCE).increment(1);
                                        Ok(())
                                    }
                                    Err(e) => {
                                        log::error!("Error in stream handler: {}", e);
                                        Err(e)
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!(
                                    "Could not restore data from file while handling stream for key {}: {}",
                                    stream_msg.key,
                                    e
                                );
                                Err(e)
                            }
                        }
                    })
                }
            }),
        );

        // Create and start the gRPC transport receiver
        let server_task = GrpcTransportReceiver::create(
            self.network_id.clone(),
            self.local_peer.clone(),
            self.port,
            self.cert.clone(),
            self.key.clone(),
            self.max_message_size,
            self.max_stream_message_size,
            buffers_map,
            message_handlers,
            self.parallelism,
            self.cache.clone(),
        )
        .await?;

        Ok(server_task)
    }
}
