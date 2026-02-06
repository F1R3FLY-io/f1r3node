// See comm/src/main/scala/coop/rchain/comm/transport/GrpcTransportClient.scala

use async_trait::async_trait;
use hex;
use std::collections::HashMap;
use std::error::Error as StdError;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, OnceCell};
use tonic::service::interceptor::InterceptedService;
use tonic::transport::Channel;

use crate::rust::{
    errors::CommError,
    peer_node::PeerNode,
    transport::{
        f1r3fly_connector::F1r3flyConnector,
        grpc_transport::GrpcTransport,
        packet_ops::PacketOps,
        ssl_session_client_interceptor::SslSessionClientInterceptor,
        stream_observable::StreamObservable,
        transport_layer::{Blob, TransportLayer},
    },
    utils::resolve_hostname_to_ip,
};

use models::routing::{transport_layer_client::TransportLayerClient, Protocol};

type StreamCache = Arc<dashmap::DashMap<String, Vec<u8>>>;

/// GRPC channel with a message buffer protecting it from resource exhaustion
#[derive(Debug)]
pub struct BufferedGrpcStreamChannel {
    /// Underlying gRPC channel
    pub grpc_transport: Channel,
    /// Pre-created transport client with SSL interceptor applied (for reuse)
    pub transport_client: Arc<
        tokio::sync::Mutex<
            TransportLayerClient<InterceptedService<Channel, SslSessionClientInterceptor>>,
        >,
    >,
    /// Buffer implementing some kind of overflow policy
    pub buffer: StreamObservable,
    /// Buffer subscriber handle
    pub buffer_subscriber: tokio::task::JoinHandle<()>,
    /// Max message size (to be applied to individual service clients)
    pub max_message_size: i32,
}

impl BufferedGrpcStreamChannel {
    /// Create a new BufferedGrpcStreamChannel with pre-created client
    pub fn new(
        grpc_transport: Channel,
        transport_client: TransportLayerClient<
            InterceptedService<Channel, SslSessionClientInterceptor>,
        >,
        buffer: StreamObservable,
        buffer_subscriber: tokio::task::JoinHandle<()>,
        max_message_size: i32,
    ) -> Self {
        Self {
            grpc_transport,
            transport_client: Arc::new(tokio::sync::Mutex::new(transport_client)),
            buffer,
            buffer_subscriber,
            max_message_size,
        }
    }

    /// Get a clone of the pre-created transport client (for use in tasks)
    pub fn get_transport_client(
        &self,
    ) -> Arc<
        tokio::sync::Mutex<
            TransportLayerClient<InterceptedService<Channel, SslSessionClientInterceptor>>,
        >,
    > {
        self.transport_client.clone()
    }
}

/// GrpcTransportClient - gRPC client implementation
#[derive(Clone)]
pub struct GrpcTransportClient {
    network_id: String,
    cert: String,
    key: String,
    max_message_size: i32,
    packet_chunk_size: i32,
    client_queue_size: i32,
    channels_map: Arc<Mutex<HashMap<PeerNode, Arc<OnceCell<Arc<BufferedGrpcStreamChannel>>>>>>,
    default_send_timeout: Duration,
    cache: StreamCache,
}

impl GrpcTransportClient {
    /// Create a new GrpcTransportClient
    pub fn new(
        network_id: String,
        cert: String,
        key: String,
        max_message_size: i32,
        packet_chunk_size: i32,
        client_queue_size: i32,
        channels_map: Arc<Mutex<HashMap<PeerNode, Arc<OnceCell<Arc<BufferedGrpcStreamChannel>>>>>>,
        network_timeout: Duration,
    ) -> Result<Self, CommError> {
        Ok(Self {
            network_id,
            cert,
            key,
            max_message_size,
            packet_chunk_size,
            client_queue_size,
            channels_map,
            default_send_timeout: network_timeout,
            cache: Arc::new(dashmap::DashMap::new()),
        })
    }

    async fn create_channel(
        &self,
        peer: &PeerNode,
    ) -> Result<BufferedGrpcStreamChannel, CommError> {
        tracing::info!("Creating new F1r3fly channel to peer {}", peer.to_address());

        // **F1r3fly Custom TLS Integration Architecture**
        // This method creates tonic gRPC channels using F1r3flyConnector with connect_with_connector()
        // providing direct integration of F1r3fly TLS verification with tonic's gRPC layer

        // Step 1: Create F1r3flyConnector with peer's F1r3fly address for TLS hostname verification
        let f1r3fly_id_hex = hex::encode(&peer.id.key);
        tracing::debug!(
            "Creating F1r3flyConnector with F1r3fly address for TLS hostname: {}",
            f1r3fly_id_hex
        );

        let f1r3fly_connector = F1r3flyConnector::new(
            self.network_id.clone(),
            &self.cert,
            &self.key,
            f1r3fly_id_hex.clone(),
        )
        .map_err(|e| CommError::ConfigError(format!("Failed to create F1r3flyConnector: {}", e)))?;

        let uri_address = resolve_hostname_to_ip(&peer.endpoint.host, peer.endpoint.tcp_port)
            .await?
            .ip()
            .to_string();

        // Step 2: Create tonic Endpoint with HTTP scheme (not HTTPS)
        // since F1r3flyConnector handles TLS internally
        let endpoint_uri = format!("http://{}:{}/", uri_address, peer.endpoint.tcp_port);
        tracing::debug!(
            "Creating F1r3fly gRPC channel to {} with TLS hostname verification against: {}",
            endpoint_uri,
            f1r3fly_id_hex
        );

        let endpoint = Channel::from_shared(endpoint_uri.clone()).map_err(|e| {
            tracing::error!(
                "Failed to create gRPC endpoint: {} {} {}",
                e,
                endpoint_uri,
                peer.endpoint.host
            );
            CommError::InternalCommunicationError(format!("Invalid endpoint URI: {}", e))
        })?;

        // Step 3: Use F1r3flyConnector with tonic's connect_with_connector API
        // The F1r3flyConnector will handle TLS hostname verification against the F1r3fly address
        let grpc_channel = endpoint
            .connect_with_connector(f1r3fly_connector)
            .await
            .map_err(|e| {
                tracing::error!(
                    "Failed to connect with F1r3flyConnector to {}: {}",
                    endpoint_uri,
                    e
                );
                if let Some(source) = e.source() {
                    tracing::error!("Error source: {}", source);
                }
                CommError::InternalCommunicationError(format!(
                    "Failed to establish gRPC connection: {}",
                    e
                ))
            })?;

        tracing::info!("gRPC channel created for {}", peer.to_address());

        // Step 4: Create SSL session interceptor for application-level validation
        let ssl_interceptor = SslSessionClientInterceptor::new(self.network_id.clone());
        let intercepted_channel = InterceptedService::new(grpc_channel.clone(), ssl_interceptor);

        // Step 5: Create transport client with interceptor
        let transport_client = TransportLayerClient::new(intercepted_channel)
            .max_encoding_message_size(self.max_message_size as usize)
            .max_decoding_message_size(self.max_message_size as usize);

        // Step 6: Create stream buffer and handler
        let mut buffer = StreamObservable::new(
            peer.clone(),
            self.client_queue_size as usize,
            self.cache.clone(),
        );

        // Create buffer subscriber
        let buffer_subscriber = {
            let peer_clone = peer.clone();
            let cache_clone = self.cache.clone();
            let network_id = self.network_id.clone();
            let default_send_timeout = self.default_send_timeout;
            let packet_chunk_size = self.packet_chunk_size;

            // Get subscription from buffer
            let mut subscription = buffer.subscribe().ok_or_else(|| {
                CommError::InternalCommunicationError(
                    "Failed to create stream subscription".to_string(),
                )
            })?;

            // Clone the transport client for use in spawned task
            let client_for_task = Arc::new(tokio::sync::Mutex::new(transport_client.clone()));

            // Spawn task to handle stream messages
            tokio::spawn(async move {
                use tokio_stream::StreamExt;

                while let Some(stream_msg) = subscription.next().await {
                    // Call static stream_blob_file_with_client method with pre-created client
                    let result = Self::stream_blob_file_with_client(
                        &stream_msg.key,
                        &peer_clone,
                        &stream_msg.sender,
                        &cache_clone,
                        &network_id,
                        default_send_timeout,
                        packet_chunk_size,
                        client_for_task.clone(),
                    )
                    .await;

                    // Log any errors but continue processing
                    if let Err(e) = result {
                        tracing::debug!(
                            "Error in stream_blob_file for key {}: {}",
                            stream_msg.key,
                            e
                        );
                    }

                    // Clean up cache
                    // This happens regardless of success/failure
                    cache_clone.remove(&stream_msg.key);
                }
            })
        };

        Ok(BufferedGrpcStreamChannel::new(
            grpc_channel,
            transport_client,
            buffer,
            buffer_subscriber,
            self.max_message_size,
        ))
    }

    /// Get or create a channel for the specified peer
    ///
    /// Uses atomic operations to ensure only one channel is created per peer,
    /// with proper cleanup and retry logic for terminated channels.
    async fn get_channel(
        &self,
        peer: &PeerNode,
    ) -> Result<Arc<BufferedGrpcStreamChannel>, CommError> {
        loop {
            // Create a new OnceCell for potential new channel
            let new_once_cell = Arc::new(OnceCell::new());

            // Atomic operation: check if peer exists, if not add new OnceCell
            let (once_cell, is_new_channel) = {
                let mut channels_map = self.channels_map.lock().await;

                if let Some(existing_once_cell) = channels_map.get(peer) {
                    // Peer exists, use existing OnceCell
                    (existing_once_cell.clone(), false)
                } else {
                    // Peer doesn't exist, add new OnceCell
                    channels_map.insert(peer.clone(), new_once_cell.clone());
                    (new_once_cell, true)
                }
            };

            // If this is a new channel, create it and store in OnceCell
            if is_new_channel {
                let channel = self.create_channel(peer).await?;
                let _ = once_cell.set(Arc::new(channel));
            }

            // Get the channel from OnceCell (wait if another thread is creating it)
            let channel = once_cell
                .get_or_try_init(|| async {
                    // This should not happen if logic is correct, but just in case
                    let ch = self.create_channel(peer).await?;
                    Ok(Arc::new(ch))
                })
                .await?;

            // Check if channel is terminated and handle cleanup/retry
            if Self::is_channel_terminated(channel).await {
                tracing::debug!(
                    "Channel to peer {} is terminated; removing from connections map",
                    peer.to_address()
                );

                // Clean up: cancel subscriber and remove from map
                channel.buffer_subscriber.abort();

                {
                    let mut channels_map = self.channels_map.lock().await;
                    channels_map.remove(peer);
                }

                // Retry by continuing the loop
                continue;
            }

            // Channel is valid, return it
            return Ok(channel.clone());
        }
    }

    /// Execute a request with a gRPC client, handling timeouts and errors
    async fn with_client<A, F, Fut>(
        &self,
        peer: &PeerNode,
        timeout: Duration,
        request: F,
    ) -> Result<A, CommError>
    where
        F: FnOnce(
            TransportLayerClient<InterceptedService<Channel, SslSessionClientInterceptor>>,
        ) -> Fut,
        Fut: std::future::Future<Output = Result<A, CommError>>,
    {
        // Apply timeout to the entire operation
        let timed_operation = tokio::time::timeout(timeout, async {
            let channel = self.get_channel(peer).await?;

            let client_guard = channel.transport_client.lock().await;
            let client = client_guard.clone(); // Clone the client for use
            drop(client_guard); // Release the lock immediately

            let result = request(client).await?;

            // Return control to caller thread
            // In Rust, this is handled automatically by the async runtime
            tokio::task::yield_now().await;

            Ok::<A, CommError>(result)
        });

        // Handle timeout and other errors
        match timed_operation.await {
            Ok(Ok(success)) => Ok(success),
            Ok(Err(comm_error)) => {
                tracing::error!(
                    "Request failed for peer {}: {}",
                    peer.to_address(),
                    comm_error
                );
                Err(comm_error)
            }
            Err(_timeout_error) => {
                let timeout_error = crate::rust::errors::protocol_exception(format!(
                    "Request to {} timed out after {}ms",
                    peer.to_address(),
                    timeout.as_millis()
                ));
                tracing::error!("Request timeout: {}", timeout_error);
                Err(timeout_error)
            }
        }
    }

    /// Stream a blob file from cache to a peer using pre-created client
    ///
    /// This method uses a pre-created transport client, eliminating the need for
    /// complex connection management in spawned tasks.
    async fn stream_blob_file_with_client(
        key: &str,
        peer: &PeerNode,
        sender: &PeerNode,
        cache: &StreamCache,
        network_id: &str,
        default_send_timeout: Duration,
        packet_chunk_size: i32,
        client: Arc<
            tokio::sync::Mutex<
                TransportLayerClient<InterceptedService<Channel, SslSessionClientInterceptor>>,
            >,
        >,
    ) -> Result<(), CommError> {
        // Timeout calculation
        let calculate_timeout = |packet: &models::routing::Packet| -> Duration {
            let packet_based_timeout = Duration::from_micros(packet.content.len() as u64 * 5);
            std::cmp::max(packet_based_timeout, default_send_timeout)
        };

        // Restore packet from cache
        match PacketOps::restore(key, cache) {
            Ok(packet) => {
                let timeout = calculate_timeout(&packet);

                tracing::debug!(
                    "Attempting to stream packet to {} with timeout: {}ms",
                    peer.to_address(),
                    timeout.as_millis()
                );

                // Create blob for streaming
                let blob = Blob {
                    sender: sender.clone(),
                    packet: packet.clone(),
                };

                // Use pre-created client directly (no connection management needed)
                let mut client_guard = client.lock().await;
                let stream_result = GrpcTransport::stream(
                    &mut *client_guard,
                    peer,
                    network_id,
                    &blob,
                    packet_chunk_size as usize,
                )
                .await;
                drop(client_guard); // Release lock

                // Handle the result
                match stream_result {
                    Ok(()) => {
                        tracing::debug!("Streamed packet {} to {}", key, peer.to_address());
                    }
                    Err(error) => {
                        tracing::debug!(
                            "Error while streaming packet to {} (timeout: {}ms): {}",
                            peer.to_address(),
                            timeout.as_millis(),
                            error
                        );
                    }
                }
                Ok(())
            }
            Err(error) => {
                tracing::error!(
                    "Error while streaming packet {} to {}: {}",
                    key,
                    peer.to_address(),
                    error
                );
                Ok(())
            }
        }
    }

    /// Check if a channel is terminated
    async fn is_channel_terminated(channel: &BufferedGrpcStreamChannel) -> bool {
        // 1. First check: Is the buffer subscriber task finished?
        // This indicates the streaming mechanism is broken
        if channel.buffer_subscriber.is_finished() {
            tracing::debug!("Channel terminated: buffer subscriber task finished");
            return true;
        }

        // 2. Second check: Test if the channel can create a client
        let test_result = tokio::time::timeout(
            Duration::from_millis(50), // Very short timeout for quick check
            async {
                // Try to create a transport client - this will fail if channel is terminated
                let _client = channel.get_transport_client();
                // If we got here, the channel is still functional
                false
            },
        )
        .await;

        match test_result {
            Ok(is_terminated) => is_terminated,
            Err(_timeout) => {
                // Timeout suggests the channel is not responsive
                tracing::debug!("Channel terminated: client creation timed out");
                true
            }
        }
    }
}

#[async_trait]
impl TransportLayer for GrpcTransportClient {
    /// Send a Protocol message to a peer
    async fn send(&self, peer: &PeerNode, msg: &Protocol) -> Result<(), CommError> {
        self.with_client(peer, self.default_send_timeout, |mut client| async move {
            GrpcTransport::send(&mut client, peer, msg).await
        })
        .await
    }

    /// Broadcast a Protocol message to multiple peers in parallel
    async fn broadcast(&self, peers: &[PeerNode], msg: &Protocol) -> Result<(), CommError> {
        if peers.is_empty() {
            return Ok(());
        }

        // Create a vector of futures for parallel execution
        let send_futures: Vec<_> = peers.iter().map(|peer| self.send(peer, msg)).collect();

        // Execute all sends in parallel and collect results
        let results = futures::future::join_all(send_futures).await;

        // Check if any send failed - if so, return the first error
        for result in results {
            result?; // Return early on first error
        }

        Ok(())
    }

    /// Stream a blob to a peer by enqueueing it in the buffer
    async fn stream(&self, peer: &PeerNode, blob: &Blob) -> Result<(), CommError> {
        let channel = self.get_channel(peer).await?;
        channel.buffer.enque(blob).await
    }

    /// Stream a blob to multiple peers in parallel
    async fn stream_mult(&self, peers: &[PeerNode], blob: &Blob) -> Result<(), CommError> {
        if peers.is_empty() {
            return Ok(());
        }

        // Create a vector of futures for parallel execution
        let stream_futures: Vec<_> = peers.iter().map(|peer| self.stream(peer, blob)).collect();

        // Execute all streams in parallel and collect results
        let results = futures::future::join_all(stream_futures).await;

        // Check if any stream failed - if so, return the first error
        for result in results {
            result?; // Return early on first error
        }

        Ok(())
    }

    /// Disconnect from a peer, shutting down any gRPC channels
    async fn disconnect(&self, peer: &PeerNode) -> Result<(), CommError> {
        let mut channels_map = self.channels_map.lock().await;
        if let Some(channel_cell) = channels_map.remove(peer) {
            tracing::info!("Shutting down gRPC channel to peer {}", peer.to_address());
            // If the channel was initialized, abort its buffer subscriber
            if let Some(channel) = channel_cell.get() {
                channel.buffer_subscriber.abort();
            }
            // The channel itself will be dropped when removed from the map
        }
        Ok(())
    }

    /// Get the set of peers that have active channels
    async fn get_channeled_peers(&self) -> Result<std::collections::HashSet<PeerNode>, CommError> {
        let channels_map = self.channels_map.lock().await;
        Ok(channels_map.keys().cloned().collect())
    }
}
