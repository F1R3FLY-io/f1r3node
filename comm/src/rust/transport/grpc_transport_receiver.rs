// See comm/src/main/scala/coop/rchain/comm/transport/GrpcTransportReceiver.scala

use futures::stream::StreamExt;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{Mutex, OnceCell};
use tokio::task::JoinHandle;
use tonic::{Request, Response, Status};

use crate::rust::rp::protocol_helper;
use crate::rust::rp::rp_conf::RPConf;
use crate::rust::transport::limited_buffer::LimitedBuffer;
use crate::rust::{
    errors::CommError,
    metrics_constants::{
        PACKETS_DROPPED_METRIC, PACKETS_ENQUEUED_METRIC, PACKETS_RECEIVED_METRIC,
        STREAM_CHUNKS_DROPPED_METRIC, STREAM_CHUNKS_ENQUEUED_METRIC, STREAM_CHUNKS_RECEIVED_METRIC,
        TRANSPORT_METRICS_SOURCE,
    },
    peer_node::PeerNode,
};
use models::routing::transport_layer_server::{TransportLayer, TransportLayerServer};
use models::routing::{Chunk, TlRequest, TlResponse};
use prost::Message;

use super::limited_buffer::{FlumeLimitedBuffer, LimitedBufferObservable};
use super::messages::{Send as CommSend, StreamMessage};
use super::packet_ops::StreamCache;
use super::ssl_session_server_interceptor::SslSessionServerInterceptor;
use super::stream_handler::{Circuit, StreamError, StreamHandler, Streamed};
use shared::rust::shared::recent_hash_filter::RecentHashFilter;

/// Calculate a deterministic hash of bytes for gossip deduplication.
fn calculate_hash(bytes: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

// Circuit breaker parameters for thread-local storage
thread_local! {
    static CIRCUIT_BREAKER_PARAMS: std::cell::RefCell<Option<(String, u64)>> = std::cell::RefCell::new(None);
}

/// Circuit breaker function that uses thread-local parameters
fn circuit_breaker_with_params(streamed: &Streamed) -> Circuit {
    CIRCUIT_BREAKER_PARAMS.with(|params| {
        if let Some((network_id, max_size)) = params.borrow().as_ref() {
            if let Some(header) = &streamed.header {
                if header.network_id != *network_id {
                    return Circuit::opened(StreamError::wrong_network_id());
                }
            }

            if streamed.read_so_far > *max_size {
                return Circuit::opened(StreamError::circuit_opened());
            }
        }

        Circuit::closed()
    })
}

/// Type alias for message buffers using Arc for shared access
pub type MessageBuffers = (
    Arc<FlumeLimitedBuffer<CommSend>>,
    Arc<FlumeLimitedBuffer<StreamMessage>>,
    Arc<JoinHandle<()>>,
);

/// Type alias for message handlers
pub type MessageHandlers = (
    Arc<
        dyn Fn(CommSend) -> Pin<Box<dyn Future<Output = Result<(), CommError>> + Send>>
            + Send
            + Sync,
    >,
    Arc<
        dyn Fn(StreamMessage) -> Pin<Box<dyn Future<Output = Result<(), CommError>> + Send>>
            + Send
            + Sync,
    >,
);

/// Transport Layer Service Implementation
///
/// This implements the tonic-generated TransportLayer trait to handle
/// incoming gRPC requests with SSL session validation.
pub struct TransportLayerService {
    network_id: String,
    rp_config: RPConf,
    max_stream_message_size: u64,
    buffers_map: Arc<Mutex<HashMap<PeerNode, Arc<OnceCell<MessageBuffers>>>>>,
    message_handlers: MessageHandlers,
    cache: StreamCache,
    parallelism: usize,
    /// Filter to avoid redundant gossip of already seen block hashes
    recent_hash_filter: RecentHashFilter,
}

/// Default capacity for the recent hash filter
const RECENT_HASH_FILTER_CAPACITY: usize = 8192;

impl TransportLayerService {
    pub fn new(
        network_id: String,
        rp_config: RPConf,
        max_stream_message_size: u64,
        buffers_map: Arc<Mutex<HashMap<PeerNode, Arc<OnceCell<MessageBuffers>>>>>,
        message_handlers: MessageHandlers,
        cache: StreamCache,
        parallelism: usize,
    ) -> Self {
        Self {
            network_id,
            rp_config,
            max_stream_message_size,
            buffers_map,
            message_handlers,
            cache,
            parallelism,
            recent_hash_filter: RecentHashFilter::new(RECENT_HASH_FILTER_CAPACITY),
        }
    }

    /// Get or create message buffers for a peer
    async fn get_buffers(&self, peer: &PeerNode) -> Result<MessageBuffers, CommError> {
        let (once_cell, is_new_peer) = {
            let mut buffers_map = self.buffers_map.lock().await;

            // Check if peer already exists
            if let Some(existing_once_cell) = buffers_map.get(peer) {
                // Peer exists
                (existing_once_cell.clone(), false)
            } else {
                // Peer doesn't exist
                let new_once_cell = Arc::new(OnceCell::new());
                buffers_map.insert(peer.clone(), new_once_cell.clone());
                (new_once_cell, true)
            }
        };

        // If this is a new peer, create buffers
        if is_new_peer {
            tracing::info!("Creating inbound message queue for {}.", peer.to_address());

            // Create the actual buffers
            let (tell_buffer, blob_buffer, task_handle) =
                self.create_buffers_with_subscriptions().await;

            // Store in OnceCell
            let buffers = (
                Arc::new(tell_buffer),
                Arc::new(blob_buffer),
                Arc::new(task_handle),
            );
            let _ = once_cell.set(buffers);
        }

        // Get the buffers
        // This will wait if another thread is creating them, or return immediately if they exist
        let buffers = once_cell
            .get_or_try_init(|| async {
                let (tell_buffer, blob_buffer, task_handle) =
                    self.create_buffers_with_subscriptions().await;
                Ok((
                    Arc::new(tell_buffer),
                    Arc::new(blob_buffer),
                    Arc::new(task_handle),
                ))
            })
            .await?;

        Ok(buffers.clone())
    }

    /// Create buffers and set up background processing
    async fn create_buffers_with_subscriptions(
        &self,
    ) -> (
        FlumeLimitedBuffer<CommSend>,
        FlumeLimitedBuffer<StreamMessage>,
        tokio::task::JoinHandle<()>,
    ) {
        // Create the buffers
        let mut tell_buffer = FlumeLimitedBuffer::<CommSend>::drop_new(64);
        let mut blob_buffer = FlumeLimitedBuffer::<StreamMessage>::drop_new(8);

        // Set up subscriptions
        let tell_subscription = tell_buffer
            .subscribe()
            .expect("Failed to subscribe to tell buffer");
        let blob_subscription = blob_buffer
            .subscribe()
            .expect("Failed to subscribe to blob buffer");

        let message_handlers = self.message_handlers.clone();
        let parallelism = self.parallelism;

        // Set up background processing
        let tell_handler = message_handlers.0.clone();
        let tell_cancellable = tokio::spawn(async move {
            tell_subscription
                .map(|send_msg| {
                    let handler = tell_handler.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handler(send_msg).await {
                            tracing::error!("Error processing Send message: {}", e);
                        }
                    })
                })
                .buffer_unordered(parallelism)
                .for_each(|_| async {}) // consume all results
                .await;
        });

        let blob_handler = message_handlers.1.clone();
        let blob_cancellable = tokio::spawn(async move {
            blob_subscription
                .map(|stream_msg| {
                    let handler = blob_handler.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handler(stream_msg).await {
                            tracing::error!("Error processing StreamMessage: {}", e);
                        }
                    })
                })
                .buffer_unordered(parallelism)
                .for_each(|_| async {}) // consume all results
                .await;
        });

        // Combine both cancellables
        let combined_task = tokio::spawn(async move {
            tokio::select! {
                _ = tell_cancellable => {
                    tracing::debug!("Tell buffer processing completed");
                }
                _ = blob_cancellable => {
                    tracing::debug!("Blob buffer processing completed");
                }
            }
        });

        // Return the buffers (they can still be pushed to via the sender)
        (tell_buffer, blob_buffer, combined_task)
    }

    /// Get the tell buffer for a peer
    async fn get_tell_buffer(
        &self,
        peer: &PeerNode,
    ) -> Result<Arc<FlumeLimitedBuffer<CommSend>>, CommError> {
        let (tell_buffer, _, _) = self.get_buffers(peer).await?;
        Ok(tell_buffer)
    }

    /// Get the blob buffer for a peer
    async fn get_blob_buffer(
        &self,
        peer: &PeerNode,
    ) -> Result<Arc<FlumeLimitedBuffer<StreamMessage>>, CommError> {
        let (_, blob_buffer, _) = self.get_buffers(peer).await?;
        Ok(blob_buffer)
    }

    /// Create ACK response
    fn create_ack_response(&self, src: &PeerNode) -> TlResponse {
        TlResponse {
            payload: Some(models::routing::tl_response::Payload::Ack(
                models::routing::Ack {
                    header: Some(protocol_helper::header(src, &self.network_id)),
                },
            )),
        }
    }

    /// Create InternalServerError response
    fn create_internal_server_error_response(&self, message: String) -> TlResponse {
        TlResponse {
            payload: Some(models::routing::tl_response::Payload::InternalServerError(
                models::routing::InternalServerError {
                    error: prost::bytes::Bytes::from(message),
                },
            )),
        }
    }

    /// Handle stream using the public StreamHandler API with thread-local parameters
    async fn handle_stream_with_params<S>(
        &self,
        stream: S,
        network_id: &str,
        max_size: u64,
    ) -> Result<StreamMessage, StreamError>
    where
        S: futures::stream::Stream<Item = Chunk> + Unpin,
    {
        // Set thread-local parameters for the circuit breaker
        CIRCUIT_BREAKER_PARAMS.with(|params| {
            *params.borrow_mut() = Some((network_id.to_string(), max_size));
        });

        // Use the public StreamHandler API
        let result =
            StreamHandler::handle_stream(stream, circuit_breaker_with_params, &self.cache).await;

        // Clear thread-local parameters
        CIRCUIT_BREAKER_PARAMS.with(|params| {
            *params.borrow_mut() = None;
        });

        result
    }
}

#[tonic::async_trait]
impl TransportLayer for TransportLayerService {
    /// Handle Send requests with SSL validation
    async fn send(&self, request: Request<TlRequest>) -> Result<Response<TlResponse>, Status> {
        // Validate the request using SSL session server interceptor
        SslSessionServerInterceptor::validate_tl_request(&request)?;

        // Extract the TLRequest message
        let tl_request = request.get_ref();
        let protocol = tl_request
            .protocol
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("Missing protocol in request"))?;

        let header = protocol
            .header
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("Missing header in protocol"))?;

        let sender_node = header
            .sender
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("Missing sender in header"))?;

        // Extract peer from request
        let peer = PeerNode::from_node(sender_node.clone())
            .map_err(|e| Status::internal(format!("Failed to convert to PeerNode: {}", e)))?;

        metrics::counter!(PACKETS_RECEIVED_METRIC, "source" => TRANSPORT_METRICS_SOURCE)
            .increment(1);

        // Determine if this is a gossip message (not a request/response)
        // Only filter pure gossip announcements: BlockHashMessage and HasBlock
        // Other messages (approvals, requests, handshakes, heartbeats) must pass through
        let is_gossip = match &protocol.message {
            Some(models::routing::protocol::Message::Packet(packet)) => {
                packet.type_id == "BlockHashMessage" || packet.type_id == "HasBlock"
            }
            _ => false,
        };

        // Deduplicate redundant gossip - skip if we've seen this hash recently
        // Only apply to gossip messages to avoid blocking legitimate requests/responses
        if is_gossip {
            let protocol_bytes = protocol.encode_to_vec();
            let hash_tag = format!("{:x}", calculate_hash(&protocol_bytes));

            if self.recent_hash_filter.seen_before(&hash_tag) {
                tracing::debug!(
                    "[GOSSIP] Suppressed redundant hash broadcast {} from {}",
                    hash_tag,
                    peer.endpoint.host
                );
                return Ok(Response::new(
                    self.create_ack_response(&self.rp_config.local),
                ));
            }
        }

        // Get target buffer
        let tell_buffer = self
            .get_tell_buffer(&peer)
            .await
            .map_err(|e| Status::internal(format!("Failed to get tell buffer: {}", e)))?;

        // Push message to buffer and handle result
        // TODO(perf): protocol.clone() copies entire message which may contain block data.
        // Consider using Arc<Protocol> or passing ownership if buffer can take it.
        let send_msg = CommSend::new(protocol.clone());

        let response = if tell_buffer.push_next(send_msg) {
            // Successfully enqueued
            metrics::counter!(PACKETS_ENQUEUED_METRIC, "source" => TRANSPORT_METRICS_SOURCE)
                .increment(1);
            self.create_ack_response(&self.rp_config.local)
        } else {
            // Buffer full
            let packet_dropped_msg = format!(
                "Packet dropped, {} packet queue overflown.",
                peer.endpoint.host
            );
            metrics::counter!(PACKETS_DROPPED_METRIC, "source" => TRANSPORT_METRICS_SOURCE)
                .increment(1);
            self.create_internal_server_error_response(packet_dropped_msg)
        };

        Ok(Response::new(response))
    }

    /// Handle Stream requests with SSL validation
    async fn stream(
        &self,
        request: Request<tonic::Streaming<Chunk>>,
    ) -> Result<Response<TlResponse>, Status> {
        // Validate the request using SSL session server interceptor
        // Note: For streaming requests, we validate the TLS session context
        // The actual message content validation happens in StreamHandler
        SslSessionServerInterceptor::validate_stream_request(&request)?;

        let stream = request.into_inner();

        // Convert tonic::Streaming<Chunk> to Stream<Item = Chunk> by handling Results
        let chunk_stream = stream.map(|result| match result {
            Ok(chunk) => chunk,
            Err(status) => {
                tracing::error!("gRPC stream error: {}", status);
                Chunk { content: None }
            }
        });

        // Use our custom handler with parameters
        let stream_result = self
            .handle_stream_with_params(chunk_stream, &self.network_id, self.max_stream_message_size)
            .await;

        let response = match stream_result {
            Err(StreamError::Unexpected { ref error }) => {
                tracing::error!("Stream error: {}", error);
                self.create_internal_server_error_response(error.clone())
            }
            Err(ref error) => {
                tracing::warn!("Stream error: {}", error.message());
                self.create_internal_server_error_response(error.message())
            }
            Ok(stream_msg) => {
                metrics::counter!(STREAM_CHUNKS_RECEIVED_METRIC, "source" => TRANSPORT_METRICS_SOURCE).increment(1);
                let msg_enqueued = format!(
                    "Stream chunk pushed to message buffer. Sender {}, message {}, size {}, file {}.",
                    stream_msg.sender.endpoint.host,
                    stream_msg.type_id,
                    stream_msg.content_length,
                    stream_msg.key
                );
                let msg_dropped = format!(
                    "Stream chunk dropped, {} stream queue overflown.",
                    stream_msg.sender.endpoint.host
                );

                // Get target buffer for the sender
                match self.get_blob_buffer(&stream_msg.sender).await {
                    Ok(target_buffer) => {
                        // Try to push message to buffer
                        if target_buffer.push_next(stream_msg.clone()) {
                            metrics::counter!(STREAM_CHUNKS_ENQUEUED_METRIC, "source" => TRANSPORT_METRICS_SOURCE).increment(1);
                            tracing::debug!("{}", msg_enqueued);
                            self.create_ack_response(&self.rp_config.local)
                        } else {
                            metrics::counter!(STREAM_CHUNKS_DROPPED_METRIC, "source" => TRANSPORT_METRICS_SOURCE).increment(1);
                            tracing::debug!("{}", msg_dropped);
                            // Clean up cache on overflow
                            self.cache.remove(&stream_msg.key);
                            self.create_internal_server_error_response(msg_dropped)
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to get blob buffer: {}", e);
                        self.create_internal_server_error_response(format!("Buffer error: {}", e))
                    }
                }
            }
        };

        Ok(Response::new(response))
    }
}

/// GrpcTransportReceiver for handling incoming gRPC messages
pub struct GrpcTransportReceiver;

impl GrpcTransportReceiver {
    /// Create a new gRPC transport receiver with F1r3fly custom TLS
    pub async fn create(
        network_id: String,
        rp_config: RPConf,
        port: u16,
        cert_pem: String,
        key_pem: String,
        max_message_size: i32,
        max_stream_message_size: u64,
        buffers_map: Arc<Mutex<HashMap<PeerNode, Arc<OnceCell<MessageBuffers>>>>>,
        message_handlers: MessageHandlers,
        parallelism: usize,
        cache: StreamCache,
    ) -> Result<JoinHandle<()>, CommError> {
        use std::net::SocketAddr;
        use tonic::transport::Server;

        // Import our custom F1r3fly server
        use super::f1r3fly_server::F1r3flyServer;

        let addr: SocketAddr = format!("0.0.0.0:{}", port)
            .parse()
            .map_err(|e| CommError::ConfigError(format!("Invalid address: {}", e)))?;

        // Create SSL session server interceptor
        let ssl_interceptor = SslSessionServerInterceptor::new(network_id.clone());

        // Create the transport layer service implementation
        let transport_service = TransportLayerService::new(
            network_id.clone(),
            rp_config,
            max_stream_message_size,
            buffers_map,
            message_handlers,
            cache,
            parallelism,
        );

        // Create F1r3fly server with custom TLS configuration
        let f1r3fly_server = F1r3flyServer::builder(network_id.clone(), &cert_pem, &key_pem, addr)
            .map_err(|e| CommError::ConfigError(format!("F1r3fly server creation failed: {}", e)))?
            // Configure TCP settings to match the previous tonic configuration
            .tcp_keepalive(Some(std::time::Duration::from_secs(600))) // 10 minutes
            .tcp_nodelay(true)
            .http2_keepalive_interval(Some(std::time::Duration::from_secs(30)))
            .http2_keepalive_timeout(Some(std::time::Duration::from_secs(5)));

        // Create incoming connection stream with F1r3fly TLS
        let incoming = f1r3fly_server.incoming().await.map_err(|e| {
            CommError::ConfigError(format!("Failed to create F1r3fly incoming stream: {}", e))
        })?;

        // Create the gRPC server with F1r3fly TLS configuration
        let server_task = tokio::spawn(async move {
            tracing::info!(
                "Starting F1r3fly TLS-enabled gRPC transport receiver on {}",
                addr
            );

            let server_result = Server::builder()
                // Request timeout (30s): Maximum time for a single gRPC request to complete.
                // Prevents hanging requests from consuming resources indefinitely.
                // Essential for blockchain P2P networks where nodes can be slow or unresponsive.
                // 30 seconds allows time for large block transfers but prevents infinite waits.
                .timeout(std::time::Duration::from_secs(30))
                // TCP keepalive - handled by F1r3flyServer configuration above
                // TCP nodelay - handled by F1r3flyServer configuration above
                // HTTP/2 keepalive interval - handled by F1r3flyServer configuration above
                // HTTP/2 keepalive timeout - handled by F1r3flyServer configuration above
                // Configure HTTP/2 max frame size
                .max_frame_size(Some(max_message_size as u32))
                // **F1r3fly Message Size Architecture**
                //
                // Unlike Scala's NettyServerBuilder.maxInboundMessageSize(), tonic does not provide
                // server-wide message size configuration. Instead, tonic requires per-service limits
                // via Grpc<T>.max_decoding_message_size(), but TransportLayerServer::with_interceptor()
                // doesn't expose the underlying Grpc<T> instance.
                //
                // Our F1r3fly architecture addresses this limitation through multiple layers:
                //
                // 1. **HTTP/2 Frame Limits** (configured above): Provides network-level protection
                //    by limiting individual HTTP/2 frames to prevent oversized packets
                //
                // 2. **Application-Level Buffer Management**: TransportLayerService implements
                //    intelligent buffer overflow policies for both regular and streaming messages
                //
                // 3. **Client-Side Configuration**: GrpcTransportClient correctly configures both
                //    max_encoding_message_size and max_decoding_message_size per connection
                //
                // 4. **Stream-Based Protection**: Large messages use our streaming protocol with
                //    configurable max_stream_message_size limits and circuit breaker patterns
                //
                // This multi-layered approach provides equivalent protection to the Scala implementation
                // while working within tonic's architectural constraints. Server defaults: 4MB decoding,
                // unlimited encoding, with streaming handling for larger payloads.
                //
                // Add the transport layer service with SSL interceptor
                .add_service(TransportLayerServer::with_interceptor(
                    transport_service,
                    ssl_interceptor,
                ))
                // Use F1r3fly incoming stream instead of standard TLS configuration
                .serve_with_incoming(incoming)
                .await;

            if let Err(e) = server_result {
                tracing::error!("F1r3fly gRPC server error: {}", e);
            }

            Ok::<(), CommError>(())
        });

        // Handle the Result from the spawn task
        Ok(tokio::spawn(async move {
            if let Err(e) = server_task.await {
                tracing::error!("F1r3fly server task join error: {}", e);
            }
        }))
    }
}
