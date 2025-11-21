// See comm/src/test/scala/coop/rchain/comm/transport/TransportLayerRuntime.scala
// See comm/src/test/scala/coop/rchain/comm/transport/TcpTransportLayerSpec.scala

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::OnceCell;

use comm::rust::test_instances::create_rp_conf_ask;
use comm::rust::{
    errors::CommError,
    peer_node::PeerNode,
    rp::protocol_helper,
    transport::{
        communication_response::CommunicationResponse,
        grpc_transport_client::{BufferedGrpcStreamChannel, GrpcTransportClient},
        grpc_transport_server::{
            DispatchFn, GrpcTransportServer, HandleStreamedFn, TransportServer,
        },
        transport_layer::{Blob, TransportLayer},
    },
};
use crypto::rust::util::certificate_helper::{CertificateHelper, CertificatePrinter};
use models::routing::Protocol;

/// TLS Environment for transport layer testing
pub struct TlsEnvironment {
    pub host: String,
    pub port: u16,
    pub cert: String,
    pub key: String,
    pub peer: PeerNode,
}

/// Test runtime for transport layer testing
pub struct TransportLayerTestRuntime {
    network_id: String,
    pub max_message_size: i32,
    max_stream_message_size: u64,
}

impl TransportLayerTestRuntime {
    pub fn new(network_id: String) -> Self {
        Self {
            network_id,
            max_message_size: 256 * 1024,               // 256KB
            max_stream_message_size: 200 * 1024 * 1024, // 200MB
        }
    }

    /// Create environment with TLS certificates and peer node
    pub async fn create_environment(&self, port: u16) -> Result<TlsEnvironment, CommError> {
        let host = "127.0.0.1".to_string();

        let (secret_key, public_key) = CertificateHelper::generate_key_pair(true);

        let cert_der = CertificateHelper::generate_certificate(&secret_key, &public_key)
            .map_err(|e| CommError::ConfigError(format!("Certificate generation failed: {}", e)))?;
        let cert = CertificatePrinter::print_certificate(&cert_der);

        let key = CertificatePrinter::print_private_key_from_secret(&secret_key)
            .map_err(|e| CommError::ConfigError(format!("Key serialization failed: {}", e)))?;

        let id = CertificateHelper::public_address(&public_key).ok_or_else(|| {
            CommError::ConfigError("Failed to generate public address".to_string())
        })?;
        let id_hex = hex::encode(&id);

        let address = format!("rnode://{}@{}?protocol={}&discovery=0", id_hex, host, port);
        let peer = PeerNode::from_address(&address)?;

        Ok(TlsEnvironment {
            host,
            port,
            cert,
            key,
            peer,
        })
    }

    /// Create transport layer client
    pub fn create_transport_layer(
        &self,
        env: &TlsEnvironment,
    ) -> Result<GrpcTransportClient, CommError> {
        let channels_map = Arc::new(tokio::sync::Mutex::new(HashMap::<
            PeerNode,
            Arc<OnceCell<Arc<BufferedGrpcStreamChannel>>>,
        >::new()));

        GrpcTransportClient::new(
            self.network_id.clone(),
            env.cert.clone(),
            env.key.clone(),
            self.max_message_size,
            self.max_message_size,
            100,
            channels_map,
        )
    }

    /// Create transport layer server
    pub fn create_transport_layer_server(&self, env: &TlsEnvironment) -> TransportServer {
        // Create RP configuration for the server
        let rp_config = create_rp_conf_ask(env.peer.clone(), None, None);

        let transport_server = GrpcTransportServer::new(
            rp_config,
            self.network_id.clone(),
            env.port,
            env.cert.clone(),
            env.key.clone(),
            self.max_message_size,
            self.max_stream_message_size,
            4, // parallelism
        );

        TransportServer::new(transport_server)
    }

    /// Two nodes test execution pattern
    pub async fn run_two_nodes_test<T, F, Fut>(
        &self,
        execute_fn: F,
        protocol_dispatcher: Option<TestProtocolDispatcher>,
        stream_dispatcher: Option<TestStreamDispatcher>,
        block_until_dispatched: bool,
    ) -> Result<TwoNodesResult<T>, CommError>
    where
        F: FnOnce(GrpcTransportClient, PeerNode, PeerNode) -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        // Create two environments
        let port1 = get_free_port()
            .await
            .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;
        let port2 = get_free_port()
            .await
            .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;

        let env1 = self.create_environment(port1).await?;
        let env2 = self.create_environment(port2).await?;

        // Create transport layer and server
        let local_transport = self.create_transport_layer(&env1)?;
        let remote_transport_server = self.create_transport_layer_server(&env2);

        let local = env1.peer.clone();
        let remote = env2.peer.clone();

        // Use provided dispatchers or create defaults
        let protocol_dispatcher =
            protocol_dispatcher.unwrap_or_else(|| TestProtocolDispatcher::new());
        let stream_dispatcher = stream_dispatcher.unwrap_or_else(|| TestStreamDispatcher::new());

        // Create dispatcher callback
        let callback = DispatcherCallback::new();

        // Start the remote server
        let _server_handle = remote_transport_server
            .start(
                protocol_dispatcher.to_handler(remote.clone(), Some(callback.clone())),
                stream_dispatcher.to_handler(remote.clone(), Some(callback.clone())),
            )
            .await?;

        // Give the server a moment to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Execute the test function
        let result = execute_fn(local_transport, local.clone(), remote.clone()).await;

        // Wait for message processing if requested
        if block_until_dispatched {
            // Wait with timeout to avoid hanging tests
            let wait_result =
                tokio::time::timeout(Duration::from_secs(5), callback.wait_until_dispatched())
                    .await;

            if wait_result.is_err() {
                tracing::warn!("Timeout waiting for message dispatch in two nodes test");
            }
        } else {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        // Stop the server
        remote_transport_server.stop().await?;

        Ok(TwoNodesResult {
            local_node: local,
            remote_node: remote,
            result,
        })
    }

    /// Three nodes test execution pattern
    pub async fn run_three_nodes_test<T, F, Fut>(
        &self,
        execute_fn: F,
        protocol_dispatcher: Option<TestProtocolDispatcher>,
        stream_dispatcher: Option<TestStreamDispatcher>,
        block_until_dispatched: bool,
    ) -> Result<ThreeNodesResult<T>, CommError>
    where
        F: FnOnce(GrpcTransportClient, PeerNode, PeerNode, PeerNode) -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        // Create three environments
        let port1 = get_free_port()
            .await
            .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;
        let port2 = get_free_port()
            .await
            .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;
        let port3 = get_free_port()
            .await
            .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;

        let env1 = self.create_environment(port1).await?;
        let env2 = self.create_environment(port2).await?;
        let env3 = self.create_environment(port3).await?;

        // Create transport layer and servers
        let local_transport = self.create_transport_layer(&env1)?;
        let remote_transport_server1 = self.create_transport_layer_server(&env2);
        let remote_transport_server2 = self.create_transport_layer_server(&env3);

        let local = env1.peer.clone();
        let remote1 = env2.peer.clone();
        let remote2 = env3.peer.clone();

        // Use provided dispatchers or create defaults
        let protocol_dispatcher =
            protocol_dispatcher.unwrap_or_else(|| TestProtocolDispatcher::new());
        let stream_dispatcher = stream_dispatcher.unwrap_or_else(|| TestStreamDispatcher::new());

        // Create dispatcher callbacks
        let callback1 = DispatcherCallback::new();
        let callback2 = DispatcherCallback::new();

        // Start both remote servers
        let _server_handle1 = remote_transport_server1
            .start(
                protocol_dispatcher.to_handler(remote1.clone(), Some(callback1.clone())),
                stream_dispatcher.to_handler(remote1.clone(), Some(callback1.clone())),
            )
            .await?;

        let _server_handle2 = remote_transport_server2
            .start(
                protocol_dispatcher.to_handler(remote2.clone(), Some(callback2.clone())),
                stream_dispatcher.to_handler(remote2.clone(), Some(callback2.clone())),
            )
            .await?;

        // Give the servers a moment to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Execute the test function
        let result = execute_fn(
            local_transport,
            local.clone(),
            remote1.clone(),
            remote2.clone(),
        )
        .await;

        // Wait for message processing if requested
        if block_until_dispatched {
            // Wait for both callbacks with timeout
            let wait1 =
                tokio::time::timeout(Duration::from_secs(5), callback1.wait_until_dispatched());
            let wait2 =
                tokio::time::timeout(Duration::from_secs(5), callback2.wait_until_dispatched());

            let (result1, result2) = tokio::join!(wait1, wait2);
            if result1.is_err() || result2.is_err() {
                tracing::warn!("Timeout waiting for message dispatch in three nodes test");
            }
        } else {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        // Stop both servers
        let stop1 = remote_transport_server1.stop();
        let stop2 = remote_transport_server2.stop();
        let (_, _) = tokio::join!(stop1, stop2);

        Ok(ThreeNodesResult {
            local_node: local,
            remote_node1: remote1,
            remote_node2: remote2,
            result,
        })
    }

    /// Two nodes test with remote dead (no server started)
    pub async fn run_two_nodes_test_remote_dead<T, F, Fut>(
        &self,
        execute_fn: F,
    ) -> Result<TwoNodesResult<T>, CommError>
    where
        F: FnOnce(GrpcTransportClient, PeerNode, PeerNode) -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        // Create two environments
        let port1 = get_free_port()
            .await
            .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;
        let port2 = get_free_port()
            .await
            .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;

        let env1 = self.create_environment(port1).await?;
        let env2 = self.create_environment(port2).await?;

        // Create transport layer (but NO server for remote - simulates dead peer)
        let local_transport = self.create_transport_layer(&env1)?;

        let local = env1.peer.clone();
        let remote = env2.peer.clone();

        // Execute the test function - this should fail because no server is running
        let result = execute_fn(local_transport, local.clone(), remote.clone()).await;

        Ok(TwoNodesResult {
            local_node: local,
            remote_node: remote,
            result,
        })
    }
}

/// Test protocol dispatcher that records received Protocol messages
#[derive(Clone)]
pub struct TestProtocolDispatcher {
    /// Records (receiver_peer, protocol_message)
    received: Arc<Mutex<Vec<(PeerNode, Protocol)>>>,
    response_type: ResponseType,
    delay: Option<Duration>,
}

#[derive(Clone)]
pub enum ResponseType {
    /// Return success response (handledWithoutMessage)
    Success,
    /// Return internal error response
    InternalError(String),
}

impl TestProtocolDispatcher {
    pub fn new() -> Self {
        Self {
            received: Arc::new(Mutex::new(Vec::new())),
            response_type: ResponseType::Success,
            delay: None,
        }
    }

    pub fn with_response_type(response_type: ResponseType) -> Self {
        Self {
            received: Arc::new(Mutex::new(Vec::new())),
            response_type,
            delay: None,
        }
    }

    pub fn with_delay(delay: Duration) -> Self {
        Self {
            received: Arc::new(Mutex::new(Vec::new())),
            response_type: ResponseType::Success,
            delay: Some(delay),
        }
    }

    /// Convert to the required DispatchFn type for TransportLayerServer::handle_receive
    pub fn to_handler(
        &self,
        receiver_peer: PeerNode,
        callback: Option<DispatcherCallback>,
    ) -> DispatchFn {
        let received = self.received.clone();
        let response_type = self.response_type.clone();
        let delay = self.delay;

        Arc::new(move |protocol: Protocol| {
            let received = received.clone();
            let receiver_peer = receiver_peer.clone();
            let protocol_clone = protocol.clone();
            let response_type = response_type.clone();
            let callback = callback.clone();

            Box::pin(async move {
                // Record the message (receiver, protocol)
                {
                    let mut received = received.lock().unwrap();
                    received.push((receiver_peer, protocol_clone));
                }

                // Apply delay if configured
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }

                // Notify callback if provided
                if let Some(callback) = callback {
                    callback.notify_that_dispatched();
                }

                // Return appropriate response based on response type
                match response_type {
                    ResponseType::Success => Ok(CommunicationResponse::handled_without_message()),
                    ResponseType::InternalError(msg) => {
                        Err(CommError::InternalCommunicationError(msg))
                    }
                }
            })
                as Pin<Box<dyn Future<Output = Result<CommunicationResponse, CommError>> + Send>>
        })
    }

    /// Get received messages
    pub fn received(&self) -> Vec<(PeerNode, Protocol)> {
        self.received.lock().unwrap().clone()
    }
}

/// Test stream dispatcher that records received Blob messages
#[derive(Clone)]
pub struct TestStreamDispatcher {
    /// Records (receiver_peer, blob)
    received: Arc<Mutex<Vec<(PeerNode, Blob)>>>,
    delay: Option<Duration>,
}

impl TestStreamDispatcher {
    pub fn new() -> Self {
        Self {
            received: Arc::new(Mutex::new(Vec::new())),
            delay: None,
        }
    }

    pub fn with_delay(delay: Duration) -> Self {
        Self {
            received: Arc::new(Mutex::new(Vec::new())),
            delay: Some(delay),
        }
    }

    /// Convert to the required HandleStreamedFn type for TransportLayerServer::handle_receive
    pub fn to_handler(
        &self,
        receiver_peer: PeerNode,
        callback: Option<DispatcherCallback>,
    ) -> HandleStreamedFn {
        let received = self.received.clone();
        let delay = self.delay;

        Arc::new(move |blob: Blob| {
            let received = received.clone();
            let receiver_peer = receiver_peer.clone();
            let blob_clone = blob.clone();
            let callback = callback.clone();

            Box::pin(async move {
                // Record the message (receiver, blob)
                {
                    let mut received = received.lock().unwrap();
                    received.push((receiver_peer, blob_clone));
                }

                // Apply delay if configured
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }

                // Notify callback if provided
                if let Some(callback) = callback {
                    callback.notify_that_dispatched();
                }

                Ok(())
            }) as Pin<Box<dyn Future<Output = Result<(), CommError>> + Send>>
        })
    }

    /// Get received messages
    pub fn received(&self) -> Vec<(PeerNode, Blob)> {
        self.received.lock().unwrap().iter().cloned().collect()
    }
}

/// Dispatcher callback for test synchronization
#[derive(Clone)]
pub struct DispatcherCallback {
    notified: Arc<tokio::sync::Mutex<bool>>,
    notify: Arc<tokio::sync::Notify>,
}

impl DispatcherCallback {
    pub fn new() -> Self {
        Self {
            notified: Arc::new(tokio::sync::Mutex::new(false)),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    /// Notify that a message has been dispatched
    pub fn notify_that_dispatched(&self) {
        let notify = self.notify.clone();
        let notified = self.notified.clone();

        tokio::spawn(async move {
            let mut notified = notified.lock().await;
            if !*notified {
                *notified = true;
                notify.notify_one();
            }
        });
    }

    /// Wait until a message has been dispatched
    pub async fn wait_until_dispatched(&self) {
        loop {
            {
                let notified = self.notified.lock().await;
                if *notified {
                    return;
                }
            }
            self.notify.notified().await;
        }
    }
}

/// Get free port for testing
pub async fn get_free_port() -> Result<u16, std::io::Error> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    Ok(port)
}

/// Send heartbeat message to remote peer
pub async fn send_heartbeat(
    transport: Arc<dyn TransportLayer>,
    local: &PeerNode,
    remote: &PeerNode,
    network_id: &str,
) -> Result<(), CommError> {
    let msg = protocol_helper::heartbeat(local, network_id);
    transport.send(remote, &msg).await
}

/// Broadcast heartbeat message to multiple peers
pub async fn broadcast_heartbeat(
    transport: Arc<dyn TransportLayer>,
    local: &PeerNode,
    remotes: &[PeerNode],
    network_id: &str,
) -> Result<(), CommError> {
    let msg = protocol_helper::heartbeat(local, network_id);
    transport.broadcast(remotes, &msg).await
}

pub struct TwoNodesResult<T> {
    pub local_node: PeerNode,
    pub remote_node: PeerNode,
    pub result: T,
}

pub struct ThreeNodesResult<T> {
    pub local_node: PeerNode,
    pub remote_node1: PeerNode,
    pub remote_node2: PeerNode,
    pub result: T,
}
