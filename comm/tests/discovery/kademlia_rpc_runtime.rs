// See comm/src/test/scala/coop/rchain/comm/discovery/KademliaRPCRuntime.scala
// See comm/src/test/scala/coop/rchain/comm/discovery/GrpcKademliaRPCSpec.scala

use prost::bytes::Bytes;
use rand::RngCore;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::TcpListener;

use comm::rust::{
    discovery::{
        grpc_kademlia_rpc::GrpcKademliaRPC,
        grpc_kademlia_rpc_server::{LookupHandler, PingHandler},
        utils::acquire_kademlia_rpc_server,
    },
    errors::CommError,
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
};
use shared::rust::grpc::grpc_server::GrpcServer;

pub struct GrpcEnvironment {
    pub host: String,
    pub port: u16,
    pub peer: PeerNode,
}

pub struct TestRuntime {
    network_id: String,
}

impl TestRuntime {
    pub fn new(network_id: String) -> Self {
        Self { network_id }
    }

    /// Create environment with random peer ID and given port
    pub fn create_environment(&self, port: u16) -> GrpcEnvironment {
        let host = "127.0.0.1".to_string();
        let mut bytes = [0u8; 40];
        rand::rng().fill_bytes(&mut bytes);
        let peer = PeerNode {
            id: NodeIdentifier {
                key: Bytes::from(bytes.to_vec()),
            },
            endpoint: Endpoint::new(host.clone(), port as u32, port as u32),
        };
        GrpcEnvironment { host, port, peer }
    }

    /// Create Kademlia RPC client
    pub fn create_kademlia_rpc(&self, env: &GrpcEnvironment) -> GrpcKademliaRPC {
        GrpcKademliaRPC::new(
            self.network_id.clone(),
            Duration::from_millis(500),
            true, // allow_private_addresses
            env.peer.clone(),
        )
    }

    /// Create Kademlia RPC server with handlers
    pub async fn create_kademlia_rpc_server(
        &self,
        env: &GrpcEnvironment,
        ping_handler: PingHandler,
        lookup_handler: LookupHandler,
    ) -> Result<GrpcServer, CommError> {
        acquire_kademlia_rpc_server(
            self.network_id.clone(),
            env.port,
            ping_handler,
            lookup_handler,
        )
        .await
    }
}

/// Test ping handler that records received messages
#[derive(Clone)]
pub struct TestPingHandler {
    /// Records (receiver_peer, sender_peer)
    received: Arc<Mutex<Vec<(PeerNode, PeerNode)>>>,
    delay: Option<Duration>,
}

impl TestPingHandler {
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

    /// Convert to the required PingHandler type
    pub fn to_handler(&self, receiver_peer: PeerNode) -> PingHandler {
        let received = self.received.clone();
        let delay = self.delay;

        Box::new(move |sender_peer: &PeerNode| {
            let received = received.clone();
            let receiver_peer = receiver_peer.clone();
            let sender_peer = sender_peer.clone();

            Box::pin(async move {
                // Record the message (receiver, sender)
                {
                    let mut received = received.lock().unwrap();
                    received.push((receiver_peer, sender_peer));
                }

                // Apply delay if configured
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }
            }) as Pin<Box<dyn Future<Output = ()> + Send>>
        })
    }

    /// Get received messages
    pub fn received(&self) -> Vec<(PeerNode, PeerNode)> {
        self.received.lock().unwrap().clone()
    }
}

/// Test lookup handler
#[derive(Clone)]
pub struct TestLookupHandler {
    received: Arc<Mutex<Vec<(PeerNode, (PeerNode, Vec<u8>))>>>,
    response: Vec<PeerNode>,
    delay: Option<Duration>,
}

impl TestLookupHandler {
    pub fn new() -> Self {
        Self {
            received: Arc::new(Mutex::new(Vec::new())),
            response: Vec::new(),
            delay: None,
        }
    }

    pub fn with_response(response: Vec<PeerNode>) -> Self {
        Self {
            received: Arc::new(Mutex::new(Vec::new())),
            response,
            delay: None,
        }
    }

    pub fn with_delay(delay: Duration) -> Self {
        Self {
            received: Arc::new(Mutex::new(Vec::new())),
            response: Vec::new(),
            delay: Some(delay),
        }
    }

    /// Convert to the required LookupHandler type
    pub fn to_handler(&self, receiver_peer: PeerNode) -> LookupHandler {
        let received = self.received.clone();
        let response = self.response.clone();
        let delay = self.delay;

        Box::new(move |sender_peer: &PeerNode, key: &[u8]| {
            let received = received.clone();
            let receiver_peer = receiver_peer.clone();
            let sender_peer = sender_peer.clone();
            let key = key.to_vec();
            let response = response.clone();

            Box::pin(async move {
                // Record the message
                {
                    let mut received = received.lock().unwrap();
                    received.push((receiver_peer, (sender_peer, key)));
                }

                // Apply delay if configured
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }

                response
            }) as Pin<Box<dyn Future<Output = Vec<PeerNode>> + Send>>
        })
    }

    pub fn received(&self) -> Vec<(PeerNode, (PeerNode, Vec<u8>))> {
        self.received.lock().unwrap().clone()
    }
}

/// Get free port
pub async fn get_free_port() -> Result<u16, std::io::Error> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    Ok(port)
}

/// Simplified two nodes test result
pub struct TwoNodesResult<T> {
    pub result: T,
    pub local_node: PeerNode,
    pub remote_node: PeerNode,
}

/// Two nodes test execution
pub async fn run_two_nodes_test<T, F, Fut>(
    runtime: &TestRuntime,
    execute_fn: F,
    ping_handler: Option<TestPingHandler>,
    lookup_handler: Option<TestLookupHandler>,
) -> Result<(TwoNodesResult<T>, TestPingHandler, TestLookupHandler), CommError>
where
    F: FnOnce(GrpcKademliaRPC, PeerNode, PeerNode) -> Fut,
    Fut: std::future::Future<Output = T>,
{
    // Create two environments
    let port1 = get_free_port()
        .await
        .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;
    let port2 = get_free_port()
        .await
        .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;

    let env1 = runtime.create_environment(port1);
    let env2 = runtime.create_environment(port2);

    let local = env1.peer.clone();
    let remote = env2.peer.clone();

    // Use provided handlers or create defaults
    let ping_handler = ping_handler.unwrap_or_else(|| TestPingHandler::new());
    let lookup_handler = lookup_handler.unwrap_or_else(|| TestLookupHandler::new());

    // Create local RPC client
    let local_rpc = runtime.create_kademlia_rpc(&env1);

    // Create and start remote RPC server
    let mut remote_server = runtime
        .create_kademlia_rpc_server(
            &env2,
            ping_handler.to_handler(remote.clone()),
            lookup_handler.to_handler(remote.clone()),
        )
        .await?;

    // Give the server a moment to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Execute the test function
    let result = execute_fn(local_rpc, local.clone(), remote.clone()).await;

    // Stop the server
    remote_server
        .stop()
        .await
        .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;

    let test_result = TwoNodesResult {
        result,
        local_node: local,
        remote_node: remote,
    };

    Ok((test_result, ping_handler, lookup_handler))
}

/// Two nodes test execution without starting remote server (simulates dead remote peer)
pub async fn run_two_nodes_test_remote_dead<T, F, Fut>(
    runtime: &TestRuntime,
    execute_fn: F,
    ping_handler: Option<TestPingHandler>,
    lookup_handler: Option<TestLookupHandler>,
) -> Result<(TwoNodesResult<T>, TestPingHandler, TestLookupHandler), CommError>
where
    F: FnOnce(GrpcKademliaRPC, PeerNode, PeerNode) -> Fut,
    Fut: std::future::Future<Output = T>,
{
    // Create two environments
    let port1 = get_free_port()
        .await
        .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;
    let port2 = get_free_port()
        .await
        .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;

    let env1 = runtime.create_environment(port1);
    let env2 = runtime.create_environment(port2);

    let local = env1.peer.clone();
    let remote = env2.peer.clone();

    // Use provided handlers or create defaults (even though they won't be used)
    let ping_handler = ping_handler.unwrap_or_else(|| TestPingHandler::new());
    let lookup_handler = lookup_handler.unwrap_or_else(|| TestLookupHandler::new());

    // Create local RPC client
    let local_rpc = runtime.create_kademlia_rpc(&env1);

    // NOTE: Unlike run_two_nodes_test, we do NOT start a remote server here
    // This simulates the "peer is not listening" scenario

    // Execute the test function - this should fail because no server is running
    let result = execute_fn(local_rpc, local.clone(), remote.clone()).await;

    let test_result = TwoNodesResult {
        result,
        local_node: local,
        remote_node: remote,
    };

    Ok((test_result, ping_handler, lookup_handler))
}
