// See comm/src/main/scala/coop/rchain/comm/discovery/GrpcKademliaRPC.scala

use std::time::{Duration, Instant};

use async_trait::async_trait;
use prost::bytes::Bytes;
use tonic::transport::{Channel, Endpoint};

use crate::{
    comm::{kademlia_rpc_service_client::KademliaRpcServiceClient, Lookup, Ping},
    rust::{
        discovery::{
            kademlia_rpc::KademliaRPC,
            utils::{to_node, to_peer_node},
        },
        errors::CommError,
        metrics_constants::{DISCOVERY_GRPC_METRICS_SOURCE, LOOKUP_METRIC, LOOKUP_TIME_METRIC, PING_METRIC, PING_TIME_METRIC},
        peer_node::PeerNode,
        utils::{is_valid_inet_address, is_valid_public_inet_address},
    },
};

/// Rust implementation of GrpcKademliaRPC
pub struct GrpcKademliaRPC {
    network_id: String,
    timeout: Duration,
    allow_private_addresses: bool,
    local_peer: PeerNode,
}

impl GrpcKademliaRPC {
    pub fn new(
        network_id: String,
        timeout: Duration,
        allow_private_addresses: bool,
        local_peer: PeerNode,
    ) -> Self {
        Self {
            network_id,
            timeout,
            allow_private_addresses,
            local_peer,
        }
    }

    /// Create a gRPC client channel for the given peer
    async fn client_channel(&self, peer: &PeerNode) -> Result<Channel, CommError> {
        let endpoint = format!("http://{}:{}", peer.endpoint.host, peer.endpoint.udp_port);

        let endpoint = Endpoint::from_shared(endpoint).map_err(|e| {
            CommError::InternalCommunicationError(format!("Invalid endpoint: {}", e))
        })?;

        let channel = endpoint.connect().await.map_err(|e| {
            CommError::InternalCommunicationError(format!("Connection failed: {}", e))
        })?;

        Ok(channel)
    }

    /// Execute a function with a gRPC client, handling resource management
    /// This includes: channel creation, deadline setting on client, and proper cleanup
    async fn with_client_ping(&self, peer: &PeerNode, ping_msg: Ping) -> Result<bool, CommError> {
        let start = Instant::now();
        // Create channel
        let channel = match self.client_channel(peer).await {
            Ok(c) => c,
            Err(_) => {
                log::error!("Failed to connect to peer for ping");
                return Ok(false); // Return false for connection failures
            }
        };

        // Create client
        let mut client = KademliaRpcServiceClient::new(channel.clone());

        // Execute the operation with timeout
        let result = tokio::time::timeout(
            self.timeout,
            client.send_ping(tonic::Request::new(ping_msg)),
        )
        .await;

        // Cleanup: channel will be dropped automatically
        drop(channel);

        let duration = start.elapsed();
        metrics::histogram!(PING_TIME_METRIC, "source" => DISCOVERY_GRPC_METRICS_SOURCE).record(duration.as_millis() as f64);

        match result {
            Ok(Ok(response)) => {
                let pong = response.into_inner();
                if pong.network_id == self.network_id {
                    Ok(true) // Success - network IDs match
                } else {
                    log::warn!("Network ID mismatch in pong");
                    Ok(false)
                }
            }
            Ok(Err(status)) => {
                log::error!("Ping failed: {:?}", status);
                Ok(false)
            }
            Err(_) => {
                log::error!("Ping timed out");
                Ok(false)
            }
        }
    }

    /// Execute lookup with proper resource management
    async fn with_client_lookup(
        &self,
        peer: &PeerNode,
        lookup_msg: Lookup,
    ) -> Result<Vec<PeerNode>, CommError> {
        let start = Instant::now();
        // Create channel
        let channel = match self.client_channel(peer).await {
            Ok(c) => c,
            Err(_) => {
                log::error!("Failed to connect to peer for lookup");
                return Ok(Vec::new()); // Return empty list for connection failures
            }
        };

        // Create client
        let mut client = KademliaRpcServiceClient::new(channel.clone());

        // Execute the operation with timeout
        let result = tokio::time::timeout(
            self.timeout,
            client.send_lookup(tonic::Request::new(lookup_msg)),
        )
        .await;

        // Cleanup: channel will be dropped automatically
        drop(channel);

        let duration = start.elapsed();
        metrics::histogram!(LOOKUP_TIME_METRIC, "source" => DISCOVERY_GRPC_METRICS_SOURCE).record(duration.as_millis() as f64);

        match result {
            Ok(Ok(response)) => {
                let lookup_response = response.into_inner();
                if lookup_response.network_id == self.network_id {
                    // Convert nodes to PeerNodes and filter valid ones
                    let mut valid_peers = Vec::new();
                    for node in lookup_response.nodes {
                        let peer_node = to_peer_node(&node);
                        if self.is_valid_peer(&peer_node).await {
                            valid_peers.push(peer_node);
                        }
                    }
                    Ok(valid_peers)
                } else {
                    log::warn!("Network ID mismatch in lookup response");
                    Ok(Vec::new())
                }
            }
            Ok(Err(status)) => {
                log::error!("Lookup failed: {:?}", status);
                Ok(Vec::new())
            }
            Err(_) => {
                log::error!("Lookup timed out");
                Ok(Vec::new())
            }
        }
    }

    /// Validate if a peer has a valid address
    async fn is_valid_peer(&self, peer: &PeerNode) -> bool {
        if self.allow_private_addresses {
            is_valid_inet_address(&peer.endpoint.host).unwrap_or(false)
        } else {
            is_valid_public_inet_address(&peer.endpoint.host).unwrap_or(false)
        }
    }
}

#[async_trait]
impl KademliaRPC for GrpcKademliaRPC {
    async fn ping(&self, peer: &PeerNode) -> Result<bool, CommError> {
        metrics::counter!(PING_METRIC, "source" => DISCOVERY_GRPC_METRICS_SOURCE).increment(1);
        let ping_msg = Ping {
            sender: Some(to_node(&self.local_peer)),
            network_id: self.network_id.clone(),
        };

        self.with_client_ping(peer, ping_msg).await
    }

    async fn lookup(&self, key: &[u8], peer: &PeerNode) -> Result<Vec<PeerNode>, CommError> {
        metrics::counter!(LOOKUP_METRIC, "source" => DISCOVERY_GRPC_METRICS_SOURCE).increment(1);
        let lookup_msg = Lookup {
            id: Bytes::from(key.to_vec()),
            sender: Some(to_node(&self.local_peer)),
            network_id: self.network_id.clone(),
        };

        self.with_client_lookup(peer, lookup_msg).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::peer_node::{Endpoint, NodeIdentifier};
    use std::{sync::Once, time::Duration};

    static INIT: Once = Once::new();

    fn init_logger() {
        INIT.call_once(|| {
            env_logger::builder()
                .is_test(true) // ensures logs show up in test output
                .filter_level(log::LevelFilter::Debug)
                .try_init()
                .unwrap();
        });
    }

    fn test_peer() -> PeerNode {
        let id = NodeIdentifier {
            key: Bytes::from("test_key".as_bytes().to_vec()),
        };
        let endpoint = Endpoint::new("localhost".to_string(), 8080, 8080);
        PeerNode { id, endpoint }
    }

    #[test]
    fn test_grpc_kademlia_rpc_creation() {
        init_logger();
        let local_peer = test_peer();
        let rpc = GrpcKademliaRPC::new(
            "test_network".to_string(),
            Duration::from_millis(500),
            true,
            local_peer,
        );

        assert_eq!(rpc.network_id, "test_network");
        assert_eq!(rpc.timeout, Duration::from_millis(500));
        assert!(rpc.allow_private_addresses);
    }

    #[test]
    fn test_node_conversions() {
        init_logger();
        let peer = test_peer();
        let comm_node = to_node(&peer);
        let converted_back = to_peer_node(&comm_node);

        assert_eq!(peer.id.key, converted_back.id.key);
        assert_eq!(peer.endpoint.host, converted_back.endpoint.host);
        assert_eq!(peer.endpoint.tcp_port, converted_back.endpoint.tcp_port);
        assert_eq!(peer.endpoint.udp_port, converted_back.endpoint.udp_port);
    }

    #[tokio::test]
    async fn test_ping_timeout_behavior() {
        init_logger();
        let local_peer = test_peer();
        let rpc = GrpcKademliaRPC::new(
            "test_network".to_string(),
            Duration::from_millis(100), // Reasonable timeout for localhost connection attempts
            true,
            local_peer.clone(),
        );

        let non_existent_peer = PeerNode {
            id: NodeIdentifier {
                key: Bytes::from("other_key".as_bytes().to_vec()),
            },
            endpoint: Endpoint::new("127.0.0.1".to_string(), 65432, 65432), // Localhost with high port
        };

        // Test that ping returns false for non-existent peer (should fail quickly)
        let ping_result = rpc.ping(&non_existent_peer).await;
        assert!(ping_result.is_ok());
        assert_eq!(ping_result.unwrap(), false);
    }

    #[tokio::test]
    async fn test_lookup_timeout_behavior() {
        init_logger();
        let local_peer = test_peer();
        let rpc = GrpcKademliaRPC::new(
            "test_network".to_string(),
            Duration::from_millis(100), // Reasonable timeout for localhost connection attempts
            true,
            local_peer.clone(),
        );

        let non_existent_peer = PeerNode {
            id: NodeIdentifier {
                key: Bytes::from("other_key".as_bytes().to_vec()),
            },
            endpoint: Endpoint::new("127.0.0.1".to_string(), 65433, 65433), // Localhost with high port
        };

        let key = b"lookup_key";

        // This should fail quickly and return empty list
        let result = rpc.lookup(key, &non_existent_peer).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Vec::<PeerNode>::new());
    }
}
