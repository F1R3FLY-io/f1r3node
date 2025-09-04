// See comm/src/main/scala/coop/rchain/comm/discovery/GrpcKademliaRPCServer.scala

use std::future::Future;
use std::pin::Pin;

use async_trait::async_trait;
use tonic::{Request, Response, Status};

use crate::{
    comm::{kademlia_rpc_service_server::KademliaRpcService, Lookup, LookupResponse, Ping, Pong},
    rust::{
        discovery::utils::{to_node, to_peer_node},
        peer_node::PeerNode,
    },
};

/// Type alias for ping handler function
pub type PingHandler =
    Box<dyn Fn(&PeerNode) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Type alias for lookup handler function  
pub type LookupHandler = Box<
    dyn Fn(&PeerNode, &[u8]) -> Pin<Box<dyn Future<Output = Vec<PeerNode>> + Send>> + Send + Sync,
>;

/// Rust implementation of GrpcKademliaRPCServer
pub struct GrpcKademliaRPCServer {
    network_id: String,
    ping_handler: PingHandler,
    lookup_handler: LookupHandler,
}

impl GrpcKademliaRPCServer {
    pub fn new(
        network_id: String,
        ping_handler: PingHandler,
        lookup_handler: LookupHandler,
    ) -> Self {
        Self {
            network_id,
            ping_handler,
            lookup_handler,
        }
    }
}

#[async_trait]
impl KademliaRpcService for GrpcKademliaRPCServer {
    /// Handle incoming ping requests
    async fn send_ping(&self, request: Request<Ping>) -> Result<Response<Pong>, Status> {
        let ping = request.into_inner();

        if ping.network_id == self.network_id {
            // Network ID matches - process the ping
            if let Some(sender_node) = ping.sender {
                let sender: PeerNode = to_peer_node(&sender_node);

                // Call the ping handler
                (self.ping_handler)(&sender).await;

                // Return successful pong with matching network ID
                let pong = Pong {
                    network_id: self.network_id.clone(),
                };
                Ok(Response::new(pong))
            } else {
                Err(Status::invalid_argument("Missing sender in ping request"))
            }
        } else {
            // Network ID mismatch - return pong with our network ID
            let pong = Pong {
                network_id: self.network_id.clone(),
            };
            Ok(Response::new(pong))
        }
    }

    /// Handle incoming lookup requests
    async fn send_lookup(
        &self,
        request: Request<Lookup>,
    ) -> Result<Response<LookupResponse>, Status> {
        let lookup = request.into_inner();

        if lookup.network_id == self.network_id {
            // Network ID matches - process the lookup
            if let Some(sender_node) = lookup.sender {
                let id = lookup.id.to_vec();
                let sender: PeerNode = to_peer_node(&sender_node);

                // Call the lookup handler
                let peers = (self.lookup_handler)(&sender, &id).await;

                // Convert peers to protobuf nodes and create response
                let nodes: Vec<_> = peers.iter().map(to_node).collect();
                let response = LookupResponse {
                    nodes,
                    network_id: self.network_id.clone(),
                };
                Ok(Response::new(response))
            } else {
                Err(Status::invalid_argument("Missing sender in lookup request"))
            }
        } else {
            // Network ID mismatch - return empty response
            let response = LookupResponse {
                nodes: Vec::new(),
                network_id: self.network_id.clone(),
            };
            Ok(Response::new(response))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        comm::Node,
        rust::peer_node::{Endpoint, NodeIdentifier},
    };
    use std::sync::{Arc, Mutex};

    fn test_peer() -> PeerNode {
        PeerNode {
            id: NodeIdentifier {
                key: prost::bytes::Bytes::from("test_key".as_bytes().to_vec()),
            },
            endpoint: Endpoint::new("localhost".to_string(), 8080, 8080),
        }
    }

    fn test_node() -> Node {
        to_node(&test_peer())
    }

    #[tokio::test]
    async fn test_ping_with_matching_network_id() {
        let network_id = "test_network".to_string();
        let ping_called = Arc::new(Mutex::new(false));
        let ping_called_clone = ping_called.clone();

        // Create ping handler that sets a flag when called
        let ping_handler: PingHandler = Box::new(move |_peer| {
            let flag = ping_called_clone.clone();
            Box::pin(async move {
                *flag.lock().unwrap() = true;
            })
        });

        // Dummy lookup handler
        let lookup_handler: LookupHandler =
            Box::new(|_peer, _key| Box::pin(async move { Vec::new() }));

        let server = GrpcKademliaRPCServer::new(network_id.clone(), ping_handler, lookup_handler);

        // Create ping request with matching network ID
        let ping = Ping {
            sender: Some(test_node()),
            network_id: network_id.clone(),
        };
        let request = Request::new(ping);

        // Process ping
        let response = server.send_ping(request).await.unwrap();
        let pong = response.into_inner();

        // Verify response
        assert_eq!(pong.network_id, network_id);
        assert!(*ping_called.lock().unwrap()); // Handler should have been called
    }

    #[tokio::test]
    async fn test_ping_with_mismatched_network_id() {
        let network_id = "test_network".to_string();
        let ping_called = Arc::new(Mutex::new(false));
        let ping_called_clone = ping_called.clone();

        // Create ping handler that sets a flag when called
        let ping_handler: PingHandler = Box::new(move |_peer| {
            let flag = ping_called_clone.clone();
            Box::pin(async move {
                *flag.lock().unwrap() = true;
            })
        });

        // Dummy lookup handler
        let lookup_handler: LookupHandler =
            Box::new(|_peer, _key| Box::pin(async move { Vec::new() }));

        let server = GrpcKademliaRPCServer::new(network_id.clone(), ping_handler, lookup_handler);

        // Create ping request with different network ID
        let ping = Ping {
            sender: Some(test_node()),
            network_id: "different_network".to_string(),
        };
        let request = Request::new(ping);

        // Process ping
        let response = server.send_ping(request).await.unwrap();
        let pong = response.into_inner();

        // Verify response - should return our network ID, not the request's
        assert_eq!(pong.network_id, network_id);
        assert!(!*ping_called.lock().unwrap()); // Handler should NOT have been called
    }

    #[tokio::test]
    async fn test_lookup_with_matching_network_id() {
        let network_id = "test_network".to_string();
        let lookup_called = Arc::new(Mutex::new(false));
        let lookup_called_clone = lookup_called.clone();

        // Dummy ping handler
        let ping_handler: PingHandler = Box::new(|_peer| Box::pin(async move {}));

        // Create lookup handler that returns test peers
        let lookup_handler: LookupHandler = Box::new(move |_peer, _key| {
            let flag = lookup_called_clone.clone();
            Box::pin(async move {
                *flag.lock().unwrap() = true;
                vec![test_peer()] // Return one test peer
            })
        });

        let server = GrpcKademliaRPCServer::new(network_id.clone(), ping_handler, lookup_handler);

        // Create lookup request with matching network ID
        let lookup = Lookup {
            id: prost::bytes::Bytes::from("lookup_key".as_bytes().to_vec()),
            sender: Some(test_node()),
            network_id: network_id.clone(),
        };
        let request = Request::new(lookup);

        // Process lookup
        let response = server.send_lookup(request).await.unwrap();
        let lookup_response = response.into_inner();

        // Verify response
        assert_eq!(lookup_response.network_id, network_id);
        assert_eq!(lookup_response.nodes.len(), 1); // Should return one peer
        assert!(*lookup_called.lock().unwrap()); // Handler should have been called
    }

    #[tokio::test]
    async fn test_lookup_with_mismatched_network_id() {
        let network_id = "test_network".to_string();
        let lookup_called = Arc::new(Mutex::new(false));
        let lookup_called_clone = lookup_called.clone();

        // Dummy ping handler
        let ping_handler: PingHandler = Box::new(|_peer| Box::pin(async move {}));

        // Create lookup handler that sets a flag when called
        let lookup_handler: LookupHandler = Box::new(move |_peer, _key| {
            let flag = lookup_called_clone.clone();
            Box::pin(async move {
                *flag.lock().unwrap() = true;
                vec![test_peer()]
            })
        });

        let server = GrpcKademliaRPCServer::new(network_id.clone(), ping_handler, lookup_handler);

        // Create lookup request with different network ID
        let lookup = Lookup {
            id: prost::bytes::Bytes::from("lookup_key".as_bytes().to_vec()),
            sender: Some(test_node()),
            network_id: "different_network".to_string(),
        };
        let request = Request::new(lookup);

        // Process lookup
        let response = server.send_lookup(request).await.unwrap();
        let lookup_response = response.into_inner();

        // Verify response
        assert_eq!(lookup_response.network_id, network_id); // Should return our network ID
        assert_eq!(lookup_response.nodes.len(), 0); // Should return empty list
        assert!(!*lookup_called.lock().unwrap()); // Handler should NOT have been called
    }

    #[tokio::test]
    async fn test_ping_missing_sender() {
        let network_id = "test_network".to_string();

        // Dummy handlers
        let ping_handler: PingHandler = Box::new(|_peer| Box::pin(async move {}));
        let lookup_handler: LookupHandler =
            Box::new(|_peer, _key| Box::pin(async move { Vec::new() }));

        let server = GrpcKademliaRPCServer::new(network_id.clone(), ping_handler, lookup_handler);

        // Create ping request without sender
        let ping = Ping {
            sender: None, // Missing sender
            network_id: network_id.clone(),
        };
        let request = Request::new(ping);

        // Process ping - should return error
        let result = server.send_ping(request).await;
        assert!(result.is_err());

        if let Err(status) = result {
            assert_eq!(status.code(), tonic::Code::InvalidArgument);
            assert!(status.message().contains("Missing sender"));
        }
    }

    #[tokio::test]
    async fn test_lookup_missing_sender() {
        let network_id = "test_network".to_string();

        // Dummy handlers
        let ping_handler: PingHandler = Box::new(|_peer| Box::pin(async move {}));
        let lookup_handler: LookupHandler =
            Box::new(|_peer, _key| Box::pin(async move { Vec::new() }));

        let server = GrpcKademliaRPCServer::new(network_id.clone(), ping_handler, lookup_handler);

        // Create lookup request without sender
        let lookup = Lookup {
            id: prost::bytes::Bytes::from("lookup_key".as_bytes().to_vec()),
            sender: None, // Missing sender
            network_id: network_id.clone(),
        };
        let request = Request::new(lookup);

        // Process lookup - should return error
        let result = server.send_lookup(request).await;
        assert!(result.is_err());

        if let Err(status) = result {
            assert_eq!(status.code(), tonic::Code::InvalidArgument);
            assert!(status.message().contains("Missing sender"));
        }
    }
}
