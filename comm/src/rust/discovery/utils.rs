// See comm/src/main/scala/coop/rchain/comm/discovery/package.scala

use prost::bytes::Bytes;
use shared::rust::grpc::grpc_server::GrpcServer;
use tonic::transport::Server as TonicServer;

use crate::{
    comm::{kademlia_rpc_service_server::KademliaRpcServiceServer, Node},
    rust::{
        discovery::grpc_kademlia_rpc_server::{GrpcKademliaRPCServer, LookupHandler, PingHandler},
        errors::CommError,
        peer_node::{Endpoint, NodeIdentifier, PeerNode},
    },
};

/// Converts a protobuf Node to a PeerNode
pub fn to_peer_node(n: &Node) -> PeerNode {
    PeerNode {
        id: NodeIdentifier { key: n.id.clone() },
        endpoint: Endpoint::new(
            String::from_utf8_lossy(&n.host).to_string(),
            n.tcp_port,
            n.udp_port,
        ),
    }
}

/// Converts a PeerNode to a protobuf Node  
pub fn to_node(n: &PeerNode) -> Node {
    Node {
        id: n.id.key.clone(),
        host: Bytes::from(n.endpoint.host.as_bytes().to_vec()),
        udp_port: n.endpoint.udp_port,
        tcp_port: n.endpoint.tcp_port,
    }
}

/// Acquire a Kademlia RPC server
pub async fn acquire_kademlia_rpc_server(
    network_id: String,
    port: u16,
    ping_handler: PingHandler,
    lookup_handler: LookupHandler,
) -> Result<GrpcServer, CommError> {
    // Create the Kademlia RPC server implementation
    let kademlia_server = GrpcKademliaRPCServer::new(network_id, ping_handler, lookup_handler);

    // Create the tonic service
    let service = KademliaRpcServiceServer::new(kademlia_server);

    // Create the server builder and add service
    let router = TonicServer::builder().add_service(service);

    // Create a GrpcServer instance
    let mut grpc_server = GrpcServer::new(port);

    // Start the server with the router
    grpc_server
        .start_with_router(router)
        .await
        .map_err(|e| CommError::InternalCommunicationError(e.to_string()))?;

    Ok(grpc_server)
}
