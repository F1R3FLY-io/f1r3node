// See comm/src/main/scala/coop/rchain/comm/discovery/NodeDiscovery.scala

use std::sync::Arc;

use crate::rust::{
    errors::CommError,
    peer_node::{NodeIdentifier, PeerNode},
};

use super::{
    kademlia_node_discovery::KademliaNodeDiscovery, kademlia_rpc::KademliaRPC,
    kademlia_store::KademliaStore,
};

#[async_trait::async_trait]
pub trait NodeDiscovery: Send + Sync {
    async fn discover(&self) -> Result<(), CommError>;

    fn peers(&self) -> Result<Vec<PeerNode>, CommError>;

    /// Remove a peer from the discovery store (e.g., Kademlia table)
    ///
    /// This should be called when a peer fails health checks to ensure
    /// aggressive cleanup of the discovery table, not just the connections.
    fn remove_peer(&self, peer: &PeerNode) -> Result<(), CommError>;
}

pub fn kademlia<T: KademliaRPC + Send + Sync + 'static>(
    id: NodeIdentifier,
    kademlia_rpc: Arc<T>,
    kademlia_store: Arc<KademliaStore<T>>,
) -> KademliaNodeDiscovery<T> {
    KademliaNodeDiscovery::new(kademlia_store, kademlia_rpc, id)
}
