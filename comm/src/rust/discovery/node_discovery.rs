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

pub trait NodeDiscovery {
    fn discover(&self) -> Result<(), CommError>;

    fn peers(&self) -> Result<Vec<PeerNode>, CommError>;
}

pub fn kademlia<T: KademliaRPC + Clone>(
    id: NodeIdentifier,
    kademlia_rpc: Arc<T>,
) -> KademliaNodeDiscovery<T> {
    KademliaNodeDiscovery::new(KademliaStore::new(id, kademlia_rpc.clone()), kademlia_rpc)
}
