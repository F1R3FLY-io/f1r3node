// See comm/src/main/scala/coop/rchain/comm/discovery/NodeDiscovery.scala

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

pub fn kademlia<'a, T: KademliaRPC + Clone>(
    id: NodeIdentifier,
    kademlia_rpc: &'a T,
) -> KademliaNodeDiscovery<'a, T> {
    KademliaNodeDiscovery::new(KademliaStore::new(id, kademlia_rpc), kademlia_rpc)
}
