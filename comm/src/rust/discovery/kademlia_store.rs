// See comm/src/main/scala/coop/rchain/comm/discovery/KademliaStore.scala

use std::sync::Arc;

use prost::bytes::Bytes;

use crate::rust::{
    errors::CommError,
    peer_node::{NodeIdentifier, PeerNode},
};

use super::{kademlia_rpc::KademliaRPC, peer_table::PeerTable};

pub struct KademliaStore<T: KademliaRPC> {
    table: PeerTable<T>,
}

impl<T: KademliaRPC> KademliaStore<T> {
    pub fn new(id: NodeIdentifier, kademlia_rpc: Arc<T>) -> Self {
        Self {
            table: PeerTable::new(id.key, None, None, kademlia_rpc),
        }
    }

    pub fn peers(&self) -> Result<Vec<PeerNode>, CommError> {
        self.table.peers()
    }

    pub fn sparseness(&self) -> Result<Vec<usize>, CommError> {
        self.table.sparseness()
    }

    pub fn lookup(&self, key: &Bytes) -> Result<Vec<PeerNode>, CommError> {
        self.table.lookup(key)
    }

    pub fn find(&self, key: &Bytes) -> Result<Option<PeerNode>, CommError> {
        self.table.find(key)
    }

    pub fn remove(&self, key: &Bytes) -> Result<(), CommError> {
        self.table.remove(key)
    }

    pub async fn update_last_seen(&self, peer_node: &PeerNode) -> Result<(), CommError> {
        self.table.update_last_seen(peer_node).await
    }
}
