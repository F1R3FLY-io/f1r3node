//! Effects module for F1r3fly node. This is a porting to Rust of the Scala module `node/main/scala/coop/rchain/node/effects`.
//!
//! This module provides various effectful operations and clients for the F1r3fly node.

use std::sync::Arc;

use comm::rust::{
    discovery::{
        kademlia_rpc::KademliaRPC,
        kademlia_store::KademliaStore,
        node_discovery::{kademlia, NodeDiscovery},
    },
    peer_node::NodeIdentifier,
};

pub mod console_io;
pub mod repl_client;

pub async fn node_discover<T: KademliaRPC + Send + Sync + 'static>(
    id: NodeIdentifier,
    kademlia_rpc: Arc<T>,
    kademlia_store: Arc<KademliaStore<T>>,
) -> eyre::Result<Arc<dyn NodeDiscovery + Send + Sync + 'static>> {
    Ok(Arc::new(kademlia(id, kademlia_rpc, kademlia_store)))
}
