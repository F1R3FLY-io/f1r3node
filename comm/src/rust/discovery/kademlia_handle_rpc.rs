// See comm/src/main/scala/coop/rchain/comm/discovery/KademliaHandleRPC.scala

use std::sync::Arc;

use prost::bytes::Bytes;

use crate::rust::{errors::CommError, peer_node::PeerNode};

use super::kademlia_store::KademliaStore;
use shared::rust::shared::metrics::Metrics;

/// Metrics source for discovery kademlia handlers
const METRICS_SOURCE: &str = "discovery.kademlia";

/// Handle incoming ping requests
///
/// This function increments the "handle.ping" counter and updates the last seen
/// timestamp for the peer in the Kademlia store.
pub async fn handle_ping<T>(
    peer: PeerNode,
    store: Arc<KademliaStore<T>>,
    metrics: Option<Arc<dyn Metrics>>,
) -> Result<(), CommError>
where
    T: super::kademlia_rpc::KademliaRPC,
{
    if let Some(metrics_impl) = metrics {
        let counter_name = format!("{}.handle.ping", METRICS_SOURCE);
        if let Err(e) = metrics_impl.increment_counter(&counter_name) {
            log::warn!(
                "Failed to increment metrics counter {}: {}",
                counter_name,
                e
            );
        }
    }

    store.update_last_seen(&peer).await?;

    Ok(())
}

/// Handle incoming lookup requests
///
/// This function increments the "handle.lookup" counter, updates the last seen
/// timestamp for the peer, and performs a lookup in the Kademlia store.
pub async fn handle_lookup<T>(
    peer: PeerNode,
    id: Vec<u8>,
    store: Arc<KademliaStore<T>>,
    metrics: Option<Arc<dyn Metrics>>,
) -> Result<Vec<PeerNode>, CommError>
where
    T: super::kademlia_rpc::KademliaRPC,
{
    // Increment metrics counter if metrics are available
    if let Some(metrics_impl) = metrics {
        let counter_name = format!("{}.handle.lookup", METRICS_SOURCE);
        if let Err(e) = metrics_impl.increment_counter(&counter_name) {
            log::warn!(
                "Failed to increment metrics counter {}: {}",
                counter_name,
                e
            );
        }
    }

    store.update_last_seen(&peer).await?;

    let id_bytes = Bytes::from(id);
    store.lookup(&id_bytes)
}
