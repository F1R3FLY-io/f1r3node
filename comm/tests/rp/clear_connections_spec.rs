// See comm/src/test/scala/coop/rchain/comm/rp/ClearConnectionsSpec.scala

use prost::bytes::Bytes;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use comm::rust::test_instances::{NodeDiscoveryStub, TransportLayerStub, NETWORK_ID};
use comm::rust::{
    errors::{timeout, CommError},
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
    rp::{
        connect::{clear_connections, Connections, ConnectionsCell},
        rp_conf::{ClearConnectionsConf, RPConf},
    },
};

/// Helper function to create a peer with given name and default host/port
fn peer(name: &str) -> PeerNode {
    peer_with_host(name, "host")
}

/// Helper function to create a peer with given name and host
fn peer_with_host(name: &str, host: &str) -> PeerNode {
    let key = Bytes::from(name.as_bytes().to_vec());
    let id = NodeIdentifier { key };
    let endpoint = Endpoint::new(host.to_string(), 80, 80);
    PeerNode { id, endpoint }
}

/// Helper function to create a ConnectionsCell with given peers
fn mk_connections(peers: &[PeerNode]) -> ConnectionsCell {
    let connections_cell = ConnectionsCell::new();
    let connections = Connections::from_vec(peers.to_vec());
    connections_cell.flat_modify(|_| Ok(connections)).unwrap();
    connections_cell
}

/// Helper function to create RPConf for testing
fn conf(
    max_num_of_connections: usize,
    num_of_connections_pinged: Option<usize>,
    bootstrap: Option<PeerNode>,
) -> RPConf {
    RPConf {
        local: peer("src"),
        network_id: NETWORK_ID.to_string(),
        bootstrap,
        default_timeout: Duration::from_millis(1),
        max_num_of_connections,
        clear_connections: ClearConnectionsConf::new(num_of_connections_pinged.unwrap_or(5)),
    }
}

/// Always successful response function
fn always_success(
    _peer: &PeerNode,
    _protocol: &models::routing::Protocol,
) -> Result<(), CommError> {
    Ok(())
}

/// Always failing response function
fn always_fail(_peer: &PeerNode, _protocol: &models::routing::Protocol) -> Result<(), CommError> {
    Err(timeout())
}

/// NodeDiscovery implementation that tracks which peers were removed.
/// Used to verify bootstrap pinning behavior.
struct TrackingNodeDiscovery {
    removed_keys: Arc<Mutex<Vec<Bytes>>>,
}

impl TrackingNodeDiscovery {
    fn new() -> Self {
        Self {
            removed_keys: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn get_removed_keys(&self) -> Vec<Bytes> {
        self.removed_keys.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl comm::rust::discovery::node_discovery::NodeDiscovery for TrackingNodeDiscovery {
    async fn discover(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn peers(&self) -> Result<Vec<PeerNode>, CommError> {
        Ok(Vec::new())
    }

    fn remove_peer(&self, peer: &PeerNode) -> Result<(), CommError> {
        self.removed_keys.lock().unwrap().push(peer.id.key.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_clear_connections_small_number_should_not_clear_any() {
        // given
        let connections = mk_connections(&[peer("A"), peer("B")]);
        let rp_conf = conf(5, None, None);
        let transport = TransportLayerStub::new();
        transport.set_responses(always_success);
        let discovery = NodeDiscoveryStub::new();

        // when
        let (cleared, failed_peers) =
            clear_connections(&connections, &rp_conf, &transport, &discovery)
                .await
                .unwrap();

        // then
        let final_connections = connections.read().unwrap();
        assert_eq!(final_connections.len(), 2);
        assert!(final_connections.as_slice().contains(&peer("A")));
        assert!(final_connections.as_slice().contains(&peer("B")));
        assert_eq!(cleared, 0);
        assert!(failed_peers.is_empty());
    }

    #[tokio::test]
    async fn test_clear_connections_should_report_zero_cleared() {
        // given
        let connections = mk_connections(&[peer("A"), peer("B")]);
        let rp_conf = conf(5, None, None);
        let transport = TransportLayerStub::new();
        transport.set_responses(always_success);
        let discovery = NodeDiscoveryStub::new();

        // when
        let (cleared, failed_peers) =
            clear_connections(&connections, &rp_conf, &transport, &discovery)
                .await
                .unwrap();

        // then
        assert_eq!(cleared, 0);
        assert!(failed_peers.is_empty());
    }

    #[tokio::test]
    async fn test_clear_connections_should_ping_first_few_nodes_with_heartbeat() {
        // given
        let connections = mk_connections(&[peer("A"), peer("B"), peer("C"), peer("D")]);
        let rp_conf = conf(5, Some(2), None);
        let transport = TransportLayerStub::new();
        transport.set_responses(always_success);
        let discovery = NodeDiscoveryStub::new();

        // when
        let _ = clear_connections(&connections, &rp_conf, &transport, &discovery)
            .await
            .unwrap();

        // then
        assert_eq!(transport.request_count(), 2);
        let requests = transport.get_all_requests();
        let peer_names: Vec<String> = requests
            .iter()
            .map(|req| String::from_utf8(req.peer.id.key.to_vec()).unwrap())
            .collect();
        assert!(peer_names.contains(&"A".to_string()));
        assert!(peer_names.contains(&"B".to_string()));
    }

    #[tokio::test]
    async fn test_clear_connections_should_remove_peers_that_did_not_respond() {
        // given
        let connections = mk_connections(&[peer("A"), peer("B"), peer("C"), peer("D")]);
        let rp_conf = conf(5, Some(2), None);
        let transport = TransportLayerStub::new();

        // Set responses: A fails, others succeed
        transport.set_responses(|peer, _protocol| {
            let peer_name = String::from_utf8(peer.id.key.to_vec()).unwrap();
            if peer_name == "A" {
                always_fail(peer, _protocol)
            } else {
                always_success(peer, _protocol)
            }
        });
        let discovery = NodeDiscoveryStub::new();

        // when
        let (cleared, failed_peers) =
            clear_connections(&connections, &rp_conf, &transport, &discovery)
                .await
                .unwrap();

        // then
        let final_connections = connections.read().unwrap();
        assert_eq!(final_connections.len(), 3);
        assert!(!final_connections.as_slice().contains(&peer("A")));
        assert!(final_connections.as_slice().contains(&peer("B")));
        assert!(final_connections.as_slice().contains(&peer("C")));
        assert!(final_connections.as_slice().contains(&peer("D")));

        // Verify failed peers are returned
        assert_eq!(cleared, 1);
        assert_eq!(failed_peers.len(), 1);
        assert!(failed_peers.contains(&peer("A")));
    }

    #[tokio::test]
    async fn test_clear_connections_should_put_responding_peers_to_end_of_list() {
        // given
        let connections = mk_connections(&[peer("A"), peer("B"), peer("C"), peer("D")]);
        let rp_conf = conf(5, Some(3), None);
        let transport = TransportLayerStub::new();

        // Set responses: A fails, others succeed
        transport.set_responses(|peer, _protocol| {
            let peer_name = String::from_utf8(peer.id.key.to_vec()).unwrap();
            if peer_name == "A" {
                always_fail(peer, _protocol)
            } else {
                always_success(peer, _protocol)
            }
        });
        let discovery = NodeDiscoveryStub::new();

        // when
        let _ = clear_connections(&connections, &rp_conf, &transport, &discovery)
            .await
            .unwrap();

        // then
        let final_connections = connections.read().unwrap();
        assert_eq!(final_connections.len(), 3);

        // The order should be: D (not pinged), B, C (pinged and successful, moved to end)
        let connection_vec = final_connections.as_slice();
        assert_eq!(connection_vec[0], peer("D"));
        assert_eq!(connection_vec[1], peer("B"));
        assert_eq!(connection_vec[2], peer("C"));
    }

    #[tokio::test]
    async fn test_clear_connections_should_report_number_of_removed_connections() {
        // given
        let connections = mk_connections(&[peer("A"), peer("B"), peer("C"), peer("D")]);
        let rp_conf = conf(5, Some(3), None);
        let transport = TransportLayerStub::new();

        // Set responses: A fails, others succeed
        transport.set_responses(|peer, _protocol| {
            let peer_name = String::from_utf8(peer.id.key.to_vec()).unwrap();
            if peer_name == "A" {
                always_fail(peer, _protocol)
            } else {
                always_success(peer, _protocol)
            }
        });
        let discovery = NodeDiscoveryStub::new();

        // when
        let (cleared, failed_peers) =
            clear_connections(&connections, &rp_conf, &transport, &discovery)
                .await
                .unwrap();

        // then
        assert_eq!(cleared, 1);
        assert_eq!(failed_peers.len(), 1);
        assert!(failed_peers.contains(&peer("A")));
    }

    #[tokio::test]
    async fn test_should_not_remove_bootstrap_peer_from_kademlia_when_heartbeat_fails() {
        // given: peer("A") is the bootstrap peer and its heartbeat fails
        let connections = mk_connections(&[peer("A"), peer("B")]);
        let rp_conf = conf(5, Some(2), Some(peer("A")));
        let transport = TransportLayerStub::new();
        transport.set_responses(|peer, _protocol| {
            let peer_name = String::from_utf8(peer.id.key.to_vec()).unwrap();
            if peer_name == "A" {
                always_fail(peer, _protocol)
            } else {
                always_success(peer, _protocol)
            }
        });
        let discovery = TrackingNodeDiscovery::new();

        // when
        let (cleared, _failed_peers) =
            clear_connections(&connections, &rp_conf, &transport, &discovery)
                .await
                .unwrap();

        // then: peer("A") should NOT have been removed from Kademlia
        assert_eq!(cleared, 1);
        let removed_keys = discovery.get_removed_keys();
        assert!(
            !removed_keys.contains(&peer("A").id.key),
            "Bootstrap peer should not be removed from KademliaStore"
        );

        // but peer("A") SHOULD be removed from ConnectionsCell (TCP cleanup)
        let final_connections = connections.read().unwrap();
        assert!(!final_connections.as_slice().contains(&peer("A")));
        assert!(final_connections.as_slice().contains(&peer("B")));
    }

    #[tokio::test]
    async fn test_should_still_remove_non_bootstrap_peers_from_kademlia_when_heartbeat_fails() {
        // given: peer("BOOT") is the bootstrap, but peer("A") fails heartbeat
        let connections = mk_connections(&[peer("A"), peer("B")]);
        let rp_conf = conf(5, Some(2), Some(peer("BOOT")));
        let transport = TransportLayerStub::new();
        transport.set_responses(|peer, _protocol| {
            let peer_name = String::from_utf8(peer.id.key.to_vec()).unwrap();
            if peer_name == "A" {
                always_fail(peer, _protocol)
            } else {
                always_success(peer, _protocol)
            }
        });
        let discovery = TrackingNodeDiscovery::new();

        // when
        let (cleared, _failed_peers) =
            clear_connections(&connections, &rp_conf, &transport, &discovery)
                .await
                .unwrap();

        // then: peer("A") SHOULD have been removed from Kademlia (it's not bootstrap)
        assert_eq!(cleared, 1);
        let removed_keys = discovery.get_removed_keys();
        assert!(
            removed_keys.contains(&peer("A").id.key),
            "Non-bootstrap peer should be removed from KademliaStore"
        );
    }
}
