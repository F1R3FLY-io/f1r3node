// See comm/src/test/scala/coop/rchain/comm/rp/ConnectionsSpec.scala

use prost::bytes::Bytes;

#[cfg(test)]
mod tests {
    use comm::rust::{
        peer_node::{Endpoint, NodeIdentifier, PeerNode},
        rp::connect::Connections,
    };

    use super::*;

    /// Helper function to create an endpoint with given host and port
    fn endpoint(host: &str, port: u32) -> Endpoint {
        Endpoint::new(host.to_string(), port, port)
    }

    /// Helper function to create a peer with given name, host, and port
    fn peer(name: &str, host: Option<&str>, port: Option<u32>) -> PeerNode {
        let host = host.unwrap_or("host");
        let port = port.unwrap_or(8080);

        // Create NodeIdentifier from name bytes
        let key = Bytes::from(name.as_bytes().to_vec());
        let id = NodeIdentifier { key };

        PeerNode {
            id,
            endpoint: endpoint(host, port),
        }
    }

    /// Helper function to create a peer with just name (uses default host and port)
    fn peer_simple(name: &str) -> PeerNode {
        peer(name, None, None)
    }

    /// Helper function to create a peer with name, host, and port
    fn peer_with_endpoint(name: &str, host: &str, port: u32) -> PeerNode {
        peer(name, Some(host), Some(port))
    }

    #[test]
    fn test_add_conn_new_peers_at_end() {
        // when
        let connections = Connections::empty()
            .add_conn(peer_simple("A"))
            .unwrap()
            .add_conn(peer_simple("B"))
            .unwrap()
            .add_conn(peer_simple("C"))
            .unwrap();

        // then
        let expected = vec![peer_simple("A"), peer_simple("B"), peer_simple("C")];
        assert_eq!(connections.into_vec(), expected);
    }

    #[test]
    fn test_add_conn_no_duplicate_peers() {
        // when
        let connections = Connections::empty()
            .add_conn(peer_simple("A"))
            .unwrap()
            .add_conn(peer_simple("A"))
            .unwrap();

        // then
        let expected = vec![peer_simple("A")];
        assert_eq!(connections.into_vec(), expected);
    }

    #[test]
    fn test_add_conn_move_peer_to_end() {
        // when
        let connections = Connections::empty()
            .add_conn(peer_simple("A"))
            .unwrap()
            .add_conn(peer_simple("B"))
            .unwrap()
            .add_conn(peer_simple("C"))
            .unwrap()
            .add_conn(peer_simple("A"))
            .unwrap();

        // then
        let expected = vec![peer_simple("B"), peer_simple("C"), peer_simple("A")];
        assert_eq!(connections.into_vec(), expected);
    }

    #[test]
    fn test_add_conn_update_endpoint_information() {
        // when
        let connections = Connections::empty()
            .add_conn(peer_with_endpoint("A", "10.10.0.1", 80))
            .unwrap()
            .add_conn(peer_simple("B"))
            .unwrap()
            .add_conn(peer_with_endpoint("A", "10.11.11.11", 8080))
            .unwrap();

        // then
        let expected = vec![
            peer_simple("B"),
            peer_with_endpoint("A", "10.11.11.11", 8080),
        ];
        assert_eq!(connections.into_vec(), expected);
    }

    #[test]
    fn test_remove_conn_list_unmodified_if_not_in_list() {
        // when
        let connections = Connections::empty()
            .add_conn(peer_simple("A"))
            .unwrap()
            .add_conn(peer_simple("B"))
            .unwrap()
            .remove_conn(peer_simple("C"))
            .unwrap();

        // then
        let expected = vec![peer_simple("A"), peer_simple("B")];
        assert_eq!(connections.into_vec(), expected);
    }

    #[test]
    fn test_remove_conn_removes_existing_connection() {
        // when
        let connections = Connections::empty()
            .add_conn(peer_simple("A"))
            .unwrap()
            .add_conn(peer_simple("B"))
            .unwrap()
            .add_conn(peer_simple("C"))
            .unwrap()
            .remove_conn(peer_simple("B"))
            .unwrap();

        // then
        let expected = vec![peer_simple("A"), peer_simple("C")];
        assert_eq!(connections.into_vec(), expected);
    }
}
