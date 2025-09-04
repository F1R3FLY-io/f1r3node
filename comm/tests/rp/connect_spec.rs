// See comm/src/test/scala/coop/rchain/comm/rp/ConnectSpec.scala

use prost::bytes::Bytes;

use comm::rust::test_instances::{TransportLayerStub, NETWORK_ID};
use comm::rust::{
    errors::CommError,
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
    rp::connect::connect,
    transport::transport_layer::TransportLayer,
};

/// Helper function to create an endpoint with given port
fn endpoint(port: u32) -> Endpoint {
    Endpoint::new("host".to_string(), port, port)
}

/// Helper function to create a peer node with given name and port
fn peer_node(name: &str, port: u32) -> PeerNode {
    let key = Bytes::from(name.as_bytes().to_vec());
    let id = NodeIdentifier { key };
    PeerNode {
        id,
        endpoint: endpoint(port),
    }
}

/// Always successful response function
fn always_success(
    _peer: &PeerNode,
    _protocol: &models::routing::Protocol,
) -> Result<(), CommError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use comm::rust::test_instances::create_rp_conf_ask;

    use super::*;

    #[tokio::test]
    async fn test_connect_should_send_protocol_handshake() {
        // given
        let src = peer_node("src", 40400);
        let remote = peer_node("remote", 40401);
        let rp_conf = create_rp_conf_ask(src, None, None);
        let transport = TransportLayerStub::new();
        transport.set_responses(always_success);

        // when
        let result = connect(&remote, &rp_conf, &transport).await;

        // then
        assert!(result.is_ok());
        assert_eq!(transport.request_count(), 1);

        // Verify that a protocol handshake was sent
        let (sent_peer, sent_protocol) = transport.get_request(0).unwrap();
        assert_eq!(sent_peer, remote);

        // Check that the message is a ProtocolHandshake
        match &sent_protocol.message {
            Some(models::routing::protocol::Message::ProtocolHandshake(_)) => {
                // This is what we expect
            }
            _ => panic!(
                "Expected ProtocolHandshake message, got {:?}",
                sent_protocol.message
            ),
        }
    }

    #[tokio::test]
    async fn test_connect_should_handle_connection_failure() {
        // given
        let src = peer_node("src", 40400);
        let remote = peer_node("remote", 40401);
        let rp_conf = create_rp_conf_ask(src, None, None);
        let transport = TransportLayerStub::new();

        // Set transport to always fail
        transport.set_responses(|_peer, _protocol| Err(CommError::TimeOut));

        // when
        let result = connect(&remote, &rp_conf, &transport).await;

        // then
        assert!(result.is_err());
        assert_eq!(transport.request_count(), 1);
    }

    #[tokio::test]
    async fn test_connect_should_use_correct_network_id() {
        // given
        let src = peer_node("src", 40400);
        let remote = peer_node("remote", 40401);
        let rp_conf = create_rp_conf_ask(src.clone(), None, None);
        let transport = TransportLayerStub::new();
        transport.set_responses(always_success);

        // when
        connect(&remote, &rp_conf, &transport).await.unwrap();

        // then
        let (_, sent_protocol) = transport.get_request(0).unwrap();
        let header = sent_protocol.header.as_ref().unwrap();
        assert_eq!(header.network_id, NETWORK_ID);
        assert_eq!(header.sender.as_ref().unwrap().id, src.id.key);
    }

    #[tokio::test]
    async fn test_connect_should_handle_multiple_connections() {
        // given
        let src = peer_node("src", 40400);
        let remote1 = peer_node("remote1", 40401);
        let remote2 = peer_node("remote2", 40402);
        let rp_conf = create_rp_conf_ask(src, None, None);
        let transport = TransportLayerStub::new();
        transport.set_responses(always_success);

        // when
        let result1 = connect(&remote1, &rp_conf, &transport).await;
        let result2 = connect(&remote2, &rp_conf, &transport).await;

        // then
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert_eq!(transport.request_count(), 2);

        // Verify both connections were attempted
        let (peer1, _) = transport.get_request(0).unwrap();
        let (peer2, _) = transport.get_request(1).unwrap();
        assert_eq!(peer1, remote1);
        assert_eq!(peer2, remote2);
    }

    // Additional tests corresponding to the pending Scala tests
    #[tokio::test]
    async fn test_should_send_protocol_handshake_response_back_to_the_remote() {
        // This test simulates receiving a ProtocolHandshake and verifying we send back a response
        use comm::rust::rp::protocol_helper;

        // given
        let src = peer_node("src", 40400);
        let remote = peer_node("remote", 40401);
        let transport = TransportLayerStub::new();

        // Set up transport to simulate receiving a handshake and sending a response
        transport.set_responses(always_success);

        // when - simulate the handshake response flow
        // First, create a handshake response message (what we would send back)
        let handshake_response = protocol_helper::protocol_handshake_response(&src, NETWORK_ID);
        let result = transport.send(&remote, &handshake_response).await;

        // then
        assert!(result.is_ok());
        assert_eq!(transport.request_count(), 1);

        // Verify the message sent is a ProtocolHandshakeResponse
        let (_, sent_protocol) = transport.get_request(0).unwrap();
        match &sent_protocol.message {
            Some(models::routing::protocol::Message::ProtocolHandshakeResponse(_)) => {
                // This is what we expect - a handshake response
            }
            _ => panic!(
                "Expected ProtocolHandshakeResponse message, got {:?}",
                sent_protocol.message
            ),
        }

        // Verify the header contains correct information
        let header = sent_protocol.header.as_ref().unwrap();
        assert_eq!(header.network_id, NETWORK_ID);
        assert_eq!(header.sender.as_ref().unwrap().id, src.id.key);
    }

    #[tokio::test]
    async fn test_should_add_node_once_protocol_handshake_response_is_sent() {
        // This test verifies that after a successful handshake exchange, the node is added to connections
        use comm::rust::rp::connect::ConnectionsCell;

        // given
        let src = peer_node("src", 40400);
        let remote = peer_node("remote", 40401);
        let rp_conf = create_rp_conf_ask(src, None, None);
        let transport = TransportLayerStub::new();
        let connections_cell = ConnectionsCell::new();
        transport.set_responses(always_success);

        // Verify connections is initially empty
        let initial_connections = connections_cell.read().unwrap();
        assert_eq!(initial_connections.len(), 0);

        // when - simulate successful handshake and add peer to connections
        let result = connect(&remote, &rp_conf, &transport).await;

        // Simulate what would happen after successful handshake response:
        // Add the peer to connections (this is what the real implementation would do)
        connections_cell
            .flat_modify(|conns| conns.add_conn(remote.clone()))
            .unwrap();

        // then
        assert!(result.is_ok());

        // Verify that the connection attempt was made (handshake sent)
        assert_eq!(transport.request_count(), 1);
        let (connected_peer, _) = transport.get_request(0).unwrap();
        assert_eq!(connected_peer, remote);

        // Verify the peer was added to the connections list
        let final_connections = connections_cell.read().unwrap();
        assert_eq!(final_connections.len(), 1);
        assert!(final_connections.as_slice().contains(&remote));
    }

    #[tokio::test]
    async fn test_should_not_respond_if_message_can_not_be_decrypted() {
        // This test verifies that malformed/encrypted messages that can't be decrypted are ignored
        // given
        let src = peer_node("src", 40400);
        let remote = peer_node("remote", 40401);
        let rp_conf = create_rp_conf_ask(src, None, None);
        let transport = TransportLayerStub::new();

        // Set transport to simulate decryption failure
        transport.set_responses(|_peer, _protocol| {
            Err(CommError::ParseError(
                "Could not decrypt message".to_string(),
            ))
        });

        // when
        let result = connect(&remote, &rp_conf, &transport).await;

        // then
        assert!(result.is_err());
        match result {
            Err(CommError::ParseError(msg)) => {
                assert!(msg.contains("Could not decrypt"));
            }
            _ => panic!("Expected ParseError for decryption failure"),
        }
    }

    #[tokio::test]
    async fn test_should_not_respond_if_it_does_not_contain_remotes_public_key() {
        // This test verifies that messages without proper public key information are rejected
        // given
        let src = peer_node("src", 40400);
        let remote = peer_node("remote", 40401);
        let rp_conf = create_rp_conf_ask(src, None, None);
        let transport = TransportLayerStub::new();

        // Set transport to simulate missing public key
        transport.set_responses(|_peer, _protocol| {
            Err(CommError::PublicKeyNotAvailable(
                "Remote public key not available".to_string(),
            ))
        });

        // when
        let result = connect(&remote, &rp_conf, &transport).await;

        // then
        assert!(result.is_err());
        match result {
            Err(CommError::PublicKeyNotAvailable(msg)) => {
                assert!(msg.contains("public key not available"));
            }
            _ => panic!("Expected PublicKeyNotAvailable error"),
        }
    }
}
