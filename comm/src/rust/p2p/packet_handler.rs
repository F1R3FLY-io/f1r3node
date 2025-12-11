// See comm/src/main/scala/coop/rchain/p2p/effects/PacketHandler.scala

use async_trait::async_trait;
use models::routing::Packet;

use crate::rust::{
    errors::{unknown_protocol, CommError},
    peer_node::PeerNode,
};

/// Trait for handling packets from peers
#[async_trait]
pub trait PacketHandler: Send + Sync {
    async fn handle_packet(&self, peer: &PeerNode, packet: &Packet) -> Result<(), CommError>;
}

/// A no-operation packet handler that does nothing
pub struct NOPPacketHandler;

impl NOPPacketHandler {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl PacketHandler for NOPPacketHandler {
    async fn handle_packet(&self, _peer: &PeerNode, _packet: &Packet) -> Result<(), CommError> {
        Ok(())
    }
}

/// A packet handler that uses partial functions for different peers
pub struct PartialFunctionPacketHandler<F>
where
    F: Fn(&PeerNode, &Packet) -> Option<Result<(), CommError>> + Send + Sync,
{
    handler_fn: F,
}

impl<F> PartialFunctionPacketHandler<F>
where
    F: Fn(&PeerNode, &Packet) -> Option<Result<(), CommError>> + Send + Sync,
{
    pub fn new(handler_fn: F) -> Self {
        Self { handler_fn }
    }
}

#[async_trait]
impl<F> PacketHandler for PartialFunctionPacketHandler<F>
where
    F: Fn(&PeerNode, &Packet) -> Option<Result<(), CommError>> + Send + Sync,
{
    async fn handle_packet(&self, peer: &PeerNode, packet: &Packet) -> Result<(), CommError> {
        match (self.handler_fn)(peer, packet) {
            Some(result) => result,
            None => {
                let error_msg = format!("Unable to handle packet {:?}", packet);
                tracing::error!("{}", error_msg);
                Err(unknown_protocol(error_msg))
            }
        }
    }
}

/// Helper function to create a packet handler from a closure
pub fn packet_handler_from_fn<F>(handler_fn: F) -> PartialFunctionPacketHandler<F>
where
    F: Fn(&PeerNode, &Packet) -> Option<Result<(), CommError>> + Send + Sync,
{
    PartialFunctionPacketHandler::new(handler_fn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rust::peer_node::{Endpoint, NodeIdentifier};
    use prost::bytes::Bytes;

    fn test_peer() -> PeerNode {
        let id = NodeIdentifier {
            key: Bytes::from("test".as_bytes().to_vec()),
        };
        let endpoint = Endpoint::new("localhost".to_string(), 8080, 8080);
        PeerNode { id, endpoint }
    }

    fn test_packet() -> Packet {
        Packet {
            type_id: "test".to_string(),
            content: Bytes::new(),
        }
    }

    #[tokio::test]
    async fn test_nop_packet_handler() {
        let handler = NOPPacketHandler::new();
        let peer = test_peer();
        let packet = test_packet();

        let result = handler.handle_packet(&peer, &packet).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_partial_function_packet_handler_success() {
        let handler = packet_handler_from_fn(|_peer, packet| {
            if packet.type_id == "test" {
                Some(Ok(()))
            } else {
                None
            }
        });

        let peer = test_peer();
        let packet = test_packet();

        let result = handler.handle_packet(&peer, &packet).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_partial_function_packet_handler_unhandled() {
        let handler = packet_handler_from_fn(|_peer, packet| {
            if packet.type_id == "other" {
                Some(Ok(()))
            } else {
                None
            }
        });

        let peer = test_peer();
        let packet = test_packet();

        let result = handler.handle_packet(&peer, &packet).await;
        assert!(result.is_err());
        if let Err(CommError::UnknownProtocolError(msg)) = result {
            assert!(msg.contains("Unable to handle packet"));
        } else {
            panic!("Expected UnknownProtocolError error");
        }
    }
}
