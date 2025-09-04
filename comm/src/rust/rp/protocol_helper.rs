// See comm/src/main/scala/coop/rchain/comm/rp/ProtocolHelper.scala

use models::{
    routing::{
        Disconnect, Header, Heartbeat, Node, Packet, Protocol, ProtocolHandshake,
        ProtocolHandshakeResponse,
    },
    rust::casper::protocol::packet_type_tag::ToPacket,
};
use prost::bytes::Bytes;

use crate::rust::{
    errors::{unknown_protocol, CommError},
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
    transport::transport_layer::Blob,
};

pub fn to_protocol_bytes(x: &str) -> Vec<u8> {
    x.as_bytes().to_vec()
}

pub fn header(src: &PeerNode, network_id: &str) -> Header {
    Header {
        sender: Some(node(src)),
        network_id: network_id.to_string(),
    }
}

pub fn node(n: &PeerNode) -> Node {
    Node {
        id: n.id.key.clone(),
        host: n.endpoint.host.clone().into(),
        tcp_port: n.endpoint.tcp_port,
        udp_port: n.endpoint.udp_port,
    }
}

pub fn sender(proto: &Protocol) -> PeerNode {
    to_peer_node(proto.header.as_ref().unwrap().sender.as_ref().unwrap())
}

pub fn to_peer_node(n: &Node) -> PeerNode {
    PeerNode {
        id: NodeIdentifier { key: n.id.clone() },
        endpoint: Endpoint::new(format!("{:?}", n.host), n.tcp_port, n.udp_port),
    }
}

pub fn protocol(src: &PeerNode, network_id: &str) -> Protocol {
    Protocol {
        header: Some(header(src, network_id)),
        ..Default::default()
    }
}

pub fn protocol_handshake(src: &PeerNode, network_id: &str) -> Protocol {
    Protocol {
        header: Some(header(src, network_id)),
        message: Some(models::routing::protocol::Message::ProtocolHandshake(
            ProtocolHandshake {
                nonce: Bytes::new(),
            },
        )),
    }
}

pub fn protocol_handshake_response(src: &PeerNode, network_id: &str) -> Protocol {
    Protocol {
        header: Some(header(src, network_id)),
        message: Some(
            models::routing::protocol::Message::ProtocolHandshakeResponse(
                ProtocolHandshakeResponse {
                    nonce: Bytes::new(),
                },
            ),
        ),
    }
}

pub fn heartbeat(src: &PeerNode, network_id: &str) -> Protocol {
    Protocol {
        header: Some(header(src, network_id)),
        message: Some(models::routing::protocol::Message::Heartbeat(Heartbeat {})),
    }
}

pub fn packet(src: &PeerNode, network_id: &str, packet: Packet) -> Protocol {
    Protocol {
        header: Some(header(src, network_id)),
        message: Some(models::routing::protocol::Message::Packet(packet)),
    }
}

pub fn packet_with_content<A>(src: &PeerNode, network_id: &str, content: A) -> Protocol
where
    A: ToPacket,
{
    packet(src, network_id, content.mk_packet())
}

pub fn to_packet(proto: &Protocol) -> Result<Packet, CommError> {
    match &proto.message {
        Some(models::routing::protocol::Message::Packet(packet)) => Ok(packet.clone()),
        _ => Err(unknown_protocol(format!(
            "Was expecting Packet, got {:?}",
            proto.message
        ))),
    }
}

pub fn disconnect(src: &PeerNode, network_id: &str) -> Protocol {
    Protocol {
        header: Some(header(src, network_id)),
        message: Some(models::routing::protocol::Message::Disconnect(
            Disconnect {},
        )),
    }
}

pub fn blob(sender: &PeerNode, type_id: &str, content: &[u8]) -> Blob {
    Blob {
        sender: sender.clone(),
        packet: Packet {
            type_id: type_id.to_string(),
            content: Bytes::copy_from_slice(content),
        },
    }
}
