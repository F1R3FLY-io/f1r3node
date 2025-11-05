// See comm/src/main/scala/coop/rchain/comm/rp/HandleMessages.scala

use std::sync::Arc;

use models::routing::{Packet, Protocol};

use crate::rust::{
    errors::CommError,
    p2p::packet_handler::PacketHandler,
    peer_node::PeerNode,
    rp::{connect::ConnectionsCell, protocol_helper, rp_conf::RPConf},
    transport::{communication_response::CommunicationResponse, transport_layer::TransportLayer},
};

pub async fn handle(
    protocol: &Protocol,
    transport_layer: Arc<dyn TransportLayer + Send + Sync + 'static>,
    packet_handler: Arc<dyn PacketHandler + Send + Sync + 'static>,
    connections_cell: &ConnectionsCell,
    rp_conf: &RPConf,
) -> Result<CommunicationResponse, CommError> {
    let sender = protocol_helper::sender(protocol);

    match &protocol.message {
        Some(models::routing::protocol::Message::Heartbeat(_)) => {
            handle_heartbeat(&sender, connections_cell)
        }

        Some(models::routing::protocol::Message::ProtocolHandshake(_)) => {
            handle_protocol_handshake(&sender, transport_layer, connections_cell, rp_conf).await
        }

        Some(models::routing::protocol::Message::ProtocolHandshakeResponse(_)) => {
            handle_protocol_handshake_response(&sender, connections_cell)
        }

        Some(models::routing::protocol::Message::Disconnect(_)) => {
            handle_disconnect(&sender, connections_cell)
        }

        Some(models::routing::protocol::Message::Packet(packet)) => {
            handle_packet(&sender, packet, packet_handler).await
        }

        None => {
            let msg_str = format!("{:?}", protocol.message);
            log::error!("Unexpected message type {}", msg_str);

            Ok(CommunicationResponse::not_handled(
                CommError::UnexpectedMessage(msg_str),
            ))
        }
    }
}

pub fn handle_disconnect(
    sender: &PeerNode,
    connections_cell: &ConnectionsCell,
) -> Result<CommunicationResponse, CommError> {
    log::info!("Forgetting about {}", sender);
    connections_cell
        .flat_modify(|connections| connections.remove_conn_and_report(sender.clone()))?;
    Ok(CommunicationResponse::handled_without_message())
}

pub async fn handle_packet(
    remote: &PeerNode,
    packet: &Packet,
    packet_handler: Arc<dyn PacketHandler + Send + Sync + 'static>,
) -> Result<CommunicationResponse, CommError> {
    log::debug!("Received packet from {}", remote);
    packet_handler.handle_packet(remote, packet).await?;
    Ok(CommunicationResponse::handled_without_message())
}

pub fn handle_protocol_handshake_response(
    peer: &PeerNode,
    connections_cell: &ConnectionsCell,
) -> Result<CommunicationResponse, CommError> {
    log::debug!("Received protocol handshake response from {}", peer);
    connections_cell.flat_modify(|connections| connections.add_conn_and_report(peer.clone()))?;
    Ok(CommunicationResponse::handled_without_message())
}

pub async fn handle_protocol_handshake(
    peer: &PeerNode,
    transport_layer: Arc<dyn TransportLayer + Send + Sync + 'static>,
    connections_cell: &ConnectionsCell,
    rp_conf: &RPConf,
) -> Result<CommunicationResponse, CommError> {
    let response =
        protocol_helper::protocol_handshake_response(&rp_conf.local, &rp_conf.network_id);

    match transport_layer.send(peer, &response).await {
        Ok(_) => {
            log::info!("Responded to protocol handshake request from {}", peer);
            let _ = connections_cell
                .flat_modify(|connections| connections.add_conn_and_report(peer.clone()));
        }
        Err(_) => {}
    }

    Ok(CommunicationResponse::handled_without_message())
}

pub fn handle_heartbeat(
    peer: &PeerNode,
    connections_cell: &ConnectionsCell,
) -> Result<CommunicationResponse, CommError> {
    let _ = connections_cell.flat_modify(|connections| connections.refresh_conn(peer.clone()));
    Ok(CommunicationResponse::handled_without_message())
}
