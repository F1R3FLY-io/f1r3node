// See casper/src/main/scala/coop/rchain/casper/util/comm/CasperPacketHandler.scala

use async_trait::async_trait;
use comm::rust::{errors::CommError, p2p::packet_handler::PacketHandler, peer_node::PeerNode};
use models::routing::Packet;

use crate::rust::{
    engine::engine_cell::EngineCell,
    protocol::{casper_message_from_proto, to_casper_message_proto},
};

pub struct CasperPacketHandler {
    engine_cell: EngineCell,
}

impl CasperPacketHandler {
    pub fn new(engine_cell: EngineCell) -> Self {
        Self { engine_cell }
    }
}

#[async_trait]
impl PacketHandler for CasperPacketHandler {
    async fn handle_packet(&self, peer: &PeerNode, packet: &Packet) -> Result<(), CommError> {
        let parse_result = to_casper_message_proto(packet).get();

        if parse_result.is_err() {
            log::warn!(
                "Could not extract casper message from packet sent by {}: {}",
                peer,
                parse_result.clone().err().unwrap()
            );
        }

        let message = casper_message_from_proto(parse_result.unwrap())
            .map_err(|e| CommError::UnexpectedMessage(e))?;

        // let engine = self
        //     .engine_cell
        //     .read()
        //     .await
        //     .map_err(|e| CommError::CasperError(e.to_string()))?;

        // engine
        //     .handle(peer.clone(), message)
        //     .await
        //     .map_err(|e| CommError::CasperError(e.to_string()))?;

        // Ok(())

        todo!()
    }
}
