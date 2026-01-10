// See comm/src/main/scala/coop/rchain/comm/transport/TransportLayer.scala
// See comm/src/main/scala/coop/rchain/comm/transport/TransportLayerSyntax.scala
// See casper/src/main/scala/coop/rchain/casper/util/comm/CommUtil.scala

use async_trait::async_trait;
use tracing::{info, warn};
use prost::bytes::Bytes;
use std::{sync::Arc, time::Duration};

use models::{
    casper::{
        ApprovedBlockRequestProto, BlockHashMessageProto, BlockRequestProto,
        ForkChoiceTipRequestProto, HasBlockRequestProto,
    },
    routing::{Packet, Protocol},
    rust::{
        block_hash::BlockHash,
        casper::{pretty_printer::PrettyPrinter, protocol::packet_type_tag::ToPacket},
    },
};

use crate::rust::{
    errors::CommError,
    peer_node::PeerNode,
    rp::{connect::ConnectionsCell, protocol_helper, rp_conf::RPConf},
};

#[derive(Clone)]
pub struct Blob {
    pub sender: PeerNode,
    pub packet: Packet,
}

#[async_trait]
pub trait TransportLayer {
    async fn send(&self, peer: &PeerNode, msg: &Protocol) -> Result<(), CommError>;

    async fn broadcast(&self, peers: &[PeerNode], msg: &Protocol) -> Result<(), CommError>;

    async fn stream(&self, peer: &PeerNode, blob: &Blob) -> Result<(), CommError>;

    async fn stream_mult(&self, peers: &[PeerNode], blob: &Blob) -> Result<(), CommError>;

    // See comm/src/main/scala/coop/rchain/comm/transport/TransportLayerSyntax.scala

    async fn send_packet_to_peer(
        &self,
        conf: &RPConf,
        peer: &PeerNode,
        msg: Packet,
    ) -> Result<(), CommError> {
        let protocol_msg = protocol_helper::packet(&conf.local, &conf.network_id, msg);
        self.send(peer, &protocol_msg).await
    }

    async fn send_message_to_peer(
        &self,
        conf: &RPConf,
        peer: &PeerNode,
        msg: Arc<dyn ToPacket + Send + Sync>,
    ) -> Result<(), CommError> {
        let packet = msg.mk_packet();
        self.send_packet_to_peer(conf, peer, packet).await
    }

    async fn stream_packet_to_peer(
        &self,
        conf: &RPConf,
        peer: &PeerNode,
        packet: Packet,
    ) -> Result<(), CommError> {
        let blob = Blob {
            sender: conf.local.clone(),
            packet,
        };
        self.stream(peer, &blob).await
    }

    async fn stream_message_to_peer(
        &self,
        conf: &RPConf,
        peer: &PeerNode,
        msg: Arc<dyn ToPacket + Send + Sync>,
    ) -> Result<(), CommError> {
        let packet = msg.mk_packet();
        self.stream_packet_to_peer(conf, peer, packet).await
    }

    async fn send_to_bootstrap(
        &self,
        conf: &RPConf,
        msg: Arc<dyn ToPacket + Send + Sync>,
    ) -> Result<(), CommError> {
        let bootstrap = conf
            .bootstrap
            .as_ref()
            .ok_or_else(|| CommError::ConfigError("No bootstrap peer configured".to_string()))?;
        self.send_message_to_peer(conf, bootstrap, msg).await
    }

    // See casper/src/main/scala/coop/rchain/casper/util/comm/CommUtil.scala

    async fn send_packet_to_peers(
        &self,
        connections_cell: &ConnectionsCell,
        conf: &RPConf,
        message: Packet,
        scope_size: Option<usize>,
    ) -> Result<(), CommError> {
        let max = scope_size.unwrap_or(conf.max_num_of_connections);
        let peers = connections_cell.random(max)?;
        let protocol_msg = protocol_helper::packet(&conf.local, &conf.network_id, message);
        self.broadcast(&peers.0, &protocol_msg).await
    }

    async fn stream_packet_to_peers(
        &self,
        connections_cell: &ConnectionsCell,
        conf: &RPConf,
        message: Packet,
        scope_size: Option<usize>,
    ) -> Result<(), CommError> {
        let max = scope_size.unwrap_or(conf.max_num_of_connections);
        let peers = connections_cell.random(max)?;
        let blob = Blob {
            sender: conf.local.clone(),
            packet: message,
        };
        self.stream_mult(&peers.0, &blob).await
    }

    async fn send_with_retry(
        &self,
        conf: &RPConf,
        message: Packet,
        peer: &PeerNode,
        retry_after: Duration,
        msg_type_name: &str,
    ) -> Result<(), CommError> {
        let protocol_msg = protocol_helper::packet(&conf.local, &conf.network_id, message);

        info!("Starting to request {}", msg_type_name);

        loop {
            match self.send(peer, &protocol_msg).await {
                Ok(()) => {
                    info!("Successfully sent {} to {}", msg_type_name, peer);
                    return Ok(());
                }
                Err(error) => {
                    warn!(
                        "Failed to send {} to {} because of {}. Retrying in {:?}...",
                        msg_type_name, peer, error, retry_after
                    );
                    tokio::time::sleep(retry_after).await;
                }
            }
        }
    }

    async fn request_for_block(
        &self,
        conf: &RPConf,
        peer: &PeerNode,
        hash: BlockHash,
    ) -> Result<(), CommError> {
        tracing::debug!(
            "Requesting {} from {}",
            PrettyPrinter::build_string_no_limit(&hash),
            peer.endpoint.host,
        );

        self.send_packet_to_peer(conf, peer, BlockRequestProto { hash }.mk_packet())
            .await
    }

    // CommUtilOps

    async fn send_message_to_peers(
        &self,
        connections_cell: &ConnectionsCell,
        conf: &RPConf,
        message: Arc<dyn ToPacket + Send + Sync>,
        scope_size: Option<usize>,
    ) -> Result<(), CommError> {
        self.send_packet_to_peers(connections_cell, conf, message.mk_packet(), scope_size)
            .await
    }

    async fn stream_message_to_peers(
        &self,
        connections_cell: &ConnectionsCell,
        conf: &RPConf,
        message: Arc<dyn ToPacket + Send + Sync>,
        scope_size: Option<usize>,
    ) -> Result<(), CommError> {
        self.stream_packet_to_peers(connections_cell, conf, message.mk_packet(), scope_size)
            .await
    }

    async fn send_block_hash(
        &self,
        connections_cell: &ConnectionsCell,
        conf: &RPConf,
        hash: &BlockHash,
        block_creator: &Bytes,
    ) -> Result<(), CommError> {
        let result = self
            .send_message_to_peers(
                connections_cell,
                conf,
                Arc::new(BlockHashMessageProto {
                    hash: hash.clone(),
                    block_creator: block_creator.clone(),
                }),
                None,
            )
            .await;

        tracing::info!(
            "Sent hash {} to peers",
            PrettyPrinter::build_string_no_limit(hash)
        );

        result
    }

    async fn broadcast_has_block_request(
        &self,
        connections_cell: &ConnectionsCell,
        conf: &RPConf,
        hash: &BlockHash,
    ) -> Result<(), CommError> {
        self.send_message_to_peers(
            connections_cell,
            conf,
            Arc::new(HasBlockRequestProto { hash: hash.clone() }),
            None,
        )
        .await
    }

    async fn broadcast_request_for_block(
        &self,
        connections_cell: &ConnectionsCell,
        conf: &RPConf,
        hash: &BlockHash,
    ) -> Result<(), CommError> {
        self.send_message_to_peers(
            connections_cell,
            conf,
            Arc::new(BlockRequestProto { hash: hash.clone() }),
            None,
        )
        .await
    }

    async fn send_fork_choice_tip_request(
        &self,
        connections_cell: &ConnectionsCell,
        conf: &RPConf,
    ) -> Result<(), CommError> {
        let result = self
            .send_message_to_peers(
                connections_cell,
                conf,
                Arc::new(ForkChoiceTipRequestProto {}),
                None,
            )
            .await;

        tracing::info!("Requested fork tip from peers");

        result
    }

    async fn request_approved_block(
        &self,
        conf: &RPConf,
        trim_state: Option<bool>,
    ) -> Result<(), CommError> {
        self.send_with_retry(
            conf,
            ApprovedBlockRequestProto {
                identifier: "".to_string(),
                trim_state: trim_state.unwrap_or(true),
            }
            .mk_packet(),
            conf.bootstrap.as_ref().ok_or_else(|| {
                CommError::ConfigError(
                    "StandaloneNodeSendToBootstrapError: No bootstrap peer configured".to_string(),
                )
            })?,
            Duration::from_secs(10),
            "ApprovedBlockRequest",
        )
        .await
    }
}
