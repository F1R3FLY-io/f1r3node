// See node/src/main/scala/coop/rchain/node/runtime/Setup.scala

use models::rust::{block_hash::BlockHash, casper::protocol::casper_message::BlockMessage};
use std::{collections::HashSet, sync::Arc};
use tokio::sync::{mpsc, oneshot, RwLock};

use casper::rust::{
    blocks::{
        block_processor::BlockProcessor,
        proposer::proposer::{ProductionProposer, ProposerResult},
    },
    casper::Casper,
    engine::{block_retriever::BlockRetriever, casper_launch::CasperLaunch},
    state::instances::ProposerState,
    ProposeFunction,
};
use comm::rust::{
    discovery::node_discovery::NodeDiscovery,
    p2p::packet_handler::PacketHandler,
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    transport::transport_layer::TransportLayer,
};
use shared::rust::shared::f1r3fly_events::EventPublisher;

use crate::rust::{
    api::{admin_web_api::AdminWebApi, web_api::WebApi},
    configuration::NodeConf,
    runtime::{
        api_servers::APIServers,
        node_runtime::{CasperLoop, EngineInit},
    },
    web::reporting_routes::ReportingHttpRoutes,
};

pub fn setup_node_program<T: TransportLayer + Send + Sync>(
    rp_connections: ConnectionsCell,
    rp_conf: RPConf,
    transport_layer: Arc<T>,
    block_retriever: BlockRetriever<T>,
    conf: NodeConf,
    event_publisher: Arc<dyn EventPublisher>,
    node_discovery: Arc<dyn NodeDiscovery>,
) -> (
    Arc<dyn PacketHandler>,
    APIServers,
    CasperLoop,
    CasperLoop,
    EngineInit,
    Arc<dyn CasperLaunch>,
    ReportingHttpRoutes,
    Arc<dyn WebApi>,
    Arc<dyn AdminWebApi>,
    Option<ProductionProposer<T>>,
    mpsc::UnboundedReceiver<(Arc<dyn Casper>, bool, oneshot::Sender<ProposerResult>)>,
    // TODO move towards having a single node state - OLD
    Option<RwLock<ProposerState>>,
    BlockProcessor<T>,
    RwLock<HashSet<BlockHash>>,
    mpsc::UnboundedSender<(Arc<dyn Casper>, BlockMessage)>,
    Option<Arc<ProposeFunction>>,
) {
    // let (proposer_queue_tx, proposer_queue_rx) =
    //   mpsc::unbounded_channel::<(Arc<dyn Casper>, bool, oneshot::Sender<ProposerResult>)>();

    // let (block_processor_queue_tx, block_processor_queue_rx) =
    // mpsc::unbounded_channel::<(Arc<dyn Casper>, BlockMessage)>();

    todo!()
}
