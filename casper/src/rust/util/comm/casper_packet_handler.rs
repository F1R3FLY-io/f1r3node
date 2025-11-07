// See casper/src/main/scala/coop/rchain/casper/util/comm/CasperPacketHandler.scala

use async_trait::async_trait;
use comm::rust::{errors::CommError, p2p::packet_handler::PacketHandler, peer_node::PeerNode};
use models::rust::casper::pretty_printer::PrettyPrinter;
use models::{routing::Packet, rust::casper::protocol::casper_message::CasperMessage};
use prost::bytes::Bytes;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::rust::{
    engine::engine_cell::EngineCell,
    errors::CasperError,
    protocol::{casper_message_from_proto, to_casper_message_proto},
    util::comm::fair_round_robin_dispatcher::{
        Dispatch, DispatcherConfig, FairRoundRobinDispatcher,
    },
};

use tokio::sync::Semaphore;

#[derive(Clone)]
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

        let engine = self.engine_cell.get().await;

        engine
            .handle(peer.clone(), message)
            .await
            .map_err(|e| CommError::CasperError(e.to_string()))?;

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct BlockCreator {
    value: Bytes,
}

impl BlockCreator {
    pub fn new(value: Bytes) -> Self {
        Self { value }
    }

    /// Create an empty BlockCreator (used for non-BlockHashMessage types).
    pub fn empty() -> Self {
        Self {
            value: Bytes::new(),
        }
    }

    /// Extract the BlockCreator from a CasperMessage.
    /// Returns the block creator for BlockHashMessage, empty for all other message types.
    pub fn from_message(message: &CasperMessage) -> Self {
        match message {
            CasperMessage::BlockHashMessage(bhm) => Self::new(bhm.block_creator.clone()),
            _ => Self::empty(),
        }
    }

    /// Access the underlying value (primarily for tests and debugging).
    pub fn value(&self) -> &Bytes {
        &self.value
    }
}

impl Hash for BlockCreator {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl PartialEq for BlockCreator {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for BlockCreator {}

impl Display for BlockCreator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", PrettyPrinter::build_string_no_limit(&self.value))
    }
}

/// Wrapper for (PeerNode, CasperMessage) tuple used by the dispatcher.
///
/// This struct enables custom equality and display implementations for messages
/// flowing through the fair round-robin dispatcher.
#[derive(Clone, Debug)]
pub struct DispatcherMessage {
    pub peer: PeerNode,
    pub message: CasperMessage,
}

impl DispatcherMessage {
    pub fn new(peer: PeerNode, message: CasperMessage) -> Self {
        Self { peer, message }
    }
}

/// Implement equality based on message content only (ignoring peer).
///
/// For BlockHashMessage: compare only block hashes
/// For other messages: use default equality
impl PartialEq for DispatcherMessage {
    fn eq(&self, other: &Self) -> bool {
        match (&self.message, &other.message) {
            (CasperMessage::BlockHashMessage(bhm1), CasperMessage::BlockHashMessage(bhm2)) => {
                bhm1.block_hash == bhm2.block_hash
            }
            _ => self.message == other.message,
        }
    }
}

impl Eq for DispatcherMessage {}

impl Display for DispatcherMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.message {
            CasperMessage::BlockHashMessage(bhm) => {
                write!(
                    f,
                    "[{}]",
                    PrettyPrinter::build_string_no_limit(&bhm.block_hash)
                )
            }
            _ => {
                write!(
                    f,
                    "[Unexpected message {:?} from {}!!!]",
                    self.message, self.peer
                )
            }
        }
    }
}

/// Filter function to determine how to dispatch a message.
///
/// For BlockHashMessage: checks if block already exists in casper
/// - If exists → Drop (duplicate)
/// - If not exists → Handle (new block)
///
/// For other messages → Pass (bypass queue, handle immediately)
async fn check_message(
    engine_cell: &EngineCell,
    message: &DispatcherMessage,
) -> Result<Dispatch, CasperError> {
    match &message.message {
        CasperMessage::BlockHashMessage(bhm) => {
            let engine = engine_cell.get().await;
            match engine.with_casper() {
                Some(casper) => {
                    if casper.contains(&bhm.block_hash) {
                        Ok(Dispatch::Drop)
                    } else {
                        Ok(Dispatch::Handle)
                    }
                }
                None => Ok(Dispatch::Handle), // If no casper, treat as new block
            }
        }
        _ => Ok(Dispatch::Pass),
    }
}

/// Handle function to process messages after dispatch.
///
/// Reads the engine from EngineCell and delegates message handling to it.
/// The block_creator parameter is only used for source identification by the dispatcher.
async fn handle_message(
    engine_cell: &EngineCell,
    _block_creator: BlockCreator,
    message: DispatcherMessage,
) -> Result<(), CasperError> {
    tracing::debug!(target: "rchain.casper", "Casper message received");
    let engine = engine_cell.get().await;
    let result = engine.handle(message.peer, message.message).await;
    tracing::debug!(target: "rchain.casper", "Casper message handle done");
    result
}

/// Packet handler that uses fair round-robin dispatcher for message processing.
///
/// This handler ensures equitable processing among multiple block creators,
/// with duplicate detection and queue management.
pub struct FairDispatcherPacketHandler {
    dispatcher: Arc<FairRoundRobinDispatcher<BlockCreator, DispatcherMessage>>,
}

impl FairDispatcherPacketHandler {
    /// Create a new FairDispatcherPacketHandler with the given dispatcher.
    pub fn new(dispatcher: Arc<FairRoundRobinDispatcher<BlockCreator, DispatcherMessage>>) -> Self {
        Self { dispatcher }
    }
}

#[async_trait]
impl PacketHandler for FairDispatcherPacketHandler {
    async fn handle_packet(&self, peer: &PeerNode, packet: &Packet) -> Result<(), CommError> {
        let parse_result = to_casper_message_proto(packet).get();

        if parse_result.is_err() {
            log::warn!(
                "Could not extract casper message from packet sent by {}: {}",
                peer,
                parse_result.clone().err().unwrap()
            );
            return Ok(());
        }

        let message = casper_message_from_proto(parse_result.unwrap())
            .map_err(|e| CommError::UnexpectedMessage(e))?;

        log::debug!("Received message {:?} from {}", message, peer);

        let block_creator = BlockCreator::from_message(&message);
        let dispatcher_message = DispatcherMessage::new(peer.clone(), message);

        self.dispatcher
            .dispatch(block_creator, dispatcher_message)
            .await
            .map_err(|e| CommError::CasperError(e.to_string()))?;

        Ok(())
    }
}

/// Create a fair dispatcher packet handler.
///
/// This factory function constructs a fully configured FairDispatcherPacketHandler
/// with the given parameters.
///
/// # Parameters
///
/// * `engine_cell` - The engine cell for accessing casper and handling messages
/// * `max_peer_queue_size` - Maximum messages per peer queue
/// * `give_up_after_skipped` - Give up on a peer after this many skips
/// * `drop_peer_after_retries` - Drop a peer after this many retries
///
/// # Returns
///
/// A configured FairDispatcherPacketHandler ready to use as a PacketHandler
pub async fn fair_dispatcher(
    engine_cell: EngineCell,
    max_peer_queue_size: usize,
    give_up_after_skipped: usize,
    drop_peer_after_retries: usize,
) -> Result<FairDispatcherPacketHandler, CasperError> {
    // Create semaphore lock for dispatcher
    let lock = Arc::new(Semaphore::new(1));

    // Create filter closure that captures engine_cell
    let engine_cell_for_filter = engine_cell.clone();
    let filter = move |message: &DispatcherMessage| {
        let engine_cell = engine_cell_for_filter.clone();
        let message = message.clone();
        Box::pin(async move {
            match check_message(&engine_cell, &message).await {
                Ok(dispatch) => dispatch,
                Err(e) => {
                    log::warn!("Error checking message, defaulting to Handle: {}", e);
                    Dispatch::Handle
                }
            }
        }) as std::pin::Pin<Box<dyn std::future::Future<Output = Dispatch> + Send>>
    };

    // Create handle closure that captures engine_cell
    let engine_cell_for_handle = engine_cell.clone();
    let handle = move |block_creator: BlockCreator, message: DispatcherMessage| {
        let engine_cell = engine_cell_for_handle.clone();
        Box::pin(async move {
            if let Err(e) = handle_message(&engine_cell, block_creator, message).await {
                log::error!("Error handling message: {}", e);
            }
        }) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
    };

    // Create dispatcher configuration
    let config = DispatcherConfig::new(
        max_peer_queue_size,
        give_up_after_skipped,
        drop_peer_after_retries,
    );

    // Create the fair round-robin dispatcher
    let dispatcher = FairRoundRobinDispatcher::new(filter, handle, config, lock);

    // Wrap in Arc and create packet handler
    Ok(FairDispatcherPacketHandler::new(Arc::new(dispatcher)))
}
