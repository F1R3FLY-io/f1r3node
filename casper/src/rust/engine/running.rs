// See casper/src/main/scala/coop/rchain/casper/engine/Running.scala

use tokio::sync::mpsc;

use crate::rust::{
    casper::MultiParentCasper,
    engine::{
        block_retriever::{self, BlockRetriever},
        engine::{self, Engine},
        engine_cell::EngineCell,
    },
    errors::CasperError,
};
use async_trait::async_trait;
use comm::rust::{
    peer_node::PeerNode,
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    transport::transport_layer::TransportLayer,
};
use models::rust::{
    block_hash::BlockHash,
    casper::{
        pretty_printer::PrettyPrinter,
        protocol::casper_message::{
            self, ApprovedBlock, ApprovedBlockCandidate, BlockHashMessage, BlockMessage,
            BlockRequest, CasperMessage, HasBlock, HasBlockRequest,
        },
    },
};

use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    state::{
        exporters::rspace_exporter_items::RSpaceExporterItems,
        rspace_exporter::RSpaceExporterInstance,
    },
};
use std::future::Future;
use std::pin::Pin;
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CasperMessageStatus {
    BlockIsInDag,
    BlockIsInCasperBuffer,
    BlockIsReceived,
    BlockIsWaitingForCasper,
    BlockIsInProcessing,
    DoNotIgnore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgnoreCasperMessageStatus {
    pub do_ignore: bool,
    pub status: CasperMessageStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LastFinalizedBlockNotFoundError;

impl std::fmt::Display for LastFinalizedBlockNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Last finalized block not found in the block storage.")
    }
}

impl std::error::Error for LastFinalizedBlockNotFoundError {}

/**
 * As we introduced synchrony constraint - there might be situation when node is stuck.
 * As an edge case with `sync = 0.99`, if node misses the block that is the last one to meet sync constraint,
 * it has no way to request it after it was broadcasted. So it will never meet synchrony constraint.
 * To mitigate this issue we can update fork choice tips if current fork-choice tip has old timestamp,
 * which means node does not propose new blocks and no new blocks were received recently.
 */
pub async fn update_fork_choice_tips_if_stuck<T: TransportLayer + Send + Sync>(
    engine_cell: &EngineCell,
    transport: &Arc<T>,
    connections_cell: &ConnectionsCell,
    conf: &RPConf,
    delay_threshold: Duration,
) -> Result<(), CasperError> {
    // Get engine from engine cell
    let engine = engine_cell.get().await;

    // Check if we have casper
    if let Some(casper) = engine.with_casper() {
        // Get latest messages from block dag
        let latest_messages = casper.block_dag().await?.latest_message_hashes();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        // Check if any latest message is recent
        let mut has_recent_latest_message = false;
        for entry in latest_messages.iter() {
            let block_hash = entry.value();
            if let Ok(Some(block)) = casper.block_store().get(block_hash) {
                let block_timestamp = block.header.timestamp;
                if (now - block_timestamp) < delay_threshold.as_millis() as i64 {
                    has_recent_latest_message = true;
                    break;
                }
            }
        }

        // If stuck, request fork choice tips
        let stuck = !has_recent_latest_message;
        if stuck {
            log::info!(
                "Requesting tips update as newest latest message is more than {:?} old. Might be network is faulty.",
                delay_threshold
            );
            transport
                .send_fork_choice_tip_request(connections_cell, conf)
                .await?;
        }
    }

    Ok(())
}

#[async_trait]
impl<T: TransportLayer + Send + Sync + 'static> Engine for Running<T> {
    async fn init(&self) -> Result<(), CasperError> {
        {
            let mut init_called = self.init_called.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire init lock".to_string())
            })?;

            if *init_called {
                return Err(CasperError::RuntimeError(
                    "Init function already called".to_string(),
                ));
            }

            *init_called = true;
        }

        // Call the async init function and await it
        (self.the_init)().await?;
        Ok(())
    }

    async fn handle(&self, peer: PeerNode, msg: CasperMessage) -> Result<(), CasperError> {
        match msg {
            CasperMessage::BlockHashMessage(h) => {
                self.handle_block_hash_message(peer, h, |hash| self.ignore_casper_message(hash))
                    .await
            }
            CasperMessage::BlockMessage(b) => {
                if let Some(id) = self.casper.get_validator() {
                    if b.sender == id.public_key.bytes {
                        log::warn!(
                            "There is another node {} proposing using the same private key as you. Or did you restart your node?",
                            peer
                        );
                    }
                }
                if self.ignore_casper_message(b.block_hash.clone())? {
                    log::debug!(
                        "Ignoring BlockMessage {} from {}",
                        PrettyPrinter::build_string_block_message(&b, true),
                        peer.endpoint.host
                    );
                } else {
                    log::debug!(
                        "Incoming BlockMessage {} from {}",
                        PrettyPrinter::build_string_block_message(&b, true),
                        peer.endpoint.host
                    );
                    self.block_processing_queue_tx
                        .send((self.casper.clone(), b))
                        .map_err(|e| {
                            CasperError::RuntimeError(format!(
                                "Failed to send block to queue: {}",
                                e
                            ))
                        })?;
                }
                Ok(())
            }
            CasperMessage::BlockRequest(br) => self.handle_block_request(peer, br).await,

            // TODO should node say it has block only after it is in DAG, or CasperBuffer is enough? Or even just BlockStore?
            // https://github.com/rchain/rchain/pull/2943#discussion_r449887701 -- OLD
            CasperMessage::HasBlockRequest(hbr) => {
                self.handle_has_block_request(peer, hbr, |hash| self.casper.dag_contains(&hash))
                    .await
            }
            CasperMessage::HasBlock(hb) => {
                self.handle_has_block_message(peer, hb, |hash| self.ignore_casper_message(hash))
                    .await
            }
            CasperMessage::ForkChoiceTipRequest(_) => {
                self.handle_fork_choice_tip_request(peer).await
            }
            CasperMessage::ApprovedBlockRequest(abr) => {
                let last_finalized_block_hash =
                    self.casper.block_dag().await?.last_finalized_block();

                // Create approved block from last finalized block
                let last_finalized_block = self
                    .casper
                    .block_store()
                    .get(&last_finalized_block_hash)?
                    .ok_or_else(|| {
                        CasperError::RuntimeError(LastFinalizedBlockNotFoundError.to_string())
                    })?;

                // Each approved block should be justified by validators signatures
                // ATM we have signatures only for genesis approved block - we also have to have a procedure
                // for gathering signatures for each approved block post genesis.
                // Now new node have to trust bootstrap if it wants to trim state when connecting to the network.
                // TODO We need signatures of Validators supporting this block -- OLD
                let last_approved_block = ApprovedBlock {
                    candidate: ApprovedBlockCandidate {
                        block: last_finalized_block,
                        required_sigs: 0,
                    },
                    sigs: vec![],
                };

                let approved_block = if abr.trim_state {
                    // If Last Finalized State is requested return Last Finalized block as Approved block
                    last_approved_block
                } else {
                    // Respond with approved block that this node is started from.
                    // The very first one is genesis, but this node still might start from later block,
                    // so it will not necessary be genesis.
                    self.approved_block.clone()
                };

                self.handle_approved_block_request(peer, approved_block)
                    .await
            }
            CasperMessage::NoApprovedBlockAvailable(na) => {
                engine::log_no_approved_block_available(&na.node_identifier);
                Ok(())
            }
            CasperMessage::StoreItemsMessageRequest(req) => {
                let start = req
                    .start_path
                    .iter()
                    .map(RSpaceExporterInstance::path_pretty)
                    .collect::<Vec<_>>()
                    .join(" ");

                log::info!(
                    "Received request for store items, startPath: [{}], chunk: {}, skip: {}, from: {}",
                    start,
                    req.take,
                    req.skip,
                    peer
                );

                if !self.disable_state_exporter {
                    self.handle_state_items_message_request(
                        peer,
                        req.start_path,
                        req.skip as u32,
                        req.take as u32,
                    )
                    .await
                } else {
                    log::info!(
                        "Received StoreItemsMessage request but the node is configured to not respond to StoreItemsMessage, from {}.",
                        peer
                    );
                    Ok(())
                }
            }
            _ => Ok(()),
        }
    }

    /// Running always contains casper; enables `EngineDynExt::with_casper(...)`
    /// to mirror Scala `Engine.withCasper` behavior.
    fn with_casper(&self) -> Option<Arc<dyn MultiParentCasper + Send + Sync>> {
        Some(Arc::clone(&self.casper) as Arc<dyn MultiParentCasper + Send + Sync>)
    }
}

// NOTE: Changed to use Arc<dyn MultiParentCasper> directly instead of generic M
// based on discussion with Steven for TestFixture compatibility - avoids ?Sized issues
pub struct Running<T: TransportLayer + Send + Sync> {
    block_processing_queue_tx:
        mpsc::UnboundedSender<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>,
    blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
    casper: Arc<dyn MultiParentCasper + Send + Sync>,
    approved_block: ApprovedBlock,
    // Scala: theInit: F[Unit] - lazy async computation
    the_init: Arc<
        dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync,
    >,
    init_called: Arc<Mutex<bool>>,
    disable_state_exporter: bool,
    transport: Arc<T>,
    conf: RPConf,
    block_retriever: BlockRetriever<T>,
}

impl<T: TransportLayer + Send + Sync> Running<T> {
    pub fn new(
        block_processing_queue_tx: mpsc::UnboundedSender<(
            Arc<dyn MultiParentCasper + Send + Sync>,
            BlockMessage,
        )>,
        blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
        casper: Arc<dyn MultiParentCasper + Send + Sync>,
        approved_block: ApprovedBlock,
        the_init: Arc<
            dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync,
        >,
        disable_state_exporter: bool,
        transport: Arc<T>,
        conf: RPConf,
        block_retriever: BlockRetriever<T>,
    ) -> Self {
        Running {
            block_processing_queue_tx,
            blocks_in_processing,
            casper,
            approved_block,
            the_init,
            init_called: Arc::new(Mutex::new(false)),
            disable_state_exporter,
            transport,
            conf,
            block_retriever,
        }
    }

    fn ignore_casper_message(&self, hash: BlockHash) -> Result<bool, CasperError> {
        let blocks_in_processing = self
            .blocks_in_processing
            .lock()
            .map_err(|_| {
                CasperError::RuntimeError("Failed to lock blocks_in_processing".to_string())
            })?
            .contains(&hash);
        let buffer_contains = self.casper.buffer_contains(&hash);
        let dag_contains = self.casper.dag_contains(&hash);
        Ok(blocks_in_processing || buffer_contains || dag_contains)
    }

    pub async fn handle_block_hash_message(
        &self,
        peer: PeerNode,
        bhm: BlockHashMessage,
        ignore_message_f: impl Fn(BlockHash) -> Result<bool, CasperError>,
    ) -> Result<(), CasperError> {
        let h = bhm.block_hash;
        if ignore_message_f(h.clone())? {
            log::debug!(
                "Ignoring {} hash broadcast",
                PrettyPrinter::build_string_bytes(&h)
            );
        } else {
            log::debug!(
                "Incoming BlockHashMessage {} from {}",
                PrettyPrinter::build_string_bytes(&h),
                peer.endpoint.host
            );
            self.block_retriever
                .admit_hash(
                    h,
                    Some(peer),
                    block_retriever::AdmitHashReason::HashBroadcastReceived,
                )
                .await?;
        }
        Ok(())
    }

    pub async fn handle_has_block_message(
        &self,
        peer: PeerNode,
        hb: HasBlock,
        ignore_message_f: impl Fn(BlockHash) -> Result<bool, CasperError>,
    ) -> Result<(), CasperError> {
        let h = hb.hash;
        if ignore_message_f(h.clone())? {
            log::debug!(
                "Ignoring {} HasBlockMessage",
                PrettyPrinter::build_string_bytes(&h)
            );
        } else {
            log::debug!(
                "Incoming HasBlockMessage {} from {}",
                PrettyPrinter::build_string_bytes(&h),
                peer.endpoint.host
            );
            self.block_retriever
                .admit_hash(
                    h,
                    Some(peer),
                    block_retriever::AdmitHashReason::HasBlockMessageReceived,
                )
                .await?;
        }
        Ok(())
    }

    pub async fn handle_block_request(
        &self,
        peer: PeerNode,
        br: BlockRequest,
    ) -> Result<(), CasperError> {
        let maybe_block = self.casper.block_store().get(&br.hash)?;
        if let Some(block) = maybe_block {
            log::info!(
                "Received request for block {} from {}. Response sent.",
                PrettyPrinter::build_string_bytes(&br.hash),
                peer
            );
            self.transport
                .stream_message_to_peer(&self.conf, &peer, Arc::new(block.to_proto()))
                .await?;
        } else {
            log::info!(
                "Received request for block {} from {}. No response given since block not found.",
                PrettyPrinter::build_string_bytes(&br.hash),
                peer
            );
        }
        Ok(())
    }

    pub async fn handle_has_block_request(
        &self,
        peer: PeerNode,
        hbr: HasBlockRequest,
        block_lookup: impl Fn(BlockHash) -> bool,
    ) -> Result<(), CasperError> {
        if block_lookup(hbr.hash.clone()) {
            let has_block = HasBlock { hash: hbr.hash };
            self.transport
                .send_message_to_peer(&self.conf, &peer, Arc::new(has_block.to_proto()))
                .await?;
        }
        Ok(())
    }

    /**
     * Peer asks for fork-choice tip
     */
    // TODO name for this message is misleading, as its a request for all tips, not just fork choice. -- OLD
    pub async fn handle_fork_choice_tip_request(&self, peer: PeerNode) -> Result<(), CasperError> {
        log::info!("Received ForkChoiceTipRequest from {}", peer.endpoint.host);
        let latest_messages = self.casper.block_dag().await?.latest_message_hashes();
        let tips: Vec<BlockHash> = latest_messages
            .iter()
            .map(|entry| entry.value().clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        log::info!(
            "Sending tips {} to {}",
            tips.iter()
                .map(|tip| PrettyPrinter::build_string_bytes(tip))
                .collect::<Vec<_>>()
                .join(", "),
            peer.endpoint.host
        );
        for tip in tips {
            let has_block = HasBlock { hash: tip };
            self.transport
                .send_message_to_peer(&self.conf, &peer, Arc::new(has_block.to_proto()))
                .await?;
        }
        Ok(())
    }

    pub async fn handle_approved_block_request(
        &self,
        peer: PeerNode,
        approved_block: ApprovedBlock,
    ) -> Result<(), CasperError> {
        log::info!("Received ApprovedBlockRequest from {}", peer);
        self.transport
            .stream_message_to_peer(&self.conf, &peer, Arc::new(approved_block.to_proto()))
            .await?;
        log::info!("ApprovedBlock sent to {}", peer);
        Ok(())
    }

    async fn handle_state_items_message_request(
        &self,
        peer: PeerNode,
        start_path: Vec<(Blake2b256Hash, Option<u8>)>,
        skip: u32,
        take: u32,
    ) -> Result<(), CasperError> {
        let exporter = self.casper.get_history_exporter().await;

        let (history, data) = RSpaceExporterItems::get_history_and_data(
            exporter,
            start_path.clone(),
            skip as i32,
            take as i32,
        );
        let resp = casper_message::StoreItemsMessage {
            start_path: start_path,
            last_path: history.last_path,
            history_items: history
                .items
                .into_iter()
                .map(|(k, v)| (k, prost::bytes::Bytes::from(v)))
                .collect(),
            data_items: data
                .items
                .into_iter()
                .map(|(k, v)| (k, prost::bytes::Bytes::from(v)))
                .collect(),
        };
        let resp_proto = resp.to_proto();
        log::info!("Read {:?}", &resp_proto);
        self.transport
            .stream_message_to_peer(&self.conf, &peer, Arc::new(resp_proto))
            .await?;

        log::info!("Store items sent to {}", peer);
        Ok(())
    }
}
