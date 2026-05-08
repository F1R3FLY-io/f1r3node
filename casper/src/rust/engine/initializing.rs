// See casper/src/main/scala/coop/rchain/casper/engine/Initializing.scala

use async_trait::async_trait;
use dashmap::DashSet;
use futures::stream::StreamExt;
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicUsize, Ordering},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio::time::sleep;

use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
    deploy::{
        key_value_deploy_storage::KeyValueDeployStorage,
        key_value_rejected_deploy_buffer::KeyValueRejectedDeployBuffer,
    },
    key_value_block_store::KeyValueBlockStore,
};
use comm::rust::{
    peer_node::PeerNode,
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    transport::transport_layer::TransportLayer,
};
use models::rust::casper::protocol::casper_message::{
    MergeableEntryRequest, StoreItemsMessageRequest,
};
use models::rust::{
    block_hash::BlockHash,
    casper::{
        pretty_printer::PrettyPrinter,
        protocol::casper_message::{
            ApprovedBlock, BlockMessage, CasperMessage, MergeableEntryResponse, StoreItemsMessage,
        },
    },
};
use rspace_plus_plus::rspace::history::Either;
use rspace_plus_plus::rspace::state::rspace_importer::RSpaceImporterInstance;
use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash, state::rspace_importer::RSpaceImporter,
};
use shared::rust::{
    ByteString,
    shared::{f1r3fly_event::F1r3flyEvent, f1r3fly_events::F1r3flyEvents},
};

use crate::rust::block_status::ValidBlock;
use crate::rust::engine::lfs_tuple_space_requester::StatePartPath;
use crate::rust::estimator::Estimator;
use crate::rust::metrics_constants::{
    CASPER_INIT_APPROVED_BLOCK_RECEIVED_METRIC, CASPER_INIT_ATTEMPTS_METRIC,
    CASPER_INIT_RETRY_NO_APPROVED_BLOCK_METRIC, CASPER_INIT_TIME_TO_APPROVED_BLOCK_METRIC,
    CASPER_INIT_TIME_TO_RUNNING_METRIC, CASPER_METRICS_SOURCE,
    INIT_BLOCK_MESSAGE_QUEUE_PENDING_METRIC, INIT_TUPLE_SPACE_QUEUE_PENDING_METRIC,
};
use crate::rust::validate::Validate;
use crate::rust::{
    casper::{CasperShardConf, MultiParentCasper},
    engine::{
        block_retriever::BlockRetriever,
        engine::{
            Engine, log_no_approved_block_available, send_no_approved_block_available,
            transition_to_running,
        },
        engine_cell::EngineCell,
        lfs_block_requester::{self, BlockRequesterOps},
        lfs_tuple_space_requester::{self, TupleSpaceRequesterOps},
    },
    errors::CasperError,
    util::proto_util,
    util::rholang::runtime_manager::RuntimeManager,
    validator_identity::ValidatorIdentity,
};

/// Scala equivalent: `class Initializing[F[_]](...) extends Engine[F]`
///
/// Initializing engine makes sure node receives Approved State and transitions to Running after
pub struct Initializing<T: TransportLayer + Send + Sync + Clone + 'static> {
    transport_layer: T,
    rp_conf_ask: RPConf,
    connections_cell: ConnectionsCell,
    last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
    block_store: KeyValueBlockStore,
    block_dag_storage: BlockDagKeyValueStorage,
    deploy_storage: KeyValueDeployStorage,
    rejected_deploy_buffer: Arc<Mutex<KeyValueRejectedDeployBuffer>>,
    casper_buffer_storage: CasperBufferKeyValueStorage,
    rspace_state_manager: RSpaceStateManager,

    // Block processing queue - matches Scala's blockProcessingQueue: Queue[F, (Casper[F], BlockMessage)]
    // Using trait object to support different MultiParentCasper implementations
    block_processing_queue_tx:
        mpsc::Sender<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>,
    blocks_in_processing: Arc<DashSet<BlockHash>>,
    casper_shard_conf: CasperShardConf,
    validator_id: Option<ValidatorIdentity>,
    the_init: Arc<
        dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync,
    >,
    block_message_rx: Arc<Mutex<Option<mpsc::Receiver<BlockMessage>>>>,
    tuple_space_rx: Arc<Mutex<Option<mpsc::Receiver<StoreItemsMessage>>>>,
    /// Receives `MergeableEntryResponse` routed from `handle_message_recv`.
    /// Drained by `lfs_block_requester::stream` during `request_approved_state`.
    mergeable_message_rx: Arc<Mutex<Option<mpsc::Receiver<MergeableEntryResponse>>>>,
    // Senders to enqueue messages from `handle` (producer side)
    pub block_message_tx: Arc<Mutex<Option<mpsc::Sender<BlockMessage>>>>,
    pub tuple_space_tx: Arc<Mutex<Option<mpsc::Sender<StoreItemsMessage>>>>,
    pub mergeable_message_tx: Arc<Mutex<Option<mpsc::Sender<MergeableEntryResponse>>>>,
    block_message_queue_pending: Arc<AtomicUsize>,
    tuple_space_queue_pending: Arc<AtomicUsize>,
    trim_state: bool,
    disable_state_exporter: bool,

    // TEMP: flag for single call for process approved block (Scala: `val startRequester = Ref.unsafe(true)`)
    start_requester: Arc<Mutex<bool>>,
    init_started_at: Arc<Mutex<Option<Instant>>>,
    no_approved_block_retries: Arc<Mutex<u64>>,
    /// Event publisher for F1r3fly events
    event_publisher: F1r3flyEvents,

    block_retriever: BlockRetriever<T>,
    engine_cell: Arc<EngineCell>,
    runtime_manager: Arc<RuntimeManager>,
    estimator: Arc<Mutex<Option<Estimator>>>,
    /// Shared reference to heartbeat signal for triggering immediate wake on deploy
    heartbeat_signal_ref: crate::rust::heartbeat_signal::HeartbeatSignalRef,
}

impl<T: TransportLayer + Send + Sync + Clone> Initializing<T> {
    /// Scala equivalent: Constructor for `Initializing` class
    #[allow(clippy::too_many_arguments)]
    // NOTE: Parameter types adapted to match GenesisValidator changes
    // based on discussion with Steven for TestFixture compatibility
    pub fn new(
        transport_layer: T,
        rp_conf_ask: RPConf,
        connections_cell: ConnectionsCell,
        last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
        block_store: KeyValueBlockStore,
        block_dag_storage: BlockDagKeyValueStorage,
        deploy_storage: KeyValueDeployStorage,
        rejected_deploy_buffer: Arc<Mutex<KeyValueRejectedDeployBuffer>>,
        casper_buffer_storage: CasperBufferKeyValueStorage,
        rspace_state_manager: RSpaceStateManager,
        block_processing_queue_tx: mpsc::Sender<(
            Arc<dyn MultiParentCasper + Send + Sync>,
            BlockMessage,
        )>,
        blocks_in_processing: Arc<DashSet<BlockHash>>,
        casper_shard_conf: CasperShardConf,
        validator_id: Option<ValidatorIdentity>,
        the_init: Arc<
            dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync,
        >,
        block_message_tx: mpsc::Sender<BlockMessage>,
        block_message_rx: mpsc::Receiver<BlockMessage>,
        tuple_space_tx: mpsc::Sender<StoreItemsMessage>,
        tuple_space_rx: mpsc::Receiver<StoreItemsMessage>,
        mergeable_message_tx: mpsc::Sender<MergeableEntryResponse>,
        mergeable_message_rx: mpsc::Receiver<MergeableEntryResponse>,
        trim_state: bool,
        disable_state_exporter: bool,
        event_publisher: F1r3flyEvents,
        block_retriever: BlockRetriever<T>,
        engine_cell: Arc<EngineCell>,
        runtime_manager: Arc<RuntimeManager>,
        estimator: Estimator,
        heartbeat_signal_ref: crate::rust::heartbeat_signal::HeartbeatSignalRef,
    ) -> Self {
        let state = Self {
            transport_layer,
            rp_conf_ask,
            connections_cell,
            last_approved_block,
            block_store,
            block_dag_storage,
            deploy_storage,
            rejected_deploy_buffer,
            casper_buffer_storage,
            rspace_state_manager,
            block_processing_queue_tx,
            blocks_in_processing,
            casper_shard_conf,
            validator_id,
            the_init,
            block_message_rx: Arc::new(Mutex::new(Some(block_message_rx))),
            tuple_space_rx: Arc::new(Mutex::new(Some(tuple_space_rx))),
            mergeable_message_rx: Arc::new(Mutex::new(Some(mergeable_message_rx))),
            block_message_tx: Arc::new(Mutex::new(Some(block_message_tx))),
            tuple_space_tx: Arc::new(Mutex::new(Some(tuple_space_tx))),
            mergeable_message_tx: Arc::new(Mutex::new(Some(mergeable_message_tx))),
            block_message_queue_pending: Arc::new(AtomicUsize::new(0)),
            tuple_space_queue_pending: Arc::new(AtomicUsize::new(0)),
            trim_state,
            disable_state_exporter,
            start_requester: Arc::new(Mutex::new(true)),
            init_started_at: Arc::new(Mutex::new(None)),
            no_approved_block_retries: Arc::new(Mutex::new(0)),
            event_publisher,
            block_retriever,
            engine_cell,
            runtime_manager,
            estimator: Arc::new(Mutex::new(Some(estimator))),
            heartbeat_signal_ref,
        };
        metrics::gauge!(
            INIT_BLOCK_MESSAGE_QUEUE_PENDING_METRIC,
            "source" => CASPER_METRICS_SOURCE
        )
        .set(0.0);
        metrics::gauge!(
            INIT_TUPLE_SPACE_QUEUE_PENDING_METRIC,
            "source" => CASPER_METRICS_SOURCE
        )
        .set(0.0);
        state
    }

    fn update_init_queue_metrics(&self) {
        metrics::gauge!(
            INIT_BLOCK_MESSAGE_QUEUE_PENDING_METRIC,
            "source" => CASPER_METRICS_SOURCE
        )
        .set(self.block_message_queue_pending.load(Ordering::Relaxed) as f64);
        metrics::gauge!(
            INIT_TUPLE_SPACE_QUEUE_PENDING_METRIC,
            "source" => CASPER_METRICS_SOURCE
        )
        .set(self.tuple_space_queue_pending.load(Ordering::Relaxed) as f64);
    }
}

#[async_trait]
impl<T: TransportLayer + Send + Sync + Clone + 'static> Engine for Initializing<T> {
    async fn init(&self) -> Result<(), CasperError> {
        metrics::counter!(
            CASPER_INIT_ATTEMPTS_METRIC,
            "source" => CASPER_METRICS_SOURCE
        )
        .increment(1);
        {
            let mut started_at = self.init_started_at.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire init_started_at lock".to_string())
            })?;
            if started_at.is_none() {
                *started_at = Some(Instant::now());
            }
        }
        (self.the_init)().await?;
        // Proactively request ApprovedBlock on init to handle the race condition where
        // the ApprovedBlock was broadcast while this node was still in GenesisValidator state
        // (verifying the UnapprovedBlock). Without this, the node could get stuck forever
        // waiting for an ApprovedBlock that was already sent and dropped.
        self.transport_layer
            .request_approved_block(&self.rp_conf_ask, Some(self.trim_state))
            .await
            .map_err(CasperError::CommError)
    }

    async fn handle(&self, peer: PeerNode, msg: CasperMessage) -> Result<(), CasperError> {
        match msg {
            CasperMessage::ApprovedBlock(approved_block) => {
                self.on_approved_block(peer, approved_block, self.disable_state_exporter)
                    .await
            }
            CasperMessage::ApprovedBlockRequest(approved_block_request) => {
                send_no_approved_block_available(
                    &self.rp_conf_ask,
                    &self.transport_layer,
                    &approved_block_request.identifier,
                    peer,
                )
                .await
            }
            CasperMessage::NoApprovedBlockAvailable(no_approved_block_available) => {
                let retry_count = {
                    let mut retries = self.no_approved_block_retries.lock().map_err(|_| {
                        CasperError::RuntimeError(
                            "Failed to acquire no_approved_block_retries lock".to_string(),
                        )
                    })?;
                    *retries += 1;
                    *retries
                };
                metrics::counter!(
                    CASPER_INIT_RETRY_NO_APPROVED_BLOCK_METRIC,
                    "source" => CASPER_METRICS_SOURCE
                )
                .increment(1);
                log_no_approved_block_available(&no_approved_block_available.node_identifier);
                tracing::info!(
                    retry_count = retry_count,
                    "Retrying approved block request after NoApprovedBlockAvailable"
                );
                sleep(Duration::from_secs(10)).await;
                self.transport_layer
                    .request_approved_block(&self.rp_conf_ask, Some(self.trim_state))
                    .await
                    .map_err(CasperError::CommError)
            }
            CasperMessage::StoreItemsMessage(store_items_message) => {
                tracing::info!(
                    "Received {} from {}.",
                    store_items_message.clone().pretty(),
                    peer
                );
                // Enqueue into tuple space channel for requester stream
                let sender = self.tuple_space_tx.lock().unwrap().as_ref().cloned();
                if let Some(tx) = sender {
                    match tx.send(store_items_message).await {
                        Ok(()) => {
                            let _ = self.tuple_space_queue_pending.fetch_update(
                                Ordering::AcqRel,
                                Ordering::Acquire,
                                |curr| Some(curr + 1),
                            );
                            self.update_init_queue_metrics();
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to enqueue StoreItemsMessage into tuple_space channel: {:?}",
                                e
                            );
                        }
                    }
                } else {
                    tracing::warn!(
                        "tuple_space_tx sender is None; tuple space channel not available (message not enqueued)"
                    );
                }
                Ok(())
            }
            CasperMessage::BlockMessage(block_message) => {
                tracing::info!(
                    "BlockMessage received {} from {}.",
                    PrettyPrinter::build_string_block_message(&block_message, true),
                    peer
                );
                // Enqueue into block message channel for requester stream
                let sender = self.block_message_tx.lock().unwrap().as_ref().cloned();
                if let Some(tx) = sender {
                    match tx.send(block_message).await {
                        Ok(()) => {
                            let _ = self.block_message_queue_pending.fetch_update(
                                Ordering::AcqRel,
                                Ordering::Acquire,
                                |curr| Some(curr + 1),
                            );
                            self.update_init_queue_metrics();
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to enqueue BlockMessage into block_message channel: {:?}",
                                e
                            );
                        }
                    }
                } else {
                    tracing::warn!(
                        "block_message_tx sender is None; block message channel not available (message not enqueued)"
                    );
                }
                Ok(())
            }
            CasperMessage::MergeableEntryResponse(resp) => {
                // Forward to the channel drained by lfs_block_requester::stream.
                let sender = self.mergeable_message_tx.lock().unwrap().as_ref().cloned();
                if let Some(tx) = sender {
                    if let Err(e) = tx.send(resp).await {
                        tracing::warn!(
                            "Failed to enqueue MergeableEntryResponse into mergeable channel: {:?}",
                            e
                        );
                    }
                } else {
                    tracing::warn!(
                        "mergeable_message_tx sender is None; mergeable channel not available (message dropped)"
                    );
                }
                Ok(())
            }
            _ => {
                // **Scala equivalent**: `case _ => ().pure`
                Ok(())
            }
        }
    }

    /// Scala equivalent: Engine trait - Initializing doesn't have casper yet, so withCasper returns default
    /// In Scala: `def withCasper[A](f: MultiParentCasper[F] => F[A], default: F[A]): F[A] = default`
    fn with_casper(&self) -> Option<Arc<dyn MultiParentCasper + Send + Sync>> {
        None
    }
}

impl<T: TransportLayer + Send + Sync + Clone> Initializing<T> {
    async fn on_approved_block(
        &self,
        sender: PeerNode,
        approved_block: ApprovedBlock,
        _disable_state_exporter: bool,
    ) -> Result<(), CasperError> {
        let sender_is_bootstrap = self
            .rp_conf_ask
            .bootstrap
            .as_ref()
            .map(|bootstrap| bootstrap == &sender)
            .unwrap_or(false);
        let received_shard = approved_block.candidate.block.shard_id.clone();
        let expected_shard = self.casper_shard_conf.shard_name.clone();
        let shard_name_is_valid = received_shard == expected_shard;

        async fn handle_approved_block<T: TransportLayer + Send + Sync + Clone>(
            initializing: &Initializing<T>,
            approved_block: &ApprovedBlock,
        ) -> Result<(), CasperError> {
            let block = &approved_block.candidate.block;

            tracing::info!(
                "Valid approved block {} received. Restoring approved state.",
                PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true)
            );

            initializing.block_dag_storage.insert(block, false, true)?;

            initializing.request_approved_state(approved_block).await?;

            initializing
                .block_store
                .put_approved_block(approved_block)?;

            {
                let mut last_approved = initializing.last_approved_block.lock().unwrap();
                *last_approved = Some(approved_block.clone());
            }

            let _ = initializing
                .event_publisher
                .publish(F1r3flyEvent::approved_block_received(
                    PrettyPrinter::build_string_no_limit(&block.block_hash),
                ));

            tracing::info!(
                "Approved state for block {} is successfully restored.",
                PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true)
            );

            Ok(())
        }

        // TODO: Scala resolve validation of approved block - we should be sure that bootstrap is not lying
        // Might be Validate.approvedBlock is enough but have to check
        let validate_ok = Validate::approved_block(&approved_block);
        let is_valid = sender_is_bootstrap && shard_name_is_valid && validate_ok;

        if is_valid {
            tracing::info!("Received approved block from bootstrap node.");
        } else {
            tracing::info!("Invalid LastFinalizedBlock received; refusing to add.");
        }

        if !shard_name_is_valid {
            tracing::info!(
                "Connected to the wrong shard. Approved block received from bootstrap is in shard \
                '{}' but expected is '{}'. Check configuration option shard-name.",
                received_shard,
                expected_shard
            );
        }

        let start = {
            let mut requester = self.start_requester.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire start_requester lock".to_string())
            })?;
            match (*requester, is_valid) {
                (true, true) => {
                    *requester = false;
                    true
                }
                (true, false) => {
                    // *requester stays true (no change needed)
                    false
                }
                _ => false,
            }
        };

        if start {
            metrics::counter!(
                CASPER_INIT_APPROVED_BLOCK_RECEIVED_METRIC,
                "source" => CASPER_METRICS_SOURCE
            )
            .increment(1);
            let no_approved_block_retries =
                *self.no_approved_block_retries.lock().map_err(|_| {
                    CasperError::RuntimeError(
                        "Failed to acquire no_approved_block_retries lock".to_string(),
                    )
                })?;
            if let Some(started_at) = *self.init_started_at.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire init_started_at lock".to_string())
            })? {
                let elapsed = started_at.elapsed();
                metrics::histogram!(
                    CASPER_INIT_TIME_TO_APPROVED_BLOCK_METRIC,
                    "source" => CASPER_METRICS_SOURCE
                )
                .record(elapsed.as_secs_f64());
                tracing::info!(
                    retries = no_approved_block_retries,
                    elapsed_ms = elapsed.as_millis(),
                    "Approved block accepted during initialization"
                );
            }
            handle_approved_block(self, &approved_block).await?;
        }
        Ok(())
    }

    /// **Scala equivalent**: `def requestApprovedState(approvedBlock: ApprovedBlock): F[Unit]`
    ///
    /// This function is functionally equivalent to the Scala version, though the implementation differs
    /// due to fundamental differences between Scala fs2 streams and Rust tokio channels:
    ///
    /// Scala approach:
    /// - Uses fs2 Queue (async) for both blockMessageQueue and tupleSpaceQueue
    /// - Passes queues directly to LfsBlockRequester.stream and LfsTupleSpaceRequester.stream
    /// - fs2 handles async message passing internally
    ///
    /// Rust approach (this implementation):
    /// - block_message_queue is Arc<Mutex<VecDeque>> (sync) for thread-safe access
    /// - tuple_space_queue is mpsc::Sender (async channel sender)
    /// - For block messages: drains existing sync queue into new async channel, then uses that channel
    /// - For tuple space: uses existing sender directly
    ///
    /// The functional result is identical: both block and tuple space streams are processed
    /// concurrently, DAG is populated with final state, and system transitions to Running.
    /// The difference is in the underlying queue/channel implementation details.
    async fn request_approved_state(
        &self,
        approved_block: &ApprovedBlock,
    ) -> Result<(), CasperError> {
        // Starting minimum block height. When latest blocks are downloaded new minimum will be calculated.
        let block = &approved_block.candidate.block;
        let start_block_number = proto_util::block_number(block);
        let min_block_number_for_deploy_lifespan = std::cmp::max(
            0,
            start_block_number - self.casper_shard_conf.deploy_lifespan,
        );
        // Forward-horizon lower bound: also fetch every block reachable as a
        // parent of any block within `max_parent_depth + depth_buffer` of LFB.
        // Without this, lfs_block_requester rejects side-branch blocks below
        // `deploy_lifespan` (default 50) even though they're still legitimate
        // parents of upcoming proposals. After Running, gossip blocks
        // referencing those rejected ancestors trigger
        // `DAG storage is missing hash` (KvStoreError) and
        // `validateAndSetCurrentRoot FAILED ... not in roots store`
        // (RootRepositoryDivergence) on the joiner.
        let max_parent_depth = self.casper_shard_conf.max_parent_depth as i64;
        let depth_buffer = self
            .casper_shard_conf
            .mergeable_channels_gc_depth_buffer as i64;
        let min_block_number_for_horizon = if max_parent_depth >= i32::MAX as i64 {
            // Disabled depth check → no horizon clamping needed. Caller can
            // opt into full replay (`disable-lfs = true`) for that scenario.
            min_block_number_for_deploy_lifespan
        } else {
            std::cmp::max(0, start_block_number - max_parent_depth - depth_buffer)
        };
        // Take the LOWER of the two so LFS covers both the deploy-lifespan
        // window AND the parent-merge horizon.
        let min_block_number_for_deploy_lifespan = std::cmp::min(
            min_block_number_for_deploy_lifespan,
            min_block_number_for_horizon,
        );

        tracing::info!(
            "request_approved_state: start (block {}, min_height {})",
            PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true),
            min_block_number_for_deploy_lifespan
        );

        // Use external block message receiver provided by test (equivalent to Scala blockMessageQueue)
        let response_message_rx =
            self.block_message_rx
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| {
                    CasperError::RuntimeError("Block message receiver not available".to_string())
                })?;

        let mergeable_response_rx = self
            .mergeable_message_rx
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| {
                CasperError::RuntimeError(
                    "Mergeable-entry response receiver not available".to_string(),
                )
            })?;

        // Create block requester wrapper with needed components and stream
        let mut block_requester = BlockRequesterWrapper::new(
            &self.transport_layer,
            &self.connections_cell,
            &self.rp_conf_ask,
            self.block_store.clone(),
            Box::new(|block| self.validate_block(block)),
        )
        .with_runtime_manager(self.runtime_manager.clone());

        // Create empty queue for block requester (must be created outside tokio::join! for lifetime reasons)
        let empty_queue = VecDeque::new(); // Empty queue since we drained it above

        // Use external tuple space message receiver provided by test (equivalent to Scala tupleSpaceQueue)
        let tuple_space_rx = self.tuple_space_rx.lock().unwrap().take().ok_or_else(|| {
            CasperError::RuntimeError("Tuple space receiver not available".to_string())
        })?;
        let tuple_space_requester =
            TupleSpaceRequester::new(&self.transport_layer, &self.rp_conf_ask);

        // Keep LFS retry cadence configurable instead of hard-coding a long startup delay.
        // Falls back to 5s when env var is absent or invalid.
        let lfs_request_timeout = Duration::from_secs(5);

        // **Scala equivalent**: Create both streams (blockRequestStream and tupleSpaceStream)
        let (block_request_stream_result, tuple_space_stream_result) = tokio::join!(
            lfs_block_requester::stream(
                &approved_block,
                &empty_queue,
                response_message_rx,
                self.block_message_queue_pending.clone(),
                mergeable_response_rx,
                min_block_number_for_deploy_lifespan,
                lfs_request_timeout,
                &mut block_requester,
            ),
            lfs_tuple_space_requester::stream(
                &approved_block,
                tuple_space_rx,
                self.tuple_space_queue_pending.clone(),
                lfs_request_timeout,
                tuple_space_requester,
                self.rspace_state_manager.importer.clone(),
            )
        );

        let block_request_stream = block_request_stream_result?;
        let tuple_space_stream = tuple_space_stream_result?;

        // **Scala equivalent**: `blockRequestAddDagStream = blockRequestStream.last.unNoneTerminate.evalMap { st => populateDag(...) }`
        // Process block request stream and return the final state for later DAG population
        let block_request_future = async move {
            // Process the stream to completion and get the last state
            let mut stream = Box::pin(block_request_stream);
            let mut last_st = None;
            while let Some(st) = stream.next().await {
                last_st = Some(st);
            }
            Ok::<Option<lfs_block_requester::ST<BlockHash>>, CasperError>(last_st)
        };

        // **Scala equivalent**: `tupleSpaceLogStream = tupleSpaceStream ++ fs2.Stream.eval(Log[F].info(...)).drain`
        // Process tuple space stream and log completion message
        let tuple_space_future = async move {
            // Stream items are processed by the stream itself, we just consume them to completion
            let mut stream = Box::pin(tuple_space_stream);
            while let Some(_) = stream.next().await {}
            tracing::info!("Rholang state received and saved to store.");
            Ok::<(), CasperError>(())
        };

        // **Scala equivalent**: `fs2.Stream(blockRequestAddDagStream, tupleSpaceLogStream).parJoinUnbounded.compile.drain`
        // Run both futures to completion; avoid canceling one branch if the other errors first.
        let (final_state_result, tuple_space_result) =
            tokio::join!(block_request_future, tuple_space_future);
        let final_state_result = final_state_result?;
        tuple_space_result?;

        // Now populate DAG with the final state (equivalent to evalMap in Scala)
        if let Some(st) = final_state_result {
            self.populate_dag(
                approved_block.candidate.block.clone(),
                st.lower_bound,
                st.height_map,
            )
            .await?;
        } else {
            tracing::warn!(
                "request_approved_state: block_request_stream returned no final state (None)"
            );
        }

        // Forward-horizon rspace history sync — ship rspace post-state for
        // every block within `max_parent_depth + depth_buffer` of LFB so
        // subsequent block validation never hits `UnknownRootError`. See
        // `casper/src/rust/util/rspace_history_horizon.rs` for the
        // reachability calc and `casper/src/rust/engine/lfs_horizon_requester.rs`
        // for the orchestrator. Companion to the proposer-side
        // `Estimator::filterDeepParents` and the validator-side parent-depth
        // check in `validate::parents`.
        {
            let dag = self.block_dag_storage.get_representation();
            let horizon_roots = crate::rust::util::rspace_history_horizon::compute_forward_horizon_roots(
                &dag,
                &self.block_store,
                &approved_block.candidate.block,
                &self.casper_shard_conf,
            )
            .map_err(|e| CasperError::KvStoreError(e))?;

            if !horizon_roots.is_empty() {
                // Phase 1's tuple_space_message_receiver was consumed by
                // lfs_tuple_space_requester::stream. Install a fresh
                // (tx, rx) pair on `tuple_space_tx` so handle_message_recv
                // routes incoming `StoreItemsMessage`s to the orchestrator.
                let (horizon_tx, horizon_rx) = mpsc::channel::<StoreItemsMessage>(50);
                {
                    let mut sender_slot = self.tuple_space_tx.lock().unwrap();
                    *sender_slot = Some(horizon_tx);
                }

                let request_timeout = Duration::from_secs(30);
                tracing::info!(
                    "LFS forward-horizon: requesting {} ancestor rspace roots below LFB",
                    horizon_roots.len()
                );

                // Consume the streaming-parallel orchestrator the same way
                // `lfs_tuple_space_requester::stream` is consumed above:
                // drive to completion, then check the final ST.is_finished()
                // to detect incomplete sync.
                use futures::StreamExt;
                let horizon_requester = HorizonRequester::new(
                    &self.transport_layer,
                    &self.rp_conf_ask,
                );
                let horizon_stream =
                    crate::rust::engine::lfs_horizon_requester::stream(
                        horizon_roots,
                        self.runtime_manager.clone(),
                        self.rspace_state_manager.importer.clone(),
                        horizon_requester,
                        horizon_rx,
                        request_timeout,
                    )
                    .await;

                // Drop the temporary sender so subsequent StoreItemsMessages
                // (none expected once Running) don't queue indefinitely.
                let final_horizon_state = match horizon_stream {
                    Ok(stream) => {
                        let mut stream = Box::pin(stream);
                        let mut final_state = None;
                        while let Some(st) = stream.next().await {
                            final_state = Some(st);
                        }
                        Ok(final_state)
                    }
                    Err(e) => Err(e),
                };
                {
                    let mut sender_slot = self.tuple_space_tx.lock().unwrap();
                    *sender_slot = None;
                }

                // Loud failure: cannot transition to Running without a
                // complete forward horizon. Subsequent block validation
                // would hit `UnknownRootError` and cascade-invalidate.
                match final_horizon_state? {
                    Some(st) if st.is_finished() => {}
                    Some(st) => {
                        return Err(CasperError::RuntimeError(format!(
                            "LFS forward-horizon: incomplete sync; {} chunk paths still pending (state machine has {} entries)",
                            st.len() - st.done_count(),
                            st.len(),
                        )));
                    }
                    None => {
                        return Err(CasperError::RuntimeError(
                            "LFS forward-horizon: stream produced no final state".to_string(),
                        ));
                    }
                }
            } else {
                tracing::info!(
                    "LFS forward-horizon: skipped (max_parent_depth unlimited or LFB at genesis)"
                );
            }
        }

        // Transition to Running state
        tracing::info!("request_approved_state: transitioning to Running");
        self.create_casper_and_transition_to_running(&approved_block)
            .await?;
        tracing::info!("request_approved_state: transition_to_running completed");

        Ok(())
    }

    fn validate_block(&self, block: &BlockMessage) -> bool {
        let block_number = proto_util::block_number(block);
        if block_number == 0 {
            // TODO: validate genesis (zero) block correctly - OLD
            true
        } else {
            match Validate::block_hash(block) {
                Either::Right(ValidBlock::Valid) => true,
                _ => false,
            }
        }
    }

    async fn populate_dag(
        &self,
        start_block: BlockMessage,
        min_height: i64,
        height_map: BTreeMap<i64, HashSet<BlockHash>>,
    ) -> Result<(), CasperError> {
        async fn add_block_to_dag<T: TransportLayer + Send + Sync + Clone>(
            initializing: &Initializing<T>,
            block: &BlockMessage,
            is_invalid: bool,
        ) -> Result<(), CasperError> {
            tracing::info!(
                "Adding {}, invalid = {}.",
                PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true),
                is_invalid
            );

            // Scala equivalent: `BlockDagStorage[F].insert(block, invalid = isInvalid)`
            initializing
                .block_dag_storage
                .insert(block, is_invalid, false)?;

            Ok(())
        }

        tracing::info!("Adding blocks for approved state to DAG.");

        let slashed_validators: Vec<ByteString> = start_block
            .body
            .state
            .bonds
            .iter()
            .filter(|bond| bond.stake == 0)
            .map(|bond| bond.validator.to_vec())
            .collect();

        let invalid_blocks: HashSet<ByteString> = start_block
            .justifications
            .iter()
            .filter(|justification| slashed_validators.contains(&justification.validator.to_vec()))
            .map(|justification| justification.latest_block_hash.to_vec())
            .collect();

        // Add sorted DAG in order from approved block to oldest
        for hash in height_map
            .values()
            .flat_map(|hashes| hashes.iter())
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            // NOTE: This is not in original Scala code. Added because we changed block_store
            // to Option<KeyValueBlockStore> to support moving it in create_casper_and_transition_to_running
            let block = self.block_store.get_unsafe(&hash);
            // If sender has stake 0 in approved block, this means that sender has been slashed and block is invalid
            let is_invalid = invalid_blocks.contains(&block.block_hash.to_vec());
            // Filter older not necessary blocks
            let block_height = proto_util::block_number(&block);
            let block_height_ok = block_height >= min_height;

            // Add block to DAG
            if block_height_ok {
                add_block_to_dag(self, &block, is_invalid).await?;
            }
        }

        tracing::info!("Blocks for approved state added to DAG.");
        Ok(())
    }

    /// **Scala equivalent**: `private def createCasperAndTransitionToRunning(approvedBlock: ApprovedBlock): F[Unit]`
    async fn create_casper_and_transition_to_running(
        &self,
        approved_block: &ApprovedBlock,
    ) -> Result<(), CasperError> {
        let ab = approved_block.candidate.block.clone();
        let genesis_post_state_hash = ab.body.state.post_state_hash.clone();

        // RuntimeManager is lock-free Arc<RuntimeManager>; clone the Arc.
        let runtime_manager = self.runtime_manager.clone();

        let estimator = self
            .estimator
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| CasperError::RuntimeError("Estimator not available".to_string()))?;

        // Pass Arc<RuntimeManager> directly to hash_set_casper
        let casper = crate::rust::casper::hash_set_casper(
            self.block_retriever.clone(),
            self.event_publisher.clone(),
            runtime_manager,
            estimator,
            self.block_store.clone(),
            self.block_dag_storage.clone(),
            self.deploy_storage.clone(),
            self.rejected_deploy_buffer.clone(),
            self.casper_buffer_storage.clone(),
            self.validator_id.clone(),
            self.casper_shard_conf.clone(),
            ab,
            self.heartbeat_signal_ref.clone(),
        )?;

        tracing::info!(
            "create_casper_and_transition_to_running: MultiParentCasper instance created"
        );

        // **Scala equivalent**: `transitionToRunning[F](...)`
        tracing::info!("create_casper_and_transition_to_running: calling transition_to_running");

        // Create empty async init (matches Scala ().pure[F])
        let the_init = Arc::new(|| {
            Box::pin(async { Ok(()) })
                as Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>>
        });

        transition_to_running(
            self.block_processing_queue_tx.clone(),
            self.blocks_in_processing.clone(),
            Arc::new(casper),
            approved_block.clone(),
            the_init,
            self.disable_state_exporter,
            Arc::new(self.transport_layer.clone()),
            self.rp_conf_ask.clone(),
            self.block_retriever.clone(),
            &*self.engine_cell,
            &self.event_publisher,
        )
        .await?;

        if let Some(started_at) = *self.init_started_at.lock().map_err(|_| {
            CasperError::RuntimeError("Failed to acquire init_started_at lock".to_string())
        })? {
            let elapsed = started_at.elapsed();
            metrics::histogram!(
                CASPER_INIT_TIME_TO_RUNNING_METRIC,
                "source" => CASPER_METRICS_SOURCE
            )
            .record(elapsed.as_secs_f64());
        }

        tracing::info!(
            "create_casper_and_transition_to_running: transition_to_running completed successfully"
        );

        // Guard joiners (first-time connections requesting an approved block from
        // peers) against config drift: the node's local native-token-* values
        // must match what this network baked into the TokenMetadata contract at
        // genesis. See casper/src/rust/util/token_metadata_check.rs for details.
        crate::rust::util::token_metadata_check::verify_token_metadata_matches_config(
            &self.runtime_manager,
            &genesis_post_state_hash,
            &self.casper_shard_conf.native_token_name,
            &self.casper_shard_conf.native_token_symbol,
            self.casper_shard_conf.native_token_decimals,
        )
        .await?;

        self.transport_layer
            .send_fork_choice_tip_request(&self.connections_cell, &self.rp_conf_ask)
            .await
            .map_err(CasperError::CommError)?;

        Ok(())
    }
}

/// **Scala equivalent**: Engine trait implementation
// Remove the following block:
// impl<T: TransportLayer + Send + Sync> Engine for Initializing<T> { ... }

// Implement BlockRequesterOps trait for the wrapper struct
#[async_trait]
impl<T: TransportLayer + Send + Sync> BlockRequesterOps for BlockRequesterWrapper<'_, T> {
    async fn request_for_block(&self, block_hash: &BlockHash) -> Result<(), CasperError> {
        self.transport_layer
            .broadcast_request_for_block(&self.connections_cell, &self.rp_conf_ask, block_hash)
            .await?;
        Ok(())
    }

    fn contains_block(&self, block_hash: &BlockHash) -> Result<bool, CasperError> {
        Ok(self.block_store.contains(block_hash)?)
    }

    fn get_block_from_store(&self, block_hash: &BlockHash) -> BlockMessage {
        self.block_store.get_unsafe(block_hash)
    }

    fn put_block_to_store(
        &mut self,
        block_hash: BlockHash,
        block: &BlockMessage,
    ) -> Result<(), CasperError> {
        Ok(self.block_store.put(block_hash, &block)?)
    }

    fn validate_block(&self, block: &BlockMessage) -> bool {
        (self.validate_block_fn)(block)
    }

    async fn request_for_mergeable_entry(&self, block_hash: &BlockHash) -> Result<(), CasperError> {
        let req = MergeableEntryRequest {
            block_hash: block_hash.clone(),
        };
        self.transport_layer
            .send_message_to_peers(
                &self.connections_cell,
                &self.rp_conf_ask,
                Arc::new(req.to_proto()),
                None,
            )
            .await
            .map_err(|e| CasperError::RuntimeError(e.to_string()))?;
        Ok(())
    }

    fn put_mergeable_entry(
        &self,
        block_hash: &BlockHash,
        serialized_entry: &[u8],
    ) -> Result<(), CasperError> {
        if serialized_entry.is_empty() {
            return Ok(());
        }
        // Look up the block to compute the local mergeable_key (must match
        // the server's key exactly; both sides derive it from the block's
        // post_state/sender/seq).
        let block = self.block_store.get_unsafe(block_hash);
        let key_bytes = RuntimeManager::mergeable_key_bytes_for_block(&block)?;
        let runtime_manager = self.runtime_manager.as_ref().ok_or_else(|| {
            CasperError::RuntimeError(
                "BlockRequesterWrapper missing runtime_manager (mergeable-entry import)"
                    .to_string(),
            )
        })?;
        runtime_manager.put_mergeable_entry_bytes(key_bytes, serialized_entry.to_vec())
    }
}

/// Wrapper struct for block request operations
pub struct BlockRequesterWrapper<'a, T: TransportLayer> {
    transport_layer: &'a T,
    connections_cell: &'a ConnectionsCell,
    rp_conf_ask: &'a RPConf,
    block_store: KeyValueBlockStore,
    validate_block_fn: Box<dyn Fn(&BlockMessage) -> bool + Send + Sync + 'a>,
    /// Optional runtime_manager handle for the mergeable-channels store
    /// import path. Required in production; optional for tests that don't
    /// exercise the mergeable path.
    runtime_manager: Option<Arc<RuntimeManager>>,
}

impl<'a, T: TransportLayer> BlockRequesterWrapper<'a, T> {
    pub fn new(
        transport_layer: &'a T,
        connections_cell: &'a ConnectionsCell,
        rp_conf_ask: &'a RPConf,
        block_store: KeyValueBlockStore,
        validate_block_fn: Box<dyn Fn(&BlockMessage) -> bool + Send + Sync + 'a>,
    ) -> Self {
        Self {
            transport_layer,
            connections_cell,
            rp_conf_ask,
            block_store,
            validate_block_fn,
            runtime_manager: None,
        }
    }

    /// Attach a `RuntimeManager` so the wrapper can import mergeable-channel
    /// entries.
    pub fn with_runtime_manager(
        mut self,
        runtime_manager: Arc<RuntimeManager>,
    ) -> Self {
        self.runtime_manager = Some(runtime_manager);
        self
    }
}

/// Wrapper struct for tuple space request operations
pub struct TupleSpaceRequester<'a, T: TransportLayer> {
    transport_layer: &'a T,
    rp_conf_ask: &'a RPConf,
}

impl<'a, T: TransportLayer> TupleSpaceRequester<'a, T> {
    pub fn new(transport_layer: &'a T, rp_conf_ask: &'a RPConf) -> Self {
        Self {
            transport_layer,
            rp_conf_ask,
        }
    }
}

// Implement TupleSpaceRequesterOps trait for the wrapper struct
#[async_trait]
impl<T: TransportLayer + Send + Sync> TupleSpaceRequesterOps for TupleSpaceRequester<'_, T> {
    async fn request_for_store_item(
        &self,
        path: &StatePartPath,
        page_size: i32,
    ) -> Result<(), CasperError> {
        let message = StoreItemsMessageRequest {
            start_path: path.clone(),
            skip: 0,
            take: page_size,
        };

        let message_proto = message.to_proto();

        self.transport_layer
            .send_to_bootstrap(&self.rp_conf_ask, Arc::new(message_proto))
            .await?;
        Ok(())
    }

    fn validate_tuple_space_items(
        &self,
        history_items: Vec<(Blake2b256Hash, Vec<u8>)>,
        data_items: Vec<(Blake2b256Hash, Vec<u8>)>,
        start_path: StatePartPath,
        page_size: i32,
        skip: i32,
        get_from_history: Arc<dyn RSpaceImporter>,
    ) -> Result<(), CasperError> {
        Ok(RSpaceImporterInstance::validate_state_items(
            history_items,
            data_items,
            start_path,
            page_size,
            skip,
            get_from_history,
        ))
    }
}

/// Wrapper struct for forward-horizon request operations. Mirrors
/// `TupleSpaceRequester` — same trait shape, same single-peer
/// `send_to_bootstrap` body. Lives here (not in the requester module)
/// for the same reason: the requester module is transport-agnostic and
/// this wrapper is the seam where `TransportLayer` is plugged in.
pub struct HorizonRequester<'a, T: TransportLayer> {
    transport_layer: &'a T,
    rp_conf_ask: &'a RPConf,
}

impl<'a, T: TransportLayer> HorizonRequester<'a, T> {
    pub fn new(transport_layer: &'a T, rp_conf_ask: &'a RPConf) -> Self {
        Self {
            transport_layer,
            rp_conf_ask,
        }
    }
}

#[async_trait]
impl<T: TransportLayer + Send + Sync>
    crate::rust::engine::lfs_horizon_requester::HorizonRequesterOps
    for HorizonRequester<'_, T>
{
    async fn request_for_horizon_chunk(
        &self,
        path: &StatePartPath,
        page_size: i32,
    ) -> Result<(), CasperError> {
        let message = StoreItemsMessageRequest {
            start_path: path.clone(),
            skip: 0,
            take: page_size,
        };
        let message_proto = message.to_proto();
        self.transport_layer
            .send_to_bootstrap(&self.rp_conf_ask, Arc::new(message_proto))
            .await?;
        Ok(())
    }
}
