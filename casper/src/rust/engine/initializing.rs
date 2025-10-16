// See casper/src/main/scala/coop/rchain/casper/engine/Initializing.scala

use async_trait::async_trait;
use futures::stream::StreamExt;
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::mpsc;
use tokio::time::sleep;

use block_storage::rust::{
    casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
    deploy::key_value_deploy_storage::KeyValueDeployStorage,
    key_value_block_store::KeyValueBlockStore,
};
use comm::rust::{
    peer_node::PeerNode,
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    transport::transport_layer::TransportLayer,
};
use models::rust::casper::protocol::casper_message::StoreItemsMessageRequest;
use models::rust::{
    block_hash::BlockHash,
    casper::{
        pretty_printer::PrettyPrinter,
        protocol::casper_message::{ApprovedBlock, BlockMessage, CasperMessage, StoreItemsMessage},
    },
};
use rspace_plus_plus::rspace::history::Either;
use rspace_plus_plus::rspace::state::rspace_importer::RSpaceImporterInstance;
use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash, state::rspace_importer::RSpaceImporter,
};
use shared::rust::{
    shared::{f1r3fly_event::F1r3flyEvent, f1r3fly_events::F1r3flyEvents},
    ByteString,
};

use crate::rust::block_status::ValidBlock;
use crate::rust::engine::lfs_tuple_space_requester::StatePartPath;
use crate::rust::estimator::Estimator;
use crate::rust::validate::Validate;
use crate::rust::{
    casper::{CasperShardConf, MultiParentCasper},
    engine::{
        block_retriever::BlockRetriever,
        engine::{
            log_no_approved_block_available, send_no_approved_block_available,
            transition_to_running, Engine,
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
    block_store: Arc<Mutex<Option<KeyValueBlockStore>>>,
    block_dag_storage: Arc<Mutex<Option<BlockDagKeyValueStorage>>>,
    deploy_storage: Arc<Mutex<Option<KeyValueDeployStorage>>>,
    casper_buffer_storage: Arc<Mutex<Option<CasperBufferKeyValueStorage>>>,
    rspace_state_manager: Arc<Mutex<Option<RSpaceStateManager>>>,

    // Block processing queue - matches Scala's blockProcessingQueue: Queue[F, (Casper[F], BlockMessage)]
    // Using trait object to support different MultiParentCasper implementations
    block_processing_queue:
        Arc<Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>>,
    blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
    casper_shard_conf: CasperShardConf,
    validator_id: Option<ValidatorIdentity>,
    the_init: Arc<
        dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync,
    >,
    block_message_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<BlockMessage>>>>,
    tuple_space_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<StoreItemsMessage>>>>,
    // Senders to enqueue messages from `handle` (producer side)
    pub block_message_tx: Arc<Mutex<Option<mpsc::UnboundedSender<BlockMessage>>>>,
    pub tuple_space_tx: Arc<Mutex<Option<mpsc::UnboundedSender<StoreItemsMessage>>>>,
    trim_state: bool,
    disable_state_exporter: bool,

    // TEMP: flag for single call for process approved block (Scala: `val startRequester = Ref.unsafe(true)`)
    start_requester: Arc<Mutex<bool>>,
    /// Event publisher for F1r3fly events
    event_publisher: Arc<F1r3flyEvents>,

    block_retriever: Arc<BlockRetriever<T>>,
    engine_cell: Arc<EngineCell>,
    runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
    estimator: Arc<Mutex<Option<Estimator>>>,
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
        casper_buffer_storage: CasperBufferKeyValueStorage,
        rspace_state_manager: RSpaceStateManager,
        block_processing_queue: Arc<
            Mutex<VecDeque<(Arc<dyn MultiParentCasper + Send + Sync>, BlockMessage)>>,
        >,
        blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
        casper_shard_conf: CasperShardConf,
        validator_id: Option<ValidatorIdentity>,
        the_init: Arc<
            dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync,
        >,
        block_message_tx: mpsc::UnboundedSender<BlockMessage>,
        block_message_rx: mpsc::UnboundedReceiver<BlockMessage>,
        tuple_space_tx: mpsc::UnboundedSender<StoreItemsMessage>,
        tuple_space_rx: mpsc::UnboundedReceiver<StoreItemsMessage>,
        trim_state: bool,
        disable_state_exporter: bool,
        event_publisher: Arc<F1r3flyEvents>,
        block_retriever: Arc<BlockRetriever<T>>,
        engine_cell: Arc<EngineCell>,
        runtime_manager: Arc<tokio::sync::Mutex<RuntimeManager>>,
        estimator: Estimator,
    ) -> Self {
        Self {
            transport_layer,
            rp_conf_ask,
            connections_cell,
            last_approved_block,
            block_store: Arc::new(Mutex::new(Some(block_store))),
            block_dag_storage: Arc::new(Mutex::new(Some(block_dag_storage))),
            deploy_storage: Arc::new(Mutex::new(Some(deploy_storage))),
            casper_buffer_storage: Arc::new(Mutex::new(Some(casper_buffer_storage))),
            rspace_state_manager: Arc::new(Mutex::new(Some(rspace_state_manager))),
            block_processing_queue,
            blocks_in_processing,
            casper_shard_conf,
            validator_id,
            the_init,
            block_message_rx: Arc::new(Mutex::new(Some(block_message_rx))),
            tuple_space_rx: Arc::new(Mutex::new(Some(tuple_space_rx))),
            block_message_tx: Arc::new(Mutex::new(Some(block_message_tx))),
            tuple_space_tx: Arc::new(Mutex::new(Some(tuple_space_tx))),
            trim_state,
            disable_state_exporter,
            start_requester: Arc::new(Mutex::new(true)),
            event_publisher,
            block_retriever,
            engine_cell,
            runtime_manager,
            estimator: Arc::new(Mutex::new(Some(estimator))),
        }
    }
}

#[async_trait]
impl<T: TransportLayer + Send + Sync + Clone + 'static> Engine for Initializing<T> {
    async fn init(&self) -> Result<(), CasperError> {
        (self.the_init)().await
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
                log_no_approved_block_available(&no_approved_block_available.node_identifier);
                sleep(Duration::from_secs(10)).await;
                self.transport_layer
                    .request_approved_block(&self.rp_conf_ask, Some(self.trim_state))
                    .await
                    .map_err(CasperError::CommError)
            }
            CasperMessage::StoreItemsMessage(store_items_message) => {
                log::info!(
                    "Received {} from {}.",
                    store_items_message.clone().pretty(),
                    peer
                );
                // Enqueue into tuple space channel for requester stream
                let send_res = self
                    .tuple_space_tx
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|tx| tx.send(store_items_message));
                match send_res {
                    Some(Ok(())) => {}
                    Some(Err(e)) => {
                        log::warn!(
                            "Failed to enqueue StoreItemsMessage into tuple_space channel: {:?}",
                            e
                        );
                    }
                    None => {
                        log::warn!(
                            "tuple_space_tx sender is None; tuple space channel not available (message not enqueued)"
                        );
                    }
                }
                Ok(())
            }
            CasperMessage::BlockMessage(block_message) => {
                log::info!(
                    "BlockMessage received {} from {}.",
                    PrettyPrinter::build_string_block_message(&block_message, true),
                    peer
                );
                // Enqueue into block message channel for requester stream
                let send_res = self
                    .block_message_tx
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|tx| tx.send(block_message));
                match send_res {
                    Some(Ok(())) => {}
                    Some(Err(e)) => {
                        log::warn!(
                            "Failed to enqueue BlockMessage into block_message channel: {:?}",
                            e
                        );
                    }
                    None => {
                        log::warn!(
                            "block_message_tx sender is None; block message channel not available (message not enqueued)"
                        );
                    }
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
    fn with_casper(&self) -> Option<&dyn MultiParentCasper> {
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

            log::info!(
                "Valid approved block {} received. Restoring approved state.",
                PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true)
            );

            if let Some(storage) = initializing.block_dag_storage.lock().unwrap().as_mut() {
                storage.insert(block, false, true)?;
            } else {
                return Err(CasperError::RuntimeError(
                    "BlockDag storage not available".to_string(),
                ));
            }

            initializing.request_approved_state(approved_block).await?;

            if let Some(store) = initializing.block_store.lock().unwrap().as_mut() {
                store.put_approved_block(approved_block)?;
            } else {
                return Err(CasperError::RuntimeError(
                    "Block store not available when persisting ApprovedBlock".to_string(),
                ));
            }

            {
                let mut last_approved = initializing.last_approved_block.lock().unwrap();
                *last_approved = Some(approved_block.clone());
            }

            let _ = initializing
                .event_publisher
                .publish(F1r3flyEvent::approved_block_received(
                    PrettyPrinter::build_string_no_limit(&block.block_hash),
                ));

            log::info!(
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
            log::info!("Received approved block from bootstrap node.");
        } else {
            log::info!("Invalid LastFinalizedBlock received; refusing to add.");
        }

        if !shard_name_is_valid {
            log::info!(
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
    /// - tuple_space_queue is mpsc::UnboundedSender (async channel sender)
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

        log::info!(
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

        // Create block requester wrapper with needed components and stream
        // Clone Arc<Mutex<Option<KeyValueBlockStore>>> to pass to the wrapper
        let block_store_for_requester = self.block_store.clone();

        let mut block_requester = BlockRequesterWrapper::new(
            &self.transport_layer,
            &self.connections_cell,
            &self.rp_conf_ask,
            block_store_for_requester,
            Box::new(|block| self.validate_block(block)),
        );

        // Create empty queue for block requester (must be created outside tokio::join! for lifetime reasons)
        let empty_queue = VecDeque::new(); // Empty queue since we drained it above

        // Use external tuple space message receiver provided by test (equivalent to Scala tupleSpaceQueue)
        let tuple_space_rx = self.tuple_space_rx.lock().unwrap().take().ok_or_else(|| {
            CasperError::RuntimeError("Tuple space receiver not available".to_string())
        })?;
        let tuple_space_requester =
            TupleSpaceRequester::new(&self.transport_layer, &self.rp_conf_ask);

        // **Scala equivalent**: Create both streams (blockRequestStream and tupleSpaceStream)
        let (block_request_stream_result, tuple_space_stream_result) = tokio::join!(
            lfs_block_requester::stream(
                &approved_block,
                &empty_queue,
                response_message_rx,
                min_block_number_for_deploy_lifespan,
                Duration::from_secs(30),
                &mut block_requester,
            ),
            lfs_tuple_space_requester::stream(
                &approved_block,
                tuple_space_rx,
                Duration::from_secs(120),
                tuple_space_requester,
                self.rspace_state_manager
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .importer
                    .clone(),
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
            Box::pin(tuple_space_stream).for_each(|_| async {}).await;
            log::info!("Rholang state received and saved to store.");
            Ok::<(), CasperError>(())
        };

        // **Scala equivalent**: `fs2.Stream(blockRequestAddDagStream, tupleSpaceLogStream).parJoinUnbounded.compile.drain`
        // Run both futures in parallel until completion
        let (final_state_result, _) = tokio::try_join!(block_request_future, tuple_space_future)?;

        // Now populate DAG with the final state (equivalent to evalMap in Scala)
        if let Some(st) = final_state_result {
            self.populate_dag(
                approved_block.candidate.block.clone(),
                st.lower_bound,
                st.height_map,
            )
            .await?;
        } else {
            log::warn!(
                "request_approved_state: block_request_stream returned no final state (None)"
            );
        }

        // Transition to Running state
        log::info!("request_approved_state: transitioning to Running");
        self.create_casper_and_transition_to_running(&approved_block)
            .await?;
        log::info!("request_approved_state: transition_to_running completed");

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
            log::info!(
                "Adding {}, invalid = {}.",
                PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true),
                is_invalid
            );

            // Scala equivalent: `BlockDagStorage[F].insert(block, invalid = isInvalid)`
            if let Some(storage) = initializing.block_dag_storage.lock().unwrap().as_mut() {
                storage.insert(block, is_invalid, false)?;
            } else {
                return Err(CasperError::RuntimeError(
                    "BlockDag storage not available".to_string(),
                ));
            }
            Ok(())
        }

        log::info!("Adding blocks for approved state to DAG.");

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
            let block = self
                .block_store
                .lock()
                .unwrap()
                .as_ref()
                .ok_or_else(|| {
                    CasperError::RuntimeError(
                        "Block store not available in populate_dag".to_string(),
                    )
                })?
                .get_unsafe(&hash);
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

        log::info!("Blocks for approved state added to DAG.");
        Ok(())
    }

    /// **Scala equivalent**: `private def createCasperAndTransitionToRunning(approvedBlock: ApprovedBlock): F[Unit]`
    async fn create_casper_and_transition_to_running(
        &self,
        approved_block: &ApprovedBlock,
    ) -> Result<(), CasperError> {
        let ab = approved_block.candidate.block.clone();

        let block_retriever_for_casper = BlockRetriever::new(
            Arc::new(self.transport_layer.clone()),
            self.connections_cell.clone(),
            self.rp_conf_ask.clone(),
        );

        let events_for_casper = (*self.event_publisher).clone();
        // RuntimeManager is now Arc<Mutex<RuntimeManager>>, so we clone the Arc
        let runtime_manager = self.runtime_manager.clone();

        let estimator = self
            .estimator
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| CasperError::RuntimeError("Estimator not available".to_string()))?;
        let block_store = self
            .block_store
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| CasperError::RuntimeError("Block store not available".to_string()))?
            .clone();
        let block_dag_storage = self
            .block_dag_storage
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| {
                CasperError::RuntimeError("BlockDag storage not available".to_string())
            })?;
        let deploy_storage =
            self.deploy_storage.lock().unwrap().take().ok_or_else(|| {
                CasperError::RuntimeError("Deploy storage not available".to_string())
            })?;
        let casper_buffer_storage = self
            .casper_buffer_storage
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| {
                CasperError::RuntimeError("Casper buffer storage not available".to_string())
            })?;

        // Pass Arc<Mutex<RuntimeManager>> directly to hash_set_casper
        let casper = crate::rust::casper::hash_set_casper(
            block_retriever_for_casper,
            events_for_casper,
            runtime_manager,
            estimator,
            block_store,
            block_dag_storage,
            deploy_storage,
            casper_buffer_storage,
            self.validator_id.clone(),
            self.casper_shard_conf.clone(),
            ab,
        )?;

        log::info!("create_casper_and_transition_to_running: MultiParentCasper instance created");

        // **Scala equivalent**: `transitionToRunning[F](...)`
        log::info!("create_casper_and_transition_to_running: calling transition_to_running");

        // Create empty async init (matches Scala ().pure[F])
        let the_init = Arc::new(|| {
            Box::pin(async { Ok(()) })
                as Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>>
        });

        transition_to_running(
            self.block_processing_queue.clone(),
            self.blocks_in_processing.clone(),
            Arc::new(casper),
            approved_block.clone(),
            the_init,
            self.disable_state_exporter,
            self.connections_cell.clone(),
            Arc::new(self.transport_layer.clone()),
            self.rp_conf_ask.clone(),
            self.block_retriever.clone(),
            &*self.engine_cell,
            &self.event_publisher,
        )
        .await?;

        log::info!(
            "create_casper_and_transition_to_running: transition_to_running completed successfully"
        );

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
        let store_guard = self.block_store.lock().unwrap();
        let store = store_guard.as_ref().ok_or_else(|| {
            CasperError::RuntimeError("Block store not available in contains_block".to_string())
        })?;
        Ok(store.contains(block_hash)?)
    }

    fn get_block_from_store(&self, block_hash: &BlockHash) -> BlockMessage {
        let store_guard = self.block_store.lock().unwrap();
        let store = store_guard
            .as_ref()
            .expect("Block store not available in get_block_from_store");
        store.get_unsafe(block_hash)
    }

    fn put_block_to_store(
        &mut self,
        block_hash: BlockHash,
        block: &BlockMessage,
    ) -> Result<(), CasperError> {
        let mut store_guard = self.block_store.lock().unwrap();
        let store = store_guard.as_mut().ok_or_else(|| {
            CasperError::RuntimeError("Block store not available in put_block_to_store".to_string())
        })?;
        Ok(store.put(block_hash, &block)?)
    }

    fn validate_block(&self, block: &BlockMessage) -> bool {
        (self.validate_block_fn)(block)
    }
}

/// Wrapper struct for block request operations
pub struct BlockRequesterWrapper<'a, T: TransportLayer> {
    transport_layer: &'a T,
    connections_cell: &'a ConnectionsCell,
    rp_conf_ask: &'a RPConf,
    block_store: Arc<Mutex<Option<KeyValueBlockStore>>>,
    validate_block_fn: Box<dyn Fn(&BlockMessage) -> bool + Send + Sync + 'a>,
}

impl<'a, T: TransportLayer> BlockRequesterWrapper<'a, T> {
    pub fn new(
        transport_layer: &'a T,
        connections_cell: &'a ConnectionsCell,
        rp_conf_ask: &'a RPConf,
        block_store: Arc<Mutex<Option<KeyValueBlockStore>>>,
        validate_block_fn: Box<dyn Fn(&BlockMessage) -> bool + Send + Sync + 'a>,
    ) -> Self {
        Self {
            transport_layer,
            connections_cell,
            rp_conf_ask,
            block_store,
            validate_block_fn,
        }
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
            .send_to_bootstrap(&self.rp_conf_ask, &message_proto)
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
