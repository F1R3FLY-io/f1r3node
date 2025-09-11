// See casper/src/main/scala/coop/rchain/casper/engine/Initializing.scala

use async_trait::async_trait;
use futures::stream::StreamExt;
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
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
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::history::Either;
use rspace_plus_plus::rspace::state::rspace_importer::RSpaceImporterInstance;
use rspace_plus_plus::rspace::state::rspace_state_manager::RSpaceStateManager;
use shared::rust::{
    shared::{f1r3fly_event::F1r3flyEvent, f1r3fly_events::F1r3flyEvents},
    ByteString, ByteVector,
};

use crate::rust::block_status::ValidBlock;
use crate::rust::engine::lfs_tuple_space_requester::StatePartPath;
use crate::rust::estimator::Estimator;
use crate::rust::validate::Validate;
use crate::rust::{
    casper::{CasperShardConf, MultiParentCasper},
    engine::{
        block_retriever::BlockRetriever,
        engine::{log_no_approved_block_available, send_no_approved_block_available, Engine, transition_to_running},
        engine_cell::EngineCell,
        lfs_block_requester::{self, BlockRequesterOps},
        lfs_tuple_space_requester::{self, TupleSpaceRequesterOps},
    },
    errors::CasperError,
    util::rholang::runtime_manager::RuntimeManager,
    util::proto_util,
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
    block_store: Option<KeyValueBlockStore>,
    block_dag_storage: Option<BlockDagKeyValueStorage>,
    deploy_storage: Option<KeyValueDeployStorage>,
    casper_buffer_storage: Option<CasperBufferKeyValueStorage>,
    rspace_state_manager: Option<RSpaceStateManager>,

    // Block processing queue - matches Scala's blockProcessingQueue: Queue[F, (Casper[F], BlockMessage)]
    // Using concrete type to match transition_to_running signature
    block_processing_queue: Arc<Mutex<VecDeque<(Arc<crate::rust::multi_parent_casper_impl::MultiParentCasperImpl<T>>, BlockMessage)>>>,
    blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
    casper_shard_conf: CasperShardConf,
    validator_id: Option<ValidatorIdentity>,
    the_init: Arc<Mutex<Option<Box<dyn FnOnce() -> Result<(), CasperError> + Send + Sync>>>>,
    block_message_queue: Arc<Mutex<VecDeque<BlockMessage>>>,
    tuple_space_queue: mpsc::UnboundedSender<StoreItemsMessage>,
    trim_state: bool,
    disable_state_exporter: bool,

    // TEMP: flag for single call for process approved block (Scala: `val startRequester = Ref.unsafe(true)`)
    start_requester: Arc<Mutex<bool>>,
    /// Event publisher for F1r3fly events
    event_publisher: Arc<F1r3flyEvents>,

    block_retriever: Arc<BlockRetriever<T>>,
    engine_cell: EngineCell,
    runtime_manager: Option<RuntimeManager>,
    estimator: Option<Estimator>,
}

impl<T: TransportLayer + Send + Sync + Clone> Initializing<T> {
    /// Scala equivalent: Constructor for `Initializing` class
    #[allow(clippy::too_many_arguments)]
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
        block_processing_queue: Arc<Mutex<VecDeque<(Arc<crate::rust::multi_parent_casper_impl::MultiParentCasperImpl<T>>, BlockMessage)>>>,
        blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
        casper_shard_conf: CasperShardConf,
        validator_id: Option<ValidatorIdentity>,
        the_init: Box<dyn FnOnce() -> Result<(), CasperError> + Send + Sync>,
        tuple_space_queue: mpsc::UnboundedSender<StoreItemsMessage>,
        trim_state: bool,
        disable_state_exporter: bool,
        event_publisher: Arc<F1r3flyEvents>,
        block_retriever: Arc<BlockRetriever<T>>,
        engine_cell: EngineCell,
        runtime_manager: RuntimeManager,
        estimator: Estimator,
    ) -> Self {
        Self {
            transport_layer,
            rp_conf_ask,
            connections_cell,
            last_approved_block,
            block_store: Some(block_store),
            block_dag_storage: Some(block_dag_storage),
            deploy_storage: Some(deploy_storage),
            casper_buffer_storage: Some(casper_buffer_storage),
            rspace_state_manager: Some(rspace_state_manager),
            block_processing_queue,
            blocks_in_processing,
            casper_shard_conf,
            validator_id,
            the_init: Arc::new(Mutex::new(Some(the_init))),
            block_message_queue: Arc::new(Mutex::new(VecDeque::new())),
            tuple_space_queue,
            trim_state,
            disable_state_exporter,
            start_requester: Arc::new(Mutex::new(true)),
            event_publisher,
            block_retriever,
            engine_cell,
            runtime_manager: Some(runtime_manager),
            estimator: Some(estimator),
        }
    }
}

#[async_trait(?Send)]
impl<T: TransportLayer + Send + Sync + Clone + 'static> Engine for Initializing<T> {
    async fn init(&self) -> Result<(), CasperError> {
        if let Ok(mut guard) = self.the_init.lock() {
            if let Some(init_fn) = guard.take() {
                init_fn()?;
            }
        }
        Ok(())
    }

    async fn handle(&mut self, peer: PeerNode, msg: CasperMessage) -> Result<(), CasperError> {
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
                self.tuple_space_queue
                    .send(store_items_message)
                    .map_err(|e| {
                        CasperError::RuntimeError(format!(
                            "Failed to enqueue StoreItemsMessage: {}",
                            e
                        ))
                    })
            }
            CasperMessage::BlockMessage(block_message) => {
                log::info!(
                    "BlockMessage received {} from {}.",
                    PrettyPrinter::build_string_block_message(&block_message, true),
                    peer
                );
                let mut queue = self.block_message_queue.lock().map_err(|_| {
                    CasperError::RuntimeError(
                        "Failed to acquire block_message_queue lock".to_string(),
                    )
                })?;
                queue.push_back(block_message);
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

    /// Rust-specific: Engine trait object cloning (no Scala equivalent)
    /// Scala doesn't need this since it uses type parameters instead of trait objects
    fn clone_box(&self) -> Box<dyn Engine> {
        // Initializing engine contains non-cloneable resources (Mutex, channels, etc.)
        // and is designed to transition to Running state, not to be cloned.
        // This matches Scala behavior where engines are not cloned.
        panic!("Initializing engine is not designed to be cloned - it transitions to Running state")
    }
}

impl<T: TransportLayer + Send + Sync + Clone> Initializing<T> {
    async fn on_approved_block(
        &mut self,
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
            initializing: &mut Initializing<T>,
            approved_block: &ApprovedBlock,
        ) -> Result<(), CasperError> {
            let block = &approved_block.candidate.block;

            log::info!(
                "Valid approved block {} received. Restoring approved state.",
                PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true)
            );

            if let Some(storage) = initializing.block_dag_storage.as_mut() {
                storage.insert(block, false, true)?;
            } else {
                return Err(CasperError::RuntimeError(
                    "BlockDag storage not available".to_string(),
                ));
            }

            initializing.request_approved_state(approved_block).await?;

            // NOTE: This is not in original Scala code. Added because we changed block_store 
            // to Option<KeyValueBlockStore> to support moving it in create_casper_and_transition_to_running
            initializing
                .block_store
                .as_mut()
                .unwrap()
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

            log::info!(
                "Approved state for block {} is successfully restored.",
                PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true)
            );

            Ok(())
        }

        // TODO: Scala resolve validation of approved block - we should be sure that bootstrap is not lying
        // Might be Validate.approvedBlock is enough but have to check
        let is_valid =
            sender_is_bootstrap && shard_name_is_valid && Validate::approved_block(&approved_block);

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
        &mut self,
        approved_block: &ApprovedBlock,
    ) -> Result<(), CasperError> {
        // Starting minimum block height. When latest blocks are downloaded new minimum will be calculated.
        let block = &approved_block.candidate.block;
        let start_block_number = proto_util::block_number(block);
        let min_block_number_for_deploy_lifespan = std::cmp::max(
            0,
            start_block_number - self.casper_shard_conf.deploy_lifespan,
        );

        // Create channel for incoming block messages (equivalent to Scala's blockMessageQueue)
        // Use unbounded channel to mirror fs2.Queue without backpressure
        let (response_message_tx, response_message_rx) = tokio::sync::mpsc::unbounded_channel();

        // Drain any pre-received messages to the response channel (preserves Scala behaviour)
        {
            let mut queue = self.block_message_queue.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire block_message_queue lock".to_string())
            })?;
            while let Some(msg) = queue.pop_front() {
                response_message_tx.send(msg).map_err(|_| {
                    CasperError::StreamError("Failed to send initial block message".to_string())
                })?;
            }
        }

        // Create block requester wrapper with needed components and stream
        let mut block_requester = BlockRequesterWrapper::new(
            &self.transport_layer,
            &self.connections_cell,
            &self.rp_conf_ask,
            // NOTE: This is not in original Scala code. Added because we changed block_store 
            // to Option<KeyValueBlockStore> to support moving it in create_casper_and_transition_to_running
            self.block_store.as_mut().ok_or_else(|| {
                CasperError::RuntimeError("Block store not available in request_approved_state".to_string())
            })?,
        );

        // Create empty queue for block requester (must be created outside tokio::join! for lifetime reasons)
        let empty_queue = VecDeque::new(); // Empty queue since we drained it above

        // Create tuple space channel outside tokio::join!
        let (_tuple_space_tx, tuple_space_rx) = tokio::sync::mpsc::unbounded_channel();
        let tuple_space_requester = TupleSpaceRequester::new(&self.transport_layer, &self.rp_conf_ask);

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
                self.rspace_state_manager.as_ref().unwrap().importer.clone(),
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
            let mut stream = Box::pin(tuple_space_stream);
            while let Some(_st) = stream.next().await {}
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
        }

        // Transition to Running state
        self.create_casper_and_transition_to_running(&approved_block)
            .await?;

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
        &mut self,
        start_block: BlockMessage,
        min_height: i64,
        height_map: BTreeMap<i64, HashSet<BlockHash>>,
    ) -> Result<(), CasperError> {
        async fn add_block_to_dag<T: TransportLayer + Send + Sync + Clone>(
            initializing: &mut Initializing<T>,
            block: &BlockMessage,
            is_invalid: bool,
        ) -> Result<(), CasperError> {
            log::info!(
                "Adding {}, invalid = {}.",
                PrettyPrinter::build_string(CasperMessage::BlockMessage(block.clone()), true),
                is_invalid
            );

            // Scala equivalent: `BlockDagStorage[F].insert(block, invalid = isInvalid)`
            if let Some(storage) = initializing.block_dag_storage.as_mut() {
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
            let block = self.block_store.as_ref().ok_or_else(|| {
                CasperError::RuntimeError("Block store not available in populate_dag".to_string())
            })?.get_unsafe(&hash);
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
        &mut self,
        approved_block: &ApprovedBlock,
    ) -> Result<(), CasperError> {
        let ab = approved_block.candidate.block.clone();

        // TODO: IDL this clones()
        // Currently we need to clone dependencies (transport_layer, connections_cell, rp_conf_ask, etc.)
        // because hash_set_casper takes ownership of them, but they're also used elsewhere in Initializing.
        // For now, cloning is necessary to satisfy Rust's ownership rules while maintaining
        // compatibility with the existing architecture.
        // let block_retriever_for_casper = BlockRetriever::new(
        //     Arc::new(self.transport_layer.clone()),
        //     Arc::new(self.connections_cell.clone()),
        //     Arc::new(self.rp_conf_ask.clone()),
        // );
        //
        // let events_for_casper = (*self.event_publisher).clone();
        // let runtime_manager = self
        //     .runtime_manager
        //     .take()
        //     .ok_or_else(|| CasperError::RuntimeError("RuntimeManager not available".to_string()))?;
        // let estimator = self
        //     .estimator
        //     .take()
        //     .ok_or_else(|| CasperError::RuntimeError("Estimator not available".to_string()))?;
        // let block_store = self
        //     .block_store
        //     .take()
        //     .ok_or_else(|| CasperError::RuntimeError("Block store not available".to_string()))?;
        // let block_dag_storage = self
        //     .block_dag_storage
        //     .take()
        //     .ok_or_else(|| CasperError::RuntimeError("BlockDag storage not available".to_string()))?;
        // let deploy_storage = self
        //     .deploy_storage
        //     .take()
        //     .ok_or_else(|| CasperError::RuntimeError("Deploy storage not available".to_string()))?;
        // let casper_buffer_storage = self
        //     .casper_buffer_storage
        //     .take()
        //     .ok_or_else(|| CasperError::RuntimeError("Casper buffer storage not available".to_string()))?;
        // let rspace_state_manager = self
        //     .rspace_state_manager
        //     .take()
        //     .ok_or_else(|| CasperError::RuntimeError("RSpace state manager not available".to_string()))?;
        //
        // let casper = crate::rust::casper::hash_set_casper(
        //     block_retriever_for_casper,
        //     events_for_casper,
        //     runtime_manager,
        //     estimator,
        //     block_store,
        //     block_dag_storage,
        //     deploy_storage,
        //     casper_buffer_storage,
        //     self.validator_id.clone(),
        //     self.casper_shard_conf.clone(),
        //     ab,
        //     rspace_state_manager,
        // )?;
        //
        // log::info!("MultiParentCasper instance created.");
        //
        // // **Scala equivalent**: `transitionToRunning[F](...)`
        // crate::rust::engine::engine::transition_to_running(
        //     self.block_processing_queue.clone(),
        //     self.blocks_in_processing.clone(),
        //     casper,
        //     approved_block.clone(),
        //     Box::new(|| Ok(())),
        //     self.disable_state_exporter,
        //     self.connections_cell.clone(),
        //     Arc::new(self.transport_layer.clone()),
        //     self.rp_conf_ask.clone(),
        //     self.block_retriever.clone(),
        //     &self.engine_cell,
        //     &self.event_publisher,
        // )
        // .await?;
        //
        // self.transport_layer
        //     .send_fork_choice_tip_request(&self.connections_cell, &self.rp_conf_ask)
        //     .await
        //     .map_err(CasperError::CommError)?;

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
}

/// Wrapper struct for block request operations
pub struct BlockRequesterWrapper<'a, T: TransportLayer> {
    transport_layer: &'a T,
    connections_cell: &'a ConnectionsCell,
    rp_conf_ask: &'a RPConf,
    block_store: &'a mut KeyValueBlockStore,
}

impl<'a, T: TransportLayer> BlockRequesterWrapper<'a, T> {
    pub fn new(
        transport_layer: &'a T,
        connections_cell: &'a ConnectionsCell,
        rp_conf_ask: &'a RPConf,
        block_store: &'a mut KeyValueBlockStore,
    ) -> Self {
        Self {
            transport_layer,
            connections_cell,
            rp_conf_ask,
            block_store,
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
        get_from_history: impl Fn(Blake2b256Hash) -> Option<ByteVector> + Send + 'static,
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
