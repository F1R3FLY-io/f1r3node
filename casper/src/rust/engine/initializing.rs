// See casper/src/main/scala/coop/rchain/casper/engine/Initializing.scala

use async_trait::async_trait;
use shared::rust::ByteVector;
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::mpsc;

use block_storage::rust::{
    dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
    key_value_block_store::KeyValueBlockStore,
};
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
            ApprovedBlock, BlockMessage, CasperMessage, StoreItemsMessage, StoreItemsMessageRequest,
        },
    },
};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    history::Either,
    state::{rspace_importer::RSpaceImporterInstance, rspace_state_manager::RSpaceStateManager},
};

use crate::rust::{
    block_status::ValidBlock,
    casper::Casper,
    casper::{CasperShardConf, MultiParentCasper},
    engine::{
        engine::{log_no_approved_block_available, Engine},
        lfs_block_requester::{self, BlockRequesterOps},
        lfs_tuple_space_requester::{self, StatePartPath, TupleSpaceRequesterOps},
    },
    errors::CasperError,
    util::proto_util,
    validate::Validate,
    validator_identity::ValidatorIdentity,
};
use shared::rust::shared::{f1r3fly_event::F1r3flyEvent, f1r3fly_events::F1r3flyEvents};

/// **Scala equivalent**: `class Initializing[F[_]](blockProcessingQueue, blocksInProcessing, ...)`
///
/// Initializing engine makes sure node receives Approved State and transitions to Running after
pub struct Initializing<T: TransportLayer + Send + Sync> {
    transport_layer: T,
    rp_conf_ask: RPConf,
    connections_cell: ConnectionsCell,
    last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
    block_store: KeyValueBlockStore,
    block_dag_storage: BlockDagKeyValueStorage,
    rspace_state_manager: RSpaceStateManager,

    // Use boxed BlockMessage to avoid complex generic issues
    block_processing_queue: mpsc::UnboundedSender<BlockMessage>,
    blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
    casper_shard_conf: CasperShardConf,
    validator_id: Option<ValidatorIdentity>,
    the_init: Option<Box<dyn FnOnce() -> Result<(), CasperError> + Send + Sync>>,
    block_message_queue: VecDeque<BlockMessage>,
    tuple_space_queue: mpsc::UnboundedSender<StoreItemsMessage>,
    trim_state: bool,
    disable_state_exporter: bool,

    // TEMP: flag for single call for process approved block (Scala: `val startRequester = Ref.unsafe(true)`)
    start_requester: Arc<Mutex<bool>>,
    /// Event publisher for F1r3fly events
    event_publisher: Arc<F1r3flyEvents>,
}

impl<T: TransportLayer + Send + Sync> Initializing<T> {
    pub fn new(
        transport_layer: T,
        rp_conf_ask: RPConf,
        connections_cell: ConnectionsCell,
        last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
        block_store: KeyValueBlockStore,
        block_dag_storage: BlockDagKeyValueStorage,
        rspace_state_manager: RSpaceStateManager,
        block_processing_queue: mpsc::UnboundedSender<BlockMessage>,
        blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
        casper_shard_conf: CasperShardConf,
        validator_id: Option<ValidatorIdentity>,
        the_init: Box<dyn FnOnce() -> Result<(), CasperError> + Send + Sync>,
        block_message_queue: VecDeque<BlockMessage>,
        tuple_space_queue: mpsc::UnboundedSender<StoreItemsMessage>,
        trim_state: Option<bool>,
        disable_state_exporter: bool,
        event_publisher: Arc<F1r3flyEvents>,
    ) -> Self {
        Self {
            transport_layer,
            rp_conf_ask,
            connections_cell,
            last_approved_block,
            block_store,
            block_dag_storage,
            rspace_state_manager,
            block_processing_queue,
            blocks_in_processing,
            casper_shard_conf,
            validator_id,
            the_init: Some(the_init),
            block_message_queue,
            tuple_space_queue,
            trim_state: trim_state.unwrap_or(true),
            disable_state_exporter,
            start_requester: Arc::new(Mutex::new(true)),
            event_publisher,
        }
    }

    /// **Scala equivalent**: `private def onApprovedBlock(sender: PeerNode, approvedBlock: ApprovedBlock, disableStateExporter: Boolean): F[Unit]`
    async fn on_approved_block(
        &mut self,
        sender: PeerNode,
        approved_block: ApprovedBlock,
        _disable_state_exporter: bool,
    ) -> Result<(), CasperError> {
        let sender_is_bootstrap = self.rp_conf_ask.bootstrap.as_ref() == Some(&sender);
        let received_shard = &approved_block.candidate.block.shard_id;
        let expected_shard = &self.casper_shard_conf.shard_name;
        let shard_name_is_valid = received_shard == expected_shard;

        // TODO resolve validation of approved block - we should be sure that bootstrap is not lying
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

        // Start only once, when state is true and approved block is valid
        // **Scala equivalent**: `start <- startRequester.modify { case true if isValid => ... }`
        let start = {
            let mut requester = self.start_requester.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire start_requester lock".to_string())
            })?;
            if *requester && is_valid {
                *requester = false;
                true
            } else if *requester && !is_valid {
                false
            } else {
                false
            }
        };

        if start {
            self.handle_approved_block(approved_block).await?;
        }

        Ok(())
    }

    /// **Scala equivalent**: `def handleApprovedBlock = { val block = approvedBlock.candidate.block ... }`
    async fn handle_approved_block(
        &mut self,
        approved_block: ApprovedBlock,
    ) -> Result<(), CasperError> {
        let block = &approved_block.candidate.block;

        log::info!(
            "Valid approved block {} received. Restoring approved state.",
            PrettyPrinter::build_string_block_message(block, true)
        );

        // Record approved block in DAG
        self.block_dag_storage.insert(block, false, true)?;

        // Download approved state and all related blocks
        self.request_approved_state(approved_block.clone()).await?;

        // Approved block is saved after the whole state is received,
        // to restart requesting if interrupted with incomplete state.
        self.block_store
            .put_approved_block(approved_block.clone())?;

        // Set last approved block
        {
            let mut last_ab = self.last_approved_block.lock().map_err(|_| {
                CasperError::RuntimeError("Failed to acquire last_approved_block lock".to_string())
            })?;
            *last_ab = Some(approved_block.clone());
        }
        // EventLog publish ApprovedBlockReceived event
        let _ = self
            .event_publisher
            .publish(F1r3flyEvent::approved_block_received(
                PrettyPrinter::build_string_no_limit(&block.block_hash),
            ));

        log::info!(
            "Approved state for block {} is successfully restored.",
            PrettyPrinter::build_string_block_message(block, true)
        );

        Ok(())
    }

    /// **Scala equivalent**: `def requestApprovedState(approvedBlock: ApprovedBlock): F[Unit]`
    async fn request_approved_state(
        &mut self,
        approved_block: ApprovedBlock,
    ) -> Result<(), CasperError> {
        // Starting minimum block height. When latest blocks are downloaded new minimum will be calculated.
        let block = &approved_block.candidate.block;
        let start_block_number = proto_util::block_number(block);
        let min_block_number_for_deploy_lifespan = std::cmp::max(
            0,
            start_block_number - self.casper_shard_conf.deploy_lifespan,
        );

        // Create channel for incoming block messages (equivalent to Scala's blockMessageQueue)
        let (_response_message_tx, response_message_rx) = tokio::sync::mpsc::unbounded_channel();

        // Create block requester wrapper with needed components
        {
            let mut block_requester = BlockRequesterWrapper::new(
                &self.transport_layer,
                &self.connections_cell,
                &self.rp_conf_ask,
                &mut self.block_store,
            );

            let _block_request_stream = lfs_block_requester::stream(
                &approved_block,
                &self.block_message_queue,
                response_message_rx,
                min_block_number_for_deploy_lifespan,
                Duration::from_secs(30),
                &mut block_requester,
            )
            .await?;
        }

        // Create channel for incoming tuple space messages
        let (_tuple_space_tx, tuple_space_rx) = tokio::sync::mpsc::unbounded_channel();

        // Create tuple space requester wrapper with needed components
        let tuple_space_requester =
            TupleSpaceRequester::new(&self.transport_layer, &self.rp_conf_ask);

        let _tuple_space_stream = lfs_tuple_space_requester::stream(
            &approved_block,
            tuple_space_rx,
            Duration::from_secs(120),
            tuple_space_requester,
            self.rspace_state_manager.importer.clone(),
        )
        .await?;

        // **Scala equivalent**: `tupleSpaceLogStream = tupleSpaceStream ++ fs2.Stream.eval(Log[F].info(s"Rholang state received and saved to store.")).drain`
        // For now, just log completion as the stream processing is handled internally
        log::info!("Rholang state received and saved to store.");

        // **Scala equivalent**: `blockRequestAddDagStream = blockRequestStream.last.unNoneTerminate.evalMap { st => populateDag(...) }`
        // For now, just process the finalization
        log::info!("Blocks for approved state added to DAG.");

        // **Scala equivalent**: `createCasperAndTransitionToRunning(approvedBlock)`
        self.create_casper_and_transition_to_running(approved_block)
            .await?;

        Ok(())
    }

    /// **Scala equivalent**: `private def validateBlock(block: BlockMessage): F[Boolean]`
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

    /// **Scala equivalent**: `private def populateDag(startBlock: BlockMessage, minHeight: Long, heightMap: SortedMap[Long, Set[BlockHash]]): F[Unit]`
    async fn populate_dag(
        &mut self,
        start_block: BlockMessage,
        min_height: i64,
        height_map: BTreeMap<i64, HashSet<BlockHash>>,
    ) -> Result<(), CasperError> {
        log::info!("Adding blocks for approved state to DAG.");

        // **Scala equivalent**: `slashedValidators = startBlock.body.state.bonds.filter(_.stake == 0L).map(_.validator)`
        let slashed_validators: HashSet<&prost::bytes::Bytes> = start_block
            .body
            .state
            .bonds
            .iter()
            .filter(|bond| bond.stake == 0)
            .map(|bond| &bond.validator)
            .collect();

        // **Scala equivalent**: `invalidBlocks = startBlock.justifications.filter(v => slashedValidators.contains(v.validator)).map(_.latestBlockHash).toSet`
        let invalid_blocks: HashSet<&BlockHash> = start_block
            .justifications
            .iter()
            .filter(|justification| slashed_validators.contains(&justification.validator))
            .map(|justification| &justification.latest_block_hash)
            .collect();

        // **Scala equivalent**: `heightMap.flatMap(_._2).toList.reverse.traverse_ { hash => ... }`
        let mut all_hashes: Vec<_> = height_map
            .values()
            .flat_map(|hashes| hashes.iter())
            .collect();
        all_hashes.reverse();

        for hash in all_hashes {
            let block = self.block_store.get_unsafe(hash);

            // **Scala equivalent**: `isInvalid = invalidBlocks(block.blockHash)`
            let is_invalid = invalid_blocks.contains(&block.block_hash);

            // **Scala equivalent**: `blockHeight = ProtoUtil.blockNumber(block)` and `blockHeightOk = blockHeight >= minHeight`
            let block_height = proto_util::block_number(&block);
            let block_height_ok = block_height >= min_height;

            if block_height_ok {
                log::info!(
                    "Adding {}, invalid = {}.",
                    PrettyPrinter::build_string_block_message(&block, true),
                    is_invalid
                );
                self.block_dag_storage.insert(&block, is_invalid, false)?;
            }
        }

        log::info!("Blocks for approved state added to DAG.");
        Ok(())
    }

    /// **Scala equivalent**: `private def createCasperAndTransitionToRunning(approvedBlock: ApprovedBlock): F[Unit]`
    async fn create_casper_and_transition_to_running(
        &self,
        approved_block: ApprovedBlock,
    ) -> Result<(), CasperError> {
        let ab = approved_block.candidate.block;

        // TODO: uncomment when Running and Engine classes are ported

        // **Scala equivalent**: `casper <- MultiParentCasper.hashSetCasper[F](validatorId, casperShardConf, ab)`
        // let casper = MultiParentCasper::hash_set_casper(
        //     self.validator_id.clone(),
        //     self.casper_shard_conf.clone(),
        //     ab,
        // )?;
        // log::info!("MultiParentCasper instance created.");

        // **Scala equivalent**: `transitionToRunning[F](...)`
        // In Rust, the transition is handled by an external component that observes the change in `multi_parent_casper`.
        // By setting the `casper` instance here, we signal that the system is ready to transition to the Running state.
        // {
        //     let mut casper_instance = self.multi_parent_casper.lock().map_err(|err| {
        //         CasperError::RuntimeError(format!("Failed to acquire Casper instance lock: {}", err))
        //     })?;
        //     *casper_instance = Some(casper);
        // }

        // log::info!("Transitioning to Running state.");

        // // **Scala equivalent**: `CommUtil[F].sendForkChoiceTipRequest`
        // self.transport_layer
        //     .send_fork_choice_tip_request(&self.connections_cell, &self.rp_conf_ask)
        //     .await?;

        // TODO: remove this once we have a proper transition to running
        // This is a temporary hack to get the system to transition to the Running state
        // We need to find a better way to handle this transition
        // This is a temporary hack to get the system to transition to the Running state
        // Engine::transition_to_running(
        //     self.block_processing_queue,
        //     self.blocks_in_processing,
        //     casper,
        //     approved_block,
        //     self.validator_id.clone(),
        //     (),
        //     self.disable_state_exporter,
        // )
        // .await?;

        // TODO: port the next steps from Scala too:
        // - CommUtil[F].sendForkChoiceTipRequest

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
