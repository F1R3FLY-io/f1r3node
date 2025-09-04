// See casper/src/main/scala/coop/rchain/casper/engine/Initializing.scala

use async_trait::async_trait;
use shared::rust::ByteVector;
use std::{
    collections::{HashSet, VecDeque},
    sync::{Arc, Mutex},
    time::Duration,
};

use block_storage::rust::{
    dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
    key_value_block_store::KeyValueBlockStore,
};
use comm::rust::{
    rp::{connect::ConnectionsCell, rp_conf::RPConf},
    transport::transport_layer::TransportLayer,
};
use models::rust::{
    block_hash::BlockHash,
    casper::protocol::casper_message::{
        ApprovedBlock, BlockMessage, StoreItemsMessage, StoreItemsMessageRequest,
    },
};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    history::Either,
    state::{rspace_importer::RSpaceImporterInstance, rspace_state_manager::RSpaceStateManager},
};

use crate::rust::{
    block_status::ValidBlock,
    casper::CasperShardConf,
    engine::{
        lfs_block_requester::{self, BlockRequesterOps},
        lfs_tuple_space_requester::{self, StatePartPath, TupleSpaceRequesterOps},
    },
    errors::CasperError,
    util::proto_util,
    validate::Validate,
    validator_identity::ValidatorIdentity,
};

pub struct Initializing<T: TransportLayer> {
    transport_layer: T,
    rp_conf_ask: RPConf,
    connections_cell: ConnectionsCell,
    last_approved_block: Arc<Mutex<Option<ApprovedBlock>>>,
    block_store: KeyValueBlockStore,
    block_dag_storage: BlockDagKeyValueStorage,
    rspace_state_manager: RSpaceStateManager,

    block_processing_queue: VecDeque<BlockMessage>,
    blocks_in_processing: Arc<Mutex<HashSet<BlockHash>>>,
    casper_shard_conf: CasperShardConf,
    validator_id: Option<ValidatorIdentity>,
    the_init: Box<dyn FnOnce() -> () + Send + Sync>,
    block_message_queue: VecDeque<BlockMessage>,
    tuple_space_message_receiver: tokio::sync::mpsc::UnboundedReceiver<StoreItemsMessage>,
    trim_state: Option<bool>,
    disable_state_exporter: bool,
}

impl<T: TransportLayer + Sync> Initializing<T> {
    pub async fn request_approved_state(
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

        // TODO: Actually process both streams in parallel
        log::info!("Both LFS streams created successfully (processing not yet implemented)");

        Ok(())
    }
}

// Implement BlockRequesterOps trait for the wrapper struct
#[async_trait]
impl<T: TransportLayer + Sync> BlockRequesterOps for BlockRequesterWrapper<'_, T> {
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
impl<T: TransportLayer + Sync> TupleSpaceRequesterOps for TupleSpaceRequester<'_, T> {
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
