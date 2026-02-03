// See block-storage/src/test/scala/coop/rchain/blockstorage/dag/IndexedBlockDagStorage.scala

use dashmap::DashMap;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use models::rust::{block_hash::BlockHash, casper::protocol::casper_message::BlockMessage};
use shared::rust::store::key_value_store::KvStoreError;

use crate::rust::dag::{
    block_dag_key_value_storage::{BlockDagKeyValueStorage, KeyValueDagRepresentation},
    equivocation_tracker_store::EquivocationTrackerStore,
};

pub struct IndexedBlockDagStorage {
    underlying: BlockDagKeyValueStorage,
    id_to_blocks: DashMap<i64, BlockMessage>,
    current_id: Arc<Mutex<i64>>,
}

impl IndexedBlockDagStorage {
    pub fn new(underlying: BlockDagKeyValueStorage) -> Self {
        Self {
            underlying,
            id_to_blocks: DashMap::new(),
            current_id: Arc::new(Mutex::new(-1)),
        }
    }

    pub fn get_representation(&self) -> KeyValueDagRepresentation {
        // Use underlying's lock for consistency
        let _lock_guard = self.underlying.global_lock.lock().unwrap();
        self.underlying.get_representation_internal()
    }

    pub fn insert(
        &mut self,
        block: &BlockMessage,
        invalid: bool,
        approved: bool,
    ) -> Result<KeyValueDagRepresentation, KvStoreError> {
        // Use underlying's lock for consistency
        let _lock_guard = self.underlying.global_lock.lock().unwrap();
        self.underlying.insert_internal(block, invalid, approved)
    }

    pub fn insert_indexed(
        &mut self,
        block: &BlockMessage,
        genesis: &BlockMessage,
        invalid: bool,
    ) -> Result<BlockMessage, KvStoreError> {
        // Lock the entire operation to match Scala's lock.withPermit
        // Use underlying's lock to ensure atomicity
        let _lock_guard = self.underlying.global_lock.lock().unwrap();
        
        // Use internal methods to avoid re-acquiring lock
        self.underlying.insert_internal(genesis, false, true)?;
        let dag = self.underlying.get_representation_internal();
        let next_creator_seq_num = if block.seq_num == 0 {
            dag.latest_message(&block.sender)?
                .map_or(-1, |b| b.sequence_number)
                + 1
        } else {
            block.seq_num
        };

        let mut current_id = self.current_id.lock().unwrap();
        let next_id = if block.seq_num == 0 {
            *current_id + 1
        } else {
            block.seq_num.into()
        };

        let mut new_post_state = block.body.state.clone();
        new_post_state.block_number = next_id;

        let mut modified_block = block.clone();
        modified_block.seq_num = next_creator_seq_num;
        modified_block.body.state = new_post_state;

        self.underlying.insert_internal(&modified_block, invalid, false)?;
        self.id_to_blocks.insert(next_id, modified_block.clone());
        *current_id = next_id;

        Ok(modified_block)
    }

    pub fn inject(
        &mut self,
        index: i64,
        block: BlockMessage,
        invalid: bool,
    ) -> Result<(), KvStoreError> {
        // Use underlying's lock for consistency
        let _lock_guard = self.underlying.global_lock.lock().unwrap();
        self.id_to_blocks.insert(index, block.clone());
        self.underlying.insert_internal(&block, invalid, false)?;

        Ok(())
    }

    pub fn access_equivocations_tracker<A>(
        &self,
        f: impl Fn(&EquivocationTrackerStore) -> Result<A, KvStoreError>,
    ) -> Result<A, KvStoreError> {
        // Use underlying's access_equivocations_tracker which has its own lock
        self.underlying.access_equivocations_tracker(f)
    }

    pub async fn record_directly_finalized<F, Fut>(
        &mut self,
        block_hash: BlockHash,
        finalization_effect: F,
    ) -> Result<(), KvStoreError>
    where
        F: FnMut(&HashSet<BlockHash>) -> Fut,
        Fut: std::future::Future<Output = Result<(), KvStoreError>>,
    {
        self.underlying
            .record_directly_finalized(block_hash, finalization_effect)
            .await
    }

    pub fn lookup_by_id(&self, id: i64) -> Result<Option<BlockMessage>, KvStoreError> {
        // DashMap is already thread-safe, so no additional lock needed
        Ok(self.id_to_blocks.get(&id).map(|b| b.clone()))
    }

    pub fn lookup_by_id_unsafe(&self, id: i64) -> BlockMessage {
        // DashMap is already thread-safe, so no additional lock needed
        self.id_to_blocks.get(&id).unwrap().clone()
    }
}
