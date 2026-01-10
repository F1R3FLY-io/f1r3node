// See block-storage/src/main/scala/coop/rchain/blockstorage/finality/LastFinalizedStorage.scala

use models::rust::block_hash::BlockHash;
use shared::rust::store::key_value_store::KvStoreError;

/// Trait for storing and retrieving the last finalized block hash
pub trait LastFinalizedStorage: Send + Sync {
    /// Store a block hash as the last finalized block
    fn put(&self, block_hash: BlockHash) -> Result<(), KvStoreError>;

    /// Retrieve the last finalized block hash
    fn get(&self) -> Result<Option<BlockHash>, KvStoreError>;

    /// Get the last finalized block or return the provided default
    fn get_or_else(&self, block_hash: BlockHash) -> Result<BlockHash, KvStoreError> {
        Ok(self.get()?.unwrap_or(block_hash))
    }

    /// Get the last finalized block or return an error if not found
    fn get_unsafe(&self) -> Result<BlockHash, KvStoreError> {
        self.get()?
            .ok_or_else(|| KvStoreError::KeyNotFound("LastFinalizedBlockNotFound".to_string()))
    }
}

pub mod last_finalized_key_value_storage;
pub mod last_finalized_memory_storage;

pub use last_finalized_key_value_storage::LastFinalizedKeyValueStorage;
pub use last_finalized_memory_storage::LastFinalizedMemoryStorage;

