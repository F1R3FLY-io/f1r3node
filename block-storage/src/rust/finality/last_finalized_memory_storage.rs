// See block-storage/src/main/scala/coop/rchain/blockstorage/finality/LastFinalizedMemoryStorage.scala

use std::sync::{Arc, Mutex};

use models::rust::block_hash::BlockHash;
use shared::rust::store::key_value_store::KvStoreError;

use super::LastFinalizedStorage;

/// In-memory implementation of LastFinalizedStorage
/// Uses Arc<Mutex<>> for thread-safe mutable state
pub struct LastFinalizedMemoryStorage {
    state: Arc<Mutex<Option<BlockHash>>>,
}

impl LastFinalizedMemoryStorage {
    /// Create a new LastFinalizedMemoryStorage with empty initial state
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(None)),
        }
    }
}

impl Default for LastFinalizedMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl LastFinalizedStorage for LastFinalizedMemoryStorage {
    fn put(&self, block_hash: BlockHash) -> Result<(), KvStoreError> {
        let mut state = self.state.lock().map_err(|e| {
            KvStoreError::LockError(format!(
                "LastFinalizedMemoryStorage: Failed to acquire lock: {}",
                e
            ))
        })?;
        *state = Some(block_hash);
        Ok(())
    }

    fn get(&self) -> Result<Option<BlockHash>, KvStoreError> {
        let state = self.state.lock().map_err(|e| {
            KvStoreError::LockError(format!(
                "LastFinalizedMemoryStorage: Failed to acquire lock: {}",
                e
            ))
        })?;
        Ok(state.clone())
    }
}