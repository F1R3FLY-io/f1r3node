use std::sync::Arc;

use super::{
    in_mem_key_value_store::InMemoryKeyValueStore, key_value_store_manager::KeyValueStoreManager,
};
use async_trait::async_trait;
use dashmap::DashMap;
use shared::rust::store::key_value_store::KeyValueStore;

// See shared/src/main/scala/coop/rchain/store/InMemoryStoreManager.scala
// Simple in-memory key value store manager
pub struct InMemoryStoreManager {
    state: DashMap<String, InMemoryKeyValueStore>,
}

#[async_trait]
impl KeyValueStoreManager for InMemoryStoreManager {
    // This method will return a new instance of the store because of cloning the store
    async fn store(&mut self, name: String) -> Result<Arc<dyn KeyValueStore>, heed::Error> {
        let kv_store = self
            .state
            .entry(name)
            .or_insert_with(|| InMemoryKeyValueStore::new());

        Ok(Arc::new(kv_store.value().clone()))
    }

    async fn shutdown(&mut self) -> Result<(), heed::Error> {
        self.state.clear();
        Ok(())
    }
}

impl InMemoryStoreManager {
    pub fn new() -> Self {
        InMemoryStoreManager {
            state: DashMap::new(),
        }
    }
}
