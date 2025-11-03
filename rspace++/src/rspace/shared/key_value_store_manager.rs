use std::sync::Arc;

use crate::rspace::rspace::RSpaceStore;
use async_trait::async_trait;
use shared::rust::store::key_value_store::KeyValueStore;

// See shared/src/main/scala/coop/rchain/store/KeyValueStoreManager.scala
#[async_trait]
pub trait KeyValueStoreManager: Send + Sync {
    async fn store(&mut self, name: String) -> Result<Arc<dyn KeyValueStore>, heed::Error>;

    async fn shutdown(&mut self) -> Result<(), heed::Error>;

    async fn r_space_stores(&mut self) -> Result<RSpaceStore, heed::Error> {
        self.get_stores("rspace").await
    }

    async fn eval_stores(&mut self) -> Result<RSpaceStore, heed::Error> {
        self.get_stores("eval").await
    }

    // TODO: This function should be private
    async fn get_stores(&mut self, db_prefix: &str) -> Result<RSpaceStore, heed::Error> {
        let history = self.store(format!("{}-history", db_prefix)).await?;
        let roots = self.store(format!("{}-roots", db_prefix)).await?;
        let cold = self.store(format!("{}-cold", db_prefix)).await?;

        Ok(RSpaceStore {
            history,
            roots,
            cold,
        })
    }
}
