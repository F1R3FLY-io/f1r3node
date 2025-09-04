// See block-storage/src/main/scala/coop/rchain/blockstorage/deploy/KeyValueDeployStorage.scala

use std::collections::HashSet;

use crypto::rust::signatures::signed::Signed;
use models::rust::casper::protocol::casper_message::DeployData;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use shared::rust::{
    store::{
        key_value_store::KvStoreError, key_value_typed_store::KeyValueTypedStore,
        key_value_typed_store_impl::KeyValueTypedStoreImpl,
    },
    ByteString,
};

pub struct KeyValueDeployStorage {
    store: KeyValueTypedStoreImpl<ByteString, Signed<DeployData>>,
}

impl KeyValueDeployStorage {
    pub async fn new(kvm: &mut impl KeyValueStoreManager) -> Result<Self, KvStoreError> {
        let deploy_storage_kv_store = kvm.store("deploy-storage".to_string()).await?;
        let deploy_storage_db: KeyValueTypedStoreImpl<ByteString, Signed<DeployData>> =
            KeyValueTypedStoreImpl::new(deploy_storage_kv_store);
        Ok(Self {
            store: deploy_storage_db,
        })
    }

    pub fn add(&mut self, deploys: Vec<Signed<DeployData>>) -> Result<(), KvStoreError> {
        self.store.put(
            deploys
                .into_iter()
                .map(|d| (d.sig.clone().into(), d))
                .collect(),
        )
    }

    pub fn remove(&mut self, deploys: Vec<Signed<DeployData>>) -> Result<(), KvStoreError> {
        self.store
            .delete(deploys.into_iter().map(|d| d.sig.clone().into()).collect())
    }

    pub fn read_all(&self) -> Result<HashSet<Signed<DeployData>>, KvStoreError> {
        self.store
            .to_map()
            .map(|map| map.into_iter().map(|(_, v)| v).collect())
    }
}
