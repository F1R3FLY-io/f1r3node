// Per-node local buffer of deploys rejected during multi-parent merge.
//
// When the merge algorithm drops a deploy from the canonical merged state,
// its data is placed here so the block creator can re-propose it in a
// subsequent block. Each validator maintains its own buffer; there is no
// cross-validator coordination.
//
// Mirrors KeyValueDeployStorage in shape and storage backing.

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

#[derive(Clone)]
pub struct KeyValueRejectedDeployBuffer {
    pub store: KeyValueTypedStoreImpl<ByteString, Signed<DeployData>>,
}

impl KeyValueRejectedDeployBuffer {
    pub async fn new(kvm: &mut impl KeyValueStoreManager) -> Result<Self, KvStoreError> {
        let buffer_kv_store = kvm.store("rejected_deploy_buffer".to_string()).await?;
        let buffer_db: KeyValueTypedStoreImpl<ByteString, Signed<DeployData>> =
            KeyValueTypedStoreImpl::new(buffer_kv_store);
        Ok(Self { store: buffer_db })
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

    pub fn remove_by_sig(&mut self, sig: &[u8]) -> Result<bool, KvStoreError> {
        let key: ByteString = sig.to_vec();
        let exists = self
            .store
            .contains(vec![key.clone()])?
            .into_iter()
            .next()
            .unwrap_or(false);
        if !exists {
            return Ok(false);
        }
        self.store.delete(vec![key])?;
        Ok(true)
    }

    pub fn contains_sig(&self, sig: &[u8]) -> Result<bool, KvStoreError> {
        let key: ByteString = sig.to_vec();
        let exists = self
            .store
            .contains(vec![key])?
            .into_iter()
            .next()
            .unwrap_or(false);
        Ok(exists)
    }

    pub fn get_by_sig(&self, sig: &[u8]) -> Result<Option<Signed<DeployData>>, KvStoreError> {
        let key: ByteString = sig.to_vec();
        let results = self.store.get(&vec![key])?;
        Ok(results.into_iter().next().flatten())
    }

    pub fn read_all(&self) -> Result<HashSet<Signed<DeployData>>, KvStoreError> {
        self.store
            .to_map()
            .map(|map| map.into_iter().map(|(_, v)| v).collect())
    }

    pub fn non_empty(&self) -> Result<bool, KvStoreError> {
        self.store.non_empty()
    }
}
