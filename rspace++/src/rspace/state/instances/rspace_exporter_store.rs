use crate::rspace::errors::RootError;
use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use crate::rspace::history::instances::radix_history::RadixHistory;
use crate::rspace::history::roots_store::{RootsStore, RootsStoreInstances};
use crate::rspace::shared::trie_exporter::{KeyHash, NodePath, TrieExporter, TrieNode, Value};
use crate::rspace::state::rspace_exporter::RSpaceExporter;
use crate::rspace::state::rspace_exporter::RSpaceExporterInstance;
use shared::rust::store::key_value_store::{KeyValueStore, KvStoreError};
use std::sync::Arc;

// See rspace/src/main/scala/coop/rchain/rspace/state/instances/RSpaceExporterStore.scala
pub struct RSpaceExporterStore;

impl RSpaceExporterStore {
    pub fn create(
        history_store: Arc<dyn KeyValueStore>,
        value_store: Arc<dyn KeyValueStore>,
        roots_store: Arc<dyn KeyValueStore>,
    ) -> impl RSpaceExporter {
        RSpaceExporterImpl {
            source_history_store: history_store,
            source_value_store: value_store,
            source_roots_store: roots_store,
        }
    }
}

pub struct RSpaceExporterImpl {
    pub source_history_store: Arc<dyn KeyValueStore>,
    pub source_value_store: Arc<dyn KeyValueStore>,
    pub source_roots_store: Arc<dyn KeyValueStore>,
}

impl RSpaceExporterImpl {
    fn get_items(
        &self,
        store: Arc<dyn KeyValueStore>,
        keys: Vec<Blake2b256Hash>,
    ) -> Result<Vec<(Blake2b256Hash, Value)>, KvStoreError> {
        let loaded = store.get(&keys.iter().map(|key| key.bytes()).collect())?;

        Ok(keys
            .into_iter()
            .zip(loaded.into_iter())
            .filter_map(|(key, value_option)| value_option.map(|value| (key, value)))
            .collect())
    }
}

impl RSpaceExporter for RSpaceExporterImpl {
    fn get_root(&self) -> Result<Blake2b256Hash, RootError> {
        let roots_store = RootsStoreInstances::roots_store(self.source_roots_store.clone());
        let maybe_root = roots_store.current_root()?;
        match maybe_root {
            Some(root) => Ok(root),
            None => Err(RootError::UnknownRootError("No root found".to_string())),
        }
    }
}

impl TrieExporter for RSpaceExporterImpl {
    fn get_nodes(&self, start_path: NodePath, skip: i32, take: i32) -> Vec<TrieNode<KeyHash>> {
        let source_trie_store = RadixHistory::create_store(self.source_history_store.clone());

        let nodes = RSpaceExporterInstance::traverse_history(
            start_path,
            skip,
            take,
            Arc::new(move |key| source_trie_store.get_one(key).ok().flatten()),
        );
        nodes
    }

    fn get_history_items(
        &self,
        keys: Vec<Blake2b256Hash>,
    ) -> Result<Vec<(KeyHash, Value)>, KvStoreError> {
        self.get_items(self.source_history_store.clone(), keys)
    }

    fn get_data_items(
        &self,
        keys: Vec<Blake2b256Hash>,
    ) -> Result<Vec<(KeyHash, Value)>, KvStoreError> {
        let serialized_keys: Vec<_> = keys
            .iter()
            .map(|key| bincode::serialize(&key).unwrap())
            .collect();

        let loaded = self.source_value_store.get(&serialized_keys)?;

        Ok(keys
            .into_iter()
            .zip(loaded.into_iter())
            .filter_map(|(key, value_option)| value_option.map(|value| (key, value)))
            .collect())
    }
}
