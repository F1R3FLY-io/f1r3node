use std::sync::{Arc, Mutex};

use shared::rust::{ByteVector, store::key_value_store::KeyValueStore};

use crate::rspace::{
    history::roots_store::{RootsStore, RootsStoreInstances},
    shared::{
        trie_exporter::{KeyHash, Value},
        trie_importer::TrieImporter,
    },
    state::rspace_importer::RSpaceImporter,
};

// See rspace/src/main/scala/coop/rchain/rspace/state/instances/RSpaceImporterStore.scala
pub struct RSpaceImporterStore;

impl RSpaceImporterStore {
    pub fn create(
        history_store: Arc<Mutex<Box<dyn KeyValueStore>>>,
        value_store: Arc<Mutex<Box<dyn KeyValueStore>>>,
        roots_store: Arc<Mutex<Box<dyn KeyValueStore>>>,
    ) -> impl RSpaceImporter {
        RSpaceImporterImpl {
            history_store,
            value_store,
            roots_store,
        }
    }
}

#[derive(Clone)]
pub struct RSpaceImporterImpl {
    pub history_store: Arc<Mutex<Box<dyn KeyValueStore>>>,
    pub value_store: Arc<Mutex<Box<dyn KeyValueStore>>>,
    pub roots_store: Arc<Mutex<Box<dyn KeyValueStore>>>,
}

impl RSpaceImporter for RSpaceImporterImpl {
    fn get_history_item(&self, hash: KeyHash) -> Option<ByteVector> {
        let history_store_lock = self
            .history_store
            .lock()
            .expect("RSpace Importer Store: Failed to acquire lock on history_store");

        history_store_lock
            .get(&vec![hash.bytes()])
            .expect("RSpace Importer: history store get failed")
            .into_iter()
            .next()
            .flatten()
    }
}

impl TrieImporter for RSpaceImporterImpl {
    fn set_history_items(&self, data: Vec<(KeyHash, Value)>) -> () {
        let mut history_store_lock = self
            .history_store
            .lock()
            .expect("RSpace Importer Store: Failed to acquire lock on history_store");

        history_store_lock
            .put(
                data.iter()
                    .map(|pair| (pair.0.bytes(), pair.1.clone()))
                    .collect(),
            )
            .expect("Rspace Importer: failed to put in history store");

        // println!("\nhistory store after set_history_items: {:?}", history_store_lock.to_map());
    }

    fn set_data_items(&self, data: Vec<(KeyHash, Value)>) -> () {
        let mut value_store_lock = self
            .value_store
            .lock()
            .expect("RSpace Importer Store: Failed to acquire lock on value_store");

        value_store_lock
            .put(
                data.iter()
                    .map(|pair| (pair.0.bytes(), pair.1.clone()))
                    .collect(),
            )
            .expect("Rspace Importer: failed to put in value store")
    }

    fn set_root(&self, key: &KeyHash) -> () {
        let roots = RootsStoreInstances::roots_store(self.roots_store.clone());
        roots
            .record_root(key)
            .expect("Rspace Importer: failed to record root")
    }
}
