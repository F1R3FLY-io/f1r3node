use crate::rspace::errors::HistoryError;
use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use crate::rspace::history::history_action::HistoryAction;
use crate::rspace::history::instances::radix_history::RadixHistory;
use shared::rust::store::key_value_store::KeyValueStore;
use std::sync::{Arc, Mutex};

// See rspace/src/main/scala/coop/rchain/rspace/history/History.scala
pub trait History: Send + Sync {
    fn read(&self, key: Vec<u8>) -> Result<Option<Vec<u8>>, HistoryError>;

    fn process(&self, actions: Vec<HistoryAction>) -> Result<Box<dyn History>, HistoryError>;

    fn root(&self) -> Blake2b256Hash;

    fn reset(&self, root: &Blake2b256Hash) -> Result<Box<dyn History>, HistoryError>;
}

pub struct HistoryInstances;

impl HistoryInstances {
    pub fn create(
        root: Blake2b256Hash,
        store: Arc<Mutex<Box<dyn KeyValueStore>>>,
    ) -> Result<RadixHistory, HistoryError> {
        let typed_store = RadixHistory::create_store(store.to_owned());
        RadixHistory::create(root, typed_store)
    }
}
