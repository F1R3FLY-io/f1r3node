// See shared/src/main/scala/coop/rchain/store/InMemoryKeyValueStore.scala

use std::collections::BTreeMap;
use std::sync::Arc;

use dashmap::DashMap;
use shared::rust::store::key_value_store::{KeyValueStore, KvStoreError};
use shared::rust::{ByteBuffer, ByteVector};

#[derive(Clone)]
pub struct InMemoryKeyValueStore {
    state: Arc<DashMap<ByteBuffer, ByteVector>>,
}

impl KeyValueStore for InMemoryKeyValueStore {
    fn get(&self, keys: &Vec<ByteBuffer>) -> Result<Vec<Option<ByteBuffer>>, KvStoreError> {
        let result = keys
            .iter()
            .map(|key| self.state.get(key).map(|entry| entry.value().clone()))
            .collect::<Vec<Option<ByteBuffer>>>();

        Ok(result)
    }

    fn put(&self, kv_pairs: Vec<(ByteBuffer, ByteBuffer)>) -> Result<(), KvStoreError> {
        for (key, value) in kv_pairs {
            self.state.insert(key, value);
        }

        Ok(())
    }

    fn delete(&self, keys: Vec<ByteBuffer>) -> Result<usize, KvStoreError> {
        Ok(keys
            .into_iter()
            .filter_map(|key| self.state.remove(&key).map(|(_, v)| v))
            .count())
    }

    fn iterate(&self, _f: fn(ByteBuffer, ByteBuffer)) -> Result<(), KvStoreError> {
        todo!()
    }

    fn clone_box(&self) -> Box<dyn KeyValueStore> {
        Box::new(self.clone())
    }

    fn to_map(&self) -> Result<BTreeMap<ByteBuffer, ByteBuffer>, KvStoreError> {
        let mut map = BTreeMap::new();

        for entry in self.state.iter() {
            map.insert(entry.key().to_vec(), entry.value().to_vec());
        }

        Ok(map)
    }

    fn size_bytes(&self) -> usize {
        self.state
            .iter()
            .map(|entry| entry.key().len() + entry.value().len())
            .sum()
    }

    fn print_store(&self) -> Result<(), KvStoreError> {
        println!("\nIn Mem Key Value Store: {:?}", self.to_map()?);
        Ok(())
    }

    fn non_empty(&self) -> Result<bool, KvStoreError> {
        Ok(!self.state.is_empty())
    }
}

impl InMemoryKeyValueStore {
    pub fn new() -> Self {
        InMemoryKeyValueStore {
            state: Arc::new(DashMap::new()),
        }
    }

    pub fn clear(&self) {
        self.state.clear();
    }

    pub fn num_records(&self) -> usize {
        self.state.len()
    }
}
