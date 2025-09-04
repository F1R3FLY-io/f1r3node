// See shared/src/main/scala/coop/rchain/store/InMemoryKeyValueStore.scala

use std::collections::BTreeMap;

use shared::rust::store::key_value_store::{KeyValueStore, KvStoreError};
use shared::rust::{ByteBuffer, ByteVector};

#[derive(Clone)]
pub struct InMemoryKeyValueStore {
    state: BTreeMap<ByteBuffer, ByteVector>,
}

impl KeyValueStore for InMemoryKeyValueStore {
    fn get(&self, keys: &Vec<ByteBuffer>) -> Result<Vec<Option<ByteBuffer>>, KvStoreError> {
        // println!("\nin_mem_state get: {:?}", self.state);
        // println!("\nin_mem_state get keys: {:?}", keys);
        let result = keys
            .into_iter()
            .map(|key| {
                self.state.get(key).map(|value| {
                    // println!(
                    //     "\nRetrieved value for key {:?}: {:?}",
                    //     key_value.key(),
                    //     key_value.value()
                    // );
                    value.clone()
                })
            })
            .collect::<Vec<Option<ByteBuffer>>>();

        // println!("\nresults in get: {:?}", result);

        Ok(result)
    }

    fn put(&mut self, kv_pairs: Vec<(ByteBuffer, ByteBuffer)>) -> Result<(), KvStoreError> {
        // println!("\nhit put in mem_kv");
        // println!("\nin_mem_state before put: {:?}", self.state);
        for (key, value) in kv_pairs {
            self.state.insert(key, value);
        }

        // println!("\nin_mem_state after put: {:?}", self.state);

        Ok(())
    }

    fn delete(&mut self, keys: Vec<ByteBuffer>) -> Result<usize, KvStoreError> {
        Ok(keys
            .into_iter()
            .filter_map(|key| self.state.remove(&key))
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
            let (key, value) = entry;
            map.insert(
                // bincode::deserialize(key).expect("Mem Key Value Store: Unable to deserialize"),
                // bincode::deserialize(value).expect("Mem Key Value Store: Unable to deserialize"),
                key.to_vec(),
                value.to_vec(),
            );
        }
        Ok(map)
    }

    fn size_bytes(&self) -> usize {
        self.state
            .iter()
            .map(|ref_entry| {
                let (key, value) = ref_entry;
                key.len() + value.len()
            })
            .sum()
    }

    fn print_store(&self) -> () {
        println!("\nIn Mem Key Value Store: {:?}", self.to_map().unwrap())
    }
}

impl InMemoryKeyValueStore {
    pub fn new() -> Self {
        InMemoryKeyValueStore {
            state: BTreeMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.state.clear();
    }

    pub fn num_records(&self) -> usize {
        self.state.len()
    }
}
