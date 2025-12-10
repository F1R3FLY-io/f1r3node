// See shared/src/test/scala/coop/rchain/store/InMemoryKeyValueStoreSpec.scala
//
// NOTE: This file is located in casper/tests/util instead of shared/tests/store
// because adding rspace_plus_plus as a dependency to shared would create a cyclic dependency:
// shared -> rspace_plus_plus -> shared
// Since KeyValueStoreSut needs KeyValueStoreManager from rspace_plus_plus, and casper already
// depends on both shared and rspace_plus_plus, we place it here to avoid the cycle.

use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use shared::rust::store::{
    key_value_store::{KeyValueStore, KvStoreError},
    key_value_typed_store::KeyValueTypedStore,
};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

/// Typed store wrapper for (i64, String) pairs
/// Similar to report_store.rs CompressedBlockEventInfoStore
struct Int64StringStore {
    store: Arc<dyn KeyValueStore>,
}

impl Int64StringStore {
    fn new(store: Arc<dyn KeyValueStore>) -> Self {
        Self { store }
    }

    fn encode_key(&self, key: &i64) -> Vec<u8> {
        key.to_le_bytes().to_vec()
    }

    fn decode_key(&self, bytes: &[u8]) -> Result<i64, KvStoreError> {
        bytes[0..8]
            .try_into()
            .map(i64::from_le_bytes)
            .map_err(|_| KvStoreError::SerializationError("Invalid key bytes".to_string()))
    }

    fn encode_value(&self, value: &String) -> Vec<u8> {
        value.as_bytes().to_vec()
    }

    fn decode_value(&self, bytes: &[u8]) -> Result<String, KvStoreError> {
        String::from_utf8(bytes.to_vec())
            .map_err(|e| KvStoreError::SerializationError(format!("Invalid UTF-8: {}", e)))
    }
}

impl KeyValueTypedStore<i64, String> for Int64StringStore {
    fn get(&self, keys: &Vec<i64>) -> Result<Vec<Option<String>>, KvStoreError> {
        let keys_encoded: Vec<Vec<u8>> = keys.iter().map(|k| self.encode_key(k)).collect();
        let values_bytes = self.store.get(&keys_encoded)?;

        values_bytes
            .iter()
            .map(|value_opt| {
                value_opt
                    .as_ref()
                    .map(|bytes| self.decode_value(bytes))
                    .transpose()
            })
            .collect()
    }

    fn put(&self, kv_pairs: Vec<(i64, String)>) -> Result<(), KvStoreError> {
        let pairs_encoded: Vec<(Vec<u8>, Vec<u8>)> = kv_pairs
            .iter()
            .map(|(k, v)| (self.encode_key(k), self.encode_value(v)))
            .collect();
        self.store.put(pairs_encoded)
    }

    fn delete(&self, keys: Vec<i64>) -> Result<(), KvStoreError> {
        let keys_encoded: Vec<Vec<u8>> = keys.iter().map(|k| self.encode_key(k)).collect();
        self.store.delete(keys_encoded).map(|_| ())
    }

    fn contains(&self, keys: Vec<i64>) -> Result<Vec<bool>, KvStoreError> {
        let keys_encoded: Vec<Vec<u8>> = keys.iter().map(|k| self.encode_key(k)).collect();
        let results = self.store.get(&keys_encoded)?;
        Ok(results.iter().map(|r| r.is_some()).collect())
    }

    fn collect<F, T>(&self, mut f: F) -> Result<Vec<T>, KvStoreError>
    where
        F: FnMut((&i64, &String)) -> Option<T>,
    {
        let store_map = self.store.to_map()?;
        let mut result = Vec::new();

        for (key_bytes, value_bytes) in store_map {
            let key = self.decode_key(&key_bytes)?;
            let value = self.decode_value(&value_bytes)?;

            if let Some(item) = f((&key, &value)) {
                result.push(item);
            }
        }

        Ok(result)
    }

    fn to_map(&self) -> Result<HashMap<i64, String>, KvStoreError> {
        let mut result = HashMap::new();
        let store_map = self.store.to_map()?;

        for (key_bytes, value_bytes) in store_map {
            let key = self.decode_key(&key_bytes)?;
            let value = self.decode_value(&value_bytes)?;
            result.insert(key, value);
        }

        Ok(result)
    }
}

pub struct KeyValueStoreSut {
    kvm: Box<dyn KeyValueStoreManager>,
}

impl KeyValueStoreSut {
    pub fn new(kvm: Box<dyn KeyValueStoreManager>) -> Self {
        Self { kvm }
    }

    /// Explicitly shutdown the underlying KeyValueStoreManager.
    pub async fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.kvm.shutdown().await?;
        Ok(())
    }

    async fn copy_to_db(
        &mut self,
        data: HashMap<i64, String>,
    ) -> Result<Int64StringStore, Box<dyn Error>> {
        let store = self.kvm.store("test".to_string()).await?;
        let typed_store = Int64StringStore::new(store);

        let kv_pairs: Vec<(i64, String)> = data.into_iter().collect();
        typed_store.put(kv_pairs)?;

        Ok(typed_store)
    }

    pub async fn test_put_get(
        &mut self,
        input: HashMap<i64, String>,
    ) -> Result<HashMap<i64, String>, Box<dyn Error>> {
        let store = self.copy_to_db(input.clone()).await?;
        let keys: Vec<i64> = input.keys().copied().collect();
        let values = store.get(&keys)?;

        let result: HashMap<i64, String> = keys
            .iter()
            .zip(values.iter())
            .filter_map(|(k, v)| v.as_ref().map(|val| (*k, val.clone())))
            .collect();

        Ok(result)
    }

    pub async fn test_put_delete_get(
        &mut self,
        input: HashMap<i64, String>,
        delete_keys: Vec<i64>,
    ) -> Result<HashMap<i64, String>, Box<dyn Error>> {
        let store = self.copy_to_db(input.clone()).await?;
        store.delete(delete_keys)?;
        let result = store.to_map()?;

        Ok(result)
    }

    pub async fn test_put_iterate(
        &mut self,
        input: HashMap<i64, String>,
    ) -> Result<HashMap<i64, String>, Box<dyn Error>> {
        let store = self.copy_to_db(input.clone()).await?;
        let result = store.to_map()?;

        Ok(result)
    }

    pub async fn test_put_collect<F>(
        &mut self,
        input: HashMap<i64, String>,
        pf: F,
    ) -> Result<HashMap<i64, String>, Box<dyn std::error::Error>>
    where
        F: FnMut((&i64, &String)) -> Option<(i64, String)>,
    {
        let store = self.copy_to_db(input.clone()).await?;
        let collect_result = store.collect(pf)?;

        let result: HashMap<i64, String> = collect_result.into_iter().collect();

        Ok(result)
    }
}

// Scala: class InMemoryKeyValueStoreSpec extends FlatSpec with Matchers with GeneratorDrivenPropertyChecks
#[cfg(test)]
mod tests {
    use super::*;
    use lazy_static::lazy_static;
    use proptest::collection::hash_map;
    use proptest::prelude::*;
    use rspace_plus_plus::rspace::shared::in_mem_store_manager::InMemoryStoreManager;

    // Optimization: proptest! macro generates sync functions but our tests are async.
    // Creating a new Runtime for each test case is expensive (proptest runs 256 cases by default).
    // Using a shared lazy_static Runtime is much more efficient.
    lazy_static! {
        static ref RUNTIME: tokio::runtime::Runtime = tokio::runtime::Runtime::new().unwrap();
    }

    fn gen_data() -> impl Strategy<Value = HashMap<i64, String>> {
        hash_map(any::<i64>(), any::<String>(), 0..2000)
    }

    proptest! {
        #[test]
        fn in_memory_key_value_store_should_put_and_get_data_from_the_store(expected in gen_data()) {
            RUNTIME.block_on(async {
                let kvm = InMemoryStoreManager::new();
                let mut sut = KeyValueStoreSut::new(Box::new(kvm));
                let result = sut.test_put_get(expected.clone()).await.unwrap();
                assert_eq!(result, expected);
            });
        }
    }

    proptest! {
        #[test]
        fn in_memory_key_value_store_should_put_and_get_all_data_from_the_store(expected in gen_data()) {
            RUNTIME.block_on(async {
                let kvm = InMemoryStoreManager::new();
                let mut sut = KeyValueStoreSut::new(Box::new(kvm));
                let result = sut.test_put_iterate(expected.clone()).await.unwrap();
                assert_eq!(result, expected);
            });
        }
    }

    proptest! {
        #[test]
        fn in_memory_key_value_store_should_put_and_collect_partial_data_from_the_store(expected in gen_data()) {
            RUNTIME.block_on(async {
                let kvm = InMemoryStoreManager::new();
                let mut sut = KeyValueStoreSut::new(Box::new(kvm));

                if expected.is_empty() {
                    return;
                }

                let keys: Vec<i64> = expected.keys().copied().collect();

                // Scala: val kMin = keys.min
                // Scala: val kMax = keys.min  // Bug in original Scala code - should be keys.max
                // Scala: val kAvg = kMax - kMin / 2
                // Note: Fixed the bug here - using max() instead of min()
                let k_min = *keys.iter().min().unwrap();
                let k_max = *keys.iter().max().unwrap(); // Fixed: was keys.min in Scala
                let k_avg = k_max - k_min / 2;

                let expected_filtered: HashMap<i64, String> = expected
                    .iter()
                    .filter(|(k, _)| **k >= k_avg)
                    .map(|(k, v)| (*k, v.clone()))
                    .collect();

                let result = sut.test_put_collect(expected.clone(), |(&k, v)| {
                    if k >= k_avg {
                        Some((k, v.clone()))
                    } else {
                        None
                    }
                }).await.unwrap();

                assert_eq!(result, expected_filtered);
            });
        }
    }

    proptest! {
        #[test]
        fn in_memory_key_value_store_should_not_have_deleted_keys_in_the_store(input in gen_data()) {
            RUNTIME.block_on(async {
                let kvm = InMemoryStoreManager::new();
                let mut sut = KeyValueStoreSut::new(Box::new(kvm));
                let all_keys: Vec<i64> = input.keys().copied().collect();

                // Take some keys for deletion
                let split_at = all_keys.len() / 2;
                let get_keys: Vec<i64> = all_keys[..split_at].to_vec();
                let delete_keys: Vec<i64> = all_keys[split_at..].to_vec();
                // Expected input without deleted keys
                let expected: HashMap<i64, String> = get_keys
                    .iter()
                    .filter_map(|k| input.get(k).map(|v| (*k, v.clone())))
                    .collect();


                let result = sut.test_put_delete_get(input.clone(), delete_keys).await.unwrap();

                assert_eq!(result, expected);
            });
        }
    }
}
