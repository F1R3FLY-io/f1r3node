// Shared test mock implementations to avoid duplication across test files

use shared::rust::store::key_value_store::{KeyValueStore, KvStoreError};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

/// A mock KeyValueStore implementation for testing that uses in-memory HashMap storage.
/// This implementation is thread-safe and supports cloning for use in multi-threaded tests.
#[derive(Clone, Default)]
pub struct MockKeyValueStore {
    data: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
}

impl MockKeyValueStore {
    /// Create a new empty MockKeyValueStore
    pub fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new MockKeyValueStore that shares the same underlying data as another store
    /// This is useful for creating stores that share state between clones
    pub fn with_shared_data(shared_data: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>) -> Self {
        Self { data: shared_data }
    }

    /// Get the current size of the store (number of key-value pairs)
    pub fn len(&self) -> usize {
        self.data.lock().unwrap().len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.data.lock().unwrap().is_empty()
    }
}

impl KeyValueStore for MockKeyValueStore {
    fn get(&self, keys: &Vec<Vec<u8>>) -> Result<Vec<Option<Vec<u8>>>, KvStoreError> {
        let data = self.data.lock().unwrap();
        let results: Vec<Option<Vec<u8>>> = keys.iter().map(|key| data.get(key).cloned()).collect();
        Ok(results)
    }

    fn put(&self, kv_pairs: Vec<(Vec<u8>, Vec<u8>)>) -> Result<(), KvStoreError> {
        let mut data = self.data.lock().unwrap();
        for (key, value) in kv_pairs {
            data.insert(key, value);
        }
        Ok(())
    }

    fn delete(&self, keys: Vec<Vec<u8>>) -> Result<usize, KvStoreError> {
        let mut data = self.data.lock().unwrap();
        let mut count = 0;
        for key in keys {
            if data.remove(&key).is_some() {
                count += 1;
            }
        }
        Ok(count)
    }

    fn contains(&self, keys: &Vec<Vec<u8>>) -> Result<Vec<bool>, KvStoreError> {
        let data = self.data.lock().unwrap();
        let results = keys.iter().map(|key| data.contains_key(key)).collect();
        Ok(results)
    }

    fn to_map(&self) -> Result<BTreeMap<Vec<u8>, Vec<u8>>, KvStoreError> {
        let data = self.data.lock().unwrap();
        Ok(data.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }

    fn iterate(&self, f: fn(Vec<u8>, Vec<u8>)) -> Result<(), KvStoreError> {
        let data = self.data.lock().unwrap();
        for (k, v) in data.iter() {
            f(k.clone(), v.clone());
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn KeyValueStore> {
        Box::new(self.clone())
    }

    fn print_store(&self) -> Result<(), KvStoreError> {
        let data = self.data.lock().unwrap();
        println!("MockKeyValueStore contents ({} items):", data.len());
        for (k, v) in data.iter() {
            println!("  {:?} -> {:?}", k, v);
        }
        Ok(())
    }

    fn size_bytes(&self) -> usize {
        let data = self.data.lock().unwrap();
        // Rough estimate: key size + value size + HashMap overhead
        data.iter()
            .map(|(k, v)| k.len() + v.len() + 16)
            .sum::<usize>()
            + data.len() * 8
    }
}

/// A simple empty KeyValueStore implementation that always returns empty results.
/// Useful for cases where you need a KeyValueStore but don't need it to store anything.
#[derive(Clone, Default)]
pub struct EmptyKeyValueStore;

impl KeyValueStore for EmptyKeyValueStore {
    fn get(&self, keys: &Vec<Vec<u8>>) -> Result<Vec<Option<Vec<u8>>>, KvStoreError> {
        Ok(vec![None; keys.len()])
    }

    fn put(&self, _kv_pairs: Vec<(Vec<u8>, Vec<u8>)>) -> Result<(), KvStoreError> {
        Ok(()) // No-op
    }

    fn delete(&self, _keys: Vec<Vec<u8>>) -> Result<usize, KvStoreError> {
        Ok(0) // Nothing to delete
    }

    fn contains(&self, keys: &Vec<Vec<u8>>) -> Result<Vec<bool>, KvStoreError> {
        Ok(vec![false; keys.len()])
    }

    fn to_map(&self) -> Result<BTreeMap<Vec<u8>, Vec<u8>>, KvStoreError> {
        Ok(BTreeMap::new())
    }

    fn iterate(&self, _f: fn(Vec<u8>, Vec<u8>)) -> Result<(), KvStoreError> {
        Ok(()) // Nothing to iterate over
    }

    fn clone_box(&self) -> Box<dyn KeyValueStore> {
        Box::new(self.clone())
    }

    fn print_store(&self) -> Result<(), KvStoreError> {
        println!("EmptyKeyValueStore (always empty)");
        Ok(())
    }

    fn size_bytes(&self) -> usize {
        0 // Always empty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_key_value_store_basic_operations() {
        let store = MockKeyValueStore::new();

        // Test empty store
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        // Test put and get
        let key = b"test_key".to_vec();
        let value = b"test_value".to_vec();
        store.put(vec![(key.clone(), value.clone())]).unwrap();

        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);

        let results = store.get(&vec![key.clone()]).unwrap();
        assert_eq!(results, vec![Some(value.clone())]);

        // Test contains
        let contains_results = store
            .contains(&vec![key.clone(), b"nonexistent".to_vec()])
            .unwrap();
        assert_eq!(contains_results, vec![true, false]);

        // Test delete
        let deleted_count = store.delete(vec![key]).unwrap();
        assert_eq!(deleted_count, 1);
        assert!(store.is_empty());
    }

    #[test]
    fn test_mock_key_value_store_clone() {
        let store1 = MockKeyValueStore::new();
        let key = b"shared_key".to_vec();
        let value = b"shared_value".to_vec();

        store1.put(vec![(key.clone(), value.clone())]).unwrap();

        // Clone should share the same underlying data
        let store2 = store1.clone();
        let results = store2.get(&vec![key]).unwrap();
        assert_eq!(results, vec![Some(value)]);
    }

    #[test]
    fn test_empty_key_value_store() {
        let store = EmptyKeyValueStore;

        let key = b"any_key".to_vec();
        let results = store.get(&vec![key.clone()]).unwrap();
        assert_eq!(results, vec![None]);

        let contains_results = store.contains(&vec![key]).unwrap();
        assert_eq!(contains_results, vec![false]);

        assert_eq!(store.size_bytes(), 0);
    }
}
