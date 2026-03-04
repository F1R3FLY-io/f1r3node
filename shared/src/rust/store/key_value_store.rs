use std::collections::BTreeMap;
use std::fmt::Debug;

use crate::rust::ByteBuffer;

// See shared/src/main/scala/coop/rchain/store/KeyValueStore.scala
pub trait KeyValueStore: Send + Sync {
    fn get(&self, keys: &Vec<ByteBuffer>) -> Result<Vec<Option<ByteBuffer>>, KvStoreError>;

    fn put(&self, kv_pairs: Vec<(ByteBuffer, ByteBuffer)>) -> Result<(), KvStoreError>;

    fn delete(&self, keys: Vec<ByteBuffer>) -> Result<usize, KvStoreError>;

    fn iterate(&self, f: fn(ByteBuffer, ByteBuffer)) -> Result<(), KvStoreError>;

    fn clone_box(&self) -> Box<dyn KeyValueStore>;

    fn to_map(&self) -> Result<BTreeMap<ByteBuffer, ByteBuffer>, KvStoreError>;

    fn print_store(&self) -> Result<(), KvStoreError>;

    /// Check if the store contains any entries. O(1) time and space.
    fn non_empty(&self) -> Result<bool, KvStoreError>;

    fn contains(&self, keys: &Vec<ByteBuffer>) -> Result<Vec<bool>, KvStoreError> {
        let results = self.get(keys)?;

        Ok(results
            .into_iter()
            .map(|result| !result.is_none())
            .collect())
    }

    // See shared/src/main/scala/coop/rchain/store/KeyValueStoreSyntax.scala

    fn get_one(&self, key: &ByteBuffer) -> Result<Option<ByteBuffer>, KvStoreError> {
        let values = self.get(&vec![key.to_vec()])?;

        match values.split_first() {
            Some((first_value, _)) => Ok(first_value.clone()),
            None => Ok(None),
        }
    }

    fn put_one(&self, key: ByteBuffer, value: ByteBuffer) -> Result<(), KvStoreError> {
        self.put(vec![(key, value)])
    }

    fn put_if_absent(&self, kv_pairs: Vec<(ByteBuffer, ByteBuffer)>) -> Result<(), KvStoreError> {
        let keys: Vec<ByteBuffer> = kv_pairs.iter().map(|(k, _)| k.clone()).collect();
        let if_absent = self.contains(&keys)?;
        let kv_if_absent: Vec<_> = kv_pairs.into_iter().zip(if_absent).collect();
        let kv_absent: Vec<_> = kv_if_absent
            .clone()
            .into_iter()
            .filter(|(_, is_present)| !is_present)
            .map(|(kv, _)| kv)
            .collect();

        self.put(kv_absent)
    }

    fn size_bytes(&self) -> usize;
}

impl Clone for Box<dyn KeyValueStore> {
    fn clone(&self) -> Box<dyn KeyValueStore> {
        self.clone_box()
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum KvStoreError {
    KeyNotFound(String),
    IoError(String),
    SerializationError(String),
    InvalidArgument(String),
    LockError(String),
}

impl std::fmt::Display for KvStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            KvStoreError::KeyNotFound(key) => write!(f, "Key not found: {}", key),
            KvStoreError::IoError(e) => write!(f, "I/O error: {}", e),
            KvStoreError::SerializationError(e) => write!(f, "SerializationError error: {}", e),
            KvStoreError::InvalidArgument(e) => write!(f, "Invalid argument: {}", e),
            KvStoreError::LockError(e) => write!(f, "Lock error: {}", e),
        }
    }
}

impl From<heed::Error> for KvStoreError {
    fn from(error: heed::Error) -> Self {
        KvStoreError::IoError(error.to_string())
    }
}

impl From<Box<bincode::ErrorKind>> for KvStoreError {
    fn from(error: Box<bincode::ErrorKind>) -> Self {
        KvStoreError::SerializationError(error.to_string())
    }
}
