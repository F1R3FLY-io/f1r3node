use heed::types::SerdeBincode;
use heed::{Database, Env};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::rust::ByteBuffer;

use super::key_value_store::{KeyValueStore, KvStoreError};

pub struct LmdbKeyValueStore {
    pub env: Arc<Env>,
    pub db: Arc<Mutex<Database<SerdeBincode<ByteBuffer>, SerdeBincode<ByteBuffer>>>>,
}

impl KeyValueStore for LmdbKeyValueStore {
    fn get(&self, keys: &Vec<ByteBuffer>) -> Result<Vec<Option<ByteBuffer>>, KvStoreError> {
        let db = self.db.lock().map_err(|_| {
            KvStoreError::LockError(
                "LMDB Key Value Store: Failed to acquire lock on db".to_string(),
            )
        })?;

        let reader = self.env.read_txn()?;
        let results = keys
            .into_iter()
            .map(|key| db.get(&reader, &key).map_err(|e| e.into()))
            .collect();
        reader.commit()?;
        results
    }

    fn put(&self, kv_pairs: Vec<(ByteBuffer, ByteBuffer)>) -> Result<(), KvStoreError> {
        let db = self.db.lock().map_err(|_| {
            KvStoreError::LockError(
                "LMDB Key Value Store: Failed to acquire lock on db".to_string(),
            )
        })?;

        let mut writer = self.env.write_txn()?;
        for (key, value) in kv_pairs {
            db.put(&mut writer, &key, &value)?;
        }
        writer.commit()?;

        Ok(())
    }

    fn delete(&self, keys: Vec<ByteBuffer>) -> Result<usize, KvStoreError> {
        let db = self.db.lock().map_err(|_| {
            KvStoreError::LockError(
                "LMDB Key Value Store: Failed to acquire lock on db".to_string(),
            )
        })?;

        let mut writer = self.env.write_txn()?;
        let mut delete_count = 0;
        for key in &keys {
            if db.delete(&mut writer, key)? {
                delete_count += 1;
            }
        }
        writer.commit()?;
        Ok(delete_count)
    }

    fn iterate(&self, f: fn(ByteBuffer, ByteBuffer)) -> Result<(), KvStoreError> {
        let db = self.db.lock().map_err(|_| {
            KvStoreError::LockError(
                "LMDB Key Value Store: Failed to acquire lock on db".to_string(),
            )
        })?;

        let reader = self.env.read_txn()?;
        let iter = db.iter(&reader)?;
        for result in iter {
            let (key, value) = result?;
            f(key.to_vec(), value);
        }
        reader.commit()?;
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn KeyValueStore> {
        Box::new(self.clone())
    }

    fn to_map(&self) -> Result<BTreeMap<ByteBuffer, ByteBuffer>, KvStoreError> {
        let db = self.db.lock().map_err(|_| {
            KvStoreError::LockError(
                "LMDB Key Value Store: Failed to acquire lock on db".to_string(),
            )
        })?;

        let reader = self.env.read_txn()?;
        let iter = db.iter(&reader)?;
        let mut map = BTreeMap::new();
        for result in iter {
            let (key, value) = result?;
            map.insert(key.to_vec(), value);
        }
        reader.commit()?;
        Ok(map)
    }

    // This is only needed for testing purposes
    fn size_bytes(&self) -> usize {
        todo!()
    }

    fn print_store(&self) -> Result<(), KvStoreError> {
        let kv_store_map = self.to_map()?;

        for (key, value) in &kv_store_map {
            println!(
                "Key: {:?}, Value: {:?}",
                hex::encode(key),
                hex::encode(value)
            );
        }

        Ok(())
    }
}

impl LmdbKeyValueStore {
    pub fn new(env: Env, db: Database<SerdeBincode<ByteBuffer>, SerdeBincode<ByteBuffer>>) -> Self {
        let env_arc = Arc::new(env);
        let db_arc = Arc::new(Mutex::new(db));

        LmdbKeyValueStore {
            env: env_arc,
            db: db_arc,
        }
    }
}

impl Clone for LmdbKeyValueStore {
    fn clone(&self) -> Self {
        Self {
            db: self.db.clone(),
            env: self.env.clone(),
        }
    }
}
