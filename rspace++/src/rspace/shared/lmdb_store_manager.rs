use crate::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use async_trait::async_trait;
use heed::{Database, Env, EnvFlags, EnvOpenOptions, WithoutTls, types::SerdeBincode};
use shared::rust::store::{
    key_value_store::KeyValueStore, lmdb_key_value_store::LmdbKeyValueStore,
};
use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;

// See shared/src/main/scala/coop/rchain/store/LmdbStoreManager.scala
pub struct LmdbStoreManager {
    dir_path: PathBuf,
    max_env_size: usize,
    // Store the env directly (created on first use) so it can be reused for multiple databases
    env: Option<Env<WithoutTls>>,
    dbs: Arc<Mutex<HashMap<String, Database<SerdeBincode<Vec<u8>>, SerdeBincode<Vec<u8>>>>>>,
}

impl LmdbStoreManager {
    pub fn new(dir_path: PathBuf, max_env_size: usize) -> Box<dyn KeyValueStoreManager> {
        Box::new(LmdbStoreManager {
            dir_path,
            max_env_size,
            env: None,
            dbs: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn create_env(&self) -> Result<Env<WithoutTls>, heed::Error> {
        fs::create_dir_all(&self.dir_path)?;

        let mut env_builder = EnvOpenOptions::new().read_txn_without_tls();
        env_builder.map_size(self.max_env_size);
        env_builder.max_dbs(20);
        env_builder.max_readers(2048);

        // SAFETY: We ensure the directory exists and is writable above.
        // The environment is properly managed by this struct's lifecycle.
        unsafe {
            env_builder.flags(EnvFlags::NO_READ_AHEAD);
        }

        let env = unsafe { env_builder.open(&self.dir_path)? };
        Ok(env)
    }
}

#[async_trait]
impl KeyValueStoreManager for LmdbStoreManager {
    async fn store(&mut self, name: String) -> Result<Arc<dyn KeyValueStore>, heed::Error> {
        // Ensure environment is created first
        if self.env.is_none() {
            self.env = Some(self.create_env()?);
        }
        let env = self.env.as_ref().unwrap();

        // Check if database already exists
        {
            let dbs = self.dbs.lock().await;
            if let Some(db) = dbs.get(&name) {
                return Ok(Arc::new(LmdbKeyValueStore::new(env.clone(), db.clone())));
            }
        }

        // Create the database (heed v0.22 requires a write transaction)
        let mut wtxn = env.write_txn()?;
        let db = env.create_database(&mut wtxn, Some(&name))?;
        wtxn.commit()?;

        // Store database reference
        {
            let mut dbs = self.dbs.lock().await;
            dbs.insert(name, db.clone());
        }

        Ok(Arc::new(LmdbKeyValueStore::new(env.clone(), db)))
    }

    async fn shutdown(&mut self) -> Result<(), heed::Error> {
        // Clear the databases HashMap
        let mut dbs = self.dbs.lock().await;
        dbs.clear();

        // Drop the environment
        self.env = None;

        Ok(())
    }
}

// This ensures LMDB environment is closed when the manager is dropped
impl Drop for LmdbStoreManager {
    fn drop(&mut self) {
        // Use try_lock() for synchronous access in Drop
        if let Ok(mut dbs) = self.dbs.try_lock() {
            dbs.clear();
        }
        // The heed::Env Drop implementation will handle closing file handles
        // when self.env is dropped
    }
}
