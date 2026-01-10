use super::key_value_store_manager::KeyValueStoreManager;
use super::lmdb_store_manager::LmdbStoreManager;
use async_trait::async_trait;
use futures::channel::oneshot;
use shared::rust::store::key_value_store::KeyValueStore;
use std::sync::Arc;
use std::{collections::HashMap, path::PathBuf};
use tokio::sync::Mutex;

/**
 * Specification for LMDB database: unique identifier and database name
 *
 * @param id unique identifier
 * @param nameOverride name to use as database name instead of [[id]]
 */
#[derive(Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct Db {
    id: String,
    name_override: Option<String>,
}

impl Db {
    pub fn new(id: String, name_override: Option<String>) -> Self {
        Db { id, name_override }
    }

    pub fn id(&self) -> &str {
        &self.id
    }
}

// Mega, giga and tera bytes
pub const MB: usize = 1024 * 1024;
pub const GB: usize = 1024 * MB;
pub const TB: usize = 1024 * GB;

#[derive(Clone)]
pub struct LmdbEnvConfig {
    pub name: String,
    pub max_env_size: usize,
    pub max_dbs: u32,
}

impl LmdbEnvConfig {
    pub fn new(name: String, max_env_size: usize) -> Self {
        LmdbEnvConfig { name, max_env_size, max_dbs: 20 }
    }

    pub fn with_max_dbs(mut self, max_dbs: u32) -> Self {
        self.max_dbs = max_dbs;
        self
    }
}

// See shared/src/main/scala/coop/rchain/store/LmdbDirStoreManager.scala
// The idea for this class is to manage multiple of key-value lmdb databases.
// For LMDB this allows control which databases are part of the same environment (file).
pub struct LmdbDirStoreManager {
    dir_path: PathBuf,
    db_mapping: HashMap<Db, LmdbEnvConfig>,
    managers_state: Arc<Mutex<StoreState>>,
}

struct StoreState {
    envs: HashMap<String, oneshot::Receiver<Box<dyn KeyValueStoreManager>>>,
}

#[async_trait]
impl KeyValueStoreManager for LmdbDirStoreManager {
    async fn store(&mut self, db_name: String) -> Result<Arc<dyn KeyValueStore>, heed::Error> {
        let db_instance_mapping: HashMap<&String, (&Db, &LmdbEnvConfig)> = self
            .db_mapping
            .iter()
            .map(|(db, cfg)| (&db.id, (db, cfg)))
            .collect();

        let (sender, receiver) = oneshot::channel::<Box<dyn KeyValueStoreManager>>();

        let action = {
            let mut state = self.managers_state.lock().await;

            let (db, cfg) = db_instance_mapping.get(&db_name).ok_or({
                heed::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("LMDB_Dir_Store_Manager: Key {} was not found", db_name),
                ))
            })?;

            let man_name = cfg.name.to_string();

            let is_new = !state.envs.contains_key(&man_name);
            if is_new {
                state.envs.insert(man_name.to_string(), receiver);
            }

            (is_new, db, cfg)
        };
        let (is_new, db, man_cfg) = action;

        if is_new {
            self.create_lmdb_manager(man_cfg, sender)?;
        }

        let mut manager = {
            let mut state = self.managers_state.lock().await;
            let receiver = state.envs.remove(&man_cfg.name).ok_or({
                heed::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("LMDB_Dir_Store_Manager: Receiver not found"),
                ))
            })?;

            receiver.await.map_err(|e| {
                heed::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("LMDB_Dir_Store_Manager: Failed to receive manager, {}", e),
                ))
            })?
        };

        let database_name = db.name_override.clone().unwrap_or(db.id.clone());
        let database = manager.store(database_name);

        database.await
    }

    async fn shutdown(&mut self) -> Result<(), heed::Error> {
        let mut state = self.managers_state.lock().await;
        for manager_future in state.envs.values_mut() {
            if let Ok(mut manager) = manager_future.await {
                let _ = manager.shutdown().await;
            }
        }

        Ok(())
    }
}

impl LmdbDirStoreManager {
    pub fn new(
        dir_path: PathBuf,
        db_instance_mapping: HashMap<Db, LmdbEnvConfig>,
    ) -> impl KeyValueStoreManager {
        LmdbDirStoreManager {
            dir_path,
            db_mapping: db_instance_mapping,
            managers_state: Arc::new(Mutex::new(StoreState {
                envs: HashMap::new(),
            })),
        }
    }

    fn create_lmdb_manager(
        &self,
        config: &LmdbEnvConfig,
        sender: oneshot::Sender<Box<dyn KeyValueStoreManager>>,
    ) -> Result<(), heed::Error> {
        let manager = LmdbStoreManager::new(self.dir_path.join(&config.name), config.max_env_size, config.max_dbs);
        sender.send(manager).map_err(|_| {
            heed::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to send LMDB manager for {}", config.name),
            ))
        })?;

        Ok(())
    }
}

// Implement Drop
// This ensures all LMDB environments are closed when the manager is dropped
impl Drop for LmdbDirStoreManager {
    fn drop(&mut self) {
        // Use try_lock() for synchronous access in Drop
        if let Ok(mut state) = self.managers_state.try_lock() {
            // Attempt to receive and drop all pending managers
            // This allows their Drop implementations to clean up LMDB environments
            for (_name, mut receiver) in state.envs.drain() {
                // In Drop context we can't await, so we use try_recv()
                // If the receiver is ready, we get the manager and let it drop
                if let Ok(Some(_manager)) = receiver.try_recv() {
                    // Manager will be dropped here, triggering its Drop implementation
                }
            }
        }
    }
}
