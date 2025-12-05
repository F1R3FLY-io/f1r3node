use super::key_value_store_manager::KeyValueStoreManager;
use super::lmdb_store_manager::LmdbStoreManager;
use async_trait::async_trait;
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
}

// Mega, giga and tera bytes
pub const MB: usize = 1024 * 1024;
pub const GB: usize = 1024 * MB;
pub const TB: usize = 1024 * GB;

#[derive(Clone)]
pub struct LmdbEnvConfig {
    pub name: String,
    pub max_env_size: usize,
}

impl LmdbEnvConfig {
    pub fn new(name: String, max_env_size: usize) -> Self {
        LmdbEnvConfig { name, max_env_size }
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
    // Store the actual managers (not receivers) so they can be reused for multiple databases
    // sharing the same LMDB environment path (e.g., rspace-history and rspace-roots both use "rspace/history")
    managers: HashMap<String, Box<dyn KeyValueStoreManager>>,
}

#[async_trait]
impl KeyValueStoreManager for LmdbDirStoreManager {
    async fn store(&mut self, db_name: String) -> Result<Arc<dyn KeyValueStore>, heed::Error> {
        // Build mapping from db.id -> (Db, LmdbEnvConfig)
        let db_instance_mapping: HashMap<&String, (&Db, &LmdbEnvConfig)> = self
            .db_mapping
            .iter()
            .map(|(db, cfg)| (&db.id, (db, cfg)))
            .collect();

        // Look up the database configuration
        let (db, cfg) = db_instance_mapping.get(&db_name).ok_or_else(|| {
            heed::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("LMDB_Dir_Store_Manager: Key {} was not found", db_name),
            ))
        })?;

        let man_name = cfg.name.clone();
        let database_name = db.name_override.clone().unwrap_or(db.id.clone());

        // Check if manager exists, create if not
        {
            let mut state = self.managers_state.lock().await;
            if !state.managers.contains_key(&man_name) {
                // Create new manager for this environment path
                let manager =
                    LmdbStoreManager::new(self.dir_path.join(&cfg.name), cfg.max_env_size);
                state.managers.insert(man_name.clone(), manager);
            }
        }

        // Get the manager and create the database/store
        let mut state = self.managers_state.lock().await;
        let manager = state.managers.get_mut(&man_name).ok_or_else(|| {
            heed::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("LMDB_Dir_Store_Manager: Manager not found for {}", man_name),
            ))
        })?;

        manager.store(database_name).await
    }

    async fn shutdown(&mut self) -> Result<(), heed::Error> {
        let mut state = self.managers_state.lock().await;
        for manager in state.managers.values_mut() {
            let _ = manager.shutdown().await;
        }
        state.managers.clear();
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
                managers: HashMap::new(),
            })),
        }
    }
}

// Implement Drop
// This ensures all LMDB environments are closed when the manager is dropped
impl Drop for LmdbDirStoreManager {
    fn drop(&mut self) {
        // Use try_lock() for synchronous access in Drop
        if let Ok(mut state) = self.managers_state.try_lock() {
            // Clear all managers, triggering their Drop implementations
            // which will close LMDB environments
            state.managers.clear();
        }
    }
}
