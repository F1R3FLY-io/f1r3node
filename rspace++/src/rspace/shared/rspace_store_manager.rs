use crate::rspace::rspace::RSpaceStore;
use heed::EnvOpenOptions;
use lazy_static::lazy_static;
use shared::rust::store::lmdb_key_value_store::LmdbKeyValueStore;
use std::collections::{HashMap, HashSet};
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::rspace::shared::{
    key_value_store_manager::KeyValueStoreManager, lmdb_dir_store_manager::LmdbDirStoreManager,
};

use super::lmdb_dir_store_manager::{Db, LmdbEnvConfig};

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/RholangCLI.scala
pub fn mk_rspace_store_manager(dir_path: PathBuf, map_size: usize) -> impl KeyValueStoreManager {
    let rspace_history_env_config = LmdbEnvConfig::new("history".to_owned(), map_size);
    let rspace_cold_env_config = LmdbEnvConfig::new("cold".to_owned(), map_size);
    let channel_env_config = LmdbEnvConfig::new("channels".to_owned(), map_size);

    let mut db_mapping = HashMap::new();
    db_mapping
        .insert(Db::new("rspace-history".to_string(), None), rspace_history_env_config.clone());
    db_mapping.insert(Db::new("rspace-roots".to_string(), None), rspace_history_env_config);
    db_mapping.insert(Db::new("rspace-cold".to_string(), None), rspace_cold_env_config);
    db_mapping.insert(Db::new("rspace-channels".to_string(), None), channel_env_config);

    LmdbDirStoreManager::new(dir_path, db_mapping)
}

lazy_static! {
    static ref LMDB_DIR_STATE_MANAGER: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
}

pub fn get_or_create_rspace_store(
    lmdb_path: &str,
    map_size: usize,
) -> Result<RSpaceStore, heed::Error> {
    if Path::new(lmdb_path).exists() {
        tracing::debug!("RSpace++ storage path {} already exists (reopening)", lmdb_path);

        // In Scala (and Rust rnode_db_mapping), RSpace envs are in subfolders: rspace/history and rspace/cold
        let history_env_path = format!("{}/rspace/history", lmdb_path);
        let cold_env_path = format!("{}/rspace/cold", lmdb_path);

        let history_store = open_lmdb_store(&history_env_path, "rspace-history")?;
        let roots_store = open_lmdb_store(&history_env_path, "rspace-roots")?;
        let cold_store = open_lmdb_store(&cold_env_path, "rspace-cold")?;

        let rspace_store = RSpaceStore {
            history: Arc::new(history_store),
            roots: Arc::new(roots_store),
            cold: Arc::new(cold_store),
        };

        Ok(rspace_store)
    } else {
        tracing::debug!("RSpace++ storage path {} does not exist, creating new", lmdb_path);
        create_dir_all(lmdb_path).expect("Failed to create RSpace++ storage directory");

        // Create subfolders consistent with rnode_db_mapping
        let history_env_path = format!("{}/rspace/history", lmdb_path);
        let cold_env_path = format!("{}/rspace/cold", lmdb_path);
        create_dir_all(&history_env_path).expect("Failed to create RSpace++ history directory");
        create_dir_all(&cold_env_path).expect("Failed to create RSpace++ cold directory");

        let history_store = create_lmdb_store(&history_env_path, "rspace-history", map_size)?;
        let roots_store = create_lmdb_store(&history_env_path, "rspace-roots", map_size)?;
        let cold_store = create_lmdb_store(&cold_env_path, "rspace-cold", map_size)?;

        let rspace_store = RSpaceStore {
            history: Arc::new(history_store),
            roots: Arc::new(roots_store),
            cold: Arc::new(cold_store),
        };

        Ok(rspace_store)
    }
}

pub fn close_rspace_store(rspace_store: RSpaceStore) {
    drop(rspace_store);
}

fn create_lmdb_store(
    lmdb_path: &str,
    db_name: &str,
    max_env_size: usize,
) -> Result<LmdbKeyValueStore, heed::Error> {
    let mut env_builder = EnvOpenOptions::new();
    env_builder.map_size(max_env_size);
    env_builder.max_dbs(20);
    env_builder.max_readers(2048);

    let env = env_builder.open(&lmdb_path)?;
    let db = env.create_database(Some(db_name))?;

    Ok(LmdbKeyValueStore::new(env, db))
}

fn open_lmdb_store(lmdb_path: &str, db_name: &str) -> Result<LmdbKeyValueStore, heed::Error> {
    let mut env_builder = EnvOpenOptions::new();
    env_builder.max_dbs(20);

    let env = env_builder.open(lmdb_path)?;
    let db = env.open_database(Some(db_name))?;
    match db {
        Some(open_db) => Ok(LmdbKeyValueStore {
            env: env.into(),
            db: Arc::new(Mutex::new(open_db)),
        }),
        None => panic!("\nFailed to open database: {}", db_name),
    }
}
