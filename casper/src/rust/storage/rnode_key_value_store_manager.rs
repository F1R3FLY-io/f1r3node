// See casper/src/main/scala/coop/rchain/casper/storage/RNodeKeyValueStoreManager.scala

use std::path::PathBuf;

use rspace_plus_plus::rspace::shared::{
    key_value_store_manager::KeyValueStoreManager,
    lmdb_dir_store_manager::{Db, LmdbDirStoreManager, LmdbEnvConfig, GB, TB},
};

pub fn new_key_value_store_manager(
    dir_path: PathBuf,
    legacy_rspace_paths: Option<bool>,
) -> impl KeyValueStoreManager {
    LmdbDirStoreManager::new(
        dir_path,
        rnode_db_mapping(legacy_rspace_paths).into_iter().collect(),
    )
}

// Config name is used as a sub-folder for LMDB files

// RSpace
fn rspace_history_env_config() -> LmdbEnvConfig {
    LmdbEnvConfig::new("rspace/history".to_string(), 1 * TB)
}

fn rspace_cold_env_config() -> LmdbEnvConfig {
    LmdbEnvConfig::new("rspace/cold".to_string(), 1 * TB)
}

// RSpace evaluator
fn eval_history_env_config() -> LmdbEnvConfig {
    LmdbEnvConfig::new("eval/history".to_string(), 1 * TB)
}

fn eval_cold_env_config() -> LmdbEnvConfig {
    LmdbEnvConfig::new("eval/cold".to_string(), 1 * TB)
}

// Blocks
fn block_storage_env_config() -> LmdbEnvConfig {
    LmdbEnvConfig::new("blockstorage".to_string(), 1 * TB)
}

fn dag_storage_env_config() -> LmdbEnvConfig {
    LmdbEnvConfig::new("dagstorage".to_string(), 100 * GB)
}

fn deploy_storage_env_config() -> LmdbEnvConfig {
    LmdbEnvConfig::new("deploystorage".to_string(), 1 * GB)
}

// Temporary storage / cache
fn casper_buffer_env_config() -> LmdbEnvConfig {
    LmdbEnvConfig::new("casperbuffer".to_string(), 1 * GB)
}

fn reporting_env_config() -> LmdbEnvConfig {
    LmdbEnvConfig::new("reporting".to_string(), 10 * TB)
}

fn transaction_env_config() -> LmdbEnvConfig {
    LmdbEnvConfig::new("transaction".to_string(), 1 * GB)
}

// Legacy RSpace paths
fn legacy_env_config(dir: &str) -> LmdbEnvConfig {
    LmdbEnvConfig::new(format!("{}/{}", "rspace/casper/v2", dir), 1 * TB)
}

// Database name to store instance name mapping (sub-folder for LMDB store)
// - keys with the same instance will be in one LMDB file (environment)
pub fn rnode_db_mapping(legacy_rspace_paths: Option<bool>) -> Vec<(Db, LmdbEnvConfig)> {
    let legacy_rspace_paths = legacy_rspace_paths.unwrap_or(false);

    let mut mappings = vec![
        // Block storage
        (
            Db::new("blocks".to_string(), None),
            block_storage_env_config(),
        ),
        // Block metadata storage
        (
            Db::new("blocks-approved".to_string(), None),
            dag_storage_env_config(),
        ),
        (
            Db::new("block-metadata".to_string(), None),
            dag_storage_env_config(),
        ),
        (
            Db::new("equivocation-tracker".to_string(), None),
            dag_storage_env_config(),
        ),
        (
            Db::new("latest-messages".to_string(), None),
            dag_storage_env_config(),
        ),
        (
            Db::new("invalid-blocks".to_string(), None),
            dag_storage_env_config(),
        ),
        (
            Db::new("deploy-index".to_string(), None),
            dag_storage_env_config(),
        ),
        (
            Db::new("last-finalized-block".to_string(), None),
            dag_storage_env_config(),
        ),
        // Runtime mergeable store (cache of mergeable channels for block-merge)
        (
            Db::new("mergeable-channel-cache".to_string(), None),
            dag_storage_env_config(),
        ),
        // Deploy storage
        (
            Db::new("deploy_storage".to_string(), None),
            deploy_storage_env_config(),
        ),
        // Reporting (trace) cache
        (
            Db::new("reporting-cache".to_string(), None),
            reporting_env_config(),
        ),
        // CasperBuffer
        (
            Db::new("parents-map".to_string(), None),
            casper_buffer_env_config(),
        ),
        // Rholang evaluator store
        (
            Db::new("eval-history".to_string(), None),
            eval_history_env_config(),
        ),
        (
            Db::new("eval-roots".to_string(), None),
            eval_history_env_config(),
        ),
        (
            Db::new("eval-cold".to_string(), None),
            eval_cold_env_config(),
        ),
        // Transaction store
        (
            Db::new("transaction".to_string(), None),
            transaction_env_config(),
        ),
    ];

    // RSpace
    if !legacy_rspace_paths {
        // History and roots maps are part of the same LMDB file (environment)
        mappings.push((
            Db::new("rspace-history".to_string(), None),
            rspace_history_env_config(),
        ));
        mappings.push((
            Db::new("rspace-roots".to_string(), None),
            rspace_history_env_config(),
        ));
        mappings.push((
            Db::new("rspace-cold".to_string(), None),
            rspace_cold_env_config(),
        ));
    } else {
        // Legacy config has the same database name for all maps
        mappings.push((
            Db::new("rspace-history".to_string(), Some("db".to_string())),
            legacy_env_config("history"),
        ));
        mappings.push((
            Db::new("rspace-roots".to_string(), Some("db".to_string())),
            legacy_env_config("roots"),
        ));
        mappings.push((
            Db::new("rspace-cold".to_string(), Some("db".to_string())),
            legacy_env_config("cold"),
        ));
        mappings.push((
            Db::new("rspace-channels".to_string(), Some("db".to_string())),
            legacy_env_config("channels"),
        ));
    }

    mappings
}
