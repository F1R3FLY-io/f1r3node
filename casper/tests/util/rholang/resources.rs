// See casper/src/test/scala/coop/rchain/casper/util/rholang/Resources.scala

use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use block_storage::rust::dag::block_metadata_store::BlockMetadataStore;
use casper::rust::casper::{CasperShardConf, CasperSnapshot, OnChainCasperState};
use dashmap::{DashMap, DashSet};
use models::rust::block_hash::BlockHash;
use prost::bytes::Bytes;
use rspace_plus_plus::rspace::shared::in_mem_key_value_store::InMemoryKeyValueStore;
use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tempfile::{Builder, TempDir};

use casper::rust::{
    genesis::genesis::Genesis, storage::rnode_key_value_store_manager::rnode_db_mapping,
    util::rholang::runtime_manager::RuntimeManager,
};
use models::rhoapi::Par;
use rholang::rust::interpreter::{
    rho_runtime::RhoHistoryRepository, test_utils::resources::mk_temp_dir_guard,
};
use rspace_plus_plus::rspace::shared::{
    key_value_store_manager::KeyValueStoreManager,
    lmdb_dir_store_manager::{Db, LmdbDirStoreManager, LmdbEnvConfig, MB},
};

pub fn mk_test_rnode_store_manager(dir_path: PathBuf) -> impl KeyValueStoreManager {
    // Limit maximum environment (file) size for LMDB in tests
    let limit_size = 100 * MB;

    let db_mappings: Vec<(Db, LmdbEnvConfig)> = rnode_db_mapping(None)
        .into_iter()
        .map(|(db, mut conf)| {
            let new_conf = if conf.max_env_size > limit_size {
                conf.max_env_size = limit_size;
                conf
            } else {
                conf
            };

            (db, new_conf)
        })
        .collect();

    LmdbDirStoreManager::new(dir_path, db_mappings.into_iter().collect())
}

/// Creates a RuntimeManager with a temporary directory that has automatic cleanup.
/// Returns (TempDir guard, RuntimeManager) - keep the guard alive to prevent cleanup.
pub async fn mk_runtime_manager(
    prefix: &str,
    mergeable_tag_name: Option<Par>,
) -> (TempDir, RuntimeManager) {
    let temp_dir = mk_temp_dir_guard(prefix);
    let dir_path = temp_dir.path().to_path_buf();
    let kvm = mk_test_rnode_store_manager(dir_path);
    let runtime_manager = mk_runtime_manager_at(kvm, mergeable_tag_name).await;
    (temp_dir, runtime_manager)
}

pub async fn mk_runtime_manager_at(
    mut kvm: impl KeyValueStoreManager,
    mergeable_tag_name: Option<Par>,
) -> RuntimeManager {
    let mergeable_tag_name =
        mergeable_tag_name.unwrap_or(Genesis::non_negative_mergeable_tag_name());

    let r_store = kvm.r_space_stores().await.unwrap();
    let m_store = RuntimeManager::mergeable_store(&mut kvm).await.unwrap();
    RuntimeManager::create_with_store(r_store, m_store, mergeable_tag_name)
}

pub async fn mk_runtime_manager_with_history_at(
    mut kvm: impl KeyValueStoreManager,
) -> (RuntimeManager, RhoHistoryRepository) {
    let r_store = kvm.r_space_stores().await.unwrap();
    let m_store = RuntimeManager::mergeable_store(&mut kvm).await.unwrap();
    let (rt_manager, history_repo) = RuntimeManager::create_with_history(
        r_store,
        m_store,
        Genesis::non_negative_mergeable_tag_name(),
    );
    (rt_manager, history_repo)
}

/// Creates a managed temporary directory that will be automatically removed when the TempDir is dropped
pub fn with_temp_dir<F, R>(prefix: &str, f: F) -> R
where
    F: FnOnce(&Path) -> R,
{
    let temp_dir = Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("Failed to create temp dir");

    // Run the function with the temp_dir path
    let result = f(temp_dir.path());

    // TempDir will be dropped here and automatically cleaned up
    // unless we've decided to manually persist it
    result
}

/// Copy a template storage directory to a new temporary directory with automatic cleanup.
/// Returns (PathBuf, TempDir) - the TempDir guard must be kept alive to prevent cleanup.
/// When the TempDir is dropped, the directory is automatically deleted.
pub fn copy_storage(storage_template_path: PathBuf) -> (PathBuf, TempDir) {
    let temp_dir = Builder::new()
        .prefix("casper-test-")
        .tempdir()
        .expect("Failed to create temp dir");
    let path = temp_dir.path().to_path_buf();
    copy_dir(&storage_template_path, &path).expect("Failed to copy directory");
    (path, temp_dir)
}

fn copy_dir<P: AsRef<Path>, Q: AsRef<Path>>(src: P, dest: Q) -> io::Result<()> {
    let src = src.as_ref();
    let dest = dest.as_ref();

    if !dest.exists() {
        fs::create_dir_all(dest)?;
    }

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let entry_path = entry.path();
        let dest_path = dest.join(entry_path.strip_prefix(src).unwrap());

        if entry_path.is_dir() {
            fs::create_dir_all(&dest_path)?;
            copy_dir(&entry_path, &dest_path)?;
        } else {
            fs::copy(&entry_path, &dest_path)?;
        }
    }

    Ok(())
}

fn new_key_value_dag_representation() -> KeyValueDagRepresentation {
    let block_metadata_store = KeyValueTypedStoreImpl::new(Arc::new(InMemoryKeyValueStore::new()));

    KeyValueDagRepresentation {
        dag_set: Arc::new(DashSet::new()),
        latest_messages_map: Arc::new(DashMap::new()),
        child_map: Arc::new(DashMap::new()),
        height_map: Arc::new(RwLock::new(BTreeMap::new())),
        invalid_blocks_set: Arc::new(DashSet::new()),
        last_finalized_block_hash: BlockHash::new(),
        finalized_blocks_set: Arc::new(DashSet::new()),
        block_metadata_index: Arc::new(RwLock::new(BlockMetadataStore::new(block_metadata_store))),
        deploy_index: Arc::new(RwLock::new(KeyValueTypedStoreImpl::new(Arc::new(
            InMemoryKeyValueStore::new(),
        )))),
    }
}

pub fn mk_dummy_casper_snapshot() -> CasperSnapshot {
    let dag = new_key_value_dag_representation();

    CasperSnapshot {
        dag,
        last_finalized_block: Bytes::new(),
        lca: Bytes::new(),
        tips: Vec::new(),
        parents: Vec::new(),
        justifications: DashSet::new(),
        invalid_blocks: HashMap::new(),
        deploys_in_scope: DashSet::new(),
        max_block_num: 0,
        max_seq_nums: DashMap::new(),
        on_chain_state: OnChainCasperState {
            shard_conf: CasperShardConf::new(),
            bonds_map: HashMap::new(),
            active_validators: Vec::new(),
        },
    }
}
