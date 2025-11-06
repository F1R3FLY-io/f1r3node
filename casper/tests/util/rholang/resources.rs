// See casper/src/test/scala/coop/rchain/casper/util/rholang/Resources.scala

use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use block_storage::rust::dag::block_metadata_store::BlockMetadataStore;
use casper::rust::casper::{CasperShardConf, CasperSnapshot, OnChainCasperState};
use casper::rust::errors::CasperError;
use dashmap::{DashMap, DashSet};
use models::rust::block_hash::BlockHash;
use models::rust::casper::protocol::casper_message::BlockMessage;
use prost::bytes::Bytes;
use rspace_plus_plus::rspace::shared::in_mem_key_value_store::InMemoryKeyValueStore;
use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use tempfile::Builder;

use casper::rust::{
    genesis::genesis::Genesis, storage::rnode_key_value_store_manager::rnode_db_mapping,
    util::rholang::runtime_manager::RuntimeManager,
};
use models::rhoapi::Par;
use rholang::rust::interpreter::{
    rho_runtime::RhoHistoryRepository, test_utils::resources::mk_temp_dir,
};
use rspace_plus_plus::rspace::shared::{
    key_value_store_manager::KeyValueStoreManager,
    lmdb_dir_store_manager::{Db, LmdbDirStoreManager, LmdbEnvConfig, MB},
};

use crate::init_logger;
use crate::util::genesis_builder::GenesisBuilder;
use crate::util::genesis_builder::GenesisContext;

static CACHED_GENESIS: OnceLock<Arc<Mutex<Option<GenesisContext>>>> = OnceLock::new();

pub async fn genesis_context() -> Result<GenesisContext, CasperError> {
    let genesis_arc = CACHED_GENESIS
        .get_or_init(|| Arc::new(Mutex::new(None)))
        .clone();

    let mut genesis_guard = genesis_arc.lock().unwrap();

    if genesis_guard.is_none() {
        let mut genesis_builder = GenesisBuilder::new();
        let new_genesis = genesis_builder.build_genesis_with_parameters(None).await?;
        *genesis_guard = Some(new_genesis);
    }

    Ok(genesis_guard.as_ref().unwrap().clone())
}

pub async fn with_runtime_manager<F, Fut, R>(f: F) -> Result<R, CasperError>
where
    F: FnOnce(RuntimeManager, GenesisContext, BlockMessage) -> Fut,
    Fut: Future<Output = R>,
{
    init_logger();
    let genesis_context = genesis_context().await?;
    let genesis_block = genesis_context.genesis_block.clone();

    let storage_dir = copy_storage(genesis_context.storage_directory.clone());
    let kvm = mk_test_rnode_store_manager(storage_dir);
    let runtime_manager = mk_runtime_manager_at(kvm, None).await;

    Ok(f(runtime_manager, genesis_context, genesis_block).await)
}

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

pub async fn mk_runtime_manager(prefix: &str, mergeable_tag_name: Option<Par>) -> RuntimeManager {
    let dir_path = mk_temp_dir(prefix);
    let kvm = mk_test_rnode_store_manager(dir_path);

    mk_runtime_manager_at(kvm, mergeable_tag_name).await
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

/// Creates a temporary directory that will be persisted (not automatically cleaned up)
pub fn create_persisted_temp_dir(prefix: &str) -> PathBuf {
    let temp_dir = Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("Failed to create temp dir");

    // Convert to PathBuf which will persist even after TempDir is dropped
    let path = temp_dir.keep();
    path
}

/// Copy a template storage directory to a new temporary directory that is persisted
/// If the source directory doesn't exist, it creates an empty directory instead
pub fn copy_storage(storage_template_path: PathBuf) -> PathBuf {
    // Create a persistent temporary directory instead of using with_temp_dir
    let temp_path_buf = create_persisted_temp_dir("casper-test-");
    copy_dir(storage_template_path, temp_path_buf.clone()).expect("Failed to copy directory");
    temp_path_buf
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
