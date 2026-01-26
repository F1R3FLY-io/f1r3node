// See casper/src/test/scala/coop/rchain/casper/util/rholang/Resources.scala

use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use block_storage::rust::dag::block_metadata_store::BlockMetadataStore;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use casper::rust::casper::{CasperShardConf, CasperSnapshot, OnChainCasperState};
use casper::rust::errors::CasperError;
use dashmap::{DashMap, DashSet};
use lazy_static::lazy_static;
use models::rust::block_hash::BlockHash;
use models::rust::casper::protocol::casper_message::BlockMessage;
use prost::bytes::Bytes;
use rspace_plus_plus::rspace::shared::in_mem_key_value_store::InMemoryKeyValueStore;
use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use tempfile::{Builder, TempDir};
use uuid::Uuid;

use casper::rust::{
    genesis::genesis::Genesis, storage::rnode_key_value_store_manager::rnode_db_mapping,
    util::rholang::runtime_manager::RuntimeManager,
};
use models::rhoapi::Par;
use rholang::rust::interpreter::rho_runtime::RhoHistoryRepository;
use rspace_plus_plus::rspace::shared::{
    key_value_store_manager::KeyValueStoreManager,
    lmdb_dir_store_manager::{Db, LmdbDirStoreManager, LmdbEnvConfig, MB},
};

use crate::init_logger;
use crate::util::genesis_builder::GenesisBuilder;
use crate::util::genesis_builder::GenesisContext;

static CACHED_GENESIS: OnceLock<Arc<Mutex<Option<GenesisContext>>>> = OnceLock::new();

// Shared LMDB environment for all tests.
//
// This single environment is shared across all tests to avoid exhausting OS resources.
// Test isolation is achieved through scoped database names (UUID prefixes) rather than
// separate environments. This allows hundreds of tests to run efficiently without
// hitting file descriptor or LMDB environment limits.
//
// Resource Management:
// - Single LMDB environment instead of 300+ separate environments
// - Automatic cleanup when TempDir is dropped (at program exit)
// - Global lock ensures test isolation when using shared LMDB
lazy_static! {
    static ref SHARED_LMDB_ENV: (PathBuf, TempDir) = {
        let temp_dir = Builder::new()
            .prefix("casper-shared-lmdb-")
            .tempdir()
            .expect("Failed to create shared LMDB temp dir");
        let path = temp_dir.path().to_path_buf();
        (path, temp_dir)
    };
    
    /// Global lock to ensure test isolation when using shared LMDB.
    ///
    /// ## Why is this needed?
    ///
    /// Unlike Scala tests where each test creates its own LMDB database, Rust tests share
    /// a single LMDB environment (SHARED_LMDB_ENV) for performance. This creates a race condition:
    ///
    /// 1. Each test creates its own BlockDagKeyValueStorage with its own global_lock
    /// 2. These per-test locks only serialize operations WITHIN a single test
    /// 3. Multiple tests can write to the same shared LMDB concurrently
    /// 4. Result: Test A inserts block_A, Test B inserts block_B concurrently,
    ///    Test A tries to read block_B → CRASH: "DAG storage is missing hash"
    ///
    /// ## Solution:
    ///
    /// SHARED_LMDB_LOCK is a single global Mutex that ALL tests must acquire before
    /// accessing shared LMDB (via with_genesis/with_storage helpers). This ensures
    /// tests run sequentially when using shared storage, preventing race conditions.
    ///
    /// ## Trade-off:
    ///
    /// - Sequential execution is slower than parallel
    /// - But still faster than creating 300+ separate LMDB databases (Scala approach)
    /// - And guaranteed correctness is more important than speed
    pub static ref SHARED_LMDB_LOCK: Mutex<()> = Mutex::new(());
}

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

    // Use the same scope_id as genesis to access all genesis data including RSpace history
    // This ensures tests can reset to the genesis state root hash
    let mut kvm = mk_test_rnode_store_manager_from_genesis(&genesis_context);
    // Use create_with_history to ensure tests can reset to genesis state root hash
    let (runtime_manager, _history_repo) = mk_runtime_manager_with_history_at(&mut *kvm).await;

    Ok(f(runtime_manager, genesis_context, genesis_block).await)
}

pub fn mk_test_rnode_store_manager_with_scope(
    dir_path: PathBuf,
    scope_id: Option<String>,
) -> impl KeyValueStoreManager {
    let limit_size = 500 * MB;

    let db_mappings: Vec<(Db, LmdbEnvConfig)> = rnode_db_mapping(None)
        .into_iter()
        .map(|(db, mut conf)| {
            let new_conf = if conf.max_env_size > limit_size {
                conf.max_env_size = limit_size;
                conf
            } else {
                conf
            }
            .with_max_dbs(10_000);

            // If scope_id is provided, create a scoped database name using name_override
            // This ensures test isolation while keeping the original ID for lookup
            let scoped_db = if let Some(ref scope) = scope_id {
                let scoped_name = format!("{}-{}", scope, db.id());
                Db::new(db.id().to_string(), Some(scoped_name))
            } else {
                db
            };

            (scoped_db, new_conf)
        })
        .collect();

    LmdbDirStoreManager::new(dir_path, db_mappings.into_iter().collect())
}

/// Creates a test store manager using a shared LMDB environment.
///
/// This is the recommended approach for tests to avoid exhausting OS resources
/// (file descriptors, LMDB environments). All tests share a single LMDB environment,
/// with test isolation achieved through scoped database names (UUID prefixes).
///
/// # Best Practices
/// - Always use this function instead of `mk_test_rnode_store_manager()` for tests
/// - Each test gets a unique scope_id via `generate_scope_id()`
/// - The shared environment is automatically cleaned up when tests complete
/// - Works efficiently with parallel test execution (test-threads=4-8 recommended)
pub fn mk_test_rnode_store_manager_shared(scope_id: String) -> Box<dyn KeyValueStoreManager> {
    let (shared_path, _temp_dir) = &*SHARED_LMDB_ENV;
    // Create the manager with scoped database names in the mapping
    // This ensures isolation at the LMDB level while keeping lookup by original name
    Box::new(mk_test_rnode_store_manager_with_scope(
        shared_path.clone(),
        Some(scope_id),
    ))
}

/// Generates a unique scope ID for test isolation.
///
/// Each test should use a unique scope ID to ensure database isolation
/// within the shared LMDB environment.
pub fn generate_scope_id() -> String {
    Uuid::new_v4().to_string()
}

/// Returns the path to the shared LMDB environment.
///
/// This is useful for logging/debugging purposes when tests need a path
/// to reference, but actual LMDB storage is in the shared environment.
pub fn get_shared_lmdb_path() -> PathBuf {
    let (shared_path, _temp_dir) = &*SHARED_LMDB_ENV;
    shared_path.clone()
}

/// Creates a test store manager with dual scoping for RSpace and other stores.
///
/// This function creates a manager where:
/// - RSpace stores (rspace-history, rspace-roots, rspace-cold) use `rspace_scope`
/// - All other stores (blocks, DAG, deploys, etc.) use `node_scope`
///
/// This allows multiple nodes within a test to share RSpace state (see each other's
/// committed roots) while maintaining isolation for block and DAG stores.
pub fn mk_test_rnode_store_manager_with_dual_scope(
    node_scope: String,
    rspace_scope: String,
) -> impl KeyValueStoreManager {
    let (shared_path, _temp_dir) = &*SHARED_LMDB_ENV;
    let limit_size = 100 * MB;

    let db_mappings: Vec<(Db, LmdbEnvConfig)> = rnode_db_mapping(None)
        .into_iter()
        .map(|(db, mut conf)| {
            let new_conf = if conf.max_env_size > limit_size {
                conf.max_env_size = limit_size;
                conf
            } else {
                conf
            }
            .with_max_dbs(10_000);

            // Determine which scope to use based on database type
            let scope_to_use = if db.id().starts_with("rspace-") {
                &rspace_scope
            } else {
                &node_scope
            };

            // Create scoped database name
            let scoped_name = format!("{}-{}", scope_to_use, db.id());
            let scoped_db = Db::new(db.id().to_string(), Some(scoped_name));

            (scoped_db, new_conf)
        })
        .collect();

    LmdbDirStoreManager::new(shared_path.clone(), db_mappings.into_iter().collect())
}

/// Creates a test store manager with genesis data and shared RSpace scope.
///
/// This function:
/// 1. Generates a new unique scope for this node's blocks/DAG stores
/// 2. Uses the shared RSpace scope from genesis for RSpace stores
/// 3. Copies genesis block and DAG data to the new node's stores
///
/// This ensures all nodes in the same test can see each other's RSpace state
/// (committed roots) while maintaining isolation for block and DAG data.
pub async fn mk_test_rnode_store_manager_with_shared_rspace(
    genesis_context: &GenesisContext,
    shared_rspace_scope: &str,
) -> Result<Box<dyn KeyValueStoreManager>, CasperError> {
    let new_node_scope = generate_scope_id();
    let mut new_kvm = Box::new(mk_test_rnode_store_manager_with_dual_scope(
        new_node_scope,
        shared_rspace_scope.to_string(),
    ));

    // Copy genesis block to the new scope's block store
    let new_block_store = KeyValueBlockStore::create_from_kvm(&mut *new_kvm).await?;
    new_block_store.put(
        genesis_context.genesis_block.block_hash.clone(),
        &genesis_context.genesis_block,
    )?;

    // Copy genesis DAG metadata to the new scope's DAG storage
    let new_dag_storage = block_dag_storage_from_dyn(&mut *new_kvm)
        .await
        .map_err(|e| CasperError::RuntimeError(format!("Failed to create DAG storage: {:?}", e)))?;
    new_dag_storage.insert(&genesis_context.genesis_block, false, true)?;

    Ok(new_kvm)
}

/// Creates a test store manager using the genesis rspace_scope_id directly.
///
/// This function reuses the same RSpace scope where genesis was created,
/// giving direct access to the genesis RSpace history and roots without copying.
///
/// CRITICAL: Uses rspace_scope_id, not scope_id! Genesis RSpace data is stored
/// in the rspace_scope_id, which ensures tests can access the genesis state and
/// its committed roots in the RootsStore.
///
/// Note: Multiple tests using this will share the same RSpace state.
/// Use `mk_test_rnode_store_manager_with_genesis` for complete test isolation.
pub fn mk_test_rnode_store_manager_from_genesis(
    genesis_context: &GenesisContext,
) -> Box<dyn KeyValueStoreManager> {
    mk_test_rnode_store_manager_shared(genesis_context.rspace_scope_id.clone())
}

type MergeableStore = shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl<
    shared::rust::ByteVector,
    Vec<rholang::rust::interpreter::merging::rholang_merging_logic::DeployMergeableData>,
>;

pub async fn mergeable_store_from_dyn(
    kvm: &mut dyn KeyValueStoreManager,
) -> Result<MergeableStore, shared::rust::store::key_value_store::KvStoreError> {
    use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;

    let store = kvm
        .store("mergeable-channel-cache".to_string())
        .await
        .map_err(|e| {
            shared::rust::store::key_value_store::KvStoreError::IoError(format!(
                "Failed to get mergeable store: {:?}",
                e
            ))
        })?;
    Ok(KeyValueTypedStoreImpl::new(store))
}

pub async fn block_dag_storage_from_dyn(
    kvm: &mut dyn KeyValueStoreManager,
) -> Result<
    block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage,
    shared::rust::store::key_value_store::KvStoreError,
> {
    use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
    use block_storage::rust::dag::block_metadata_store::BlockMetadataStore;
    use block_storage::rust::dag::equivocation_tracker_store::EquivocationTrackerStore;
    use models::rust::block_hash::BlockHashSerde;
    use models::rust::block_metadata::BlockMetadata;
    use models::rust::equivocation_record::SequenceNumber;
    use models::rust::validator::ValidatorSerde;
    use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
    use std::collections::BTreeSet;
    use std::sync::{Arc, RwLock};

    let block_metadata_kv_store = kvm.store("block-metadata".to_string()).await.map_err(|e| {
        shared::rust::store::key_value_store::KvStoreError::IoError(format!(
            "Failed to get block-metadata store: {:?}",
            e
        ))
    })?;
    let block_metadata_db: KeyValueTypedStoreImpl<BlockHashSerde, BlockMetadata> =
        KeyValueTypedStoreImpl::new(block_metadata_kv_store);
    let block_metadata_store = BlockMetadataStore::new(block_metadata_db);

    let equivocation_tracker_kv_store = kvm
        .store("equivocation-tracker".to_string())
        .await
        .map_err(|e| {
            shared::rust::store::key_value_store::KvStoreError::IoError(format!(
                "Failed to get equivocation-tracker store: {:?}",
                e
            ))
        })?;
    let equivocation_tracker_db: KeyValueTypedStoreImpl<
        (ValidatorSerde, SequenceNumber),
        BTreeSet<BlockHashSerde>,
    > = KeyValueTypedStoreImpl::new(equivocation_tracker_kv_store);
    let equivocation_tracker_store = EquivocationTrackerStore::new(equivocation_tracker_db);

    let latest_messages_kv_store = kvm
        .store("latest-messages".to_string())
        .await
        .map_err(|e| {
            shared::rust::store::key_value_store::KvStoreError::IoError(format!(
                "Failed to get latest-messages store: {:?}",
                e
            ))
        })?;
    let latest_messages_db: KeyValueTypedStoreImpl<ValidatorSerde, BlockHashSerde> =
        KeyValueTypedStoreImpl::new(latest_messages_kv_store);

    let invalid_blocks_kv_store = kvm.store("invalid-blocks".to_string()).await.map_err(|e| {
        shared::rust::store::key_value_store::KvStoreError::IoError(format!(
            "Failed to get invalid-blocks store: {:?}",
            e
        ))
    })?;
    let invalid_blocks_db: KeyValueTypedStoreImpl<BlockHashSerde, BlockMetadata> =
        KeyValueTypedStoreImpl::new(invalid_blocks_kv_store);

    let deploy_index_kv_store = kvm.store("deploy-index".to_string()).await.map_err(|e| {
        shared::rust::store::key_value_store::KvStoreError::IoError(format!(
            "Failed to get deploy-index store: {:?}",
            e
        ))
    })?;
    let deploy_index_db: KeyValueTypedStoreImpl<
        block_storage::rust::dag::block_dag_key_value_storage::DeployId,
        BlockHashSerde,
    > = KeyValueTypedStoreImpl::new(deploy_index_kv_store);

    Ok(BlockDagKeyValueStorage {
        global_lock: Arc::new(Mutex::new(())),
        block_metadata_index: Arc::new(RwLock::new(block_metadata_store)),
        deploy_index: Arc::new(RwLock::new(deploy_index_db)),
        invalid_blocks_index: invalid_blocks_db,
        equivocation_tracker_index: equivocation_tracker_store,
        latest_messages_index: latest_messages_db,
    })
}

pub async fn key_value_deploy_storage_from_dyn(
    kvm: &mut dyn KeyValueStoreManager,
) -> Result<
    block_storage::rust::deploy::key_value_deploy_storage::KeyValueDeployStorage,
    shared::rust::store::key_value_store::KvStoreError,
> {
    use block_storage::rust::deploy::key_value_deploy_storage::KeyValueDeployStorage;
    use crypto::rust::signatures::signed::Signed;
    use models::rust::casper::protocol::casper_message::DeployData;
    use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
    use shared::rust::ByteString;

    let deploy_storage_kv_store = kvm.store("deploy_storage".to_string()).await.map_err(|e| {
        shared::rust::store::key_value_store::KvStoreError::IoError(format!(
            "Failed to get deploy_storage store: {:?}",
            e
        ))
    })?;
    let deploy_storage_db: KeyValueTypedStoreImpl<ByteString, Signed<DeployData>> =
        KeyValueTypedStoreImpl::new(deploy_storage_kv_store);

    Ok(KeyValueDeployStorage {
        store: deploy_storage_db,
    })
}

pub async fn casper_buffer_storage_from_dyn(
    kvm: &mut dyn KeyValueStoreManager,
) -> Result<
    block_storage::rust::casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage,
    shared::rust::store::key_value_store::KvStoreError,
> {
    use block_storage::rust::casperbuffer::casper_buffer_key_value_storage::CasperBufferKeyValueStorage;
    use models::rust::block_hash::BlockHashSerde;
    use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;
    use std::collections::HashSet;

    let parents_store_kv = kvm.store("parents-map".to_string()).await.map_err(|e| {
        shared::rust::store::key_value_store::KvStoreError::IoError(format!(
            "Failed to get parents-map store: {:?}",
            e
        ))
    })?;
    let parents_store: KeyValueTypedStoreImpl<BlockHashSerde, HashSet<BlockHashSerde>> =
        KeyValueTypedStoreImpl::new(parents_store_kv);

    CasperBufferKeyValueStorage::new_from_kv_store(parents_store)
        .await
        .map_err(|e| {
            shared::rust::store::key_value_store::KvStoreError::IoError(format!(
                "Failed to create CasperBufferKeyValueStorage: {:?}",
                e
            ))
        })
}

pub async fn mk_runtime_manager(_prefix: &str, mergeable_tag_name: Option<Par>) -> RuntimeManager {
    let scope_id = generate_scope_id();
    let mut kvm = mk_test_rnode_store_manager_shared(scope_id);

    mk_runtime_manager_at(&mut *kvm, mergeable_tag_name).await
}

pub async fn mk_runtime_manager_at(
    kvm: &mut dyn KeyValueStoreManager,
    mergeable_tag_name: Option<Par>,
) -> RuntimeManager {
    let mergeable_tag_name =
        mergeable_tag_name.unwrap_or(Genesis::non_negative_mergeable_tag_name());

    let r_store = kvm.r_space_stores().await.unwrap();
    let m_store = mergeable_store_from_dyn(kvm).await.unwrap();
    RuntimeManager::create_with_store(r_store, m_store, mergeable_tag_name)
}

pub async fn mk_runtime_manager_with_history_at(
    kvm: &mut dyn KeyValueStoreManager,
) -> (RuntimeManager, RhoHistoryRepository) {
    let r_store = kvm.r_space_stores().await.unwrap();
    let m_store = mergeable_store_from_dyn(kvm).await.unwrap();
    let (rt_manager, history_repo) = RuntimeManager::create_with_history(
        r_store,
        m_store,
        Genesis::non_negative_mergeable_tag_name(),
    );
    (rt_manager, history_repo)
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
