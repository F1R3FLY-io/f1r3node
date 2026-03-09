// See casper/src/test/scala/coop/rchain/casper/helper/BlockDagStorageFixture.scala
//
// ## Race Condition Fix with Shared LMDB
//
// Unlike Scala tests where each test gets its own separate LMDB database, Rust tests use a 
// SHARED_LMDB_ENV (see resources.rs) for performance optimization. This means all 300+ tests 
// write to the same LMDB database concurrently.
//
// ### The Problem:
// Each test creates its own BlockDagKeyValueStorage with its own global_lock. These locks
// only serialize operations WITHIN a single test, but do NOT prevent race conditions BETWEEN tests:
//
// ```
// Test A: insert(block_A) → unlock → get_representation() → reads snapshot
// Test B: insert(block_B) → unlock → (writes to same LMDB!)
// Test A: validate() → tries to lookup block_B → CRASH: "DAG storage is missing hash"
// ```
//
// ### The Solution:
// Use SHARED_LMDB_LOCK - a global static Mutex that ALL tests must acquire before accessing
// the shared LMDB. This ensures tests run SEQUENTIALLY (one at a time) when using shared storage.
//
// ### Trade-off:
// - ✅ Fixes race condition completely
// - ⚠️ Tests run slower (sequential vs parallel)
// - Still faster than creating 300+ separate LMDB databases (Scala approach)

use std::future::Future;

use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use block_storage::rust::test::indexed_block_dag_storage::IndexedBlockDagStorage;
use casper::rust::util::rholang::runtime_manager::RuntimeManager;

use crate::init_logger;
use crate::util::genesis_builder::GenesisContext;
use crate::util::rholang::resources;

pub async fn with_genesis<F, Fut, R>(context: GenesisContext, f: F) -> R
where
    F: FnOnce(KeyValueBlockStore, IndexedBlockDagStorage, RuntimeManager) -> Fut,
    Fut: Future<Output = R>,
{
    // Acquire global lock for shared LMDB to ensure test isolation.
    // This prevents concurrent tests from interfering with each other when using shared LMDB.
    // The lock is held for the entire test duration to guarantee consistency.
    let _lock_guard = resources::SHARED_LMDB_LOCK.lock().unwrap();
    
    async fn create(
        genesis_context: &GenesisContext,
    ) -> (KeyValueBlockStore, IndexedBlockDagStorage, RuntimeManager) {
        let scope_id = genesis_context.rspace_scope_id.clone();
        let mut kvm = resources::mk_test_rnode_store_manager_shared(scope_id);

        let blocks = KeyValueBlockStore::create_from_kvm(&mut *kvm)
            .await
            .unwrap();
        blocks
            .put(
                genesis_context.genesis_block.block_hash.clone(),
                &genesis_context.genesis_block,
            )
            .expect("Failed to put genesis block");

        let dag = resources::block_dag_storage_from_dyn(&mut *kvm)
            .await
            .unwrap();
        dag.insert(&genesis_context.genesis_block, false, true)
            .expect("Failed to insert genesis block into DAG");

        let indexed_dag = IndexedBlockDagStorage::new(dag);

        let (runtime, _history_repo) =
            resources::mk_runtime_manager_with_history_at(&mut *kvm).await;

        (blocks, indexed_dag, runtime)
    }

    let (blocks, indexed_dag, runtime) = create(&context).await;
    f(blocks, indexed_dag, runtime).await
}

pub async fn with_storage<F, Fut, R>(f: F) -> R
where
    F: FnOnce(KeyValueBlockStore, IndexedBlockDagStorage) -> Fut,
    Fut: Future<Output = R>,
{
    // Acquire global lock for shared LMDB to ensure test isolation.
    // Same reason as with_genesis - prevents race conditions with shared LMDB.
    let _lock_guard = resources::SHARED_LMDB_LOCK.lock().unwrap();
    
    async fn create() -> (KeyValueBlockStore, IndexedBlockDagStorage) {
        let scope_id = resources::generate_scope_id();
        let mut kvm = resources::mk_test_rnode_store_manager_shared(scope_id);
        let blocks = KeyValueBlockStore::create_from_kvm(&mut *kvm)
            .await
            .unwrap();
        let dag = resources::block_dag_storage_from_dyn(&mut *kvm)
            .await
            .unwrap();
        let indexed_dag = IndexedBlockDagStorage::new(dag);

        (blocks, indexed_dag)
    }

    init_logger();

    let (blocks, indexed_dag) = create().await;
    f(blocks, indexed_dag).await
}
