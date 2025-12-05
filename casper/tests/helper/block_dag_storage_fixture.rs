// See casper/src/test/scala/coop/rchain/casper/helper/BlockDagStorageFixture.scala

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
    async fn create(genesis_context: &GenesisContext) -> (KeyValueBlockStore, IndexedBlockDagStorage, RuntimeManager) {
        // Use mk_test_rnode_store_manager_with_genesis to get a new scope with genesis data copied
        // This ensures test isolation while having access to genesis block and DAG data
        let mut kvm = resources::mk_test_rnode_store_manager_with_genesis(genesis_context).await
            .expect("Failed to create store manager with genesis");
        let blocks = KeyValueBlockStore::create_from_kvm(&mut *kvm).await.unwrap();
        
        let dag = resources::block_dag_storage_from_dyn(&mut *kvm).await.unwrap();
        
        let indexed_dag = IndexedBlockDagStorage::new(dag);
        // Use create_with_history to ensure tests can reset to genesis state root hash
        // Note: RSpace history will be empty in new scope, but tests can still create blocks
        // For tests that need to reset to genesis state, they should use with_runtime_manager instead
        let (runtime, _history_repo) = resources::mk_runtime_manager_with_history_at(&mut *kvm).await;

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
    async fn create() -> (KeyValueBlockStore, IndexedBlockDagStorage) {
        let scope_id = resources::generate_scope_id();
        let mut kvm = resources::mk_test_rnode_store_manager_shared(scope_id);
        let blocks = KeyValueBlockStore::create_from_kvm(&mut *kvm).await.unwrap();
        let dag = resources::block_dag_storage_from_dyn(&mut *kvm).await.unwrap();
        let indexed_dag = IndexedBlockDagStorage::new(dag);

        (blocks, indexed_dag)
    }

    init_logger();

    let (blocks, indexed_dag) = create().await;
    f(blocks, indexed_dag).await
}
