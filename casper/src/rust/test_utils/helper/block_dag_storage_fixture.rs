// See casper/src/test/scala/coop/rchain/casper/helper/BlockDagStorageFixture.scala
// Moved from casper/tests/helper/block_dag_storage_fixture.rs to casper/src/rust/test_utils/helper/block_dag_storage_fixture.rs
// All imports fixed for library crate context

use std::future::Future;

use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use block_storage::rust::test::indexed_block_dag_storage::IndexedBlockDagStorage;
use crate::rust::util::rholang::runtime_manager::RuntimeManager;

use crate::rust::test_utils::util::genesis_builder::GenesisContext;
use crate::rust::test_utils::util::rholang::resources;

pub async fn with_genesis<F, Fut, R>(context: GenesisContext, f: F) -> R
where
    F: FnOnce(KeyValueBlockStore, IndexedBlockDagStorage, RuntimeManager) -> Fut,
    Fut: Future<Output = R>,
{
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

    // Note: init_logger removed - logging should be initialized by test framework
    let (blocks, indexed_dag) = create().await;
    f(blocks, indexed_dag).await
}


