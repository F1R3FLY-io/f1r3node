// See casper/src/test/scala/coop/rchain/casper/helper/BlockDagStorageFixture.scala

use std::future::Future;
use std::path::PathBuf;

use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;
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
    async fn create(dir: PathBuf) -> (KeyValueBlockStore, IndexedBlockDagStorage, RuntimeManager) {
        let mut kvm = resources::mk_test_rnode_store_manager(dir);
        let blocks = KeyValueBlockStore::create_from_kvm(&mut kvm).await.unwrap();
        let dag = BlockDagKeyValueStorage::new(&mut kvm).await.unwrap();
        let indexed_dag = IndexedBlockDagStorage::new(dag);
        let runtime = resources::mk_runtime_manager_at(kvm, None).await;

        (blocks, indexed_dag, runtime)
    }

    let storage_dir = resources::copy_storage(context.storage_directory);
    let (blocks, indexed_dag, runtime) = create(storage_dir).await;
    f(blocks, indexed_dag, runtime).await
}

pub async fn with_storage<F, Fut, R>(f: F) -> R
where
    F: FnOnce(KeyValueBlockStore, IndexedBlockDagStorage) -> Fut,
    Fut: Future<Output = R>,
{
    async fn create(dir: PathBuf) -> (KeyValueBlockStore, IndexedBlockDagStorage) {
        let mut kvm = resources::mk_test_rnode_store_manager(dir);
        let blocks = KeyValueBlockStore::create_from_kvm(&mut kvm).await.unwrap();
        let dag = BlockDagKeyValueStorage::new(&mut kvm).await.unwrap();
        let indexed_dag = IndexedBlockDagStorage::new(dag);

        (blocks, indexed_dag)
    }

    init_logger();

    let temp_dir = rholang::rust::interpreter::test_utils::resources::mk_temp_dir(
        "casper-block-dag-storage-test-",
    );
    let (blocks, indexed_dag) = create(temp_dir).await;
    f(blocks, indexed_dag).await
}
