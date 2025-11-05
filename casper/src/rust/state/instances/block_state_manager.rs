// See casper/src/main/scala/coop/rchain/casper/state/instances/BlockStateManagerImpl.scala

use block_storage::rust::dag::block_dag_key_value_storage::BlockDagKeyValueStorage;

use crate::rust::errors::CasperError;

#[derive(Clone)]
pub struct BlockStateManager {
    block_dag_storage: BlockDagKeyValueStorage,
}

impl BlockStateManager {
    pub fn new(block_dag_storage: BlockDagKeyValueStorage) -> Self {
        Self { block_dag_storage }
    }

    /// Checks if the block state is empty by checking if the DAG has any blocks
    pub fn is_empty(&self) -> Result<bool, CasperError> {
        let dag = self.block_dag_storage.get_representation();
        let first_hash = dag.topo_sort(0, Some(1))?;
        Ok(first_hash.is_empty())
    }
}
