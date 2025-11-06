// See block-storage/src/main/scala/coop/rchain/blockstorage/casperbuffer/CasperBufferKeyValueStorage.scala
// See block-storage/src/test/scala/coop/rchain/blockstorage/casperbuffer/CasperBufferStorageTest.scala

use dashmap::DashSet;
use std::collections::HashSet;

use models::rust::block_hash::BlockHashSerde;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use shared::rust::store::{
    key_value_store::KvStoreError, key_value_typed_store::KeyValueTypedStore,
    key_value_typed_store_impl::KeyValueTypedStoreImpl,
};

use crate::rust::util::doubly_linked_dag_operations::BlockDependencyDag;

/**
 * @param parentsStore - persistent map {hash -> parents set}
 * @param blockDependencyDag - in-memory dependency DAG, recreated from parentsStore on node startup
 */
#[derive(Clone)]
pub struct CasperBufferKeyValueStorage {
    parents_store: KeyValueTypedStoreImpl<BlockHashSerde, HashSet<BlockHashSerde>>,
    block_dependency_dag: BlockDependencyDag,
}

impl CasperBufferKeyValueStorage {
    pub async fn new_from_kvm(kvm: &mut impl KeyValueStoreManager) -> Result<Self, KvStoreError> {
        let parents_store_kv = kvm.store("parents-map".to_string()).await?;
        let parents_store: KeyValueTypedStoreImpl<BlockHashSerde, HashSet<BlockHashSerde>> =
            KeyValueTypedStoreImpl::new(parents_store_kv);

        Self::new_from_kv_store(parents_store).await
    }

    pub async fn new_from_kv_store(
        kv_store: KeyValueTypedStoreImpl<BlockHashSerde, HashSet<BlockHashSerde>>,
    ) -> Result<Self, KvStoreError> {
        let in_mem_store = {
            let parents_map = kv_store.to_map()?;
            parents_map
                .into_iter()
                .fold(BlockDependencyDag::empty(), |bdd, (key, parents)| {
                    parents.iter().cloned().fold(bdd, |mut bdd, p| {
                        bdd.add(p, key.clone());
                        bdd
                    })
                })
        };

        Ok(Self {
            parents_store: kv_store,
            block_dependency_dag: in_mem_store,
        })
    }

    pub fn add_relation(
        &self,
        parent: BlockHashSerde,
        child: BlockHashSerde,
    ) -> Result<(), KvStoreError> {
        let mut parents = self.parents_store.get_one(&child)?.unwrap_or_default();
        parents.insert(parent.clone());
        self.parents_store.put_one(child.clone(), parents)?;
        self.block_dependency_dag.add(parent, child);
        Ok(())
    }

    pub fn put_pendant(&self, block: BlockHashSerde) -> Result<(), KvStoreError> {
        let temp_block = BlockHashSerde(prost::bytes::Bytes::from_static(b"tempblock"));
        self.add_relation(temp_block.clone(), block)?;
        self.remove(temp_block)?;
        Ok(())
    }

    pub fn remove(&self, hash: BlockHashSerde) -> Result<(), KvStoreError> {
        let (hashes_affected, hashes_removed) = self.block_dependency_dag.remove(hash)?;

        // Process each affected hash
        let mut changes = Vec::new();
        for h in &hashes_affected {
            let mut parents = HashSet::new();
            if let Some(dash_parents) = self
                .block_dependency_dag
                .child_to_parent_adjacency_list
                .get(h)
            {
                for parent in dash_parents.iter() {
                    parents.insert(parent.clone());
                }
            }
            changes.push((h.clone(), parents));
        }

        self.parents_store.put(changes)?;
        let hashes_to_delete: Vec<BlockHashSerde> = hashes_removed.into_iter().collect();
        self.parents_store.delete(hashes_to_delete)?;

        Ok(())
    }

    pub fn get_parents(&self, block_hash: &BlockHashSerde) -> Option<DashSet<BlockHashSerde>> {
        self.block_dependency_dag
            .child_to_parent_adjacency_list
            .get(&block_hash)
            .map(|kv_ref| kv_ref.value().clone())
    }

    pub fn get_children(&self, block_hash: &BlockHashSerde) -> Option<DashSet<BlockHashSerde>> {
        self.block_dependency_dag
            .parent_to_child_adjacency_list
            .get(&block_hash)
            .map(|kv_ref| kv_ref.value().clone())
    }

    pub fn get_pendants(&self) -> DashSet<BlockHashSerde> {
        self.block_dependency_dag.dependency_free.clone()
    }

    // Block is considered to be in CasperBuffer when there is a records about its parents
    pub fn contains(&self, block_hash: &BlockHashSerde) -> bool {
        self.block_dependency_dag
            .child_to_parent_adjacency_list
            .get(block_hash)
            .is_some()
    }

    pub fn to_doubly_linked_dag(&self) -> BlockDependencyDag {
        self.block_dependency_dag.clone()
    }

    pub fn size(&self) -> usize {
        self.block_dependency_dag
            .child_to_parent_adjacency_list
            .len()
    }

    pub fn is_pendant(&self, block_hash: &BlockHashSerde) -> bool {
        self.block_dependency_dag
            .dependency_free
            .contains(block_hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::rust::block_hash::BlockHashSerde;
    use prost::bytes::Bytes;
    use rspace_plus_plus::rspace::shared::in_mem_store_manager::InMemoryStoreManager;

    fn create_block_hash(data: &[u8]) -> BlockHashSerde {
        BlockHashSerde(Bytes::copy_from_slice(data))
    }

    #[tokio::test]
    async fn casper_buffer_storage_should_work() -> Result<(), KvStoreError> {
        let mut kvm = InMemoryStoreManager::new();
        let store = kvm.store("parents-map".to_string()).await?;
        let typed_store = KeyValueTypedStoreImpl::new(store);

        let a = create_block_hash(b"A");
        let b = create_block_hash(b"B");
        let c = create_block_hash(b"C");
        let d = create_block_hash(b"D");

        typed_store.put_one(c.clone(), HashSet::from([d.clone()]))?;

        let mut casper_buffer = CasperBufferKeyValueStorage::new_from_kv_store(typed_store).await?;

        // CasperBufferStorage be able to restore state on startup
        let c_parents = casper_buffer.get_parents(&c);
        assert!(c_parents.is_some());
        assert!(c_parents.unwrap().contains(&d));

        let d_children = casper_buffer.get_children(&d);
        assert!(d_children.is_some());
        assert!(d_children.unwrap().contains(&c));

        // Add relation should change parents set and children set
        casper_buffer.add_relation(a.clone(), b.clone())?;

        let b_parents = casper_buffer.get_parents(&b);
        assert!(b_parents.is_some());
        assert!(b_parents.unwrap().contains(&a));

        let a_children = casper_buffer.get_children(&a);
        assert!(a_children.is_some());
        assert!(a_children.unwrap().contains(&b));

        // Block that has no parents should be pendant
        casper_buffer.add_relation(a.clone(), b.clone())?;
        assert!(casper_buffer.is_pendant(&a));

        // When removed hash A is the last parent for hash B, key B should be removed from parents store
        let h1 = casper_buffer.parents_store.get_one(&b)?;
        assert!(h1.is_some());
        assert!(h1.unwrap().contains(&a));
        casper_buffer.remove(a.clone())?;
        let h2 = casper_buffer.parents_store.get_one(&b)?;
        assert!(h2.is_none());

        // When removed hash A is the last parent for hash B, B should be pendant
        assert!(casper_buffer.is_pendant(&b));

        Ok(())
    }
}
