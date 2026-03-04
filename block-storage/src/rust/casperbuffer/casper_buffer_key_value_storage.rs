// See block-storage/src/main/scala/coop/rchain/blockstorage/casperbuffer/CasperBufferKeyValueStorage.scala
// See block-storage/src/test/scala/coop/rchain/blockstorage/casperbuffer/CasperBufferStorageTest.scala

use dashmap::DashSet;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{SystemTime, UNIX_EPOCH};

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
    block_dependency_dag: Arc<BlockDependencyDag>,
    first_seen_ms: Arc<dashmap::DashMap<BlockHashSerde, u64>>,
    last_prune_ms: Arc<AtomicU64>,
    state_lock: Arc<RwLock<()>>,
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
                    parents.iter().cloned().fold(bdd, |bdd, p| {
                        bdd.add(p, key.clone());
                        bdd
                    })
                })
        };

        Ok(Self {
            parents_store: kv_store,
            block_dependency_dag: Arc::new(in_mem_store),
            first_seen_ms: Arc::new(dashmap::DashMap::new()),
            last_prune_ms: Arc::new(AtomicU64::new(0)),
            state_lock: Arc::new(RwLock::new(())),
        })
    }

    fn read_guard(&self) -> RwLockReadGuard<'_, ()> {
        self.state_lock.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write_guard(&self) -> RwLockWriteGuard<'_, ()> {
        self.state_lock.write().unwrap_or_else(|e| e.into_inner())
    }

    fn add_relation_unlocked(
        &self,
        parent: BlockHashSerde,
        child: BlockHashSerde,
    ) -> Result<(), KvStoreError> {
        self.track_hash_first_seen(&parent);
        self.track_hash_first_seen(&child);
        let mut parents = self.parents_store.get_one(&child)?.unwrap_or_default();
        parents.insert(parent.clone());
        self.parents_store.put_one(child.clone(), parents)?;
        self.block_dependency_dag.add(parent, child);
        Ok(())
    }

    fn remove_unlocked(&self, hash: BlockHashSerde) -> Result<(), KvStoreError> {
        let (hashes_affected, hashes_removed, orphaned_hashes) =
            self.block_dependency_dag.remove(hash.clone())?;
        self.first_seen_ms.remove(&hash);

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
        let mut hashes_to_delete_with_node = hashes_to_delete;
        hashes_to_delete_with_node.push(hash);
        hashes_to_delete_with_node.extend(orphaned_hashes);
        for h in &hashes_to_delete_with_node {
            self.first_seen_ms.remove(h);
        }
        self.parents_store.delete(hashes_to_delete_with_node)?;

        Ok(())
    }

    pub fn add_relation(
        &self,
        parent: BlockHashSerde,
        child: BlockHashSerde,
    ) -> Result<(), KvStoreError> {
        let _guard = self.write_guard();
        self.add_relation_unlocked(parent, child)
    }

    pub fn put_pendant(&self, block: BlockHashSerde) -> Result<(), KvStoreError> {
        let _guard = self.write_guard();
        let temp_block = BlockHashSerde(prost::bytes::Bytes::from_static(b"tempblock"));
        self.add_relation_unlocked(temp_block.clone(), block)?;
        self.remove_unlocked(temp_block)?;
        Ok(())
    }

    pub fn remove(&self, hash: BlockHashSerde) -> Result<(), KvStoreError> {
        let _guard = self.write_guard();
        self.remove_unlocked(hash)
    }

    pub fn get_parents(&self, block_hash: &BlockHashSerde) -> Option<DashSet<BlockHashSerde>> {
        let _guard = self.read_guard();
        self.block_dependency_dag
            .child_to_parent_adjacency_list
            .get(&block_hash)
            .map(|kv_ref| kv_ref.value().clone())
    }

    pub fn get_children(&self, block_hash: &BlockHashSerde) -> Option<DashSet<BlockHashSerde>> {
        let _guard = self.read_guard();
        self.block_dependency_dag
            .parent_to_child_adjacency_list
            .get(&block_hash)
            .map(|kv_ref| kv_ref.value().clone())
    }

    pub fn get_pendants(&self) -> DashSet<BlockHashSerde> {
        let _guard = self.read_guard();
        self.block_dependency_dag.dependency_free.clone()
    }

    // Block is considered to be in CasperBuffer when there is a records about its parents
    pub fn contains(&self, block_hash: &BlockHashSerde) -> bool {
        let _guard = self.read_guard();
        self.block_dependency_dag
            .child_to_parent_adjacency_list
            .get(block_hash)
            .is_some()
    }

    pub fn to_doubly_linked_dag(&self) -> BlockDependencyDag {
        let _guard = self.read_guard();
        self.block_dependency_dag.as_ref().clone()
    }

    pub fn requested_as_dependency(&self, block_hash: &BlockHashSerde) -> bool {
        let _guard = self.read_guard();
        self.block_dependency_dag
            .parent_to_child_adjacency_list
            .contains_key(block_hash)
    }

    pub fn size(&self) -> usize {
        let _guard = self.read_guard();
        self.block_dependency_dag
            .child_to_parent_adjacency_list
            .len()
    }

    pub fn approx_node_count(&self) -> usize {
        let _guard = self.read_guard();
        self.block_dependency_dag
            .child_to_parent_adjacency_list
            .len()
            + self
                .block_dependency_dag
                .parent_to_child_adjacency_list
                .len()
    }

    pub fn is_pendant(&self, block_hash: &BlockHashSerde) -> bool {
        let _guard = self.read_guard();
        self.block_dependency_dag
            .dependency_free
            .contains(block_hash)
    }

    fn dependency_free_nodes_with_age_ms(&self, now_ms: u64) -> Vec<(u64, BlockHashSerde)> {
        let mut nodes = Vec::new();

        for hash in self.block_dependency_dag.dependency_free.iter() {
            let seen_ms = self
                .first_seen_ms
                .get(hash.key())
                .map(|seen| *seen)
                .unwrap_or(now_ms);
            nodes.push((now_ms.saturating_sub(seen_ms), hash.key().clone()));
        }

        nodes
    }

    pub fn enforce_limits(
        &self,
        max_approx_nodes: usize,
        stale_ttl_ms: u64,
        max_prune_batch: usize,
        prune_interval_ms: u64,
    ) -> Result<(usize, usize), KvStoreError> {
        let now = Self::now_millis();
        let last_prune = self.last_prune_ms.load(Ordering::Relaxed);
        if now.saturating_sub(last_prune) < prune_interval_ms {
            return Ok((0, 0));
        }
        self.last_prune_ms.store(now, Ordering::Relaxed);

        let mut stale_candidates: Vec<(u64, BlockHashSerde)> = self
            .dependency_free_nodes_with_age_ms(now)
            .into_iter()
            .filter(|(age_ms, _)| *age_ms >= stale_ttl_ms)
            .collect();
        stale_candidates.sort_by(|a, b| b.0.cmp(&a.0));

        let mut stale_pruned = 0usize;
        for (_, hash) in stale_candidates.into_iter().take(max_prune_batch) {
            match self.remove(hash) {
                Ok(_) => stale_pruned += 1,
                Err(KvStoreError::InvalidArgument(_)) => {}
                Err(e) => return Err(e),
            }
        }

        let mut overflow_pruned = 0usize;
        let mut approx_nodes = self.approx_node_count();
        let mut attempts = 0usize;
        while overflow_pruned < max_prune_batch
            && attempts < max_prune_batch
            && approx_nodes > max_approx_nodes
        {
            let mut oldest_nodes: Vec<(u64, BlockHashSerde)> = self
                .dependency_free_nodes_with_age_ms(now)
                .into_iter()
                .collect();
            oldest_nodes.sort_by(|a, b| a.0.cmp(&b.0));

            let Some((_, hash)) = oldest_nodes.into_iter().next() else {
                break;
            };

            let removed = self.remove(hash);
            attempts += 1;

            match removed {
                Ok(_) => {
                    overflow_pruned += 1;
                    approx_nodes = self.approx_node_count();
                }
                Err(KvStoreError::InvalidArgument(_)) => {}
                Err(e) => return Err(e),
            }
        }

        Ok((stale_pruned, overflow_pruned))
    }

    fn track_hash_first_seen(&self, hash: &BlockHashSerde) {
        self.first_seen_ms
            .entry(hash.clone())
            .or_insert_with(Self::now_millis);
    }

    fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
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

        let casper_buffer = CasperBufferKeyValueStorage::new_from_kv_store(typed_store).await?;

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

    #[tokio::test]
    async fn casper_buffer_put_pendant_stays_dependency_free() -> Result<(), KvStoreError> {
        let mut kvm = InMemoryStoreManager::new();
        let store = kvm.store("parents-map".to_string()).await?;
        let typed_store = KeyValueTypedStoreImpl::new(store);
        let casper_buffer = CasperBufferKeyValueStorage::new_from_kv_store(typed_store).await?;

        let block = create_block_hash(b"dependent_block");
        let temp_block = BlockHashSerde(prost::bytes::Bytes::from_static(b"tempblock"));
        casper_buffer.put_pendant(block.clone())?;

        assert!(casper_buffer.contains(&block) == false);
        assert!(casper_buffer.is_pendant(&block));
        assert!(!casper_buffer.contains(&temp_block));
        assert!(!casper_buffer.is_pendant(&temp_block));
        assert!(casper_buffer.get_parents(&block).is_none());
        assert!(casper_buffer.get_children(&temp_block).is_none());

        let pendants = casper_buffer.get_pendants();
        assert!(pendants.contains(&block));
        Ok(())
    }

    #[tokio::test]
    async fn casper_buffer_remove_repairs_stale_parent_links() -> Result<(), KvStoreError> {
        let mut kvm = InMemoryStoreManager::new();
        let store = kvm.store("parents-map".to_string()).await?;
        let typed_store = KeyValueTypedStoreImpl::new(store);
        let casper_buffer = CasperBufferKeyValueStorage::new_from_kv_store(typed_store).await?;

        let block = create_block_hash(b"orphan");
        let valid_parent = create_block_hash(b"valid-parent");
        let stale_parent = create_block_hash(b"stale-parent");

        casper_buffer.add_relation(valid_parent.clone(), block.clone())?;

        {
            let parents = casper_buffer
                .block_dependency_dag
                .child_to_parent_adjacency_list
                .get_mut(&block)
                .expect("expected block to have parent links");
            parents.insert(stale_parent.clone());
        }

        let before = casper_buffer
            .get_parents(&block)
            .expect("expected pendant-parent linkage");
        assert!(before.contains(&valid_parent));
        assert!(before.contains(&stale_parent));
        assert!(casper_buffer.first_seen_ms.get(&valid_parent).is_some());

        casper_buffer.remove(block.clone())?;

        assert!(!casper_buffer.contains(&block));
        assert!(casper_buffer.get_parents(&block).is_none());
        assert!(casper_buffer.parents_store.get_one(&block)?.is_none());
        assert!(!casper_buffer.requested_as_dependency(&valid_parent));
        assert!(casper_buffer.first_seen_ms.get(&valid_parent).is_none());

        Ok(())
    }
}
