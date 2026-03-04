// See block-storage/src/main/scala/coop/rchain/blockstorage/util/DoublyLinkedDagOperations.scala

use dashmap::{DashMap, DashSet};
use shared::rust::store::key_value_store::KvStoreError;
use std::collections::HashSet;

use models::rust::block_hash::BlockHashSerde;

#[derive(Debug, Clone)]
pub struct BlockDependencyDag {
    pub parent_to_child_adjacency_list: DashMap<BlockHashSerde, DashSet<BlockHashSerde>>,
    pub child_to_parent_adjacency_list: DashMap<BlockHashSerde, DashSet<BlockHashSerde>>,
    pub dependency_free: DashSet<BlockHashSerde>,
}

impl BlockDependencyDag {
    pub fn empty() -> Self {
        BlockDependencyDag {
            parent_to_child_adjacency_list: DashMap::new(),
            child_to_parent_adjacency_list: DashMap::new(),
            dependency_free: DashSet::new(),
        }
    }

    pub fn add(&self, parent: BlockHashSerde, child: BlockHashSerde) {
        if let Some(set) = self.parent_to_child_adjacency_list.get_mut(&parent) {
            set.insert(child.clone());
        } else {
            let set = DashSet::new();
            set.insert(child.clone());
            self.parent_to_child_adjacency_list
                .insert(parent.clone(), set);
        }

        if let Some(set) = self.child_to_parent_adjacency_list.get_mut(&child) {
            set.insert(parent.clone());
        } else {
            let set = DashSet::new();
            set.insert(parent.clone());
            self.child_to_parent_adjacency_list
                .insert(child.clone(), set);
        }

        if !self.child_to_parent_adjacency_list.contains_key(&parent) {
            self.dependency_free.insert(parent);
        }

        self.dependency_free.remove(&child);
    }

    pub fn remove(
        &self,
        element: BlockHashSerde,
    ) -> Result<
        (
            HashSet<BlockHashSerde>,
            HashSet<BlockHashSerde>,
            HashSet<BlockHashSerde>,
        ),
        KvStoreError,
    > {
        let mut orphaned_parents = HashSet::new();

        let parent_links: Vec<BlockHashSerde> =
            match self.child_to_parent_adjacency_list.get(&element) {
                Some(parents) => parents.iter().map(|parent| parent.key().clone()).collect(),
                None => Vec::new(),
            };

        // Remove incoming links from all direct parents so this node does not
        // remain as a dangling child after removal.
        for parent in parent_links {
            let mut remove_parent_entry = false;
            if let Some(children) = self.parent_to_child_adjacency_list.get_mut(&parent) {
                children.remove(&element);
                if children.is_empty() {
                    remove_parent_entry = true;
                }
            }

            if remove_parent_entry {
                self.parent_to_child_adjacency_list.remove(&parent);
                orphaned_parents.insert(parent.clone());
                self.dependency_free.remove(&parent);
            }
        }

        // Get children first and release the lock
        let children: Vec<BlockHashSerde> = match self.parent_to_child_adjacency_list.get(&element)
        {
            Some(children) => {
                let mut vec = Vec::new();
                for child in children.value().iter() {
                    vec.push(child.clone());
                }
                vec
            }
            None => Vec::new(),
        };

        self.child_to_parent_adjacency_list.remove(&element);

        let mut new_dependency_free = HashSet::new();
        let mut children_affected = HashSet::new();
        let mut children_removed = HashSet::new();

        // Process each child independently
        for child in children {
            // Get parents and release the lock
            let parents: HashSet<BlockHashSerde> =
                match self.child_to_parent_adjacency_list.get(&child) {
                    Some(parents) => {
                        let mut set = HashSet::new();
                        for parent in parents.value().iter() {
                            set.insert(parent.clone());
                        }
                        set
                    }
                    None => {
                        // A stale forward edge can exist in parent_to_child without a reverse edge.
                        // Treat it as a removable orphan relation and continue.
                        HashSet::new()
                    }
                };

            // Create new parents set without the element
            let updated_parents: HashSet<_> =
                parents.into_iter().filter(|p| !p.eq(&element)).collect();

            if updated_parents.is_empty() {
                self.child_to_parent_adjacency_list.remove(&child);
                new_dependency_free.insert(child.clone());
                children_removed.insert(child);
            } else {
                let dash_set = DashSet::new();
                for parent in updated_parents {
                    dash_set.insert(parent);
                }
                self.child_to_parent_adjacency_list
                    .insert(child.clone(), dash_set);
                children_affected.insert(child);
            }
        }

        // Update the DAG state
        self.parent_to_child_adjacency_list.remove(&element);
        for item in new_dependency_free {
            self.dependency_free.insert(item);
        }
        self.dependency_free.remove(&element);

        for parent in &orphaned_parents {
            if self.child_to_parent_adjacency_list.contains_key(&parent) {
                continue;
            }

            self.dependency_free.remove(&parent);
        }

        Ok((children_affected, children_removed, orphaned_parents))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::bytes::Bytes;

    fn create_block_hash(value: &[u8]) -> BlockHashSerde {
        models::rust::block_hash::BlockHashSerde(Bytes::from(value.to_vec()))
    }

    #[test]
    fn test_empty_dag() {
        let dag = BlockDependencyDag::empty();
        assert!(dag.parent_to_child_adjacency_list.is_empty());
        assert!(dag.child_to_parent_adjacency_list.is_empty());
        assert!(dag.dependency_free.is_empty());
    }

    #[test]
    fn test_add_single_edge() {
        let dag = BlockDependencyDag::empty();
        let parent = create_block_hash(b"parent");
        let child = create_block_hash(b"child");

        dag.add(parent.clone(), child.clone());

        // Check parent -> child mapping
        assert!(dag.parent_to_child_adjacency_list.contains_key(&parent));
        assert!(dag
            .parent_to_child_adjacency_list
            .get(&parent)
            .unwrap()
            .contains(&child));

        // Check child -> parent mapping
        assert!(dag.child_to_parent_adjacency_list.contains_key(&child));
        assert!(dag
            .child_to_parent_adjacency_list
            .get(&child)
            .unwrap()
            .contains(&parent));

        // Check dependency free set
        assert!(dag.dependency_free.contains(&parent));
        assert!(!dag.dependency_free.contains(&child));
    }

    #[test]
    fn test_add_multiple_children() {
        let dag = BlockDependencyDag::empty();
        let parent = create_block_hash(b"parent");
        let child1 = create_block_hash(b"child1");
        let child2 = create_block_hash(b"child2");

        dag.add(parent.clone(), child1.clone());
        dag.add(parent.clone(), child2.clone());

        // Check parent -> children mapping
        let children = dag.parent_to_child_adjacency_list.get(&parent).unwrap();
        assert!(children.contains(&child1));
        assert!(children.contains(&child2));

        // Check children -> parent mapping
        assert!(dag
            .child_to_parent_adjacency_list
            .get(&child1)
            .unwrap()
            .contains(&parent));
        assert!(dag
            .child_to_parent_adjacency_list
            .get(&child2)
            .unwrap()
            .contains(&parent));

        // Check dependency free set
        assert!(dag.dependency_free.contains(&parent));
        assert!(!dag.dependency_free.contains(&child1));
        assert!(!dag.dependency_free.contains(&child2));
    }

    #[test]
    fn test_remove_leaf_node() {
        let dag = BlockDependencyDag::empty();
        let parent = create_block_hash(b"parent");
        let child = create_block_hash(b"child");

        dag.add(parent.clone(), child.clone());
        let (affected, removed, _orphaned_parents) = dag.remove(parent.clone()).unwrap();

        // Check that parent is removed from all structures
        assert!(!dag.parent_to_child_adjacency_list.contains_key(&parent));
        assert!(!dag.child_to_parent_adjacency_list.contains_key(&parent));
        assert!(!dag.dependency_free.contains(&parent));

        // Check that child is now dependency-free
        assert!(dag.dependency_free.contains(&child));

        // Check returned sets
        assert!(affected.is_empty());
        assert!(removed.contains(&child));
    }

    #[test]
    fn test_remove_child_cleans_orphan_parent_from_dependency_free() {
        let dag = BlockDependencyDag::empty();
        let parent = create_block_hash(b"orphan-parent");
        let child = create_block_hash(b"child");

        dag.add(parent.clone(), child.clone());
        assert!(dag.dependency_free.contains(&parent));

        let (affected, removed, orphaned_parents) = dag.remove(child.clone()).unwrap();

        assert!(!dag.dependency_free.contains(&parent));
        assert!(dag.parent_to_child_adjacency_list.get(&parent).is_none());
        assert!(!dag.child_to_parent_adjacency_list.contains_key(&parent));
        assert!(orphaned_parents.contains(&parent));
        assert!(affected.is_empty());
        assert!(removed.is_empty());
    }

    #[test]
    fn test_remove_node_with_multiple_children() {
        let dag = BlockDependencyDag::empty();
        let parent = create_block_hash(b"parent");
        let child1 = create_block_hash(b"child1");
        let child2 = create_block_hash(b"child2");

        dag.add(parent.clone(), child1.clone());
        dag.add(parent.clone(), child2.clone());
        let (affected, removed, _orphaned_parents) = dag.remove(parent.clone()).unwrap();

        // Check that parent is removed
        assert!(!dag.parent_to_child_adjacency_list.contains_key(&parent));
        assert!(!dag.child_to_parent_adjacency_list.contains_key(&parent));
        assert!(!dag.dependency_free.contains(&parent));

        // Check that children are now dependency-free
        assert!(dag.dependency_free.contains(&child1));
        assert!(dag.dependency_free.contains(&child2));

        // Check returned sets
        assert!(affected.is_empty());
        assert!(removed.contains(&child1));
        assert!(removed.contains(&child2));
    }

    #[test]
    fn test_remove_node_with_remaining_parents() {
        let dag = BlockDependencyDag::empty();
        let parent1 = create_block_hash(b"parent1");
        let parent2 = create_block_hash(b"parent2");
        let child = create_block_hash(b"child");

        dag.add(parent1.clone(), child.clone());
        dag.add(parent2.clone(), child.clone());
        let (affected, removed, _orphaned_parents) = dag.remove(parent1.clone()).unwrap();

        // Check that parent1 is removed
        assert!(!dag.parent_to_child_adjacency_list.contains_key(&parent1));
        assert!(!dag.child_to_parent_adjacency_list.contains_key(&parent1));
        assert!(!dag.dependency_free.contains(&parent1));

        // Check that child still has parent2
        assert!(dag
            .child_to_parent_adjacency_list
            .get(&child)
            .unwrap()
            .contains(&parent2));
        assert!(!dag.dependency_free.contains(&child));

        // Check returned sets
        assert!(affected.contains(&child));
        assert!(removed.is_empty());
    }

    #[test]
    fn test_remove_node_with_parents_only() {
        let dag = BlockDependencyDag::empty();
        let parent = create_block_hash(b"parent");
        let child = create_block_hash(b"child");

        dag.add(parent.clone(), child.clone());
        let (affected, removed, _orphaned_parents) = dag.remove(child.clone()).unwrap();

        assert!(!dag
            .parent_to_child_adjacency_list
            .get(&parent)
            .is_some_and(|children| children.contains(&child)));
        assert!(!dag.parent_to_child_adjacency_list.contains_key(&child));
        assert!(!dag.child_to_parent_adjacency_list.contains_key(&child));
        assert!(affected.is_empty());
        assert!(removed.is_empty());
    }

    #[test]
    fn test_remove_node_with_children_but_no_parents() {
        let dag = BlockDependencyDag::empty();
        let parent = create_block_hash(b"tempblock");
        let child = create_block_hash(b"child");

        dag.add(parent.clone(), child.clone());
        let result = dag.remove(parent.clone());
        assert!(result.is_ok());

        let (affected, removed, _orphaned_parents) = result.unwrap();
        assert!(affected.is_empty());
        assert!(removed.contains(&child));
        assert!(!dag.parent_to_child_adjacency_list.contains_key(&parent));
        assert!(!dag.child_to_parent_adjacency_list.contains_key(&parent));
        assert!(!dag.child_to_parent_adjacency_list.contains_key(&child));
        assert!(dag.dependency_free.contains(&child));
    }

    #[test]
    fn test_remove_tolerates_stale_parent_links() {
        let dag = BlockDependencyDag::empty();
        let valid_parent = create_block_hash(b"valid-parent");
        let stale_parent = create_block_hash(b"stale-parent");
        let child = create_block_hash(b"child");

        dag.add(valid_parent.clone(), child.clone());

        // Inject a stale parent link for the child that has no corresponding forward edge.
        if let Some(child_parents) = dag.child_to_parent_adjacency_list.get_mut(&child) {
            child_parents.insert(stale_parent.clone());
        }

        assert!(dag
            .child_to_parent_adjacency_list
            .get(&child)
            .is_some_and(|parents| parents.contains(&stale_parent)));

        let (_affected, _removed, _orphaned_parents) = dag.remove(child.clone()).unwrap();

        assert!(!dag.child_to_parent_adjacency_list.contains_key(&child));
        assert!(!dag.parent_to_child_adjacency_list.contains_key(&child));
        assert!(!dag
            .parent_to_child_adjacency_list
            .get(&valid_parent)
            .is_some_and(|children| children.contains(&child)));
        assert!(!dag
            .parent_to_child_adjacency_list
            .get(&stale_parent)
            .is_some_and(|children| children.contains(&child)));
    }

    #[test]
    fn test_remove_tolerates_stale_child_to_parent_entry() {
        let dag = BlockDependencyDag::empty();
        let parent = create_block_hash(b"parent");
        let child = create_block_hash(b"child");

        dag.add(parent.clone(), child.clone());

        // Simulate a stale/partial edge where reverse lookup is missing.
        dag.child_to_parent_adjacency_list.remove(&child);

        let (affected, removed, _orphaned_parents) = dag.remove(parent.clone()).unwrap();

        assert!(!dag.parent_to_child_adjacency_list.contains_key(&parent));
        assert!(!dag.child_to_parent_adjacency_list.contains_key(&parent));
        assert!(!dag.parent_to_child_adjacency_list.contains_key(&child));
        assert!(!dag.child_to_parent_adjacency_list.contains_key(&child));
        assert!(removed.contains(&child));
        assert!(affected.is_empty());
    }
}
