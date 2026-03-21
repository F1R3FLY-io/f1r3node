// See casper/src/main/scala/coop/rchain/casper/util/DagOperations.scala

use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use models::rust::{block_hash::BlockHash, block_metadata::BlockMetadata};
use shared::rust::store::key_value_store::KvStoreError;
use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};

pub struct DagOperations;

// Wrapper for BlockMetadata to implement ordering for BinaryHeap
#[derive(Clone, Debug, PartialEq, Eq)]
struct OrderedBlockMetadata(BlockMetadata);

impl PartialOrd for OrderedBlockMetadata {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedBlockMetadata {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap in Rust is a max-heap and we want highest block_number first (like Scala PriorityQueue)
        // Scala uses BlockMetadata.orderingByNum, which orders by block_number, then by block_hash
        // Since BinaryHeap is max-heap and we want max block_number first, use direct ordering
        BlockMetadata::ordering_by_num(&self.0, &other.0)
    }
}

// Wrapper for BlockMetadata with reverse ordering for BTreeSet
#[derive(Clone, Debug, PartialEq, Eq)]
struct ReverseOrderedBlockMetadata(BlockMetadata);

impl PartialOrd for ReverseOrderedBlockMetadata {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReverseOrderedBlockMetadata {
    fn cmp(&self, other: &Self) -> Ordering {
        // Equivalent to Scala's SortedSet with BlockMetadata.orderingByNum.reverse
        // Reverse ordering: highest blocknum first (equivalent to orderingByNum.reverse)
        BlockMetadata::ordering_by_num(&self.0, &other.0).reverse()
    }
}

impl DagOperations {
    fn metadata_from_cache_or_dag(
        metadata_cache: &mut HashMap<BlockHash, BlockMetadata>,
        block_hash: &BlockHash,
        dag: &KeyValueDagRepresentation,
    ) -> Result<BlockMetadata, KvStoreError> {
        if let Some(metadata) = metadata_cache.get(block_hash) {
            return Ok(metadata.clone());
        }

        let metadata = dag.lookup_unsafe(block_hash)?;
        metadata_cache.insert(block_hash.clone(), metadata.clone());
        Ok(metadata)
    }

    /// Determines the ancestors to a set of blocks which are not common to all
    /// blocks in the set. Each starting block is assigned an index (hence the
    /// usage of slice with indices) and this is used to refer to that block in the result.
    /// A block B is an ancestor of a starting block with index i if the BitSet for
    /// B contains i.
    ///
    /// The `blocks` parameter is a slice of blocks to determine uncommon ancestors of.
    /// The `dag` parameter provides the DAG representation for traversing parent relationships.
    ///
    /// Returns a map from uncommon ancestor blocks to BitSets, where a block B is
    /// an ancestor of starting block with index i if B's BitSet contains i.
    pub async fn uncommon_ancestors(
        blocks: &[BlockMetadata],
        dag: &KeyValueDagRepresentation,
    ) -> Result<HashMap<BlockMetadata, HashSet<u8>>, KvStoreError> {
        let common_set: HashSet<u8> = (0..blocks.len()).map(|i| i as u8).collect();

        async fn parents(
            b: &BlockMetadata,
            dag: &KeyValueDagRepresentation,
        ) -> Result<Vec<BlockMetadata>, KvStoreError> {
            b.parents
                .iter()
                .map(|b| dag.lookup_unsafe(b))
                .collect::<Result<Vec<_>, _>>()
        }

        fn is_common(set: &HashSet<u8>, common_set: &HashSet<u8>) -> bool {
            set == common_set
        }

        let init_map: HashMap<BlockMetadata, HashSet<u8>> = blocks
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let mut set = HashSet::new();
                set.insert(i as u8);
                (b.clone(), set)
            })
            .collect();

        let mut q = BinaryHeap::new();
        for block in blocks {
            q.push(OrderedBlockMetadata(block.clone()));
        }

        let mut curr_map = init_map;
        let mut enqueued: HashSet<BlockMetadata> = HashSet::new();
        let mut uncommon_enqueued: HashSet<BlockMetadata> = blocks.iter().cloned().collect();

        while !uncommon_enqueued.is_empty() {
            let curr_block = q.pop().ok_or_else(|| {
                KvStoreError::InvalidArgument(
                    "Priority queue became empty during uncommon ancestor traversal".to_string(),
                )
            })?;
            let curr_block = curr_block.0;

            // Note: Instead of BitSet in Rust and union function from models/util.rs,
            // We have used HashSets<u8> and extend function which provides the correct set semantic that match the original Scala BitSet behavior,
            // while the current BitSet implementation and union functions in the Rust side is designed for different use cases (bitwise operations, not set operations).
            let curr_set = curr_map.get(&curr_block).cloned().unwrap_or_default();
            let curr_parents = parents(&curr_block, dag).await?;

            enqueued.remove(&curr_block);
            uncommon_enqueued.remove(&curr_block);

            for p in curr_parents {
                if !enqueued.contains(&p) {
                    q.push(OrderedBlockMetadata(p.clone()));
                }

                let mut p_set = curr_map.get(&p).cloned().unwrap_or_default();
                p_set.extend(curr_set.iter().copied());

                if is_common(&p_set, &common_set) {
                    uncommon_enqueued.remove(&p);
                } else {
                    uncommon_enqueued.insert(p.clone());
                }

                curr_map.insert(p.clone(), p_set);
                enqueued.insert(p);
            }

            if is_common(&curr_set, &common_set) {
                curr_map.remove(&curr_block);
            }
        }

        Ok(curr_map
            .into_iter()
            .filter(|(_, set)| !is_common(set, &common_set))
            .collect())
    }

    /// Conceptually, the LUCA is the lowest point at which the histories of b1 and b2 diverge.
    /// We compute by finding the first block that is the "lowest" (has highest blocknum) block common
    /// for both blocks' ancestors.
    pub async fn lowest_universal_common_ancestor_many(
        blocks: &[BlockMetadata],
        dag: &KeyValueDagRepresentation,
    ) -> Result<BlockMetadata, KvStoreError> {
        if blocks.is_empty() {
            return Err(KvStoreError::InvalidArgument(
                "Cannot compute LUCA for an empty block set".to_string(),
            ));
        }

        if blocks.len() == 1 {
            return Ok(blocks[0].clone());
        }

        let mut current: BTreeSet<ReverseOrderedBlockMetadata> = BTreeSet::new();
        let mut metadata_cache: HashMap<BlockHash, BlockMetadata> = HashMap::new();

        for block in blocks {
            metadata_cache.insert(block.block_hash.clone(), block.clone());
            current.insert(ReverseOrderedBlockMetadata(block.clone()));
        }

        loop {
            if current.len() == 1 {
                break current;
            }

            let (head, tail) = (
                current
                    .iter()
                    .next()
                    .expect("BTreeSet should not be empty")
                    .0
                    .clone(),
                current.iter().skip(1).cloned(),
            );

            let mut next: BTreeSet<ReverseOrderedBlockMetadata> = tail.collect();

            for parent_hash in &head.parents {
                let parent =
                    Self::metadata_from_cache_or_dag(&mut metadata_cache, parent_hash, dag)?;
                next.insert(ReverseOrderedBlockMetadata(parent));
            }

            current = next;
        }
        .into_iter()
        .next()
        .map(|wrapper| wrapper.0)
        .ok_or_else(|| KvStoreError::KeyNotFound("No common ancestor found".to_string()))
    }

    /// Conceptually, the LUCA is the lowest point at which the histories of b1 and b2 diverge.
    /// We compute by finding the first block that is the "lowest" (has highest blocknum) block common
    /// for both blocks' ancestors.
    pub async fn lowest_universal_common_ancestor(
        b1: &BlockMetadata,
        b2: &BlockMetadata,
        dag: &KeyValueDagRepresentation,
    ) -> Result<BlockMetadata, KvStoreError> {
        if b1 == b2 {
            return Ok(b1.clone());
        }

        Self::lowest_universal_common_ancestor_many(&[b1.clone(), b2.clone()], dag).await
    }
}
