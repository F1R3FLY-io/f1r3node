// See casper/src/main/scala/coop/rchain/casper/util/DagOperations.scala

use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use models::rust::block_metadata::BlockMetadata;
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

        async fn loop_impl(
            curr_map: HashMap<BlockMetadata, HashSet<u8>>,
            enqueued: HashSet<BlockMetadata>,
            uncommon_enqueued: HashSet<BlockMetadata>,
            q: &mut BinaryHeap<OrderedBlockMetadata>,
            dag: &KeyValueDagRepresentation,
            common_set: &HashSet<u8>,
        ) -> Result<HashMap<BlockMetadata, HashSet<u8>>, KvStoreError> {
            if uncommon_enqueued.is_empty() {
                return Ok(curr_map);
            }

            let curr_block = q
                .pop()
                .expect("Priority queue should not be empty during traversal")
                .0;

            // Note: Instead of BitSet in Rust and union function from models/util.rs,
            // We have used HashSets<u8> and extend function which provides the correct set semantic that match the original Scala BitSet behavior,
            // while the current BitSet implementation and union functions in the Rust side is designed for different use cases (bitwise operations, not set operations).
            let curr_set = curr_map.get(&curr_block).cloned().unwrap_or_default();

            let curr_parents = parents(&curr_block, dag).await?;

            let new_map = curr_map;
            let mut new_enqueued = enqueued;
            let mut new_uncommon = uncommon_enqueued;

            new_enqueued.remove(&curr_block);
            new_uncommon.remove(&curr_block);

            let (mut new_map, new_enqueued, new_uncommon) = curr_parents.iter().fold(
                (new_map, new_enqueued, new_uncommon),
                |(mut map, mut enq, mut unc), p| {
                    if !enq.contains(p) {
                        q.push(OrderedBlockMetadata(p.clone()));
                    }

                    let mut p_set = map.get(p).cloned().unwrap_or_default();
                    p_set.extend(&curr_set);

                    if is_common(&p_set, &common_set) {
                        unc.remove(&p);
                    } else {
                        unc.insert(p.clone());
                    }

                    map.insert(p.clone(), p_set);
                    enq.insert(p.clone());

                    (map, enq, unc)
                },
            );

            if is_common(&curr_set, common_set) {
                new_map.remove(&curr_block);
                Box::pin(loop_impl(
                    new_map,
                    new_enqueued,
                    new_uncommon,
                    q,
                    dag,
                    common_set,
                ))
                .await
            } else {
                Box::pin(loop_impl(
                    new_map,
                    new_enqueued,
                    new_uncommon,
                    q,
                    dag,
                    common_set,
                ))
                .await
            }
        }

        loop_impl(
            init_map,
            HashSet::new(),
            blocks.iter().cloned().collect::<HashSet<_>>(),
            &mut q,
            dag,
            &common_set,
        )
        .await
        .map(|result| {
            result
                .into_iter()
                .filter(|(_, set)| !is_common(set, &common_set))
                .collect()
        })
    }

    /// Conceptually, the LUCA is the lowest point at which the histories of b1 and b2 diverge.
    /// We compute by finding the first block that is the "lowest" (has highest blocknum) block common
    /// for both blocks' ancestors.
    pub async fn lowest_universal_common_ancestor(
        b1: &BlockMetadata,
        b2: &BlockMetadata,
        dag: &KeyValueDagRepresentation,
    ) -> Result<BlockMetadata, KvStoreError> {
        async fn get_parents(
            p: &BlockMetadata,
            dag: &KeyValueDagRepresentation,
        ) -> Result<HashSet<BlockMetadata>, KvStoreError> {
            let parents: Result<Vec<_>, _> =
                p.parents.iter().map(|b| dag.lookup_unsafe(b)).collect();
            Ok(parents?.into_iter().collect())
        }

        async fn extract_parents_from_highest_num_block(
            blocks: BTreeSet<ReverseOrderedBlockMetadata>,
            dag: &KeyValueDagRepresentation,
        ) -> Result<BTreeSet<ReverseOrderedBlockMetadata>, KvStoreError> {
            let (head, tail) = (
                blocks.iter().next().expect("BTreeSet should not be empty"),
                blocks.iter().skip(1).cloned(),
            );

            get_parents(&head.0, dag).await.map(|new_blocks| {
                tail.chain(new_blocks.into_iter().map(ReverseOrderedBlockMetadata))
                    .collect()
            })
        }

        if b1 == b2 {
            Ok(b1.clone())
        } else {
            let mut start = BTreeSet::new();
            start.insert(ReverseOrderedBlockMetadata(b1.clone()));
            start.insert(ReverseOrderedBlockMetadata(b2.clone()));

            let final_result = {
                let mut current = start;
                loop {
                    current = extract_parents_from_highest_num_block(current, dag).await?;
                    if current.len() == 1 {
                        break current;
                    }
                }
            };

            final_result
                .into_iter()
                .next()
                .map(|wrapper| wrapper.0)
                .ok_or_else(|| KvStoreError::KeyNotFound("No common ancestor found".to_string()))
        }
    }
}
