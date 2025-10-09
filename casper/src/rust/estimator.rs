// See casper/src/main/scala/coop/rchain/casper/Estimator.scala

use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use futures::stream::{self, StreamExt, TryStreamExt};
use models::rust::{
    block_hash::BlockHash, block_metadata::BlockMetadata,
    casper::protocol::casper_message::BlockMessage, validator::Validator,
};
use shared::rust::dag::dag_ops;
use shared::rust::shared::list_ops::ListOps;
use shared::rust::store::key_value_store::KvStoreError;
use std::collections::{HashMap, HashSet};

use crate::rust::util::dag_operations::DagOperations;
use crate::rust::util::proto_util;

/// Tips of the DAG, ranked against LCA
#[derive(Debug, Clone, PartialEq)]
pub struct ForkChoice {
    pub tips: Vec<BlockHash>,
    pub lca: BlockHash,
}

#[derive(Debug, Clone)]
pub struct Estimator {
    max_number_of_parents: i32,
    max_parent_depth_opt: Option<i32>,
}

impl Estimator {
    pub const UNLIMITED_PARENTS: i32 = i32::MAX;
    const LATEST_MESSAGE_MAX_DEPTH: i64 = 1000;

    pub fn apply(max_number_of_parents: i32, max_parent_depth_opt: Option<i32>) -> Self {
        Self {
            max_number_of_parents,
            max_parent_depth_opt,
        }
    }

    pub async fn tips(
        &self,
        dag: &mut KeyValueDagRepresentation,
        genesis: &BlockMessage,
    ) -> Result<ForkChoice, KvStoreError> {
        let arc_latest_message_hashes = dag.latest_message_hashes();
        let latest_message_hashes: HashMap<Validator, BlockHash> = arc_latest_message_hashes
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        self.tips_with_latest_messages(dag, genesis, latest_message_hashes)
            .await
    }

    /// When the BlockDag has an empty latestMessages, tips will return IndexedSeq(genesis.blockHash)
    pub async fn tips_with_latest_messages(
        &self,
        dag: &mut KeyValueDagRepresentation,
        genesis: &BlockMessage,
        latest_messages_hashes: HashMap<Validator, BlockHash>,
    ) -> Result<ForkChoice, KvStoreError> {
        let invalid_latest_messages =
            dag.invalid_latest_messages_from_hashes(latest_messages_hashes.clone())?;

        let mut filtered_latest_messages_hashes = latest_messages_hashes;
        filtered_latest_messages_hashes
            .retain(|validator, _| !invalid_latest_messages.contains_key(validator));

        let genesis_metadata = BlockMetadata::from_block(genesis, false, None, None);
        let lca =
            Self::calculate_lca(dag, &genesis_metadata, &filtered_latest_messages_hashes).await?;

        let scores_map =
            Self::build_scores_map(dag, &filtered_latest_messages_hashes, &lca).await?;

        let ranked_latest_messages_hashes =
            Self::rank_forkchoices(vec![lca.clone()], dag, &scores_map).await?;

        let ranked_shallow_hashes = self
            .filter_deep_parents(ranked_latest_messages_hashes, dag)
            .await?;

        Ok(ForkChoice {
            tips: ranked_shallow_hashes
                .into_iter()
                .take(self.max_number_of_parents as usize)
                .collect(),
            lca,
        })
    }

    async fn filter_deep_parents(
        &self,
        ranked_latest_hashes: Vec<BlockHash>,
        dag: &KeyValueDagRepresentation,
    ) -> Result<Vec<BlockHash>, KvStoreError> {
        match self.max_parent_depth_opt {
            Some(max_parent_depth) => {
                let (main_hash, secondary_hashes) = ranked_latest_hashes.split_first().unwrap();

                let max_block_number = dag.lookup_unsafe(main_hash)?.block_number;

                let secondary_parents: Vec<BlockMetadata> = secondary_hashes
                    .iter()
                    .map(|hash| dag.lookup_unsafe(hash))
                    .collect::<Result<Vec<_>, _>>()?;

                let shallow_parents: Vec<BlockMetadata> = secondary_parents
                    .into_iter()
                    .filter(|p| max_block_number - p.block_number <= max_parent_depth as i64)
                    .collect();

                Ok(std::iter::once(main_hash.clone())
                    .chain(shallow_parents.into_iter().map(|p| p.block_hash))
                    .collect())
            }
            None => Ok(ranked_latest_hashes),
        }
    }

    async fn calculate_lca(
        block_dag: &KeyValueDagRepresentation,
        genesis: &BlockMetadata,
        latest_messages_hashes: &HashMap<Validator, BlockHash>,
    ) -> Result<BlockHash, KvStoreError> {
        let latest_messages: Vec<BlockMetadata> = latest_messages_hashes
            .values()
            .map(|hash| block_dag.lookup(hash))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        let top_block_number = block_dag.latest_block_number();

        let filtered_lm: Vec<BlockMetadata> = latest_messages
            .into_iter()
            .filter(|msg| msg.block_number > top_block_number - Self::LATEST_MESSAGE_MAX_DEPTH)
            .collect();

        let result = if filtered_lm.is_empty() {
            genesis.block_hash.clone()
        } else {
            let (head, tail) = filtered_lm.split_first().unwrap();

            let mut acc = head.clone();
            // TODO: Scala message - Change to mainParentLCA
            for latest_message in tail {
                acc = DagOperations::lowest_universal_common_ancestor(
                    &acc,
                    latest_message,
                    block_dag,
                )
                .await?;
            }

            acc.block_hash
        };

        Ok(result)
    }

    async fn build_scores_map(
        block_dag: &mut KeyValueDagRepresentation,
        latest_messages_hashes: &HashMap<Validator, BlockHash>,
        lowest_common_ancestor: &BlockHash,
    ) -> Result<HashMap<BlockHash, i64>, KvStoreError> {
        fn hash_parents(
            hash: &BlockHash,
            last_finalized_block_number: i64,
            block_dag: &KeyValueDagRepresentation,
        ) -> Result<Vec<BlockHash>, KvStoreError> {
            let current_block_number = block_dag.lookup_unsafe(hash)?.block_number;

            if current_block_number < last_finalized_block_number {
                Ok(Vec::new())
            } else {
                Ok(block_dag.lookup_unsafe(hash)?.parents)
            }
        }

        async fn add_validator_weight_down_supporting_chain(
            score_map: HashMap<BlockHash, i64>,
            validator: &Validator,
            latest_block_hash: &BlockHash,
            block_dag: &mut KeyValueDagRepresentation,
            lowest_common_ancestor: &BlockHash,
        ) -> Result<HashMap<BlockHash, i64>, KvStoreError> {
            let lca_block_num = block_dag
                .lookup_unsafe(lowest_common_ancestor)?
                .block_number;

            let traversed_hashes: Vec<BlockHash> =
                dag_ops::bf_traverse(vec![latest_block_hash.clone()], |hash| {
                    hash_parents(hash, lca_block_num, block_dag).expect("Failed to get parents")
                });

            let mut result = score_map;
            for hash in traversed_hashes {
                let curr_score = *result.get(&hash).unwrap_or(&0);
                let validator_weight =
                    proto_util::weight_from_validator_by_dag(block_dag, &hash, validator)?;
                result.insert(hash, curr_score + validator_weight);
            }

            Ok(result)
        }

        // TODO: Scala message - Since map scores are additive it should be possible to do this in parallel
        let mut scores_map: HashMap<BlockHash, i64> = HashMap::new();
        for (validator, latest_block_hash) in latest_messages_hashes.iter() {
            scores_map = add_validator_weight_down_supporting_chain(
                scores_map,
                validator,
                latest_block_hash,
                block_dag,
                lowest_common_ancestor,
            )
            .await?;
        }

        Ok(scores_map)
    }

    async fn rank_forkchoices(
        blocks: Vec<BlockHash>,
        block_dag: &KeyValueDagRepresentation,
        scores: &HashMap<BlockHash, i64>,
    ) -> Result<Vec<BlockHash>, KvStoreError> {
        let unsorted_new_blocks: Vec<BlockHash> = stream::iter(blocks.iter())
            .then(|block| Self::replace_block_hash_with_children(block, block_dag, scores))
            .try_fold(Vec::new(), |mut acc, children| async move {
                acc.extend(children);
                Ok(acc)
            })
            .await?;

        let unique_blocks: Vec<BlockHash> = unsorted_new_blocks
            .into_iter()
            .collect::<HashSet<_>>() // distinct
            .into_iter()
            .collect();

        let new_blocks = ListOps::sort_by_with_decreasing_order(unique_blocks, scores);

        if Self::still_same(&blocks, &new_blocks) {
            Ok(blocks)
        } else {
            Box::pin(Self::rank_forkchoices(new_blocks, block_dag, scores)).await
        }
    }

    fn non_empty_list(elements: &HashSet<BlockHash>) -> Option<Vec<BlockHash>> {
        if elements.is_empty() {
            None
        } else {
            Some(elements.iter().cloned().collect())
        }
    }

    /// Only include children that have been scored,
    /// this ensures that the search does not go beyond
    /// the messages defined by blockDag.latestMessages
    async fn replace_block_hash_with_children(
        b: &BlockHash,
        block_dag: &KeyValueDagRepresentation,
        scores: &HashMap<BlockHash, i64>,
    ) -> Result<Vec<BlockHash>, KvStoreError> {
        match block_dag.children(b) {
            Some(children_set) => {
                let scored_children: HashSet<BlockHash> = children_set
                    .iter()
                    .filter_map(|child| {
                        let child_hash = child.clone();
                        if scores.contains_key(&child_hash) {
                            Some(child_hash)
                        } else {
                            None
                        }
                    })
                    .collect();

                match Self::non_empty_list(&scored_children) {
                    Some(non_empty_children) => Ok(non_empty_children),
                    None => Ok(vec![b.clone()]),
                }
            }
            None => Ok(vec![b.clone()]),
        }
    }

    fn still_same(blocks: &[BlockHash], new_blocks: &[BlockHash]) -> bool {
        new_blocks == blocks
    }
}
