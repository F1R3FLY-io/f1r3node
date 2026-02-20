// See block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockMetadataStore.scala

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
};

use models::rust::block_hash::{BlockHash, BlockHashSerde};
use models::rust::block_metadata::BlockMetadata;
use models::rust::casper::pretty_printer::PrettyPrinter;
use shared::rust::store::key_value_store::KvStoreError;
use shared::rust::store::key_value_typed_store::KeyValueTypedStore;
use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;

pub struct BlockMetadataStore {
    store: KeyValueTypedStoreImpl<BlockHashSerde, BlockMetadata>,
    dag_state: Arc<RwLock<DagState>>,
}

/// In-memory DAG state using persistent immutable collections (imbl).
/// Clone is O(1) via structural sharing, enabling race-free snapshots.
pub(crate) struct DagState {
    pub(crate) dag_set: imbl::HashSet<BlockHash>,
    pub(crate) child_map: imbl::HashMap<BlockHash, imbl::HashSet<BlockHash>>,
    pub(crate) height_map: imbl::OrdMap<i64, imbl::HashSet<BlockHash>>,
    // In general - at least genesis should be LFB.
    // But dagstate can be empty, as it is initialized before genesis is inserted.
    // Also lots of tests do not have genesis properly initialised, so fixing all this is pain.
    // So this is Option.
    pub(crate) last_finalized_block: Option<(BlockHash, i64)>,
    pub(crate) finalized_block_set: imbl::HashSet<BlockHash>,
}

impl DagState {
    fn new() -> Self {
        Self {
            dag_set: imbl::HashSet::new(),
            child_map: imbl::HashMap::new(),
            height_map: imbl::OrdMap::new(),
            last_finalized_block: Some((BlockHash::new(), 0)),
            finalized_block_set: imbl::HashSet::new(),
        }
    }
}

struct BlockInfo {
    hash: BlockHash,
    parents: Vec<BlockHash>,
    block_num: i64,
    is_invalid: bool,
    is_directly_finalized: bool,
    is_finalized: bool,
}

impl BlockMetadataStore {
    pub fn new(
        block_metadata_store: KeyValueTypedStoreImpl<BlockHashSerde, BlockMetadata>,
    ) -> Self {
        let blocks_info_result = block_metadata_store
            .collect(|(hash, metadata)| {
                Some((
                    hash.0.clone(),
                    Self::block_metadata_to_info(&hash.0, metadata),
                ))
            })
            .unwrap_or_else(|e| {
                tracing::warn!(
                    "Warning: Failed to collect block metadata: {}. Continuing with empty store.",
                    e
                );
                Vec::new()
            });

        let blocks_info_map = blocks_info_result.into_iter().collect::<HashMap<_, _>>();
        let dag_state = Self::recreate_in_memory_state(blocks_info_map);

        Self {
            store: block_metadata_store,
            dag_state,
        }
    }

    fn block_metadata_to_info(hash: &BlockHash, block_metadata: &BlockMetadata) -> BlockInfo {
        BlockInfo {
            hash: hash.clone(),
            parents: block_metadata.parents.clone(),
            block_num: block_metadata.block_number,
            is_invalid: block_metadata.invalid,
            is_directly_finalized: block_metadata.directly_finalized,
            is_finalized: block_metadata.finalized,
        }
    }

    pub fn add(&mut self, block_metadata: BlockMetadata) -> Result<(), KvStoreError> {
        let block_hash = block_metadata.block_hash.clone();
        let block_info = Self::block_metadata_to_info(&block_hash, &block_metadata);

        self.dag_state = Self::validate_dag_state(Self::add_block_to_dag_state(
            self.dag_state.clone(),
            block_info,
        ));

        // Update persistent block metadata store
        self.store
            .put_one(BlockHashSerde(block_hash), block_metadata)?;

        Ok(())
    }

    /** Record new last finalized lock. Directly finalized is the output of finalizer,
     * indirectly finalized are new LFB ancestors. */
    pub fn record_finalized(
        &mut self,
        directly: BlockHash,
        indirectly: HashSet<BlockHash>,
    ) -> Result<(), KvStoreError> {
        let indirectly_serde: Vec<BlockHashSerde> = indirectly
            .iter()
            .map(|hash| BlockHashSerde(hash.clone()))
            .collect();

        let cur_metas_for_if = self.store.get_batch(&indirectly_serde)?;

        // new values to persist
        let mut new_meta_for_df = self.store.get_unsafe(&BlockHashSerde(directly.clone()))?;
        new_meta_for_df.finalized = true;
        new_meta_for_df.directly_finalized = true;

        let new_metas_for_if: Vec<(BlockHashSerde, BlockMetadata)> = cur_metas_for_if
            .into_iter()
            .map(|mut v| {
                v.finalized = true;
                (BlockHashSerde(v.block_hash.clone()), v)
            })
            .collect();

        // Add all blocks to finalized set
        let mut dag_state_guard = self.dag_state.write().unwrap();
        for hash in indirectly {
            dag_state_guard.finalized_block_set.insert(hash);
        }
        dag_state_guard
            .finalized_block_set
            .insert(directly.clone());

        // update lastFinalizedBlock only when current one is lower
        if dag_state_guard.last_finalized_block.is_none()
            || dag_state_guard.last_finalized_block.as_ref().unwrap().1
                <= new_meta_for_df.block_number
        {
            dag_state_guard.last_finalized_block =
                Some((directly.clone(), new_meta_for_df.block_number));
        }
        drop(dag_state_guard);

        // persist new values all at once
        let mut new_values = Vec::with_capacity(1 + new_metas_for_if.len());
        new_values.push((BlockHashSerde(directly), new_meta_for_df));
        new_values.extend(new_metas_for_if);
        self.store.put(new_values)?;

        Ok(())
    }

    pub fn get(&self, hash: &BlockHash) -> Result<Option<BlockMetadata>, KvStoreError> {
        self.store.get_one(&BlockHashSerde(hash.clone()))
    }

    pub fn get_unsafe(&self, hash: &BlockHash) -> Result<BlockMetadata, KvStoreError> {
        self.get(hash)?.ok_or_else(|| {
            KvStoreError::KeyNotFound(format!(
                "BlockMetadataStore is missing key {}",
                PrettyPrinter::build_string_bytes(&hash.to_vec())
            ))
        })
    }

    // DAG state operations — all return O(1) clones via imbl structural sharing

    pub(crate) fn dag_state(&self) -> &Arc<RwLock<DagState>> {
        &self.dag_state
    }

    pub fn contains(&self, hash: &BlockHash) -> bool {
        self.dag_state.read().unwrap().dag_set.contains(hash)
    }

    pub fn last_finalized_block(&self) -> BlockHash {
        self.dag_state
            .read()
            .unwrap()
            .last_finalized_block
            .as_ref()
            .expect("DagState does not contain lastFinalizedBlock. Are you calling this on empty BlockDagStorage? Otherwise there is a bug.")
            .0
            .clone()
    }

    fn add_block_to_dag_state(
        state: Arc<RwLock<DagState>>,
        block_info: BlockInfo,
    ) -> Arc<RwLock<DagState>> {
        let hash = &block_info.hash;
        let mut state_guard = state.write().unwrap();

        // Update dag set / all blocks in the DAG
        state_guard.dag_set.insert(hash.clone());

        // Update children relation map
        // Create entry for current block (with empty children set initially)
        if !state_guard.child_map.contains_key(hash) {
            state_guard
                .child_map
                .insert(hash.clone(), imbl::HashSet::new());
        }

        // Add current block as child to all its parents
        for parent in block_info.parents.iter() {
            let children = state_guard
                .child_map
                .remove(parent)
                .unwrap_or_else(imbl::HashSet::new);
            let mut children = children;
            children.insert(hash.clone());
            state_guard.child_map.insert(parent.clone(), children);
        }

        // Update height map
        if !block_info.is_invalid {
            let hashes = state_guard
                .height_map
                .remove(&block_info.block_num)
                .unwrap_or_else(imbl::HashSet::new);
            let mut hashes = hashes;
            hashes.insert(hash.clone());
            state_guard
                .height_map
                .insert(block_info.block_num, hashes);
        }

        if block_info.is_directly_finalized
            && state_guard
                .last_finalized_block
                .as_ref()
                .map_or(true, |&(_, height)| height <= block_info.block_num)
        {
            state_guard.last_finalized_block = Some((hash.clone(), block_info.block_num));
        }

        if block_info.is_finalized {
            state_guard.finalized_block_set.insert(block_info.hash);
        }

        state.clone()
    }

    fn validate_dag_state(dag_state: Arc<RwLock<DagState>>) -> Arc<RwLock<DagState>> {
        let dag_state_guard = dag_state.read().unwrap();
        let height_map = &dag_state_guard.height_map;
        // Validate height map index (block numbers) are in sequence without holes
        let (min, max) = if !height_map.is_empty() {
            (
                height_map.get_min().unwrap().0,
                height_map.get_max().unwrap().0 + 1,
            )
        } else {
            (0, 0)
        };
        assert!(
            max - min == height_map.len() as i64,
            "DAG store height map has numbers not in sequence."
        );
        drop(dag_state_guard);
        dag_state.clone()
    }

    fn recreate_in_memory_state(
        blocks_info_map: HashMap<BlockHash, BlockInfo>,
    ) -> Arc<RwLock<DagState>> {
        let empty_state = Arc::new(RwLock::new(DagState::new()));

        // Add blocks to DAG state
        let dag_state = blocks_info_map
            .into_iter()
            .fold(empty_state, |state, (_, block_info)| {
                Self::add_block_to_dag_state(state, block_info)
            });

        Self::validate_dag_state(dag_state)
    }
}
