// See block-storage/src/main/scala/coop/rchain/blockstorage/dag/BlockMetadataStore.scala

use dashmap::{DashMap, DashSet};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
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

struct DagState {
    dag_set: Arc<DashSet<BlockHash>>,
    child_map: Arc<DashMap<BlockHash, Arc<DashSet<BlockHash>>>>,
    height_map: Arc<RwLock<BTreeMap<i64, DashSet<BlockHash>>>>,
    // In general - at least genesis should be LFB.
    // But dagstate can be empty, as it is initialized before genesis is inserted.
    // Also lots of tests do not have genesis properly initialised, so fixing all this is pain.
    // So this is Option.
    last_finalized_block: Option<(BlockHash, i64)>,
    finalized_block_set: Arc<DashSet<BlockHash>>,
}

impl DagState {
    fn new() -> Self {
        Self {
            dag_set: Arc::new(DashSet::new()),
            child_map: Arc::new(DashMap::new()),
            height_map: Arc::new(RwLock::new(BTreeMap::new())),
            last_finalized_block: Some((BlockHash::new(), 0)),
            finalized_block_set: Arc::new(DashSet::new()),
        }
    }
}

struct BlockInfo {
    hash: BlockHash,
    parents: Arc<DashSet<BlockHash>>,
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
        let parents = Arc::new(block_metadata.parents.iter().cloned().collect());

        BlockInfo {
            hash: hash.clone(),
            parents,
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
        let mut current_dag_state = self.dag_state.write().unwrap();
        for hash in indirectly {
            current_dag_state.finalized_block_set.insert(hash);
        }
        current_dag_state
            .finalized_block_set
            .insert(directly.clone());

        // update lastFinalizedBlock only when current one is lower
        if current_dag_state.last_finalized_block.is_none()
            || current_dag_state.last_finalized_block.as_ref().unwrap().1
                <= new_meta_for_df.block_number
        {
            current_dag_state.last_finalized_block =
                Some((directly.clone(), new_meta_for_df.block_number));
        }
        drop(current_dag_state);

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

    // DAG state operations

    pub fn dag_set(&self) -> Arc<DashSet<BlockHash>> {
        self.dag_state.read().unwrap().dag_set.clone()
    }

    pub fn contains(&self, hash: &BlockHash) -> bool {
        self.dag_state.read().unwrap().dag_set.contains(hash)
    }

    pub fn child_map(&self) -> Arc<DashMap<BlockHash, Arc<DashSet<BlockHash>>>> {
        self.dag_state.read().unwrap().child_map.clone()
    }

    pub fn height_map(&self) -> Arc<RwLock<BTreeMap<i64, DashSet<BlockHash>>>> {
        self.dag_state.read().unwrap().height_map.clone()
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

    pub fn finalized_block_set(&self) -> Arc<DashSet<BlockHash>> {
        self.dag_state.read().unwrap().finalized_block_set.clone()
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
        state_guard
            .child_map
            .entry(hash.clone())
            .or_insert_with(|| Arc::new(DashSet::new()));

        // Add current block as child to all its parents
        for parent in block_info.parents.iter() {
            let children_set = state_guard
                .child_map
                .entry(parent.clone())
                .or_insert_with(|| Arc::new(DashSet::new()))
                .clone();
            children_set.insert(hash.clone());
        }

        // Update height map
        if !block_info.is_invalid {
            let mut height_map_guard = state_guard.height_map.write().unwrap();
            height_map_guard
                .entry(block_info.block_num)
                .or_insert_with(|| DashSet::new())
                .insert(hash.clone());
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
        let height_map_guard = dag_state_guard.height_map.read().unwrap();
        // Validate height map index (block numbers) are in sequence without holes
        let (min, max) = if !height_map_guard.is_empty() {
            (
                *height_map_guard.first_key_value().unwrap().0,
                *height_map_guard.last_key_value().unwrap().0 + 1,
            )
        } else {
            (0, 0)
        };
        assert!(
            max - min == height_map_guard.len() as i64,
            "DAG store height map has numbers not in sequence."
        );
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
