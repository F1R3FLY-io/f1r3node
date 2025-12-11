// See block-storage/src/main/scala/coop/rchain/blockstorage/finality/LastFinalizedKeyValueStorage.scala

use prost::bytes::Bytes;
use std::collections::HashMap;

use models::rust::block_hash::{BlockHash, BlockHashSerde};
use models::rust::block_metadata::BlockMetadata;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use shared::rust::dag::dag_ops;
use shared::rust::store::key_value_store::KvStoreError;
use shared::rust::store::key_value_typed_store::KeyValueTypedStore;
use shared::rust::store::key_value_typed_store_impl::KeyValueTypedStoreImpl;

use super::LastFinalizedStorage;
use crate::rust::key_value_block_store::KeyValueBlockStore;

/// LMDB-backed implementation of LastFinalizedStorage
pub struct LastFinalizedKeyValueStorage {
    last_finalized_block_db: KeyValueTypedStoreImpl<i32, BlockHashSerde>,
    fixed_key: i32,
}

impl LastFinalizedKeyValueStorage {
    /// Sentinel value to mark migration as complete (32 bytes of 0xFF)
    const DONE: BlockHash = Bytes::from_static(&[
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF,
    ]);

    /// Create a new LastFinalizedKeyValueStorage from a typed store
    pub fn new(last_finalized_block_db: KeyValueTypedStoreImpl<i32, BlockHashSerde>) -> Self {
        Self {
            last_finalized_block_db,
            fixed_key: 1,
        }
    }

    /// Create a new LastFinalizedKeyValueStorage from a KeyValueStoreManager
    pub async fn create_from_kvm(
        kvm: &mut dyn KeyValueStoreManager,
    ) -> Result<Self, KvStoreError> {
        let last_finalized_kv_store = kvm.store("last-finalized-block".to_string()).await?;
        let last_finalized_block_db: KeyValueTypedStoreImpl<i32, BlockHashSerde> =
            KeyValueTypedStoreImpl::new(last_finalized_kv_store);
        Ok(Self::new(last_finalized_block_db))
    }

    /// Check if migration from old LastFinalizedStorage format is required
    pub fn require_migration(&self) -> Result<bool, KvStoreError> {
        let value = self.get()?;
        Ok(value.map_or(false, |hash| hash != Self::DONE))
    }

    /// Migrate LastFinalizedStorage to BlockDagStorage
    ///
    /// This migration:
    /// 1. Reads the LFB from old storage or ApprovedBlock
    /// 2. Marks it as directly finalized in block metadata
    /// 3. Traverses DAG breadth-first to find all ancestor blocks
    /// 4. Marks all ancestors as finalized in chunks of 10,000
    /// 5. Marks migration as complete with DONE sentinel
    pub async fn migrate_lfb(
        &self,
        kvm: &mut dyn KeyValueStoreManager,
        block_store: &KeyValueBlockStore,
    ) -> Result<(), KvStoreError> {
        tracing::info!("Starting migration of LastFinalizedStorage to BlockDagStorage");

        let err_no_lfb_in_storage =
            "No LFB in LastFinalizedStorage nor ApprovedBlock found when attempting migration.";
        let err_no_metadata_for_lfb = "No metadata found for LFB when attempting migration.";

        // Get block metadata database
        let block_metadata_kv_store = kvm.store("block-metadata".to_string()).await?;
        let block_metadata_db: KeyValueTypedStoreImpl<BlockHashSerde, BlockMetadata> =
            KeyValueTypedStoreImpl::new(block_metadata_kv_store);

        // Get LFB from LastFinalizedStorage or ApprovedBlock
        let persisted_lfb_opt = self.get()?;
        let approved_block_hash_opt = block_store
            .get_approved_block()?
            .map(|ab| ab.candidate.block.block_hash);

        let lfb = persisted_lfb_opt
            .or(approved_block_hash_opt)
            .ok_or_else(|| KvStoreError::KeyNotFound(err_no_lfb_in_storage.to_string()))?;

        // Mark LFB as directly finalized
        let mut cur_v = block_metadata_db
            .get_one(&BlockHashSerde(lfb.clone()))?
            .ok_or_else(|| KvStoreError::KeyNotFound(err_no_metadata_for_lfb.to_string()))?;

        cur_v.directly_finalized = true;
        cur_v.finalized = true;
        block_metadata_db.put_one(BlockHashSerde(lfb.clone()), cur_v)?;

        // Build blocks info map for DAG traversal
        let blocks_info_map: HashMap<BlockHash, BlockInfo> = block_metadata_db
            .collect(|(hash_serde, metadata)| {
                Some((
                    hash_serde.0.clone(),
                    BlockInfo {
                        parents: metadata.parents.clone(),
                    },
                ))
            })?
            .into_iter()
            .collect();

        tracing::info!("Migration of LFB done. Starting finalized blocks traversal.");

        // Traverse DAG breadth-first to find all finalized blocks
        let finalized_block_set = dag_ops::bf_traverse(vec![lfb.clone()], |bh| {
            blocks_info_map
                .get(bh)
                .map(|info| {
                    // Filter parents that exist in the blockmetadataDB
                    // (trimmed state might have removed some)
                    info.parents
                        .iter()
                        .filter(|p| blocks_info_map.contains_key(*p))
                        .cloned()
                        .collect()
                })
                .unwrap_or_else(Vec::new)
        });

        tracing::info!(
            "Found {} finalized blocks to record",
            finalized_block_set.len()
        );

        // Process blocks in chunks of 10,000
        let chunk_size = 10000;
        let total_blocks = finalized_block_set.len();
        let chunks: Vec<&[BlockHash]> = finalized_block_set.chunks(chunk_size).collect();

        let mut processed = 0;
        for chunk in chunks {
            self.process_chunk(&block_metadata_db, chunk)?;
            processed += chunk.len();
            tracing::info!(
                "Finalized blocks recorded: {} of {}",
                processed,
                total_blocks
            );
        }

        // Mark migration as done
        self.put(Self::DONE)?;
        tracing::info!("Migration complete. LastFinalizedStorage marked as migrated.");

        Ok(())
    }

    /// Process a chunk of block hashes, marking them as finalized
    fn process_chunk(
        &self,
        block_metadata_db: &KeyValueTypedStoreImpl<BlockHashSerde, BlockMetadata>,
        chunk: &[BlockHash],
    ) -> Result<(), KvStoreError> {
        // Convert to BlockHashSerde for batch get
        let chunk_serde: Vec<BlockHashSerde> = chunk
            .iter()
            .map(|hash| BlockHashSerde(hash.clone()))
            .collect();

        // Get current values
        let cur_vs = block_metadata_db.get_batch(&chunk_serde)?;

        // Update finalization flags
        // WARNING: migration should be done before block merge, as it assumes all blocks are directly finalized.
        let new_values: Vec<(BlockHashSerde, BlockMetadata)> = cur_vs
            .into_iter()
            .map(|mut v| {
                v.directly_finalized = true;
                v.finalized = true;
                (BlockHashSerde(v.block_hash.clone()), v)
            })
            .collect();

        // Persist updated values
        block_metadata_db.put(new_values)?;
        Ok(())
    }
}

impl LastFinalizedStorage for LastFinalizedKeyValueStorage {
    fn put(&self, block_hash: BlockHash) -> Result<(), KvStoreError> {
        self.last_finalized_block_db
            .put_one(self.fixed_key, BlockHashSerde(block_hash))
    }

    fn get(&self) -> Result<Option<BlockHash>, KvStoreError> {
        self.last_finalized_block_db
            .get_one(&self.fixed_key)
            .map(|opt| opt.map(|hash_serde| hash_serde.0))
    }
}

/// Helper struct for DAG traversal during migration
struct BlockInfo {
    parents: Vec<BlockHash>,
}

