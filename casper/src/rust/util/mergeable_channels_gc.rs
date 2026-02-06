//! Mergeable Channels Garbage Collection
//!
//! Garbage collects mergeable channel data for blocks that are provably unreachable.
//! This is required for multi-parent mode where immediate deletion during finalization
//! can cause data races.

use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use models::rust::block_hash::BlockHash;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use shared::rust::store::key_value_store::KvStoreError;
use shared::rust::store::key_value_typed_store::KeyValueTypedStore;

use crate::rust::casper::CasperShardConf;
use crate::rust::util::rholang::runtime_manager::RuntimeManager;

/// Garbage collects mergeable channel data for blocks that are provably unreachable.
///
/// A block's mergeable data is safe to delete when:
/// 1. The block is finalized
/// 2. All validators' latest messages are descendants of the block's children
/// 3. The block is deeper than maxParentDepth + depthBuffer from current tips
pub async fn collect_garbage(
    dag: &KeyValueDagRepresentation,
    block_store: &KeyValueBlockStore,
    runtime_manager: &std::sync::Arc<tokio::sync::Mutex<RuntimeManager>>,
    casper_shard_conf: &CasperShardConf,
) -> Result<usize, KvStoreError> {
    let mut deleted_count = 0;

    // Get all finalized blocks by traversing from genesis
    // Note: This could be optimized by tracking pending GC blocks
    let finalized_blocks = get_finalized_blocks(dag)?;

    for block_hash in finalized_blocks {
        if is_safe_to_delete(dag, &block_hash, casper_shard_conf)? {
            // Get block to access its state hash
            if let Some(block) = block_store.get(&block_hash)? {
                let state_hash =
                    Blake2b256Hash::from_bytes_prost(&block.body.state.post_state_hash);

                // Delete mergeable data
                runtime_manager
                    .lock()
                    .await
                    .mergeable_store
                    .delete(vec![state_hash.bytes()])?;

                deleted_count += 1;
                tracing::debug!(
                    "GC: Deleted mergeable data for block {}",
                    hex::encode(&block_hash)
                );
            }
        }
    }

    if deleted_count > 0 {
        metrics::counter!("mergeable_channels_gc_deleted").increment(deleted_count as u64);
        tracing::info!(
            "Mergeable channels GC: Deleted {} blocks' data",
            deleted_count
        );
    } else {
        tracing::debug!("Mergeable channels GC: No data to delete");
    }

    Ok(deleted_count)
}

/// Check if a block's mergeable data is safe to delete.
fn is_safe_to_delete(
    dag: &KeyValueDagRepresentation,
    block_hash: &BlockHash,
    casper_shard_conf: &CasperShardConf,
) -> Result<bool, KvStoreError> {
    // 1. Check if block is finalized
    if !dag.is_finalized(block_hash) {
        return Ok(false);
    }

    // 2. Check depth constraint
    let block_meta = dag.lookup_unsafe(block_hash)?;
    let max_block_number = dag.latest_block_number();
    let depth_from_tip = max_block_number - block_meta.block_number;
    let max_allowed_depth = (casper_shard_conf.max_parent_depth as i64)
        + (casper_shard_conf.mergeable_channels_gc_depth_buffer as i64);

    if depth_from_tip <= max_allowed_depth {
        return Ok(false);
    }

    // 3. Check if all validators have moved past this block
    let children = match dag.children(block_hash) {
        Some(children_set) => children_set,
        None => return Ok(false), // No children means no one can have moved past
    };

    if children.is_empty() {
        return Ok(false);
    }

    let latest_message_hashes = dag.latest_message_hashes();

    // For each validator's latest message, check if it's a descendant of any child (via main chain)
    for entry in latest_message_hashes.iter() {
        let latest_msg_hash = entry.value();

        if latest_msg_hash == block_hash {
            // Validator's latest is still this block
            return Ok(false);
        }

        // Check if latest message is descendant of any child (via main chain)
        let mut found_in_child_chain = false;
        for child_hash in children.iter() {
            let child_hash_ref = child_hash.key();
            if dag.is_in_main_chain(child_hash_ref, latest_msg_hash)? {
                found_in_child_chain = true;
                break;
            }
        }

        if !found_in_child_chain {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Get all finalized blocks from the DAG.
/// Note: This is a simple implementation that could be optimized.
fn get_finalized_blocks(dag: &KeyValueDagRepresentation) -> Result<Vec<BlockHash>, KvStoreError> {
    // Get all blocks via topo_sort and filter for finalized ones
    let all_blocks = dag.topo_sort(0, None)?;

    let finalized: Vec<BlockHash> = all_blocks
        .into_iter()
        .flatten()
        .filter(|hash| dag.is_finalized(hash))
        .collect();

    Ok(finalized)
}

#[cfg(test)]
mod tests {
    // Tests would go here
}
