//! Forward-horizon rspace history reachability calculation.
//!
//! Used by joiner-side LFS sync to determine which rspace post-state roots
//! must be in the joiner's local roots store before transitioning to Running.
//! The calculation is the inverse of `mergeable_channels_gc::is_safe_to_delete`:
//! whereas the GC computes "what's safe to forget," this computes "what's
//! still reachable as a parent of an upcoming proposal."
//!
//! A joiner that has just LFS-synced to LFB at height N needs rspace state
//! for every block that an honest proposer could legitimately reference as
//! a parent: the proposer-side `Estimator::filterDeepParents` filter permits
//! any block within `max_parent_depth + depth_buffer` from the highest tip.
//! For a joiner whose highest tip is the LFB (just after sync), that means
//! every block in the DAG with `height ≥ LFB.height − (max_parent_depth +
//! depth_buffer)` — main chain AND side branches.
//!
//! The validator-side parent-depth check in `validate::parents` rejects any
//! block whose parents fall outside this same horizon, so the set of
//! potentially-validatable blocks is exactly bounded by it. There is no
//! out-of-horizon case to handle: horizon-internal blocks have their roots
//! synced by `sync_forward_horizon`, and out-of-horizon blocks are rejected
//! on consensus rules before validation queries the rspace history.
//!
//! For each in-horizon block we emit BOTH `pre_state_hash` and
//! `post_state_hash`. Single-parent blocks have `pre_state_hash =
//! parent.post_state_hash`, so this is mostly a redundant restatement of
//! a hash that's already collected from the parent. For multi-parent
//! blocks, however, `pre_state_hash` is the merge result computed by the
//! proposer's `dag_merger::merge` via `apply_trie_actions_fn` →
//! `do_checkpoint`'s `store_root`. That merge intermediate is recorded in
//! the proposer's roots_store but is NOT any block's post-state, so a
//! joiner that only collects post-states is left without it. When the
//! joiner later attempts to replay a child of one of those merged blocks
//! (or processes a parallel sibling), the spawn/reset path validates the
//! merge-result root against the joiner's roots_store and fires
//! `RootRepositoryDivergence` if absent. Including pre-states closes
//! that gap — peers serve the radix history for these merge results
//! because their `do_checkpoint` already wrote the trie nodes plus the
//! root tag for them.

use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use models::rust::casper::protocol::casper_message::BlockMessage;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use shared::rust::store::key_value_store::KvStoreError;
use std::collections::HashSet;

use crate::rust::casper::CasperShardConf;

/// Compute the set of rspace roots a joiner needs in its local roots store
/// before transitioning to Running. Walks every block in the DAG with
/// `height ≥ LFB.height − (max_parent_depth + depth_buffer)` and emits
/// both its `pre_state_hash` and `post_state_hash`, deduped, ordered by
/// descending block_number (LFB-side first — most likely to be referenced
/// as a parent by an early incoming block). Within a single block the
/// post-state is emitted before the pre-state so the joiner imports the
/// "outer" state before its merge intermediate.
///
/// Including pre-states is what fixes multi-parent merge intermediates:
/// these hashes only ever exist as the result of the proposer's
/// `dag_merger::merge` (recorded via `do_checkpoint`'s `store_root`), and
/// would otherwise never reach the joiner because they are not any block's
/// `post_state_hash`. See module-level docs for the full rationale.
///
/// Returns an empty vec if the LFB is at depth 0 (genesis) or if the
/// horizon would extend below genesis (clamped to height 0).
pub fn compute_forward_horizon_roots(
    dag: &KeyValueDagRepresentation,
    block_store: &KeyValueBlockStore,
    lfb: &BlockMessage,
    casper_shard_conf: &CasperShardConf,
) -> Result<Vec<Blake2b256Hash>, KvStoreError> {
    if casper_shard_conf.max_parent_depth == i32::MAX {
        // Depth check disabled — joiner can validate against any historical
        // block. Fall back to no horizon sync; caller can opt into full
        // replay (`disable-lfs = true`) instead.
        return Ok(Vec::new());
    }

    let lfb_height = lfb.body.state.block_number;
    let horizon_depth = (casper_shard_conf.max_parent_depth as i64)
        + (casper_shard_conf.mergeable_channels_gc_depth_buffer as i64);
    let min_height = std::cmp::max(0, lfb_height - horizon_depth);

    if min_height > lfb_height {
        return Ok(Vec::new());
    }

    // topo_sort returns Vec<Vec<BlockHash>> — one inner vec per height,
    // covering all blocks at that height (main chain + side branches).
    // Ordering: ascending by height. Reverse to get LFB-side first.
    let layers = dag.topo_sort(min_height, Some(lfb_height))?;

    let mut roots: Vec<Blake2b256Hash> = Vec::new();
    let mut seen: HashSet<Blake2b256Hash> = HashSet::new();
    for layer in layers.iter().rev() {
        for block_hash in layer {
            let block = match block_store.get(block_hash)? {
                Some(b) => b,
                None => {
                    tracing::warn!(
                        "compute_forward_horizon_roots: block {} in DAG but missing from block_store",
                        models::rust::casper::pretty_printer::PrettyPrinter::build_string_bytes(
                            block_hash
                        )
                    );
                    continue;
                }
            };
            let post = Blake2b256Hash::from_bytes_prost(&block.body.state.post_state_hash);
            if seen.insert(post.clone()) {
                roots.push(post);
            }
            let pre = Blake2b256Hash::from_bytes_prost(&block.body.state.pre_state_hash);
            if seen.insert(pre.clone()) {
                roots.push(pre);
            }
        }
    }
    Ok(roots)
}

// Unit tests for `compute_forward_horizon_roots` are in
// `casper/tests/util/rspace_history_horizon_test.rs` (alongside the rest of
// the casper-test mod tree) where the test fixtures `with_storage` and
// `create_chain` are available.
