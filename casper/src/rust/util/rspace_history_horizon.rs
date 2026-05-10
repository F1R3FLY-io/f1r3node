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
use models::rust::block_hash::BlockHash;
use models::rust::casper::protocol::casper_message::BlockMessage;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use shared::rust::store::key_value_store::KvStoreError;
use std::collections::HashSet;

use crate::rust::casper::CasperShardConf;

/// Result of walking the forward-horizon DAG window. Carries both the block
/// hashes (for storage-layers keyed by block — mergeable entries, block
/// metadata) and the rspace state roots (for the rspace history sync).
/// Single walk; both projections in one pass.
pub struct ForwardHorizon {
    /// Every block hash in the horizon, ordered LFB-side first (highest
    /// block_number first), deduped.
    pub block_hashes: Vec<BlockHash>,
    /// Every rspace state root referenced by horizon blocks (post-state and
    /// pre-state hashes), ordered LFB-side first, deduped. Within a single
    /// block the post-state precedes the pre-state — see module-level docs.
    pub roots: Vec<Blake2b256Hash>,
}

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
    Ok(compute_forward_horizon(dag, block_store, lfb, casper_shard_conf)?.roots)
}

/// Walk the forward-horizon DAG window and return both the block hashes
/// (for storage layers keyed by block — mergeable entries, block metadata)
/// and the rspace state roots (for rspace history sync) in a single pass.
/// Both projections are deduped and ordered LFB-side first.
///
/// Returns an empty `ForwardHorizon` if the LFB is at depth 0 (genesis) or
/// if the horizon would extend below genesis (clamped to height 0). Also
/// returns empty when `max_parent_depth == i32::MAX` (depth check disabled
/// — the unbounded horizon is a deliberate opt-out; operators can use
/// `disable-lfs = true` for full replay instead).
pub fn compute_forward_horizon(
    dag: &KeyValueDagRepresentation,
    block_store: &KeyValueBlockStore,
    lfb: &BlockMessage,
    casper_shard_conf: &CasperShardConf,
) -> Result<ForwardHorizon, KvStoreError> {
    let empty = ForwardHorizon {
        block_hashes: Vec::new(),
        roots: Vec::new(),
    };

    if casper_shard_conf.max_parent_depth == i32::MAX {
        return Ok(empty);
    }

    let lfb_height = lfb.body.state.block_number;
    let horizon_depth = (casper_shard_conf.max_parent_depth as i64)
        + (casper_shard_conf.mergeable_channels_gc_depth_buffer as i64);
    let min_height = std::cmp::max(0, lfb_height - horizon_depth);

    if min_height > lfb_height {
        return Ok(empty);
    }

    // topo_sort returns Vec<Vec<BlockHash>> — one inner vec per height,
    // covering all blocks at that height (main chain + side branches).
    // Ordering: ascending by height. Reverse to get LFB-side first.
    let layers = dag.topo_sort(min_height, Some(lfb_height))?;

    let mut block_hashes: Vec<BlockHash> = Vec::new();
    let mut block_seen: HashSet<BlockHash> = HashSet::new();
    let mut roots: Vec<Blake2b256Hash> = Vec::new();
    let mut root_seen: HashSet<Blake2b256Hash> = HashSet::new();
    for layer in layers.iter().rev() {
        for block_hash in layer {
            if !block_seen.insert(block_hash.clone()) {
                continue;
            }
            let block = match block_store.get(block_hash)? {
                Some(b) => b,
                None => {
                    tracing::warn!(
                        "compute_forward_horizon: block {} in DAG but missing from block_store",
                        models::rust::casper::pretty_printer::PrettyPrinter::build_string_bytes(
                            block_hash
                        )
                    );
                    continue;
                }
            };
            block_hashes.push(block_hash.clone());
            let post = Blake2b256Hash::from_bytes_prost(&block.body.state.post_state_hash);
            if root_seen.insert(post.clone()) {
                roots.push(post);
            }
            let pre = Blake2b256Hash::from_bytes_prost(&block.body.state.pre_state_hash);
            if root_seen.insert(pre.clone()) {
                roots.push(pre);
            }
        }
    }
    Ok(ForwardHorizon {
        block_hashes,
        roots,
    })
}

/// Compute the LFS lower-bound block number for joiner-side block download.
///
/// `lfs_block_requester::stream` clamps how far back from LFB it will accept
/// blocks during sync. The bound has two contributing constraints:
///
///   1. `deploy_lifespan` — blocks within this window of LFB carry deploys
///      that may still be reproposable (the original Scala-era reason).
///   2. `max_parent_depth + mergeable_channels_gc_depth_buffer` — the
///      forward-horizon: any block within this window of LFB may still be
///      legitimately referenced as a parent by an upcoming proposal, so the
///      joiner needs its DAG metadata even if it's older than
///      `deploy_lifespan` from LFB.
///
/// Take the lower of the two bounds (= older floor) so LFS coverage spans
/// both windows. The `i32::MAX` sentinel for `max_parent_depth` disables
/// the parent-depth check — in that mode the forward-horizon is unbounded
/// and `deploy_lifespan` alone determines the floor.
///
/// Negative results clamp to 0 (genesis) so the bound is always a valid
/// block number.
pub fn lfs_min_block_number(
    start_block_number: i64,
    deploy_lifespan: i64,
    max_parent_depth: i32,
    depth_buffer: i32,
) -> i64 {
    let lifespan_bound = std::cmp::max(0, start_block_number - deploy_lifespan);
    let horizon_bound = if max_parent_depth as i64 >= i32::MAX as i64 {
        // Sentinel: depth check disabled. Forward-horizon is unbounded,
        // so the lifespan bound alone determines the floor.
        lifespan_bound
    } else {
        std::cmp::max(
            0,
            start_block_number - (max_parent_depth as i64) - (depth_buffer as i64),
        )
    };
    std::cmp::min(lifespan_bound, horizon_bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_block_number_takes_lifespan_when_horizon_wider() {
        // start=200, lifespan=50 -> lifespan_bound=150
        // max_parent_depth=10, buffer=0 -> horizon_bound=190
        // min(150, 190) = 150 (lifespan tighter)
        assert_eq!(lfs_min_block_number(200, 50, 10, 0), 150);
    }

    #[test]
    fn min_block_number_takes_horizon_when_lifespan_wider() {
        // start=200, lifespan=50 -> lifespan_bound=150
        // max_parent_depth=100, buffer=10 -> horizon_bound=90
        // min(150, 90) = 90 (horizon tighter — this is the §2.14 case:
        // forward-horizon goes deeper than deploy_lifespan)
        assert_eq!(lfs_min_block_number(200, 50, 100, 10), 90);
    }

    #[test]
    fn min_block_number_clamps_to_zero_at_genesis() {
        // start=10, lifespan=50 -> max(0, 10-50) = 0
        assert_eq!(lfs_min_block_number(10, 50, 100, 10), 0);
    }

    #[test]
    fn min_block_number_clamps_horizon_to_zero() {
        // start=10, lifespan=5 -> lifespan_bound=5
        // max_parent_depth=100, buffer=10 -> horizon raw=10-110=-100 -> clamped 0
        // min(5, 0) = 0
        assert_eq!(lfs_min_block_number(10, 5, 100, 10), 0);
    }

    #[test]
    fn min_block_number_disabled_depth_falls_back_to_lifespan() {
        // max_parent_depth=i32::MAX (sentinel) → horizon_bound=lifespan_bound
        // The function returns lifespan_bound directly (no horizon clamping).
        assert_eq!(lfs_min_block_number(200, 50, i32::MAX, 10), 150);
        assert_eq!(lfs_min_block_number(200, 50, i32::MAX, 0), 150);
    }

    #[test]
    fn min_block_number_zero_buffer() {
        // No buffer → pure max_parent_depth.
        // start=300, lifespan=50, max_parent_depth=200 -> horizon=100
        // min(250, 100) = 100
        assert_eq!(lfs_min_block_number(300, 50, 200, 0), 100);
    }
}

// Integration coverage for the full `compute_forward_horizon_roots`
// reachability calc lives in `casper/tests/util/rspace_history_horizon_test.rs`
// (alongside the rest of the casper-test mod tree) where the test fixtures
// `with_storage` and `create_chain` are available.
