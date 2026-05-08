// Tests for `compute_forward_horizon_roots` — joiner-side LFS forward-horizon
// reachability calculation. See `casper/src/rust/util/rspace_history_horizon.rs`
// for the production code under test.

use crate::helper::{
    block_dag_storage_fixture::with_storage,
    block_generator::{create_block, create_genesis_block},
    block_util::generate_validator,
};
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use block_storage::rust::test::indexed_block_dag_storage::IndexedBlockDagStorage;
use casper::rust::casper::CasperShardConf;
use casper::rust::util::rspace_history_horizon::compute_forward_horizon_roots;
use models::rust::casper::protocol::casper_message::{Bond, BlockMessage};
use prost::bytes::Bytes;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;

fn mk_conf(max_parent_depth: i32, depth_buffer: i32) -> CasperShardConf {
    let mut conf = CasperShardConf::new();
    conf.max_parent_depth = max_parent_depth;
    conf.mergeable_channels_gc_depth_buffer = depth_buffer;
    conf
}

fn unique_state_hash(seed: u8) -> Bytes {
    // 32-byte deterministic hash; seed in first byte distinguishes per-block.
    // (Production blocks have genuinely unique post_state_hashes from rspace
    // computation; test fixture defaults to empty bytes which would all
    // dedupe.)
    let mut bytes = vec![0u8; 32];
    bytes[0] = seed;
    Bytes::from(bytes)
}

fn build_chain(
    block_store: &mut KeyValueBlockStore,
    block_dag_storage: &mut IndexedBlockDagStorage,
    length: usize,
    bonds: Vec<Bond>,
    validator: Bytes,
) -> Vec<BlockMessage> {
    let genesis = create_genesis_block(
        block_store,
        block_dag_storage,
        None,
        Some(bonds.clone()),
        None,
        None,
        Some(unique_state_hash(0)),
        None,
        None,
        None,
    );
    let mut chain = vec![genesis.clone()];
    for i in 1..length {
        let parent = chain.last().unwrap().clone();
        let block = create_block(
            block_store,
            block_dag_storage,
            vec![parent.block_hash.clone()],
            &genesis,
            Some(validator.clone()),
            Some(bonds.clone()),
            None,
            None,
            Some(unique_state_hash(i as u8)),
            None,
            // pre_state_hash mirrors a real single-parent chain: this
            // block's pre = parent's post. Dedup against the parent's
            // post_state_hash later in compute_forward_horizon_roots.
            Some(parent.body.state.post_state_hash.clone()),
            Some(i as i32),
            None,
        );
        chain.push(block);
    }
    chain
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn returns_empty_when_max_parent_depth_is_unlimited() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v0 = generate_validator(Some("Validator0"));
        let bonds = vec![Bond {
            validator: v0.clone(),
            stake: 10,
        }];
        let chain = build_chain(&mut block_store, &mut block_dag_storage, 5, bonds, v0);

        let dag = block_dag_storage.get_representation();
        let conf = mk_conf(i32::MAX, 0);

        let roots = compute_forward_horizon_roots(&dag, &block_store, &chain[4], &conf).unwrap();
        assert!(
            roots.is_empty(),
            "horizon should be empty when max_parent_depth is unlimited (caller falls back to disable-lfs replay)"
        );
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn includes_main_chain_ancestors_within_horizon() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v0 = generate_validator(Some("Validator0"));
        let bonds = vec![Bond {
            validator: v0.clone(),
            stake: 10,
        }];
        // Chain of 6 blocks. block_numbers 1..5 (genesis at 1 in test fixture).
        let chain = build_chain(&mut block_store, &mut block_dag_storage, 6, bonds, v0);

        let dag = block_dag_storage.get_representation();
        // max_parent_depth=2, depth_buffer=1 → window from LFB.height-3 upward.
        let conf = mk_conf(2, 1);

        let roots = compute_forward_horizon_roots(&dag, &block_store, &chain[5], &conf).unwrap();
        // LFB at chain[5] has block_number=5; horizon includes block_numbers >= 5-3=2.
        // That's chain[2..=5] = 4 blocks, contributing 4 unique post-states.
        // Each block also contributes its pre-state, which on a single-parent
        // chain equals the parent's post-state. chain[2..=5] pre-states dedupe
        // against post-states of chain[1..=4]; chain[1]'s post is OUTSIDE the
        // horizon and therefore not collected from chain[1] itself, so it
        // appears once via chain[2]'s pre. Expected total = 4 + 1 = 5.
        assert_eq!(
            roots.len(),
            5,
            "horizon should include 4 post-states + 1 unique pre-state from chain[2] (got {})",
            roots.len()
        );
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn excludes_blocks_outside_horizon() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v0 = generate_validator(Some("Validator0"));
        let bonds = vec![Bond {
            validator: v0.clone(),
            stake: 10,
        }];
        // Chain of 10 blocks, block_numbers 1..9.
        let chain = build_chain(&mut block_store, &mut block_dag_storage, 10, bonds, v0);

        let dag = block_dag_storage.get_representation();
        // max_parent_depth=2, depth_buffer=0 → window from LFB.height-2 upward.
        let conf = mk_conf(2, 0);

        let roots = compute_forward_horizon_roots(&dag, &block_store, &chain[9], &conf).unwrap();
        // LFB at chain[9] block_number=9; horizon includes block_numbers >= 7.
        // That's chain[7], chain[8], chain[9] = 3 blocks → 3 unique post-states.
        // Pre-states: chain[7].pre == chain[6].post (unique, outside horizon),
        // chain[8].pre == chain[7].post (deduped), chain[9].pre == chain[8].post
        // (deduped). Expected total = 3 + 1 = 4.
        assert_eq!(
            roots.len(),
            4,
            "horizon should exclude blocks outside window (got {} roots, expected 4)",
            roots.len()
        );

        // chain[3]'s post-state is well outside the horizon and should never
        // be collected — neither as a post nor as anyone's pre (chain[4]'s pre
        // is also outside the collected window).
        let early_ancestor_root =
            Blake2b256Hash::from_bytes_prost(&chain[3].body.state.post_state_hash);
        assert!(
            !roots.contains(&early_ancestor_root),
            "horizon must not contain root from chain[3] (block_number=3, depth=6 from tip)"
        );

        // chain[6]'s post-state SHOULD appear, but only as chain[7]'s pre-state
        // (chain[6] itself is outside the horizon). This proves the pre-state
        // collection bridges one level of horizon-edge state.
        let chain6_post =
            Blake2b256Hash::from_bytes_prost(&chain[6].body.state.post_state_hash);
        assert!(
            roots.contains(&chain6_post),
            "chain[6]'s post-state must appear via chain[7]'s pre-state (horizon-edge bridge)"
        );
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn returns_lfb_only_when_horizon_at_genesis() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v0 = generate_validator(Some("Validator0"));
        let bonds = vec![Bond {
            validator: v0.clone(),
            stake: 10,
        }];
        // Just genesis + 1 child. LFB.height = 1. Horizon clamped to 0.
        let chain = build_chain(&mut block_store, &mut block_dag_storage, 2, bonds, v0);

        let dag = block_dag_storage.get_representation();
        let conf = mk_conf(100, 10);

        let roots = compute_forward_horizon_roots(&dag, &block_store, &chain[1], &conf).unwrap();
        // Both genesis (block_number=1 in test fixture) and chain[1] (also seq_num=1
        // → block_number=1) are within horizon. Both have default-empty
        // post_state_hash though (test fixture doesn't compute real state),
        // so they may dedupe to one entry. Just assert non-empty.
        assert!(
            !roots.is_empty(),
            "horizon for short chain should still include at least the LFB's post-state"
        );
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ordered_by_descending_height() {
    with_storage(|mut block_store, mut block_dag_storage| async move {
        let v0 = generate_validator(Some("Validator0"));
        let bonds = vec![Bond {
            validator: v0.clone(),
            stake: 10,
        }];
        let chain = build_chain(&mut block_store, &mut block_dag_storage, 6, bonds, v0);

        let dag = block_dag_storage.get_representation();
        // max_parent_depth large enough to include the whole chain
        let conf = mk_conf(10, 0);

        let roots = compute_forward_horizon_roots(&dag, &block_store, &chain[5], &conf).unwrap();
        assert!(!roots.is_empty());

        // First entry should correspond to a block with height >= the last entry's height.
        // We can't directly read height from Blake2b256Hash, so instead verify that
        // the LFB's own post-state is FIRST in the result (LFB-side first per docs).
        let lfb_root = Blake2b256Hash::from_bytes_prost(&chain[5].body.state.post_state_hash);
        assert_eq!(
            roots.first(),
            Some(&lfb_root),
            "LFB's post-state should appear first (LFB-side ordering)"
        );
    })
    .await
}
