// Slash-recovery coverage for the merge-rejected-slash re-issuance
// path in `block_creator::create`. Two tests:
//   * `slash_for_equivocator_survives_multi_parent_merge` — end-to-end:
//     equivocation, both honest validators slash, multi-parent merge,
//     slash effect lands in canonical post-state.
//   * `e1c_re_issues_merge_rejected_slash` — focused: a synthetic
//     `RejectedSlash` is injected into the parents-post-state cache so
//     `block_creator::create` exercises the re-issuance loop and emits
//     a SlashDeploy in the proposed block's body.

use casper::rust::casper::Casper;
use casper::rust::merging::rejected_slash::RejectedSlash;
use casper::rust::util::construct_deploy;
use casper::rust::util::rholang::runtime_manager::ParentsPostStateCacheKey;
use models::rust::casper::protocol::casper_message::{ProcessedSystemDeploy, SystemDeployData};

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::{GenesisBuilder, GenesisContext};

struct TestContext {
    genesis: GenesisContext,
    shard_id: String,
}

impl TestContext {
    async fn new() -> Self {
        let genesis = GenesisBuilder::new()
            .build_genesis_with_parameters(None)
            .await
            .expect("Failed to build genesis");
        let shard_id = genesis.genesis_block.shard_id.clone();
        Self { genesis, shard_id }
    }
}

#[tokio::test]
async fn slash_for_equivocator_survives_multi_parent_merge() {
    let ctx = TestContext::new().await;

    let mut nodes = TestNode::create_network(ctx.genesis.clone(), 3, None, None, None, None)
        .await
        .expect("create_network(3)");
    let equivocator_pk = nodes[0]
        .validator_id_opt
        .as_ref()
        .expect("node 0 has validator identity")
        .public_key
        .clone();
    let merge_proposer_pk = nodes[1]
        .validator_id_opt
        .as_ref()
        .expect("node 1 has validator identity")
        .public_key
        .clone();

    // Equivocation: a forged copy of node[0]'s block with a mutated
    // seq_num. Honest validators see this as InvalidBlockHash and queue
    // a slashing deploy for the equivocator.
    let deploy_data = construct_deploy::basic_deploy_data(0, None, Some(ctx.shard_id.clone()))
        .expect("build deploy");
    nodes[0]
        .casper
        .deploy(deploy_data)
        .expect("validator 0 deploy");
    let signed_block = nodes[0]
        .create_block_unsafe(&[])
        .await
        .expect("validator 0 creates signed_block");
    let invalid_block = {
        let mut b = signed_block.clone();
        b.seq_num = 47;
        b
    };
    nodes[1]
        .process_block(invalid_block.clone())
        .await
        .expect("node 1 processes invalid_block");
    nodes[2]
        .process_block(invalid_block.clone())
        .await
        .expect("node 2 processes invalid_block");

    // Each honest validator proposes a block containing its own
    // auto-emitted SlashDeploy via prepare_slashing_deploys.
    let deploy_data_a = construct_deploy::basic_deploy_data(1, None, Some(ctx.shard_id.clone()))
        .expect("build deploy a");
    nodes[1]
        .casper
        .deploy(deploy_data_a)
        .expect("validator 1 deploy");
    let block_1 = nodes[1]
        .create_block_unsafe(&[])
        .await
        .expect("validator 1 creates block_1");
    nodes[1]
        .process_block(block_1.clone())
        .await
        .expect("node 1 processes its own block_1");

    let deploy_data_b = construct_deploy::basic_deploy_data(2, None, Some(ctx.shard_id.clone()))
        .expect("build deploy b");
    nodes[2]
        .casper
        .deploy(deploy_data_b)
        .expect("validator 2 deploy");
    let block_2 = nodes[2]
        .create_block_unsafe(&[])
        .await
        .expect("validator 2 creates block_2");
    nodes[2]
        .process_block(block_2.clone())
        .await
        .expect("node 2 processes its own block_2");

    let slashes_in =
        |block: &models::rust::casper::protocol::casper_message::BlockMessage| -> Vec<prost::bytes::Bytes> {
            block
                .body
                .system_deploys
                .iter()
                .filter_map(|psd| match psd {
                    ProcessedSystemDeploy::Succeeded {
                        system_deploy:
                            SystemDeployData::Slash {
                                invalid_block_hash, ..
                            },
                        ..
                    } => Some(invalid_block_hash.clone()),
                    _ => None,
                })
                .collect()
        };
    assert!(
        slashes_in(&block_1).contains(&invalid_block.block_hash),
        "block_1 must contain a SlashDeploy for the equivocator's invalid_block"
    );
    assert!(
        slashes_in(&block_2).contains(&invalid_block.block_hash),
        "block_2 must contain a SlashDeploy for the equivocator's invalid_block"
    );

    // Sync block_2 into node 1 so the next propose can take both as parents.
    nodes[1]
        .process_block(block_2.clone())
        .await
        .expect("node 1 processes block_2");
    assert!(
        nodes[1].contains(&block_2.block_hash),
        "node 1 must observe block_2 after process_block"
    );

    // A fresh user deploy keeps create_block from short-circuiting
    // on NoNewDeploys when the merge proposer's own slash detection is
    // already covered by the merged parent state.
    let marker_deploy = construct_deploy::basic_deploy_data(3, None, Some(ctx.shard_id.clone()))
        .expect("build marker deploy");
    nodes[1]
        .casper
        .deploy(marker_deploy)
        .expect("validator 1 deploys marker");
    let merge_block = nodes[1]
        .create_block_unsafe(&[])
        .await
        .expect("validator 1 creates merge_block");

    let merge_parents: Vec<&prost::bytes::Bytes> =
        merge_block.header.parents_hash_list.iter().collect();
    assert!(
        merge_parents.iter().any(|h| **h == block_1.block_hash),
        "merge_block parents must include block_1"
    );
    assert!(
        merge_parents.iter().any(|h| **h == block_2.block_hash),
        "merge_block parents must include block_2"
    );

    // Post-merge bonds: equivocator must be at the bond floor
    // (<=1; tests currently use floor 0). Catches a regression where
    // the slash effect failed to land in canonical state through the
    // multi-parent merge.
    let post_merge_bonds = nodes[1]
        .runtime_manager
        .compute_bonds(&casper::rust::util::proto_util::post_state_hash(
            &merge_block,
        ))
        .await
        .expect("compute_bonds");
    let equivocator_stake = post_merge_bonds
        .iter()
        .find(|b| b.validator == equivocator_pk.bytes)
        .map(|b| b.stake)
        .expect("equivocator must still appear in bonds map");
    assert!(
        equivocator_stake <= 1,
        "post-merge equivocator stake must be at the bond floor (<=1); got {}",
        equivocator_stake
    );

    // Catches a regression where the slash hits the merge proposer
    // instead of (or in addition to) the equivocator.
    let proposer_stake = post_merge_bonds
        .iter()
        .find(|b| b.validator == merge_proposer_pk.bytes)
        .map(|b| b.stake)
        .expect("merge proposer must still appear in bonds map");
    let proposer_genesis_stake = ctx
        .genesis
        .genesis_block
        .body
        .state
        .bonds
        .iter()
        .find(|b| b.validator == merge_proposer_pk.bytes)
        .map(|b| b.stake)
        .expect("merge proposer must be bonded at genesis");
    assert_eq!(
        proposer_stake, proposer_genesis_stake,
        "merge proposer's stake must be unchanged after the merge"
    );
}

// Exercises the merge-rejected-slash recovery path
// (block_creator.rs:594-609). A synthetic `RejectedSlash` is written
// into the parents-post-state cache so the proposer's
// `compute_parents_post_state` call returns it as if the merge engine
// had rejected a slash chain. The synthetic entry uses a different
// `issuer_public_key` from any own-detected slash so `filter_recoverable`
// keeps it; the E1c loop then emits a SlashDeploy under the proposer's
// identity, which executes against an already-slashed PoS state and
// produces a Failed system-deploy entry in the proposed block's body.
//
// The proposer needs sibling parents (neither a descendant of the
// other) so `compute_parents_post_state` runs the cache-consulting
// merge path. The single-parent path and the descendant-fast-path both
// bypass the cache and skip E1c.
#[tokio::test]
async fn e1c_re_issues_merge_rejected_slash() {
    let ctx = TestContext::new().await;

    let mut nodes = TestNode::create_network(ctx.genesis.clone(), 3, None, None, None, None)
        .await
        .expect("create_network(3)");
    let alt_issuer_pk = nodes[2]
        .validator_id_opt
        .as_ref()
        .expect("node 2 has validator identity")
        .public_key
        .clone();

    // Forge an equivocation. node 1 processes the invalid block so
    // own-detection will emit a SlashDeploy under node 1's pk.
    let deploy_data = construct_deploy::basic_deploy_data(0, None, Some(ctx.shard_id.clone()))
        .expect("build deploy");
    nodes[0]
        .casper
        .deploy(deploy_data)
        .expect("validator 0 deploy");
    let signed_block = nodes[0]
        .create_block_unsafe(&[])
        .await
        .expect("validator 0 creates signed_block");
    let invalid_block = {
        let mut b = signed_block.clone();
        b.seq_num = 47;
        b
    };
    nodes[1]
        .process_block(invalid_block.clone())
        .await
        .expect("node 1 processes invalid_block");
    nodes[2]
        .process_block(invalid_block.clone())
        .await
        .expect("node 2 processes invalid_block");

    // Each honest validator proposes a sibling block at block_number=1
    // so the merge proposer's tip set contains two non-ancestor parents.
    // Without that, `compute_parents_post_state` skips the cache via
    // either the single-parent path or the descendant-fast-path.
    let deploy_a = construct_deploy::basic_deploy_data(1, None, Some(ctx.shard_id.clone()))
        .expect("build deploy a");
    nodes[1]
        .casper
        .deploy(deploy_a)
        .expect("validator 1 deploy a");
    let block_a = nodes[1]
        .create_block_unsafe(&[])
        .await
        .expect("validator 1 creates block_a");
    nodes[1]
        .process_block(block_a.clone())
        .await
        .expect("node 1 processes its own block_a");

    let deploy_b = construct_deploy::basic_deploy_data(2, None, Some(ctx.shard_id.clone()))
        .expect("build deploy b");
    nodes[2]
        .casper
        .deploy(deploy_b)
        .expect("validator 2 deploy b");
    let block_b = nodes[2]
        .create_block_unsafe(&[])
        .await
        .expect("validator 2 creates block_b");
    nodes[1]
        .process_block(block_b.clone())
        .await
        .expect("node 1 processes block_b");

    // Snapshot the proposer's view to derive the cache key the next
    // propose will compute. The merge runs once here so we obtain the
    // real merged pre-state and rejected-deploy list — overwriting the
    // cache entry then lets us augment rejected_slashes without
    // disturbing the rest of the merged-state computation.
    let snapshot = nodes[1].casper.get_snapshot().await.expect("get_snapshot");
    assert!(
        snapshot.parents.len() >= 2,
        "test setup requires multi-parent proposer view; got {} parent(s)",
        snapshot.parents.len()
    );
    let mut sorted_parent_hashes: Vec<prost::bytes::Bytes> = snapshot
        .parents
        .iter()
        .map(|p| p.block_hash.clone())
        .collect();
    sorted_parent_hashes.sort();
    let cache_key = ParentsPostStateCacheKey {
        sorted_parent_hashes,
        snapshot_lfb_hash: snapshot.last_finalized_block.clone(),
        disable_late_block_filtering: snapshot
            .on_chain_state
            .shard_conf
            .disable_late_block_filtering,
    };

    let (merged_state, merged_rejected, _) =
        casper::rust::util::rholang::interpreter_util::compute_parents_post_state(
            &nodes[1].block_store,
            snapshot.parents.clone(),
            &snapshot,
            &nodes[1].runtime_manager,
            None,
            Some(&nodes[1].rejected_deploy_buffer),
        )
        .expect("real merge to seed cache value");

    let synthetic = RejectedSlash {
        invalid_block_hash: invalid_block.block_hash.clone(),
        issuer_public_key: alt_issuer_pk,
        source_block_hash: invalid_block.block_hash.clone(),
    };
    nodes[1]
        .runtime_manager
        .put_cached_parents_post_state(cache_key, (merged_state, merged_rejected, vec![synthetic]));
    drop(snapshot);

    // Propose. A user deploy keeps `create_block` from short-circuiting
    // on `NoNewDeploys`; the slash entries are then driven by
    // own-detection plus the cache-injected RejectedSlash.
    let user_deploy = construct_deploy::basic_deploy_data(1, None, Some(ctx.shard_id.clone()))
        .expect("build user deploy");
    nodes[1]
        .casper
        .deploy(user_deploy)
        .expect("validator 1 deploys");
    let block = nodes[1]
        .create_block_unsafe(&[])
        .await
        .expect("validator 1 creates block");

    // Own-detection at the merge proposer is filtered out: parents.first()
    // post-state already shows the equivocator at bond floor, so
    // `prepare_slashing_deploys` returns an empty list. The single
    // SlashDeploy entry in the body therefore comes from the E1c
    // re-issuance loop driven by the cache-injected RejectedSlash. PoS's
    // slash entry-point is idempotent for already-slashed validators, so
    // the re-issued slash succeeds (returns true with no further state
    // change), producing a Succeeded entry in the body.
    let succeeded_slash_for_invalid_block = block
        .body
        .system_deploys
        .iter()
        .filter(|psd| {
            matches!(
                psd,
                ProcessedSystemDeploy::Succeeded {
                    system_deploy: SystemDeployData::Slash { invalid_block_hash, .. },
                    ..
                } if *invalid_block_hash == invalid_block.block_hash
            )
        })
        .count();
    assert_eq!(
        succeeded_slash_for_invalid_block, 1,
        "merge_block.body must contain exactly one Succeeded SlashDeploy \
         for invalid_block. The cache-injected RejectedSlash should \
         survive `filter_recoverable` (different issuer pk than any \
         own-detected slash) and reach the E1c re-issuance loop. Got {} \
         entries — if 0, the E1c loop in `block_creator::create` is not \
         emitting a SlashDeploy for cache-supplied RejectedSlashes.",
        succeeded_slash_for_invalid_block
    );
}

// Regression for the empty-block skip path. A heartbeat-disabled proposer
// (allow_empty_blocks=false, the production default) used to fast-fail on
// `NoNewDeploys` whenever it had no user deploys and no own-detected
// slashes — even when the parent merge had produced rejected slashes that
// only this proposer could re-issue. The fix moves the merge above the
// skip check so `recovered_rejected_slashes` can keep the proposer alive.
//
// Setup mirrors `e1c_re_issues_merge_rejected_slash` (cache-injected
// RejectedSlash, own-detection filtered by bond floor) but omits the
// keep-alive user deploy. Pre-fix this test would error on `NoNewDeploys`;
// post-fix the proposer must still emit the cache-supplied SlashDeploy.
#[tokio::test]
async fn rejected_slash_recovery_keeps_empty_proposer_alive() {
    let ctx = TestContext::new().await;

    let mut nodes = TestNode::create_network(ctx.genesis.clone(), 3, None, None, None, None)
        .await
        .expect("create_network(3)");
    let alt_issuer_pk = nodes[2]
        .validator_id_opt
        .as_ref()
        .expect("node 2 has validator identity")
        .public_key
        .clone();

    let deploy_data = construct_deploy::basic_deploy_data(0, None, Some(ctx.shard_id.clone()))
        .expect("build deploy");
    nodes[0]
        .casper
        .deploy(deploy_data)
        .expect("validator 0 deploy");
    let signed_block = nodes[0]
        .create_block_unsafe(&[])
        .await
        .expect("validator 0 creates signed_block");
    let invalid_block = {
        let mut b = signed_block.clone();
        b.seq_num = 47;
        b
    };
    nodes[1]
        .process_block(invalid_block.clone())
        .await
        .expect("node 1 processes invalid_block");
    nodes[2]
        .process_block(invalid_block.clone())
        .await
        .expect("node 2 processes invalid_block");

    let deploy_a = construct_deploy::basic_deploy_data(1, None, Some(ctx.shard_id.clone()))
        .expect("build deploy a");
    nodes[1]
        .casper
        .deploy(deploy_a)
        .expect("validator 1 deploy a");
    let block_a = nodes[1]
        .create_block_unsafe(&[])
        .await
        .expect("validator 1 creates block_a");
    nodes[1]
        .process_block(block_a.clone())
        .await
        .expect("node 1 processes its own block_a");

    let deploy_b = construct_deploy::basic_deploy_data(2, None, Some(ctx.shard_id.clone()))
        .expect("build deploy b");
    nodes[2]
        .casper
        .deploy(deploy_b)
        .expect("validator 2 deploy b");
    let block_b = nodes[2]
        .create_block_unsafe(&[])
        .await
        .expect("validator 2 creates block_b");
    nodes[1]
        .process_block(block_b.clone())
        .await
        .expect("node 1 processes block_b");

    let snapshot = nodes[1].casper.get_snapshot().await.expect("get_snapshot");
    assert!(
        snapshot.parents.len() >= 2,
        "test setup requires multi-parent proposer view; got {} parent(s)",
        snapshot.parents.len()
    );
    let mut sorted_parent_hashes: Vec<prost::bytes::Bytes> = snapshot
        .parents
        .iter()
        .map(|p| p.block_hash.clone())
        .collect();
    sorted_parent_hashes.sort();
    let cache_key = ParentsPostStateCacheKey {
        sorted_parent_hashes,
        snapshot_lfb_hash: snapshot.last_finalized_block.clone(),
        disable_late_block_filtering: snapshot
            .on_chain_state
            .shard_conf
            .disable_late_block_filtering,
    };

    let (merged_state, merged_rejected, _) =
        casper::rust::util::rholang::interpreter_util::compute_parents_post_state(
            &nodes[1].block_store,
            snapshot.parents.clone(),
            &snapshot,
            &nodes[1].runtime_manager,
            None,
            Some(&nodes[1].rejected_deploy_buffer),
        )
        .expect("real merge to seed cache value");

    let synthetic = RejectedSlash {
        invalid_block_hash: invalid_block.block_hash.clone(),
        issuer_public_key: alt_issuer_pk,
        source_block_hash: invalid_block.block_hash.clone(),
    };
    nodes[1]
        .runtime_manager
        .put_cached_parents_post_state(cache_key, (merged_state, merged_rejected, vec![synthetic]));
    drop(snapshot);

    // No user deploy. With allow_empty_blocks=false (TestNode default) and
    // own-detection filtered out by bond floor, the only thing keeping
    // the proposer alive is the cache-injected RejectedSlash flowing
    // through `recovered_rejected_slashes`. If the skip check is still
    // pre-merge, `create_block` returns NoNewDeploys and `create_block_unsafe`
    // errors here.
    let block = nodes[1].create_block_unsafe(&[]).await.expect(
        "validator 1 must propose a block even with no user deploys and no own-detected \
         slashes — a pending merge-rejected slash should keep the proposer alive. If this \
         fails with NoNewDeploys, the empty-block skip check is running before the merge \
         and dropping rejected-slash recovery.",
    );

    let succeeded_slash_for_invalid_block = block
        .body
        .system_deploys
        .iter()
        .filter(|psd| {
            matches!(
                psd,
                ProcessedSystemDeploy::Succeeded {
                    system_deploy: SystemDeployData::Slash { invalid_block_hash, .. },
                    ..
                } if *invalid_block_hash == invalid_block.block_hash
            )
        })
        .count();
    assert_eq!(
        succeeded_slash_for_invalid_block, 1,
        "block.body must contain exactly one Succeeded SlashDeploy for invalid_block. \
         Got {} entries — the skip check should have allowed the proposer through on \
         the strength of recovered_rejected_slashes alone.",
        succeeded_slash_for_invalid_block
    );
}
