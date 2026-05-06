// Reproduces the integration-suite `test_contract_lifecycle` failure mode at
// unit level: two bridge contracts deployed concurrently from different
// validators, then a multi-parent merge block. Under the original
// (pre-bitmask-tag) merge semantics, the second bridge's deploy is rejected
// at merge because the registry's TreeHashMap interior nodes conflict.
//
// This test captures the merge block's `body.rejected_deploys` field and
// asserts it is empty. With the bitmask-tag fix correctly engaged, the
// registry's interior-node bitmaps are OR-merged across chains and no
// rejection should occur. Without the fix, one bridge gets rejected and
// this test fails — that is the observable repro.
//
// Note on bridge `findOrCreate` regression: the bridge-v2.rho fixture in
// this branch also includes a `SystemVault.findOrCreate(bridgeVaultAddr)`
// call at bridge init time. That fix is required for the integration-level
// `test_multi_block_state_evolution` test to pass — without it, transfers
// to the bridge's vault have their `_deposit` send orphaned. This unit test
// does not directly exercise the lock flow (constructing a lock deploy and
// reading its deployId-channel response in TestNode is finicky); the
// integration test owns that regression guard.
//
// Diagnostic mode: run with `RUST_LOG=f1r3fly.merge.tag_check=trace` to
// observe whether `is_mergeable_channel` ever returns `Some(BitmaskOr)`
// during bridge deployment. Three possible outcomes:
//
//   (a) Test passes (no rejections), bitmask trace fires on registry channels
//       → bitmask fix is working end-to-end.
//   (b) Test fails (rejection observed), bitmask trace NEVER fires on
//       registry channels → bitmask tag binding isn't reaching Registry.rho
//       (URI binding issue, Par-equality issue, scope issue).
//   (c) Test fails (rejection observed), bitmask trace DOES fire → bitmask
//       fix engages on registry but the rejection is on a different shared
//       channel (vault, gas accumulator, etc.).

use casper::rust::util::construct_deploy;

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::{GenesisBuilder, GenesisContext};

struct TestContext {
    genesis: GenesisContext,
}

impl TestContext {
    async fn new() -> Self {
        let genesis = GenesisBuilder::new()
            .build_genesis_with_parameters(None)
            .await
            .unwrap();

        Self { genesis }
    }
}

/// Reads `bridge-v2.rho` from the test resources dir.
fn read_bridge_rho() -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/resources/bridge-v2.rho"),
    )
    .expect("read bridge-v2.rho")
}

/// Mirrors the integration-test scenario: two validators each deploy a bridge
/// concurrently (siblings off genesis), then a third validator proposes a
/// merge block that multi-parents over both. Asserts that no deploy gets
/// rejected at merge.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_concurrent_bridges_should_merge_without_rejection() {
    crate::init_logger();

    let ctx = TestContext::new().await;

    // 3-validator network. Default unlimited max-parents; no synchrony
    // constraint. validators[0] = bridge1 proposer, validators[1] = bridge2
    // proposer, validators[2] = merge proposer.
    let mut nodes = TestNode::create_network(ctx.genesis.clone(), 3, None, None, None, None)
        .await
        .unwrap();

    let bridge_rho = read_bridge_rho();
    let shard_id = ctx.genesis.genesis_block.shard_id.clone();

    // Two distinct bridge deploys signed by different genesis-funded keys.
    // Different timestamps avoid signature-collision edge cases.
    // Bridge deploys need a large phlo budget — they register 3 contracts via
    // insertArbitrary (each is a TreeHashMap insert) plus call findOrCreate
    // for the bridge's own vault. The integration test uses 500M; we match.
    let bridge1_deploy = {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        construct_deploy::source_deploy_now_full(
            bridge_rho.clone(),
            Some(500_000_000),
            None,
            Some(construct_deploy::DEFAULT_SEC.clone()),
            None,
            Some(shard_id.clone()),
        )
        .unwrap()
    };
    let bridge2_deploy = {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        construct_deploy::source_deploy_now_full(
            bridge_rho.clone(),
            Some(500_000_000),
            None,
            Some(construct_deploy::DEFAULT_SEC2.clone()),
            None,
            Some(shard_id.clone()),
        )
        .unwrap()
    };
    // Trigger deploy for the merge block. Without this, block_creator returns
    // NoNewDeploys and the multi-parent merge path doesn't fire.
    let trigger_deploy = {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        construct_deploy::source_deploy_now_full(
            "Nil".to_string(),
            None,
            None,
            Some(construct_deploy::DEFAULT_SEC.clone()),
            None,
            Some(shard_id.clone()),
        )
        .unwrap()
    };

    // Third sibling deploy so all three validators have a sibling block,
    // ensuring fork choice on the merge proposer multi-parents over them.
    let bridge3_deploy = {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        construct_deploy::source_deploy_now_full(
            "Nil".to_string(),
            None,
            None,
            Some(construct_deploy::DEFAULT_SEC2.clone()),
            None,
            Some(shard_id.clone()),
        )
        .unwrap()
    };

    let bridge1_sig = bridge1_deploy.sig.clone();
    let bridge2_sig = bridge2_deploy.sig.clone();

    // Phase 1: each validator proposes a sibling block off genesis without
    // sync between them — three concurrent siblings.
    let block1 = nodes[0]
        .add_block_from_deploys(&[bridge1_deploy.clone()])
        .await
        .expect("validator 0 propose bridge1");
    let block2 = nodes[1]
        .add_block_from_deploys(&[bridge2_deploy.clone()])
        .await
        .expect("validator 1 propose bridge2");
    let block3 = nodes[2]
        .add_block_from_deploys(&[bridge3_deploy.clone()])
        .await
        .expect("validator 2 propose third sibling");

    eprintln!(
        "PHASE 1 siblings: block1={} (bridge1), block2={} (bridge2), block3={} (third)",
        hex::encode(&block1.block_hash[..std::cmp::min(8, block1.block_hash.len())]),
        hex::encode(&block2.block_hash[..std::cmp::min(8, block2.block_hash.len())]),
        hex::encode(&block3.block_hash[..std::cmp::min(8, block3.block_hash.len())]),
    );

    // Phase 2: full pairwise sync so every node has both sibling blocks
    // in its DAG. After this, validators[2]'s next proposal will multi-parent
    // over both.
    for sender_idx in 0..3 {
        for receiver_idx in 0..3 {
            if sender_idx == receiver_idx {
                continue;
            }
            let (first_idx, second_idx) = if sender_idx < receiver_idx {
                (sender_idx, receiver_idx)
            } else {
                (receiver_idx, sender_idx)
            };
            let (left, right) = nodes.split_at_mut(second_idx);
            let first = &mut left[first_idx];
            let second = &mut right[0];
            first
                .sync_with_one(second)
                .await
                .expect("sync first → second");
        }
    }

    // Phase 3: validators[0] proposes a follow-up block. With its own sibling
    // and both peers' siblings visible, fork choice picks all three as
    // parents, exercising the multi-parent merge path. The merge block's
    // `rejected_deploys` field captures any deploys that were rejected during
    // the merge — that's the observable we assert against.
    let merge_block = nodes[0]
        .add_block_from_deploys(&[trigger_deploy.clone()])
        .await
        .expect("validator 0 propose merge block");

    eprintln!(
        "PHASE 3 merge block: hash={}, parents={}, rejected_deploys={}",
        hex::encode(&merge_block.block_hash[..std::cmp::min(8, merge_block.block_hash.len())]),
        merge_block.header.parents_hash_list.len(),
        merge_block.body.rejected_deploys.len(),
    );

    assert!(
        merge_block.header.parents_hash_list.len() >= 2,
        "Expected merge block to multi-parent over the two sibling proposals; got {} parents",
        merge_block.header.parents_hash_list.len(),
    );

    // The critical assertion. Without the bitmask-tag fix, one of bridge1
    // or bridge2 ends up here. With the fix correctly engaged on registry
    // interior-node channels, neither does.
    let rejected_sigs: Vec<Vec<u8>> = merge_block
        .body
        .rejected_deploys
        .iter()
        .map(|r| r.sig.to_vec())
        .collect();

    let bridge1_rejected = rejected_sigs.iter().any(|s| s == &bridge1_sig.to_vec());
    let bridge2_rejected = rejected_sigs.iter().any(|s| s == &bridge2_sig.to_vec());

    eprintln!(
        "REJECTION OBSERVED: bridge1_rejected={}, bridge2_rejected={}, total_rejected_deploys={}",
        bridge1_rejected,
        bridge2_rejected,
        rejected_sigs.len(),
    );

    assert!(
        !bridge1_rejected && !bridge2_rejected,
        "REPRO: bridge rejection observed at multi-parent merge. \
         bridge1_rejected={}, bridge2_rejected={}. \
         If `is_mergeable_channel` trace shows BitmaskOr firing on registry \
         channels, the conflict source is something OTHER than the registry's \
         TreeHashMap (case 2 — vault, gas, or other shared channel). \
         If the trace never shows BitmaskOr firing, the bitmask tag binding \
         isn't reaching Registry.rho (case 1 or 3 — URI scope or Par equality).",
        bridge1_rejected,
        bridge2_rejected,
    );

    // The bridge `findOrCreate` fix (bridge-v2.rho calls findOrCreate on its
    // own vault address at init) is exercised at integration level — see
    // `integration-tests/test/tests/shared/test_contract_lifecycle.py::
    // test_multi_block_state_evolution`. That test calls bridge.lock and
    // verifies the response chain completes (deployId is populated).
    //
    // A code-level equivalent that constructs a lock deploy and reads its
    // deployId data via `runtime_manager.get_data` was attempted but the
    // in-process TestNode's per-deploy data lookups don't expose deployId
    // data the way live gRPC `getDataAtName` does. Rather than fight that,
    // we let the integration test guard the lock-flow regression and keep
    // this code-level test scoped to merge behavior, which is what the
    // multi-node TestNode infrastructure is best at.
}
