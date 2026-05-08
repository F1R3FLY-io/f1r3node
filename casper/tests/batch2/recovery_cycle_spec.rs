// End-to-end regression test for the rejected-deploy recovery pipeline:
// a deploy that is conflict-rejected during a multi-parent merge lands
// in `KeyValueRejectedDeployBuffer` and is re-included in a subsequent
// proposer's body.
//
// Conflict generator: two deploys signed by the SAME funded key, each
// requesting a phlogiston precharge that would individually leave the
// shared vault solvent but together would drive the vault balance below
// zero. `conflict_set_merger::fold_rejection` rejects whichever branch
// it processes second to keep the merged state non-negative. The
// Rholang body is `Nil`, so play execution has no `|` parallel
// composition and is fully deterministic.

use casper::rust::util::construct_deploy;
use prost::bytes::Bytes;
use serial_test::serial;

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

/// Trivial deploy body. The conflict comes from the system-level
/// precharge against the source vault, not from anything in the Rholang.
const CONFLICT_RHO: &str = r#"
Nil
"#;

/// Phlogiston pricing per deploy. The actual REV drain on the source
/// vault is `cost * phlo_price` (precharge is `phlo_limit * phlo_price`,
/// refunded down to `cost * phlo_price`).
///
/// `phlo_limit = 8` keeps the precharge under the 9_000_000 REV vault
/// cap (`8 * 1_000_000 = 8_000_000`). The deploy's actual cost is ~5
/// phlo, so per-deploy net drain ≈ `5 * 1_000_000 = 5_000_000` REV. Two
/// such deploys against the same vault sum to `10_000_000`, exceeding
/// the 9_000_000 balance and triggering the merge-engine's
/// negative-balance rejection.
const PHLO_LIMIT: i64 = 8;
const PHLO_PRICE: i64 = 1_000_000;

/// Recovery cycle end-to-end.
///
/// DAG shape:
///
///         genesis
///         /     \
///     block_a   block_b      same-key deploys; block_a's deploy is the
///         \     /            larger-sig one and gets merge-rejected
///       merge_block          proposed by validator 1 (NOT validator 0)
///            |
///     recovery_block         proposed by validator 0; the rejected sig
///                            must surface in body.deploys
///
/// The flow exercises:
///   1. Multi-parent merge in `compute_parents_post_state`, where
///      `dag_merger::merge` returns the rejected sig and
///      `compute_rejected_buffer_admits` admits it to the buffer.
///   2. Buffer population on the recovery proposer via
///      `validate_block_checkpoint` when it syncs merge_block.
///   3. `prepare_user_deploys` pulling from the buffer and the
///      self-chain dedup filter exempting `rejected_in_scope` sigs so
///      the recovered deploy actually reaches `body.deploys`.
///
/// Determinism notes:
///
/// * Both deploys are signed by the same key (`DEFAULT_SEC`). At equal
///   cost/size the merge engine's tiebreak orders deploys via
///   `DeployChainIndex::Ord`, which compares sigs ascending. The
///   lex-LARGER sig is processed second by `fold_rejection` and gets
///   rejected.
///
/// * The larger-sig deploy is routed to `nodes[0]`'s block_a so the
///   rejected sig lives in validator 0's own previous block.
///
/// * Validator 0 must NOT propose merge_block. Validator 1 does. That
///   keeps validator 0's `latest_message_hash` at block_a, so when
///   validator 0 later creates recovery_block,
///   `collect_self_chain_deploy_sigs` walks `block_a → genesis` and
///   block_a's body deploys (including the rejected sig) always land
///   in `self_chain_deploy_sigs`. The hash-asc tiebreak that decides
///   merge_block's main parent is irrelevant — we never traverse
///   merge_block via the self-chain walk.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn recovery_cycle_rejected_deploy_is_buffered_and_re_proposed() {
    let ctx = TestContext::new().await;
    let shard_id = ctx.genesis.genesis_block.shard_id.clone();

    // Two validators, no synchrony constraint, unlimited parents so the
    // multi-parent merge actually happens.
    let mut nodes = TestNode::create_network(ctx.genesis.clone(), 2, None, None, None, None)
        .await
        .expect("create_network(2)");

    // Build the two conflicting deploys. Both are signed by the same
    // funded key; different timestamps (enforced by the sleeps) keep
    // their signatures distinct.
    let deploy_x = {
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        construct_deploy::source_deploy_now_full(
            CONFLICT_RHO.to_string(),
            Some(PHLO_LIMIT),
            Some(PHLO_PRICE),
            Some(construct_deploy::DEFAULT_SEC.clone()),
            None,
            Some(shard_id.clone()),
        )
        .expect("build deploy_x")
    };
    let deploy_y = {
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        construct_deploy::source_deploy_now_full(
            CONFLICT_RHO.to_string(),
            Some(PHLO_LIMIT),
            Some(PHLO_PRICE),
            Some(construct_deploy::DEFAULT_SEC.clone()),
            None,
            Some(shard_id.clone()),
        )
        .expect("build deploy_y")
    };

    // Route the lex-LARGER sig to deploy_a (validator 0's block) so
    // validator 0's own block contains the deploy that the merge engine
    // will reject.
    let (deploy_a, deploy_b) = if deploy_x.sig >= deploy_y.sig {
        (deploy_x, deploy_y)
    } else {
        (deploy_y, deploy_x)
    };
    let sig_a: Bytes = deploy_a.sig.clone();
    let sig_b: Bytes = deploy_b.sig.clone();
    assert!(
        sig_a > sig_b,
        "deploy_a must hold the lex-larger sig so the negative-balance \
         merge rejection picks validator 0's deploy"
    );

    // Sibling blocks: validator 0 proposes block_a, validator 1
    // proposes block_b. Neither has seen the other's block yet, so each
    // executes its deploy against the genesis post-state independently.
    let block_a = nodes[0]
        .add_block_from_deploys(&[deploy_a.clone()])
        .await
        .expect("validator 0 proposes block_a");
    let block_b = nodes[1]
        .add_block_from_deploys(&[deploy_b.clone()])
        .await
        .expect("validator 1 proposes block_b");
    assert_ne!(
        block_a.block_hash, block_b.block_hash,
        "block_a and block_b must be distinct sibling blocks"
    );

    // Sync both ways so each validator can include the other's block as
    // a parent in its next propose.
    {
        let (a, b) = nodes.split_at_mut(1);
        a[0].sync_with_one(&mut b[0]).await.expect("sync 0 -> 1");
    }
    {
        let (a, b) = nodes.split_at_mut(1);
        b[0].sync_with_one(&mut a[0]).await.expect("sync 1 -> 0");
    }
    assert!(
        nodes[0].contains(&block_b.block_hash),
        "validator 0 must observe block_b after sync"
    );
    assert!(
        nodes[1].contains(&block_a.block_hash),
        "validator 1 must observe block_a after sync"
    );

    // Validator 1 proposes merge_block. Validator 0 deliberately does
    // not propose it: keeping validator 0's latest at block_a is what
    // makes the recovery propose's self-chain walk deterministic.
    //
    // The marker deploy gives `create_block` something fresh to commit
    // so it doesn't short-circuit on `NoNewDeploys`.
    let marker_deploy = {
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        construct_deploy::basic_deploy_data(0, None, Some(shard_id.clone()))
            .expect("build marker_deploy")
    };
    let merge_block = nodes[1]
        .add_block_from_deploys(&[marker_deploy.clone()])
        .await
        .expect("validator 1 proposes merge_block over [block_a, block_b]");

    // The merge block must merge both branches. Inactive validators in
    // the bond set may also pin genesis as an additional parent, so we
    // assert presence of the two real chains rather than an exact count.
    assert!(
        merge_block.header.parents_hash_list.len() >= 2,
        "merge_block must merge at least 2 branches (got {} parents)",
        merge_block.header.parents_hash_list.len()
    );
    assert!(
        merge_block
            .header
            .parents_hash_list
            .iter()
            .any(|h| *h == block_a.block_hash),
        "merge_block parents must include block_a"
    );
    assert!(
        merge_block
            .header
            .parents_hash_list
            .iter()
            .any(|h| *h == block_b.block_hash),
        "merge_block parents must include block_b"
    );

    // The merge engine's negative-balance check must have rejected one
    // of the two deploys, and it must be deploy_a (the lex-larger sig).
    let rejected_sigs: Vec<Bytes> = merge_block
        .body
        .rejected_deploys
        .iter()
        .map(|rd| rd.sig.clone())
        .collect();
    assert!(
        !rejected_sigs.is_empty(),
        "merge_block.body.rejected_deploys must be non-empty — combined \
         precharge from two same-key deploys must drive the source vault \
         balance below zero, which `conflict_set_merger::fold_rejection` \
         catches by rejecting the second branch"
    );
    let conflict_sig = rejected_sigs
        .iter()
        .find(|s| **s == sig_a || **s == sig_b)
        .cloned()
        .expect("the rejected sig must be one of the two conflicting deploys");
    assert_eq!(
        conflict_sig,
        sig_a,
        "the rejected sig must be deploy_a's (the lex-larger sig that \
         `fold_rejection` processes second). Got rejected sigs={:?}, \
         sig_a={}, sig_b={}",
        rejected_sigs.iter().map(hex::encode).collect::<Vec<_>>(),
        hex::encode(&sig_a),
        hex::encode(&sig_b)
    );
    let surviving_sig = sig_b.clone();

    // Sync merge_block from validator 1 back to validator 0. The
    // receive-side `validate_block_checkpoint` runs
    // `compute_parents_post_state` with the buffer arg, which populates
    // validator 0's own `KeyValueRejectedDeployBuffer`. The recovery
    // proposer's snapshot BFS then sees merge_block's `rejected_deploys`
    // and populates `rejected_in_scope`.
    {
        let (a, b) = nodes.split_at_mut(1);
        a[0].sync_with_one(&mut b[0])
            .await
            .expect("sync merge_block 1 -> 0");
    }
    assert!(
        nodes[0].contains(&merge_block.block_hash),
        "validator 0 must observe merge_block before recovery propose"
    );

    // Validator 0's buffer must contain the rejected sig after sync.
    {
        let buffer_guard = nodes[0].rejected_deploy_buffer.lock().expect("buffer lock");
        let contains_rejected = buffer_guard
            .contains_sig(&conflict_sig)
            .expect("buffer.contains_sig");
        assert!(
            contains_rejected,
            "validator 0's buffer must contain the rejected sig {} after \
             syncing merge_block",
            hex::encode(&conflict_sig)
        );
    }

    // Drive the recovery: validator 0 proposes recovery_block.
    // `collect_self_chain_deploy_sigs` walks validator 0's prior latest
    // (block_a) and finds deploy_a's sig there. Without the
    // `rejected_in_scope` exemption, the self-chain dedup filter would
    // drop the recovered sig from `prepared.deploys` and the new block's
    // body would be missing the recovered deploy.
    let marker_deploy_2 = {
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        construct_deploy::basic_deploy_data(1, None, Some(shard_id.clone()))
            .expect("build marker_deploy_2")
    };
    let recovery_block = nodes[0]
        .add_block_from_deploys(&[marker_deploy_2.clone()])
        .await
        .expect("validator 0 proposes recovery_block");

    let recovery_sigs: Vec<&Bytes> = recovery_block
        .body
        .deploys
        .iter()
        .map(|pd| &pd.deploy.sig)
        .collect();
    assert!(
        recovery_sigs.iter().any(|s| **s == conflict_sig),
        "recovery_block.body.deploys must contain the recovered sig {} \
         (pulled from the rejected-deploy buffer); got body.deploys \
         sigs = {:?}. If this fires, check that both `prepare_user_deploys` \
         and `collect_self_chain_deploy_sigs` exempt `rejected_in_scope` \
         sigs from their in-scope dedup filters",
        hex::encode(&conflict_sig),
        recovery_sigs
            .iter()
            .map(|s| hex::encode(s.as_ref()))
            .collect::<Vec<_>>()
    );

    // The surviving sig must remain reachable in the canonical view via
    // the deploy index, pointing back to its pre-merge block.
    assert!(
        nodes[0]
            .block_dag_storage
            .get_representation()
            .lookup_by_deploy_id(&surviving_sig.to_vec())
            .ok()
            .flatten()
            .is_some(),
        "the surviving sig {} must be reachable in the canonical view via \
         the deploy index",
        hex::encode(&surviving_sig)
    );
}
