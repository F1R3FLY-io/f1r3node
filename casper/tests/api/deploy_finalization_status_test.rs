// Proof tests for BlockAPI::deploy_finalization_status. Covers the
// API-surface states that can be triggered with the existing
// single-node TestNode fixture.

use std::collections::HashMap;
use std::sync::Arc;

use casper::rust::api::block_api::BlockAPI;
use casper::rust::api::deploy_finalization_status::{self, DeployFinalizationState};
use casper::rust::casper::MultiParentCasper;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::engine::engine_with_casper::EngineWithCasper;
use casper::rust::multi_parent_casper_impl::MultiParentCasperImpl;
use crypto::rust::public_key::PublicKey;

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::{GenesisBuilder, GenesisContext};

struct TestContext {
    genesis: GenesisContext,
}

impl TestContext {
    async fn new() -> Self {
        fn bonds_function(validators: Vec<PublicKey>) -> HashMap<PublicKey, i64> {
            validators
                .into_iter()
                .zip(vec![10i64, 10i64, 10i64])
                .collect()
        }

        let parameters =
            GenesisBuilder::build_genesis_parameters_with_defaults(Some(bonds_function), None);
        let genesis = GenesisBuilder::new()
            .build_genesis_with_parameters(Some(parameters))
            .await
            .expect("Failed to build genesis");

        Self { genesis }
    }
}

async fn create_engine_cell(node: &TestNode) -> EngineCell {
    let casper_for_engine = Arc::new(MultiParentCasperImpl {
        block_retriever: node.casper.block_retriever.clone(),
        event_publisher: node.casper.event_publisher.clone(),
        runtime_manager: node.casper.runtime_manager.clone(),
        estimator: node.casper.estimator.clone(),
        block_store: node.casper.block_store.clone(),
        block_dag_storage: node.casper.block_dag_storage.clone(),
        deploy_storage: node.casper.deploy_storage.clone(),
        rejected_deploy_buffer: node.casper.rejected_deploy_buffer.clone(),
        casper_buffer_storage: node.casper.casper_buffer_storage.clone(),
        validator_id: node.casper.validator_id.clone(),
        casper_shard_conf: node.casper.casper_shard_conf.clone(),
        approved_block: node.casper.approved_block.clone(),
        finalization_in_progress: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        finalizer_task_in_progress: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        finalizer_task_queued: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heartbeat_signal_ref: casper::rust::heartbeat_signal::new_heartbeat_signal_ref(),
        deploys_in_scope_cache: std::sync::Arc::new(std::sync::Mutex::new(None)),
        active_validators_cache: std::sync::Arc::new(tokio::sync::Mutex::new(
            std::collections::HashMap::new(),
        )),
    });
    let engine = EngineWithCasper::new(casper_for_engine);
    let engine_cell = EngineCell::init();
    engine_cell.set(Arc::new(engine)).await;
    engine_cell
}

/// A sig never seen anywhere in the DAG returns the "unknown" pending state:
/// no rejection count, no latest block hash. Regression guard for the most
/// common polling case (client polls right after deploy submission).
#[tokio::test]
async fn unknown_sig_returns_pending_with_empty_fields() {
    let ctx = TestContext::new().await;
    let nodes = TestNode::create_network(ctx.genesis.clone(), 1, None, None, None, None)
        .await
        .unwrap();
    let engine_cell = create_engine_cell(&nodes[0]).await;

    let unknown_sig = vec![0xAA; 32];
    let status = BlockAPI::deploy_finalization_status(&engine_cell, &unknown_sig)
        .await
        .expect("resolver should not fail");

    assert_eq!(status.state, DeployFinalizationState::Pending);
    assert_eq!(status.rejection_count, 0);
    assert!(
        status.latest_block_hash.is_none(),
        "unknown sig must have no latest_block_hash, got {:?}",
        status.latest_block_hash
    );
}

/// Calls the pure `resolve` function directly (bypassing the async
/// `BlockAPI` wrapper) to confirm it is callable from non-engine-cell
/// contexts. This path is what the catchup gate in
/// `compute_parents_post_state` uses — the gate is not invoked in this
/// single-node test, but the pure-function signature contract is.
#[tokio::test]
async fn resolve_pure_function_returns_pending_for_unknown_sig() {
    let ctx = TestContext::new().await;
    let nodes = TestNode::create_network(ctx.genesis.clone(), 1, None, None, None, None)
        .await
        .unwrap();

    let dag = nodes[0]
        .casper
        .block_dag()
        .await
        .expect("fetch dag representation");
    let block_store = nodes[0].casper.block_store();
    let deploy_lifespan = nodes[0].casper.casper_shard_conf().deploy_lifespan;

    let unknown_sig = vec![0xBB; 32];
    let status =
        deploy_finalization_status::resolve(&dag, block_store, deploy_lifespan, &unknown_sig)
            .expect("resolve should not fail for unknown sig");

    assert_eq!(status.state, DeployFinalizationState::Pending);
    assert_eq!(status.rejection_count, 0);
    assert!(status.latest_block_hash.is_none());
}

/// Regression test for the resolver's multi-parent DAG coverage.
///
/// Builds a minimal multi-parent DAG:
///
/// ```text
///     genesis (h=0)
///       |   |
///       A   B       both at h=1, children of genesis
///       |   |
///        \ /
///         C         at h=2, parents=[A, B] with A as main-parent; LFB
/// ```
///
/// The deploy sig under test lives only in `B.body.deploys`. B reaches
/// canonical state via C's secondary-parent slot, not via the main-parent
/// chain from C.
///
/// A main-parent-only walk (`dag.main_parent_chain(C, _)`) visits
/// `C → A → genesis` and never touches B, so it misses the sig and the
/// resolver reports `Pending`. A BFS over all parents visits B through C's
/// secondary slot, finds the sig in `body.deploys`, and reports `Finalized`.
///
/// This test exists to keep the BFS semantics (over `parents_hash_list`, not
/// just `main_parent`) locked in.
#[tokio::test]
async fn resolve_finds_sig_in_secondary_parent_branch() {
    use crate::util::rholang::resources::{
        block_dag_storage_from_dyn, mk_test_rnode_store_manager_from_genesis,
    };
    use block_storage::rust::key_value_block_store::KeyValueBlockStore;
    use casper::rust::util::construct_deploy;
    use models::rust::block_implicits;
    use models::rust::casper::protocol::casper_message::ProcessedDeploy;

    let ctx = TestContext::new().await;
    let genesis_block = ctx.genesis.genesis_block.clone();
    let genesis_hash = genesis_block.block_hash.clone();

    let mut kvm = mk_test_rnode_store_manager_from_genesis(&ctx.genesis);
    let block_store = KeyValueBlockStore::create_from_kvm(&mut *kvm)
        .await
        .expect("block store");
    let dag_storage = block_dag_storage_from_dyn(&mut *kvm)
        .await
        .expect("dag storage");

    block_store
        .put_block_message(&genesis_block)
        .expect("store genesis");
    dag_storage
        .insert(&genesis_block, false, true)
        .expect("dag genesis");

    let deploy_b =
        construct_deploy::source_deploy_now_full("Nil".to_string(), None, None, None, None, None)
            .expect("construct deploy_b");
    let deploy_b_sig = deploy_b.sig.to_vec();

    // Block A: empty-body sibling of genesis at h=1.
    let block_a = block_implicits::get_random_block(
        Some(1),
        Some(1),
        None,
        None,
        None,
        None,
        Some(0),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(genesis_block.body.state.bonds.clone()),
        Some(genesis_block.shard_id.clone()),
        None,
    );
    // Block B: sibling of A at h=1, carries deploy_b in body.deploys.
    let block_b = block_implicits::get_random_block(
        Some(1),
        Some(2),
        None,
        None,
        None,
        None,
        Some(0),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(vec![ProcessedDeploy::empty(deploy_b)]),
        Some(Vec::new()),
        Some(genesis_block.body.state.bonds.clone()),
        Some(genesis_block.shard_id.clone()),
        None,
    );
    // Block C: merge of [A, B] with A as main parent.
    let block_c = block_implicits::get_random_block(
        Some(2),
        Some(1),
        None,
        None,
        None,
        None,
        Some(0),
        Some(vec![block_a.block_hash.clone(), block_b.block_hash.clone()]),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(genesis_block.body.state.bonds.clone()),
        Some(genesis_block.shard_id.clone()),
        None,
    );

    block_store.put_block_message(&block_a).expect("store A");
    block_store.put_block_message(&block_b).expect("store B");
    block_store.put_block_message(&block_c).expect("store C");
    dag_storage
        .insert(&block_a, false, false)
        .expect("dag insert A");
    dag_storage
        .insert(&block_b, false, false)
        .expect("dag insert B");
    dag_storage
        .insert(&block_c, false, false)
        .expect("dag insert C");

    // Promote C to LFB so the resolver's scan starts there. The DAG state
    // normally bumps LFB only via the finalization pipeline; for this unit
    // test we overwrite the representation's field directly.
    let mut dag = dag_storage.get_representation();
    dag.last_finalized_block_hash = block_c.block_hash.clone();

    let deploy_lifespan = 50i64;
    let status =
        deploy_finalization_status::resolve(&dag, &block_store, deploy_lifespan, &deploy_b_sig)
            .expect("resolve should not fail");

    assert_eq!(
        status.state,
        DeployFinalizationState::Finalized,
        "sig in secondary-parent ancestor of LFB should be Finalized; got {:?}",
        status.state
    );
    assert_eq!(
        status.latest_block_hash.as_ref(),
        Some(&block_b.block_hash),
        "latest_block_hash must point at B (the block actually containing the sig)"
    );
    assert_eq!(status.rejection_count, 0);
}

/// Parity regression for the `resolve` / `resolve_batch` refactor on a
/// **production-shape multi-parent DAG**.
///
/// `resolve_batch` lifts the per-sig BFS state into a struct so a single
/// canonical-chain walk can update many sigs at once (one BFS per merge
/// step instead of one per rejected deploy).
/// Both entry points share `run_prelude`, `bfs_finalized_window`, and
/// `finalize_sig_state`; this test ensures the *external* contract
/// stays the same — for every sig, `resolve(sig)` returns the same
/// `DeployFinalizationStatus` as `resolve_batch(set)[sig]`.
///
/// Single-chain DAGs miss the multi-parent BFS branches that are the
/// entire reason `bfs_finalized_window` walks `parents_hash_list`
/// instead of just main-parents. The production f1r3fly shard
/// produces multi-parent merges constantly, and the resolver's
/// canonical-descendant invalidation rule has different outcomes in
/// multi-parent vs. single-chain topology.
///
/// DAG shape:
///
/// ```text
///         genesis (h=0)
///            |
///            A             (h=1, parent=genesis) — CANONICAL
///           / \
///          B   S           (both h=2, both have main_parent=A)
///           \ /            B is on the canonical main-parent chain of LFB.
///            C             S is a non-canonical sibling of B, reachable
///         (LFB, h=3)       from C only via C's secondary-parent slot.
/// ```
///
/// `is_in_main_chain` semantics (verified in
/// `block_dag_key_value_storage.rs::is_in_main_chain`): walks main
/// parents from `descendant` down to `ancestor`'s height; returns true
/// iff that walk lands on `ancestor`. For this DAG:
///
/// - `is_in_main_chain(A, B) = true`  (B's main parent is A)
/// - `is_in_main_chain(A, S) = true`  (S's main parent is also A — even
///                                    though S itself is non-canonical)
/// - `is_in_main_chain(B, C) = true`  (B is on LFB's main-parent chain)
/// - `is_in_main_chain(S, C) = false` (S is not on LFB's main-parent
///                                    chain — C's main parent is B)
///
/// Coverage matrix:
///
/// | Sig                                | Body location                                | Expected state |
/// |------------------------------------|----------------------------------------------|----------------|
/// | `clean_via_secondary`              | only in S.body.deploys (secondary-parent)    | `Finalized`    |
/// | `failed_canonical`                 | A.body.deploys with `is_failed=true`         | `Failed`       |
/// | `clean_canonical_reject_canonical` | clean in A, rejected in B (canonical desc.)  | `Pending`      |
/// | `clean_canonical_reject_sibling`   | clean in A, rejected in S (non-canonical)    | `Finalized`    |
/// | `unknown`                          | not in DAG                                   | `Pending` (unknown) |
///
/// **Regression guard for canonical-descendant invalidation.**
/// `clean_canonical_reject_sibling` is the case that distinguishes the
/// resolver's *intended* semantics ("rejection invalidates clean only
/// when the rejection is on the canonical chain") from a buggier
/// reading ("rejection invalidates clean when the rejection's main
/// parent walk lands on the clean block"). Under the latter,
/// `is_in_main_chain(A, S) = true` would invalidate clean inclusion at
/// A even though S itself is a non-canonical sibling whose effects are
/// not in canonical post-state. The correct resolver behavior is to
/// require both:
///   1. clean is in reject's main-parent ancestry, AND
///   2. reject is itself on the LFB's main-parent chain.
///
/// This test exercises that distinction.
#[tokio::test]
async fn resolve_and_resolve_batch_agree_across_states() {
    use crate::util::rholang::resources::{
        block_dag_storage_from_dyn, mk_test_rnode_store_manager_from_genesis,
    };
    use block_storage::rust::key_value_block_store::KeyValueBlockStore;
    use casper::rust::api::deploy_finalization_status::resolve_batch;
    use casper::rust::util::construct_deploy;
    use models::rust::block_implicits;
    use models::rust::casper::protocol::casper_message::{ProcessedDeploy, RejectedDeploy};
    use prost::bytes::Bytes;
    use std::collections::HashSet;

    let ctx = TestContext::new().await;
    let genesis_block = ctx.genesis.genesis_block.clone();
    let genesis_hash = genesis_block.block_hash.clone();

    let mut kvm = mk_test_rnode_store_manager_from_genesis(&ctx.genesis);
    let block_store = KeyValueBlockStore::create_from_kvm(&mut *kvm)
        .await
        .expect("block store");
    let dag_storage = block_dag_storage_from_dyn(&mut *kvm)
        .await
        .expect("dag storage");

    block_store
        .put_block_message(&genesis_block)
        .expect("store genesis");
    dag_storage
        .insert(&genesis_block, false, true)
        .expect("dag genesis");

    // Construct four user deploys; the fifth sig (unknown) is not
    // deployed. Each deploy needs a distinct source string — same
    // source + millisecond-resolution `SystemTime::now()` timestamp +
    // default key would produce identical signatures within a single
    // millisecond, causing the "deploys" to collide on one sig.
    let deploy_clean_via_secondary = construct_deploy::source_deploy_now_full(
        "@1!(1)".to_string(),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("deploy_clean_via_secondary");
    let deploy_failed = construct_deploy::source_deploy_now_full(
        "@2!(2)".to_string(),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("deploy_failed");
    let deploy_clean_canonical_reject_canonical = construct_deploy::source_deploy_now_full(
        "@3!(3)".to_string(),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("deploy_clean_canonical_reject_canonical");
    let deploy_clean_canonical_reject_sibling = construct_deploy::source_deploy_now_full(
        "@4!(4)".to_string(),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("deploy_clean_canonical_reject_sibling");

    let sig_clean_via_secondary = deploy_clean_via_secondary.sig.clone();
    let sig_failed = deploy_failed.sig.clone();
    let sig_clean_canonical_reject_canonical = deploy_clean_canonical_reject_canonical.sig.clone();
    let sig_clean_canonical_reject_sibling = deploy_clean_canonical_reject_sibling.sig.clone();
    let sig_unknown = Bytes::from(vec![0xCDu8; 32]);

    // ProcessedDeploy::empty defaults is_failed=false; flip for the
    // `failed_canonical` deploy.
    let mut pd_failed = ProcessedDeploy::empty(deploy_failed);
    pd_failed.is_failed = true;

    // Block A: canonical h=1. Carries failed_canonical (with is_failed)
    // and the two clean-then-reject sigs in body.deploys.
    let block_a = block_implicits::get_random_block(
        Some(1),
        Some(1),
        None,
        None,
        None,
        None,
        Some(0),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(vec![
            pd_failed,
            ProcessedDeploy::empty(deploy_clean_canonical_reject_canonical),
            ProcessedDeploy::empty(deploy_clean_canonical_reject_sibling),
        ]),
        Some(Vec::new()),
        Some(genesis_block.body.state.bonds.clone()),
        Some(genesis_block.shard_id.clone()),
        None,
    );

    // Block B: canonical h=2, main-parent A. Carries the canonical
    // rejection of `clean_canonical_reject_canonical`. B is the main
    // parent of LFB so `is_in_main_chain(B, C)` is true → B is
    // canonical, and its rejection should invalidate the clean
    // inclusion at A.
    //
    // `get_random_block` has no rejected-deploys parameter, so we
    // mutate `body.rejected_deploys` directly after construction.
    let mut block_b = block_implicits::get_random_block(
        Some(2),
        Some(1),
        None,
        None,
        None,
        None,
        Some(0),
        Some(vec![block_a.block_hash.clone()]),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(genesis_block.body.state.bonds.clone()),
        Some(genesis_block.shard_id.clone()),
        None,
    );
    block_b.body.rejected_deploys = vec![RejectedDeploy {
        sig: sig_clean_canonical_reject_canonical.clone(),
    }];

    // Block S: non-canonical sibling of B at h=2. Has main parent A
    // (so `is_in_main_chain(A, S)` is true) but is NOT on LFB's
    // main-parent chain (`is_in_main_chain(S, C)` is false).
    //
    // Body:
    //   - body.deploys = [clean_via_secondary]
    //     Tests that BFS finds clean inclusions reachable only via
    //     secondary-parent slots. With no rejection of this sig
    //     anywhere, the resolver should return Finalized.
    //   - body.rejected_deploys = [clean_canonical_reject_sibling]
    //     Tests the canonical-descendant invalidation gap: with the
    //     buggy implementation, `is_in_main_chain(A, S) = true`
    //     causes invalidation of the clean inclusion at A even
    //     though S itself is non-canonical. With the fix that
    //     additionally requires the rejection block to be on LFB's
    //     main-parent chain, this rejection is correctly ignored
    //     and the sig stays Finalized.
    let mut block_s = block_implicits::get_random_block(
        Some(2),
        Some(2), // distinct seq number from B to give a different block hash
        None,
        None,
        None,
        None,
        Some(0),
        Some(vec![block_a.block_hash.clone()]),
        Some(Vec::new()),
        Some(vec![ProcessedDeploy::empty(deploy_clean_via_secondary)]),
        Some(Vec::new()),
        Some(genesis_block.body.state.bonds.clone()),
        Some(genesis_block.shard_id.clone()),
        None,
    );
    block_s.body.rejected_deploys = vec![RejectedDeploy {
        sig: sig_clean_canonical_reject_sibling.clone(),
    }];

    // Block C: LFB. Multi-parent merge of [B, S]. Main parent = B,
    // secondary parent = S. BFS from C visits both B (canonical) and
    // S (non-canonical) via the parents_hash_list slots.
    let block_c = block_implicits::get_random_block(
        Some(3),
        Some(1),
        None,
        None,
        None,
        None,
        Some(0),
        Some(vec![block_b.block_hash.clone(), block_s.block_hash.clone()]),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(Vec::new()),
        Some(genesis_block.body.state.bonds.clone()),
        Some(genesis_block.shard_id.clone()),
        None,
    );

    block_store.put_block_message(&block_a).expect("store A");
    block_store.put_block_message(&block_b).expect("store B");
    block_store.put_block_message(&block_s).expect("store S");
    block_store.put_block_message(&block_c).expect("store C");
    dag_storage
        .insert(&block_a, false, false)
        .expect("dag insert A");
    dag_storage
        .insert(&block_b, false, false)
        .expect("dag insert B");
    dag_storage
        .insert(&block_s, false, false)
        .expect("dag insert S");
    dag_storage
        .insert(&block_c, false, false)
        .expect("dag insert C");

    let mut dag = dag_storage.get_representation();
    dag.last_finalized_block_hash = block_c.block_hash.clone();

    let deploy_lifespan = 50i64;

    // Per-sig single resolve.
    let single = |sig: &Bytes| {
        deploy_finalization_status::resolve(&dag, &block_store, deploy_lifespan, sig)
            .expect("resolve should not fail")
    };
    let single_clean_via_secondary = single(&sig_clean_via_secondary);
    let single_failed = single(&sig_failed);
    let single_clean_canonical_reject_canonical = single(&sig_clean_canonical_reject_canonical);
    let single_clean_canonical_reject_sibling = single(&sig_clean_canonical_reject_sibling);
    let single_unknown = single(&sig_unknown);

    // Sanity: state assignments match the coverage matrix above. If
    // any of these assertions break, the parity result below is
    // misleading.
    assert_eq!(
        single_clean_via_secondary.state,
        DeployFinalizationState::Finalized,
        "clean_via_secondary sig must be Finalized — BFS must reach S via C's secondary-parent slot"
    );
    assert_eq!(
        single_failed.state,
        DeployFinalizationState::Failed,
        "failed_canonical sig must be Failed"
    );
    assert_eq!(
        single_clean_canonical_reject_canonical.state,
        DeployFinalizationState::Pending,
        "clean_canonical_reject_canonical must be Pending — rejection at B (main-parent of C) is canonical and invalidates clean at A"
    );
    assert_eq!(
        single_clean_canonical_reject_canonical.rejection_count, 1,
        "clean_canonical_reject_canonical must have rejection_count=1"
    );
    assert_eq!(
        single_clean_canonical_reject_sibling.state,
        DeployFinalizationState::Finalized,
        "clean_canonical_reject_sibling must be Finalized — rejection at S is on a non-canonical sibling chain (S is not on LFB's main-parent chain) and must NOT invalidate the clean inclusion at A. \
         If this assertion fails with state=Pending, the resolver's canonical-descendant check is over-aggressive: it accepts any reject_block whose main-parent walk passes through clean_block, without verifying reject_block itself is canonical."
    );
    assert_eq!(
        single_clean_canonical_reject_sibling.rejection_count, 1,
        "clean_canonical_reject_sibling must have rejection_count=1 (the rejection is observed and counted; only the *invalidation* of the clean inclusion is rejected)"
    );
    assert_eq!(
        single_unknown.state,
        DeployFinalizationState::Pending,
        "unknown sig must be Pending"
    );
    assert!(
        single_unknown.latest_block_hash.is_none(),
        "unknown sig must have no latest_block_hash"
    );

    // Batched resolve over the same sigs in one BFS.
    let mut sigs = HashSet::new();
    sigs.insert(sig_clean_via_secondary.clone());
    sigs.insert(sig_failed.clone());
    sigs.insert(sig_clean_canonical_reject_canonical.clone());
    sigs.insert(sig_clean_canonical_reject_sibling.clone());
    sigs.insert(sig_unknown.clone());

    let batch = resolve_batch(&dag, &block_store, deploy_lifespan, &sigs)
        .expect("resolve_batch should not fail");

    // Parity: every sig has a result, and that result equals the single-
    // sig result.
    fn assert_parity(
        label: &str,
        single: &deploy_finalization_status::DeployFinalizationStatus,
        batch: &deploy_finalization_status::DeployFinalizationStatus,
    ) {
        assert_eq!(
            batch.state, single.state,
            "{label}: state mismatch (single={:?}, batch={:?})",
            single.state, batch.state
        );
        assert_eq!(
            batch.rejection_count, single.rejection_count,
            "{label}: rejection_count mismatch (single={}, batch={})",
            single.rejection_count, batch.rejection_count
        );
        assert_eq!(
            batch.latest_block_hash, single.latest_block_hash,
            "{label}: latest_block_hash mismatch"
        );
    }

    assert_parity(
        "clean_via_secondary",
        &single_clean_via_secondary,
        batch
            .get(&sig_clean_via_secondary)
            .expect("batch missing clean_via_secondary"),
    );
    assert_parity(
        "failed",
        &single_failed,
        batch.get(&sig_failed).expect("batch missing failed"),
    );
    assert_parity(
        "clean_canonical_reject_canonical",
        &single_clean_canonical_reject_canonical,
        batch
            .get(&sig_clean_canonical_reject_canonical)
            .expect("batch missing clean_canonical_reject_canonical"),
    );
    assert_parity(
        "clean_canonical_reject_sibling",
        &single_clean_canonical_reject_sibling,
        batch
            .get(&sig_clean_canonical_reject_sibling)
            .expect("batch missing clean_canonical_reject_sibling"),
    );
    assert_parity(
        "unknown",
        &single_unknown,
        batch.get(&sig_unknown).expect("batch missing unknown"),
    );

    // Empty batch returns empty map (regression guard).
    let empty = resolve_batch(&dag, &block_store, deploy_lifespan, &HashSet::new())
        .expect("empty batch should not fail");
    assert!(empty.is_empty(), "empty batch must return empty map");

    // Single-element batch matches single resolve (regression guard
    // for the single-input branch of `resolve_batch`).
    let mut single_set = HashSet::new();
    single_set.insert(sig_clean_via_secondary.clone());
    let one = resolve_batch(&dag, &block_store, deploy_lifespan, &single_set)
        .expect("single-element batch should not fail");
    assert_eq!(one.len(), 1, "single-element batch must return one entry");
    assert_parity(
        "single-element-batch",
        &single_clean_via_secondary,
        one.get(&sig_clean_via_secondary)
            .expect("missing single-element entry"),
    );
}
