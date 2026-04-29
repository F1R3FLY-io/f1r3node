// Proof tests for BlockAPI::deploy_finalization_status.
//
// Covers the API-surface states that can be triggered with the existing
// single-node TestNode fixture. Deep end-to-end coverage (multi-
// equivocation, cost-starvation simulation, merge-rejection-then-
// recovery) requires additional test infrastructure and is tracked
// separately.

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
