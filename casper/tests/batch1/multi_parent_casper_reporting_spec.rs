// See casper/src/test/scala/coop/rchain/casper/batch1/MultiParentCasperReportingSpec.scala

use crate::util::rholang::resources::mk_test_rnode_store_manager_shared;
use casper::rust::reporting_casper::{rho_reporter, ReportingCasper};
use casper::rust::util::construct_deploy;
use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use models::rust::casper::protocol::casper_message::Event;
use rholang::rust::interpreter::external_services::ExternalServices;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use rspace_plus_plus::rspace::history::history_repository::HistoryRepositoryInstances;
use rspace_plus_plus::rspace::reporting_rspace::ReportingEvent;
use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
use rspace_plus_plus::rspace::state::exporters::rspace_exporter_items::RSpaceExporterItems;

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::GenesisBuilder;

/// Mirrors `MultiParentCasperReportingSpec` from the Scala suite. The
/// reporter must converge to the same post-state hash as production for
/// every block, and its replayed COMM-event count for each deploy must
/// equal what the production execution recorded in the deploy log.
///
/// Anything else means the reporter's replay diverges from the
/// production-recorded execution — the bug class observed on
/// LFS-synced READONLY observers where every block's precharge system
/// deploy returns `ConsumeFailed` and `create_checkpoint` panics on
/// unconsumed COMM events.
#[tokio::test]
async fn reporting_casper_should_behave_the_same_way_as_multi_parent_casper() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let mut node = TestNode::standalone(genesis.clone()).await.unwrap();

    let correct_rholang = r#" for(@a <- @"1"){ Nil } | @"1"!("x") "#;

    let deploy = construct_deploy::source_deploy_now(
        correct_rholang.to_string(),
        None,
        None,
        Some(genesis.genesis_block.shard_id.clone()),
    )
    .unwrap();

    let signed_block = node.add_block_from_deploys(&[deploy]).await.unwrap();

    let reporter: std::sync::Arc<dyn ReportingCasper> = rho_reporter(
        &node.rspace_store,
        &node.block_store,
        &node.block_dag_storage,
        ExternalServices::noop(),
    );

    let trace = reporter
        .trace(&signed_block)
        .await
        .expect("reporter.trace must succeed for a freshly-played block");

    assert_eq!(
        trace.deploy_report_result.len(),
        signed_block.body.deploys.len(),
        "reporter must produce one deploy report per processed deploy",
    );

    // COMM-event count: reporter's collected events for the first deploy
    // must equal the production deploy_log's COMM count. (Persistent-mode
    // produces / consumes can drift on non-COMM event kinds; only COMM
    // events are required to match one-for-one.)
    let production_comm_events = signed_block.body.deploys[0]
        .deploy_log
        .iter()
        .filter(|event| matches!(event, Event::Comm(_)))
        .count();
    let reporter_comm_events = trace.deploy_report_result[0]
        .events
        .iter()
        .flatten()
        .filter(|event| matches!(event, ReportingEvent::ReportingComm(_)))
        .count();
    assert_eq!(
        reporter_comm_events, production_comm_events,
        "reporter must produce the same COMM event count as production for \
         the same deploy. Mismatch means replay diverged from production.",
    );

    // Post-state hash is the strongest single check: it folds in every
    // committed COMM event and channel write across all of the block's
    // deploys + system deploys.
    let production_post_state = signed_block.body.state.post_state_hash.clone();
    let reporter_post_state: Vec<u8> = trace.post_state_hash.to_vec();
    assert_eq!(
        reporter_post_state,
        production_post_state.to_vec(),
        "reporter post_state_hash must match production for the same block",
    );
}

/// Reproduces the observer-side reporter divergence at the unit-test
/// level. Mirrors the production scenario:
///
///   1. Node A plays a deploy and produces a block (the "producer").
///   2. A second, empty rspace store B is created (the "joiner").
///   3. B is populated by exporting A's rspace via the rspace exporter
///      and importing the items via the rspace importer — exactly the
///      contract the LFS forward-horizon sync uses on a real joiner.
///   4. A reporter is constructed against B's freshly-imported store
///      and asked to trace the same block.
///   5. The reporter MUST converge to the same post-state hash and the
///      same COMM-event count as the producer.
///
/// In production this fails on every observer that LFS-syncs to a live
/// shard: the reporter's per-block precharge replay returns
/// `ConsumeFailed`, deploy replay produces zero events, and
/// `create_checkpoint` panics on the unconsumed COMM events left in
/// the rigged trace. The same reporter against a freshly-played rspace
/// (the test above) works correctly, so the divergence is specific to
/// LFS-imported state, not to the reporter logic.
#[tokio::test]
async fn reporting_casper_works_against_lfs_imported_rspace() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let mut node_a = TestNode::standalone(genesis.clone()).await.unwrap();

    let correct_rholang = r#" for(@a <- @"1"){ Nil } | @"1"!("x") "#;
    let deploy = construct_deploy::source_deploy_now(
        correct_rholang.to_string(),
        None,
        None,
        Some(genesis.genesis_block.shard_id.clone()),
    )
    .unwrap();

    let signed_block = node_a.add_block_from_deploys(&[deploy]).await.unwrap();
    let pre_state = Blake2b256Hash::from_bytes_prost(&signed_block.body.state.pre_state_hash);
    let post_state = Blake2b256Hash::from_bytes_prost(&signed_block.body.state.post_state_hash);

    // Producer-side history repo (for the exporter). We construct a
    // fresh typed view over A's underlying KV stores so the exporter
    // walks the same trie production wrote into.
    let history_repo_a = HistoryRepositoryInstances::<
        Par,
        BindPattern,
        ListParWithRandom,
        TaggedContinuation,
    >::lmdb_repository(
        node_a.rspace_store.history.clone(),
        node_a.rspace_store.roots.clone(),
        node_a.rspace_store.cold.clone(),
    )
    .expect("Failed to construct producer history repository");
    let exporter_a = history_repo_a.exporter();

    // Joiner-side: brand-new on-disk LMDB, brand-new rspace_store. Nothing
    // in it yet — this is the state observer1 starts in before LFS sync.
    let scope_id = format!(
        "reporter-lfs-import-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let mut kvm_b = mk_test_rnode_store_manager_shared(scope_id);
    let rspace_store_b = (&mut *kvm_b)
        .r_space_stores()
        .await
        .expect("Failed to construct joiner rspace stores");
    let history_repo_b = HistoryRepositoryInstances::<
        Par,
        BindPattern,
        ListParWithRandom,
        TaggedContinuation,
    >::lmdb_repository(
        rspace_store_b.history.clone(),
        rspace_store_b.roots.clone(),
        rspace_store_b.cold.clone(),
    )
    .expect("Failed to construct joiner history repository");
    let importer_b = history_repo_b.importer();

    // Mirror the LFS forward-horizon import for both pre_state and
    // post_state: walk the trie via the exporter in chunks, and feed
    // each chunk to the joiner's importer until pagination terminates.
    // Bounded loop: any honest peer with finite trie depth must
    // terminate well below this bound.
    let max_chunks_per_root = 4096;
    let page_size = 1024;
    for root in [&pre_state, &post_state] {
        let mut start_path = vec![(root.clone(), None)];
        let mut chunks = 0;
        loop {
            assert!(
                chunks < max_chunks_per_root,
                "exporter did not terminate within {} chunks for root {}",
                max_chunks_per_root,
                root
            );
            let (history, data) = RSpaceExporterItems::get_history_and_data(
                exporter_a.clone(),
                start_path.clone(),
                0,
                page_size,
            );
            let next_start = history.last_path.clone();
            importer_b.set_history_items(history.items);
            importer_b.set_data_items(data.items);
            chunks += 1;
            if next_start == start_path {
                // Terminal cursor: subtree exhausted at this start_path.
                break;
            }
            start_path = next_start;
        }
        importer_b.set_root(root);
    }

    // Construct the reporter against the joiner's LFS-imported store.
    // BlockStore + BlockDagStorage reuse A's instances so the reporter
    // can resolve the block; only the rspace state is freshly imported.
    let reporter: std::sync::Arc<dyn ReportingCasper> = rho_reporter(
        &rspace_store_b,
        &node_a.block_store,
        &node_a.block_dag_storage,
        ExternalServices::noop(),
    );

    let trace = reporter
        .trace(&signed_block)
        .await
        .expect("reporter.trace must succeed against an LFS-imported rspace");

    assert_eq!(
        trace.deploy_report_result.len(),
        signed_block.body.deploys.len(),
        "reporter must produce one deploy report per processed deploy",
    );

    let production_comm_events = signed_block.body.deploys[0]
        .deploy_log
        .iter()
        .filter(|event| matches!(event, Event::Comm(_)))
        .count();
    let reporter_comm_events = trace.deploy_report_result[0]
        .events
        .iter()
        .flatten()
        .filter(|event| matches!(event, ReportingEvent::ReportingComm(_)))
        .count();
    assert_eq!(
        reporter_comm_events, production_comm_events,
        "reporter against LFS-imported rspace must produce the same COMM event \
         count as production. Mismatch is the divergence we observe on \
         observer1: replay leaves rigged events unconsumed because the \
         imported rspace is missing some state production had.",
    );

    let production_post_state = signed_block.body.state.post_state_hash.clone();
    let reporter_post_state: Vec<u8> = trace.post_state_hash.to_vec();
    assert_eq!(
        reporter_post_state,
        production_post_state.to_vec(),
        "reporter post_state_hash against LFS-imported rspace must match production",
    );
}
