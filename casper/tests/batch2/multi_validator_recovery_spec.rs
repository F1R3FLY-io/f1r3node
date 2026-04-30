// Code-level regression test for the multi-validator buffer
// convergence scenario: two validators independently buffer the same
// conflict-rejected sig and each re-propose it in their own block,
// alongside a validator-unique marker deploy.
//
// `dag_merger::merge`'s freshness-based dedup at dag_merger.rs:153-235
// must pick exactly one chain by `(block_number, byte-lex hash)` for
// the shared recovered sig, drop the other, and route the dropped
// chain's unique marker into `collateral_lost_pairs` (orphan path).
// The shared sig itself does NOT land in `rejected_user_deploys` —
// dedup handled the duplicate cleanly without invoking conflict
// resolution.
//
// The validator-unique markers are critical to isolating the dedup
// path: they make the two chains' `deploys_with_cost` sets distinct,
// so `conflict_set_merger`'s `actual_set: HashSet` cannot collapse
// them. The dedup logic is then the only mechanism that can avoid
// surfacing the shared sig as a conflict-rejected duplicate.
//
// TDD red-green: disabling the dedup retain logic at
// dag_merger.rs:201-213 routes the shared sig through
// `conflict_set_merger`'s `same_deploy` short-circuit, surfacing it
// in `rejected_user_deploys` and failing the assertion.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use block_storage::rust::deploy::key_value_rejected_deploy_buffer::KeyValueRejectedDeployBuffer;
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use casper::rust::genesis::genesis::Genesis;
use casper::rust::{
    casper::{CasperShardConf, CasperSnapshot, OnChainCasperState},
    util::{
        construct_deploy, proto_util,
        rholang::{
            interpreter_util::{compute_deploys_checkpoint, compute_parents_post_state},
            runtime_manager::RuntimeManager,
            system_deploy_enum::SystemDeployEnum,
        },
    },
};
use dashmap::{DashMap, DashSet};
use models::rust::{
    block::state_hash::StateHash,
    block_hash::BlockHash,
    block_implicits,
    casper::protocol::casper_message::ProcessedDeploy,
};
use rholang::rust::interpreter::{
    external_services::ExternalServices, system_processes::BlockData,
};

use crate::util::rholang::resources::{
    block_dag_storage_from_dyn, mergeable_store_from_dyn,
    mk_test_rnode_store_manager_from_genesis,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_validator_recovery_dedups_re_proposed_sig() {
    crate::init_logger();
    let genesis_context = crate::util::rholang::resources::genesis_context()
        .await
        .unwrap();
    let genesis_block = genesis_context.genesis_block.clone();
    let genesis_hash = genesis_block.block_hash.clone();
    let genesis_state = proto_util::post_state_hash(&genesis_block);
    let genesis_bonds = genesis_block.body.state.bonds.clone();
    let validator_0: prost::bytes::Bytes = genesis_context.validator_pks()[0].bytes.clone().into();
    let validator_1: prost::bytes::Bytes = genesis_context.validator_pks()[1].bytes.clone().into();
    let shard_name = genesis_block.shard_id.clone();

    let mut kvm = mk_test_rnode_store_manager_from_genesis(&genesis_context);
    let rspace_store = kvm.r_space_stores().await.expect("rspace stores");
    let mergeable_store = mergeable_store_from_dyn(&mut *kvm)
        .await
        .expect("mergeable store");
    let (mut rm, _) = RuntimeManager::create_with_history(
        rspace_store,
        mergeable_store,
        Genesis::non_negative_mergeable_tag_name(),
        ExternalServices::noop(),
    );

    let mut block_store = KeyValueBlockStore::create_from_kvm(&mut *kvm)
        .await
        .expect("block store");
    let dag_storage = block_dag_storage_from_dyn(&mut *kvm)
        .await
        .expect("dag storage");
    let mut buffer_kvm =
        rspace_plus_plus::rspace::shared::in_mem_store_manager::InMemoryStoreManager::new();
    let rejected_deploy_buffer = Arc::new(Mutex::new(
        KeyValueRejectedDeployBuffer::new(&mut buffer_kvm)
            .await
            .expect("rejected deploy buffer"),
    ));

    block_store
        .put_block_message(&genesis_block)
        .expect("store genesis");
    dag_storage
        .insert(&genesis_block, false, true)
        .expect("dag genesis");

    let now_millis = || -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    };

    let mk_snapshot = |lfb: &BlockHash| -> CasperSnapshot {
        let mut snapshot = CasperSnapshot::new(dag_storage.get_representation());
        snapshot.last_finalized_block = lfb.clone();
        let max_seq_nums: DashMap<prost::bytes::Bytes, u64> = DashMap::new();
        max_seq_nums.insert(validator_0.clone(), 0);
        max_seq_nums.insert(validator_1.clone(), 0);
        snapshot.max_seq_nums = max_seq_nums;
        let mut shard_conf = CasperShardConf::new();
        shard_conf.shard_name = shard_name.clone();
        shard_conf.max_parent_depth = 0;
        shard_conf.deploy_lifespan = 50;
        let mut bonds_map = HashMap::new();
        bonds_map.insert(validator_0.clone(), 100);
        bonds_map.insert(validator_1.clone(), 100);
        snapshot.on_chain_state = OnChainCasperState {
            shard_conf,
            bonds_map,
            active_validators: vec![validator_0.clone(), validator_1.clone()],
        };
        snapshot.deploys_in_scope = Arc::new(DashSet::new());
        snapshot.rejected_in_scope = Arc::new(DashSet::new());
        snapshot
    };

    // The shared recovered deploy that both validators independently
    // re-propose. Produces on a well-known channel so the
    // validator-unique markers (which consume from the same channel)
    // become event-log-dependent on it — putting [deploy_x, marker]
    // in a single deploy chain inside each block.
    let rho_shared_producer = r#"
@"multi-validator-shared"!(42)
"#
    .to_string();
    let deploy_x = construct_deploy::source_deploy_now_full(
        rho_shared_producer,
        None,
        None,
        Some(construct_deploy::DEFAULT_SEC.clone()),
        None,
        None,
    )
    .expect("build deploy_x");
    let sig_x = deploy_x.sig.clone();

    // Validator-unique markers. Each consumes the shared produce, which
    // creates the event-log dependency that puts marker and deploy_x
    // in the same chain via `compute_related_sets`. Distinct sigs (via
    // distinct timestamps) make the two chains' `deploys_with_cost`
    // sets unequal — `conflict_set_merger`'s HashSet collapse cannot
    // merge them. The merge then relies on `dag_merger::merge`'s
    // explicit freshness-based dedup, which picks one chain by hash
    // and orphans the loser's unique marker.
    let rho_shared_consumer = r#"
for(@_v <- @"multi-validator-shared") { Nil }
"#
    .to_string();
    let marker_v0 = construct_deploy::source_deploy_now_full(
        rho_shared_consumer.clone(),
        None,
        None,
        Some(construct_deploy::DEFAULT_SEC2.clone()),
        None,
        None,
    )
    .expect("build marker_v0");
    let sig_marker_v0 = marker_v0.sig.clone();

    tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
    let marker_v1 = construct_deploy::source_deploy_now_full(
        rho_shared_consumer,
        None,
        None,
        Some(construct_deploy::DEFAULT_SEC2.clone()),
        None,
        None,
    )
    .expect("build marker_v1");
    let sig_marker_v1 = marker_v1.sig.clone();
    assert_ne!(
        sig_marker_v0, sig_marker_v1,
        "validator-unique markers must have distinct sigs"
    );

    // ── Recovery block R0 (sender = validator 0, body = [deploy_x, marker_v0]) ──
    let r0_raw = block_implicits::get_random_block(
        Some(1),
        Some(1),
        Some(genesis_state.clone()),
        Some(StateHash::default()),
        Some(validator_0.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(vec![
            ProcessedDeploy::empty(deploy_x.clone()),
            ProcessedDeploy::empty(marker_v0.clone()),
        ]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );
    let (_, post_state_r0, pd_r0, _, sys_pd_r0, bonds_r0) = compute_deploys_checkpoint(
        &mut block_store,
        vec![genesis_block.clone()],
        proto_util::deploys(&r0_raw)
            .into_iter()
            .map(|d| d.deploy)
            .collect(),
        Vec::<SystemDeployEnum>::new(),
        &mk_snapshot(&genesis_hash),
        &mut rm,
        BlockData::from_block(&r0_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute R0 checkpoint");
    for pd in &pd_r0 {
        assert!(
            !pd.is_failed,
            "deploy in R0 must execute cleanly (sig {}): {:?}",
            hex::encode(&pd.deploy.sig[..8]),
            pd.system_deploy_error
        );
    }
    let mut r0 = r0_raw;
    r0.body.state.post_state_hash = post_state_r0.clone();
    r0.body.deploys = pd_r0;
    r0.body.system_deploys = sys_pd_r0;
    r0.body.state.bonds = bonds_r0;
    block_store.put_block_message(&r0).expect("store R0");
    dag_storage
        .insert(&r0, false, false)
        .expect("dag R0");

    // ── Recovery block R1 (sender = validator 1, body = [deploy_x, marker_v1]) ──
    let r1_raw = block_implicits::get_random_block(
        Some(1),
        Some(1),
        Some(genesis_state.clone()),
        Some(StateHash::default()),
        Some(validator_1.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(vec![
            ProcessedDeploy::empty(deploy_x.clone()),
            ProcessedDeploy::empty(marker_v1.clone()),
        ]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );
    let (_, post_state_r1, pd_r1, _, sys_pd_r1, bonds_r1) = compute_deploys_checkpoint(
        &mut block_store,
        vec![genesis_block.clone()],
        proto_util::deploys(&r1_raw)
            .into_iter()
            .map(|d| d.deploy)
            .collect(),
        Vec::<SystemDeployEnum>::new(),
        &mk_snapshot(&genesis_hash),
        &mut rm,
        BlockData::from_block(&r1_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute R1 checkpoint");
    for pd in &pd_r1 {
        assert!(
            !pd.is_failed,
            "deploy in R1 must execute cleanly (sig {}): {:?}",
            hex::encode(&pd.deploy.sig[..8]),
            pd.system_deploy_error
        );
    }
    let mut r1 = r1_raw;
    r1.body.state.post_state_hash = post_state_r1.clone();
    r1.body.deploys = pd_r1;
    r1.body.system_deploys = sys_pd_r1;
    r1.body.state.bonds = bonds_r1;
    block_store.put_block_message(&r1).expect("store R1");
    dag_storage
        .insert(&r1, false, false)
        .expect("dag R1");

    assert_ne!(
        r0.block_hash, r1.block_hash,
        "R0 and R1 must have distinct hashes for the dedup tiebreak to be observable"
    );
    assert_ne!(
        r0.sender, r1.sender,
        "R0 and R1 must come from distinct validators to represent the multi-validator scenario"
    );

    // ── Merge [R0, R1] over genesis ──
    //
    // Dedup picks the chain whose source-block hash sorts byte-lex
    // smaller for sig_x. The losing chain's unique marker has no
    // fresher copy elsewhere and lands in `collateral_lost_pairs`,
    // which is unioned into `rejected_user_deploys`. The shared sig_x
    // does NOT appear there: dedup retained it via the winning chain.
    let snapshot = mk_snapshot(&genesis_hash);
    let (_merged_state, rejected_sigs, rejected_slashes) = compute_parents_post_state(
        &block_store,
        vec![r0.clone(), r1.clone()],
        &snapshot,
        &rm,
        None,
        Some(&rejected_deploy_buffer),
    )
    .expect("compute_parents_post_state over [R0, R1]");

    assert!(
        rejected_slashes.is_empty(),
        "no system slashes are involved in this fixture; rejected_slashes \
         must be empty (got {} entries)",
        rejected_slashes.len()
    );

    let rejected_set: HashSet<prost::bytes::Bytes> = rejected_sigs.iter().cloned().collect();

    // Dedup must keep the shared recovered sig out of the rejected
    // list. If it appears here, dedup did not run and conflict
    // resolution surfaced the duplicate via the `same_deploy`
    // short-circuit — meaning the multi-validator-convergence dedup at
    // dag_merger.rs:153-235 is broken or has been removed.
    assert!(
        !rejected_set.contains(&sig_x),
        "sig_x must NOT appear in `rejected_user_deploys`. Got: {:?}",
        rejected_sigs
            .iter()
            .map(|s| hex::encode(&s[..std::cmp::min(8, s.len())]))
            .collect::<Vec<_>>()
    );

    // Exactly one validator-unique marker is orphaned: the loser's
    // marker has no fresher copy in the retained chain, so the orphan
    // path (`collateral_lost_pairs` → `rejected_user_deploys`) surfaces
    // it. We don't predict which validator wins the hash tiebreak —
    // either marker_v0 or marker_v1 is acceptable, but exactly one
    // must fire.
    let v0_orphaned = rejected_set.contains(&sig_marker_v0);
    let v1_orphaned = rejected_set.contains(&sig_marker_v1);
    assert!(
        v0_orphaned ^ v1_orphaned,
        "exactly one of {{marker_v0, marker_v1}} must be orphaned (the \
         loser's unique deploy). got v0_orphaned={}, v1_orphaned={}, \
         rejected_sigs={:?}. If neither fires, the dedup orphan path \
         (`collateral_lost_pairs` push at dag_merger.rs:217-235) is not \
         classifying validator-unique markers as collateral. If both \
         fire, dedup did not retain a winning chain.",
        v0_orphaned,
        v1_orphaned,
        rejected_sigs
            .iter()
            .map(|s| hex::encode(&s[..std::cmp::min(8, s.len())]))
            .collect::<Vec<_>>()
    );
}
