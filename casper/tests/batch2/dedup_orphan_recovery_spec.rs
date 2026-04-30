// Code-level regression test for the dedup-orphan path of the
// rejected-deploy recovery pipeline.
//
// `dag_merger::merge` runs a freshness-based dedup pass over the in-scope
// chains: when the same `deploy_id` appears in multiple chains, the chain
// with the highest (block_number, byte-lex-smallest hash) wins, and any
// chain whose deploys all have a fresher copy elsewhere is dropped. The
// orphan path activates when a dropped chain contains a deploy whose
// freshest source IS the dropped chain itself — i.e., the deploy is
// unique to that chain. Such deploys land in `collateral_lost_pairs`,
// which is unioned into `rejected_user_deploys` before the merge
// returns, and admitted to the rejected-deploy buffer like
// conflict-rejected sigs.
//
// The minimum DAG that triggers this path:
//
//   block_a: body.deploys = [X, V] — V's consume depends on X's produce,
//                                     so X and V are in one event-log chain
//   block_b: body.deploys = [X, W] — W is to chain_b what V is to chain_a
//
// Dedup picks chain_a or chain_b for the shared `X` based on
// (block_number, byte-lex hash) — the loser is dropped. The dropped
// chain's UNIQUE deploy (V or W) has no fresher copy and is orphaned.
// We don't predict which chain wins; we assert that exactly one of
// {sig_V, sig_W} reaches the buffer.

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
async fn dedup_orphan_lands_in_rejected_deploy_buffer() {
    crate::init_logger();
    let genesis_context = crate::util::rholang::resources::genesis_context()
        .await
        .unwrap();
    let genesis_block = genesis_context.genesis_block.clone();
    let genesis_hash = genesis_block.block_hash.clone();
    let genesis_state = proto_util::post_state_hash(&genesis_block);
    let genesis_bonds = genesis_block.body.state.bonds.clone();
    let validator: prost::bytes::Bytes = genesis_context.validator_pks()[0].bytes.clone().into();
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
    // The buffer's underlying store is independent of the rspace / block
    // store — its only job is to hold sigs that the merge admits. Using a
    // separate in-memory KVM keeps the type concrete so
    // `KeyValueRejectedDeployBuffer::new`'s `Sized` bound is satisfied.
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
        max_seq_nums.insert(validator.clone(), 0);
        snapshot.max_seq_nums = max_seq_nums;
        let mut shard_conf = CasperShardConf::new();
        shard_conf.shard_name = shard_name.clone();
        shard_conf.max_parent_depth = 0;
        // A non-trivial deploy lifespan keeps the orphan sig in the
        // `Pending` state through `compute_rejected_buffer_admits`. The
        // default of 0 would mark any deploy whose `valid_after_block_number`
        // is below the tip height as `Expired`, which would prevent buffer
        // admission and obscure what this test is checking.
        shard_conf.deploy_lifespan = 50;
        let mut bonds_map = HashMap::new();
        bonds_map.insert(validator.clone(), 100);
        snapshot.on_chain_state = OnChainCasperState {
            shard_conf,
            bonds_map,
            active_validators: vec![validator.clone()],
        };
        snapshot.deploys_in_scope = Arc::new(DashSet::new());
        snapshot.rejected_in_scope = Arc::new(DashSet::new());
        snapshot
    };

    // The shared deploy: produces a value on a well-known channel.
    // Both block_a's chain and block_b's chain include this deploy.
    // Dedup picks one of the two chains for this deploy_id.
    let rho_shared_producer = r#"
@"dedup-orphan-shared"!(42)
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

    // The unique-to-each-block deploys: each consumes the shared produce.
    // The consume depends on deploy_x's produce in event-log terms, so
    // `compute_related_sets` groups [X, V] into a single chain in
    // block_a, and [X, W] into a single chain in block_b.
    let rho_shared_consumer = r#"
for(@_v <- @"dedup-orphan-shared") { Nil }
"#
    .to_string();
    let deploy_v = construct_deploy::source_deploy_now_full(
        rho_shared_consumer.clone(),
        None,
        None,
        Some(construct_deploy::DEFAULT_SEC2.clone()),
        None,
        None,
    )
    .expect("build deploy_v");
    let sig_v = deploy_v.sig.clone();

    // Sleep keeps the timestamp distinct so deploy_w's sig differs from
    // deploy_v's even though they share a Rholang body.
    tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
    let deploy_w = construct_deploy::source_deploy_now_full(
        rho_shared_consumer,
        None,
        None,
        Some(construct_deploy::DEFAULT_SEC2.clone()),
        None,
        None,
    )
    .expect("build deploy_w");
    let sig_w = deploy_w.sig.clone();
    assert_ne!(sig_v, sig_w, "deploy_v and deploy_w must have distinct sigs");

    // ── block_a: body.deploys = [X, V] ──
    let block_a_raw = block_implicits::get_random_block(
        Some(1),
        Some(1),
        Some(genesis_state.clone()),
        Some(StateHash::default()),
        Some(validator.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(vec![
            ProcessedDeploy::empty(deploy_x.clone()),
            ProcessedDeploy::empty(deploy_v.clone()),
        ]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );
    let (_, post_state_a, pd_a, _, sys_pd_a, bonds_a) = compute_deploys_checkpoint(
        &mut block_store,
        vec![genesis_block.clone()],
        proto_util::deploys(&block_a_raw)
            .into_iter()
            .map(|d| d.deploy)
            .collect(),
        Vec::<SystemDeployEnum>::new(),
        &mk_snapshot(&genesis_hash),
        &mut rm,
        BlockData::from_block(&block_a_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute block_a checkpoint");
    for pd in &pd_a {
        assert!(
            !pd.is_failed,
            "deploy in block_a must execute cleanly (sig {}): {:?}",
            hex::encode(&pd.deploy.sig[..8]),
            pd.system_deploy_error
        );
    }
    let mut block_a = block_a_raw;
    block_a.body.state.post_state_hash = post_state_a.clone();
    block_a.body.deploys = pd_a;
    block_a.body.system_deploys = sys_pd_a;
    block_a.body.state.bonds = bonds_a;
    block_store.put_block_message(&block_a).expect("store A");
    dag_storage
        .insert(&block_a, false, false)
        .expect("dag A");

    // ── block_b: body.deploys = [X, W] ──
    let block_b_raw = block_implicits::get_random_block(
        Some(1),
        Some(2),
        Some(genesis_state.clone()),
        Some(StateHash::default()),
        Some(validator.clone()),
        Some(1),
        Some(now_millis()),
        Some(vec![genesis_hash.clone()]),
        Some(Vec::new()),
        Some(vec![
            ProcessedDeploy::empty(deploy_x.clone()),
            ProcessedDeploy::empty(deploy_w.clone()),
        ]),
        Some(Vec::new()),
        Some(genesis_bonds.clone()),
        Some(shard_name.clone()),
        None,
    );
    let (_, post_state_b, pd_b, _, sys_pd_b, bonds_b) = compute_deploys_checkpoint(
        &mut block_store,
        vec![genesis_block.clone()],
        proto_util::deploys(&block_b_raw)
            .into_iter()
            .map(|d| d.deploy)
            .collect(),
        Vec::<SystemDeployEnum>::new(),
        &mk_snapshot(&genesis_hash),
        &mut rm,
        BlockData::from_block(&block_b_raw),
        HashMap::new(),
        None,
    )
    .await
    .expect("compute block_b checkpoint");
    for pd in &pd_b {
        assert!(
            !pd.is_failed,
            "deploy in block_b must execute cleanly (sig {}): {:?}",
            hex::encode(&pd.deploy.sig[..8]),
            pd.system_deploy_error
        );
    }
    let mut block_b = block_b_raw;
    block_b.body.state.post_state_hash = post_state_b.clone();
    block_b.body.deploys = pd_b;
    block_b.body.system_deploys = sys_pd_b;
    block_b.body.state.bonds = bonds_b;
    block_store.put_block_message(&block_b).expect("store B");
    dag_storage
        .insert(&block_b, false, false)
        .expect("dag B");

    assert_ne!(
        block_a.block_hash, block_b.block_hash,
        "block_a and block_b must have distinct hashes for dedup to fire"
    );

    // ── Merge [block_a, block_b] over genesis ──
    //
    // dag_merger's freshness rule (block_number desc, then byte-lex
    // smaller hash) breaks the tie between two block_number=1 siblings
    // by hash. The chain that loses the tiebreak is dropped; its
    // deploy_x sig has no fresher copy elsewhere, so it lands in
    // `collateral_lost_pairs`, which is unioned into the merge's
    // rejected-user output and admitted to the buffer.
    let snapshot = mk_snapshot(&genesis_hash);
    let (_merged_state, rejected_sigs, rejected_slashes) = compute_parents_post_state(
        &block_store,
        vec![block_a.clone(), block_b.clone()],
        &snapshot,
        &rm,
        None,
        Some(&rejected_deploy_buffer),
    )
    .expect("compute_parents_post_state over [block_a, block_b]");

    assert!(
        rejected_slashes.is_empty(),
        "no system slashes are involved in this fixture; rejected_slashes \
         must be empty (got {} entries)",
        rejected_slashes.len()
    );

    let rejected_set: HashSet<prost::bytes::Bytes> = rejected_sigs.iter().cloned().collect();
    let v_orphaned = rejected_set.contains(&sig_v);
    let w_orphaned = rejected_set.contains(&sig_w);
    assert!(
        v_orphaned ^ w_orphaned,
        "exactly one of {{sig_v, sig_w}} must be orphaned (the unique \
         deploy in the dropped chain). got v_orphaned={}, w_orphaned={}, \
         rejected_sigs={:?}. If both fire, dedup didn't run; if neither \
         fires, the orphan logic in `dag_merger::merge` (the \
         `collateral_lost_pairs` push at lines ~217-235 or the union at \
         lines ~563-573) is not reaching the merge output",
        v_orphaned,
        w_orphaned,
        rejected_sigs
            .iter()
            .map(|s| hex::encode(&s[..std::cmp::min(8, s.len())]))
            .collect::<Vec<_>>()
    );
    assert!(
        !rejected_set.contains(&sig_x),
        "the shared deploy_x is not orphaned — it has a fresher copy in \
         the retained chain, so it must NOT be in collateral_lost_pairs. \
         If this fires, the orphan classification in dag_merger is \
         flagging shared deploys as collateral, which would cause them to \
         be re-proposed unnecessarily"
    );

    // The orphaned sig must be admitted to the buffer. The catchup gate
    // (`compute_rejected_buffer_admits`) checks finalization status; for
    // an unfinalized sig in the merge scope, status is `Pending` and the
    // sig is admitted.
    let orphaned_sig = if v_orphaned { &sig_v } else { &sig_w };
    let buffer_contains = {
        let guard = rejected_deploy_buffer
            .lock()
            .expect("buffer lock");
        guard
            .contains_sig(orphaned_sig)
            .expect("buffer.contains_sig")
    };
    assert!(
        buffer_contains,
        "rejected-deploy buffer must contain the orphaned sig {} after \
         the merge. If this fails, the buffer-admit machinery in \
         `compute_parents_post_state` is not consuming \
         `rejected_user_deploys` entries that came from \
         `collateral_lost_pairs`",
        hex::encode(orphaned_sig)
    );

    // The shared deploy_x must NOT be in the buffer — it has a fresher
    // copy and isn't a recovery candidate.
    let buffer_has_x = {
        let guard = rejected_deploy_buffer
            .lock()
            .expect("buffer lock");
        guard
            .contains_sig(&sig_x)
            .expect("buffer.contains_sig")
    };
    assert!(
        !buffer_has_x,
        "the shared sig_x must NOT be in the buffer — it's not orphaned"
    );
}
