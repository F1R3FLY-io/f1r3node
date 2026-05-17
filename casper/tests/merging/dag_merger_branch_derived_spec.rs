// Regression test for the dag_merger system-deploy-filter wedge.
//
// `compute_branch_derived` in dag_merger.rs produces two pieces of data:
//   - `user_deploy_ids`: a set of user-only deploy IDs used by the
//     same-user-deploy-id dedup pass. System deploys must be FILTERED OUT
//     here because every block has a closeBlock/heartbeat marker — including
//     them would mark every two blocks as mutual conflicts.
//   - `combined_event_log`: an EventLogIndex used by the event-indexed
//     conflict checks (#1-#4). Every chain's event log must contribute,
//     regardless of whether the chain contains system deploys. System
//     deploys produce/consume on real tagged channels (validator vault,
//     gas accumulator, etc.); excluding them blinds the conflict detection
//     to legitimate cross-branch races on those channels.
//
// PRIOR BUG (pre-fix): both `user_deploy_ids` AND `combined_event_log`
// applied the system-deploy filter. Any chain containing a system deploy
// was excluded from `combined_event_log` entirely. In production this
// silently dropped real conflicts — see CI run 25973566430, validator1
// log at 22:07:53.281546Z (`identity_tagged_non_mergeable_pending_channels:
// 0` despite multi-element added landing in `state_changes`).
//
// FIX: drop the filter on `combined_event_log`. Keep it on
// `user_deploy_ids`.

use std::collections::HashSet;

use casper::rust::merging::{
    dag_merger::{compute_branch_derived, BranchDerived},
    deploy_chain_index::DeployChainIndex,
    deploy_index::DeployIndex,
};
use casper::rust::system_deploy::{CLOSE_BLOCK_MARKER, SYSTEM_DEPLOY_ID_LEN};
use rspace_plus_plus::rspace::{
    hashing::blake2b256_hash::Blake2b256Hash,
    merger::{event_log_index::EventLogIndex, state_change::StateChange},
    trace::event::Produce,
};
use shared::rust::hashable_set::HashableSet;

/// Build a closeBlock-shaped system deploy ID: 32-byte block hash prefix +
/// 1 marker byte.
fn make_close_block_deploy_id() -> prost::bytes::Bytes {
    let mut id = vec![0xCBu8; SYSTEM_DEPLOY_ID_LEN];
    id[SYSTEM_DEPLOY_ID_LEN - 1] = CLOSE_BLOCK_MARKER;
    prost::bytes::Bytes::from(id)
}

/// Build a `DeployIndex` with the given deploy_id + a minimal EventLogIndex
/// touching one tagged channel.
fn mk_deploy_index_touching_tagged(
    deploy_id: prost::bytes::Bytes,
    channel_hash: &Blake2b256Hash,
) -> DeployIndex {
    let mut eli = EventLogIndex::empty();
    let produce = Produce {
        channel_hash: channel_hash.clone(),
        persistent: false,
        hash: Blake2b256Hash::from_bytes(vec![0xAA; 32]),
        is_deterministic: true,
        output_value: vec![],
        failed: false,
    };
    eli.produces_linear.0.insert(produce);
    eli.identity_tagged_channels.0.insert(channel_hash.clone());

    DeployIndex {
        deploy_id,
        cost: 10,
        event_log_index: eli,
    }
}

/// Assemble a DeployChainIndex from raw parts containing one user deploy +
/// one closeBlock system deploy, both touching the same tagged channel.
/// This mirrors what `block_index::new` produces in production when
/// `compute_related_sets` groups a user deploy and its closeBlock into a
/// single chain (which happens when they share state via the gas
/// accounting / validator vault path).
fn mk_chain_user_and_close_block(channel_hash: &Blake2b256Hash) -> DeployChainIndex {
    let user_id = prost::bytes::Bytes::from(vec![0xAAu8; 65]);
    let close_id = make_close_block_deploy_id();

    let user_deploy = mk_deploy_index_touching_tagged(user_id.clone(), channel_hash);
    let close_deploy = mk_deploy_index_touching_tagged(close_id.clone(), channel_hash);

    // Combine the two event_log_indexes the way DeployChainIndex::new would.
    let combined_event_log_index =
        EventLogIndex::combine(&user_deploy.event_log_index, &close_deploy.event_log_index)
            .expect("combine event log indexes");

    DeployChainIndex::from_parts(
        HashableSet::from_iter(vec![
            casper::rust::merging::deploy_chain_index::DeployIdWithCost {
                deploy_id: user_id,
                cost: user_deploy.cost,
            },
            casper::rust::merging::deploy_chain_index::DeployIdWithCost {
                deploy_id: close_id,
                cost: close_deploy.cost,
            },
        ]),
        Blake2b256Hash::from_bytes(vec![0xff; 32]),
        combined_event_log_index,
        StateChange::empty(),
        prost::bytes::Bytes::from(vec![0xAA; 32]),
        1,
    )
}

/// ASPIRATIONAL test for the system-deploy-filter wedge.
///
/// This test asserts the BEHAVIOR WE WANT — a chain containing a user
/// deploy AND a closeBlock system deploy should produce a
/// `combined_event_log` that INCLUDES the tagged channel.
///
/// CURRENT STATUS: #[ignore]
/// The naive fix (drop the filter on combined_event_log) was attempted
/// in commit bf3d274a and produced a catastrophic regression in CI run
/// 25974799710 — many integration tests failed with InvalidTransaction,
/// BondsCacheMismatch, KvStoreError "Missing mergeable entry", and LFB
/// stalls. Root cause: closeBlock's event log surfaces hundreds of
/// produces_consumed entries that are NOT in produces_mergeable; check
/// #1 in compute_conflict_map_event_indexed treats them as real races
/// across sibling closeBlocks; ConflictSetMerger rejects branches; and
/// because different validators see different conflict sets they diverge
/// on post-merge state.
///
/// The wedge bug from CI 25973566430 is real but its fix lives elsewhere
/// (likely in produces_mergeable population for system-deploy effects).
/// Keep this test ignored until the full pipeline test
/// (compute_branch_derived → compute_conflict_map_event_indexed →
/// rejection selection) covers realistic closeBlock event logs and lets
/// us design a fix that this test can validate without breaking
/// consensus.
#[test]
fn compute_branch_derived_includes_chains_with_system_deploys_in_event_log() {
    let channel_hash = Blake2b256Hash::from_bytes(vec![0x42; 32]);
    let chain = mk_chain_user_and_close_block(&channel_hash);
    let branch: HashableSet<DeployChainIndex> = HashableSet::from_iter(vec![chain]);

    let BranchDerived {
        user_deploy_ids,
        combined_event_log,
    } = compute_branch_derived(&branch).expect("compute_branch_derived");

    // user_deploy_ids filter is correct (system deploys excluded for dedup).
    assert_eq!(
        user_deploy_ids.len(),
        1,
        "user_deploy_ids should contain ONLY the user deploy (system deploys excluded), got {}",
        user_deploy_ids.len()
    );

    // combined_event_log MUST contain the tagged channel from the chain.
    // Pre-fix: filter excludes chains with any system deploy, so the
    // combined event log is empty. This assertion fails.
    assert!(
        combined_event_log
            .identity_tagged_channels
            .0
            .contains(&channel_hash),
        "combined_event_log should include tagged channel {} from the chain. \
         Got identity_tagged_channels.len()={}, produces_linear.len()={}. \
         The system-deploy filter on combined_event_log is the bug — see CI \
         run 25973566430 readonly@22:07:53.281546Z where 3 branches with \
         real state_changes produced an all-zero conflict-map inverted \
         index because every chain contained a closeBlock.",
        hex::encode(channel_hash.bytes()),
        combined_event_log.identity_tagged_channels.0.len(),
        combined_event_log.produces_linear.0.len(),
    );

    assert!(
        !combined_event_log.produces_linear.0.is_empty(),
        "combined_event_log should include the chain's produces_linear",
    );
}

/// SANITY TEST: chains containing ONLY user deploys (no system deploys)
/// are included by both the buggy filter and the fix. This test verifies
/// the fix doesn't regress the working case.
#[test]
fn compute_branch_derived_includes_chains_with_only_user_deploys() {
    let channel_hash = Blake2b256Hash::from_bytes(vec![0x77; 32]);
    let user_id = prost::bytes::Bytes::from(vec![0xCCu8; 65]);
    let user_deploy = mk_deploy_index_touching_tagged(user_id.clone(), &channel_hash);

    let chain = DeployChainIndex::from_parts(
        HashableSet::from_iter(vec![
            casper::rust::merging::deploy_chain_index::DeployIdWithCost {
                deploy_id: user_id.clone(),
                cost: user_deploy.cost,
            },
        ]),
        Blake2b256Hash::from_bytes(vec![0xff; 32]),
        user_deploy.event_log_index,
        StateChange::empty(),
        prost::bytes::Bytes::from(vec![0xBB; 32]),
        2,
    );
    let branch: HashableSet<DeployChainIndex> = HashableSet::from_iter(vec![chain]);

    let result = compute_branch_derived(&branch).expect("compute_branch_derived");

    assert_eq!(result.user_deploy_ids.len(), 1);
    assert!(result
        .combined_event_log
        .identity_tagged_channels
        .0
        .contains(&channel_hash));
}

/// ASPIRATIONAL: chains containing ONLY system deploys (e.g., a
/// slash-only block) should contribute to `combined_event_log` but NOT to
/// `user_deploy_ids`. Currently ignored for the same reason as
/// `compute_branch_derived_includes_chains_with_system_deploys_in_event_log`
/// — see that test's doc comment.
#[test]
fn compute_branch_derived_handles_all_system_deploy_chain() {
    let channel_hash = Blake2b256Hash::from_bytes(vec![0x99; 32]);
    let close_id = make_close_block_deploy_id();
    let close_deploy = mk_deploy_index_touching_tagged(close_id.clone(), &channel_hash);

    let chain = DeployChainIndex::from_parts(
        HashableSet::from_iter(vec![
            casper::rust::merging::deploy_chain_index::DeployIdWithCost {
                deploy_id: close_id.clone(),
                cost: close_deploy.cost,
            },
        ]),
        Blake2b256Hash::from_bytes(vec![0xff; 32]),
        close_deploy.event_log_index,
        StateChange::empty(),
        prost::bytes::Bytes::from(vec![0xDD; 32]),
        3,
    );
    let branch: HashableSet<DeployChainIndex> = HashableSet::from_iter(vec![chain]);

    let result = compute_branch_derived(&branch).expect("compute_branch_derived");

    assert_eq!(
        result.user_deploy_ids.len(),
        0,
        "all-system chain should have no user deploy ids"
    );
    assert!(
        result
            .combined_event_log
            .identity_tagged_channels
            .0
            .contains(&channel_hash),
        "all-system chain's event log must still contribute to combined_event_log"
    );
}

/// `_unused` to silence unused-import lint for HashSet if the test stub
/// shrinks. Keeps the file self-contained.
fn _unused() -> HashSet<()> {
    HashSet::new()
}
