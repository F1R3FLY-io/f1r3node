// See casper/src/test/scala/coop/rchain/casper/merging/MergingCases.scala

use std::collections::HashMap;

use casper::rust::{
    merging::block_index,
    util::{
        construct_deploy,
        rholang::{costacc::close_block_deploy::CloseBlockDeploy, system_deploy_util},
    },
};
use rholang::rust::interpreter::system_processes::BlockData;
use rspace_plus_plus::rspace::{hashing::blake2b256_hash::Blake2b256Hash, merger::merging_logic};
use tokio::sync::OnceCell;

use crate::util::genesis_builder::{GenesisBuilder, GenesisContext};
use crate::util::rholang::resources::{
    copy_storage, mk_runtime_manager_at, mk_test_rnode_store_manager,
};

static GENESIS: OnceCell<GenesisContext> = OnceCell::const_new();

async fn get_genesis() -> &'static GenesisContext {
    GENESIS
        .get_or_init(|| async {
            GenesisBuilder::new()
                .build_genesis_with_parameters(None)
                .await
                .expect("Failed to build genesis")
        })
        .await
}

/**
 * Two deploys inside single state transition are using the same PVV for precharge and refund.
 * So this should be dependent over produce that puts new value into PVV balance in the first deploy.
 * TODO adjust this once/if there is a solution to make deploys touching the same PVV non dependent
 */
#[tokio::test]
async fn two_deploys_executed_inside_single_state_transition_should_be_dependent() {
    let genesis_context = get_genesis().await;
    let (storage_path, _temp_dir) = copy_storage(genesis_context.storage_directory.clone());
    let kvm = mk_test_rnode_store_manager(storage_path);
    let mut runtime_manager = mk_runtime_manager_at(kvm, None).await;

    let base_state = genesis_context
        .genesis_block
        .body
        .state
        .post_state_hash
        .clone();
    let payer1_key = &genesis_context.genesis_vaults[0].0;
    let payer2_key = &genesis_context.genesis_vaults[1].0;
    let state_transition_creator = &genesis_context.validator_key_pairs[0].1;
    let seq_num = 1;
    let block_num = 1;

    let d1 = construct_deploy::source_deploy_now_full(
        "Nil".to_string(),
        None,
        None,
        Some(payer1_key.clone()),
        None,
        None,
    )
    .unwrap();

    let d2 = construct_deploy::source_deploy_now_full(
        "Nil".to_string(),
        None,
        None,
        Some(payer2_key.clone()),
        None,
        None,
    )
    .unwrap();

    let block_data = BlockData {
        time_stamp: d1.data.time_stamp,
        seq_num,
        block_number: block_num,
        sender: state_transition_creator.clone(),
    };

    let invalid_blocks = HashMap::new();
    let user_deploys = vec![d1, d2];
    let system_deploys = vec![CloseBlockDeploy {
        initial_rand: system_deploy_util::generate_close_deploy_random_seed_from_pk(
            state_transition_creator.clone(),
            seq_num,
        ),
    }];

    let (post_state_hash, processed_deploys, _) = runtime_manager
        .compute_state(
            &base_state,
            user_deploys,
            system_deploys,
            block_data,
            Some(invalid_blocks),
        )
        .await
        .unwrap();

    assert_eq!(processed_deploys.len(), 2);

    let mergeable_channels = runtime_manager
        .load_mergeable_channels(
            &post_state_hash,
            state_transition_creator.bytes.clone(),
            seq_num,
        )
        .unwrap();

    // Combine processed deploys with cached mergeable channels data
    let processed_deploys_with_mergeable = processed_deploys
        .to_vec()
        .into_iter()
        .zip(mergeable_channels)
        .collect::<Vec<_>>();

    let idxs = processed_deploys_with_mergeable
        .into_iter()
        .map(|(d, merge_chs)| {
            block_index::create_event_log_index(
                d.deploy_log,
                runtime_manager.get_history_repo(),
                &Blake2b256Hash::from_bytes_prost(&base_state),
                merge_chs,
            )
        })
        .collect::<Vec<_>>();

    let first_depends = merging_logic::depends(&idxs[0], &idxs[1]);
    let second_depends = merging_logic::depends(&idxs[1], &idxs[0]);
    let conflicts = merging_logic::are_conflicting(&idxs[0], &idxs[1]);

    let deploy_chains = merging_logic::compute_related_sets(
        &idxs.iter().cloned().collect(),
        merging_logic::depends,
    );

    // deploys inside one state transition never conflict, as executed in a sequence (for now)
    assert!(!conflicts);
    // first deploy does not depend on the second
    assert!(!first_depends);
    // second deploy depends on the first, as it consumes produce put by first one when updating per validator vault balance
    assert!(!second_depends);
    // deploys should be be put in separate deploy chains
    assert_eq!(deploy_chains.0.len(), 2);
}
