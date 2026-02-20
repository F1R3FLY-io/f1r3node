// See casper/src/test/scala/coop/rchain/casper/batch1/MultiParentCasperFinalizationSpec.scala

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::GenesisBuilder;
use casper::rust::casper::MultiParentCasper;
use casper::rust::util::construct_deploy;
use crypto::rust::public_key::PublicKey;
use models::rust::casper::protocol::casper_message::BlockMessage;
use std::collections::HashMap;

// TODO: Round-robin finalization concept no longer applies with multi-parent merging.
// Scala deleted this test in PR #288.
#[tokio::test]
#[ignore = "Round-robin finalization concept no longer applies with multi-parent blocks"]
async fn multi_parent_casper_should_increment_last_finalized_block_as_appropriate_in_round_robin() {
    fn assert_finalized_block(node: &TestNode, expected: &BlockMessage) {
        let last_finalized_block_hash = node
            .block_dag_storage
            .get_representation()
            .last_finalized_block();

        // Scala uses withClue to add file:line context to assertions.
        // In Rust, assert_eq! automatically shows file and line on failure,
        // so I'll just add helpful hex-encoded block hashes for debugging.
        assert_eq!(
            last_finalized_block_hash,
            expected.block_hash,
            "Last finalized block mismatch\nExpected: {}\nGot: {}",
            hex::encode(&expected.block_hash),
            hex::encode(&last_finalized_block_hash)
        );
    }

    // Bonds function: _.map(pk => pk -> 10L).toMap
    fn bonds_function(validators: Vec<PublicKey>) -> HashMap<PublicKey, i64> {
        validators.into_iter().map(|pk| (pk, 10i64)).collect()
    }

    let parameters = GenesisBuilder::build_genesis_parameters_with_defaults(
        Some(bonds_function),
        None, // Use default validatorsNum = 4
    );

    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(Some(parameters))
        .await
        .expect("Failed to build genesis");

    let mut nodes = TestNode::create_network(genesis.clone(), 3, None, None, None, None)
        .await
        .unwrap();

    let deploy_datas: Vec<_> = (0..=7)
        .map(|i| {
            construct_deploy::basic_deploy_data(
                i,
                None,
                Some(genesis.genesis_block.shard_id.clone()),
            )
            .unwrap()
        })
        .collect();

    let block1 = TestNode::propagate_block_at_index(&mut nodes, 0, &[deploy_datas[0].clone()])
        .await
        .unwrap();

    let block2 = TestNode::propagate_block_at_index(&mut nodes, 1, &[deploy_datas[1].clone()])
        .await
        .unwrap();

    let block3 = TestNode::propagate_block_at_index(&mut nodes, 2, &[deploy_datas[2].clone()])
        .await
        .unwrap();

    let block4 = TestNode::propagate_block_at_index(&mut nodes, 0, &[deploy_datas[3].clone()])
        .await
        .unwrap();

    let _block5 = TestNode::propagate_block_at_index(&mut nodes, 1, &[deploy_datas[4].clone()])
        .await
        .unwrap();

    assert_finalized_block(&nodes[0], &block1);

    let _block6 = TestNode::propagate_block_at_index(&mut nodes, 2, &[deploy_datas[5].clone()])
        .await
        .unwrap();

    assert_finalized_block(&nodes[0], &block2);

    let _block7 = TestNode::propagate_block_at_index(&mut nodes, 0, &[deploy_datas[6].clone()])
        .await
        .unwrap();

    assert_finalized_block(&nodes[0], &block3);

    let _block8 = TestNode::propagate_block_at_index(&mut nodes, 1, &[deploy_datas[7].clone()])
        .await
        .unwrap();

    assert_finalized_block(&nodes[0], &block4);
}

/// This test verifies that finalization advances monotonically (block number never
/// decreases) during round-robin block production with multi-parent merging, and
/// that all validators agree on the last finalized block at the end.
#[tokio::test]
async fn multi_parent_casper_should_advance_finalization_monotonically_in_round_robin() {
    fn bonds_function(validators: Vec<PublicKey>) -> HashMap<PublicKey, i64> {
        validators.into_iter().map(|pk| (pk, 10i64)).collect()
    }

    let parameters = GenesisBuilder::build_genesis_parameters_with_defaults(
        Some(bonds_function),
        None, // Use default validatorsNum = 4
    );

    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(Some(parameters))
        .await
        .expect("Failed to build genesis");

    let mut nodes = TestNode::create_network(genesis.clone(), 3, None, None, None, None)
        .await
        .unwrap();

    let shard_id = genesis.genesis_block.shard_id.clone();

    let deploy_datas: Vec<_> = (0..=7)
        .map(|i| construct_deploy::basic_deploy_data(i, None, Some(shard_id.clone())).unwrap())
        .collect();

    // Round-robin block production across 3 validators with propagation.
    // With multi-parent merging, each block may have multiple parents
    // (one from each validator's latest message).
    let _block1 = TestNode::propagate_block_at_index(&mut nodes, 0, &[deploy_datas[0].clone()])
        .await
        .unwrap();

    let _block2 = TestNode::propagate_block_at_index(&mut nodes, 1, &[deploy_datas[1].clone()])
        .await
        .unwrap();

    let _block3 = TestNode::propagate_block_at_index(&mut nodes, 2, &[deploy_datas[2].clone()])
        .await
        .unwrap();

    let _block4 = TestNode::propagate_block_at_index(&mut nodes, 0, &[deploy_datas[3].clone()])
        .await
        .unwrap();

    let _block5 = TestNode::propagate_block_at_index(&mut nodes, 1, &[deploy_datas[4].clone()])
        .await
        .unwrap();

    // After 5 blocks in round-robin with 3 validators (equal bonds),
    // finalization should have advanced past genesis.
    let lfb_after5 = nodes[0].casper.last_finalized_block().await.unwrap();

    let _block6 = TestNode::propagate_block_at_index(&mut nodes, 2, &[deploy_datas[5].clone()])
        .await
        .unwrap();

    let lfb_after6 = nodes[0].casper.last_finalized_block().await.unwrap();

    let _block7 = TestNode::propagate_block_at_index(&mut nodes, 0, &[deploy_datas[6].clone()])
        .await
        .unwrap();

    let lfb_after7 = nodes[0].casper.last_finalized_block().await.unwrap();

    let _block8 = TestNode::propagate_block_at_index(&mut nodes, 1, &[deploy_datas[7].clone()])
        .await
        .unwrap();

    let lfb_after8 = nodes[0].casper.last_finalized_block().await.unwrap();

    // Verify finalization has advanced past genesis
    assert_ne!(
        lfb_after5.block_hash, genesis.genesis_block.block_hash,
        "LFB after 5 blocks should not be genesis"
    );

    // Verify finalization advances monotonically (block number never decreases)
    assert!(
        lfb_after6.body.state.block_number >= lfb_after5.body.state.block_number,
        "LFB block number should not decrease: after6={} < after5={}",
        lfb_after6.body.state.block_number,
        lfb_after5.body.state.block_number
    );
    assert!(
        lfb_after7.body.state.block_number >= lfb_after6.body.state.block_number,
        "LFB block number should not decrease: after7={} < after6={}",
        lfb_after7.body.state.block_number,
        lfb_after6.body.state.block_number
    );
    assert!(
        lfb_after8.body.state.block_number >= lfb_after7.body.state.block_number,
        "LFB block number should not decrease: after8={} < after7={}",
        lfb_after8.body.state.block_number,
        lfb_after7.body.state.block_number
    );

    // Verify finalization has advanced meaningfully by the end
    // With 8 blocks from 3 validators, LFB should be well past block 1
    assert!(
        lfb_after8.body.state.block_number >= 2,
        "LFB after 8 blocks should be at least block #2, got #{}",
        lfb_after8.body.state.block_number
    );

    // Verify all validators agree on finalization
    let lfb_node1 = nodes[1].casper.last_finalized_block().await.unwrap();
    let lfb_node2 = nodes[2].casper.last_finalized_block().await.unwrap();
    assert_eq!(
        lfb_node1.block_hash, lfb_after8.block_hash,
        "Node1 LFB should match Node0 LFB"
    );
    assert_eq!(
        lfb_node2.block_hash, lfb_after8.block_hash,
        "Node2 LFB should match Node0 LFB"
    );
}
