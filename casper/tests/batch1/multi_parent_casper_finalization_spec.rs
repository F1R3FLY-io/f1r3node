// See casper/src/test/scala/coop/rchain/casper/batch1/MultiParentCasperFinalizationSpec.scala

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::GenesisBuilder;
use casper::rust::util::construct_deploy;
use crypto::rust::public_key::PublicKey;
use models::rust::casper::protocol::casper_message::BlockMessage;
use std::collections::HashMap;

#[tokio::test]
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
