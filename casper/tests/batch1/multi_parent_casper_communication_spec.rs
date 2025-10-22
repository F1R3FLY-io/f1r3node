// See casper/src/test/scala/coop/rchain/casper/batch1/MultiParentCasperCommunicationSpec.scala

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::GenesisBuilder;
use casper::rust::util::construct_deploy;

#[tokio::test]
async fn multi_parent_casper_should_ask_peers_for_blocks_it_is_missing() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let mut nodes = TestNode::create_network(genesis.clone(), 3, None, None, None, None)
        .await
        .unwrap();

    let shard_id = genesis.genesis_block.shard_id.clone();

    let deploy1 = construct_deploy::source_deploy_now(
        "for(_ <- @1){ Nil } | @1!(1)".to_string(),
        None,
        None,
        Some(shard_id.clone()),
    )
    .unwrap();

    let signed_block1 = nodes[0]
        .add_block_from_deploys(&[deploy1])
        .await
        .unwrap();

    nodes[1].handle_receive().await.unwrap();

    nodes[2].shutoff().unwrap(); // nodes(2) misses this block

    let deploy2 = construct_deploy::source_deploy_now(
        "@2!(2)".to_string(),
        None,
        None,
        Some(shard_id.clone()),
    )
    .unwrap();

    let signed_block2 = nodes[0]
        .add_block_from_deploys(&[deploy2])
        .await
        .unwrap();

    // signedBlock2 has signedBlock1 as a dependency
    // When node(2) tries to add signedBlock2, it should request signedBlock1
    let _ = nodes[2].add_block(signed_block2.clone()).await;

    // Scala: r <- nodes(2).requestedBlocks.get.map(v => v.get(signedBlock1.blockHash)).map { ... }
    // Check if signedBlock1 is in requestedBlocks of node(2)
    // TestNode.requested_blocks should be shared with BlockRetriever.requested_blocks
    let is_requested = {
        let state = nodes[2].requested_blocks.lock().unwrap();
        state.contains_key(&signed_block1.block_hash)
    };

    assert!(
        is_requested,
        "signedBlock1 should be in requestedBlocks of node(2)"
    );
}

