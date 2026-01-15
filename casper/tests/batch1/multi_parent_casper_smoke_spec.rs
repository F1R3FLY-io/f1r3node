// See casper/src/test/scala/coop/rchain/casper/batch1/MultiParentCasperSmokeSpec.scala

use casper::rust::test_utils::helper::test_node::TestNode;
use casper::rust::test_utils::util::genesis_builder::GenesisBuilder;
use casper::rust::util::construct_deploy;

#[tokio::test]
async fn multi_parent_casper_should_perform_the_most_basic_deploy_successfully() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let mut node = TestNode::standalone(genesis.clone()).await.unwrap();

    let deploy = construct_deploy::source_deploy_now(
        "new x in { x!(0) }".to_string(),
        None,
        None,
        Some(genesis.genesis_block.shard_id.clone()),
    )
    .unwrap();

    node.add_block_from_deploys(&[deploy]).await.unwrap();
}

