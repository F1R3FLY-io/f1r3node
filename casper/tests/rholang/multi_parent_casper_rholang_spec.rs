// See casper/src/test/scala/coop/rchain/casper/batch1/MultiParentCasperRholangSpec.scala

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::GenesisBuilder;
use casper::rust::util::{construct_deploy, proto_util, rspace_util};

// Uncomment this to use the debugger on M2
// May need to modify if architecture required is different or if path is different. See ./scripts/build_rust_libraries.sh
// System.setProperty("jna.library.path", "../rspace++/target/x86_64-apple-darwin/debug/")

//put a new casper instance at the start of each
//test since we cannot reset it
#[tokio::test]
async fn multi_parent_casper_should_create_blocks_based_on_deploys() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let mut standalone_node = TestNode::standalone(genesis).await.unwrap();

    let deploy =
        construct_deploy::basic_deploy_data(0, None, Some(standalone_node.genesis.shard_id.clone())).unwrap();
    let block = standalone_node.create_block_unsafe(&[deploy.clone()]).await.unwrap();
    let deploys: Vec<_> = block.body.deploys.iter().map(|pd| &pd.deploy).collect();
    let parents = proto_util::parent_hashes(&block);

    assert_eq!(parents.len(), 1);
    assert_eq!(parents[0], standalone_node.genesis.block_hash);
    assert_eq!(deploys.len(), 1);
    assert_eq!(deploys[0], &deploy);

    let data =
        rspace_util::get_data_at_public_channel_block(&block, 0, &standalone_node.runtime_manager).await;
    assert_eq!(data, vec!["0"]);
}
