// See casper/src/test/scala/coop/rchain/casper/batch1/MultiParentCasperSmokeSpec.scala

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::{GenesisBuilder, GenesisContext};
use casper::rust::util::construct_deploy;
use tokio::sync::OnceCell;

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

#[tokio::test]
async fn multi_parent_casper_should_perform_the_most_basic_deploy_successfully() {
    let genesis = get_genesis().await.clone();

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
