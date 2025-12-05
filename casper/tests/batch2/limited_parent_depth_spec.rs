// See casper/src/test/scala/coop/rchain/casper/batch2/LimitedParentDepthSpec.scala

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::{GenesisBuilder, GenesisContext};
use casper::rust::errors::CasperError;
use casper::rust::util::construct_deploy;
use crypto::rust::signatures::signed::Signed;
use models::rust::casper::protocol::casper_message::DeployData;
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

struct TestContext {
    genesis: GenesisContext,
    produce_deploys: Vec<Signed<DeployData>>,
}

impl TestContext {
    async fn new() -> Self {
        let genesis = get_genesis().await.clone();

        let mut produce_deploys = Vec::new();
        for i in 0..6 {
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            let deploy = construct_deploy::basic_deploy_data(
                i,
                None,
                Some(genesis.genesis_block.shard_id.clone()),
            )
            .unwrap();
            produce_deploys.push(deploy);
        }

        Self {
            genesis,
            produce_deploys,
        }
    }

    async fn create_network(
        &self,
        max_parent_depth: Option<i32>,
    ) -> Result<Vec<TestNode>, CasperError> {
        TestNode::create_network(self.genesis.clone(), 2, None, None, max_parent_depth, None).await
    }
}

#[tokio::test]
async fn estimator_should_obey_present_parent_depth_limitation() {
    let ctx = TestContext::new().await;
    let mut nodes = ctx.create_network(Some(2)).await.unwrap();

    let _b1 = nodes[0]
        .add_block_from_deploys(&[ctx.produce_deploys[0].clone()])
        .await
        .unwrap();

    let _b2 = TestNode::propagate_block_at_index(&mut nodes, 1, &[ctx.produce_deploys[1].clone()])
        .await
        .unwrap();

    let _b3 = TestNode::propagate_block_at_index(&mut nodes, 1, &[ctx.produce_deploys[2].clone()])
        .await
        .unwrap();

    let _b4 = TestNode::propagate_block_at_index(&mut nodes, 1, &[ctx.produce_deploys[3].clone()])
        .await
        .unwrap();

    let b5 = TestNode::propagate_block_at_index(&mut nodes, 1, &[ctx.produce_deploys[4].clone()])
        .await
        .unwrap();

    let b6 = TestNode::propagate_block_at_index(&mut nodes, 0, &[ctx.produce_deploys[5].clone()])
        .await
        .unwrap();

    assert_eq!(
        b6.header.parents_hash_list,
        vec![b5.block_hash.clone()],
        "Expected b6 to have only b5 as parent due to maxParentDepth=2"
    );
}

#[tokio::test]
async fn estimator_should_obey_absent_parent_depth_limitation() {
    let ctx = TestContext::new().await;
    let mut nodes = ctx.create_network(None).await.unwrap();

    let b1 = nodes[0]
        .add_block_from_deploys(&[ctx.produce_deploys[0].clone()])
        .await
        .unwrap();

    let _b2 = TestNode::propagate_block_at_index(&mut nodes, 1, &[ctx.produce_deploys[1].clone()])
        .await
        .unwrap();

    let _b3 = TestNode::propagate_block_at_index(&mut nodes, 1, &[ctx.produce_deploys[2].clone()])
        .await
        .unwrap();

    let _b4 = TestNode::propagate_block_at_index(&mut nodes, 1, &[ctx.produce_deploys[3].clone()])
        .await
        .unwrap();

    let b5 = TestNode::propagate_block_at_index(&mut nodes, 1, &[ctx.produce_deploys[4].clone()])
        .await
        .unwrap();

    let b6 = TestNode::propagate_block_at_index(&mut nodes, 0, &[ctx.produce_deploys[5].clone()])
        .await
        .unwrap();

    assert_eq!(
        b6.header.parents_hash_list.len(),
        2,
        "Expected b6 to have exactly 2 parents"
    );
    assert!(
        b6.header.parents_hash_list.contains(&b1.block_hash),
        "Expected b6 to have b1 as a parent"
    );
    assert!(
        b6.header.parents_hash_list.contains(&b5.block_hash),
        "Expected b6 to have b5 as a parent"
    );
}
