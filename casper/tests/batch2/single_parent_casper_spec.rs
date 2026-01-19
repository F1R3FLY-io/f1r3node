// See casper/src/test/scala/coop/rchain/casper/batch2/SingleParentCasperSpec.scala

use casper::rust::block_status::{BlockError, InvalidBlock};
use casper::rust::casper::Casper;
use casper::rust::util::construct_deploy;
use casper::rust::validate::Validate;
use rspace_plus_plus::rspace::history::Either;

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::{GenesisBuilder, GenesisContext};

struct TestContext {
    genesis: GenesisContext,
}

impl TestContext {
    async fn new() -> Self {
        let genesis = GenesisBuilder::new()
            .build_genesis_with_parameters(None)
            .await
            .unwrap();

        Self { genesis }
    }
}

#[tokio::test]
async fn single_parent_casper_should_create_blocks_with_a_single_parent() {
    let ctx = TestContext::new().await;

    let mut nodes = TestNode::create_network(ctx.genesis.clone(), 2, None, Some(1), None, None)
        .await
        .unwrap();

    // Note: We create deploys one by one with sleep to ensure unique timestamps.
    // In Scala, the Time effect provides unique timestamps automatically,
    // but in Rust we need to explicitly wait between deploys to avoid NoNewDeploys error.
    let mut deploy_datas = Vec::new();
    for i in 0..=2 {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        let deploy = construct_deploy::basic_deploy_data(
            i,
            None,
            Some(ctx.genesis.genesis_block.shard_id.clone()),
        )
        .unwrap();
        deploy_datas.push(deploy);
    }

    let _b1 = nodes[0]
        .add_block_from_deploys(&[deploy_datas[0].clone()])
        .await
        .unwrap();

    let _b2 = nodes[1]
        .add_block_from_deploys(&[deploy_datas[1].clone()])
        .await
        .unwrap();

    // Note: To work around borrow checker, we need to split nodes array
    let (first_part, second_part) = nodes.split_at_mut(1);
    first_part[0]
        .sync_with_one(&mut second_part[0])
        .await
        .unwrap();

    let (first_part, second_part) = nodes.split_at_mut(1);
    second_part[0]
        .sync_with_one(&mut first_part[0])
        .await
        .unwrap();

    let b3 = nodes[0]
        .add_block_from_deploys(&[deploy_datas[2].clone()])
        .await
        .unwrap();

    assert_eq!(
        b3.header.parents_hash_list.len(),
        1,
        "Block should have exactly one parent"
    );
}

// NOTE: Storage isolation in test nodes
// Both Scala and Rust TestNodes have isolated LMDB storage (via copyStorage).
// When add_block_from_deploys() is called, blocks are stored ONLY in the creating node's storage.
// The syncWith() mechanism exchanges blocks via BlockRequest/BlockMessage protocol.
// This test now correctly matches Scala's approach: addBlock() + syncWith().
#[tokio::test]
async fn should_reject_multi_parent_blocks() {
    let ctx = TestContext::new().await;

    let mut nodes = TestNode::create_network(ctx.genesis.clone(), 2, None, Some(1), None, None)
        .await
        .unwrap();

    // Note: We create deploys one by one with sleep to ensure unique timestamps.
    let mut deploy_datas = Vec::new();
    for i in 0..=2 {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        let deploy = construct_deploy::basic_deploy_data(
            i,
            None,
            Some(ctx.genesis.genesis_block.shard_id.clone()),
        )
        .unwrap();
        deploy_datas.push(deploy);
    }

    let b1 = nodes[0]
        .add_block_from_deploys(&[deploy_datas[0].clone()])
        .await
        .unwrap();

    let b2 = nodes[1]
        .add_block_from_deploys(&[deploy_datas[1].clone()])
        .await
        .unwrap();

    let (first_part, second_part) = nodes.split_at_mut(1);
    first_part[0]
        .sync_with_one(&mut second_part[0])
        .await
        .unwrap();

    let (first_part, second_part) = nodes.split_at_mut(1);
    second_part[0]
        .sync_with_one(&mut first_part[0])
        .await
        .unwrap();

    let b3 = nodes[1]
        .add_block_from_deploys(&[deploy_datas[2].clone()])
        .await
        .unwrap();

    let two_parents = vec![b2.block_hash.clone(), b1.block_hash.clone()];

    let dual_parent_b3 = {
        let mut modified_b3 = b3.clone();
        modified_b3.header.parents_hash_list = two_parents;
        modified_b3
    };

    let mut snapshot = nodes[0].casper.get_snapshot().await.unwrap();

    // max_number_of_parents = 1 means only single parent blocks are allowed
    let max_number_of_parents = 1;
    let validate_result = Validate::parents(
        &dual_parent_b3,
        &ctx.genesis.genesis_block,
        &mut snapshot,
        max_number_of_parents,
    );

    assert_eq!(
        validate_result,
        Either::Left(BlockError::Invalid(InvalidBlock::InvalidParents)),
        "Block with multiple parents should be rejected as InvalidParents when max_number_of_parents=1"
    );
}
