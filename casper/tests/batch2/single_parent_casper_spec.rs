// See casper/src/test/scala/coop/rchain/casper/batch2/SingleParentCasperSpec.scala

use casper::rust::block_status::{BlockError, InvalidBlock};
use casper::rust::casper::Casper;
use casper::rust::util::construct_deploy;
use casper::rust::validate::Validate;
use rspace_plus_plus::rspace::history::Either;
use tokio::sync::OnceCell;

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::{GenesisBuilder, GenesisContext};

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
async fn single_parent_casper_should_create_blocks_with_a_single_parent() {
    let genesis = get_genesis().await.clone();

    let mut nodes = TestNode::create_network(genesis.clone(), 2, None, Some(1), None, None)
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
            Some(genesis.genesis_block.shard_id.clone()),
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

// TODO This port is not a 1:1 match to the Scala logic; explanation below.
// Scala test uses: b1 <- n1.addBlock(deployDatas(0))
// This translates to: create_block_unsafe + add_block(block)
//
// PROBLEM: This approach doesn't work in Rust because each TestNode has isolated storage.
// In Scala: copyStorage creates separate directories, BUT blocks added via addBlock are stored
//           only in the local node's DAG storage. The test works because Validate.parents
//           might have different behavior or there's a shared storage mechanism we're missing.
//
// Alternative 1: Use add_block_from_deploys (convenience method)
// PROBLEM: Same issue - block only exists in the creating node's DAG storage.
//          When Validate.parents runs on nodes[0], it can't find b2 (created on nodes[1]).
//          Error: "DAG storage is missing hash..."
//
// Alternative 2: Use propagate_block instead of add_block
// Propagates blocks to all nodes, ensuring they're in all DAG storages.
// ✅NOTE: Changes test semantics - Scala uses addBlock, not propagateBlock.
//       Test uses this approach and passes successfully.
//
// Alternative WORKING SOLUTION: Manually insert blocks into both nodes' storage
// Example:
//   let b1 = {
//       let block = nodes[0].create_block_unsafe(&[deploy_datas[0].clone()]).await.unwrap();
//       nodes[0].add_block(block.clone()).await.unwrap();
//       nodes[0].block_store.put(block.block_hash.clone(), &block).unwrap();
//       nodes[0].block_dag_storage.insert(&block, false, false).unwrap();
//       nodes[1].block_store.put(block.block_hash.clone(), &block).unwrap();
//       nodes[1].block_dag_storage.insert(&block, false, false).unwrap();
//       block
//   };
//
#[tokio::test]
async fn should_reject_multi_parent_blocks() {
    let genesis = get_genesis().await.clone();

    let mut nodes = TestNode::create_network(genesis.clone(), 2, None, Some(1), None, None)
        .await
        .unwrap();

    // Note: We create deploys one by one with sleep to ensure unique timestamps.
    let mut deploy_datas = Vec::new();
    for i in 0..=2 {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        let deploy = construct_deploy::basic_deploy_data(
            i,
            None,
            Some(genesis.genesis_block.shard_id.clone()),
        )
        .unwrap();
        deploy_datas.push(deploy);
    }

    let b1 = TestNode::propagate_block_at_index(&mut nodes, 0, &[deploy_datas[0].clone()])
        .await
        .unwrap();

    let b2 = TestNode::propagate_block_at_index(&mut nodes, 1, &[deploy_datas[1].clone()])
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

    let b3 = TestNode::propagate_block_at_index(&mut nodes, 1, &[deploy_datas[2].clone()])
        .await
        .unwrap();

    let two_parents = vec![b2.block_hash.clone(), b1.block_hash.clone()];

    let dual_parent_b3 = {
        let mut modified_b3 = b3.clone();
        modified_b3.header.parents_hash_list = two_parents;
        modified_b3
    };

    let mut snapshot = nodes[0].casper.get_snapshot().await.unwrap();

    let validate_result = Validate::parents(
        &dual_parent_b3,
        &genesis.genesis_block,
        &mut snapshot,
        &nodes[0].casper.estimator,
    )
    .await;

    assert_eq!(
        validate_result,
        Either::Left(BlockError::Invalid(InvalidBlock::InvalidParents)),
        "Block with multiple parents should be rejected as InvalidParents"
    );
}
