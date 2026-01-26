// See casper/src/test/scala/coop/rchain/casper/api/LastFinalizedAPITest.scala

use casper::rust::api::block_api::BlockAPI;
use casper::rust::casper::MultiParentCasper;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::engine::engine_with_casper::EngineWithCasper;
use casper::rust::multi_parent_casper_impl::MultiParentCasperImpl;
use casper::rust::util::{construct_deploy, proto_util};
use crypto::rust::public_key::PublicKey;
use models::rust::casper::protocol::casper_message::BlockMessage;
use std::collections::HashMap;
use std::sync::Arc;

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::{GenesisBuilder, GenesisContext};

/// Test fixture that holds common test data for LastFinalizedAPITest
struct TestContext {
    genesis: GenesisContext,
}

impl TestContext {
    async fn new() -> Self {
        // Note: validatorsNum not specified, so uses default = 4
        // But zip with List(10, 10, 10) means only first 3 validators get bonds
        fn bonds_function(validators: Vec<PublicKey>) -> HashMap<PublicKey, i64> {
            validators
                .into_iter()
                .zip(vec![10i64, 10i64, 10i64])
                .collect()
        }

        let parameters = GenesisBuilder::build_genesis_parameters_with_defaults(
            Some(bonds_function),
            None, // Use default validatorsNum = 4, matching Scala behavior
        );
        let genesis = GenesisBuilder::new()
            .build_genesis_with_parameters(Some(parameters))
            .await
            .expect("Failed to build genesis");

        Self { genesis }
    }
}

/// Creates an EngineCell with EngineWithCasper from a TestNode's casper instance
/// Equivalent to Scala: val engine = new EngineWithCasper[Task](n1.casperEff)
///                       engineCell <- Cell.mvarCell[Task, Engine[Task]](engine)
async fn create_engine_cell(node: &TestNode) -> EngineCell {
    let casper_for_engine = Arc::new(MultiParentCasperImpl {
        block_retriever: node.casper.block_retriever.clone(),
        event_publisher: node.casper.event_publisher.clone(),
        runtime_manager: node.casper.runtime_manager.clone(),
        estimator: node.casper.estimator.clone(),
        block_store: node.casper.block_store.clone(),
        block_dag_storage: node.casper.block_dag_storage.clone(),
        deploy_storage: node.casper.deploy_storage.clone(),
        casper_buffer_storage: node.casper.casper_buffer_storage.clone(),
        validator_id: node.casper.validator_id.clone(),
        casper_shard_conf: node.casper.casper_shard_conf.clone(),
        approved_block: node.casper.approved_block.clone(),
        finalization_in_progress: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        heartbeat_signal_ref: casper::rust::heartbeat_signal::new_heartbeat_signal_ref(),
    });
    let engine = EngineWithCasper::new(casper_for_engine);
    let engine_cell = EngineCell::init();
    engine_cell.set(Arc::new(engine)).await;
    engine_cell
}

async fn is_finalized(block: &BlockMessage, engine_cell: &EngineCell) -> bool {
    let block_hash_str = hex::encode(proto_util::hash_string(block));
    BlockAPI::is_finalized(engine_cell, &block_hash_str)
        .await
        .expect("isFinalized should not fail")
}

/*
 * DAG Looks like this:
 *
 *           b7
 *           |
 *           b6
 *           |
 *           b5 <- last finalized block
 *         / |
 *        |  b4
 *        |  |
 *       b2  b3
 *         \ |
 *           b1
 *           |
 *         genesis
 */
#[tokio::test]
#[ignore = "Scala ignore"]
async fn is_finalized_should_return_true_for_ancestors_of_last_finalized_block() {
    let ctx = TestContext::new().await;

    let mut nodes = TestNode::create_network(ctx.genesis.clone(), 3, None, None, None, None)
        .await
        .unwrap();

    // Note: We create deploys one by one with sleep to ensure unique timestamps.
    // In Scala, the Time effect provides unique timestamps automatically,
    // but in Rust we need to explicitly wait between deploys to avoid NoNewDeploys error.
    let mut produce_deploys = Vec::new();
    for i in 0..7 {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        let deploy = construct_deploy::basic_deploy_data(
            i,
            None,
            Some(ctx.genesis.genesis_block.shard_id.clone()),
        )
        .unwrap();
        produce_deploys.push(deploy);
    }

    let _b1 = TestNode::propagate_block_at_index(&mut nodes, 0, &[produce_deploys[0].clone()])
        .await
        .unwrap();

    let b2 = nodes[1]
        .publish_block(&[produce_deploys[1].clone()], &mut [])
        .await
        .unwrap();

    // Split nodes to work around borrow checker: n3.propagateBlock(...)(n1)
    let (nodes_before_2, nodes_from_2) = nodes.split_at_mut(2);
    let (node_2, _nodes_after_2) = nodes_from_2.split_at_mut(1);
    let _b3 = node_2[0]
        .propagate_block(&[produce_deploys[2].clone()], &mut [&mut nodes_before_2[0]])
        .await
        .unwrap();

    let b4 = TestNode::propagate_block_at_index(&mut nodes, 0, &[produce_deploys[3].clone()])
        .await
        .unwrap();

    let b5 = TestNode::propagate_block_at_index(&mut nodes, 1, &[produce_deploys[4].clone()])
        .await
        .unwrap();

    let _b6 = TestNode::propagate_block_at_index(&mut nodes, 0, &[produce_deploys[5].clone()])
        .await
        .unwrap();

    let _b7 = TestNode::propagate_block_at_index(&mut nodes, 1, &[produce_deploys[6].clone()])
        .await
        .unwrap();

    let last_finalized_block = nodes[0].casper.last_finalized_block().await.unwrap();

    let b5_block_hash = proto_util::hash_string(&b5);
    assert_eq!(
        proto_util::hash_string(&last_finalized_block),
        b5_block_hash,
        "Expected last finalized block to be b5"
    );

    let engine_cell = create_engine_cell(&nodes[0]).await;

    assert_eq!(
        is_finalized(&b5, &engine_cell).await,
        true,
        "b5 should be finalized"
    );

    assert_eq!(
        is_finalized(&b4, &engine_cell).await,
        true,
        "b4 (parent of b5) should be finalized"
    );

    assert_eq!(
        is_finalized(&b2, &engine_cell).await,
        true,
        "b2 (secondary parent of b5) should be finalized"
    );
}

/*
 * DAG Looks like this:
 *
 *           b5
 *             \
 *              b4
 *             /
 *        b7 b3 <- last finalized block
 *        |    \
 *        b6    b2
 *          \  /
 *           b1
 *       [n3 n1 n2]
 *           |
 *         genesis
 */
#[tokio::test]
async fn should_return_false_for_children_uncles_and_cousins_of_last_finalized_block() {
    let ctx = TestContext::new().await;

    let mut nodes = TestNode::create_network(ctx.genesis.clone(), 3, None, None, None, None)
        .await
        .unwrap();

    // Note: We create deploys one by one with sleep to ensure unique timestamps.
    let mut produce_deploys = Vec::new();
    for i in 0..7 {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        let deploy = construct_deploy::basic_deploy_data(
            i,
            None,
            Some(ctx.genesis.genesis_block.shard_id.clone()),
        )
        .unwrap();
        produce_deploys.push(deploy);
    }

    let _b1 = TestNode::propagate_block_at_index(&mut nodes, 0, &[produce_deploys[0].clone()])
        .await
        .unwrap();

    let _b2 = TestNode::propagate_block_to_one(&mut nodes, 1, 0, &[produce_deploys[1].clone()])
        .await
        .unwrap();

    let b3 = TestNode::propagate_block_to_one(&mut nodes, 0, 1, &[produce_deploys[2].clone()])
        .await
        .unwrap();

    let b4 = TestNode::propagate_block_to_one(&mut nodes, 1, 0, &[produce_deploys[3].clone()])
        .await
        .unwrap();

    let _b5 = TestNode::propagate_block_to_one(&mut nodes, 0, 1, &[produce_deploys[4].clone()])
        .await
        .unwrap();

    nodes[2].tle.test_network().clear(&nodes[2].local).unwrap(); // n3 misses b2, b3, b4, b5

    let b6 = TestNode::propagate_block_at_index(&mut nodes, 2, &[produce_deploys[5].clone()])
        .await
        .unwrap();

    let b7 = TestNode::propagate_block_at_index(&mut nodes, 2, &[produce_deploys[6].clone()])
        .await
        .unwrap();

    let last_finalized_block = nodes[0].casper.last_finalized_block().await.unwrap();

    let b3_block_hash = proto_util::hash_string(&b3);
    assert_eq!(
        proto_util::hash_string(&last_finalized_block),
        b3_block_hash,
        "Expected last finalized block to be b3"
    );

    let engine_cell = create_engine_cell(&nodes[0]).await;

    assert_eq!(
        is_finalized(&b4, &engine_cell).await,
        false,
        "b4 (child of b3) should not be finalized"
    );

    assert_eq!(
        is_finalized(&b6, &engine_cell).await,
        false,
        "b6 (uncle of b3) should not be finalized"
    );

    assert_eq!(
        is_finalized(&b7, &engine_cell).await,
        false,
        "b7 (cousin of b3) should not be finalized"
    );
}
