// See casper/src/test/scala/coop/rchain/casper/api/BondedStatusAPITest.scala

use casper::rust::api::block_api::BlockAPI;
use casper::rust::engine::engine_cell::EngineCell;
use casper::rust::engine::engine_with_casper::EngineWithCasper;
use casper::rust::multi_parent_casper_impl::MultiParentCasperImpl;
use casper::rust::util::construct_deploy;
use casper::rust::util::construct_deploy::{DEFAULT_PUB, DEFAULT_SEC};
use crypto::rust::public_key::PublicKey;
use crypto::rust::signatures::secp256k1::Secp256k1;
use crypto::rust::signatures::signatures_alg::SignaturesAlg;
use std::collections::HashMap;
use std::sync::Arc;

use crate::helper::bonding_util;
use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::{GenesisBuilder, GenesisContext, DEFAULT_VALIDATOR_KEY_PAIRS};

struct TestContext {
    genesis: GenesisContext,
}

impl TestContext {
    async fn new() -> Self {
        // Scala: buildGenesisParameters(
        //   defaultValidatorKeyPairs.take(3) :+ ConstructDeploy.defaultKeyPair,
        //   createBonds(defaultValidatorPks.take(3))
        // )
        // This means:
        // - First 3 validators: random keys from defaultValidatorKeyPairs (bonded)
        // - 4th validator (n4): ConstructDeploy.defaultKeyPair = (DEFAULT_SEC, DEFAULT_PUB)
        //   This matches genesisVaults[0] which has 9,000,000 REV, allowing n4 to pay for bonding

        let validator_key_pairs = vec![
            DEFAULT_VALIDATOR_KEY_PAIRS[0].clone(),
            DEFAULT_VALIDATOR_KEY_PAIRS[1].clone(),
            DEFAULT_VALIDATOR_KEY_PAIRS[2].clone(),
            (DEFAULT_SEC.clone(), DEFAULT_PUB.clone()), // n4 uses DEFAULT keypair to match genesisVaults[0]
        ];

        // Extract public keys for bonds
        let validator_pks: Vec<PublicKey> = validator_key_pairs
            .iter()
            .map(|(_, pk)| pk.clone())
            .collect();

        // Create bonds for first 3 validators only (n4 is unbonded)
        let bonds: HashMap<PublicKey, i64> = validator_pks
            .iter()
            .take(3)
            .enumerate()
            .map(|(i, pk)| (pk.clone(), 2 * i as i64 + 1))
            .collect();

        let parameters = GenesisBuilder::build_genesis_parameters(validator_key_pairs, &bonds);
        let genesis = GenesisBuilder::new()
            .build_genesis_with_parameters(Some(parameters))
            .await
            .expect("Failed to build genesis");

        Self { genesis }
    }
}

/// Creates an EngineCell with EngineWithCasper from a TestNode's casper instance
/// Scala equivalent:
///   val engine = new EngineWithCasper[Task](node.casperEff)
///   Cell.mvarCell[Task, Engine[Task]](engine).flatMap { implicit engineCell => ... }
async fn bonded_status(public_key: &PublicKey, node: &TestNode) -> bool {
    // Create engine and engine_cell (Scala lines 40-41)
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
        heartbeat_signal_ref: casper::rust::heartbeat_signal::new_heartbeat_signal_ref(),
    });
    let engine = EngineWithCasper::new(casper_for_engine);
    let engine_cell = EngineCell::init();
    engine_cell.set(Arc::new(engine)).await;

    let public_key_bytes = public_key.bytes.to_vec();
    BlockAPI::bond_status(&engine_cell, &public_key_bytes)
        .await
        .expect("bondStatus should not fail")
}

#[tokio::test]
async fn bond_status_should_return_true_for_bonded_validator() {
    let ctx = TestContext::new().await;

    let nodes = TestNode::create_network(ctx.genesis.clone(), 3, None, None, None, None)
        .await
        .unwrap();

    let n1_pk = nodes[0]
        .validator_id_opt
        .as_ref()
        .unwrap()
        .public_key
        .clone();
    assert_eq!(
        bonded_status(&n1_pk, &nodes[0]).await,
        true,
        "n1 should be bonded"
    );

    let n2_pk = nodes[1]
        .validator_id_opt
        .as_ref()
        .unwrap()
        .public_key
        .clone();
    assert_eq!(
        bonded_status(&n2_pk, &nodes[0]).await,
        true,
        "n2 should be bonded"
    );

    let n3_pk = nodes[2]
        .validator_id_opt
        .as_ref()
        .unwrap()
        .public_key
        .clone();
    assert_eq!(
        bonded_status(&n3_pk, &nodes[0]).await,
        true,
        "n3 should be bonded"
    );
}

#[tokio::test]
async fn bond_status_should_return_false_for_not_bonded_validators() {
    let ctx = TestContext::new().await;

    let node = TestNode::standalone(ctx.genesis.clone()).await.unwrap();

    let secp256k1 = Secp256k1;
    let (_, public_key) = secp256k1.new_key_pair();

    assert_eq!(
        bonded_status(&public_key, &node).await,
        false,
        "Unbonded validator should return false"
    );
}

#[tokio::test]
async fn bond_status_should_return_true_for_newly_bonded_validator() {
    let ctx = TestContext::new().await;

    let mut nodes = TestNode::create_network(ctx.genesis.clone(), 4, None, None, None, None)
        .await
        .unwrap();

    let mut produce_deploys = Vec::new();
    for i in 0..3 {
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        let deploy = construct_deploy::basic_deploy_data(
            i,
            None,
            Some(ctx.genesis.genesis_block.shard_id.clone()),
        )
        .unwrap();
        produce_deploys.push(deploy);
    }

    tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
    let n4_private_key = nodes[3]
        .validator_id_opt
        .as_ref()
        .unwrap()
        .private_key
        .clone();
    let bond_deploy = bonding_util::bonding_deploy(
        1000,
        &n4_private_key,
        Some(ctx.genesis.genesis_block.shard_id.clone()),
    )
    .unwrap();

    let n4_pk = nodes[3]
        .validator_id_opt
        .as_ref()
        .unwrap()
        .public_key
        .clone();

    // Scala line 81: n4 is not bonded initially
    assert_eq!(
        bonded_status(&n4_pk, &nodes[0]).await,
        false,
        "n4 should not be bonded initially"
    );

    let _b1 = TestNode::propagate_block_at_index(&mut nodes, 0, &[bond_deploy])
        .await
        .unwrap();

    let _b2 = TestNode::propagate_block_at_index(&mut nodes, 1, &[produce_deploys[0].clone()])
        .await
        .unwrap();

    assert_eq!(
        bonded_status(&n4_pk, &nodes[0]).await,
        false,
        "n4 should not be bonded yet (b1 not finalized)"
    );

    let _b3 = TestNode::propagate_block_at_index(&mut nodes, 2, &[produce_deploys[1].clone()])
        .await
        .unwrap();

    let _b4 = TestNode::propagate_block_at_index(&mut nodes, 0, &[produce_deploys[2].clone()])
        .await
        .unwrap();

    assert_eq!(
        bonded_status(&n4_pk, &nodes[0]).await,
        true,
        "n4 should be bonded now (b1 finalized)"
    );
}
