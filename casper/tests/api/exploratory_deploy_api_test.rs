// See casper/src/test/scala/coop/rchain/casper/api/ExploratoryDeployAPITest.scala
//
// Tests for the exploratory deploy API, which allows read-only queries
// against the blockchain state.

use std::collections::HashMap;

use casper::rust::api::block_api::BlockAPI;
use casper::rust::util::construct_deploy;
use crypto::rust::public_key::PublicKey;

use crate::helper::test_node::TestNode;
use crate::util::genesis_builder::GenesisBuilder;

/// Creates genesis parameters with equal bond stakes (10 each) for 3 validators.
fn bonds_function(validators: Vec<PublicKey>) -> HashMap<PublicKey, i64> {
    validators
        .into_iter()
        .zip(vec![10i64, 10i64, 10i64])
        .collect()
}

/// Exploratory deploy should get data from the read-only node.
///
/// DAG structure for finalization:
/// With 3 validators at 10 stake each (total 30), finalization requires >15 stake.
///
///     n1: genesis -> b1 -> b2
///     n2: genesis ---------> b3 (main parent: b2)
///     n3: genesis ---------> b4 (main parent: b3)
///
/// After b3 and b4, b3 accumulates 20 stake (n2 + n3) and is finalized.
#[tokio::test]
async fn exploratory_deploy_should_get_data_from_read_only_node() {
    // Build genesis with 3 validators at 10 stake each
    let parameters = GenesisBuilder::build_genesis_parameters_with_defaults(
        Some(bonds_function),
        None, // Use default validatorsNum = 4
    );
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(Some(parameters))
        .await
        .expect("Failed to build genesis");

    // Create network with 3 validators + 1 read-only node
    let mut nodes = TestNode::create_network(
        genesis.clone(),
        3,       // network_size: 3 bonded validators
        None,    // synchrony_constraint_threshold
        None,    // max_number_of_parents
        None,    // max_parent_depth
        Some(1), // with_read_only_size: 1 read-only node
    )
    .await
    .expect("Failed to create network");

    let shard_id = genesis.genesis_block.shard_id.clone();
    let stored_data = "data";

    // Create deploys
    // putDataDeploy stores data at @"store"
    let put_data_deploy = construct_deploy::source_deploy(
        format!(r#"@"store"!("{}")"#, stored_data),
        1,
        None,
        None,
        None,
        None,
        Some(shard_id.clone()),
    )
    .expect("Failed to create put data deploy");

    // produceDeploys for subsequent blocks
    let produce_deploy_0 = construct_deploy::source_deploy(
        "new x in { x!(0) }".to_string(),
        2,
        None,
        None,
        None,
        None,
        Some(shard_id.clone()),
    )
    .expect("Failed to create produce deploy 0");

    let produce_deploy_1 = construct_deploy::source_deploy(
        "new x in { x!(1) }".to_string(),
        3,
        None,
        None,
        None,
        None,
        Some(shard_id.clone()),
    )
    .expect("Failed to create produce deploy 1");

    let produce_deploy_2 = construct_deploy::source_deploy(
        "new x in { x!(2) }".to_string(),
        4,
        None,
        None,
        None,
        None,
        Some(shard_id.clone()),
    )
    .expect("Failed to create produce deploy 2");

    // b1: n1 creates block with putDataDeploy and propagates to all
    let _b1 = TestNode::propagate_block_at_index(&mut nodes, 0, &[put_data_deploy])
        .await
        .expect("n1 should create and propagate b1");

    // b2: n1 creates block with produceDeploy(0) and propagates to all
    let b2 = TestNode::propagate_block_at_index(&mut nodes, 0, &[produce_deploy_0])
        .await
        .expect("n1 should create and propagate b2");

    // b3: n2 creates block with produceDeploy(1) and propagates to all
    let b3 = TestNode::propagate_block_at_index(&mut nodes, 1, &[produce_deploy_1])
        .await
        .expect("n2 should create and propagate b3");

    // b4: n3 creates block with produceDeploy(2) and propagates to all
    // This finalizes b3 (n2 + n3 = 20 stake > 15 threshold)
    let _b4 = TestNode::propagate_block_at_index(&mut nodes, 2, &[produce_deploy_2])
        .await
        .expect("n3 should create and propagate b4");

    // Get the read-only node (index 3)
    let read_only_node = &nodes[3];

    // Use node's existing engine_cell instead of creating a new one
    // This ensures we use the same casper instance that processed the blocks
    let engine_cell = &read_only_node.engine_cell;

    // Run exploratory deploy to retrieve stored data
    let exploratory_term = r#"new return in { for (@data <- @"store") { return!(data) } }"#;

    let result = BlockAPI::exploratory_deploy(
        engine_cell,
        exploratory_term.to_string(),
        None,  // block_hash: None means use current DAG tips
        false, // use_pre_state_hash
        false, // dev_mode
    )
    .await;

    // Verify result
    match result {
        Ok((pars, last_finalized_block)) => {
            // Verify we got the stored data back
            assert!(!pars.is_empty(), "Exploratory deploy should return data");

            // The result should contain our stored data "data"
            let result_str = format!("{:?}", pars);
            assert!(
                result_str.contains(stored_data),
                "Result should contain stored data '{}', got: {:?}",
                stored_data,
                pars
            );

            // Verify last finalized block is in the expected finalized set.
            // Depending on parent tie-breaks, either b2 or b3 can be the current LFB here.
            let b2_hash_hex = hex::encode(&b2.block_hash);
            let b3_hash_hex = hex::encode(&b3.block_hash);
            let expected_lfb_hashes = [b2_hash_hex.clone(), b3_hash_hex.clone()];
            if !expected_lfb_hashes.contains(&last_finalized_block.block_hash) {
                let mut saw_expected_lfb = false;
                for _ in 0..20 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
                    let maybe_lfb = BlockAPI::last_finalized_block(engine_cell).await;
                    if let Ok(lfb) = maybe_lfb {
                        if let Some(block_info) = lfb.block_info {
                            if expected_lfb_hashes.contains(&block_info.block_hash) {
                                saw_expected_lfb = true;
                                break;
                            }
                        }
                    }
                }
                assert!(
                    saw_expected_lfb,
                    "Last finalized block should eventually be one of {:?}. observed={}",
                    expected_lfb_hashes, last_finalized_block.block_hash
                );
            }

            tracing::info!(
                "Exploratory deploy result: {:?}, LFB: {}",
                pars,
                last_finalized_block.block_hash
            );
        }
        Err(e) => {
            panic!("Exploratory deploy failed: {:?}", e);
        }
    }
}

/// Exploratory deploy should invoke contract continuations (persistent receives)
/// deployed in a finalized block.
///
/// This mirrors the embers pattern: deploy a contract with a persistent continuation,
/// finalize the block, then invoke the contract via explore-deploy on a read-only node.
///
/// DAG structure (same as the data test):
///     n1: genesis -> b1 -> b2
///     n2: genesis ---------> b3 (main parent: b2)
///     n3: genesis ---------> b4 (main parent: b3)
#[tokio::test]
async fn exploratory_deploy_should_invoke_contract_continuations() {
    let parameters = GenesisBuilder::build_genesis_parameters_with_defaults(
        Some(bonds_function),
        None,
    );
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(Some(parameters))
        .await
        .expect("Failed to build genesis");

    let mut nodes = TestNode::create_network(
        genesis.clone(),
        3,
        None,
        None,
        None,
        Some(1),
    )
    .await
    .expect("Failed to create network");

    let shard_id = genesis.genesis_block.shard_id.clone();

    // b1: deploy a persistent contract at @"test"
    // The `contract` keyword creates a persistent for (i.e., it stays after matching)
    let contract_deploy = construct_deploy::source_deploy(
        r#"contract @"test"(@"get", @key, ret) = { ret!(key) }"#.to_string(),
        1,
        None,
        None,
        None,
        None,
        Some(shard_id.clone()),
    )
    .expect("Failed to create contract deploy");

    let _b1 = TestNode::propagate_block_at_index(&mut nodes, 0, &[contract_deploy])
        .await
        .expect("n1 should create and propagate b1");

    // b2-b4: propagate blocks to finalize b1's state
    let produce_deploy_0 = construct_deploy::source_deploy(
        "new x in { x!(0) }".to_string(),
        2,
        None,
        None,
        None,
        None,
        Some(shard_id.clone()),
    )
    .expect("Failed to create produce deploy 0");

    let _b2 = TestNode::propagate_block_at_index(&mut nodes, 0, &[produce_deploy_0])
        .await
        .expect("n1 should create and propagate b2");

    let produce_deploy_1 = construct_deploy::source_deploy(
        "new x in { x!(1) }".to_string(),
        3,
        None,
        None,
        None,
        None,
        Some(shard_id.clone()),
    )
    .expect("Failed to create produce deploy 1");

    let _b3 = TestNode::propagate_block_at_index(&mut nodes, 1, &[produce_deploy_1])
        .await
        .expect("n2 should create and propagate b3");

    let produce_deploy_2 = construct_deploy::source_deploy(
        "new x in { x!(2) }".to_string(),
        4,
        None,
        None,
        None,
        None,
        Some(shard_id.clone()),
    )
    .expect("Failed to create produce deploy 2");

    let _b4 = TestNode::propagate_block_at_index(&mut nodes, 2, &[produce_deploy_2])
        .await
        .expect("n3 should create and propagate b4");

    // Explore-deploy on the read-only node: invoke the contract
    let read_only_node = &nodes[3];
    let engine_cell = &read_only_node.engine_cell;

    let exploratory_term =
        r#"new return in { @"test"!("get", "hello", *return) }"#;

    let result = BlockAPI::exploratory_deploy(
        engine_cell,
        exploratory_term.to_string(),
        None,
        false,
        false,
    )
    .await;

    match result {
        Ok((pars, _last_finalized_block)) => {
            assert!(
                !pars.is_empty(),
                "Exploratory deploy should return data from contract invocation"
            );

            let result_str = format!("{:?}", pars);
            assert!(
                result_str.contains("hello"),
                "Result should contain 'hello' from contract invocation, got: {:?}",
                pars
            );

            tracing::info!("Contract continuation test passed: {:?}", pars);
        }
        Err(e) => {
            panic!(
                "Exploratory deploy failed to invoke contract continuation: {:?}",
                e
            );
        }
    }
}

/// Exploratory deploy should return error on bonded validator.
///
/// The exploratory deploy API should only work on read-only nodes.
/// When called on a bonded validator, it should return an error.
#[tokio::test]
async fn exploratory_deploy_should_return_error_on_bonded_validator() {
    // Build genesis with default parameters
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    // Create network with 1 bonded validator (no read-only nodes)
    let mut nodes = TestNode::create_network(
        genesis.clone(),
        1,    // network_size: 1 bonded validator
        None, // synchrony_constraint_threshold
        None, // max_number_of_parents
        None, // max_parent_depth
        None, // with_read_only_size: None (no read-only nodes)
    )
    .await
    .expect("Failed to create network");

    let shard_id = genesis.genesis_block.shard_id.clone();

    // Create a deploy and propagate a block
    let produce_deploy = construct_deploy::source_deploy(
        "new x in { x!(0) }".to_string(),
        1,
        None,
        None,
        None,
        None,
        Some(shard_id),
    )
    .expect("Failed to create produce deploy");

    let _b1 = TestNode::propagate_block_at_index(&mut nodes, 0, &[produce_deploy])
        .await
        .expect("n1 should create and propagate b1");

    // Use node's existing engine_cell for the bonded validator (node 0)
    let engine_cell = &nodes[0].engine_cell;

    // Try to run exploratory deploy on bonded validator
    let result = BlockAPI::exploratory_deploy(
        engine_cell,
        "new return in { return!(1) }".to_string(),
        None,  // block_hash
        false, // use_pre_state_hash
        false, // dev_mode: false means read-only check is enforced
    )
    .await;

    // Verify it returns an error
    match result {
        Err(e) => {
            let error_message = format!("{:?}", e);
            assert!(
                error_message
                    .contains("Exploratory deploy can only be executed on read-only RNode"),
                "Expected read-only error message, got: {}",
                error_message
            );
            tracing::info!("Got expected error: {}", error_message);
        }
        Ok(_) => {
            panic!("Exploratory deploy should fail on bonded validator");
        }
    }
}
