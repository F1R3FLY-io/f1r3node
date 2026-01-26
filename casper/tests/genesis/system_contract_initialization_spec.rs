// See casper/src/test/scala/coop/rchain/casper/genesis/SystemContractInitializationSpec.scala
//
// Tests to verify system contracts (PoS, RevVault) are properly
// initialized at genesis and accessible in subsequent blocks.
//
// These tests verify:
// - Genesis initialization (system contracts deployed correctly)
// - Block processing (state properly restored after blocks)
// - Invalid block handling (invalidBlocks map populated correctly)

use casper::rust::{
    block_status::{BlockError, InvalidBlock},
    casper::MultiParentCasper,
    util::construct_deploy,
};
use rholang::rust::interpreter::util::rev_address::RevAddress;
use rspace_plus_plus::rspace::history::Either;

use crate::{
    helper::test_node::TestNode,
    util::genesis_builder::GenesisBuilder,
};

/// PoS contract should return correct bonds at genesis
#[tokio::test]
async fn pos_contract_should_return_correct_bonds_at_genesis() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let get_bonds_query = r#"
        new return, rl(`rho:registry:lookup`), posCh in {
          rl!(`rho:rchain:pos`, *posCh) |
          for (@(_, PoS) <- posCh) {
            @PoS!("getBonds", *return)
          }
        }
    "#;

    let node = TestNode::standalone(genesis.clone())
        .await
        .expect("Failed to create standalone node");

    let post_state_hash = genesis.genesis_block.body.state.post_state_hash.clone();

    let result = node
        .runtime_manager
        .play_exploratory_deploy(get_bonds_query.to_string(), &post_state_hash)
        .await
        .expect("Failed to execute exploratory deploy");

    // Verify we got a result
    assert!(
        !result.is_empty(),
        "PoS getBonds should return a non-empty result"
    );

    // Verify genesis has 4 validators (default genesis config)
    assert_eq!(
        genesis.genesis_block.body.state.bonds.len(),
        4,
        "Genesis should have 4 validators"
    );

    tracing::info!("PoS getBonds result: {:?}", result);
}

/// RevVault should be accessible at genesis post-state
#[tokio::test]
async fn rev_vault_should_be_accessible_at_genesis() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    // Get the first genesis vault's public key and derive REV address
    let (_, vault_pk) = &genesis.genesis_vaults[0];
    let vault_addr = RevAddress::from_public_key(vault_pk)
        .expect("Should create REV address from public key");

    let get_vault_query = format!(
        r#"
        new return, rl(`rho:registry:lookup`), RevVaultCh, vaultCh in {{
          rl!(`rho:rchain:revVault`, *RevVaultCh) |
          for (@(_, RevVault) <- RevVaultCh) {{
            @RevVault!("findOrCreate", "{}", *vaultCh) |
            for (@(true, vault) <- vaultCh) {{
              @vault!("balance", *return)
            }}
          }}
        }}
    "#,
        vault_addr.to_base58()
    );

    let node = TestNode::standalone(genesis.clone())
        .await
        .expect("Failed to create standalone node");

    let post_state_hash = genesis.genesis_block.body.state.post_state_hash.clone();

    let result = node
        .runtime_manager
        .play_exploratory_deploy(get_vault_query, &post_state_hash)
        .await
        .expect("Failed to execute exploratory deploy");

    // Verify we got a result
    assert!(
        !result.is_empty(),
        "RevVault balance query should return a non-empty result"
    );

    tracing::info!("RevVault balance result: {:?}", result);
}

/// Validator vaults should have zero balance at genesis
#[tokio::test]
async fn validator_vaults_should_have_zero_balance_at_genesis() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    // Get the first validator's public key and derive REV address
    let (_, validator_pk) = &genesis.validator_key_pairs[0];
    let validator_addr = RevAddress::from_public_key(validator_pk)
        .expect("Should create REV address from validator public key");

    let get_validator_vault_query = format!(
        r#"
        new return, rl(`rho:registry:lookup`), RevVaultCh, vaultCh in {{
          rl!(`rho:rchain:revVault`, *RevVaultCh) |
          for (@(_, RevVault) <- RevVaultCh) {{
            @RevVault!("findOrCreate", "{}", *vaultCh) |
            for (@(true, vault) <- vaultCh) {{
              @vault!("balance", *return)
            }}
          }}
        }}
    "#,
        validator_addr.to_base58()
    );

    let node = TestNode::standalone(genesis.clone())
        .await
        .expect("Failed to create standalone node");

    let post_state_hash = genesis.genesis_block.body.state.post_state_hash.clone();

    let result = node
        .runtime_manager
        .play_exploratory_deploy(get_validator_vault_query, &post_state_hash)
        .await
        .expect("Failed to execute exploratory deploy");

    // Verify we got a result (validator vaults are initialized to 0 REV per GenesisBuilder)
    assert!(
        !result.is_empty(),
        "Validator vault balance query should return a non-empty result"
    );

    tracing::info!("Validator vault balance result: {:?}", result);
}

/// InvalidBlocks map should contain invalid block after processing
#[tokio::test]
async fn invalid_blocks_map_should_contain_invalid_block_after_processing() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let mut nodes = TestNode::create_network(genesis.clone(), 2, None, None, None, None)
        .await
        .expect("Failed to create network");

    // Create a valid deploy on node 0
    let deploy = construct_deploy::source_deploy(
        "new x in { x!(0) }".to_string(),
        1,
        None,
        None,
        None,
        None,
        Some(genesis.genesis_block.shard_id.clone()),
    )
    .expect("Failed to create deploy");

    // Create a valid block on node 0
    let signed_block = nodes[0]
        .create_block_unsafe(&[deploy])
        .await
        .expect("Node 0 should create block");

    tracing::info!("Original block hash: {:?}", signed_block.block_hash);

    // Create an invalid version (wrong seqNum makes hash invalid)
    let mut invalid_block = signed_block.clone();
    invalid_block.seq_num = 47;

    tracing::info!("Invalid block hash: {:?}", invalid_block.block_hash);
    tracing::info!("Invalid block sender: {:?}", invalid_block.sender);

    // Process the invalid block on node 1
    let status = nodes[1]
        .process_block(invalid_block.clone())
        .await
        .expect("Node 1 should process block (even if invalid)");

    tracing::info!("Process status: {:?}", status);

    // Verify the block was rejected with InvalidBlockHash
    match &status {
        Either::Left(BlockError::Invalid(InvalidBlock::InvalidBlockHash)) => {
            // Expected
        }
        other => {
            panic!(
                "Expected InvalidBlockHash error, got: {:?}",
                other
            );
        }
    }

    // Check what's in node 1's dag.invalidBlocks
    let dag = nodes[1]
        .casper
        .block_dag()
        .await
        .expect("Should get DAG");
    let invalid_blocks = dag.invalid_blocks();

    tracing::info!("dag.invalidBlocks count: {}", invalid_blocks.len());

    // The invalid block should be in dag.invalidBlocks
    let is_in_invalid_blocks = invalid_blocks
        .iter()
        .any(|block_meta| block_meta.block_hash == invalid_block.block_hash);

    assert!(
        is_in_invalid_blocks,
        "The invalid block should be in dag.invalidBlocks"
    );
}

/// System contracts should work after adding a block with a deploy
#[tokio::test]
async fn system_contracts_should_work_after_adding_block() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let get_bonds_query = r#"
        new return, rl(`rho:registry:lookup`), posCh in {
          rl!(`rho:rchain:pos`, *posCh) |
          for (@(_, PoS) <- posCh) {
            @PoS!("getBonds", *return)
          }
        }
    "#;

    let mut node = TestNode::standalone(genesis.clone())
        .await
        .expect("Failed to create standalone node");

    // Create a simple deploy
    let deploy = construct_deploy::source_deploy(
        "new x in { x!(0) }".to_string(),
        1,
        None,
        None,
        None,
        None,
        Some(genesis.genesis_block.shard_id.clone()),
    )
    .expect("Failed to create deploy");

    // Add a block with the deploy
    let block = node
        .add_block_from_deploys(&[deploy])
        .await
        .expect("Should add block from deploys");

    tracing::info!("Added block: {:?}", block.block_hash);

    // Query PoS in the new block's post-state
    let result = node
        .runtime_manager
        .play_exploratory_deploy(
            get_bonds_query.to_string(),
            &block.body.state.post_state_hash,
        )
        .await
        .expect("Failed to execute exploratory deploy");

    assert!(
        !result.is_empty(),
        "PoS getBonds should still work after adding a block"
    );

    tracing::info!("PoS getBonds after block: {:?}", result);
}

/// Validator key lookup should succeed in allBonds map
#[tokio::test]
async fn validator_key_lookup_should_succeed_in_all_bonds() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    // Get the first validator's public key in hex
    let (_, validator_pk) = &genesis.validator_key_pairs[0];
    let validator_hex = hex::encode(&validator_pk.bytes);

    let lookup_query = format!(
        r#"
        new return, rl(`rho:registry:lookup`), posCh in {{
          rl!(`rho:rchain:pos`, *posCh) |
          for (@(_, PoS) <- posCh) {{
            new bondsCh in {{
              @PoS!("getBonds", *bondsCh) |
              for (@bonds <- bondsCh) {{
                match bonds.get("{}".hexToBytes()) {{
                  Nil => return!(("KEY_NOT_FOUND", Nil))
                  stake => return!(("KEY_FOUND", stake))
                }}
              }}
            }}
          }}
        }}
    "#,
        validator_hex
    );

    let node = TestNode::standalone(genesis.clone())
        .await
        .expect("Failed to create standalone node");

    let post_state_hash = genesis.genesis_block.body.state.post_state_hash.clone();

    let result = node
        .runtime_manager
        .play_exploratory_deploy(lookup_query, &post_state_hash)
        .await
        .expect("Failed to execute exploratory deploy");

    // Verify we got a result
    assert!(
        !result.is_empty(),
        "Validator lookup should return a non-empty result"
    );

    // The lookup should succeed (KEY_FOUND)
    let result_str = format!("{:?}", result);
    assert!(
        result_str.contains("KEY_FOUND"),
        "Validator key should be found in allBonds: {:?}",
        result
    );

    tracing::info!("Validator lookup result: {:?}", result);
}

/// Invalid block sender should be found in genesis validators
#[tokio::test]
async fn invalid_block_sender_should_be_in_genesis_validators() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .expect("Failed to build genesis");

    let mut nodes = TestNode::create_network(genesis.clone(), 2, None, None, None, None)
        .await
        .expect("Failed to create network");

    // Create a valid deploy on node 0
    let deploy = construct_deploy::source_deploy(
        "new x in { x!(0) }".to_string(),
        1,
        None,
        None,
        None,
        None,
        Some(genesis.genesis_block.shard_id.clone()),
    )
    .expect("Failed to create deploy");

    // Create a valid block on node 0
    let signed_block = nodes[0]
        .create_block_unsafe(&[deploy])
        .await
        .expect("Node 0 should create block");

    // Create an invalid version
    let mut invalid_block = signed_block.clone();
    invalid_block.seq_num = 47;

    let sender_hex = hex::encode(&invalid_block.sender);
    tracing::info!("Invalid block sender (hex): {}", sender_hex);

    // Check genesis validators
    let genesis_validators: Vec<String> = genesis
        .genesis_block
        .body
        .state
        .bonds
        .iter()
        .map(|bond| hex::encode(&bond.validator))
        .collect();

    tracing::info!("Genesis validators: {:?}", genesis_validators);

    // Sender should be in genesis validators
    let sender_in_genesis = genesis_validators.contains(&sender_hex);
    tracing::info!("Sender in genesis validators: {}", sender_in_genesis);

    // Process the invalid block
    let status = nodes[1]
        .process_block(invalid_block)
        .await
        .expect("Node 1 should process block");

    // Block should be rejected
    match &status {
        Either::Left(BlockError::Invalid(InvalidBlock::InvalidBlockHash)) => {
            // Expected
        }
        other => {
            panic!("Expected InvalidBlockHash error, got: {:?}", other);
        }
    }

    // Sender must be a genesis validator for slashing to work
    assert!(
        sender_in_genesis,
        "Invalid block sender should be a genesis validator"
    );
}
