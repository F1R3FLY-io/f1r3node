// Port of node/src/test/scala/coop/rchain/node/TransactionAPISpec.scala

use casper::rust::test_utils::helper::test_node::TestNode;
use casper::rust::test_utils::util::genesis_builder::{GenesisBuilder, GenesisContext};
use casper::rust::{api::block_report_api::BlockReportAPI, util::construct_deploy};
use crypto::rust::{
    private_key::PrivateKey,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg},
};
use node::rust::web::transaction::{TransactionAPI, TransactionAPIImpl, TransactionType};
use rholang::rust::interpreter::util::rev_address::RevAddress;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;

// Helper function equivalent to checkTransactionAPI in Scala
async fn check_transaction_api(
    term: String,
    phlo_limit: i64,
    phlo_price: i64,
    deploy_key: PrivateKey,
    genesis: &GenesisContext,
) -> Result<
    (
        Vec<node::rust::web::transaction::TransactionInfo>,
        models::rust::casper::protocol::casper_message::BlockMessage,
    ),
    casper::rust::errors::CasperError,
> {
    // Use dual scope for transaction API tests: isolated block/DAG stores, shared RSpace
    let mut nodes =
        TestNode::create_network_with_scope(genesis.clone(), 1, None, None, None, Some(1), true)
            .await?;

    // Use split_at_mut to get two non-overlapping mutable references
    let (first, second) = nodes.split_at_mut(1);
    let validator = &mut first[0];
    let readonly = &mut second[0];

    // Create deploy
    let deploy = construct_deploy::source_deploy_now_full(
        term,
        Some(phlo_limit),
        Some(phlo_price),
        Some(deploy_key),
        None,
        Some(genesis.genesis_block.shard_id.clone()),
    )?;

    // Add block from validator
    let transfer_block = validator.add_block_from_deploys(&[deploy]).await?;

    // Process block on readonly node - this stores the block in readonly.block_store
    readonly.process_block(transfer_block.clone()).await?;

    // Verify the block is actually in the store before proceeding
    let block_in_store = readonly
        .block_store
        .get(&transfer_block.block_hash)
        .map_err(|e| {
            casper::rust::errors::CasperError::RuntimeError(format!(
                "Failed to check if block is in store: {}",
                e
            ))
        })?;

    if block_in_store.is_none() {
        return Err(casper::rust::errors::CasperError::RuntimeError(
            "Block was not stored after process_block".to_string(),
        ));
    }

    // Create ReportingCasper using shared RSpace scope from genesis context
    // This ensures the ReportingCasper has access to the same RSpace history/roots
    // that were committed when the block was processed on the readonly node
    use casper::rust::reporting_casper;
    use casper::rust::test_utils::util::rholang::resources;

    // Use the shared RSpace scope from genesis to access the same RSpace stores
    // that contain the committed roots from block processing
    let mut kvm_for_rspace =
        resources::mk_test_rnode_store_manager_shared(genesis.rspace_scope_id.clone());
    let rspace_stores = kvm_for_rspace.r_space_stores().await.map_err(|e| {
        casper::rust::errors::CasperError::RuntimeError(format!(
            "Failed to get rspace stores: {}",
            e
        ))
    })?;

    let reporting_casper = reporting_casper::rho_reporter(
        &rspace_stores,
        &readonly.block_store,
        &readonly.block_dag_storage,
    );

    // Create store manager for ReportStore
    let mut kvm = casper::rust::test_utils::util::rholang::resources::mk_test_rnode_store_manager(
        readonly.data_dir.clone(),
    );

    // Create ReportStore
    let reporting_store = casper::rust::report_store::report_store(&mut kvm)
        .await
        .map_err(|e| {
            casper::rust::errors::CasperError::RuntimeError(format!(
                "Failed to create report store: {}",
                e
            ))
        })?;

    // Create BlockReportAPI
    // Note: BlockReportAPI requires engine_cell, block_store, and oracle
    // Use the readonly node's block_store directly (cloned, but shares underlying storage)
    let engine_cell = readonly.engine_cell.clone();
    let block_store = readonly.block_store.clone();
    let oracle = casper::rust::safety_oracle::CliqueOracleImpl;

    let block_report_api = BlockReportAPI::new(
        reporting_casper,
        reporting_store,
        engine_cell,
        block_store,
        oracle,
    );

    // Create TransactionAPI
    let transfer_unforgeable = node::rust::web::transaction::transfer_unforgeable();
    let transaction_api = TransactionAPIImpl::new(block_report_api, transfer_unforgeable);

    // Get transactions
    let block_hash = Blake2b256Hash::from_bytes_prost(&transfer_block.block_hash);
    let transactions = transaction_api
        .get_transaction(block_hash)
        .await
        .map_err(|e| {
            casper::rust::errors::CasperError::RuntimeError(format!(
                "Failed to get transactions: {}",
                e
            ))
        })?;

    Ok((transactions, transfer_block))
}

#[tokio::test]
async fn transfer_rev_should_be_gotten_in_transaction_api() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .unwrap();

    let from_sk = genesis.genesis_vaults_sks()[0].clone();
    let from_pk = Secp256k1.to_public(&from_sk);
    let from_addr = RevAddress::from_public_key(&from_pk)
        .expect("Failed to create from address")
        .to_base58();

    let to_sk = genesis.genesis_vaults_sks().last().unwrap().clone();
    let to_pk = Secp256k1.to_public(&to_sk);
    let to_addr = RevAddress::from_public_key(&to_pk)
        .expect("Failed to create to address")
        .to_base58();

    let amount = 1i64;
    let phlo_price = 1i64;
    let phlo_limit = 3_000_000i64;

    let transfer_rho = format!(
        r#"
new rl(`rho:registry:lookup`), RevVaultCh, vaultCh, toVaultCh, deployerId(`rho:rchain:deployerId`), revVaultKeyCh, resultCh in {{
  rl!(`rho:rchain:revVault`, *RevVaultCh) |
  for (@(_, RevVault) <- RevVaultCh) {{
    @RevVault!("findOrCreate", "{}", *vaultCh) |
    @RevVault!("findOrCreate", "{}", *toVaultCh) |
    @RevVault!("deployerAuthKey", *deployerId, *revVaultKeyCh) |
    for (@(true, vault) <- vaultCh; key <- revVaultKeyCh; @(true, toVault) <- toVaultCh) {{
      @vault!("transfer", "{}", {}, *key, *resultCh) |
      for (_ <- resultCh) {{ Nil }}
    }}
  }}
}}"#,
        from_addr, to_addr, to_addr, amount
    );

    let result = check_transaction_api(transfer_rho, phlo_limit, phlo_price, from_sk, &genesis)
        .await
        .expect("check_transaction_api failed");

    let (transactions, transfer_block) = result;

    assert_eq!(transactions.len(), 3, "Expected 3 transactions");

    for t in &transactions {
        match &t.transaction_type {
            TransactionType::UserDeploy { .. } => {
                assert_eq!(t.transaction.from_addr, from_addr);
                assert_eq!(t.transaction.to_addr, to_addr);
                assert_eq!(t.transaction.amount, amount);
                assert_eq!(t.transaction.fail_reason, None);
            }
            TransactionType::PreCharge { .. } => {
                assert_eq!(t.transaction.from_addr, from_addr);
                assert_eq!(t.transaction.amount, phlo_limit * phlo_price);
                assert_eq!(t.transaction.fail_reason, None);
            }
            TransactionType::Refund { .. } => {
                assert_eq!(t.transaction.to_addr, from_addr);
                let expected_refund =
                    phlo_limit * phlo_price - transfer_block.body.deploys[0].cost.cost as i64;
                assert_eq!(t.transaction.amount, expected_refund);
                assert_eq!(t.transaction.fail_reason, None);
            }
            _ => {}
        }
    }
}

#[tokio::test]
async fn no_user_deploy_log_should_return_only_precharge_and_refund_transaction() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .unwrap();

    let from_sk = genesis.genesis_vaults_sks()[0].clone();
    let from_pk = Secp256k1.to_public(&from_sk);
    let from_addr = RevAddress::from_public_key(&from_pk)
        .expect("Failed to create from address")
        .to_base58();

    let phlo_price = 1i64;
    let phlo_limit = 3_000_000i64;
    let deploy_rho = "new a in {}".to_string();

    let result = check_transaction_api(deploy_rho, phlo_limit, phlo_price, from_sk, &genesis)
        .await
        .expect("check_transaction_api failed");

    let (transactions, block) = result;

    assert_eq!(transactions.len(), 2, "Expected 2 transactions");

    for t in &transactions {
        match &t.transaction_type {
            TransactionType::PreCharge { .. } => {
                assert_eq!(t.transaction.from_addr, from_addr);
                assert_eq!(t.transaction.amount, phlo_limit * phlo_price);
                assert_eq!(t.transaction.fail_reason, None);
            }
            TransactionType::Refund { .. } => {
                assert_eq!(t.transaction.to_addr, from_addr);
                let expected_refund =
                    phlo_limit * phlo_price - block.body.deploys[0].cost.cost as i64;
                assert_eq!(t.transaction.amount, expected_refund);
                assert_eq!(t.transaction.fail_reason, None);
            }
            _ => {}
        }
    }
}

#[tokio::test]
async fn precharge_failed_case_should_return_1_precharge_transaction() {
    let genesis = GenesisBuilder::new()
        .build_genesis_with_parameters(None)
        .await
        .unwrap();

    let from_sk = genesis.genesis_vaults_sks()[0].clone();
    let from_pk = Secp256k1.to_public(&from_sk);
    let from_addr = RevAddress::from_public_key(&from_pk)
        .expect("Failed to create from address")
        .to_base58();

    let phlo_price = 1i64;
    let phlo_limit = 300_000_000_000i64; // Very high limit to trigger insufficient funds
    let deploy_rho = "new a in {}".to_string();

    let result = check_transaction_api(deploy_rho, phlo_limit, phlo_price, from_sk, &genesis)
        .await
        .expect("check_transaction_api failed");

    let (transactions, block) = result;

    assert_eq!(transactions.len(), 1, "Expected 1 transaction");

    let transaction = &transactions[0];
    assert!(matches!(
        transaction.transaction_type,
        TransactionType::PreCharge { .. }
    ));
    assert_eq!(transaction.transaction.from_addr, from_addr);

    // Note: The amount check might need adjustment based on actual implementation
    // The Scala test checks: phloLimit * phloPrice - block.body.deploys.head.cost.cost
    let expected_amount = phlo_limit * phlo_price - block.body.deploys[0].cost.cost as i64;
    assert_eq!(transaction.transaction.amount, expected_amount);
    assert_eq!(
        transaction.transaction.fail_reason,
        Some("Insufficient funds".to_string())
    );
}
