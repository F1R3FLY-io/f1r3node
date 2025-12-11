// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/MultiSigRevVaultSpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use crate::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[tokio::test]
async fn multi_sig_rev_vault_spec() {
    let test_object = CompiledRholangSource::load_source("MultiSigRevVaultTest.rho")
        .expect("Failed to load MultiSigRevVaultTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(),
        "MultiSigRevVaultTest.rho".to_string(),
    )
    .expect("Failed to compile MultiSigRevVaultTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests()
        .await
        .expect("MultiSigRevVaultSpec tests failed");
}

