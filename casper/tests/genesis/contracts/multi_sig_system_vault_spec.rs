// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/MultiSigSystemVaultSpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use crate::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[tokio::test]
async fn multi_sig_system_vault_spec() {
    let test_object = CompiledRholangSource::load_source("MultiSigSystemVaultTest.rho")
        .expect("Failed to load MultiSigSystemVaultTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(),
        "MultiSigSystemVaultTest.rho".to_string(),
    )
    .expect("Failed to compile MultiSigSystemVaultTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests()
        .await
        .expect("MultiSigSystemVaultSpec tests failed");
}

