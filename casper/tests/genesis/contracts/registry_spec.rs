// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/RegistrySpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use crate::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[tokio::test]
async fn registry_spec() {
    let test_object = CompiledRholangSource::load_source("RegistryTest.rho")
        .expect("Failed to load RegistryTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(),
        "RegistryTest.rho".to_string(),
    )
    .expect("Failed to compile RegistryTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests().await.expect("RegistrySpec tests failed");
}

