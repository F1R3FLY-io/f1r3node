// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/RegistryOpsSpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use casper::rust::test_utils::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[tokio::test]
async fn registry_ops_spec() {
    let test_object = CompiledRholangSource::load_source("RegistryOpsTest.rho")
        .expect("Failed to load RegistryOpsTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(),
        "RegistryOpsTest.rho".to_string(),
    )
    .expect("Failed to compile RegistryOpsTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests()
        .await
        .expect("RegistryOpsSpec tests failed");
}

