// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/RhoSpecContractSpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use casper::rust::test_utils::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[tokio::test]
async fn rho_spec_contract_spec() {
    let test_object = CompiledRholangSource::load_source("RhoSpecContractTest.rho")
        .expect("Failed to load RhoSpecContractTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(),
        "RhoSpecContractTest.rho".to_string(),
    )
    .expect("Failed to compile RhoSpecContractTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests()
        .await
        .expect("RhoSpecContractSpec tests failed");
}
