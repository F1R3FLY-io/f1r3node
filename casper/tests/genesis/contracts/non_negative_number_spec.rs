// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/NonNegativeNumberSpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use crate::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[tokio::test]
async fn non_negative_number_spec() {
    let test_object = CompiledRholangSource::load_source("NonNegativeNumberTest.rho")
        .expect("Failed to load NonNegativeNumberTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(),
        "NonNegativeNumberTest.rho".to_string(),
    )
    .expect("Failed to compile NonNegativeNumberTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests()
        .await
        .expect("NonNegativeNumberSpec tests failed");
}

