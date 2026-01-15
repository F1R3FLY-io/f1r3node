// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/AuthKeySpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use casper::rust::test_utils::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[tokio::test]
async fn auth_key_spec() {
    let test_object = CompiledRholangSource::load_source("AuthKeyTest.rho")
        .expect("Failed to load AuthKeyTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(), // NormalizerEnv.Empty
        "AuthKeyTest.rho".to_string(),
    )
    .expect("Failed to compile AuthKeyTest.rho");

    let spec = RhoSpec::new(
        compiled,
        vec![], // Seq.empty
        GENESIS_TEST_TIMEOUT,
    );

    spec.run_tests().await.expect("AuthKeySpec tests failed");
}
