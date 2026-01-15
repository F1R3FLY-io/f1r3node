// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/EitherSpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use casper::rust::test_utils::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[tokio::test]
async fn either_spec() {
    let test_object = CompiledRholangSource::load_source("EitherTest.rho")
        .expect("Failed to load EitherTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(), // NormalizerEnv.Empty
        "EitherTest.rho".to_string(),
    )
    .expect("Failed to compile EitherTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests().await.expect("EitherSpec tests failed");
}

