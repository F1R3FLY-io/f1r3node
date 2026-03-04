// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/StackSpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use crate::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[tokio::test]
async fn stack_spec() {
    let test_object = CompiledRholangSource::load_source("StackTest.rho")
        .expect("Failed to load StackTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(), // NormalizerEnv.Empty
        "StackTest.rho".to_string(),
    )
    .expect("Failed to compile StackTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests().await.expect("StackSpec tests failed");
}
