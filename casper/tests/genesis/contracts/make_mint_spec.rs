// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/MakeMintSpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use crate::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[tokio::test]
async fn make_mint_spec() {
    let test_object = CompiledRholangSource::load_source("MakeMintTest.rho")
        .expect("Failed to load MakeMintTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(),
        "MakeMintTest.rho".to_string(),
    )
    .expect("Failed to compile MakeMintTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests().await.expect("MakeMintSpec tests failed");
}

