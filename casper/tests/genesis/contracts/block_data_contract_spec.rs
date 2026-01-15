// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/BlockDataContractSpec.scala

use casper::rust::test_utils::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;
use std::time::Duration;

#[tokio::test]
async fn block_data_contract_spec() {
    let test_object = CompiledRholangSource::load_source("BlockDataContractTest.rho")
        .expect("Failed to load BlockDataContractTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(), // NormalizerEnv.Empty
        "BlockDataContractTest.rho".to_string(),
    )
    .expect("Failed to compile BlockDataContractTest.rho");

    let spec = RhoSpec::new(
        compiled,
        vec![],                      // Seq.empty
        Duration::from_secs(30),     // 30.seconds (custom timeout)
    );

    spec.run_tests()
        .await
        .expect("BlockDataContractSpec tests failed");
}

