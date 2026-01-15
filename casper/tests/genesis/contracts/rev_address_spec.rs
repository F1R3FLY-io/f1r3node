// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/RevAddressSpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use casper::rust::test_utils::helper::rho_spec::RhoSpec;
use casper::rust::util::construct_deploy::DEFAULT_PUB;
use models::rust::normalizer_env::with_deployer_id;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;

#[tokio::test]
async fn rev_address_spec() {
    let test_object = CompiledRholangSource::load_source("RevAddressTest.rho")
        .expect("Failed to load RevAddressTest.rho");

    // NormalizerEnv.withDeployerId(deployerPk)
    let normalizer_env = with_deployer_id(&DEFAULT_PUB);

    let compiled = CompiledRholangSource::new(
        test_object,
        normalizer_env,
        "RevAddressTest.rho".to_string(),
    )
    .expect("Failed to compile RevAddressTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests()
        .await
        .expect("RevAddressSpec tests failed");
}
