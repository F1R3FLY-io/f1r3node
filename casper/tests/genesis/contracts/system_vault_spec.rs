// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/SystemVaultSpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use crate::helper::rho_spec::RhoSpec;
use casper::rust::util::construct_deploy::DEFAULT_PUB;
use models::rust::normalizer_env::with_deployer_id;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;

#[tokio::test]
async fn system_vault_spec() {
    let test_object = CompiledRholangSource::load_source("SystemVaultTest.rho")
        .expect("Failed to load SystemVaultTest.rho");

    // NormalizerEnv.withDeployerId(deployerPk)
    let normalizer_env = with_deployer_id(&DEFAULT_PUB);

    let compiled = CompiledRholangSource::new(
        test_object,
        normalizer_env,
        "SystemVaultTest.rho".to_string(),
    )
    .expect("Failed to compile SystemVaultTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests().await.expect("SystemVaultSpec tests failed");
}
