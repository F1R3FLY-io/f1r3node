use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use crate::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[tokio::test]
async fn token_metadata_spec() {
    let test_object = crate::util::rholang::test_rho_loader::load_test_rho("TokenMetadataTest.rho")
        .expect("Failed to load TokenMetadataTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(),
        "TokenMetadataTest.rho".to_string(),
    )
    .expect("Failed to compile TokenMetadataTest.rho");

    let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

    spec.run_tests()
        .await
        .expect("TokenMetadataSpec tests failed");
}
