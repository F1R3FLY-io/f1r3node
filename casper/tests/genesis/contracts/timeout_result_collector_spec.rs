// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/TimeoutResultCollectorSpec.scala

use casper::rust::test_utils::helper::rho_spec::get_results;
use casper::rust::test_utils::util::genesis_builder::GenesisBuilder;
use casper::rust::helper::test_result_collector::TestResultCollector;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn test_finished_should_be_false_if_execution_hasnt_finished_within_timeout() {
    let test_object = CompiledRholangSource::load_source("TimeoutResultCollectorTest.rho")
        .expect("Failed to load TimeoutResultCollectorTest.rho");

    let compiled = CompiledRholangSource::new(
        test_object,
        HashMap::new(),
        "TimeoutResultCollectorTest.rho".to_string(),
    )
    .expect("Failed to compile TimeoutResultCollectorTest.rho");

    let test_result_collector = Arc::new(TestResultCollector::new());
    let genesis_parameters = GenesisBuilder::build_genesis_parameters_with_defaults(None, None);

    let result = get_results(
        &compiled,
        &[],
        Duration::from_secs(10),
        genesis_parameters,
        test_result_collector,
    )
    .await
    .expect("Failed to get results");

    assert_eq!(
        result.has_finished, false,
        "testFinished should be false if execution hasn't finished within timeout"
    );
}
