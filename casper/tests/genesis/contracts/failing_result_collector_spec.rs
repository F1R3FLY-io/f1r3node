// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/FailingResultCollectorSpec.scala

use crate::helper::rho_spec::get_results;
use casper::rust::test_utils::util::genesis_builder::GenesisBuilder;
use casper::rust::helper::test_result_collector::{
    RhoTestAssertion, TestResult, TestResultCollector,
};
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

struct FailingResultCollectorSpec;

impl FailingResultCollectorSpec {
    fn clue(clue_msg: &str, attempt: i64) -> String {
        format!("{} (attempt {})", clue_msg, attempt)
    }

    fn mk_test(test: (&str, &HashMap<i64, Vec<RhoTestAssertion>>)) {
        let (test_name, attempts) = test;

        println!("\nTest: {}", test_name);

        for (attempt, assertions) in attempts {
            for assertion in assertions {
                match assertion {
                    RhoTestAssertion::RhoAssertEquals {
                        expected,
                        actual,
                        clue: clue_msg,
                        ..
                    } => {
                        assert_ne!(actual, expected, "{}", Self::clue(clue_msg, *attempt));
                    }
                    RhoTestAssertion::RhoAssertTrue {
                        is_success,
                        clue: clue_msg,
                        ..
                    } => {
                        assert!(!is_success, "{}", Self::clue(clue_msg, *attempt));
                    }
                    RhoTestAssertion::RhoAssertNotEquals { .. } => {
                        panic!("Unexpected RhoAssertNotEquals");
                    }
                }
            }
        }
    }

    async fn result() -> TestResult {
        let test_object = CompiledRholangSource::load_source("FailingResultCollectorTest.rho")
            .expect("Failed to load FailingResultCollectorTest.rho");

        let compiled = CompiledRholangSource::new(
            test_object,
            HashMap::new(),
            "FailingResultCollectorTest.rho".to_string(),
        )
        .expect("Failed to compile FailingResultCollectorTest.rho");

        let test_result_collector = Arc::new(TestResultCollector::new());
        let genesis_parameters = GenesisBuilder::build_genesis_parameters_with_defaults(None, None);

        get_results(
            &compiled,
            &[],
            Duration::from_secs(10),
            genesis_parameters,
            test_result_collector,
        )
        .await
        .expect("Failed to get results")
    }
}

#[tokio::test]
async fn failing_result_collector_spec() {
    let result = FailingResultCollectorSpec::result().await;

    for (test_name, test_attempts) in &result.assertions {
        FailingResultCollectorSpec::mk_test((test_name.as_str(), test_attempts));
    }
}

#[tokio::test]
async fn failing_result_collector_spec_complete_within_timeout() {
    let result = FailingResultCollectorSpec::result().await;

    assert!(result.has_finished, "Test should complete within timeout");
}

