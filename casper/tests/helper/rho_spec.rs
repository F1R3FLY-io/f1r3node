// See casper/src/test/scala/coop/rchain/casper/helper/RhoSpec.scala

use crate::genesis::contracts::test_util::TestUtil;
use crate::helper::{
    block_data_contract, casper_invalid_blocks_contract, deployer_id_contract, rho_logger_contract,
    secp256k1_sign_contract, sys_auth_token_contract,
};
use crate::util::genesis_builder::{GenesisBuilder, GenesisParameters};
use crate::util::rholang::resources::{generate_scope_id, mk_test_rnode_store_manager_shared};
use casper::rust::genesis::genesis::Genesis;
use casper::rust::helper::test_result_collector::{
    RhoTestAssertion, TestResult, TestResultCollector,
};
use casper::rust::util::rholang::tools::Tools;
use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use crypto::rust::private_key::PrivateKey;
use crypto::rust::signatures::secp256k1::Secp256k1;
use crypto::rust::signatures::signed::Signed;
use models::rhoapi::{BindPattern, ListParWithRandom};
use models::rust::casper::protocol::casper_message::DeployData;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use rholang::rust::interpreter::errors::InterpreterError;
use rholang::rust::interpreter::matcher::r#match::Matcher;
use rholang::rust::interpreter::pretty_printer::PrettyPrinter;
use rholang::rust::interpreter::rho_runtime::{create_runtime_from_kv_store, RhoRuntime};
use rholang::rust::interpreter::system_processes::{byte_name, Definition};
use rspace_plus_plus::rspace::r#match::Match;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

const SHARD_ID: &str = "root-shard";
const RHO_SPEC_PRIVATE_KEY: &str =
    "abaa20c1d578612b568a7c3d9b16e81c68d73b931af92cf79727e02011c558c6";
const RHO_SPEC_TIMESTAMP: i64 = 1559158671800;

pub struct RhoSpec {
    pub test_object: CompiledRholangSource,
    pub extra_non_genesis_deploys: Vec<Signed<DeployData>>,
    pub execution_timeout: Duration,
    pub genesis_parameters: GenesisParameters,
}

impl RhoSpec {
    pub fn new(
        test_object: CompiledRholangSource,
        extra_non_genesis_deploys: Vec<Signed<DeployData>>,
        execution_timeout: Duration,
    ) -> Self {
        Self {
            test_object,
            extra_non_genesis_deploys,
            execution_timeout,
            genesis_parameters: GenesisBuilder::build_genesis_parameters_with_defaults(None, None),
        }
    }

    pub fn new_with_genesis_parameters(
        test_object: CompiledRholangSource,
        extra_non_genesis_deploys: Vec<Signed<DeployData>>,
        execution_timeout: Duration,
        genesis_parameters: GenesisParameters,
    ) -> Self {
        Self {
            test_object,
            extra_non_genesis_deploys,
            execution_timeout,
            genesis_parameters,
        }
    }

    fn printer() -> PrettyPrinter {
        PrettyPrinter::new()
    }

    pub fn mk_test(&self, _test_name: &str, test_attempts: &HashMap<i64, Vec<RhoTestAssertion>>) {
        assert!(
            !test_attempts.is_empty(),
            "It doesn't make sense to have less than one attempt"
        );

        let (attempt, assertions) = test_attempts
            .iter()
            .find(|(_, assertions)| Self::has_failures(assertions))
            .map(|(k, v)| (*k, v.clone()))
            .unwrap_or_else(|| {
                let first = test_attempts.iter().next().unwrap();
                (*first.0, first.1.clone())
            });

        let clue_msg = |clue: &str| format!("{} (test attempt: {})", clue, attempt);

        let mut printer = Self::printer();

        for assertion in assertions {
            match assertion {
                RhoTestAssertion::RhoAssertEquals {
                    expected,
                    actual,
                    clue,
                    ..
                } => {
                    assert_eq!(
                        printer.build_string_from_message(&expected),
                        printer.build_string_from_message(&actual),
                        "{}",
                        clue_msg(&clue)
                    );
                    assert_eq!(expected, actual, "{}", clue_msg(&clue));
                }
                RhoTestAssertion::RhoAssertNotEquals {
                    unexpected,
                    actual,
                    clue,
                    ..
                } => {
                    assert_ne!(
                        printer.build_string_from_message(&unexpected),
                        printer.build_string_from_message(&actual),
                        "{}",
                        clue_msg(&clue)
                    );
                    assert_ne!(unexpected, actual, "{}", clue_msg(&clue));
                }
                RhoTestAssertion::RhoAssertTrue {
                    is_success, clue, ..
                } => {
                    assert!(is_success, "{}", clue_msg(&clue));
                }
            }
        }
    }

    // Note: Original Scala code has a bug here - it checks `_.isSuccess` instead of `!_.isSuccess`.
    // This was fixed in Rust to correctly return true when there are failures (not successes).
    pub fn has_failures(assertions: &[RhoTestAssertion]) -> bool {
        assertions.iter().any(|a| !a.is_success())
    }

    /// Runs the tests by executing get_results and iterating through assertions
    /// Original Scala:
    /// ```scala
    /// val result = getResults(testObject, extraNonGenesisDeploys, executionTimeout).runSyncUnsafe(Duration.Inf)
    /// result.assertions.foreach(mkTest)
    /// ```
    pub async fn run_tests(&self) -> Result<TestResult, InterpreterError> {
        let test_result_collector = Arc::new(TestResultCollector::new());

        let result = get_results(
            &self.test_object,
            &self.extra_non_genesis_deploys,
            self.execution_timeout,
            self.genesis_parameters.clone(),
            test_result_collector,
        )
        .await?;

        // Run mkTest for each assertion
        for (test_name, test_attempts) in &result.assertions {
            self.mk_test(test_name, test_attempts);
        }

        Ok(result)
    }
}

pub fn test_framework_contracts(
    test_result_collector: Arc<TestResultCollector>,
) -> Vec<Definition> {
    vec![
        Definition {
            urn: "rho:test:assertAck".to_string(),
            fixed_channel: byte_name(101),
            arity: 5,
            body_ref: 101,
            handler: {
                let trc = test_result_collector.clone();
                Box::new(move |ctx| {
                    let trc = trc.clone();
                    Box::new(move |args| {
                        let trc = trc.clone();
                        let ctx = ctx.clone();
                        Box::pin(async move {
                            trc.handle_message(ctx, args).await;
                            Ok(vec![])
                        })
                    })
                })
            },
            remainder: None,
        },
        Definition {
            urn: "rho:test:testSuiteCompleted".to_string(),
            fixed_channel: byte_name(102),
            arity: 1,
            body_ref: 102,
            handler: {
                let trc = test_result_collector.clone();
                Box::new(move |ctx| {
                    let trc = trc.clone();
                    Box::new(move |args| {
                        let trc = trc.clone();
                        let ctx = ctx.clone();
                        Box::pin(async move {
                            trc.handle_message(ctx, args).await;
                            Ok(vec![])
                        })
                    })
                })
            },
            remainder: None,
        },
        Definition {
            urn: "rho:io:stdlog".to_string(),
            fixed_channel: byte_name(103),
            arity: 2,
            body_ref: 103,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { rho_logger_contract::handle_message(ctx, args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:test:deployerId:make".to_string(),
            fixed_channel: byte_name(104),
            arity: 3,
            body_ref: 104,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { deployer_id_contract::get(ctx, args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:test:crypto:secp256k1Sign".to_string(),
            fixed_channel: byte_name(105),
            arity: 3,
            body_ref: 105,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { secp256k1_sign_contract::get(ctx, args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "sys:test:authToken:make".to_string(),
            fixed_channel: byte_name(106),
            arity: 1,
            body_ref: 106,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { sys_auth_token_contract::get(ctx, args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:test:block:data:set".to_string(),
            fixed_channel: byte_name(107),
            arity: 3,
            body_ref: 107,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { block_data_contract::set(ctx, args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:test:casper:invalidBlocks:set".to_string(),
            fixed_channel: byte_name(108),
            arity: 2,
            body_ref: 108,
            handler: Box::new(|ctx| {
                Box::new(move |args| {
                    let ctx = ctx.clone();
                    Box::pin(async move { casper_invalid_blocks_contract::set(ctx, args).await })
                })
            }),
            remainder: None,
        },
    ]
}

pub async fn get_results(
    test_object: &CompiledRholangSource,
    other_libs: &[Signed<DeployData>],
    execution_timeout: Duration,
    genesis_parameters: GenesisParameters,
    test_result_collector: Arc<TestResultCollector>,
) -> Result<TestResult, InterpreterError> {
    let mut genesis_builder = GenesisBuilder::new();
    let _genesis = genesis_builder
        .build_genesis_with_parameters(Some(genesis_parameters))
        .await
        .map_err(|e| {
            InterpreterError::BugFoundError(format!("Failed to build genesis: {:?}", e))
        })?;

    let scope_id = generate_scope_id();

    let mut kvs_manager = mk_test_rnode_store_manager_shared(scope_id);
    let r_store = kvs_manager.r_space_stores().await.map_err(|e| {
        InterpreterError::BugFoundError(format!("Failed to create RSpaceStore: {}", e))
    })?;

    // NOTE: In Scala, RSpacePlusPlus_RhoTypes.create() calls Rust via JNA, where Matcher
    // is created automatically (see rspace++/libs/rspace_rhotypes/src/lib.rs).
    // In pure Rust code (without JNA), we must create the Matcher explicitly here,
    // as RSpace::create(stores, matcher) requires it as a parameter.
    let matcher =
        Arc::new(Box::new(Matcher::default()) as Box<dyn Match<BindPattern, ListParWithRandom>>);

    let mut additional_system_processes = test_framework_contracts(test_result_collector.clone());

    let runtime = create_runtime_from_kv_store(
        r_store,
        Genesis::non_negative_mergeable_tag_name(),
        true,
        &mut additional_system_processes,
        matcher,
        rholang::rust::interpreter::external_services::ExternalServices::noop(),
    )
    .await;

    println!("Starting tests from {}", test_object.path);

    let runtime = setup_runtime(runtime, other_libs).await?;

    let rand = Blake2b512Random::create_from_length(128).split_short(1);

    // Note: tokio::time::timeout similar to Scala's `monix.eval.Task.timeout()`.
    // If test execution takes longer than `execution_timeout`, it returns a timeout error.
    match tokio::time::timeout(
        execution_timeout,
        TestUtil::eval_source(test_object, &runtime, rand),
    )
    .await
    {
        Ok(result) => result?,
        Err(_) => {
            return Err(InterpreterError::BugFoundError(format!(
                "Timeout of {:?} expired while executing test from {}",
                execution_timeout, test_object.path
            )))
        }
    }

    Ok(test_result_collector.get_result())
}

async fn setup_runtime<R: RhoRuntime>(
    runtime: R,
    extra_libs: &[Signed<DeployData>],
) -> Result<R, InterpreterError> {
    eval_deploy(&rho_spec_deploy(), &runtime).await?;

    for deploy in extra_libs {
        eval_deploy(deploy, &runtime).await?;
    }

    Ok(runtime)
}

async fn eval_deploy(
    deploy: &Signed<DeployData>,
    runtime: &impl RhoRuntime,
) -> Result<(), InterpreterError> {
    use models::rust::normalizer_env::normalizer_env_from_deploy;

    let rand = Tools::unforgeable_name_rng(&deploy.pk, deploy.data.time_stamp);

    TestUtil::eval(
        &deploy.data.term,
        runtime,
        normalizer_env_from_deploy(deploy),
        rand,
    )
    .await?;

    Ok(())
}

fn rho_spec_deploy() -> Signed<DeployData> {
    let sk_bytes = hex::decode(RHO_SPEC_PRIVATE_KEY).expect("Invalid RHO_SPEC_PRIVATE_KEY hex");
    let sk = PrivateKey::from_bytes(&sk_bytes);

    let code = CompiledRholangSource::load_source("RhoSpecContract.rho")
        .expect("Failed to load RhoSpecContract.rho");

    let compiled =
        CompiledRholangSource::new(code, HashMap::new(), "RhoSpecContract.rho".to_string())
            .expect("Failed to compile RhoSpecContract.rho");

    let deploy_data = DeployData {
        term: compiled.code,
        time_stamp: RHO_SPEC_TIMESTAMP,
        phlo_price: 0,
        phlo_limit: i64::MAX,
        valid_after_block_number: 0,
        shard_id: SHARD_ID.to_string(),
    };

    Signed::create(deploy_data, Box::new(Secp256k1), sk).expect("Failed to sign RhoSpec deploy")
}
