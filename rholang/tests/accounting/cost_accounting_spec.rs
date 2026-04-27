//See rholang/src/test/scala/coop/rchain/rholang/interpreter/accounting/CostAccountingSpec.scala from main branch

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use rholang::rust::interpreter::rho_runtime::{create_replay_rho_runtime, RhoRuntimeImpl};
use rholang::rust::interpreter::system_processes::Definition;
use rholang::rust::interpreter::test_utils::resources::create_runtimes;
use rholang::rust::interpreter::{
    accounting::costs::Cost,
    external_services::ExternalServices,
    interpreter::EvaluateResult,
    matcher::r#match::Matcher,
    rho_runtime::{create_rho_runtime, RhoRuntime},
};
use rspace_plus_plus::rspace::history::history_repository::HistoryRepository;
use rspace_plus_plus::rspace::rspace::RSpaceStore;
use rspace_plus_plus::rspace::{
    rspace::RSpace,
    shared::{
        in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
    },
};

use rand::Rng;
use rholang::rust::interpreter::errors::InterpreterError;
use std::collections::{HashMap, HashSet};
use std::option::Option;
use std::sync::Arc;

async fn evaluate_with_cost_log(
    initial_phlo: i64,
    contract: String,
) -> (EvaluateResult, Vec<Cost>) {
    // Cost logging is enabled in test builds via cfg!(test) in CostManager.

    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();
    let (mut runtime, _, _) =
        create_runtimes_with_cost_log(store, Some(false), Some(&mut Vec::new())).await;

    let eval_result = runtime
        .evaluate_with_phlo(
            &contract,
            Cost::create(initial_phlo, "cost_accounting_spec setup".to_string()),
        )
        .await;

    assert!(eval_result.is_ok());
    let eval_result = eval_result.unwrap();
    let cost_log = runtime.get_cost_log();
    (eval_result, cost_log)
}

async fn create_runtimes_with_cost_log(
    stores: RSpaceStore,
    init_registry: Option<bool>,
    additional_system_processes: Option<&mut Vec<Definition>>,
) -> (
    RhoRuntimeImpl,
    RhoRuntimeImpl,
    Arc<
        Box<
            dyn HistoryRepository<Par, BindPattern, ListParWithRandom, TaggedContinuation>
                + Send
                + Sync
                + 'static,
        >,
    >,
) {
    let init_registry = init_registry.unwrap_or(false);

    let mut empty_vec = Vec::new();
    let additional_system_processes = additional_system_processes.unwrap_or(&mut empty_vec);

    let hrstores =
        RSpace::<Par, BindPattern, ListParWithRandom, TaggedContinuation>::create_with_replay(
            stores,
            Arc::new(Box::new(Matcher)),
        )
        .unwrap();

    let (space, replay) = hrstores;

    let history_repository = space.get_history_repository();

    let rho_runtime = create_rho_runtime(
        space.clone(),
        Par::default(),
        init_registry,
        additional_system_processes,
        ExternalServices::noop(),
    )
    .await;

    let replay_rho_runtime = create_replay_rho_runtime(
        replay,
        Par::default(),
        init_registry,
        additional_system_processes,
        ExternalServices::noop(),
    )
    .await;

    (rho_runtime, replay_rho_runtime, history_repository)
}

async fn evaluate_and_replay(initial_phlo: Cost, term: String) -> (EvaluateResult, EvaluateResult) {
    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();
    let (mut runtime, mut replay_runtime, _): (
        RhoRuntimeImpl,
        RhoRuntimeImpl,
        Arc<
            Box<
                dyn HistoryRepository<Par, BindPattern, ListParWithRandom, TaggedContinuation>
                    + Send
                    + Sync
                    + 'static,
            >,
        >,
    ) = create_runtimes(store, false, &mut Vec::new()).await;

    let rand = Blake2b512Random::create_from_bytes(&[]);

    let play_result = {
        runtime
            .evaluate(&term, initial_phlo.clone(), HashMap::new(), rand.clone())
            .await
            .expect("Evaluation failed")
    };

    let replay_result = {
        let checkpoint = runtime.create_checkpoint().await;
        let root = checkpoint.root;
        let log = checkpoint.log;

        replay_runtime
            .reset(&root)
            .await
            .expect("Failed to reset replay runtime");
        replay_runtime.rig(log).await.expect("Rig failed");

        let result = replay_runtime
            .evaluate(&term, initial_phlo, HashMap::new(), rand)
            .await
            .expect("Replay evaluation failed");

        replay_runtime
            .check_replay_data()
            .await
            .unwrap_or_else(|e| panic!("Replay data check failed for '{}': {:?}", term, e));

        result
    };

    (play_result, replay_result)
}

// Uses Godel numbering and a https://en.wikipedia.org/wiki/Mixed_radix
// to encode certain terms as numbers in the range [0, 0x144000000).
// Every number gets decoded into a unique term, but some terms can
// be encoded by more than one number.
fn from_long(index: i64) -> String {
    let mut remainder = index;
    let num_pars = (remainder % 4) + 1;
    remainder /= 4;

    let mut result = Vec::new();
    let mut nonlinear_send = false;
    let mut nonlinear_recv = false;

    for _ in 0..num_pars {
        let dir = remainder % 2;
        remainder /= 2;

        if dir == 0 {
            //send
            let bang = if remainder % 2 == 0 { "!" } else { "!!" };
            remainder /= 2;

            if bang == "!" || !nonlinear_recv {
                let ch = remainder % 4;
                remainder /= 4;
                result.push(format!("@{}{}(0)", ch, bang));
                nonlinear_send |= bang == "!!";
            }
        } else {
            //receive
            let arrow = match remainder % 3 {
                0 => "<-",
                1 => "<=",
                2 => "<<-",
                _ => unreachable!(),
            };
            remainder /= 3;

            if arrow != "<=" || !nonlinear_send {
                let num_joins = (remainder % 2) + 1;
                remainder /= 2;

                let mut joins = Vec::new();
                for _ in 1..=num_joins {
                    let ch = remainder % 4;
                    remainder /= 4;
                    joins.push(format!("_ {} @{}", arrow, ch));
                }

                let join_str = joins.join(" & ");
                result.push(format!("for ({}) {{ 0 }}", join_str));
                nonlinear_recv |= arrow == "<=";
            }
        }
    }

    result.join(" | ")
}

fn contracts() -> Vec<(String, i64)> {
    vec![
      (String::from("@0!(2)"), 97),
      (String::from("@0!(2) | @1!(1)"), 197),
      (String::from("for(x <- @0){ Nil }"), 128),
      (String::from("for(x <- @0){ Nil } | @0!(2)"), 329),
      (String::from("@0!!(0) | for (_ <- @0) { 0 }"), 342),
      (String::from("@0!!(0) | for (x <- @0) { 0 }"), 342),
      (String::from("@0!!(0) | for (@0 <- @0) { 0 }"), 336),
      (String::from("@0!!(0) | @0!!(0) | for (_ <- @0) { 0 }"), 443),
      (String::from("@0!!(0) | @1!!(1) | for (_ <- @0 & _ <- @1) { 0 }"), 596),
      (String::from("@0!(0) | for (_ <- @0) { 0 }"), 333),
      (String::from("@0!(0) | for (x <- @0) { 0 }"), 333),
      (String::from("@0!(0) | for (@0 <- @0) { 0 }"), 327),
      (String::from("@0!(0) | for (_ <= @0) { 0 }"), 354),
      (String::from("@0!(0) | for (x <= @0) { 0 }"), 356),
      (String::from("@0!(0) | for (@0 <= @0) { 0 }"), 341),
      (String::from("@0!(0) | @0!(0) | for (_ <= @0) { 0 }"), 574),
      (String::from("@0!(0) | for (@0 <- @0) { 0 } | @0!(0) | for (_ <- @0) { 0 }"), 663),
      (String::from("@0!(0) | for (@0 <- @0) { 0 } | @0!(0) | for (@1 <- @0) { 0 }"), 551),
      (String::from("@0!(0) | for (_ <<- @0) { 0 }"), 406),
      (String::from("@0!!(0) | for (_ <<- @0) { 0 }"), 343),
      (String::from("@0!!(0) | @0!!(0) | for (_ <<- @0) { 0 }"), 444),
      (String::from("new loop in {\n  contract loop(@n) = {\n    match n {\n      0 => Nil\n      _ => loop!(n-1)\n    }\n  } |\n  loop!(10)\n}"), 3846),
      (String::from("42 | @0!(2) | for (x <- @0) { Nil }"), 336),
      (String::from("@1!(1) |\n        for(x <- @1) { Nil } |\n        new x in { x!(10) | for(X <- x) { @2!(Set(X!(7)).add(*X).contains(10)) }} |\n        match 42 {\n          38 => Nil\n          42 =>\n@3!(42)\n        }\n     "), 1264),
      (String::from("new ret, keccak256Hash(`rho:crypto:keccak256Hash`) in {\n  keccak256Hash!(\"TEST\".toByteArray(), *ret) |\n  for (_ <- ret) { Nil }\n}"), 782),
    ]
}

fn element_counts(list: &[Cost]) -> HashSet<(Cost, usize)> {
    let mut counts = HashMap::new();
    for c in list {
        *counts.entry(c.clone()).or_insert(0) += 1;
    }
    counts.into_iter().collect()
}

async fn check_phlo_limit_exceeded(
    contract: String,
    initial_phlo: i64,
    expected_costs: Vec<Cost>,
) -> bool {
    let (evaluate_result, cost_log) = evaluate_with_cost_log(initial_phlo, contract).await;
    let expected_sum: i64 = expected_costs.iter().map(|cost| cost.value).sum();

    assert!(
        expected_sum <= initial_phlo,
        "We must not expect more costs than initialPhlo allows (duh!): {} > {}",
        expected_sum,
        initial_phlo
    );

    assert_eq!(
        evaluate_result.errors,
        vec![InterpreterError::OutOfPhlogistonsError],
        "Expected list of OutOfPhlogistonsError"
    );

    for cost in &expected_costs {
        assert!(
            cost_log.contains(cost),
            "CostLog does not contain expected cost: {:?}",
            cost
        );
    }

    assert_eq!(
        {
            element_counts(&cost_log)
                .difference(&element_counts(&expected_costs))
                .count()
        },
        1,
        "Exactly one cost should be logged past the expected ones"
    );
    assert!(
        evaluate_result.cost.value >= initial_phlo,
        "Total cost value should be >= initialPhlo"
    );

    true
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn total_cost_of_evaluation_should_be_equal_to_the_sum_of_all_costs_in_the_log() {
    for (contract, expected_cost) in contracts() {
        let initial_phlo = 10000i64;
        let (eval_result, cost_log) = evaluate_with_cost_log(initial_phlo, contract.clone()).await;
        assert_eq!(eval_result.errors, Vec::new(), "Contract errored: {}", contract);
        assert_eq!(
            eval_result.cost.value, expected_cost,
            "Cost mismatch for '{}': expected={}, got={}", contract, expected_cost, eval_result.cost.value
        );
        assert_eq!(
            cost_log.iter().map(|c| c.value).sum::<i64>(),
            expected_cost,
            "Cost log sum mismatch for: {}", contract
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cost_should_be_deterministic() {
    for (contract, _) in contracts() {
        let mut first_cost: Option<i64> = None;
        for i in 0..20 {
            let (result, _log) = evaluate_with_cost_log(i32::MAX as i64, contract.clone()).await;
            assert!(result.errors.is_empty(), "Contract errored: {}", contract);
            match first_cost {
                None => first_cost = Some(result.cost.value),
                Some(expected) => {
                    assert_eq!(
                        result.cost.value, expected,
                        "Cost not deterministic at iteration {} for '{}': expected={}, got={}",
                        i, contract, expected, result.cost.value
                    );
                }
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cost_should_be_repeatable_when_generated() {
    let phlo = Cost::create(i32::MAX as i64, "max_value".to_string());
    let mut tested = 0u32;
    let mut skipped = 0u32;
    let mut mismatches: Vec<(String, i64, i64)> = Vec::new();

    let mut rng = rand::thread_rng();
    for _ in 0..10000 {
        let long = ((rng.gen::<i64>() % 0x144000000) + 0x144000000) % 0x144000000;
        let contract = from_long(long);
        if contract.is_empty() {
            continue;
        }

        let (play, replay) = evaluate_and_replay(phlo.clone(), contract.clone()).await;

        // Skip contracts that hit the known "same-channel join" limitation.
        // These are invalid Rholang — not a replay issue.
        let is_same_channel_error = |errors: &[InterpreterError]| {
            errors.iter().any(|e| match e {
                InterpreterError::ReceiveOnSameChannelsError { .. } => true,
                InterpreterError::ParserError(msg) if msg.contains("Receiving on the same channels") => true,
                _ => false,
            })
        };
        if is_same_channel_error(&play.errors) || is_same_channel_error(&replay.errors) {
            skipped += 1;
            continue;
        }
        assert!(play.errors.is_empty(), "Unexpected play error for '{}': {:?}", contract, play.errors);
        assert!(replay.errors.is_empty(), "Unexpected replay error for '{}': {:?}", contract, replay.errors);

        if play.cost != replay.cost {
            mismatches.push((contract, play.cost.value, replay.cost.value));
        }
        tested += 1;
    }

    eprintln!("Replay cost determinism: {} tested, {} skipped, {} mismatches", tested, skipped, mismatches.len());
    for (contract, play, replay) in &mismatches {
        eprintln!("  MISMATCH: play={}, replay={}, diff={}, contract='{}'", play, replay, play - replay, contract);
    }
    assert!(tested > 100, "Too few contracts tested: {}", tested);
    assert!(mismatches.is_empty(), "{} contracts had play/replay cost mismatches", mismatches.len());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn running_out_of_phlogistons_should_stop_evaluation_upon_cost_depletion_in_a_single_execution_branch(
) {
    let parsing_cost = 6;

    check_phlo_limit_exceeded(
        "@1!(1)".to_string(),
        parsing_cost,
        vec![Cost::create(parsing_cost, "parsing".to_string())],
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn should_not_attempt_reduction_when_there_was_not_enough_phlo_for_parsing() {
    let parsing_cost = 6;

    check_phlo_limit_exceeded("@1!(1)".to_string(), parsing_cost - 1, vec![]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn should_stop_the_evaluation_of_all_execution_branches_when_one_of_them_runs_out_of_phlo() {
    let parsing_cost = 24;
    let first_step_cost = 11;
    check_phlo_limit_exceeded(
        "@1!(1) | @2!(2) | @3!(3)".to_string(),
        parsing_cost + first_step_cost,
        vec![
            Cost::create(parsing_cost, "parsing".to_string()),
            Cost::create(first_step_cost, "send eval".to_string()),
        ],
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn should_stop_the_evaluation_of_all_execution_branches_when_one_of_them_runs_out_of_phlo_with_a_more_sophisticated_contract(
) {
    let mut rng = rand::thread_rng();
    for (contract, expected_total_cost) in contracts() {
        let initial_phlo = rng.gen_range(1..expected_total_cost);

        let (result, _) = evaluate_with_cost_log(initial_phlo, contract.clone()).await;

        assert!(
            result.cost.value >= initial_phlo,
            "Total cost value should be >= initialPhlo, but got {} < {}",
            result.cost.value,
            initial_phlo
        );
    }
}

/// Regression test for F1R3FLY-io/f1r3node#178: peek consume with parallel
/// produce must preserve peeked data and produce identical play/replay costs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn peek_with_parallel_produce_should_have_deterministic_replay_cost() {
    let phlo = Cost::create(i32::MAX as i64, "max_value".to_string());

    let contracts = vec![
        // Basic peek + produce (simplified #178 repro)
        "for (@x <<- @1) { 0 } | @1!(42)",
        // Persistent produce triggering peek COMM (the bug pattern)
        "@1!!(0) | for (_ <<- @1) { 0 } | @2!(0)",
        // Peek join with persistent produce on one channel
        "for (_ <<- @2 & _ <<- @3) { 0 } | @3!!(0) | @2!(0)",
        // Peek join with persistent produces on both channels
        "for (_ <<- @1 & _ <<- @2) { 0 } | @1!!(0) | @2!!(0)",
        // Multiple peek consumes with persistent and non-persistent produces
        "for (_ <<- @1) { 0 } | @1!!(0) | for (_ <<- @2) { 0 } | @2!!(0)",
    ];

    for contract in contracts {
        for _ in 0..20 {
            let (play, replay) = evaluate_and_replay(phlo.clone(), contract.to_string()).await;
            assert!(play.errors.is_empty(), "Play error for '{}': {:?}", contract, play.errors);
            assert!(replay.errors.is_empty(), "Replay error for '{}': {:?}", contract, replay.errors);
            assert_eq!(
                play.cost, replay.cost,
                "Play/replay cost mismatch for '{}': play={}, replay={}",
                contract, play.cost.value, replay.cost.value
            );
        }
    }
}
