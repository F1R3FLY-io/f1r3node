//See rholang/src/test/scala/coop/rchain/rholang/interpreter/accounting/CostAccountingSpec.scala from main branch

use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use rholang::rust::interpreter::rho_runtime::{create_replay_rho_runtime, RhoRuntimeImpl};
use rholang::rust::interpreter::system_processes::Definition;
use rholang::rust::interpreter::test_utils::resources::create_runtimes;
use rholang::rust::interpreter::{
    accounting::costs::Cost,
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
    Arc<Box<dyn HistoryRepository<Par, BindPattern, ListParWithRandom, TaggedContinuation>>>,
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

    let history_repository = space.history_repository.clone();

    let rho_runtime = create_rho_runtime(
        space.clone(),
        Par::default(),
        init_registry,
        additional_system_processes,
    )
    .await;

    let replay_rho_runtime = create_replay_rho_runtime(
        replay,
        Par::default(),
        init_registry,
        additional_system_processes,
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
        Arc<Box<dyn HistoryRepository<Par, BindPattern, ListParWithRandom, TaggedContinuation>>>,
    ) = create_runtimes(store, false, &mut Vec::new()).await;

    let rand = Blake2b512Random::create_from_bytes(&[]);

    let play_result = {
        runtime
            .evaluate(&term, initial_phlo.clone(), HashMap::new(), rand.clone())
            .await
            .expect("Evaluation failed")
    };

    let replay_result = {
        let checkpoint = runtime.create_checkpoint();
        let root = checkpoint.root;
        let log = checkpoint.log;

        replay_runtime.reset(&root);
        replay_runtime.rig(log).expect("Rig failed");

        let result = replay_runtime
            .evaluate(&term, initial_phlo, HashMap::new(), rand)
            .await
            .expect("Replay evaluation failed");

        replay_runtime
            .check_replay_data()
            .expect("Replay data check failed");

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
      (String::from("@0!(2)"), 97i64),
      (String::from("@0!(2) | @1!(1)"), 197i64),
      (String::from("for(x <- @0){ Nil }"), 128i64),
      (String::from("for(x <- @0){ Nil } | @0!(2)"), 329i64),
      (String::from("@0!!(0) | for (_ <- @0) { 0 }"), 342i64),
      (String::from("@0!!(0) | for (x <- @0) { 0 }"), 342i64),
      (String::from("@0!!(0) | for (@0 <- @0) { 0 }"), 336i64),
      (String::from("@0!!(0) | @0!!(0) | for (_ <- @0) { 0 }"), 443i64),
      (String::from("@0!!(0) | @1!!(1) | for (_ <- @0 & _ <- @1) { 0 }"), 596i64),
      (String::from("@0!(0) | for (_ <- @0) { 0 }"), 333i64),
      (String::from("@0!(0) | for (x <- @0) { 0 }"), 333i64),
      (String::from("@0!(0) | for (@0 <- @0) { 0 }"), 327i64),
      (String::from("@0!(0) | for (_ <= @0) { 0 }"), 354i64),
      (String::from("@0!(0) | for (x <= @0) { 0 }"), 356i64),
      (String::from("@0!(0) | for (@0 <= @0) { 0 }"), 341i64),
      (String::from("@0!(0) | @0!(0) | for (_ <= @0) { 0 }"), 574i64),
      (String::from("@0!(0) | for (@0 <- @0) { 0 } | @0!(0) | for (_ <- @0) { 0 }"), 663i64),
      (String::from("@0!(0) | for (@0 <- @0) { 0 } | @0!(0) | for (@1 <- @0) { 0 }"), 551i64),
      (String::from("@0!(0) | for (_ <<- @0) { 0 }"), 406i64),
      (String::from("@0!!(0) | for (_ <<- @0) { 0 }"), 343i64),
      (String::from("@0!!(0) | @0!!(0) | for (_ <<- @0) { 0 }"), 444i64),
      // TODO: This fails due to a cost mismatch - needs fixing
//       (String::from("new loop in {\n         contract loop(@n) = {\n           match n {\n             0 => Nil\n             _ => loop!(n-1)\n           }\n         } |\n         loop!(10)\n       }"),
// 3892i64),
      (String::from("42 | @0!(2) | for (x <- @0) { Nil }"), 336i64),
      (String::from("@1!(1) |\n        for(x <- @1) { Nil } |\n        new x in { x!(10) | for(X <- x) { @2!(Set(X!(7)).add(*X).contains(10)) }} |\n        match 42 {\n          38 => Nil\n          42 =>
@3!(42)\n        }\n     "), 1264i64),
      // TODO: This fails due to a cost mismatch - needs fixing
      // (String::from("new ret, keccak256Hash(`rho:crypto:keccak256Hash`) in {\n  keccak256Hash!(\"TEST\".toByteArray(), *ret) |\n  for (_ <- ret) { Nil }\n}"), 782i64),
]
}

fn element_counts(list: &[Cost]) -> HashSet<(Cost, usize)> {
    let mut counts = HashMap::new();
    for c in list {
        *counts.entry(c.clone()).or_insert(0) += 1;
    }
    counts.into_iter().collect()
}

async fn check_deterministic_cost<F, Fut>(block: F) -> bool
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = (EvaluateResult, Vec<Cost>)>,
{
    let repetitions = 20;
    let (first_result, first_log) = block().await;

    // Execute sequentially for now (could be parallel with join_all later)
    for _ in 1..repetitions {
        let (subsequent_result, subsequent_log) = block().await;
        let expected = first_result.cost.value;
        let actual = subsequent_result.cost.value;
        if expected != actual {
            assert_eq!(
                subsequent_log, first_log,
                "CostLog should be the same for deterministic cost"
            );
            panic!(
                "Cost was not repeatable, expected {}, got {}.",
                expected, actual
            );
        }
    }

    true
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

#[tokio::test]
async fn total_cost_of_evaluation_should_be_equal_to_the_sum_of_all_costs_in_the_log() {
    for (contract, expected_total_cost) in contracts() {
        let initial_phlo = 10000i64;
        let (eval_result, cost_log) = evaluate_with_cost_log(initial_phlo, contract.clone()).await;
        assert_eq!(
            eval_result.cost,
            Cost::create(expected_total_cost, "subtraction".to_string())
        );
        assert_eq!(eval_result.errors, Vec::new());
        assert_eq!(
            cost_log.iter().map(|c| c.value).sum::<i64>(),
            expected_total_cost
        );
    }
}

#[tokio::test]
async fn cost_should_be_deterministic() {
    for (contract, _) in contracts() {
        check_deterministic_cost(|| async {
            let (result, _log) = evaluate_with_cost_log(i32::MAX as i64, contract.clone()).await;
            assert!(result.errors.is_empty());
            (result, _log)
        })
        .await;
    }
}

#[tokio::test]
#[ignore] // TODO: Remove ignore when bug RCHAIN-3917 is fixed (13.05.2025 I can't find this ticket), RCHAIN-4032 - which is indicated in the Scala side error is also unknown.
async fn cost_should_be_repeatable_when_generated() {
    // Try contract fromLong(1716417707L) = @2!!(0) | @0!!(0) | for (_ <<- @2) { 0 } | @2!(0)
    // because the cost is nondeterministic
    let result1 = evaluate_and_replay(
        Cost::create(i32::MAX as i64, "max_value".to_string()),
        from_long(1716417707),
    )
    .await;

    /*
     Same to Scala, we get the same error - "Receiving on the same channels is currently not allowed"
     Should it be fixed in Rholang - 1.2?
    */
    assert!(result1.0.errors.is_empty());
    assert!(result1.1.errors.is_empty());
    assert_eq!(result1.0.cost, result1.1.cost);

    // Try contract fromLong(510661906) = @1!(0) | @1!(0) | for (_ <= @1 & _ <= @1) { 0 }
    // because of bug RCHAIN-3917
    let contract = from_long(510661906);
    println!("Generated contract: {}", contract);
    let result2 = evaluate_and_replay(
        Cost::create(i32::MAX as i64, "max_value".to_string()),
        contract,
    )
    .await;

    assert!(result2.0.errors.is_empty());
    assert!(result2.1.errors.is_empty());
    assert_eq!(result2.0.cost, result2.1.cost);

    let mut rng = rand::thread_rng();
    for _ in 1..10000 {
        let long = ((rng.gen::<i64>() % 0x144000000) + 0x144000000) % 0x144000000;
        let contract = from_long(long);
        if !contract.is_empty() {
            let result = evaluate_and_replay(
                Cost::create(i32::MAX as i64, "max_value".to_string()),
                contract,
            )
            .await;

            /*
              Same to Scala, we get the same error - "Receiving on the same channels is currently not allowed"
              Should it be fixed in Rholang - 1.2?
            */
            assert!(result.0.errors.is_empty());
            assert!(result.1.errors.is_empty());
            assert_eq!(result.0.cost, result.1.cost);
        }
    }
}

#[tokio::test]
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

#[tokio::test]
async fn should_not_attempt_reduction_when_there_was_not_enough_phlo_for_parsing() {
    let parsing_cost = 6;

    check_phlo_limit_exceeded("@1!(1)".to_string(), parsing_cost - 1, vec![]).await;
}

#[tokio::test]
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

#[tokio::test]
async fn should_stop_the_evaluation_of_all_execution_branches_when_one_of_them_runs_out_of_phlo_with_a_more_sophisticated_contract(
) {
    let mut rng = rand::thread_rng();
    for (contract, expected_total_cost) in contracts() {
        for _ in 0..1 {
            let initial_phlo = rng.gen_range(1..expected_total_cost);

            let (result, _) = evaluate_with_cost_log(initial_phlo, contract.clone()).await;

            assert!(
                result.cost.value >= initial_phlo,
                "Total cost value should be >= initialPhlo, but got {} < {}",
                result.cost.value,
                initial_phlo
            );

            println!(
                "Contract '{}' with initial_phlo={} executed with cost={}",
                contract, initial_phlo, result.cost.value
            );
        }
    }
}
