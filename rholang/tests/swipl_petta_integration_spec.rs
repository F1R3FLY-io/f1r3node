use crypto::rust::hash::blake2b512_random::Blake2b512Random;
use models::rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation};
use rholang::rust::interpreter::{
    accounting::costs::Cost,
    external_services::ExternalServices,
    interpreter::EvaluateResult,
    rho_runtime::{RhoRuntime, RhoRuntimeImpl},
    test_utils::{resources::create_runtimes_with_services, utils::should_skip_petta_test},
};
use rspace_plus_plus::rspace::history::history_repository::HistoryRepository;
use rspace_plus_plus::rspace::shared::{
    in_mem_store_manager::InMemoryStoreManager, key_value_store_manager::KeyValueStoreManager,
};
use std::collections::HashMap;
use std::sync::Arc;

async fn evaluate_petta_term(term: &str) -> EvaluateResult {
    let mut kvm = InMemoryStoreManager::new();
    let store = kvm.r_space_stores().await.unwrap();

    let (runtime, _, _): (
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
    ) = create_runtimes_with_services(store, false, &mut Vec::new(), ExternalServices::noop())
        .await;

    let rand = Blake2b512Random::create_from_bytes(&[]);
    let initial_phlo = Cost::create(i64::MAX, "test".to_string());

    runtime
        .evaluate(term, initial_phlo, HashMap::new(), rand)
        .await
        .expect("Evaluation failed")
}

#[tokio::test]
async fn test_petta_rholang_integration_swap() {
    if should_skip_petta_test() {
        return;
    }

    let term = r#"
        new executePetta(`rho:petta:execute`), stdout(`rho:io:stdout`), retCh in {
            executePetta!("(= (swap (Pair $x $y)) (Pair $y $x)) !(swap (Pair 1 3))", *retCh) |
            for(@result <- retCh) {
                stdout!(result)
            }
        }
    "#;

    let result = evaluate_petta_term(term).await;

    assert!(
        result.errors.is_empty(),
        "PeTTa swap should execute without errors: {:?}",
        result.errors
    );
}

#[tokio::test]
async fn test_petta_rholang_integration_fibonacci() {
    if should_skip_petta_test() {
        return;
    }

    let term = r#"
        new executePetta(`rho:petta:execute`), retCh in {
            executePetta!("(= (fib-tr $n $a $b) (if (== $n 0) $a (fib-tr (- $n 1) $b (+ $a $b)))) (= (fib $n) (fib-tr $n 0 1)) !(fib 10)", *retCh)
        }
    "#;

    let result = evaluate_petta_term(term).await;

    assert!(
        result.errors.is_empty(),
        "PeTTa fibonacci should execute without errors: {:?}",
        result.errors
    );
}

#[tokio::test]
async fn test_petta_rholang_integration_arithmetic() {
    if should_skip_petta_test() {
        return;
    }

    let term = r#"
        new executePetta(`rho:petta:execute`), retCh in {
            executePetta!("!(+ 1 2)", *retCh) |
            for(@result <- retCh) {
                retCh!(result)
            }
        }
    "#;

    let result = evaluate_petta_term(term).await;

    assert!(
        result.errors.is_empty(),
        "PeTTa arithmetic should execute without errors: {:?}",
        result.errors
    );
}

#[tokio::test]
async fn test_petta_rholang_multiple_calls() {
    if should_skip_petta_test() {
        return;
    }

    let term = r#"
        new executePetta(`rho:petta:execute`), ret1, ret2 in {
            executePetta!("!(+ 1 2)", *ret1) |
            executePetta!("!(* 3 4)", *ret2) |
            for(@r1 <- ret1; @r2 <- ret2) {
                ret1!(r1) | ret2!(r2)
            }
        }
    "#;

    let result = evaluate_petta_term(term).await;

    assert!(
        result.errors.is_empty(),
        "Multiple PeTTa calls should execute without errors: {:?}",
        result.errors
    );
}

#[tokio::test]
async fn test_petta_rholang_error_handling() {
    if should_skip_petta_test() {
        return;
    }

    // Test with invalid MeTTa syntax
    let term = r#"
        new executePetta(`rho:petta:execute`), retCh in {
            executePetta!("(= incomplete", *retCh)
        }
    "#;

    let result = evaluate_petta_term(term).await;

    // Should produce an error due to invalid syntax
    assert!(
        !result.errors.is_empty(),
        "Invalid MeTTa syntax should produce errors"
    );
}

#[tokio::test]
async fn test_petta_rholang_timeout_large_computation() {
    if should_skip_petta_test() {
        return;
    }

    // Test that timeout is enforced through the full Rholang runtime
    // This fibonacci computation should timeout after 10 seconds
    let term = r#"
        new executePetta(`rho:petta:execute`), retCh in {
            executePetta!("(= (fib-tr $n $a $b) (if (== $n 0) $a (fib-tr (- $n 1) $b (+ $a $b)))) (= (fib $n) (fib-tr $n 0 1)) !(fib 10000000)", *retCh)
        }
    "#;

    let result = evaluate_petta_term(term).await;

    // Should have errors due to timeout
    assert!(
        !result.errors.is_empty(),
        "Large fibonacci computation should timeout and produce errors: {:?}",
        result.errors
    );

    // Check that at least one error mentions timeout
    let has_timeout_error = result.errors.iter().any(|err| {
        let err_str = format!("{:?}", err);
        err_str.contains("timed out") || err_str.contains("timeout")
    });

    assert!(
        has_timeout_error,
        "At least one error should mention timeout, errors: {:?}",
        result.errors
    );
}
