// See rholang/src/test/scala/coop/rchain/rholang/interpreter/AbortSpec.scala

use rholang::rust::interpreter::{
    errors::InterpreterError,
    rho_runtime::RhoRuntime,
    test_utils::resources::with_runtime,
};

/// Tests for the rho:execution:abort system process
/// 
/// The abort system process allows Rholang code to explicitly terminate execution.
/// When called, it raises a UserAbortError that propagates up and terminates the
/// deploy with an error result.

#[tokio::test]
async fn abort_should_terminate_execution_with_user_abort_error() {
    with_runtime("abort-spec-", |mut runtime| async move {
        let rho_code = r#"
            new abort(`rho:execution:abort`) in {
              abort!("Test abort")
            }
        "#;

        let result = runtime.evaluate_with_term(rho_code).await.unwrap();
        
        // Abort should result in UserAbortError and mark execution as failed
        assert!(
            result.errors.contains(&InterpreterError::UserAbortError),
            "Expected UserAbortError, got: {:?}",
            result.errors
        );
        
        // Cost should be non-zero (some execution happened before abort)
        assert!(result.cost.value > 0, "Expected non-zero cost, got: {:?}", result.cost);
    })
    .await
}

#[tokio::test]
async fn abort_without_message_should_terminate_execution() {
    with_runtime("abort-spec-no-msg-", |mut runtime| async move {
        let rho_code = r#"
            new abort(`rho:execution:abort`) in {
              abort!(Nil)
            }
        "#;

        let result = runtime.evaluate_with_term(rho_code).await.unwrap();
        
        // Abort should result in UserAbortError
        assert!(
            result.errors.contains(&InterpreterError::UserAbortError),
            "Expected UserAbortError, got: {:?}",
            result.errors
        );
    })
    .await
}

#[tokio::test]
async fn abort_should_stop_parallel_execution() {
    with_runtime("abort-spec-parallel-", |mut runtime| async move {
        // In parallel execution, abort should halt the entire computation
        let rho_code = r#"
            new abort(`rho:execution:abort`), result in {
              @"test"!(1) | abort!("Stop everything") | @"test"!(2)
            }
        "#;

        let result = runtime.evaluate_with_term(rho_code).await.unwrap();
        
        // Abort should result in UserAbortError
        assert!(
            result.errors.contains(&InterpreterError::UserAbortError),
            "Expected UserAbortError, got: {:?}",
            result.errors
        );
    })
    .await
}
