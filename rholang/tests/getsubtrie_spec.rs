// Test for getSubtrie() method

use rholang::rust::interpreter::{
    errors::InterpreterError,
    interpreter::EvaluateResult,
    rho_runtime::{RhoRuntime, RhoRuntimeImpl},
    test_utils::resources::with_runtime,
};

async fn success(runtime: &mut RhoRuntimeImpl, term: &str) -> Result<(), InterpreterError> {
    execute(runtime, term).await.map(|res| {
        assert!(
            res.errors.is_empty(),
            "{}",
            format!("Execution failed for: {}. Cause: {:?}", term, res.errors)
        )
    })
}

async fn execute(
    runtime: &mut RhoRuntimeImpl,
    term: &str,
) -> Result<EvaluateResult, InterpreterError> {
    runtime.evaluate_with_term(term).await
}

#[tokio::test]
async fn test_get_subtrie_at_prefix() {
    with_runtime("get-subtrie-prefix-", |mut runtime| async move {
        let rho_code = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              stdoutAck!("Test getSubtrie at prefix", Nil) |
              stdoutAck!(
                {| ["books", "fiction", "gatsby"], ["books", "fiction", "moby"], ["books", "nonfiction", "history"] |}
                  .readZipperAt(["books", "fiction"])
                  .getSubtrie(),
                Nil
              )
            }
        "#;

        success(&mut runtime, rho_code).await.unwrap();
    })
    .await
}

#[tokio::test]
async fn test_get_subtrie_at_root() {
    with_runtime("get-subtrie-root-", |mut runtime| async move {
        let rho_code = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              stdoutAck!("Test getSubtrie at root", Nil) |
              stdoutAck!(
                {| ["a", "b"], ["c", "d"] |}
                  .readZipper()
                  .getSubtrie(),
                Nil
              )
            }
        "#;

        success(&mut runtime, rho_code).await.unwrap();
    })
    .await
}

#[tokio::test]
async fn test_get_subtrie_on_direct_pathmap() {
    with_runtime("get-subtrie-direct-", |mut runtime| async move {
        let rho_code = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              stdoutAck!("Test getSubtrie on direct PathMap", Nil) |
              stdoutAck!(
                {| ["x", "y", "z"] |}.getSubtrie(),
                Nil
              )
            }
        "#;

        success(&mut runtime, rho_code).await.unwrap();
    })
    .await
}

#[tokio::test]
async fn test_get_subtrie_empty_result() {
    with_runtime("get-subtrie-empty-", |mut runtime| async move {
        let rho_code = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              stdoutAck!("Test getSubtrie with non-existent path", Nil) |
              stdoutAck!(
                {| ["books", "fiction", "gatsby"] |}
                  .readZipperAt(["books", "science"])
                  .getSubtrie(),
                Nil
              )
            }
        "#;

        success(&mut runtime, rho_code).await.unwrap();
    })
    .await
}


