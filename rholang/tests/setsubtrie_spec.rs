// Test for setSubtrie() method

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
async fn test_set_subtrie_at_prefix() {
    with_runtime("set-subtrie-prefix-", |mut runtime| async move {
        let rho_code = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              stdoutAck!("Test setSubtrie at prefix", Nil) |
              stdoutAck!(
                {| ["root", "a", "x"], ["root", "a", "y"], ["root", "b", "z"] |}
                  .writeZipperAt(["root", "a"])
                  .setSubtrie({| ["new1"], ["new2"] |}),
                Nil
              )
            }
        "#;

        success(&mut runtime, rho_code).await.unwrap();
    })
    .await
}

#[tokio::test]
async fn test_set_subtrie_at_root() {
    with_runtime("set-subtrie-root-", |mut runtime| async move {
        let rho_code = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              stdoutAck!("Test setSubtrie at root", Nil) |
              stdoutAck!(
                {| ["old1"], ["old2"] |}
                  .writeZipper()
                  .setSubtrie({| ["new1"], ["new2"], ["new3"] |}),
                Nil
              )
            }
        "#;

        success(&mut runtime, rho_code).await.unwrap();
    })
    .await
}

#[tokio::test]
async fn test_set_subtrie_empty_source() {
    with_runtime("set-subtrie-empty-", |mut runtime| async move {
        let rho_code = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              stdoutAck!("Test setSubtrie with empty source", Nil) |
              stdoutAck!(
                {| ["root", "a", "x"], ["root", "a", "y"], ["root", "b", "z"] |}
                  .writeZipperAt(["root", "a"])
                  .setSubtrie({| |}),
                Nil
              )
            }
        "#;

        success(&mut runtime, rho_code).await.unwrap();
    })
    .await
}

#[tokio::test]
async fn test_set_subtrie_requires_write_zipper() {
    with_runtime("set-subtrie-error-", |mut runtime| async move {
        let rho_code = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              stdoutAck!("Test setSubtrie requires write zipper", Nil) |
              stdoutAck!(
                {| ["a", "b"] |}
                  .readZipper()
                  .setSubtrie({| ["new"] |}),
                Nil
              )
            }
        "#;

        // This should produce an error
        let result = execute(&mut runtime, rho_code).await;
        assert!(result.is_err() || result.unwrap().errors.len() > 0);
    })
    .await
}

#[tokio::test]
async fn test_set_subtrie_multi_level_paths() {
    with_runtime("set-subtrie-multi-level-", |mut runtime| async move {
        let rho_code = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              stdoutAck!("Test setSubtrie with multi-level paths", Nil) |
              stdoutAck!(
                {| ["backend", "api"] |}
                  .writeZipperAt(["devops"])
                  .setSubtrie({| ["deploy", "todo"], ["monitor", "in-progress"] |}),
                Nil
              )
            }
        "#;

        success(&mut runtime, rho_code).await.unwrap();
    })
    .await
}

