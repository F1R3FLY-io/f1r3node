// Verification test for pathmap-demo.rho functionality

use rholang::rust::interpreter::{
    errors::InterpreterError,
    interpreter::EvaluateResult,
    rho_runtime::{RhoRuntime, RhoRuntimeImpl},
    test_utils::resources::with_runtime,
};

async fn execute(
    runtime: &mut RhoRuntimeImpl,
    term: &str,
) -> Result<EvaluateResult, InterpreterError> {
    runtime.evaluate_with_term(term).await
}

#[tokio::test]
async fn test_demo_scenario() {
    with_runtime("demo-verification-", |mut runtime| async move {
        // Initial database
        let db = r#"{| ["backend", "api", "done"], 
                      ["backend", "database", "in-progress"], 
                      ["frontend", "ui", "todo"],
                      ["frontend", "tests", "todo"] |}"#;
        
        // Demo 1: Query backend tasks
        let demo1 = format!(r#"
            new stdoutAck(`rho:io:stdoutAck`) in {{
              stdoutAck!({}.readZipperAt(["backend"]).getSubtrie(), Nil)
            }}
        "#, db);
        
        let res1 = execute(&mut runtime, &demo1).await.unwrap();
        assert!(res1.errors.is_empty(), "Demo 1 failed: {:?}", res1.errors);
        
        // Demo 2: Complete UI task
        let demo2 = format!(r#"
            new stdoutAck(`rho:io:stdoutAck`) in {{
              stdoutAck!(
                {}.writeZipperAt(["frontend", "ui"])
                  .setLeaf(["frontend", "ui", "done"]),
                Nil
              )
            }}
        "#, db);
        
        let res2 = execute(&mut runtime, &demo2).await.unwrap();
        assert!(res2.errors.is_empty(), "Demo 2 failed: {:?}", res2.errors);
        
        // Demo 3: Replace frontend tasks (using the result from demo2)
        let demo3 = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              stdoutAck!(
                {| ["backend", "api", "done"], 
                   ["backend", "database", "in-progress"], 
                   ["frontend", "ui", "todo"],
                   ["frontend", "tests", "todo"],
                   ["frontend", "ui", "done"] |}
                  .writeZipperAt(["frontend"])
                  .setSubtrie({| ["dashboard", "done"], ["profile", "todo"] |}),
                Nil
              )
            }
        "#;
        
        let res3 = execute(&mut runtime, demo3).await.unwrap();
        assert!(res3.errors.is_empty(), "Demo 3 failed: {:?}", res3.errors);
        
        // Demo 4: Add DevOps project (using result from demo3)
        let demo4 = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              stdoutAck!(
                {| ["backend", "api", "done"], 
                   ["backend", "database", "in-progress"],
                   ["frontend", "dashboard", "done"], 
                   ["frontend", "profile", "todo"] |}
                  .writeZipperAt(["devops"])
                  .setSubtrie({| ["deploy", "todo"], ["monitor", "in-progress"] |}),
                Nil
              )
            }
        "#;
        
        let res4 = execute(&mut runtime, demo4).await.unwrap();
        assert!(res4.errors.is_empty(), "Demo 4 failed: {:?}", res4.errors);
        
        // Demo 5: Graft operation - merge complete PathMaps (using result from demo4)
        let demo5 = r#"
            new stdoutAck(`rho:io:stdoutAck`) in {
              new zipper in {
                zipper!({| ["metrics", "cpu", "85%"], ["metrics", "memory", "60%"], ["alerts", "disk-full"] |}.readZipper()) |
                for (@z <- zipper) {
                  stdoutAck!(
                    {| ["backend", "api", "done"], 
                       ["backend", "database", "in-progress"],
                       ["frontend", "dashboard", "done"], 
                       ["frontend", "profile", "todo"],
                       ["devops", "deploy", "todo"],
                       ["devops", "monitor", "in-progress"] |}
                      .writeZipper()
                      .graft(z),
                    Nil
                  )
                }
              }
            }
        "#;
        
        let res5 = execute(&mut runtime, demo5).await.unwrap();
        assert!(res5.errors.is_empty(), "Demo 5 failed: {:?}", res5.errors);
        
    })
    .await
}

