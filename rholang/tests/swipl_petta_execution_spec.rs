use prost::Message;
use rholang::rust::interpreter::{
    swi_prolog_service::petta_execute, test_utils::utils::should_skip_petta_test,
};

/// Tests for PeTTa execution service. This service is experimental.
/// This API is not finalized. Specifically, the interface between Rholang and MeTTa
/// is not complete. Here we test simple success and failure cases.
///
/// These tests require PeTTa to be installed. Set PETTA_PATH environment variable
/// to point to the PeTTa installation directory.
/// Example: PETTA_PATH=/path/to/PeTTa cargo test

#[tokio::test]
async fn test_petta_execute_simple_swap() {
    if should_skip_petta_test() {
        return;
    }

    let metta_code = "(= (swap (Pair $x $y)) (Pair $y $x)) !(swap (Pair 1 3))";
    let result = petta_execute(metta_code).await;

    assert!(
        result.is_ok(),
        "PeTTa execution should succeed: {:?}",
        result.err()
    );
    let par = result.unwrap();

    // Verify we got a valid Par structure back
    assert!(
        !par.encode_to_vec().is_empty(),
        "Par structure should not be empty"
    );
}

#[tokio::test]
async fn test_petta_execute_fibonacci() {
    if should_skip_petta_test() {
        return;
    }

    let metta_code = r#"
        (= (fib-tr $n $a $b) (if (== $n 0) $a (fib-tr (- $n 1) $b (+ $a $b))))
        (= (fib $n) (fib-tr $n 0 1))
        !(fib 10)
    "#;
    let result = petta_execute(metta_code).await;

    assert!(
        result.is_ok(),
        "Fibonacci execution should succeed: {:?}",
        result.err()
    );
    let par = result.unwrap();

    // Verify we got a valid Par structure back (result should be 55)
    assert!(
        !par.encode_to_vec().is_empty(),
        "Par structure should not be empty"
    );
}

#[tokio::test]
async fn test_petta_execute_simple_arithmetic() {
    if should_skip_petta_test() {
        return;
    }

    let metta_code = "!(+ 1 2)";
    let result = petta_execute(metta_code).await;

    assert!(
        result.is_ok(),
        "Simple arithmetic should succeed: {:?}",
        result.err()
    );
    let par = result.unwrap();
    assert!(
        !par.encode_to_vec().is_empty(),
        "Par structure should not be empty"
    );
}

#[tokio::test]
async fn test_petta_execute_invalid_syntax() {
    if should_skip_petta_test() {
        return;
    }

    // This should fail due to invalid MeTTa syntax
    let metta_code = "(= incomplete";
    let result = petta_execute(metta_code).await;

    // We expect this to either error or return an error result
    // The exact behavior depends on PeTTa's error handling
    if result.is_err() {
        println!("Got expected error: {:?}", result.err());
    } else {
        println!(
            "PeTTa handled invalid syntax without error: {:?}",
            result.ok()
        );
    }
}

#[tokio::test]
async fn test_petta_execute_empty_code() {
    if should_skip_petta_test() {
        return;
    }

    let metta_code = "";
    let result = petta_execute(metta_code).await;

    // Empty code might succeed or fail depending on PeTTa behavior
    println!("Empty code result: {:?}", result);
}

#[tokio::test]
async fn test_petta_execute_timeout_large_fibonacci() {
    if should_skip_petta_test() {
        return;
    }

    // Fibonacci of a very large number should timeout (10 second limit)
    // fib(10000000) will take much longer than 10 seconds
    let metta_code = r#"
        (= (fib-tr $n $a $b) (if (== $n 0) $a (fib-tr (- $n 1) $b (+ $a $b))))
        (= (fib $n) (fib-tr $n 0 1))
        !(fib 10000000)
    "#;

    let result = petta_execute(metta_code).await;

    // Should fail with a timeout error
    assert!(
        result.is_err(),
        "Large fibonacci computation should timeout after 10 seconds"
    );

    let err = result.unwrap_err();
    let err_msg = format!("{:?}", err);

    println!("Timeout error (expected): {}", err_msg);

    // Error should mention timeout
    assert!(
        err_msg.contains("timed out") || err_msg.contains("timeout"),
        "Error should be a timeout error, got: {}",
        err_msg
    );
}
