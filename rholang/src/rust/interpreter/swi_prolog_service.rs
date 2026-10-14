use std::io::Write;
use std::path::PathBuf;
use std::{env, time::Duration};
// use std::process::Command;
use tokio::process::Command;

use models::rhoapi::Par;
use serde_json::Value;
use tempfile::NamedTempFile;

use crate::rust::interpreter::rho_type::{
    RhoBoolean, RhoList, RhoMap, RhoNil, RhoNumber, RhoString,
};

use super::errors::InterpreterError;

/// Executes MeTTa code through the PeTTa (SWI-Prolog) interpreter and returns the result as a Rholang Par.
///
/// # Overview
///
/// This function provides low-level access to the PeTTa interpreter, which
/// provides MeTTa execution with SWI-Prolog. The execution is sandboxed with
/// a 10-second timeout to prevent runaway computations.
///
/// # Arguments
///
/// * `metta_code` - A string containing valid MeTTa code to execute
///
/// # Returns
///
/// Returns `Ok(Par)` containing the execution result as a Rholang Par structure, or an
/// `InterpreterError::SwiplError` if execution fails.
///
/// # JSON Output Schema
///
/// PeTTa returns results as JSON in the following envelope:
/// ```json
/// {"results": [...]}
/// ```
///
/// The `results` field contains a JSON array of MeTTa execution results. This entire JSON
/// structure is converted to a Rholang Par using the following mapping:
///
/// - `null` → `RhoNil`
/// - `true`/`false` → `RhoBoolean`
/// - Numbers → `RhoNumber` (must fit in i64, otherwise error)
/// - Strings → `RhoString`
/// - Arrays → `RhoList` (recursive conversion of elements)
/// - Objects → `RhoMap` (keys converted to RhoString, values recursively converted)
///
/// # Error Conditions
///
/// - `InterpreterError::SwiplError("Can't find PeTTa.")` - PeTTa installation not found at `$PETTA_PATH`
/// - `InterpreterError::SwiplError("Can't open temp file")` - Failed to create temporary file
/// - `InterpreterError::SwiplError("MeTTa execution timed out...")` - Execution exceeded 10 seconds
/// - `InterpreterError::SwiplError("PeTTa execution failed...")` - SWI-Prolog returned error
/// - `InterpreterError::SwiplError("Can't parse JSON output...")` - Invalid JSON from PeTTa
/// - `InterpreterError::SwiplError("Could not parse number as i64")` - Number exceeds i64 range
///
/// # Environment Variables
///
/// - `PETTA_PATH` - Path to PeTTa installation directory (default: `./PeTTa`)
///
/// # Timeout
///
/// Execution is limited to 10 seconds. Long-running computations will be terminated and return
/// a timeout error. This prevents malicious or buggy MeTTa code from blocking the node.
///
/// # Examples
///
/// ```ignore
/// // Simple arithmetic
/// let result = petta_execute("!(+ 1 2)").await?;
///
/// // Pattern matching
/// let result = petta_execute(
///     "(= (swap (Pair $x $y)) (Pair $y $x)) !(swap (Pair 1 3))"
/// ).await?;
/// ```
///
/// # See Also
///
/// - [`system_processes::swipl_execute_petta`] - System process wrapper for Rholang contracts
/// - [`value_to_par`] - JSON to Par conversion logic
pub async fn petta_execute(metta_code: &str) -> Result<Par, InterpreterError> {
    // Write the MeTTa code to a temp file
    let mut metta_file = NamedTempFile::new()
        .map_err(|_| InterpreterError::SwiplError("Can't open temp file".into()))?;
    metta_file
        .write(metta_code.as_bytes())
        .map_err(|_| InterpreterError::SwiplError("Can't write MeTTa code to temp file".into()))?;

    let metta_file_path = metta_file
        .path()
        .to_str()
        .ok_or(InterpreterError::SwiplError(
            "Can't convert metta_file path to string".into(),
        ))?;

    // Get the path to PeTTa
    let metta_module_path: PathBuf = {
        let petta_path = PathBuf::from(env::var("PETTA_PATH").unwrap_or("./PeTTa".into()));
        [petta_path, PathBuf::from("src/metta.pl")].iter().collect()
    };

    if !metta_module_path.exists() {
        return Err(InterpreterError::SwiplError("Can't find PeTTa.".into()));
    }

    let goal = format!(
        r#"assertz(silent(true)),
           load_metta_file('{metta_file_path}', Results),
           use_module(library(json)),
           json_write_dict(current_output, #{{results:Results}})."#
    );

    // TODO: Make this a configuration parameter
    let timeout_secs: u64 = 10;
    let proc_handle = tokio::spawn(tokio::time::timeout(
        Duration::from_secs(timeout_secs),
        Command::new("swipl")
            .arg("-s")
            .arg(metta_module_path)
            .arg("-g")
            .arg(goal)
            .arg("-t")
            .arg("halt")
            .kill_on_drop(true)
            .output(),
    ));

    let output = proc_handle
        .await
        .map_err(|join_error| {
            InterpreterError::SwiplError(
                format!("Error while joining with the PeTTa task: {}", join_error).into(),
            )
        })?
        .map_err(|elapsed| {
            InterpreterError::SwiplError(
                format!("MeTTa execution timed out after {}", elapsed).into(),
            )
        })?
        .map_err(|e| {
            InterpreterError::SwiplError(format!("MeTTa execution failed: {}", e).into())
        })?;

    if !output.status.success() {
        return Err(InterpreterError::SwiplError(
            format!("PeTTa execution failed. {:#?}", output.stderr.as_slice()).into(),
        ));
    }

    // Get output as string
    let str_output = String::from_utf8(output.stdout)
        .map_err(|_| InterpreterError::SwiplError("Can't interpret PeTTa output".into()))?;

    let value_output = serde_json::from_str::<Value>(str_output.as_str()).map_err(|_| {
        InterpreterError::SwiplError("Can't parse JSON output from PeTTa execution".into())
    })?;
    let par_output = value_to_par(value_output)?;
    Ok(par_output)
}

/// Converts a JSON Value to a Rholang Par structure.
///
/// This function recursively transforms JSON data returned by PeTTa into Rholang's internal
/// representation (Par). It is used internally by [`petta_execute`] to convert PeTTa results.
///
/// # Type Mapping
///
/// | JSON Type | Rholang Type | Notes |
/// |-----------|--------------|-------|
/// | `null` | `RhoNil` | Represents absence of value |
/// | `boolean` | `RhoBoolean` | Direct mapping |
/// | `number` | `RhoNumber` | **Must fit in i64**, otherwise returns error |
/// | `string` | `RhoString` | Direct mapping, supports Unicode |
/// | `array` | `RhoList` | Elements recursively converted |
/// | `object` | `RhoMap` | Keys stringified, values recursively converted |
///
/// # Important Constraints
///
/// - **Numbers must fit in i64**: JSON numbers that exceed `i64::MIN` to `i64::MAX` will cause
///   an error. Floating-point numbers are truncated to integers.
/// - **Object keys become strings**: All JSON object keys are converted to `RhoString` in the
///   resulting `RhoMap`.
/// - **Recursive conversion**: Nested structures (arrays in arrays, objects in objects, etc.)
///   are fully supported and recursively converted.
///
/// # Arguments
///
/// * `v` - A `serde_json::Value` to convert
///
/// # Returns
///
/// Returns `Ok(Par)` with the converted structure, or `InterpreterError::SwiplError` if
/// conversion fails (e.g., number doesn't fit in i64).
///
/// # Examples
///
/// ```ignore
/// use serde_json::json;
///
/// // Simple values
/// let nil = value_to_par(json!(null))?;
/// let bool = value_to_par(json!(true))?;
/// let num = value_to_par(json!(42))?;
/// let str = value_to_par(json!("hello"))?;
///
/// // Collections
/// let list = value_to_par(json!([1, 2, 3]))?;
/// let map = value_to_par(json!({"key": "value"}))?;
///
/// // Nested structures
/// let nested = value_to_par(json!({
///     "list": [1, 2, 3],
///     "map": {"inner": "value"}
/// }))?;
/// ```
fn value_to_par(v: Value) -> Result<Par, InterpreterError> {
    match v {
        Value::Null => Ok(RhoNil::create_par()),
        Value::Bool(b) => Ok(RhoBoolean::create_par(b)),
        Value::Number(n) => {
            let n64 = n.as_i64().ok_or(InterpreterError::SwiplError(
                "Could not parse number as i64".into(),
            ))?;
            Ok(RhoNumber::create_par(n64))
        }
        Value::String(s) => Ok(RhoString::create_par(s)),
        Value::Array(values) => {
            let ps = values
                .into_iter()
                .map(value_to_par)
                .collect::<Result<_, _>>()?;
            Ok(RhoList::create_par(ps))
        }
        Value::Object(map) => {
            let hashmap = map
                .into_iter()
                .map(|(k, v)| {
                    let p = value_to_par(v)?;
                    Ok((RhoString::create_par(k), p))
                })
                .collect::<Result<_, InterpreterError>>()?;
            Ok(RhoMap::create_par(hashmap))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn test_value_to_par_null() {
        let result = value_to_par(json!(null)).unwrap();
        assert_eq!(result, RhoNil::create_par());
    }
    #[test]
    fn test_value_to_par_boolean() {
        let result = value_to_par(json!(true)).unwrap();
        assert_eq!(result, RhoBoolean::create_par(true));
    }
    #[test]
    fn test_value_to_par_small_number() {
        let result = value_to_par(json!(42)).unwrap();
        assert_eq!(result, RhoNumber::create_par(42));
    }
    #[test]
    fn test_value_to_par_max_i64() {
        let result = value_to_par(json!(i64::MAX)).unwrap();
        assert_eq!(result, RhoNumber::create_par(i64::MAX));
    }
    #[test]
    fn test_value_to_par_string() {
        let result = value_to_par(json!("hello")).unwrap();
        assert_eq!(result, RhoString::create_par("hello".into()));
    }
    #[test]
    fn test_value_to_par_array() {
        let result = value_to_par(json!([1, "two", true])).unwrap();
        assert_eq!(
            result,
            RhoList::create_par(vec![
                RhoNumber::create_par(1),
                RhoString::create_par("two".into()),
                RhoBoolean::create_par(true)
            ])
        );
    }
    #[test]
    fn test_value_to_par_nested_array() {
        let result = value_to_par(json!([[1, 2], [3, 4]])).unwrap();
        assert_eq!(
            result,
            RhoList::create_par(vec![
                RhoList::create_par(vec![RhoNumber::create_par(1), RhoNumber::create_par(2)]),
                RhoList::create_par(vec![RhoNumber::create_par(3), RhoNumber::create_par(4)])
            ])
        );
    }
    #[test]
    fn test_value_to_par_object() {
        let result = value_to_par(json!({"key": "value", "num": 123})).unwrap();
        assert_eq!(
            result,
            RhoMap::create_par(
                vec![
                    (
                        RhoString::create_par("key".into()),
                        RhoString::create_par("value".into())
                    ),
                    (
                        RhoString::create_par("num".into()),
                        RhoNumber::create_par(123)
                    )
                ]
                .into_iter()
                .collect()
            )
        );
    }
    #[test]
    fn test_value_to_par_nested_object() {
        let result = value_to_par(json!({
            "outer": {
                "inner": "value"
            }
        }))
        .unwrap();
        assert_eq!(
            result,
            RhoMap::create_par(
                vec![(
                    RhoString::create_par("outer".into()),
                    RhoMap::create_par(
                        vec![(
                            RhoString::create_par("inner".into()),
                            RhoString::create_par("value".into())
                        )]
                        .into_iter()
                        .collect()
                    )
                )]
                .into_iter()
                .collect()
            )
        );
    }

    #[test]
    fn test_value_to_par_complex_nested_structure() {
        let result = value_to_par(json!({
            "list": [1, 2, 3],
            "nested": {
                "bool": true,
                "null": null
            }
        }))
        .unwrap();

        assert_eq!(
            result,
            RhoMap::create_par(
                vec![
                    (
                        RhoString::create_par("list".into()),
                        RhoList::create_par(vec![
                            RhoNumber::create_par(1),
                            RhoNumber::create_par(2),
                            RhoNumber::create_par(3)
                        ])
                    ),
                    (
                        RhoString::create_par("nested".into()),
                        RhoMap::create_par(
                            vec![
                                (
                                    RhoString::create_par("bool".into()),
                                    RhoBoolean::create_par(true)
                                ),
                                (RhoString::create_par("null".into()), RhoNil::create_par())
                            ]
                            .into_iter()
                            .collect()
                        )
                    )
                ]
                .into_iter()
                .collect()
            )
        );
    }

    #[test]
    fn test_value_to_par_empty_array() {
        let result = value_to_par(json!([])).unwrap();
        assert_eq!(result, RhoList::create_par(vec![]));
    }

    #[test]
    fn test_value_to_par_empty_object() {
        let result = value_to_par(json!({})).unwrap();
        assert_eq!(result, RhoMap::create_par(std::collections::HashMap::new()));
    }

    #[test]
    fn test_value_to_par_negative_number() {
        let result = value_to_par(json!(-42)).unwrap();
        assert_eq!(result, RhoNumber::create_par(-42));
    }

    #[test]
    fn test_value_to_par_zero() {
        let result = value_to_par(json!(0)).unwrap();
        assert_eq!(result, RhoNumber::create_par(0));
    }

    #[test]
    fn test_value_to_par_false_boolean() {
        let result = value_to_par(json!(false)).unwrap();
        assert_eq!(result, RhoBoolean::create_par(false));
    }

    #[test]
    fn test_value_to_par_empty_string() {
        let result = value_to_par(json!("")).unwrap();
        assert_eq!(result, RhoString::create_par("".into()));
    }

    #[test]
    fn test_value_to_par_unicode_string() {
        let result = value_to_par(json!("Hello, 世界! 🌍")).unwrap();
        assert_eq!(result, RhoString::create_par("Hello, 世界! 🌍".into()));
    }

    /// Test that petta_execute times out after 10 seconds for long-running computations.
    /// This uses a large fibonacci number that should exceed the timeout.
    #[tokio::test]
    async fn test_petta_execute_timeout() {
        use std::env;
        use std::path::PathBuf;

        // Check if PeTTa is available
        let petta_path = PathBuf::from(env::var("PETTA_PATH").unwrap_or("./PeTTa".into()));
        let metta_module_path: PathBuf =
            [petta_path, PathBuf::from("src/metta.pl")].iter().collect();

        if !metta_module_path.exists() {
            eprintln!("Skipping timeout test: PeTTa not available");
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
            "Large fibonacci computation should timeout"
        );

        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);

        // Error should mention timeout
        assert!(
            err_msg.contains("timed out") || err_msg.contains("timeout"),
            "Error should be a timeout error, got: {}",
            err_msg
        );
    }
}
