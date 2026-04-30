//! Test-only loader for `.rho` / `.rhox` resource files.
//!
//! Production code embeds its required Rholang sources via `include_str!`
//! at compile time (see `casper::rust::genesis::contracts::embedded_rho`).
//! The casper test suite loads test fixtures from
//! `casper/src/test/resources/` at runtime. This helper exists solely for
//! that test path — it preserves the historical multi-path search ladder
//! that the production loader used before being removed.
//!
//! If a test needs to compile a fixture, prefer:
//!
//! ```ignore
//! let code = load_test_rho("AuthKeyTest.rho");
//! let compiled = CompiledRholangSource::new(code, HashMap::new(), "AuthKeyTest.rho".into())?;
//! ```

use rholang::rust::interpreter::errors::InterpreterError;
use std::fs;

/// Reads a Rholang source/template file from one of several well-known
/// test-resource locations relative to the current working directory.
/// Mirrors the path ladder the production loader used to walk before the
/// embedding refactor, so existing tests keep finding their fixtures.
pub fn load_test_rho(filepath: &str) -> Result<String, InterpreterError> {
    let candidates = [
        format!("casper/src/test/resources/{}", filepath),
        format!("casper/src/main/resources/{}", filepath),
        format!("src/test/resources/{}", filepath),
        format!("src/main/resources/{}", filepath),
        format!("../casper/src/test/resources/{}", filepath),
        format!("../casper/src/main/resources/{}", filepath),
        format!("rholang/examples/{}", filepath),
        format!("../rholang/examples/{}", filepath),
    ];

    for candidate in &candidates {
        if let Ok(content) = fs::read_to_string(candidate) {
            return Ok(content);
        }
    }

    Err(InterpreterError::from(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "Test resource '{}' not found in any of: {:?}",
            filepath, candidates
        ),
    )))
}
