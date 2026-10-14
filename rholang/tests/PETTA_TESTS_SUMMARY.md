# PeTTa Tests Summary

This document summarizes the test suite implemented for the PeTTa (MeTTa +
SWI-Prolog) execution functionality in RNode.

## Requirements

For the PeTTa tests to run, the following requirements must be met:

1. **The PeTTa installation must be available**.

   See [DEVELOPER.md](../../DEVELOPER.md) for details on how to initialize this git
   submodule.

   The installation location must be specified in the `PETTA_PATH` environment
   variable. If this variable is not set, the relative path `./PeTTa` will be used,
   which may fail or not depending on how the tests are run. Therefore, setting this
   variable is encouraged.

2. **`swipl` must be available in the `PATH`**.

   See [DEVELOPER.md](../../DEVELOPER.md) for details on how to install this
   program.

If these are not available, the tests will be skipped by default. If this behaviour
is not desirable (e.g: all tests must be run as part of CI), then the environment
variable `REQUIRE_PETTA_TESTS` must be set.

## Test Coverage Overview

The test suite consists of:
1. Unit tests for `value_to_par` function
2. Unit tests for `petta_execute` function  
3. Integration tests with Rholang runtime
4. Replay tests to verify non-deterministic operation handling

## Test Files

### 1. Unit Tests: `value_to_par` Function and `petta_execute`
**Location:** `rholang/src/rust/interpreter/swi_prolog_service.rs` (lines 127-360)

**Coverage:** 18 unit tests

Tests JSON→Par conversion for all data types:
- Basic types: `null`, `boolean`, `number`, `string`
- Collections: `array`, `object`
- Nested structures: nested arrays, nested objects.
- **Timeout test:** Large fibonacci computation that exceeds 10-second timeout

**Run:** `PETTA_PATH=/path/to/PeTTa cargo test --package rholang --lib swi_prolog_service::tests`

### 2. Direct Execution Tests
**Location:** `rholang/tests/swipl_petta_execution_spec.rs`

**Coverage:** 6 tests

Tests the `petta_execute` function directly:
- `test_petta_execute_simple_swap` - Basic pattern matching
- `test_petta_execute_fibonacci` - Recursive function execution
- `test_petta_execute_simple_arithmetic` - Simple operations
- `test_petta_execute_invalid_syntax` - Error handling
- `test_petta_execute_empty_code` - Edge case handling
- `test_petta_execute_timeout_large_fibonacci` - Timeout enforcement for long-running computations

**Run:** `PETTA_PATH=/path/to/PeTTa cargo test --package rholang --test swipl_petta_execution_spec`

### 3. Integration Tests with Rholang Runtime
**Location:** `rholang/tests/swipl_petta_integration_spec.rs`

**Coverage:** 6 tests

Same tests as above, but using the Rholang runtime.

**Run:** `PETTA_PATH=/path/to/PeTTa cargo test --package rholang --test swipl_petta_integration_spec`

### 4. Replay Tests (Non-Deterministic Operation Verification)
**Location:** `rholang/tests/swipl_petta_replay_spec.rs`

**Coverage:** 5 tests

Critical tests for consensus safety:
- `test_petta_is_registered_as_non_deterministic` - Verifies `SWIPL_EXECUTE_PETTA` in `non_deterministic_ops()`
- `test_petta_replay_consistency` - Basic replay with cached output
- `test_petta_replay_with_multiple_calls` - Multiple PeTTa calls in one contract
- `test_petta_replay_error_consistency` - Error cases are replayed correctly
- `test_petta_replay_uses_cached_output` - Verifies replay doesn't re-execute PeTTa

**Run:** `PETTA_PATH=/path/to/PeTTa cargo test --package rholang --test swipl_petta_replay_spec`

## Running All Tests

### With PeTTa Installed
```bash
# Set PeTTa path
export PETTA_PATH=/path/to/PeTTa

# Run all PeTTa tests
cargo test --package rholang --test swipl_petta_execution_spec
cargo test --package rholang --test swipl_petta_integration_spec  
cargo test --package rholang --test swipl_petta_replay_spec

# Run unit tests
cargo test --package rholang --lib swi_prolog_service::tests
```
