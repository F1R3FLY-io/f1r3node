# Rholang Directory Organization

The Rholang implementation is organized as follows:

## Core Language Implementation (`src/main/`)

### BNFC Grammar (`bnfc/`)
- `rholang_mercury.cf` - BNFC grammar definition for the language parser

### Scala Implementation (`scala/coop/rchain/rholang/`)
Main Scala implementation containing 77 source files organized into:

#### Interpreter (`interpreter/`)
Core interpreter logic with the following submodules:
- `accounting/` - Cost accounting for blockchain execution
- `compiler/` - Compilation pipeline
  - `normalizer/` - AST normalization
  - `processes/` - Process handling
- `matcher/` - Pattern matching implementation
- `merging/` - Process merging utilities
- `registry/` - Name registry
- `storage/` - Storage layer
- `util/` - Utility functions
  - `codec/` - Encoding/decoding utilities

#### Build Utilities (`build/`)
Build-related utilities and configurations

## Formal Semantics (`src/main/k/`)

Multiple K framework specifications for formal verification:
- `rholang/` - Main Rholang K semantics
- `rho/` - Core rho-calculus
- `minpi2/` - Minimal pi-calculus variant
- `ski-rho/` - SKI combinator calculus in Rholang
- `yoshida-rho/` - Yoshida's variant
- `sk/` - SK combinator calculus

## RBL Backend (`src/main/rbl/`)

Rosette Base Language files (`.rbl`) serving as compilation targets or runtime components. Contains 17 RBL files including:
- Cell implementations (`Cell.rbl`, `Cell1.rbl`, `Cell2.rbl`, `Cell3.rbl`)
- Hello World examples (`HelloWorld.rbl`, `HelloWorldAgain.rbl`, etc.)
- Test files for various features
- Token implementation (`token.rbl`)

## Test Suite (`src/test/`)

Comprehensive test coverage including:
- Interpreter tests
- Accounting tests
- Pattern matching tests
- Lexer tests
- Various specification tests

## Examples (`examples/`)

87 example Rholang programs (`.rho` files) demonstrating language features:
- Tutorial examples (`tut-*.rho`)
- Basic I/O examples (`stdout.rho`, `stdoutAck.rho`, `fileinteract.rho`)
- Domain-specific examples (`bond/` directory)
- Performance tests (`longslow.rho`, `shortslow.rho`)
- Linking examples (`linking/` directory)

## Documentation and Resources

- `README.md` - Build instructions, current status, and known issues
- `reference_doc/` - Language reference documentation
- `LICENSE` - License information
- `project/` - SBT project configuration
- `rho2rbl` - Script for converting Rholang to RBL

## Build Artifacts

- `target/` - Compiled artifacts and build outputs
- `lib/` - External dependencies and libraries

## Architecture Overview

The implementation follows a classic compiler/interpreter pattern:
1. **BNFC** generates the lexer/parser from the grammar specification
2. **Scala** implements the core interpreter with pattern matching, normalization, and execution
3. **K Framework** provides formal semantics for verification
4. **RBL** serves as a lower-level target or runtime component
5. Extensive **examples** and **tests** ensure correctness and demonstrate usage

The codebase is structured to support Rholang as a concurrent programming language based on the ρ-calculus, designed for implementing protocols and smart contracts on blockchain platforms.