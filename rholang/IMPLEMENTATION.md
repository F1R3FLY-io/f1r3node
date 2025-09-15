# Rholang Implementation Architecture

## Overview

Rholang is a concurrent programming language based on the ρ-calculus (reflective higher-order π-calculus), designed for implementing protocols and smart contracts on blockchain platforms. The implementation follows a classic compiler/interpreter pattern with formal semantics verification.

## Architecture Components

### 1. Language Syntax and Parsing

**BNFC Grammar** (`src/main/bnfc/rholang_mercury.cf`)
- Defines the complete syntax of Rholang using BNFC (Backus-Naur Form Converter)
- Generates lexer and parser automatically
- Key language constructs:
  - **Processes**: Basic computational units (Par, Send, Receive, etc.)
  - **Names**: Channels for communication (`@` for quoting processes)
  - **Pattern Matching**: `match` expressions with cases
  - **Concurrency**: Parallel composition with `|`
  - **Contracts**: Persistent processes with `contract`
  - **Bundles**: Read/write capabilities on channels

### 2. Compilation Pipeline

**Compiler** (`compiler/Compiler.scala`)
```
Source Code → AST → Normalization → Sorting → Par (Internal Representation)
```

1. **Parsing**: BNFC-generated parser converts source to AST
2. **Normalization**: AST transformed to internal `Par` representation
   - Each AST node type has a dedicated normalizer (e.g., `PSendNormalizer`, `PInputNormalizer`)
   - De Bruijn indexing for variable binding
   - Environment tracking for free variables
3. **Sorting**: Canonical ordering for deterministic execution

### 3. Core Interpreter

**InterpreterImpl** (`Interpreter.scala`)
- Entry point for code execution
- Manages cost accounting (phlogiston-based gas system)
- Error handling and result aggregation
- Integration with the reducer

**DebruijnInterpreter/Reduce** (`Reduce.scala`)
- Core evaluation engine using De Bruijn indices
- Key operations:
  - `eval`: Evaluates Par expressions
  - `produce`: Sends data on channels
  - `consume`: Receives data from channels
  - Pattern matching for channel operations

### 4. Pattern Matching

**SpatialMatcher** (`matcher/SpatialMatcher.scala`)
- Implements spatial-behavioral type system matching
- Uses StateT monad for tracking variable bindings
- Supports:
  - Structural pattern matching on processes
  - Variable capture and binding
  - Backtracking for multiple matches
  - Free variable tracking

### 5. Storage Layer

**RhoRuntime and Tuplespace**
- **RSpace**: Underlying storage for channel-based communication
  - Persistent key-value store for channels and data
  - Support for persistent and ephemeral sends
  - Pattern-based retrieval
- **ChargingRSpace**: Wrapper adding cost accounting
- Operations:
  - `produce`: Store data on a channel
  - `consume`: Retrieve and remove data matching patterns
  - `peek`: Non-destructive read

### 6. Cost Accounting

**Accounting Module** (`accounting/`)
- Phlogiston-based gas system for blockchain execution
- Cost calculation for:
  - Parsing
  - Pattern matching
  - Storage operations
  - Computation steps
- Prevents infinite loops and resource exhaustion

### 7. System Processes

**Built-in Functionality**
- Registry for persistent naming
- System channels for I/O
- Cryptographic operations
- Blockchain-specific operations (deploy info, validators)

### 8. Formal Semantics

**K Framework Specifications** (`src/main/k/`)
- Formal operational semantics in K
- Multiple calculus implementations:
  - `rholang/`: Main Rholang semantics
  - `rho/`: Core ρ-calculus
  - `minpi2/`: Minimal π-calculus
- Enables formal verification of language properties

## Execution Flow

1. **Parse**: Source code → AST via BNFC-generated parser
2. **Normalize**: AST → Par with De Bruijn indices
3. **Sort**: Canonical ordering for determinism
4. **Reduce**: Evaluate Par expression
   - Pattern match on Par type
   - Execute corresponding operation
   - Update tuplespace for sends/receives
5. **Cost Accounting**: Track and charge for operations
6. **Result**: Return evaluation result with cost and errors

## Key Design Decisions

1. **Reflective Architecture**: Processes and names are interconvertible via quote (`@`) and dereference (`*`)
2. **Pattern-Based Communication**: Receives use pattern matching, not just name equality
3. **Persistent Contracts**: `contract` creates reusable service endpoints
4. **Spatial Types**: Structure-aware type system for pattern matching
5. **Cost Accounting**: Every operation has an associated cost for blockchain deployment
6. **Formal Verification**: K framework specifications ensure correctness

## Implementation Languages

- **Scala**: Core interpreter and runtime (77 source files)
- **BNFC**: Grammar specification
- **K Framework**: Formal semantics
- **RBL (Rosette Base Language)**: Lower-level compilation target

## Testing and Examples

- **87 example programs** demonstrating language features
- **Comprehensive test suite** covering:
  - Interpreter correctness
  - Pattern matching
  - Cost accounting
  - Concurrent behavior
- **Tutorial examples** for learning the language

## Future Considerations

Based on the README, several features are currently broken or in development:
- Guarded patterns in receives
- 0-arity send/receive
- Pre-evaluation of match cases
- Various optimizations for production use