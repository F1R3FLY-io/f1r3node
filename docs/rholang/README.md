# Rholang Developer Reference

Verified against the f1r3node-rust implementation (Rust shard).

## Reading Order

### Fundamentals
1. [Language Overview](01-language-overview.md) -- Process calculus, names vs processes, core model
2. [Syntax Reference](02-syntax-reference.md) -- Complete syntax for all language constructs
3. [Data Types](03-data-types.md) -- Primitives, strings, byte arrays, URIs, booleans
4. [Operators](04-operators.md) -- Arithmetic, comparison, logical, string, interpolation

### Types and Collections
5. [Pattern Matching](05-pattern-matching.md) -- Patterns, connectives, destructuring, type guards
6. [Collections](06-collections.md) -- Lists, tuples, sets, maps with all methods
7. [PathMaps and Zippers](07-pathmaps-and-zippers.md) -- Hierarchical trie structures and navigation
8. [Numeric Types](15-numeric-types.md) -- Float, BigInt, BigRat, FixedPoint

### Concurrency and State
9. [Channels and Concurrency](08-channels-and-concurrency.md) -- Send, receive, persistent, peek, join, parallel
10. [Contracts and State](09-contracts-and-state.md) -- Contract patterns, mutable cells, CRUD

### Platform
11. [System Channels](10-system-channels.md) -- All `rho:*` URIs: I/O, crypto, registry, vaults, AI
12. [Registry](11-registry.md) -- Registry operations, TreeHashMap, URI generation
13. [Vaults and Tokens](12-vaults-and-tokens.md) -- SystemVault, MakeMint, transfers, auth keys
14. [Cost Model](13-cost-model.md) -- Phlogiston accounting, cost functions, gas scaling

### Patterns and Practice
15. [Design Patterns](14-design-patterns.md) -- Security, capabilities, common idioms
16. [Rhox Macros](16-rhox-macros.md) -- `.rhox` template format
17. [Testing with RhoSpec](17-testing-with-rhospec.md) -- Test framework, writing contract tests
18. [Deployment Workflow](18-deployment-workflow.md) -- Deploy, propose, finalize, exploratory deploy
19. [Real-World Applications](19-real-world-applications.md) -- Embers template system, production patterns

### Internal Reference
- [Crate Overview](crate-overview.md) -- Interpreter crate architecture, compilation pipeline, system processes
- [Language Analysis](rholang-language-analysis.md) -- Analysis of 88 .rho files: patterns, complexity levels, compiler implications

### Legacy Tutorials (from RChain era -- core concepts still valid)
- [Rholang Tutorial](rholangtut.md) -- Foundational language tutorial
- [Pattern Matching Tutorial](rholangmatchingtut.md) -- Deep dive into matching semantics
- [Ollama Integration](ollama.md) -- Local LLM integration guide

## Examples

Working examples are in [`rholang/examples/`](../../rholang/examples/):

| File | Topic |
|------|-------|
| `tut-hello.rho` | Contracts, channels, ack pattern |
| `tut-lists-methods.rho` | List operations |
| `tut-maps-methods.rho` | Map operations |
| `tut-sets-methods.rho` | Set operations |
| `tut-strings-methods.rho` | String operations |
| `tut-pathmap-all-methods.rho` | PathMap operations |
| `tut-pathmap-zippers.rho` | Zipper navigation (18 examples) |
| `tut-hash-functions.rho` | SHA256, Blake2b, Keccak256 |
| `tut-registry.rho` | Registry insert/lookup |
| `tut-philosophers.rho` | Dining philosophers (concurrency) |
| `numeric-types.rho` | BigInt, BigRat, Float, FixedPoint |
| `user-crud-example.rho` | Full CRUD with mutable state |
| `vault_demo/` | Vault address, balance, transfer |

## System Contracts

Production contracts are in [`casper/src/main/resources/`](../../casper/src/main/resources/):

| File | Purpose |
|------|---------|
| `Registry.rho` | Registry + TreeHashMap implementation |
| `SystemVault.rho` | Fund management and transfers |
| `MakeMint.rho` | Token/currency factory |
| `ListOps.rho` | Functional list operations library |
| `Either.rho` | Either monad (success/failure) |
| `Stack.rho` | LIFO stack data structure |
| `NonNegativeNumber.rho` | Constrained non-negative integers |
| `AuthKey.rho` | Authentication key management |
| `MultiSigSystemVault.rho` | Multi-signature vault |
| `PoS.rhox` | Proof-of-Stake validator contract (macro) |

## Test Suites

Sophisticated contract tests in [`casper/src/test/resources/`](../../casper/src/test/resources/) using the RhoSpec framework. See [Testing with RhoSpec](17-testing-with-rhospec.md).
