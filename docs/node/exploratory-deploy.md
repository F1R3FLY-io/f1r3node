# Exploratory Deploy

Exploratory deploy executes Rholang code in a read-only context against a specific block's post-state. No block is created, no phlo is consumed. Available only on readonly nodes.

Results are returned as `RhoExpr` values — see [Rholang Type System (RhoExpr)](README.md#rholang-type-system-rhoexpr) for the complete type mapping. All Rholang types are supported including extended numerics (BigInt, BigRat, FixedPoint), operators, and method calls.

## Return Channel Convention

The runtime reads results from the **first unforgeable name** created by the deploy's RNG (`GPrivate`), NOT from `rho:system:deployId` (`GDeployId`). This is by design in both Scala and Rust (see `RuntimeSyntax.scala:517-518`).

The return channel must be:
- The **first name** in the `new` binding list
- **Without** a URI binding (plain `new ret`, not `` new ret(`rho:system:deployId`) ``)

### Correct

```rholang
new ret, lookup(`rho:registry:lookup`), ch in {
  lookup!(`rho:id:...`, *ch) |
  for (val <- ch) { ret!(val) }
}
```

### Wrong (returns 0 pars)

```rholang
new ret(`rho:system:deployId`), lookup(`rho:registry:lookup`), ch in {
  lookup!(`rho:id:...`, *ch) |
  for (val <- ch) { ret!(val) }
}
```

## Reserved Keywords

Rholang's tree-sitter grammar reserves several keywords that cannot be used as variable names. The most common pitfall is `contract`:

### Wrong (parse error, silently returns empty before error propagation fix)

```rholang
new ret, lookup(`rho:registry:lookup`), ch in {
  lookup!(`rho:id:...`, *ch) |
  for (contract <- ch) {        // ERROR: 'contract' is a reserved keyword
    contract!("method", *ret)
  }
}
```

### Correct

```rholang
new ret, lookup(`rho:registry:lookup`), ch in {
  lookup!(`rho:id:...`, *ch) |
  for (c <- ch) {
    c!("method", *ret)
  }
}
```

Reserved keywords in the Rholang grammar include: `contract`, `new`, `in`, `for`, `match`, `if`, `else`, `bundle`, `select`, `Nil`, `true`, `false`, `not`, `and`, `or`. See `rholang-rs/rholang-tree-sitter/grammar.js` for the full list.

## Error Propagation

`play_exploratory_deploy` now propagates errors to the caller. Previously, all errors (including parse errors) were silently swallowed and empty results were returned. This made it impossible to distinguish "no data" from "invalid Rholang" at the client level.

The gRPC `exploratoryDeploy` endpoint returns errors in the `ExploratoryDeployResponse.Error` message field, which pyf1r3fly surfaces as `F1r3flyClientException`.

## Block Hash Parameter

When calling exploratory deploy, always pass an explicit block hash (typically the LFB hash) to ensure you're querying the expected state. Passing an empty string may resolve to a state that doesn't include recent deploys.

```python
lfb = node.last_finalized_block().blockInfo
result = node.exploratory_deploy(rholang_code, lfb.blockHash)
```

## Implementation

- `casper/src/rust/rholang/runtime.rs:858` -- `play_exploratory_deploy`
- `casper/src/rust/api/block_api.rs:1405` -- `exploratory_deploy` API handler
- `casper/src/rust/rholang/runtime.rs:1035` -- `capture_results_with_errors` (reset, evaluate, read)
