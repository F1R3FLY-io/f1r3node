# Testing with RhoSpec

RhoSpec is the Rholang test framework used in the casper test suite. It provides structured test suites with setup/teardown, assertions, and result collection.

## Test File Structure

A test file follows this pattern:

```rho
new
  rl(`rho:registry:lookup`),
  setup,
  test_case_1,
  test_case_2
in {
  // Look up RhoSpec from the registry
  new RhoSpecCh in {
    rl!(`rho:id:zphjgsfy13h1k85isc8rtwtgt3t9zzt5pjd5ihykfmyapfc4wt3x5h`, *RhoSpecCh) |
    for (@(_, RhoSpec) <- RhoSpecCh) {

      // Define the test suite
      @RhoSpec!("testSuite", *setup,
        [
          ("Test case description 1", *test_case_1),
          ("Test case description 2", *test_case_2)
        ])
    }
  } |

  // Setup: runs before each test
  contract setup(_, retCh) = {
    // Initialize test fixtures
    retCh!(fixtureData)
  } |

  // Test cases
  contract test_case_1(rhoSpec, @fixtures, ackCh) = {
    // Test logic using rhoSpec assertions
    @rhoSpec!("assert", (expected, "==", actual), "description", *ackCh)
  } |

  contract test_case_2(rhoSpec, @fixtures, ackCh) = {
    @rhoSpec!("assert", (true, "==", true), "always passes", *ackCh)
  }
}
```

## Test Suite Variants

### Without Setup

```rho
@RhoSpec!("testSuite", [
  ("test name", *test_fn)
])
```

### With Setup

```rho
@RhoSpec!("testSuite", *setup, [
  ("test name", *test_fn)
])
```

Setup contract signature: `contract setup(_, retCh) = { retCh!(fixtures) }`

### With Setup and Teardown

```rho
@RhoSpec!("testSuite", *setup, *teardown, [
  ("test name", *test_fn)
])
```

Teardown contract signature: `contract teardown(_, @fixtures, ackCh) = { ackCh!(Nil) }`

## Assertions

Assertions are called on the `rhoSpec` argument passed to each test:

### Equality

```rho
@rhoSpec!("assert", (expected, "==", actual), "description", *ackCh)
```

### Inequality

```rho
@rhoSpec!("assert", (unexpected, "!=", actual), "description", *ackCh)
```

### Boolean

```rho
@rhoSpec!("assert", true, "should be true", *ackCh)
@rhoSpec!("assert", false, "expected failure", *ackCh)
```

### Channel Assertions

Assert that a value appears on a channel:

```rho
// Assert expected value arrives on channel
@rhoSpec!("assert", (expectedValue, "== <-", resultCh), "description", *ackCh)

// Assert unexpected value does NOT arrive
@rhoSpec!("assert", (unexpectedValue, "!= <-", resultCh), "description", *ackCh)
```

## Test Case Signature

```rho
contract testName(rhoSpec, @fixtures, ackCh) = {
  // rhoSpec: the assertion API
  // fixtures: data from setup
  // ackCh: signal test completion
}
```

The `ackCh` must be signaled when the test completes. If not signaled, the test hangs.

## Real Example: MakeMint Tests

From `casper/src/test/resources/MakeMintTest.rho`:

```rho
contract setup(_, retCh) = {
  new MakeMintCh, mintACh, mintBCh in {
    rl!(`rho:system:makeMint`, *MakeMintCh) |
    for (@(_, MakeMint) <- MakeMintCh) {
      @MakeMint!(*mintACh) |
      @MakeMint!(*mintBCh) |
      for (mintA <- mintACh; mintB <- mintBCh) {
        retCh!((*mintA, *mintB))
      }
    }
  }
} |

contract test_create_purse(rhoSpec, @(mintA, mintB), ackCh) = {
  new alicePurse, bobPurse, aliceBal, bobBal in {
    @mintA!("makePurse", 100, *alicePurse) |
    @mintB!("makePurse", 50, *bobPurse) |
    for (alice <- alicePurse; bob <- bobPurse) {
      @alice!("getBalance", *aliceBal) |
      @bob!("getBalance", *bobBal) |
      @rhoSpec!("assert", (100, "== <-", *aliceBal), "Alice should have 100", *ackCh) |
      @rhoSpec!("assert", (50, "== <-", *bobBal), "Bob should have 50", *ackCh)
    }
  }
} |

contract test_cross_currency_deposit(rhoSpec, @(mintA, mintB), ackCh) = {
  new purseA, purseB, depositResult in {
    @mintA!("makePurse", 100, *purseA) |
    @mintB!("makePurse", 50, *purseB) |
    for (pA <- purseA; pB <- purseB) {
      @pA!("deposit", 10, *pB, *depositResult) |
      @rhoSpec!("assert", ("Cross-currency deposit not allowed", "== <-", *depositResult),
                "Cross-currency deposits should fail", *ackCh)
    }
  }
}
```

## Running Tests

Tests are executed as part of the casper test suite:

```bash
cargo test -p casper --test system_contract_tests
```

Or run specific test files by matching test names.

## Test Files Reference

| File | Tests | What It Covers |
|------|-------|----------------|
| `SystemVaultTest.rho` | 11 | Vault creation, balance, transfers, auth errors |
| `MakeMintTest.rho` | 7 | Purse creation, cross-currency, deposits, splits, overflow |
| `TreeHashMapTest.rho` | ~10 | Init, set/get, contains, edge cases |
| `RegistryTest.rho` | ~5 | Insert, lookup, signed insert |
| `ListOpsTest.rho` | ~10 | Append, reverse, fold, filter |
| `EitherTest.rho` | ~8 | Either monad: map, flatMap, fold |
| `StackTest.rho` | ~5 | Push, pop, peek, isEmpty |
| `NonNegativeNumberTest.rho` | ~5 | Valid/invalid add/sub, bounds |
| `AuthKeyTest.rho` | ~3 | Key creation and verification |
| `MultiSigSystemVaultTest.rho` | ~8 | Multi-sig creation, threshold, transfers |
| `PoSTest.rho` | ~15 | Bonding, slashing, rewards, epochs |

## Writing New Tests

1. Create a `.rho` file in `casper/src/test/resources/`
2. Look up RhoSpec from the registry using its URI
3. Define `setup` if you need shared fixtures
4. Define test contracts following the `(rhoSpec, @fixtures, ackCh)` signature
5. Use `@rhoSpec!("assert", ...)` for assertions
6. Always signal `ackCh` on completion
7. Register the test suite with `@RhoSpec!("testSuite", ...)`
