# PeTTa (SWI-Prolog + MeTTa) Examples

This directory contains Rholang examples demonstrating the `rho:petta:execute`
system contract, which provides access to the MeTTA language.

## What is PeTTa?

**PeTTa** is an interpreter for MeTTa written in SWI-Prolog.

This integration allows Rholang contracts to perform advanced symbolic reasoning,
pattern matching, and AI-style computations.

## Prerequisites

See the prerequisites section in the DEVELOPER.md file located in the project's
folder.

## MeTTa entry point

Standard MeTTa programs do not require an entry point. Any definitions and queries
are added and executed (respectively) in the order of the program.

However, programs provided to the `rho::petta::execute` contract MUST contain
an entry point called `main`. This entry point MUST be defined as a function
with a single parameter (which serves no purpose and should be ignored).

## Examples Overview

### 01-swap.rho - Pattern Matching Basics

Demonstrates basic MeTTa pattern matching by defining a `swap` function that reverses a pair.

### 02-fib-long.rho - Recursive Computation

Computes a large Fibonacci number (fib(1000000)) using tail recursion,
which on consumer hardware should exceed the timeout of 10 seconds currently
defined for all MeTTa computations.


**MeTTa code:**
```metta
(= (fib-tr $n $a $b) (if (== $n 0) $a (fib-tr (- $n 1) $b (+ $a $b))))
(= (fib $n) (fib-tr $n 0 1))
!(fib 1000000)
```

### Return Value Structure

PeTTa always returns results wrapped in a `{"results": [...]}` JSON object, which is converted to a Rholang map:

```rholang
{
  "results": [result1, result2, ...]  // Rholang list
}
```

## Common Patterns

### Pattern 1: Simple Computation

```rholang
new executePetta(`rho:petta:execute`), retCh in {
  executePetta!("!(+ 1 2)", *retCh) |
  for(@result <- retCh) {
    // result = {"results": [3]}
    match result {
      {"results": [answer]} => {
        stdout!(answer)  // Prints: 3
      }
    }
  }
}
```

### Pattern 2: Multiple Calls

```rholang
new executePetta(`rho:petta:execute`), ret1, ret2 in {
  executePetta!("!(+ 1 2)", *ret1) |
  executePetta!("!(* 3 4)", *ret2) |
  
  for(@r1 <- ret1; @r2 <- ret2) {
    stdout!([r1, r2])
  }
}
```

## Troubleshooting

### Error: "Can't find PeTTa"

**Cause:** `$PETTA_PATH` points to invalid location

**Solution:**
```bash
# Check PeTTa location
ls ./PeTTa/src/metta.pl

# Or set explicitly
export PETTA_PATH=/full/path/to/PeTTa
```

### Error: "swipl: command not found"

**Cause:** SWI-Prolog not installed or not in PATH

**Solution:**
```bash
# Install SWI-Prolog
brew install swi-prolog  # macOS
apt-get install swi-prolog  # Ubuntu/Debian

# Verify
which swipl
```

### Timeout Errors

**Cause:** Computation exceeded 10-second limit

**Solutions:**
1. Break computation into smaller steps
2. Use more efficient algorithms
3. Pre-compute complex results off-chain

### No Output

**Cause:** Error occurred (doesn't send on ack channel)

**Solution:** Check node logs for error details:
```bash
tail -f ~/.rnode/rnode.log
```

## Security Notes

⚠️ **Important Security Considerations: this feature is EXPERIMENTAL.**

1. **Untrusted Code:** Never execute untrusted MeTTa code - there is currently
   no sandboxing at language level
2. **Timeouts:** All execution limited to 10 seconds to prevent DoS
3. **Non-Deterministic:** Results are cached for replay (consensus safety)
4. **Resource Limits:** System memory limits apply
