# Gas Accounting Update Workflow

Use this skill when adding new step boundaries, suspension triggers, or continuation execution paths.

## Objective

Ensure each execution unit has explicit, bounded, and test-covered gas accounting.

## Checklist

1. Enumerate new gas-consuming operations.
2. Define per-step gas debit boundaries.
3. Add explicit out-of-gas suspension/error behavior.
4. Verify no hidden unmetered loops remain.
5. Add tests for near-boundary and exact-boundary behavior.
6. Verify producer/executor billing split if continuation funding is involved.

## Required verification

```bash
cargo test -p casper
cargo test -p rholang
```

## Completion report format

- Gas model changes
- Metering invariants
- Boundary test coverage
