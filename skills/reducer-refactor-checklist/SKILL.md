# Reducer Refactor Checklist

Use this skill when extracting explicit reducer machine state without changing behavior.

## Objective

Safely refactor reducer internals to introduce explicit `ExecState`/step boundaries while preserving current semantics.

## Checklist

1. Identify exact before/after execution entrypoints.
2. Confirm all consensus-visible outputs are unchanged.
3. Keep old full execution path available during transition.
4. Ensure deterministic ordering in any new collections.
5. Add focused equivalence tests (old path vs new path).
6. Document any temporary compatibility shims.

## Required verification

```bash
cargo test -p rholang
cargo test -p casper
```

## Completion report format

- Changed files
- Behavioral equivalence scope
- Residual risks
- Follow-ups
