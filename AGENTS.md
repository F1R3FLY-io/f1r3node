# LogicT Runtime Agent Instructions

This file defines repository-local execution rules for Codex tasks related to the LogicT-based execution runtime.

## Mission

Implement split execution for post-match continuation processing with strict determinism, bounded gas, and replay safety.

## Runtime invariants

- Deterministic replay across validators is mandatory.
- No wall-clock dependent logic inside consensus execution paths.
- No nondeterministic iteration in consensus-visible behavior.
- No environment-dependent serialization or hashing.
- Execution suspension must be explicit machine state, not native stack capture.
- Every execution step must have a bounded gas budget and explicit suspension boundary.
- Coordination state and continuation execution state must remain separate.

## Deterministic coding rules

- Prefer `BTreeMap` / sorted vectors when ordering affects behavior.
- If `HashMap` / `HashSet` is required, sort keys/items before consensus-visible use.
- Avoid random seeds, UUID generation, and system time in reducer/merge/replay paths.
- Keep serialization canonical and versioned when persisted.

## Module boundaries

- Reducer stepping logic belongs in runtime/reducer modules.
- Persistent suspended state belongs in a dedicated continuation store abstraction.
- Consume/produce matching state remains in the existing coordination substrate.
- Funding policy and scheduler policy must be explicit typed modules, not ad hoc flags.

## Forbidden shortcuts

- Do not bypass gas accounting to force progress.
- Do not store opaque closures / stack snapshots for suspension.
- Do not introduce hidden global mutable state for execution progress.
- Do not merge refactor + feature + cleanup in one large, mixed change unless mechanically required.

## Expected task output format

For every substantial task, provide:

1. Objective completed.
2. Files changed.
3. Behavior summary.
4. Risks.
5. Follow-ups.
6. Exact verification commands run.

## Verification commands

Primary:

```bash
./scripts/verify-runtime.sh
```

Targeted examples:

```bash
cargo test -p models --test models_tests sorted_par_hash_set_test
cargo test -p models --test models_tests sorted_par_map_test
```
