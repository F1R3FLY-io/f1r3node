# LogicT Execution Runtime Architecture (Phase 0 Baseline)

## Purpose

This document captures the architecture baseline for implementing split execution in F1r3node.

## High-level model

Post-match execution is split into bounded, resumable steps:

1. Consume-match succeeds.
2. Runtime instantiates a process with substitutions.
3. Runtime executes a bounded step (`reduce_step` in later phases).
4. Immediate effects are committed.
5. Remaining execution state is persisted as continuation state.

## Separation of concerns

- Coordination state:
  - channels / consumes / produces / matching
  - remains in existing coordination substrate
- Execution state:
  - resumable machine control state
  - continuation stack / environment / gas metadata
  - persisted in a dedicated continuation store

## Determinism constraints

- Canonical serialization for persisted state.
- Stable ordering for any consensus-visible traversal.
- No wall-clock or host-environment dependence in execution outcomes.
- Stepwise replay must produce the same result as full reduction for covered fragments.

## Initial feature boundaries

Phase 0-1 scope excludes:

- full scheduler design
- production MPC/HSM-like off-node orchestration concerns
- broad protocol migration

Initial milestone focuses on deterministic extraction of explicit execution state and bounded stepping interfaces.

## Key artifacts

- Plan: `IMPLEMENTATION_PLAN.md`
- Detailed project plan source: `logicT_execution_runtime-IMPLEMENTATION_PLAN.md`
- Verification entrypoint: `scripts/verify-runtime.sh`
- Agent constraints: `AGENTS.md`
