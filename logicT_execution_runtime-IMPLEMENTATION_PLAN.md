# LogicT-Based Execution Runtime for F1r3node — Codex Implementation Plan

## Document purpose

This plan defines how to implement the LogicT-based execution runtime in F1r3node using Codex as the primary coding agent. It is structured for incremental delivery, strong reviewability, and repeatable agent execution.

It assumes a Codex workflow that uses:
- repo-local instructions in `AGENTS.md`
- optional project-scoped `.codex/config.toml`
- small, reusable skills for recurring workflows
- Git checkpoints before and after each major task

## Codex operating model

Codex should be used in short, reviewable task slices rather than one monolithic prompt. Prefer:
1. analysis and design diffs first
2. one bounded implementation change at a time
3. tests in the same task
4. explicit verification commands
5. a short completion report listing changed files, risks, and follow-ups

Codex reads `AGENTS.md` before starting work, supports project-scoped `.codex/config.toml`, and supports reusable skills across the CLI, IDE extension, and Codex app. The official quickstart also recommends Git checkpoints around tasks. :contentReference[oaicite:0]{index=0}

## Design guidelines for Codex

### 1. Preserve deterministic execution
Any runtime change must preserve deterministic replay across validators. Codex must not introduce:
- wall-clock dependencies
- nondeterministic iteration over unordered maps
- nondeterministic serialization
- hidden environment-dependent behavior

### 2. Prefer explicit machine state over implicit control flow
Do not implement suspension by capturing native runtime stacks. Represent resumable execution explicitly as serializable abstract machine state.

### 3. Bound every execution step
Every continuation step must have a measurable upper gas bound and a clear suspension boundary.

### 4. Separate coordination state from execution state
Keep channel / receive / consume-match state in the existing coordination substrate. Put resumable continuation state in a dedicated continuation store abstraction.

### 5. Optimize for reviewability
Every Codex task should target a narrow subsystem, produce a small diff, and include tests. Avoid wide refactors unless they are purely preparatory and mechanically safe.

### 6. Keep the first milestone narrow
The first split-executable fragment is only:
- successful consume-match
- process instantiation with substitutions
- bounded post-match reducer step
- persistence of suspended remainder

Do not attempt whole-runtime suspension in the first implementation.

## Required repo scaffolding for Codex

### A. Add `AGENTS.md`
Create a repo-local `AGENTS.md` that tells Codex:
- the runtime invariants
- deterministic coding rules
- accepted module boundaries
- test commands
- forbidden shortcuts
- expected output format for each task

### B. Add project Codex config
Add `.codex/config.toml` with project defaults for:
- preferred model
- approval behavior
- sandbox preferences
- task verbosity
- optional project-specific overrides

The official docs state that Codex CLI and IDE extensions share the same config layers, including project-level `.codex/config.toml`, and that a default model can be configured there. :contentReference[oaicite:1]{index=1}

### C. Add reusable skills
Create skills for recurring tasks such as:
- reducer refactor checklist
- deterministic serialization review
- gas accounting update workflow
- continuation-store implementation workflow
- test authoring checklist
- acceptance verification checklist

Official Codex skills package instructions, resources, and optional scripts so workflows are repeatable across Codex surfaces. :contentReference[oaicite:2]{index=2}

## Delivery phases

## Phase 0 — Codex project setup

### Objective
Prepare the repository so Codex can work predictably and repeatedly.

### Tasks
- Add `AGENTS.md`
- Add `.codex/config.toml`
- Add a `skills/` directory with initial task skills
- Add `docs/runtime/` and place the architecture spec there
- Add a `scripts/verify-runtime.sh` script that runs formatting, linting, unit tests, and deterministic serialization tests
- Add `IMPLEMENTATION_PLAN.md`

### Acceptance criteria
- Codex can open the repo and infer project rules from `AGENTS.md`
- A single verification command exists and passes on main
- At least 3 reusable skills exist for implementation and verification workflows
- A new engineer can run Codex in this repo without ad hoc instructions

## Phase 1 — Abstract machine extraction

### Objective
Refactor the reducer so post-match execution state can be represented explicitly.

### Scope
Extract an internal machine state structure for the post-match instantiated process:
- control node
- environment / substitution bindings
- continuation stack
- effect buffer
- gas metadata

### Tasks
- Identify the current reducer path from consume-match to process execution
- Introduce an explicit `ExecState` or equivalent internal structure
- Refactor reducer code so execution can proceed from `ExecState`
- Keep behavior identical to the current runtime
- Add golden tests proving equivalence before and after refactor

### Design constraints
- No persistence yet
- No public API changes yet
- No scheduler yet
- No continuation funding yet

### Acceptance criteria
- There is an internal explicit execution state type for post-match execution
- Existing behavior is preserved for non-suspended execution paths
- Golden tests or trace-equivalence tests pass
- No consensus-visible behavior changes occur

## Phase 2 — Step-based reducer entrypoint

### Objective
Introduce a bounded reducer entrypoint for a single execution step.

### Scope
Add:
- `reduce_step(exec_state, gas_limit) -> StepResult`

`StepResult` should minimally support:
- emitted effects
- next suspended state
- consumed gas
- terminal status

### Tasks
- Define `StepResult`
- Implement suspension boundaries
- Define mandatory suspension triggers:
  - gas threshold
  - first externally visible effect batch
  - blocking / wait condition
  - branch expansion threshold
- Keep full reduction path available for compatibility testing

### Acceptance criteria
- `reduce_step` exists and is unit-tested
- A process can make bounded progress and suspend
- Gas accounting is explicit per step
- At least one post-match process can be stepped multiple times to completion
- Full execution and repeated step execution produce the same final state for covered test cases

## Phase 3 — Continuation persistence layer

### Objective
Persist suspended execution state as first-class continuation objects.

### Scope
Implement a dedicated continuation store, separate from coordination storage.

### Tasks
- Define `ContinuationHandle`
- Implement create / load / update / complete / expire lifecycle
- Store:
  - canonical ID
  - origin reference
  - serialized state
  - gas bound per step
  - funding policy
  - status
  - version
  - state root
- Add serialization and deserialization tests
- Add storage accounting

### Design decision
Use:
- canonical ID = origin + nonce
- state root = separate integrity field

### Acceptance criteria
- Suspended continuations can be persisted and reloaded deterministically
- Continuation identity is stable across versions
- State root changes across state transitions as expected
- Serialization round-trip tests pass
- Storage costs are metered

## Phase 4 — Integration with consume / produce matching

### Objective
Wire split execution into the real runtime path.

### Scope
After a successful consume-match:
- instantiate the process
- execute one bounded step
- commit immediate effects
- persist any remainder as a continuation

### Tasks
- Add split execution into the consume-match path behind a feature flag
- Ensure producer pays only for the initial bounded split
- Persist suspended work to the continuation store
- Add trace logging and metrics

### Acceptance criteria
- A matched continuation no longer requires full execution in one path
- Producer gas remains bounded by the configured initial split rules
- Immediate effects and suspended remainder are both committed correctly
- Feature-flagged integration tests pass

## Phase 5 — Continuation execution transaction path

### Objective
Allow explicit later execution of suspended continuations.

### Scope
Add a transaction or deploy path for:
- `execute_continuation(id, gas_limit)`

### Tasks
- Implement load-step-commit lifecycle
- Enforce gas bound per step
- Update continuation version/state root after each step
- Mark completion or failure terminally
- Add replay and re-entrancy protections as needed

### Acceptance criteria
- A suspended continuation can be resumed by ID
- Repeated calls can drive a continuation to completion
- Partial progress is persisted safely
- Terminal continuations cannot be resumed
- Invalid IDs and stale versions are rejected cleanly

## Phase 6 — Funding policy and market hooks

### Objective
Add the first consumer-funded execution model.

### Scope
Support:
- producer-funded initial split only
- public executor pays for continuation steps
- optional bounty metadata, without full marketplace complexity

### Tasks
- Implement `FundingPolicy`
- Add validation rules for executor-paid steps
- Add optional bounty attachment fields
- Add queue visibility controls
- Meter storage rent or TTL

### Acceptance criteria
- Producer and executor costs are separated
- Public executors can advance eligible continuations
- Continuations without funding support can expire or remain private
- Storage rent / TTL enforcement works
- Spam-resistant economics are test-covered

## Phase 7 — Bridge continuation subtype

### Objective
Add bridge continuations as a first-class subtype inside the shared runtime.

### Scope
Support bridge-specific metadata and scheduling hooks:
- route / lane ID
- source and target domain
- message nonce
- finality phase
- deadline
- reward policy

### Tasks
- Define a bridge continuation subtype
- Add subtype-aware scheduler hooks
- Add deadline-priority and reward-priority strategies
- Add bridge-specific observability
- Add protocol-controlled rescue / escalation policy if needed

### Acceptance criteria
- Bridge continuations use the shared continuation runtime
- Bridge metadata is available to scheduler and reward logic
- Bridge steps can be prioritized without creating a separate execution engine
- Bridge-specific tests cover deadlines, retries, and completion paths

## Phase 8 — Operational hardening

### Objective
Make the system safe for sustained use.

### Tasks
- Add continuation expiry sweeps
- Add continuation size caps
- Add branch fan-out caps
- Add metrics:
  - continuations created
  - continuations resumed
  - average steps to completion
  - expired continuations
  - gas per step
- Add property tests for determinism and idempotent replay
- Add crash-recovery tests around continuation persistence

### Acceptance criteria
- Continuation spam is economically bounded
- Metrics are emitted for operations and debugging
- Recovery tests demonstrate safe restart behavior
- Deterministic replay tests pass under repeated runs

## Codex tasking guidelines

## Prompt structure for every Codex task

Use this structure:

1. **Objective**
2. **Files likely involved**
3. **Constraints**
4. **Implementation boundaries**
5. **Tests required**
6. **Completion report format**

### Example template

#### Objective
Implement `StepResult` and a minimal `reduce_step` path for post-match instantiated execution state.

#### Files likely involved
- reducer module
- runtime types
- gas accounting module
- unit tests

#### Constraints
- deterministic only
- no public API breakage
- no persistence yet
- keep full reduction path intact

#### Tests required
- one-step progress test
- multi-step to completion test
- equivalence vs full reduction for covered cases

#### Completion report
- changed files
- behavior summary
- risks
- follow-ups

This style aligns with official Codex guidance to use clear project instructions, reusable workflows, and repo context rather than restating everything from scratch each time. :contentReference[oaicite:3]{index=3}

## Pull request guidelines

Every PR generated with Codex should:
- target a single phase or sub-phase
- include tests
- include a short architecture note if it changes invariants
- avoid mixing refactor, feature, and cleanup unless tightly coupled
- include a deterministic replay checklist

### Required PR checklist
- [ ] deterministic behavior preserved
- [ ] serialization format reviewed
- [ ] gas accounting updated
- [ ] tests added or updated
- [ ] feature flag added if rollout is partial
- [ ] metrics added if operational behavior changed
- [ ] docs updated

## Verification matrix

### Unit tests
- reducer stepping
- continuation serialization
- ID/version/state-root behavior
- gas metering
- expiry and rent logic

### Integration tests
- consume-match creates continuation
- continuation resumes to completion
- producer cost remains bounded
- executor-paid continuation path works
- bridge subtype path works

### Property tests
- repeated replay yields same outputs
- stepwise execution equals full execution for covered fragments
- suspended state round-trip is stable

### Failure tests
- invalid continuation ID
- stale version update
- gas exhaustion near split boundary
- oversized continuation
- expired continuation resume attempt

## Rollout plan

### Stage 1
Feature-flagged internal reducer stepping only.

### Stage 2
Continuation persistence enabled in non-critical paths.

### Stage 3
Public executor path enabled for selected continuation classes.

### Stage 4
Bridge continuation subtype enabled on controlled routes.

### Stage 5
Default runtime path migrated to split execution for the approved fragment set.

## Final acceptance criteria for the overall initiative

The initiative is complete when all of the following are true:

1. A successful consume-match can create a split-executable post-match process.
2. That process can be stepped deterministically under explicit gas limits.
3. Suspended state persists and reloads correctly from a dedicated continuation store.
4. Producer cost is bounded to the initial split policy.
5. Executor-funded continuation progress is supported.
6. Continuation spam is bounded through fees, rent or TTL, and size/fan-out constraints.
7. Bridge continuations run as a typed subtype within the same runtime.
8. Stepwise execution is test-demonstrated to match full execution for covered fragments.
9. Operational metrics and recovery behavior are in place.
10. The repository contains Codex-native instructions, config, and reusable skills so future tasks remain repeatable.

## Suggested initial Codex work queue

1. Add `AGENTS.md`, `.codex/config.toml`, and 3 starter skills
2. Extract explicit post-match `ExecState`
3. Add `StepResult` and `reduce_step`
4. Add equivalence tests vs full reduction
5. Add continuation store types and serialization
6. Integrate split execution behind a feature flag
7. Add `execute_continuation`
8. Add funding policy and TTL/rent
9. Add bridge continuation subtype
10. Add metrics, hardening, and rollout docs