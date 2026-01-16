# TLA+ Specifications: Introduction

This document introduces the TLA+ specifications for Reified RSpaces and explains how they complement the Rocq proofs.

## What is TLA+?

TLA+ (Temporal Logic of Actions) is a formal specification language developed by Leslie Lamport. Unlike Rocq, which proves properties about functions and data structures, TLA+ excels at specifying and verifying **concurrent and distributed systems**.

Key characteristics:
- **State machine model**: Systems are described as initial states + transition relations
- **Temporal logic**: Properties can express "eventually", "always", "leads to"
- **Model checking**: TLC model checker exhaustively explores finite state spaces
- **Abstraction**: Focus on essential behavior, abstracting implementation details

## TLA+ vs Rocq

| Aspect | Rocq | TLA+ |
|--------|------|------|
| Verification | Theorem proving | Model checking |
| Approach | Prove for all inputs | Check finite cases |
| Strength | Infinite domains, induction | Concurrency, state machines |
| Properties | Functional correctness | Temporal properties |
| Trade-off | More effort, stronger results | Less effort, bounded scope |

## How They Complement Each Other

For Reified RSpaces:

- **Rocq proves**: Individual operation correctness (produce, consume, etc.), invariant preservation, determinism of sequential operations
- **TLA+ verifies**: System-level coordination, concurrent behavior, liveness properties, state machine transitions

Example:
- Rocq proves `produce_maintains_full_invariant` (invariant preserved after produce)
- TLA+ verifies that the overall system `[]NoPendingMatch` (always safe)

## Specifications Overview

### GenericRSpace.tla

Core tuple space operations:
- **Variables**: `dataStore`, `contStore`, `joins`, `nameCounter`, `qualifier`
- **Actions**: `Produce`, `Consume`, `Gensym`, `Clear`
- **Invariants**: `TypeInvariant`, `NoPendingMatch`
- **Properties**: `Safety`, `MatchEventuallyFires`

### CheckpointReplay.tla

Checkpoint and replay mechanisms:
- **Variables**: `spaceStates`, `checkpoints`, `operationLog`, `replayMode`, `softCheckpoint`
- **Actions**: `Produce`, `Consume`, `CreateCheckpoint`, `EnterReplayMode`, `ReplayNextOperation`
- **Invariants**: `ReplayModeExclusive`, `ReplayIndexValid`
- **Properties**: `ReplayEventuallyCompletes`, `SoftCheckpointResolved`

### SpaceCoordination.tla

Multi-space coordination:
- **Variables**: `spaces`, `channelOwnership`, `useBlockStacks`, `defaultSpace`
- **Actions**: `RegisterSpace`, `RegisterChannel`, `PushUseBlock`, `PopUseBlock`
- **Invariants**: `NoOrphanedChannels`, `UseBlocksValid`, `DefaultSpaceValid`
- **Helper Functions**: `EffectiveDefaultSpace`, `ResolveSpace`, `ValidJoinPattern`

## Reading TLA+ Specifications

### Basic Syntax

```tla
\* Comment
EXTENDS Naturals, Sequences

CONSTANTS Channel, Data, NULL

VARIABLES dataStore, contStore

\* Definition
MyDef == expression

\* Action (state transition)
MyAction ==
    /\ precondition
    /\ dataStore' = new_value        \* ' means "next state"
    /\ UNCHANGED contStore
```

### Temporal Operators

| Operator | Meaning |
|----------|---------|
| `[]P` | Always P (in all states) |
| `<>P` | Eventually P (in some future state) |
| `P ~> Q` | P leads to Q (whenever P, eventually Q) |
| `WF_v(A)` | Weak fairness: if A is continuously enabled, it eventually happens |

### Common Patterns

**Invariant**:
```tla
Safety == []NoPendingMatch
```

**Liveness**:
```tla
EventuallyCompletes == started ~> completed
```

**Type Invariant**:
```tla
TypeOK ==
    /\ var1 \in SomeType
    /\ var2 \in AnotherType
```

## Running TLC Model Checker

TLC exhaustively checks properties against finite models:

```bash
# Run TLC on a specification
java -jar tla2tools.jar -config GenericRSpace.cfg GenericRSpace.tla

# With larger state space
java -Xmx4G -jar tla2tools.jar -workers 4 -config CheckpointReplay.cfg CheckpointReplay.tla
```

### Configuration Files

Each `.tla` file has a `.cfg` configuration:

```tla
\* GenericRSpace.cfg
CONSTANTS
    Channels = {c1, c2}
    Patterns = {p1, p2}
    Data = {d1, d2}
    Continuations = {k1}
    NULL = NULL

SPECIFICATION Spec

INVARIANTS
    TypeInvariant
    NoPendingMatch

PROPERTIES
    Safety
```

## Axioms and Assumptions

The TLA+ specifications make certain abstractions:

1. **Finite domains**: Constants like `Channels`, `Data` are finite sets
2. **Simplified matching**: `Matches(pattern, data) == TRUE` (wildcard)
3. **Abstract hashing**: Merkle roots are counters, not real hashes
4. **Deterministic CHOOSE**: TLC picks a fixed element for `CHOOSE`

These abstractions are appropriate because:
- TLC can only check finite state spaces
- The Rocq proofs cover the infinite/general cases
- The focus is on protocol correctness, not implementation details

## Relationship to Rust Implementation

| TLA+ Concept | Rust Implementation |
|--------------|---------------------|
| `dataStore` | `GenericRSpace::data_store` |
| `contStore` | `GenericRSpace::cont_store` |
| `Produce(ch, d, p)` | `GenericRSpace::produce()` |
| `Consume(chs, ps, k, p)` | `GenericRSpace::consume()` |
| `replayMode` | `ReplayRSpace` vs normal mode |
| `softCheckpoint` | `ISpace::create_soft_checkpoint()` |
| `useBlockStacks` | `Registry::use_block_stack` |

## Next Steps

Continue to:
- [01-generic-rspace-spec.md](01-generic-rspace-spec.md) - Core tuple space operations
- [02-checkpoint-replay-spec.md](02-checkpoint-replay-spec.md) - Checkpoint and replay
- [03-space-coordination-spec.md](03-space-coordination-spec.md) - Multi-space coordination
