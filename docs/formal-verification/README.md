# Formal Verification of Reified RSpaces

This directory contains comprehensive documentation for the formal verification of **Reified RSpaces**, a parameterized tuple space implementation for the Rholang interpreter. The proofs establish correctness guarantees for space operations, checkpoint/replay determinism, gas accounting, and multi-space coordination.

## Overview

Reified RSpaces provides a flexible, configurable tuple space where:
- **Inner collections** determine data ordering (Bag, Queue, Stack, Set, Cell, PriorityQueue, VectorDB)
- **Outer storage** determines channel indexing (HashMap, PathMap, FixedArray, CyclicArray, Vector, HashSet)
- **Qualifiers** control persistence and concurrency (Default, Temp, Seq)

The formal verification ensures that regardless of configuration, the system maintains critical safety properties.

## Verification Summary

| Category | Theorems | Admitted | Status |
|----------|----------|----------|--------|
| **Rocq (Coq) Proofs** | 173+ | 0 | Complete |
| **TLA+ Specifications** | 3 specs | N/A | Complete |

### Rocq/Coq Proofs

All theorems are **fully proven** with zero admitted axioms (except the justified cryptographic assumptions for BLAKE2b256 hash collision resistance).

### TLA+ Model Checking

Three TLA+ specifications provide model-checkable safety and liveness properties.

## Document Structure

### Rocq Documentation

| File | Topic | Key Theorems |
|------|-------|--------------|
| [00-introduction.md](rocq/00-introduction.md) | Overview of Rocq proofs | Proof techniques, axioms |
| [01-foundations.md](rocq/01-foundations.md) | Prelude.v - Core types | `space_id_eq_dec`, `mobile_not_seq` |
| [02-pattern-matching.md](rocq/02-pattern-matching.md) | Match.v - Match semantics | `match_decidable`, `wildcard_always_matches` |
| [03-generic-rspace.md](rocq/03-generic-rspace.md) | GenericRSpace.v - Core invariants | `produce_maintains_full_invariant`, `consume_multi_atomic` |
| [04-checkpoint-replay.md](rocq/04-checkpoint-replay.md) | Checkpoint.v - Replay determinism | `replay_determinism`, `soft_checkpoint_preserves_state` |
| [05-phlogiston.md](rocq/05-phlogiston.md) | Phlogiston.v - Gas accounting | `charge_succeeds_with_sufficient_balance`, `charges_preserve_non_negative` |
| [06-substitution.md](rocq/06-substitution.md) | Substitution.v - Variable substitution | `subst_par_handles_all_constructors` |
| [07-outer-storage.md](rocq/07-outer-storage.md) | OuterStorage.v - Storage implementations | `gensym_unique_fixed_array`, `vector_get_put` |
| [08-space-factory.md](rocq/08-space-factory.md) | SpaceFactory.v - Factory pattern | `factory_creates_empty`, `seq_hashset_invalid` |

### TLA+ Documentation

| File | Topic | Key Properties |
|------|-------|----------------|
| [00-introduction.md](tla/00-introduction.md) | Overview of TLA+ specifications | Model checking approach |
| [01-generic-rspace-spec.md](tla/01-generic-rspace-spec.md) | GenericRSpace.tla | `NoPendingMatch`, `Safety` |
| [02-checkpoint-replay-spec.md](tla/02-checkpoint-replay-spec.md) | CheckpointReplay.tla | `ReplayDeterminism`, `ReplayEventuallyCompletes` |
| [03-space-coordination-spec.md](tla/03-space-coordination-spec.md) | SpaceCoordination.tla | `NoOrphanedChannels`, `UseBlocksValid` |

## Core Safety Properties

### 1. No Pending Match Invariant

The fundamental safety property: **if data exists at a channel, no waiting continuation on that channel has a matching pattern**.

This ensures that the tuple space never "misses" a match - either data fires a continuation immediately, or it waits.

```
no_pending_match: forall channel data,
  In data (data_store channel) ->
  forall continuation,
    In continuation (cont_store channel) ->
    ~ matches (pattern continuation) data
```

### 2. Deterministic Replay

Checkpoint and replay is deterministic: replaying the same operation log from the same checkpoint always produces identical final state.

```
replay_determinism: forall checkpoint ops final1 final2,
  apply_operations (cp_state checkpoint) ops = Some final1 ->
  apply_operations (cp_state checkpoint) ops = Some final2 ->
  final1 = final2
```

### 3. Gas Exhaustion Safety

Gas (phlogiston) accounting ensures operations cannot proceed without sufficient resources, and successful charges maintain non-negative balance.

```
charge_preserves_non_negative: forall cm cost cm',
  (0 <= cost_value cost) ->
  charge cm cost = ChargeOk cm' ->
  (0 <= cm_available cm')
```

### 4. Gensym Uniqueness

Fresh channel generation produces unique identifiers (for non-cyclic storage types).

```
gensym_unique: forall arr val1 val2 arr1 arr2 idx1 idx2,
  put arr val1 = Some (arr1, idx1) ->
  put arr1 val2 = Some (arr2, idx2) ->
  idx1 <> idx2
```

## Axioms and Assumptions

The proofs rely on two justified cryptographic axioms (see [04-checkpoint-replay.md](rocq/04-checkpoint-replay.md)):

1. **`compute_hash_valid`**: BLAKE2b256 always produces exactly 32 bytes. This is a specification property of the algorithm (RFC 7693).

2. **`hash_collision_resistant`**: Different states produce different hashes. This models the cryptographic collision resistance of BLAKE2b256 (128-bit security against birthday attacks).

## Reading the Documentation

Each documentation file follows a consistent structure:

1. **Overview**: What the module specifies and why it matters
2. **Key Concepts**: Definitions and data structures
3. **Theorem Hierarchy**: Main theorems with their supporting lemmas
4. **Proof Techniques**: How theorems are proven
5. **Examples**: Concrete illustrations of the properties
6. **Correspondence to Rust**: How specifications relate to implementation

### Suggested Reading Order

1. Start with [00-introduction.md](rocq/00-introduction.md) for Rocq proof background
2. Read [01-foundations.md](rocq/01-foundations.md) for core type definitions
3. Study [03-generic-rspace.md](rocq/03-generic-rspace.md) for the main invariants
4. Explore other files based on interest

## File Locations

The formal specifications are located in:
- **Rocq proofs**: `formal/rocq/reified_rspaces/theories/`
- **TLA+ specs**: `formal/tla/`

## Building the Proofs

```bash
cd formal/rocq/reified_rspaces
make clean
make
```

Use resource limiting to prevent system unresponsiveness:

```bash
systemd-run --user --scope -p MemoryMax=126G -p CPUQuota=1800% -p IOWeight=30 make -j1
```

## Related Documentation

- [Reifying RSpaces Specification](../../rholang/examples/README.md) - High-level design document
- Rust implementation: `rholang/src/rust/interpreter/spaces/`
