# Checkpoint and Replay: Checkpoint.v

This document covers `Checkpoint.v`, which specifies the checkpoint/replay mechanism enabling deterministic state reconstruction.

## Overview

Checkpoints capture snapshots of tuple space state that can be:
1. Persisted to a history repository
2. Retrieved by hash
3. Used as starting points for deterministic replay

The key property is **replay determinism**: replaying the same operations from the same checkpoint always produces identical results.

## Key Concepts

### Hash Type

State is identified by BLAKE2b256 hashes:

```coq
(** BLAKE2b256 hash represented as 32 bytes *)
Definition Hash := list Z.

(** Hash length is always 32 bytes *)
Definition valid_hash (h : Hash) : Prop := length h = 32%nat.
```

### Checkpoint Record

A checkpoint captures complete state at a point in time:

```coq
Record Checkpoint := mkCheckpoint {
  cp_root : Hash;                      (* Merkle root hash *)
  cp_state : GenericRSpaceState A K;   (* The captured state *)
}.
```

### Operation Types

Operations that can be logged and replayed:

```coq
Inductive Operation (A K : Type) :=
  | Op_Produce : Channel -> A -> bool -> Operation A K
  | Op_Consume : list Channel -> list A -> K -> bool -> Operation A K
  | Op_Install : list Channel -> list A -> K -> Operation A K.
```

### Soft Checkpoints

Fast in-memory checkpoints for speculative execution:

```coq
Record SoftCheckpoint := mkSoftCheckpoint {
  sc_state : GenericRSpaceState A K;
  sc_timestamp : nat;
}.
```

Unlike regular checkpoints, soft checkpoints:
- Don't have a hash
- Aren't persisted to storage
- Are used for speculative execution and rollback

### Replay State

Tracks replay mode and progress:

```coq
Record ReplayState := mkReplayState {
  rs_is_replay : bool;                    (* Currently replaying? *)
  rs_event_log : list (Operation A K);    (* Operations to replay *)
  rs_log_position : nat;                  (* Current position *)
  rs_start_checkpoint : option (Checkpoint A K);  (* Start point *)
}.
```

## Axioms (Justified)

The proofs rely on two axioms about BLAKE2b256:

### Axiom 1: compute_hash_valid

```coq
Axiom compute_hash_valid : forall A K (state : GenericRSpaceState A K),
  valid_hash (compute_hash A K state).
```

**Justification**: BLAKE2b256 is specified (RFC 7693) to always produce exactly 32 bytes of output. This is a deterministic property of the algorithm, not a security assumption.

### Axiom 2: hash_collision_resistant

```coq
Axiom hash_collision_resistant : forall A K (s1 s2 : GenericRSpaceState A K),
  compute_hash A K s1 = compute_hash A K s2 ->
  s1 = s2.
```

**Justification**: This models collision resistance. Finding two distinct inputs with the same hash requires approximately 2^128 hash evaluations (birthday bound), which is computationally infeasible. We model this as exact equality because Rocq cannot express computational complexity.

## Theorem Hierarchy

```
REPLAY DETERMINISM (TOP-LEVEL)
├── replay_determinism              ★ Main theorem
│   └── (basic functional equality)
├── replay_from_log_determinism
│   └── replay_determinism
├── chain_replay_determinism
│   └── hash_identifies_state
│       └── hash_collision_resistant (axiom)
└── replay_from_rig_deterministic

CHECKPOINT CORRECTNESS
├── checkpoint_restore_identity
├── hash_identifies_state
│   └── hash_collision_resistant (axiom)
├── stored_checkpoint_retrievable
│   └── hr_retrieve_stored
└── contains_reflects_stored
    └── hr_contains_iff

SOFT CHECKPOINTS
├── soft_checkpoint_preserves_state      ★ Main theorem
├── soft_checkpoint_preserves_data_store
│   └── soft_checkpoint_preserves_state
├── soft_checkpoint_preserves_cont_store
│   └── soft_checkpoint_preserves_state
├── soft_checkpoint_preserves_joins
│   └── soft_checkpoint_preserves_state
├── soft_checkpoint_preserves_qualifier
│   └── soft_checkpoint_preserves_state
├── soft_checkpoints_independent
└── soft_checkpoint_timestamp_preserved

RIG OPERATIONS
├── rig_enables_replay
├── rig_and_reset_enables_replay
├── rig_and_reset_sets_checkpoint
├── rig_and_reset_sets_log
├── rig_and_reset_starts_at_zero
├── check_replay_consistent
├── rig_sets_log
├── rig_preserves_checkpoint
├── advance_increases_position
└── replay_terminates

TEMP CLEARING
├── clear_temp_preserves_qualifier
└── clear_temp_non_temp_identity
```

## Detailed Theorems

### Theorem: replay_determinism (MAIN)

**Statement**: Replaying the same operations from the same checkpoint always produces identical results.

```coq
Theorem replay_determinism :
  forall (cp : Checkpoint A K) (ops : list (Operation A K))
         (final1 final2 : GenericRSpaceState A K),
    apply_operations (cp_state cp) ops = Some final1 ->
    apply_operations (cp_state cp) ops = Some final2 ->
    final1 = final2.
```

**Intuition**: This is the fundamental determinism guarantee. It means:
- State can be reconstructed from checkpoints
- Replay produces consistent results across nodes
- The system is reproducible

**Proof Technique**: Straightforward - both applications start from the same state and apply the same operations, so by functional equality they produce the same result.

**Example**:
```
Checkpoint: { state with data_store = {ch -> [(a, true)]} }
Operations: [Op_Produce ch b false, Op_Consume [ch] [p] k false]

Replay 1: state -> state' -> state''  = final1
Replay 2: state -> state' -> state''  = final2

final1 = final2 (guaranteed)
```

---

### Theorem: hash_identifies_state

**Statement**: Checkpoints with the same hash have the same state.

```coq
Theorem hash_identifies_state :
  forall (cp1 cp2 : Checkpoint A K),
    cp_root cp1 = cp_root cp2 ->
    compute_hash A K (cp_state cp1) = cp_root cp1 ->
    compute_hash A K (cp_state cp2) = cp_root cp2 ->
    cp_state cp1 = cp_state cp2.
```

**Intuition**: Hashes uniquely identify states (under collision resistance). If two checkpoints have the same hash, they must have the same underlying state.

**Proof Technique**: Uses `hash_collision_resistant` axiom.

---

### Theorem: soft_checkpoint_preserves_state (MAIN)

**Statement**: Creating and reverting a soft checkpoint yields the original state.

```coq
Theorem soft_checkpoint_preserves_state :
  forall (state : GenericRSpaceState A K) (timestamp : nat),
    revert_to_soft_checkpoint (create_soft_checkpoint state timestamp) = state.
```

**Intuition**: Soft checkpoints are lossless. The state captured is exactly the state restored.

**Example**:
```coq
(* Create soft checkpoint *)
let sc = create_soft_checkpoint current_state 42

(* Do some speculative operations... *)
let speculative_state = do_some_operations()

(* Revert if speculation fails *)
let restored = revert_to_soft_checkpoint sc

(* restored = current_state *)
```

---

### Theorem: soft_checkpoint_preserves_data_store

**Statement**: Data store is preserved through soft checkpoint round-trip.

```coq
Theorem soft_checkpoint_preserves_data_store :
  forall (state : GenericRSpaceState A K) (timestamp : nat),
    gr_data_store (revert_to_soft_checkpoint (create_soft_checkpoint state timestamp)) =
    gr_data_store state.
```

Similar theorems exist for:
- `soft_checkpoint_preserves_cont_store`
- `soft_checkpoint_preserves_joins`
- `soft_checkpoint_preserves_qualifier`

---

### Theorem: rig_and_reset_enables_replay

**Statement**: After rig_and_reset, we're in replay mode.

```coq
Theorem rig_and_reset_enables_replay :
  forall cp log,
    rs_is_replay (rig_and_reset cp log) = true.
```

**Intuition**: The rig_and_reset operation sets up the space for deterministic replay by:
1. Recording the start checkpoint
2. Loading the event log
3. Entering replay mode

---

### Theorem: check_replay_consistent

**Statement**: Replay state is valid after rig_and_reset.

```coq
Theorem check_replay_consistent :
  forall cp log,
    check_replay_data (rig_and_reset cp log) = true.
```

**Intuition**: The validity check passes when:
- We're in replay mode
- We have a start checkpoint
- Log position is within bounds (starts at 0)

---

### Theorem: replay_terminates

**Statement**: Replay eventually completes when position exceeds log length.

```coq
Theorem replay_terminates :
  forall rs,
    rs_log_position rs >= length (rs_event_log rs) ->
    replay_complete rs = true.
```

**Intuition**: Replay is finite - once we've processed all logged operations, replay is complete.

---

### Theorem: chain_replay_determinism

**Statement**: In a valid chain, checkpoints with the same hash have the same state.

```coq
Theorem chain_replay_determinism :
  forall (chain : list CheckpointLink),
    valid_chain chain ->
    forall link1 link2,
      In link1 chain -> In link2 chain ->
      cp_root (link_from link1) = cp_root (link_from link2) ->
      compute_hash A K (cp_state (link_from link1)) = cp_root (link_from link1) ->
      compute_hash A K (cp_state (link_from link2)) = cp_root (link_from link2) ->
      cp_state (link_from link1) = cp_state (link_from link2).
```

**Intuition**: Checkpoint chains form a verifiable history. Links in the chain satisfy:
- Each link transforms from-state to to-state via its operations
- Consecutive links connect (to of one = from of next)

## Examples

### Example: Basic Checkpoint/Restore Cycle

```
1. Start with state S
2. Create checkpoint: cp = { root: hash(S), state: S }
3. Modify state: S -> S' -> S''
4. Restore from checkpoint: current = cp_state = S
5. State is back to S
```

### Example: Soft Checkpoint for Speculation

```
1. Current state: S
2. Create soft checkpoint: sc = create_soft_checkpoint S 100
3. Speculatively execute: S -> S' -> S''
4. Speculation fails!
5. Revert: current = revert_to_soft_checkpoint sc = S
6. State restored to S without disk I/O
```

### Example: Replay from Checkpoint

```
1. Start checkpoint: cp with state S0
2. Event log: [Op_Produce ch a true, Op_Consume [ch] [p] k false]
3. Rig and reset: rs = rig_and_reset cp log
4. Replay step 1: S0 -> S1 (produce)
5. Replay step 2: S1 -> S2 (consume fires)
6. Final state: S2

Determinism guarantee: Any replay of this log from S0 produces S2
```

## History Repository Trait

The specification includes an abstract interface for checkpoint storage:

```coq
Class HistoryRepository (A K : Type) := {
  hr_store : Checkpoint A K -> Prop;
  hr_retrieve : Hash -> option (GenericRSpaceState A K);
  hr_contains : Hash -> bool;

  (* Specification properties *)
  hr_retrieve_stored : forall cp,
    hr_store cp -> hr_retrieve (cp_root cp) = Some (cp_state cp);
  hr_contains_iff : forall h,
    hr_contains h = true <-> exists state, hr_retrieve h = Some state;
}.
```

## Correspondence to Rust

| Rocq Definition | Rust Implementation |
|-----------------|---------------------|
| `Checkpoint` | `rspace::checkpoint::Checkpoint` |
| `Hash` | `Blake2b256Hash` |
| `compute_hash` | `Blake2b256Hash::hash()` |
| `SoftCheckpoint` | `rspace::checkpoint::SoftCheckpoint` |
| `create_soft_checkpoint` | `ISpace::create_soft_checkpoint()` |
| `revert_to_soft_checkpoint` | `ISpace::revert_to_soft_checkpoint()` |
| `ReplayState` | Replay mode tracking in `ReplayRSpace` |
| `rig_and_reset` | `ISpace::rig_and_reset()` |
| `Operation` | Events in `trace::Log` |
| `apply_operations` | Replay loop in `ReplayRSpace` |

## Next Steps

Continue to [05-phlogiston.md](05-phlogiston.md) for the gas accounting (phlogiston) proofs.
