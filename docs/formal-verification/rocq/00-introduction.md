# Introduction to Rocq Proofs for Reified RSpaces

This document introduces the Rocq (formerly Coq) formal verification of Reified RSpaces, explaining the proof methodology, techniques used, and how to read the specifications.

## What is Rocq/Coq?

Rocq (the new name for Coq as of 2024) is an interactive theorem prover based on the Calculus of Inductive Constructions. It allows us to:

1. **Define precise specifications** using mathematical logic
2. **Prove theorems** about those specifications
3. **Extract verified code** that is guaranteed to satisfy the specifications

The proofs in this development ensure that the Reified RSpaces implementation maintains critical safety invariants under all circumstances.

## Why Formal Verification?

Tuple spaces are notoriously difficult to get right. Subtle bugs can lead to:

- **Missed matches**: Data and continuations both wait when they should fire
- **Double consumption**: Data consumed multiple times
- **Lost data**: Data disappears without triggering a continuation
- **Non-deterministic replay**: State diverges during replay

Formal verification eliminates these bugs by construction. If the proofs check, the invariants hold.

## Proof Modules Overview

The verification is organized into these modules:

| Module | Lines | Theorems | Description |
|--------|-------|----------|-------------|
| **Prelude.v** | ~210 | 4 | Core types and qualifiers |
| **Match.v** | ~315 | 9 | Pattern matching semantics |
| **GenericRSpace.v** | ~1200 | 40+ | Core space operations |
| **Checkpoint.v** | ~675 | 25+ | Checkpoint and replay |
| **Phlogiston.v** | ~500 | 25+ | Gas/cost accounting |
| **OuterStorage.v** | ~1140 | 49 | Storage implementations |
| **SpaceFactory.v** | ~390 | 12 | Factory pattern |
| **Substitution.v** | ~375 | 6 | Variable substitution |

## Proof Techniques

### 1. Structural Induction

Most proofs proceed by induction on the structure of data:

```coq
Theorem fixed_array_set_valid :
  forall arr idx val,
    idx < fixed_array_len A arr ->
    exists arr', fixed_array_set A arr idx val = Some arr'.
Proof.
  intros arr idx val Hlt.
  generalize dependent idx.
  induction arr as [| h t IH].
  - intros idx Hlt. simpl in Hlt. lia.
  - intros idx Hlt.
    destruct idx.
    + simpl. eexists. reflexivity.
    + simpl. specialize (IH idx). ...
Qed.
```

### 2. Case Analysis

Many proofs split on enumeration values or option types:

```coq
Theorem mobile_not_seq : forall q,
  is_mobile q = true <-> q <> QSeq.
Proof.
  intros q. split.
  - intros H. destruct q; simpl in H.
    + intro contra. discriminate contra.
    + intro contra. discriminate contra.
    + discriminate H.
  - intros H. destruct q; simpl; try reflexivity.
    exfalso. apply H. reflexivity.
Qed.
```

### 3. Decidability

Properties are made decidable to enable computational proofs:

```coq
Lemma space_id_eq_dec : forall (s1 s2 : SpaceId), {s1 = s2} + {s1 <> s2}.
Proof.
  decide equality.
  apply Z.eq_dec.
Defined.
```

### 4. Unfolding Definitions

Many proofs proceed by unfolding definitions and using simplification:

```coq
Theorem charge_decreases_available :
  forall cm cost cm',
    charge cm cost = ChargeOk cm' ->
    (cm_available cm' = cm_available cm - cost_value cost)%Z.
Proof.
  intros cm cost cm' Hcharge.
  unfold charge in Hcharge.
  destruct (cm_available cm <? 0)%Z eqn:Hneg; [discriminate |].
  destruct (cm_available cm <? cost_value cost)%Z eqn:Hlt; [discriminate |].
  injection Hcharge as Hcm'.
  subst cm'. simpl. reflexivity.
Qed.
```

### 5. Omega/Lia Tactics

Linear arithmetic is handled by the `lia` tactic:

```coq
Theorem charge_strictly_decreases :
  forall cm cost cm',
    (0 < cost_value cost)%Z ->
    charge cm cost = ChargeOk cm' ->
    (cm_available cm' < cm_available cm)%Z.
Proof.
  intros cm cost cm' Hpos Hcharge.
  apply charge_decreases_available in Hcharge.
  lia.
Qed.
```

## Reading Rocq Specifications

### Basic Syntax

```coq
(* Comments are in (* ... *) *)

(** Documentation comments start with (**  *)

(* Definitions *)
Definition SpaceId := list Z.

(* Inductive types (ADTs) *)
Inductive ChannelQualifier :=
  | QDefault
  | QTemp
  | QSeq.

(* Records *)
Record SpaceConfig := mkSpaceConfig {
  config_qualifier : ChannelQualifier;
  config_inner : InnerCollectionType;
  config_outer : OuterStorageType;
}.

(* Theorems *)
Theorem name : forall x y, Property x y.
Proof.
  (* proof tactics *)
Qed.
```

### Quantifiers and Connectives

| Symbol | Meaning |
|--------|---------|
| `forall x, P x` | For all x, P holds |
| `exists x, P x` | There exists x such that P holds |
| `P /\ Q` | P and Q |
| `P \/ Q` | P or Q |
| `P -> Q` | P implies Q |
| `~P` | Not P |
| `P <-> Q` | P if and only if Q |

### Common Types

| Type | Description |
|------|-------------|
| `nat` | Natural numbers (0, 1, 2, ...) |
| `Z` | Integers (..., -1, 0, 1, ...) |
| `bool` | Boolean (true, false) |
| `list A` | List of elements of type A |
| `option A` | Optional value (Some x or None) |
| `A * B` | Pair of A and B |

### Notation

The proofs use standard mathematical notation:

```coq
Notation "⊥" := None.           (* Bottom/None *)
Notation "⟨ x ⟩" := (Some x).   (* Some x *)
```

## Axioms

The proofs rely on exactly **two axioms**, both justified by cryptographic properties of BLAKE2b256:

### Axiom 1: Hash Validity

```coq
Axiom compute_hash_valid : forall A K (state : GenericRSpaceState A K),
  valid_hash (compute_hash A K state).
```

**Justification**: BLAKE2b256 is specified (RFC 7693) to always produce exactly 32 bytes (256 bits) of output. This is a deterministic property of the algorithm, not a security assumption.

### Axiom 2: Collision Resistance

```coq
Axiom hash_collision_resistant : forall A K (s1 s2 : GenericRSpaceState A K),
  compute_hash A K s1 = compute_hash A K s2 ->
  s1 = s2.
```

**Justification**: This models the cryptographic collision resistance of BLAKE2b256. Finding two distinct inputs with the same hash output requires approximately 2^128 hash evaluations (birthday bound), which is computationally infeasible.

Note: We model collision resistance as exact equality rather than computational infeasibility because Rocq cannot express computational complexity. In practice, this means we assume no collisions occur during system operation.

## Theorem Naming Conventions

Theorems follow consistent naming patterns:

| Pattern | Meaning |
|---------|---------|
| `X_maintains_Y` | Operation X preserves invariant Y |
| `X_preserves_Y` | Operation X preserves property Y |
| `X_Y_correct` | Operation X satisfies specification Y |
| `X_eq_dec` | Equality on X is decidable |
| `X_empty_Y` | Empty X satisfies Y |

## Proof Status

All 173+ theorems are **fully proven** with zero `Admitted` statements. The proofs can be verified by running:

```bash
cd formal/rocq/reified_rspaces
make clean && make
```

## Next Steps

Continue reading:

1. [01-foundations.md](01-foundations.md) - Core type definitions from Prelude.v
2. [03-generic-rspace.md](03-generic-rspace.md) - The main invariant proofs
