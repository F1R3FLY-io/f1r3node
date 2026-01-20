# Foundations: Prelude.v

This document covers the foundational definitions in `Prelude.v`, which establishes the core types and properties used throughout the Reified RSpaces verification.

## Overview

The Prelude module defines:
- Space identifiers and their properties
- Channel qualifiers (Default, Temp, Seq)
- Inner collection types (Bag, Queue, Stack, etc.)
- Outer storage types (HashMap, PathMap, etc.)
- Space configuration records
- Use block stack operations

## Key Concepts

### Space Identifiers

A `SpaceId` uniquely identifies a space instance. It's represented as a 32-byte list (matching the BLAKE2b256 hash output size):

```coq
Definition SpaceId := list Z.

(* The default/root space has all zeros *)
Definition default_space_id : SpaceId := repeat 0%Z 32.
```

### Channel Qualifiers

Qualifiers control persistence and concurrency behavior:

```coq
Inductive ChannelQualifier :=
  | QDefault  (* Persistent, concurrent access *)
  | QTemp     (* Non-persistent, concurrent access *)
  | QSeq.     (* Non-persistent, sequential, restricted mobility *)
```

| Qualifier | Persistent | Concurrent | Mobile |
|-----------|------------|------------|--------|
| Default   | Yes | Yes | Yes |
| Temp      | No | Yes | Yes |
| Seq       | No | No | No |

### Inner Collection Types

These determine how data is stored and matched at a channel:

```coq
Inductive InnerCollectionType :=
  | ICBag           (* Multiset - any element can match *)
  | ICQueue         (* FIFO - only head matches *)
  | ICStack         (* LIFO - only top matches *)
  | ICSet           (* Unique elements, idempotent sends *)
  | ICCell          (* At most one element *)
  | ICPriorityQueue (priorities : nat)  (* Priority levels *)
  | ICVectorDB (dimensions : nat).      (* Similarity-based matching *)
```

### Outer Storage Types

These determine how channels are indexed:

```coq
Inductive OuterStorageType :=
  | OSHashMap                          (* O(1) lookup by hash *)
  | OSPathMap                          (* Hierarchical with prefix matching *)
  | OSArray (max_size : nat) (cyclic : bool)  (* Fixed size *)
  | OSVector                           (* Unbounded, growable *)
  | OSHashSet.                         (* Presence-only storage *)
```

### Space Configuration

A complete space configuration combines these choices:

```coq
Record SpaceConfig := mkSpaceConfig {
  config_qualifier : ChannelQualifier;
  config_inner : InnerCollectionType;
  config_outer : OuterStorageType;
}.
```

### Use Block Stack

Use blocks create a stack of default spaces per process:

```coq
Definition UseBlockStack := list SpaceId.

Definition push_use_block (stack : UseBlockStack) (s : SpaceId) : UseBlockStack :=
  s :: stack.

Definition pop_use_block (stack : UseBlockStack) : option (SpaceId * UseBlockStack) :=
  match stack with
  | [] => None
  | s :: rest => Some (s, rest)
  end.

Definition effective_default (stack : UseBlockStack) (global : option SpaceId)
  : option SpaceId :=
  match stack with
  | [] => global
  | s :: _ => Some s
  end.
```

## Theorem Hierarchy

```
CORE TYPE PROPERTIES
├── space_id_eq_dec          (* SpaceId equality is decidable *)
├── qualifier_eq_dec         (* Qualifier equality is decidable *)
├── default_space_unique     (* Characterization of default space *)
├── mobile_not_seq           (* Seq channels are not mobile *)
└── persistent_implies_mobile (* Persistent implies mobile *)
```

## Detailed Theorems

### Theorem: space_id_eq_dec

**Statement**: SpaceId equality is decidable.

```coq
Lemma space_id_eq_dec : forall (s1 s2 : SpaceId), {s1 = s2} + {s1 <> s2}.
```

**Intuition**: We need decidable equality to compute with space identifiers. Since a SpaceId is a list of integers (Z), and Z has decidable equality, the list equality is also decidable.

**Proof Technique**: Use `decide equality` tactic which generates decidability proofs for inductive types, then supply `Z.eq_dec` for the element type.

```coq
Proof.
  decide equality.
  apply Z.eq_dec.
Defined.
```

Note: We use `Defined` instead of `Qed` to make the proof term transparent, enabling computation with this decidability result.

---

### Theorem: default_space_unique

**Statement**: The default space ID is characterized by being all zeros with length 32.

```coq
Theorem default_space_unique : forall s : SpaceId,
  s = default_space_id <-> (forall n, nth n s 0%Z = 0%Z) /\ length s = 32.
```

**Intuition**: The default space is a distinguished space (like the root namespace). This theorem provides both:
1. **Forward**: If s equals the default, all elements are zero and length is 32
2. **Backward**: If all elements are zero and length is 32, s equals the default

**Proof Technique**: Split into forward and backward directions. Forward uses properties of `repeat`. Backward uses `nth_ext` to show lists are equal when they have equal length and equal elements at all positions.

**Example**:
```coq
(* The default space ID *)
default_space_id = [0; 0; 0; ...; 0]  (* 32 zeros *)

(* Any space with all zeros of length 32 equals default *)
s = [0; 0; 0; ...; 0] -> s = default_space_id
```

---

### Theorem: mobile_not_seq

**Statement**: A qualifier allows mobility if and only if it's not Seq.

```coq
Theorem mobile_not_seq : forall q,
  is_mobile q = true <-> q <> QSeq.
```

where:

```coq
Definition is_mobile (q : ChannelQualifier) : bool :=
  match q with
  | QSeq => false
  | _ => true
  end.
```

**Intuition**: The Seq qualifier restricts channel mobility - Seq channels cannot be sent to other processes. This theorem formalizes that restriction.

**Proof Technique**: Case analysis on the qualifier, with contradiction in impossible cases.

**Example**:
```coq
(* Default qualifier is mobile *)
is_mobile QDefault = true

(* Seq qualifier is NOT mobile *)
is_mobile QSeq = false

(* Temp qualifier is mobile *)
is_mobile QTemp = true
```

---

### Theorem: persistent_implies_mobile

**Statement**: If a qualifier is persistent, it's also mobile.

```coq
Theorem persistent_implies_mobile : forall q,
  is_persistent q = true -> is_mobile q = true.
```

where:

```coq
Definition is_persistent (q : ChannelQualifier) : bool :=
  match q with
  | QDefault => true
  | _ => false
  end.
```

**Intuition**: Only QDefault is persistent, and QDefault is mobile. So persistence implies mobility. The converse doesn't hold (QTemp is mobile but not persistent).

**Proof Technique**: Case analysis on qualifier. Only QDefault satisfies `is_persistent q = true`, and it also satisfies `is_mobile q = true`.

**Example**:
```coq
(* Default: persistent AND mobile *)
is_persistent QDefault = true  ->  is_mobile QDefault = true

(* Temp: NOT persistent but mobile *)
is_persistent QTemp = false    but is_mobile QTemp = true

(* Seq: NOT persistent and NOT mobile *)
is_persistent QSeq = false     and is_mobile QSeq = false
```

## Predefined Configurations

The module provides common configurations:

```coq
(* Default: HashMap + Bag + Default qualifier *)
Definition default_space_config : SpaceConfig :=
  mkSpaceConfig QDefault ICBag OSHashMap.

(* PathMap configuration (recommended default per spec) *)
Definition pathmap_space_config : SpaceConfig :=
  mkSpaceConfig QDefault ICBag OSPathMap.

(* Temp space configuration *)
Definition temp_space_config : SpaceConfig :=
  mkSpaceConfig QTemp ICBag OSHashMap.

(* Seq space configuration *)
Definition seq_space_config : SpaceConfig :=
  mkSpaceConfig QSeq ICBag OSHashSet.

(* Queue space configuration *)
Definition queue_space_config : SpaceConfig :=
  mkSpaceConfig QDefault ICQueue OSHashMap.

(* Stack space configuration *)
Definition stack_space_config : SpaceConfig :=
  mkSpaceConfig QDefault ICStack OSHashMap.
```

## Correspondence to Rust

| Rocq Definition | Rust Implementation |
|-----------------|---------------------|
| `SpaceId` | `types::SpaceId` |
| `ChannelQualifier` | `types::SpaceQualifier` |
| `InnerCollectionType` | `types::InnerCollectionType` |
| `OuterStorageType` | `types::OuterStorageType` |
| `SpaceConfig` | `types::SpaceConfig` |
| `UseBlockStack` | `Registry::use_block_stack` field |
| `push_use_block` | `Registry::push_use_block()` |
| `pop_use_block` | `Registry::pop_use_block()` |
| `effective_default` | `Registry::effective_default_space()` |

## Notation

The Prelude introduces convenient notation:

```coq
Notation "⊥" := None (at level 0).           (* None/Bottom *)
Notation "⟨ x ⟩" := (Some x) (at level 0).   (* Some x *)
```

This makes specifications more readable:
- `⊥` represents "no value" or "undefined"
- `⟨space_id⟩` represents "some space_id"

## Next Steps

Continue to [02-pattern-matching.md](02-pattern-matching.md) for the Match.v specification of pattern matching semantics.
