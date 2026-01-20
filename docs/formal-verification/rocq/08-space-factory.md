# Space Factory: SpaceFactory.v

This document covers `SpaceFactory.v`, which specifies the factory pattern for creating properly configured RSpace instances.

## Overview

The SpaceFactory pattern ensures that created spaces:
1. Have valid configurations (qualifier/inner/outer combinations)
2. Satisfy all invariants from construction
3. Start in an empty state (no pending matches)

The specification captures the validation rules and proves that factory-created spaces are always valid.

## Key Concepts

### Factory Configuration

#### Inner Collection Types

```coq
Inductive FactoryInnerType :=
  | FI_Bag            (* Unordered multiset *)
  | FI_Queue          (* FIFO ordering *)
  | FI_Stack          (* LIFO ordering *)
  | FI_Set            (* Unique elements *)
  | FI_Cell           (* At most one element *)
  | FI_PriorityQueue  (* Priority levels *)
  | FI_VectorDB.      (* Similarity matching *)
```

#### Outer Storage Types

```coq
Inductive FactoryOuterType :=
  | FO_HashMap        (* O(1) lookup *)
  | FO_PathMap        (* Hierarchical with prefix matching *)
  | FO_FixedArray     (* Fixed size array *)
  | FO_CyclicArray    (* Wraparound array *)
  | FO_Vector         (* Growable array *)
  | FO_HashSet.       (* Presence-only storage *)
```

#### Factory Config Record

```coq
Record FactoryConfig := mkFactoryConfig {
  fc_qualifier : SpaceQualifier;    (* Default=0, Temp=1, Seq=2 *)
  fc_inner_type : FactoryInnerType;
  fc_outer_type : FactoryOuterType;
  fc_max_size : option nat;         (* For fixed/cyclic arrays *)
}.
```

### Configuration Validation

#### Seq Compatibility

Seq spaces cannot use HashSet (concurrency issues):

```coq
Definition seq_compatible_outer (outer : FactoryOuterType) : bool :=
  match outer with
  | FO_HashSet => false
  | _ => true
  end.
```

#### Array Size Validation

Fixed/Cyclic arrays require max_size:

```coq
Definition array_size_valid (outer : FactoryOuterType) (max_size : option nat) : bool :=
  match outer with
  | FO_FixedArray | FO_CyclicArray =>
    match max_size with
    | Some n => (0 <? n)%nat
    | None => false
    end
  | _ => true
  end.
```

#### VectorDB Compatibility

VectorDB requires specific outer types:

```coq
Definition vectordb_compatible_outer (inner : FactoryInnerType) (outer : FactoryOuterType) : bool :=
  match inner with
  | FI_VectorDB =>
    match outer with
    | FO_HashMap | FO_Vector => true
    | _ => false
    end
  | _ => true
  end.
```

#### Complete Validation

```coq
Definition valid_config (cfg : FactoryConfig) : bool :=
  let qual := fc_qualifier cfg in
  let inner := fc_inner_type cfg in
  let outer := fc_outer_type cfg in
  let max_size := fc_max_size cfg in
  (* Seq qualifier restrictions *)
  (if Nat.eqb qual FC_QUALIFIER_SEQ then seq_compatible_outer outer else true) &&
  (* Array size requirements *)
  array_size_valid outer max_size &&
  (* VectorDB compatibility *)
  vectordb_compatible_outer inner outer.
```

### SpaceFactory Trait

```coq
Class SpaceFactory := {
  sf_create : FactoryConfig -> option (GenericRSpaceState A K);
  sf_validate : FactoryConfig -> bool;

  (* Specification properties *)
  sf_validate_correct : forall cfg,
    sf_validate cfg = valid_config cfg;
  sf_create_valid : forall cfg,
    sf_validate cfg = true -> exists state, sf_create cfg = Some state;
  sf_create_invalid : forall cfg,
    sf_validate cfg = false -> sf_create cfg = None;
  sf_respects_qualifier : forall cfg state,
    sf_create cfg = Some state -> gr_qualifier state = fc_qualifier cfg;
  sf_produces_empty : forall cfg state,
    sf_create cfg = Some state ->
    gr_data_store state = empty_data_store /\
    gr_cont_store state = empty_cont_store /\
    gr_joins state = empty_joins;
}.
```

## Theorem Hierarchy

```
FACTORY CORRECTNESS
├── factory_respects_qualifier
│   └── sf_respects_qualifier (trait axiom)
├── factory_produces_valid_state
│   └── sf_produces_empty (trait axiom)
├── factory_creates_empty
│   ├── factory_produces_valid_state
│   └── empty_state_no_pending_match
└── empty_state_no_pending_match
    └── (vacuous - no data)

DEFAULT FACTORY INSTANCE
├── default_factory_create
├── default_create_valid
├── default_create_invalid
├── default_respects_qualifier
└── default_produces_empty

CONFIGURATION VALIDATION
├── default_config_valid
├── temp_config_valid
├── seq_hashset_invalid
├── fixed_array_no_size_invalid
├── fixed_array_with_size_valid
├── vectordb_hashmap_valid
└── vectordb_pathmap_invalid

FACTORY COMPOSITION
└── create_many_all_valid
    └── empty_state_no_pending_match
```

## Detailed Theorems

### Theorem: factory_respects_qualifier

**Statement**: Created state has correct qualifier.

```coq
Theorem factory_respects_qualifier :
  forall cfg state,
    sf_create cfg = Some state ->
    gr_qualifier state = fc_qualifier cfg.
```

**Intuition**: If you ask for a Temp space, you get a Temp space - the factory doesn't silently change qualifiers.

---

### Theorem: factory_produces_valid_state

**Statement**: Created state has empty stores.

```coq
Theorem factory_produces_valid_state :
  forall cfg state,
    sf_validate cfg = true ->
    sf_create cfg = Some state ->
    gr_data_store state = empty_data_store /\
    gr_cont_store state = empty_cont_store /\
    gr_joins state = empty_joins.
```

**Intuition**: Factory-created spaces start empty. No data, no continuations, no join relationships.

---

### Theorem: empty_state_no_pending_match

**Statement**: Empty state trivially satisfies no_pending_match.

```coq
Theorem empty_state_no_pending_match :
  forall qualifier,
    no_pending_match A K M (empty_gr_state qualifier).
```

**Intuition**: With no data, the invariant "no data coexists with matching continuations" is vacuously true.

**Proof Technique**: Unfold `no_pending_match`, show the antecedent (data exists) is false.

---

### Theorem: factory_creates_empty

**Statement**: Factory creates empty state satisfying invariant.

```coq
Theorem factory_creates_empty :
  forall cfg state,
    sf_validate cfg = true ->
    sf_create cfg = Some state ->
    no_pending_match A K M state.
```

**Intuition**: Combines `factory_produces_valid_state` with `empty_state_no_pending_match`. Every factory-created space satisfies the core safety invariant.

---

### Configuration Validation Theorems

#### Theorem: default_config_valid

**Statement**: Default configuration is valid.

```coq
Theorem default_config_valid :
  valid_config (mkFactoryConfig FC_QUALIFIER_DEFAULT FI_Bag FO_HashMap None) = true.
```

---

#### Theorem: temp_config_valid

**Statement**: Temp configuration is valid.

```coq
Theorem temp_config_valid :
  valid_config (mkFactoryConfig FC_QUALIFIER_TEMP FI_Bag FO_HashMap None) = true.
```

---

#### Theorem: seq_hashset_invalid

**Statement**: Seq with HashSet is invalid.

```coq
Theorem seq_hashset_invalid :
  valid_config (mkFactoryConfig FC_QUALIFIER_SEQ FI_Bag FO_HashSet None) = false.
```

**Intuition**: Seq channels require sequential access, but HashSet is designed for concurrent operations.

---

#### Theorem: fixed_array_no_size_invalid

**Statement**: Fixed array without size is invalid.

```coq
Theorem fixed_array_no_size_invalid :
  valid_config (mkFactoryConfig FC_QUALIFIER_DEFAULT FI_Bag FO_FixedArray None) = false.
```

**Intuition**: Fixed arrays need a capacity to be allocated.

---

#### Theorem: fixed_array_with_size_valid

**Statement**: Fixed array with size is valid.

```coq
Theorem fixed_array_with_size_valid :
  valid_config (mkFactoryConfig FC_QUALIFIER_DEFAULT FI_Bag FO_FixedArray (Some 100)) = true.
```

---

#### Theorem: vectordb_hashmap_valid

**Statement**: VectorDB with HashMap is valid.

```coq
Theorem vectordb_hashmap_valid :
  valid_config (mkFactoryConfig FC_QUALIFIER_DEFAULT FI_VectorDB FO_HashMap None) = true.
```

---

#### Theorem: vectordb_pathmap_invalid

**Statement**: VectorDB with PathMap is invalid.

```coq
Theorem vectordb_pathmap_invalid :
  valid_config (mkFactoryConfig FC_QUALIFIER_DEFAULT FI_VectorDB FO_PathMap None) = false.
```

**Intuition**: VectorDB uses similarity-based lookup that doesn't work with PathMap's prefix-based organization.

---

### Theorem: create_many_all_valid

**Statement**: All created spaces satisfy invariant.

```coq
Theorem create_many_all_valid :
  forall cfg n state,
    valid_config cfg = true ->
    In state (create_many (default_factory_create A K) cfg n) ->
    no_pending_match A K M state.
```

**Intuition**: Creating multiple spaces with the same configuration produces multiple valid spaces.

## Examples

### Example: Valid Configurations

| Qualifier | Inner | Outer | Size | Valid? | Reason |
|-----------|-------|-------|------|--------|--------|
| Default | Bag | HashMap | None | Yes | Standard config |
| Temp | Queue | HashMap | None | Yes | Temp + FIFO |
| Seq | Bag | Vector | None | Yes | Seq + growable |
| Seq | Bag | HashSet | None | **No** | Seq incompatible with HashSet |
| Default | Bag | FixedArray | None | **No** | FixedArray needs size |
| Default | Bag | FixedArray | Some 100 | Yes | FixedArray with size |
| Default | VectorDB | HashMap | None | Yes | VectorDB + HashMap |
| Default | VectorDB | PathMap | None | **No** | VectorDB incompatible with PathMap |

### Example: Factory Usage

```coq
(* Create a standard space *)
let cfg = mkFactoryConfig FC_QUALIFIER_DEFAULT FI_Bag FO_HashMap None
in match default_factory_create cfg with
   | None => (* Should not happen - config is valid *)
   | Some state =>
       (* state has:
          - gr_qualifier = 0 (Default)
          - gr_data_store = empty
          - gr_cont_store = empty
          - gr_joins = empty
          - satisfies no_pending_match *)
   end
```

### Example: Why Seq + HashSet is Invalid

```
Seq qualifier means:
  - Non-persistent
  - Non-concurrent (sequential access only)
  - Non-mobile (can't be sent to other processes)

HashSet implementation typically uses:
  - Concurrent hash tables
  - Lock-free operations
  - Atomic updates

Conflict: Seq semantics require sequential access, but HashSet is designed
for concurrent access. Using HashSet would violate Seq's sequential guarantee.
```

## Default Factory Implementation

The specification includes a concrete default factory:

```coq
Definition default_factory_create (cfg : FactoryConfig) : option (GenericRSpaceState A K) :=
  if valid_config cfg
  then Some (empty_gr_state (fc_qualifier cfg))
  else None.

Instance DefaultSpaceFactory : SpaceFactory A K := {
  sf_create := default_factory_create;
  sf_validate := valid_config;
  sf_validate_correct := valid_config_correct;
  sf_create_valid := default_create_valid;
  sf_create_invalid := default_create_invalid;
  sf_respects_qualifier := default_respects_qualifier;
  sf_produces_empty := default_produces_empty;
}.
```

## Correspondence to Rust

| Rocq Definition | Rust Implementation |
|-----------------|---------------------|
| `FactoryConfig` | `SpaceConfig` struct |
| `FactoryInnerType` | `InnerCollectionType` enum |
| `FactoryOuterType` | `OuterStorageType` enum |
| `valid_config` | `SpaceConfig::validate()` |
| `SpaceFactory` trait | `SpaceFactory` trait |
| `default_factory_create` | `DefaultSpaceFactory::create()` |
| `FC_QUALIFIER_DEFAULT` | `SpaceQualifier::Default` |
| `FC_QUALIFIER_TEMP` | `SpaceQualifier::Temp` |
| `FC_QUALIFIER_SEQ` | `SpaceQualifier::Seq` |

## Design Notes

### Why Validation is Separate

Validation (`sf_validate`) is separate from creation (`sf_create`) because:
1. Validation is cheap and can be done early
2. Allows better error messages (know which constraint failed)
3. Enables compile-time configuration checking

### Factory as Invariant Bootstrap

The factory is the "invariant bootstrap" - it ensures that spaces satisfy `no_pending_match` from the moment they're created. Combined with operation preservation theorems (`produce_maintains_full_invariant`, etc.), this establishes that valid spaces remain valid.

### Compatibility Matrix Design

The compatibility rules encode real constraints:
- **Seq + HashSet**: Implementation conflict
- **FixedArray without size**: Incomplete configuration
- **VectorDB + PathMap**: Semantic mismatch (similarity vs. prefix)

These aren't arbitrary - each invalid combination represents a real problem that would cause runtime issues.

## Next Steps

This completes the core Rocq specification documentation. Continue to the TLA+ documentation starting with [../tla/00-introduction.md](../tla/00-introduction.md) for model checking specifications.
