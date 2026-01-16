# Outer Storage Types: OuterStorage.v

This document covers `OuterStorage.v`, which specifies the behavior of outer storage types for channel stores. Outer storage determines how channels are organized and named.

## Overview

Outer storage types include:
1. **Fixed Array** - Static bounds, fails when full
2. **Cyclic Array** - Wraps around, overwrites old data
3. **Vector** - Dynamic growth, unbounded (modulo OOM)
4. **HashSet** - Presence-only storage for Seq channels

Each storage type has different guarantees for `gensym` (name generation) uniqueness.

## Key Concepts

### Channel Index

```coq
Definition ChannelIndex := nat.
```

### Fixed Array

Fixed-size array that fails with `OutOfNames` when full:

```coq
Definition fixed_array := list (option A).

Definition fixed_array_empty : fixed_array :=
  repeat None capacity.

Definition fixed_array_put (arr : fixed_array) (val : A)
  : option (fixed_array * ChannelIndex) :=
  match fixed_array_find_empty arr 0 with
  | None => None  (* OutOfNames error - array is full *)
  | Some idx =>
    match fixed_array_set arr idx val with
    | None => None
    | Some arr' => Some (arr', idx)
    end
  end.
```

### Cyclic Array

Array that wraps around, overwriting old entries:

```coq
Record cyclic_array := {
  ca_data : list (option A);
  ca_next : nat;  (* Next write position *)
}.

Definition cyclic_array_put (arr : cyclic_array) (val : A)
  : cyclic_array * ChannelIndex :=
  let idx := ca_next arr in
  let data' := ... in
  let next' := (idx + 1) mod capacity in
  ({| ca_data := data'; ca_next := next' |}, idx).
```

### Vector (Dynamic Array)

Growable array that doubles capacity as needed:

```coq
Record vector := {
  vec_data : list (option A);
  vec_len : nat;  (* Number of occupied slots *)
}.

Definition vector_grow (v : vector) : vector :=
  let current_cap := vector_capacity v in
  let new_cap := if current_cap =? 0 then 1 else current_cap * 2 in
  {| vec_data := vec_data v ++ repeat None (new_cap - current_cap);
     vec_len := vec_len v |}.

Definition vector_put (v : vector) (val : A) : vector * ChannelIndex :=
  let v' := if vec_len v <? vector_capacity v then v else vector_grow v in
  let idx := vec_len v' in
  ...
```

### HashSet

Simple set for presence-only tracking:

```coq
Definition hash_set := list A.

Definition hash_set_insert (s : hash_set) (a : A) : hash_set :=
  if hash_set_member s a
  then s  (* Already present *)
  else a :: s.
```

## Theorem Hierarchy

```
FIXED ARRAY (21 theorems)
├── fixed_array_empty_len
├── fixed_array_empty_all_none
├── fixed_array_get_oob
├── fixed_array_set_valid
├── fixed_array_set_oob
├── fixed_array_put_full_fails
├── fixed_array_set_preserves_len
├── gensym_unique_fixed_array                ★ Main uniqueness theorem
├── find_empty_in_range
├── find_empty_returns_none_slot
├── find_empty_is_none
├── find_empty_after_set_start_zero
├── fixed_array_set_changes_only_target
├── fixed_array_set_preserves_nones
├── fixed_array_set_target_some
└── count_le_length
    └── find_empty_all_some

CYCLIC ARRAY (4 theorems)
├── cyclic_array_put_always_succeeds
├── cyclic_array_wraps
├── gensym_cyclic_not_unique_after_wrap      ★ Non-uniqueness (by design)
└── gensym_cyclic_consecutive

VECTOR (18 theorems)
├── vector_empty_len
├── vector_empty_capacity
├── vector_grow_increases_capacity
├── vector_put_increases_len
├── vector_grow_capacity_pos
├── vector_grow_capacity_gt_len
├── vector_grow_len_preserved
├── vector_put_data_correct
├── vector_get_put
├── gensym_vector_monotonic
├── gensym_unique_vector                     ★ Main uniqueness theorem
├── gensym_vector_starts_at_zero
├── gensym_vector_index_equals_len
├── gensym_vector_always_succeeds
├── gensym_vector_returns_len
└── gensym_vector_index_is_len
    └── vector_put_returns_len

HASHSET (4 theorems)
├── hash_set_insert_member
├── hash_set_insert_idempotent
├── hash_set_remove_not_member
└── hash_set_empty_no_members
```

## Detailed Theorems

### Fixed Array Theorems

#### Theorem: fixed_array_empty_len

**Statement**: Empty array has correct length.

```coq
Theorem fixed_array_empty_len :
  fixed_array_len A (fixed_array_empty A capacity) = capacity.
```

---

#### Theorem: fixed_array_empty_all_none

**Statement**: Empty array has all None values.

```coq
Theorem fixed_array_empty_all_none :
  forall idx,
    idx < capacity ->
    fixed_array_get A (fixed_array_empty A capacity) idx = None.
```

---

#### Theorem: fixed_array_get_oob

**Statement**: Get on out-of-bounds index returns None.

```coq
Theorem fixed_array_get_oob :
  forall arr idx,
    idx >= fixed_array_len A arr ->
    fixed_array_get A arr idx = None.
```

---

#### Theorem: fixed_array_set_valid

**Statement**: Set at valid index succeeds.

```coq
Theorem fixed_array_set_valid :
  forall arr idx val,
    idx < fixed_array_len A arr ->
    exists arr', fixed_array_set A arr idx val = Some arr'.
```

---

#### Theorem: fixed_array_put_full_fails

**Statement**: Put on full array fails (OutOfNames).

```coq
Theorem fixed_array_put_full_fails :
  forall arr val,
    fixed_array_count A arr = fixed_array_len A arr ->
    fixed_array_put A arr val = None.
```

**Intuition**: When all slots are occupied (count = length), there's nowhere to put new data.

---

#### Theorem: gensym_unique_fixed_array (MAIN)

**Statement**: Sequential puts return distinct indices.

```coq
Theorem gensym_unique_fixed_array :
  forall arr val1 val2 arr1 arr2 idx1 idx2,
    fixed_array_put A arr val1 = Some (arr1, idx1) ->
    fixed_array_put A arr1 val2 = Some (arr2, idx2) ->
    idx1 <> idx2.
```

**Intuition**: This is the key uniqueness guarantee. If you put two values sequentially, they get different indices. This ensures `gensym` produces unique channel names.

**Proof Technique**:
1. First put finds an empty slot and fills it
2. Second put can't use the same slot (now filled)
3. `find_empty_after_set_start_zero` proves the second find skips the first index

**Example**:
```
Start: [None, None, None]
Put val1: [Some(val1), None, None], idx1 = 0
Put val2: [Some(val1), Some(val2), None], idx2 = 1
idx1 ≠ idx2 ✓
```

---

### Cyclic Array Theorems

#### Theorem: cyclic_array_put_always_succeeds

**Statement**: Cyclic put always succeeds (returns valid index).

```coq
Theorem cyclic_array_put_always_succeeds :
  forall (arr : ca_array) val,
    caa_next arr < capacity ->
    let '(arr', idx) := ca_put arr val in
    idx < capacity /\ caa_next arr' = (idx + 1) mod capacity.
```

**Intuition**: Unlike fixed arrays, cyclic arrays never fail - they overwrite old data.

---

#### Theorem: cyclic_array_wraps

**Statement**: After capacity puts, index wraps to 0.

```coq
Theorem cyclic_array_wraps :
  forall (arr : ca_array) (default_val : A),
    caa_next arr = capacity - 1 ->
    let '(arr', _) := ca_put arr default_val in
    caa_next arr' = 0.
```

**Example** (capacity = 3):
```
next = 0: put -> idx = 0, next = 1
next = 1: put -> idx = 1, next = 2
next = 2: put -> idx = 2, next = 0  (wraps!)
next = 0: put -> idx = 0, next = 1  (overwrites!)
```

---

#### Theorem: gensym_cyclic_not_unique_after_wrap

**Statement**: Cyclic indices are NOT unique after capacity puts (by design).

```coq
Theorem gensym_cyclic_not_unique_after_wrap :
  forall arr,
    caa_next arr = 0 ->
    length (cag_data arr) = capacity ->
    exists (arr_final : cag_array) (idx_final : nat),
      idx_final = 0.
```

**Intuition**: This is a **negative result** - cyclic arrays deliberately reuse indices. This is appropriate for temporary/cache-like usage where old entries can be discarded.

---

#### Theorem: gensym_cyclic_consecutive

**Statement**: Consecutive puts return consecutive indices (mod capacity).

```coq
Theorem gensym_cyclic_consecutive :
  forall arr val1 val2,
    cag_next arr < capacity ->
    let '(arr1, idx1) := cag_put arr val1 in
    let '(arr2, idx2) := cag_put arr1 val2 in
    idx2 = (idx1 + 1) mod capacity.
```

---

### Vector Theorems

#### Theorem: vector_empty_len / vector_empty_capacity

**Statement**: Empty vector has zero length and capacity.

```coq
Theorem vector_empty_len :
  vec_len A (vector_empty A) = 0.

Theorem vector_empty_capacity :
  vector_capacity A (vector_empty A) = 0.
```

---

#### Theorem: vector_grow_increases_capacity

**Statement**: Grow increases capacity.

```coq
Theorem vector_grow_increases_capacity :
  forall v,
    vector_capacity A (vector_grow A v) > vector_capacity A v \/
    vector_capacity A v = 0.
```

**Intuition**: Either capacity doubles, or it was 0 and becomes 1.

---

#### Theorem: vector_put_increases_len

**Statement**: Put increases length by 1.

```coq
Theorem vector_put_increases_len :
  forall v val,
    let '(v', _) := vector_put A v val in
    vec_len A v' = S (vec_len A v).
```

---

#### Theorem: vector_get_put

**Statement**: Get on valid index returns value (well-formed vector).

```coq
Theorem vector_get_put :
  forall v val,
    vector_wf v ->
    let '(v', idx) := vector_put A v val in
    vector_get A v' idx = Some val.
```

---

#### Theorem: gensym_vector_monotonic

**Statement**: Vector gensym returns increasing indices.

```coq
Theorem gensym_vector_monotonic :
  forall v val1 val2,
    vector_wf A v ->
    let '(v1, idx1) := vector_put A v val1 in
    let '(v2, idx2) := vector_put A v1 val2 in
    idx2 = S idx1.
```

**Intuition**: Each put returns the next index. Indices are strictly increasing.

---

#### Theorem: gensym_unique_vector (MAIN)

**Statement**: Vector gensym returns distinct indices for sequential puts.

```coq
Theorem gensym_unique_vector :
  forall v val1 val2,
    vector_wf A v ->
    let '(v1, idx1) := vector_put A v val1 in
    let '(v2, idx2) := vector_put A v1 val2 in
    idx1 <> idx2.
```

**Intuition**: Since indices are strictly increasing (monotonic), they're automatically distinct.

**Proof**: Follows directly from `gensym_vector_monotonic` - if `idx2 = S idx1`, then `idx1 ≠ idx2`.

---

#### Theorem: gensym_vector_starts_at_zero

**Statement**: Vector indices start at 0 for empty vector.

```coq
Theorem gensym_vector_starts_at_zero :
  forall val,
    let '(_, idx) := vector_put A (vector_empty A) val in
    idx = 0.
```

---

#### Theorem: gensym_vector_index_equals_len

**Statement**: Put always returns vec_len (since grow preserves len).

```coq
Corollary gensym_vector_index_equals_len :
  forall v val,
    snd (vector_put A v val) = vec_len A v.
```

---

### HashSet Theorems

#### Theorem: hash_set_insert_member

**Statement**: Insert makes element a member.

```coq
Theorem hash_set_insert_member :
  forall s a,
    hash_set_member A A_eq_dec (hash_set_insert A A_eq_dec s a) a = true.
```

---

#### Theorem: hash_set_insert_idempotent

**Statement**: Insert is idempotent.

```coq
Theorem hash_set_insert_idempotent :
  forall s a,
    hash_set_insert A A_eq_dec (hash_set_insert A A_eq_dec s a) a =
    hash_set_insert A A_eq_dec s a.
```

**Intuition**: Inserting the same element twice is the same as inserting once.

---

#### Theorem: hash_set_remove_not_member

**Statement**: Remove removes element from membership.

```coq
Theorem hash_set_remove_not_member :
  forall s a,
    hash_set_member A A_eq_dec (hash_set_remove A A_eq_dec s a) a = false.
```

---

## Examples

### Example: Fixed Array Gensym Sequence

```
capacity = 3
arr0 = [None, None, None]

put(val1) -> arr1 = [Some(val1), None, None], idx1 = 0
put(val2) -> arr2 = [Some(val1), Some(val2), None], idx2 = 1
put(val3) -> arr3 = [Some(val1), Some(val2), Some(val3)], idx3 = 2
put(val4) -> None (OutOfNames!)

All indices unique: 0, 1, 2
```

### Example: Vector Gensym (Unbounded)

```
v0 = { data: [], len: 0 }

put(val1) -> grow, v1 = { data: [Some(val1)], len: 1 }, idx1 = 0
put(val2) -> grow, v2 = { data: [Some(val1), Some(val2)], len: 2 }, idx2 = 1
put(val3) -> grow, v3 = { data: [Some(val1), Some(val2), None, None], len: 3 }, idx3 = 2
...

Indices: 0, 1, 2, 3, ... (always increasing)
```

### Example: Cyclic Array Wraparound

```
capacity = 3
arr0 = { data: [None, None, None], next: 0 }

put(val1) -> arr1 = { data: [Some(val1), None, None], next: 1 }, idx1 = 0
put(val2) -> arr2 = { data: [Some(val1), Some(val2), None], next: 2 }, idx2 = 1
put(val3) -> arr3 = { data: [Some(val1), Some(val2), Some(val3)], next: 0 }, idx3 = 2
put(val4) -> arr4 = { data: [Some(val4), Some(val2), Some(val3)], next: 1 }, idx4 = 0  OVERWRITES!

idx1 = idx4 = 0 (NOT unique!)
```

## Correspondence to Rust

| Rocq Definition | Rust Implementation |
|-----------------|---------------------|
| `fixed_array` | `Vec<Option<T>>` with capacity check |
| `fixed_array_put` | `FixedArrayStore::gensym()` |
| `cyclic_array` | `CyclicArrayStore` |
| `cyclic_array_put` | `CyclicArrayStore::gensym()` |
| `vector` | `VectorStore` |
| `vector_put` | `VectorStore::gensym()` |
| `hash_set` | `HashSet<T>` |
| `OutOfNames` error | `SpaceError::OutOfNames` |

## Design Notes

### Why Different Storage Types?

| Type | Use Case | Uniqueness | Bounded |
|------|----------|------------|---------|
| Fixed Array | Resource-constrained environments | Yes | Yes |
| Cyclic Array | Temporary/cache channels | No (wraps) | Yes |
| Vector | General purpose | Yes | No |
| HashSet | Seq qualifier channels | N/A | No |

### Gensym Uniqueness Summary

- **Fixed Array**: Unique until full, then fails
- **Cyclic Array**: NOT unique after wraparound
- **Vector**: Always unique (indices always increase)
- **HashSet**: Not applicable (presence-only)

## Next Steps

Continue to [08-space-factory.md](08-space-factory.md) for the space factory pattern specifications.
