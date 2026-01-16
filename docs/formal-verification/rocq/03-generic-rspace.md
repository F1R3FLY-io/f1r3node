# Core Space Operations: GenericRSpace.v

This document covers the largest and most important specification file, `GenericRSpace.v`, which defines the core tuple space operations and their invariant preservation proofs.

## Overview

GenericRSpace is the parameterized implementation combining:
- A **ChannelStore** (how channels are indexed)
- A **Match** strategy (how patterns match data)

The specification proves that regardless of the choice of these parameters, the fundamental safety invariants are maintained.

## Key Concepts

### State Record

The GenericRSpace state encapsulates all tuple space data:

```coq
Record GenericRSpaceState (A K : Type) := mkGRState {
  gr_data_store : Channel -> list (A * bool);       (* Data at each channel *)
  gr_cont_store : list Channel -> list (list A * K * bool);  (* Continuations *)
  gr_joins : Channel -> list (list Channel);        (* Join relationships *)
  gr_qualifier : SpaceQualifier;                    (* 0=Default, 1=Temp, 2=Seq *)
  gr_name_counter : nat;                            (* For gensym *)
}.
```

Where:
- `A` is the data type
- `K` is the continuation type
- `bool` indicates persistence (true = persistent, false = consumable)

### The Core Invariant: no_pending_match

The fundamental safety property states: **there is never simultaneously data and a matching continuation on the same channel(s)**.

```coq
Definition no_pending_match (state : GenericRSpaceState A K) : Prop :=
  forall ch data persist,
    In (data, persist) (gr_data_store state ch) ->
    forall join_channels patterns cont cpersist,
      In join_channels (gr_joins state ch) ->
      In (patterns, cont, cpersist) (gr_cont_store state join_channels) ->
      any_pattern_matches patterns data = false.
```

**Intuition**: If `no_pending_match` is violated, it means:
1. Data exists at channel `ch`
2. A continuation is waiting on channels including `ch`
3. The continuation's pattern matches the data

This state is invalid because the match should have fired immediately - data should never coexist with a matching continuation.

### Join Consistency

For multi-channel consumes (joins), we maintain:

```coq
Definition join_consistent_state (state : GenericRSpaceState A K) : Prop :=
  forall ch channels,
    In channels (gr_joins state ch) ->
    In ch channels.

Definition cont_implies_join_forward (state : GenericRSpaceState A K) : Prop :=
  forall channels patterns cont cpersist,
    In (patterns, cont, cpersist) (gr_cont_store state channels) ->
    forall ch, In ch channels ->
      In channels (gr_joins state ch).
```

### Full Invariant

The combined invariant packages all three properties:

```coq
Definition full_invariant (state : GenericRSpaceState A K) : Prop :=
  no_pending_match A K M state /\
  join_consistent_state A K state /\
  cont_implies_join_forward A K state.
```

### Operation Results

Each operation returns a result indicating what happened:

```coq
(* Produce results *)
Inductive ProduceResult :=
  | PR_Stored : GenericRSpaceState A K -> ProduceResult   (* Data stored *)
  | PR_Fired : K -> A -> GenericRSpaceState A K -> ProduceResult.  (* Continuation fired *)

(* Consume results *)
Inductive ConsumeResult :=
  | CR_Stored : GenericRSpaceState A K -> ConsumeResult   (* Continuation stored *)
  | CR_Matched : list A -> GenericRSpaceState A K -> ConsumeResult. (* Data matched *)

(* Install results *)
Inductive InstallResult :=
  | IR_Installed : GenericRSpaceState A K -> InstallResult  (* Continuation installed *)
  | IR_Matched : K -> list A -> GenericRSpaceState A K -> InstallResult. (* Data found *)
```

## Theorem Hierarchy

```
FULL INVARIANT PRESERVATION (TOP-LEVEL)
├── produce_maintains_full_invariant          ★ Main theorem
│   ├── produce_maintains_invariant           (no_pending_match for PR_Stored)
│   │   ├── find_matching_cont_none_no_match
│   │   ├── find_first_none_implies_all_false
│   │   └── any_pattern_matches_equiv
│   ├── produce_fired_maintains_invariant     (no_pending_match for PR_Fired)
│   │   └── remove_first_preserves_non_matching
│   ├── produce_preserves_join_consistency_stored
│   ├── produce_preserves_join_consistency_fired
│   ├── produce_preserves_cont_implies_join_stored
│   └── produce_preserves_cont_implies_join_fired
│
├── consume_maintains_full_invariant          ★ Main theorem (with precondition)
│   ├── consume_maintains_invariant
│   ├── consume_preserves_join_consistency
│   └── consume_preserves_cont_implies_join_forward
│
└── install_maintains_invariant               ★ Main theorem
    ├── install_preserves_join_consistency
    └── install_preserves_cont_implies_join

EMPTY STATE THEOREMS
├── empty_state_no_pending_match
├── empty_state_join_consistent_fwd
└── empty_state_full_invariant

EXCLUSIVITY THEOREMS
├── produce_exclusive    (either fires or stores)
└── install_exclusive    (either installs or matches immediately)

HELPER LEMMAS
├── find_first_none_implies_all_false
├── find_first_some_implies_pred
├── remove_first_preserves_non_matching
└── In_dec_list
```

## Detailed Theorems

### Theorem: empty_state_no_pending_match

**Statement**: An empty state satisfies the no_pending_match invariant.

```coq
Theorem empty_state_no_pending_match :
  forall qualifier, no_pending_match (empty_gr_state qualifier).
```

**Intuition**: An empty state has no data anywhere, so the invariant is vacuously true - there's nothing to mismatch.

**Proof Technique**: Unfold definitions, show that the antecedent of the invariant (data exists) is false.

---

### Theorem: produce_maintains_full_invariant (MAIN)

**Statement**: Produce maintains all invariants regardless of whether it stores data or fires a continuation.

```coq
Theorem produce_maintains_full_invariant :
  forall state ch data persist,
    full_invariant state ->
    match produce_spec A K M state ch data persist with
    | @PR_Stored _ _ state' => full_invariant state'
    | @PR_Fired _ _ _ _ state' => full_invariant state'
    end.
```

**Intuition**: This is the key safety theorem for produce. It says:
1. If the current state is valid (satisfies full_invariant)
2. Then after produce, the resulting state is also valid

This holds whether produce stores the data (no matching continuation found) or fires a continuation (match found).

**Proof Structure**:
```
produce_maintains_full_invariant
├── Case PR_Stored:
│   ├── no_pending_match preserved (produce_maintains_invariant)
│   ├── join_consistency preserved (produce_preserves_join_consistency_stored)
│   └── cont_implies_join preserved (produce_preserves_cont_implies_join_stored)
│
└── Case PR_Fired:
    ├── no_pending_match preserved (produce_fired_maintains_invariant)
    ├── join_consistency preserved (produce_preserves_join_consistency_fired)
    └── cont_implies_join preserved (produce_preserves_cont_implies_join_fired)
```

---

### Theorem: produce_maintains_invariant

**Statement**: When produce stores data (no match found), no_pending_match is preserved.

```coq
Theorem produce_maintains_invariant :
  forall state ch data persist state',
    no_pending_match A K M state ->
    produce_spec A K M state ch data persist = PR_Stored A K state' ->
    no_pending_match A K M state'.
```

**Intuition**: If `produce` returns `PR_Stored`, it means `find_matching_cont` returned `None`. This means no continuation at any of ch's join groups matches the new data. Therefore, adding the data doesn't create a pending match.

**Key Lemma Used**: `find_matching_cont_none_no_match` - if the search returns None, no continuation matches.

---

### Theorem: produce_fired_maintains_invariant

**Statement**: When produce fires a continuation, no_pending_match is preserved.

```coq
Theorem produce_fired_maintains_invariant :
  forall state ch data persist cont d state',
    no_pending_match A K M state ->
    produce_spec A K M state ch data persist = PR_Fired A K cont d state' ->
    no_pending_match A K M state'.
```

**Intuition**: When produce fires, we:
1. Don't add data to the store (it goes to the continuation)
2. Remove the matching continuation

Since we don't add data and we remove a continuation, the invariant is preserved.

**Key Lemma Used**: `remove_first_preserves_non_matching` - elements remaining after removal were in the original list.

---

### Theorem: consume_maintains_invariant

**Statement**: Consume maintains no_pending_match when no data matches.

```coq
Theorem consume_maintains_invariant :
  forall state channels patterns cont persist state',
    no_pending_match A K M state ->
    join_consistent_state A K state ->
    cont_implies_join_forward A K state ->
    consume_spec A K state channels patterns cont persist = @CR_Stored A K state' ->
    (* Precondition: no data at any channel matches the patterns *)
    (forall ch, In ch channels ->
      forall data dpersist,
        In (data, dpersist) (gr_data_store state ch) ->
        any_pattern_matches A M patterns data = false) ->
    no_pending_match A K M state'.
```

**Intuition**: The precondition ensures that adding this continuation doesn't create a pending match - we've verified that no existing data matches the new patterns.

**Note**: In the actual implementation, `consume` would first check for matching data and immediately return it (like `install` does). The specification here covers the "store continuation" case which requires the precondition.

---

### Theorem: install_maintains_invariant

**Statement**: Install maintains all invariants.

```coq
Theorem install_maintains_invariant :
  forall state channels patterns cont state',
    no_pending_match A K M state ->
    join_consistent_state A K state ->
    cont_implies_join_forward A K state ->
    install_spec state channels patterns cont = IR_Installed state' ->
    (* Precondition: no data matches *)
    (forall ch, In ch channels ->
      forall data dpersist,
        In (data, dpersist) (gr_data_store state ch) ->
        any_pattern_matches A M patterns data = false) ->
    no_pending_match A K M state'.
```

**Intuition**: Same as consume - the precondition ensures safety. Install is permanent (persist=true) but follows the same logic.

---

### Theorem: produce_exclusive

**Statement**: Produce either fires or stores, never both (and always one).

```coq
Theorem produce_exclusive :
  forall state ch data persist,
    (exists state' cont d, produce_spec A K M state ch data persist = PR_Fired A K cont d state') \/
    (exists state', produce_spec A K M state ch data persist = PR_Stored A K state').
```

**Intuition**: This is a totality and exclusivity theorem. `produce` always terminates with exactly one outcome: either it finds a matching continuation (PR_Fired) or it doesn't (PR_Stored).

---

### Theorem: install_exclusive

**Statement**: Install either installs or matches immediately, never both.

```coq
Theorem install_exclusive :
  forall state channels patterns cont,
    (exists state', install_spec state channels patterns cont = IR_Installed state') \/
    (exists matched_data, install_spec state channels patterns cont = IR_Matched cont matched_data state).
```

---

## Examples

### Example: Empty State Satisfies Invariants

```coq
(* Create empty state with Default qualifier *)
Let state := empty_gr_state 0.

(* Prove it satisfies the full invariant *)
Proof:
  - no_pending_match: vacuously true (no data)
  - join_consistent: vacuously true (no joins)
  - cont_implies_join: vacuously true (no continuations)
```

### Example: Produce Then Consume Fires

Consider a scenario:
1. Start with empty state
2. Produce data `d` at channel `ch`
3. Consume with pattern `p` that matches `d`

```
Initial:  {}  (empty)

After produce(ch, d, false):
  State = { data_store: ch -> [(d, false)] }
  Result = PR_Stored(state')

After consume([ch], [p], k, false) where matches(p, d) = true:
  Result = CR_Matched([d])  (immediate match, not stored)
```

The invariant holds throughout:
- After produce: no continuations exist, so trivially no pending match
- After consume: data was consumed (or continuation was never stored due to match)

### Example: Multi-Channel Join

```
Consume([ch1, ch2], [p1, p2], k, false)

This creates:
  - gr_joins(ch1) includes [ch1, ch2]
  - gr_joins(ch2) includes [ch1, ch2]
  - gr_cont_store([ch1, ch2]) includes (([p1, p2], k, false))

For the continuation to fire:
  - Data at ch1 must match p1
  - Data at ch2 must match p2
  - BOTH must be present simultaneously
```

## Correspondence to Rust

| Rocq Definition | Rust Implementation |
|-----------------|---------------------|
| `GenericRSpaceState` | `GenericRSpace<CS, M>` struct |
| `gr_data_store` | `ChannelStore::data_at()` |
| `gr_cont_store` | `ChannelStore::continuations_at()` |
| `gr_joins` | `ChannelStore::joins_for()` |
| `produce_spec` | `GenericRSpace::produce()` |
| `consume_spec` | `GenericRSpace::consume()` |
| `install_spec` | `GenericRSpace::install()` |
| `no_pending_match` | Invariant maintained by implementation |
| `find_matching_cont` | `GenericRSpace::find_matching_continuation()` |

## Proof Techniques Summary

1. **Induction on Lists**: Used for `find_first`, `find_matching_cont`, etc.
2. **Case Analysis**: On operation results (PR_Stored vs PR_Fired)
3. **Decidability**: `In_dec`, `list_eq_dec` for constructive proofs
4. **Unfolding Definitions**: Breaking down specs into constituent parts
5. **Preservation Arguments**: Showing unchanged parts remain valid

## Next Steps

Continue to [04-checkpoint-replay.md](04-checkpoint-replay.md) for the checkpoint and deterministic replay proofs.
