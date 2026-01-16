# Pattern Matching: Match.v

This document covers the pattern matching specification in `Match.v`, which defines the pluggable `Match<P, A>` trait used by Reified RSpaces.

## Overview

Pattern matching is fundamental to tuple space semantics. A **continuation** waits for data that **matches** its pattern. The Match module provides:

- An abstract `Match` typeclass for pluggable matching strategies
- Concrete instances: ExactMatch, WildcardMatch
- Match combinators: And, Or
- Integration with data collections
- Properties ensuring matching is well-behaved

## Key Concepts

### The Match Typeclass

The core abstraction is a typeclass parameterized by pattern type `P` and data type `A`:

```coq
Class Match (P A : Type) := {
  (** Returns true if pattern matches data *)
  match_fn : P -> A -> bool;

  (** Matcher name for debugging *)
  matcher_name : string;
}.
```

This design allows different matching strategies to be plugged into the same generic space implementation.

### Match Properties

Several properties characterize well-behaved matchers:

```coq
(** Reflexivity: A pattern that matches everything *)
Definition match_reflexive (p : P) : Prop :=
  forall a, match_fn p a = true.

(** A wildcard pattern matches all data *)
Definition is_wildcard (p : P) : Prop := match_reflexive p.

(** Pattern refinement: p1 is more specific than p2 *)
Definition pattern_refines (p1 p2 : P) : Prop :=
  forall a, match_fn p1 a = true -> match_fn p2 a = true.
```

### Match Instances

#### Exact Match

Requires equality between pattern and data:

```coq
Definition exact_match_fn (pattern data : A) : bool :=
  if A_eq_dec pattern data then true else false.
```

#### Wildcard Match

Accepts any data:

```coq
Definition wildcard_match_fn (pattern : P) (data : A) : bool := true.
```

### Match Combinators

Matchers can be combined:

```coq
(** And combinator: both matchers must match *)
Definition and_match_fn (p : P) (a : A) : bool :=
  @match_fn P A M1 p a && @match_fn P A M2 p a.

(** Or combinator: either matcher can match *)
Definition or_match_fn (p : P) (a : A) : bool :=
  @match_fn P A M1 p a || @match_fn P A M2 p a.
```

### Data Equivalence

Two data values are equivalent if they match the same patterns:

```coq
Definition data_equivalent (a1 a2 : A) : Prop :=
  forall p, match_fn p a1 = match_fn p a2.
```

## Theorem Hierarchy

```
MATCH PROPERTIES
├── match_decidable              (* Match result is decidable *)
│
├── EXACT MATCH
│   ├── exact_match_reflexive    (* a matches a *)
│   └── exact_match_symmetric    (* a1 matches a2 <-> a2 matches a1 *)
│
├── WILDCARD MATCH
│   ├── wildcard_always_matches  (* wildcard matches everything *)
│   └── wildcard_is_reflexive    (* wildcard is reflexive *)
│
├── COMBINATORS
│   ├── and_match_implies_both   (* And requires both *)
│   └── or_match_implies_either  (* Or allows either *)
│
├── SUBSTITUTION
│   └── match_substitution       (* Equivalent data preserves matches *)
│
└── COLLECTIONS
    └── find_with_matcher_correct  (* Find returns matching data *)
```

## Detailed Theorems

### Theorem: match_decidable

**Statement**: The result of a match is decidable.

```coq
Lemma match_decidable : forall p a, {match_fn p a = true} + {match_fn p a = false}.
```

**Intuition**: Since `match_fn` returns a boolean, we can always decide which case holds. This is trivial but important for computability.

**Proof**:
```coq
Proof.
  intros p a.
  destruct (match_fn p a) eqn:Hm.
  - left. reflexivity.
  - right. reflexivity.
Qed.
```

---

### Theorem: exact_match_reflexive

**Statement**: Exact match is reflexive - any value matches itself.

```coq
Theorem exact_match_reflexive : forall a, exact_match_fn a a = true.
```

**Intuition**: If the pattern equals the data (and equality is decidable), the match succeeds. When pattern and data are the same value, equality holds trivially.

**Example**:
```coq
(* The number 42 matches pattern 42 *)
exact_match_fn 42 42 = true

(* The string "hello" matches pattern "hello" *)
exact_match_fn "hello" "hello" = true
```

---

### Theorem: exact_match_symmetric

**Statement**: Exact match is symmetric.

```coq
Theorem exact_match_symmetric :
  forall a1 a2, exact_match_fn a1 a2 = true -> exact_match_fn a2 a1 = true.
```

**Intuition**: If pattern a1 matches data a2, then a1 = a2 (by exact match semantics). By symmetry of equality, a2 = a1, so pattern a2 matches data a1.

**Proof Technique**: Unfold the definition, destruct on the equality decision, and use symmetry of equality.

---

### Theorem: wildcard_always_matches

**Statement**: Wildcard matches everything.

```coq
Theorem wildcard_always_matches : forall p a, wildcard_match_fn p a = true.
```

**Intuition**: By definition, wildcard match always returns true. This is the "don't care" pattern.

**Proof**: Immediate by definition.

**Example**:
```coq
(* Wildcard matches any value *)
wildcard_match_fn _ 42 = true
wildcard_match_fn _ "anything" = true
wildcard_match_fn _ (complex_data_structure) = true
```

---

### Theorem: wildcard_is_reflexive

**Statement**: Every wildcard pattern is reflexive (matches all data).

```coq
Theorem wildcard_is_reflexive : forall p, is_wildcard p.
```

where `is_wildcard p := forall a, match_fn p a = true`.

**Intuition**: Since wildcard always returns true, any wildcard pattern satisfies the reflexivity property.

---

### Theorem: and_match_implies_both

**Statement**: If an And-combined match succeeds, both component matches succeeded.

```coq
Lemma and_match_implies_both :
  forall p a, and_match_fn p a = true ->
  @match_fn P A M1 p a = true /\ @match_fn P A M2 p a = true.
```

**Intuition**: The And combinator uses boolean AND (`&&`), which is true only when both operands are true.

**Proof Technique**: Use `andb_prop` from the standard library.

**Example**:
```coq
(* If both type check AND value check succeed... *)
and_match_fn type_pattern value = true
->
(* ...then type check succeeded AND value check succeeded *)
type_match type_pattern value = true /\
value_match type_pattern value = true
```

---

### Theorem: or_match_implies_either

**Statement**: If an Or-combined match succeeds, at least one component match succeeded.

```coq
Lemma or_match_implies_either :
  forall p a, or_match_fn p a = true ->
  @match_fn P A M1 p a = true \/ @match_fn P A M2 p a = true.
```

**Intuition**: The Or combinator uses boolean OR (`||`), which is true when at least one operand is true.

---

### Theorem: match_substitution

**Statement**: If data values are equivalent (match the same patterns), a match on one implies a match on the other.

```coq
Theorem match_substitution :
  forall p a1 a2,
    match_fn p a1 = true ->
    data_equivalent a1 a2 ->
    match_fn p a2 = true.
```

**Intuition**: Data equivalence means `forall p, match_fn p a1 = match_fn p a2`. So if a1 matches pattern p, and a1 is equivalent to a2, then a2 also matches p.

**Example**:
```coq
(* If 42 matches pattern p, and 42 is equivalent to (40+2)... *)
match_fn p 42 = true
data_equivalent 42 (40+2)
(* ...then (40+2) also matches p *)
match_fn p (40+2) = true
```

---

### Theorem: find_with_matcher_correct

**Statement**: If `find_with_matcher` succeeds, the returned data actually matches the pattern.

```coq
Theorem find_with_matcher_correct :
  forall pattern l data rest,
    find_with_matcher pattern l = Some (data, rest) ->
    @match_fn A A M pattern data = true.
```

**Intuition**: The `find_with_matcher` function traverses a list looking for the first element that matches. If it returns `Some (data, rest)`, then `data` is the matching element.

**Proof Technique**: Induction on the list structure, with case analysis on the match result.

**Example**:
```coq
(* Given a list of data *)
l = [(x, true); (y, false); (z, true)]

(* If find returns y *)
find_with_matcher pattern l = Some (y, rest)

(* Then y matches the pattern *)
match_fn pattern y = true
```

## Integration with Collections

The `find_with_matcher` function integrates matching with the data collection:

```coq
Fixpoint find_with_matcher (pattern : A) (l : list (A * bool))
  : option (A * list (A * bool)) :=
  match l with
  | [] => None
  | (data, persist) :: rest =>
    if @match_fn A A M pattern data then
      if persist then Some (data, l)       (* Keep persistent data *)
      else Some (data, rest)               (* Remove non-persistent data *)
    else
      match find_with_matcher pattern rest with
      | None => None
      | Some (found, rest') => Some (found, (data, persist) :: rest')
      end
  end.
```

This shows how:
1. The matcher is used to find qualifying data
2. Persistent data remains after matching
3. Non-persistent data is removed when matched

## Correspondence to Rust

| Rocq Definition | Rust Implementation |
|-----------------|---------------------|
| `Match` typeclass | `matcher::Match<P, A>` trait |
| `match_fn` | `Match::matches(&self, pattern, data)` |
| `exact_match_fn` | `ExactMatcher::matches` |
| `wildcard_match_fn` | Pattern with `is_wildcard = true` |
| `find_with_matcher` | `GenericRSpace::find_matching_data` |

## Use in GenericRSpace

The Match trait is a type parameter of GenericRSpace:

```rust
pub struct GenericRSpace<CS, M>
where
    CS: ChannelStore,
    M: Match<CS::Pattern, CS::Data>,
{
    // ...
}
```

This allows the same GenericRSpace implementation to work with different matching strategies:
- Rholang's structural matching
- Exact equality matching
- Similarity-based matching (for VectorDB collections)

## Next Steps

Continue to [03-generic-rspace.md](03-generic-rspace.md) for the core GenericRSpace invariants and operation proofs.
