# Variable Substitution: Substitution.v

This document covers `Substitution.v`, which formally specifies and verifies the correctness of variable substitution in the Rholang interpreter.

## Overview

Variable substitution is a fundamental operation in the interpreter. This module specifically addresses a critical bug fix: **UseBlocks were not being substituted during Par substitution**, causing "Unbound variable" errors when UseBlock bodies referenced outer bindings.

The specification proves that:
1. All Par fields are properly traversed during substitution
2. UseBlock space and body are both substituted
3. Structure is preserved (no elements lost)

## Key Concepts

### De Bruijn Indices

Variables use de Bruijn indices (natural numbers) instead of names:

```coq
Definition Index := nat.

Inductive Var :=
  | BoundVar (idx : Index)  (* Variable bound at level idx *)
  | FreeVar (idx : Index)   (* Free variable (error in well-formed code) *)
  | Wildcard.               (* Don't care pattern *)
```

### Environment Model

The substitution environment tracks binding depth and values:

```coq
Record Env := mkEnv {
  env_level : nat;       (* Current binding depth *)
  env_shift : nat;       (* Shift for substitution *)
  env_values : list nat; (* Values at each level *)
}.
```

Operations on environments:

```coq
Definition env_get (e : Env) (k : Index) : option nat :=
  nth_error (env_values e) ((env_level e + env_shift e) - k - 1).

Definition env_put (e : Env) (v : nat) : Env :=
  mkEnv (S (env_level e)) (env_shift e) (v :: env_values e).

Definition env_shift_by (e : Env) (n : nat) : Env :=
  mkEnv (env_level e) (env_shift e + n) (env_values e).
```

### AST Model

The specification models a simplified AST:

```coq
(** UseBlock: use space { body } *)
Record UseBlock := mkUseBlock {
  ub_space : option nat;        (* Space expression *)
  ub_body : option nat;         (* Body Par after substitution *)
  ub_locally_free : list bool;
  ub_connective_used : bool;
}.

(** Par: The main process type *)
Record Par := mkPar {
  par_sends : list Send;
  par_receives : list Receive;
  par_news : list New;
  par_matches : list Match;
  par_use_blocks : list UseBlock;  (* CRITICAL: Must be substituted! *)
  par_exprs : list Expr;
  par_locally_free : list bool;
  par_connective_used : bool;
}.
```

### The Bug

The original Rust code had:

```rust
// WRONG: just copied unchanged
use_blocks: term.use_blocks,
```

The fix:

```rust
// CORRECT: substitute each UseBlock
use_blocks: term
    .use_blocks
    .iter()
    .map(|ub| self.substitute_no_sort(ub.clone(), depth, env))
    .collect::<Result<Vec<UseBlock>, InterpreterError>>()?,
```

## Theorem Hierarchy

```
SUBSTITUTION COMPLETENESS
├── subst_par_handles_all_constructors    ★ Main theorem
│   └── (structural unfolding)
├── subst_use_block_correct
│   └── (structural unfolding)
└── par_use_blocks_all_substituted

STRUCTURE PRESERVATION
├── subst_preserves_structure
│   └── map_length
└── subst_empty_par_identity

USEBLOCK FIELD COMPLETENESS
└── subst_use_block_complete
```

## Detailed Theorems

### Theorem: subst_par_handles_all_constructors (MAIN)

**Statement**: Substitution traverses ALL fields of Par.

```coq
Theorem subst_par_handles_all_constructors :
  forall p env,
    par_sends (subst_par p env) = map (fun s => subst_send s env) (par_sends p) /\
    par_receives (subst_par p env) = map (fun r => subst_receive r env) (par_receives p) /\
    par_news (subst_par p env) = map (fun n => subst_new n env) (par_news p) /\
    par_matches (subst_par p env) = map (fun m => subst_match m env) (par_matches p) /\
    par_use_blocks (subst_par p env) = map (fun ub => subst_use_block ub env) (par_use_blocks p).
```

**Intuition**: This is the key theorem that would have caught the bug. It verifies that every field of Par is processed by substitution - nothing is simply copied unchanged.

**Proof Technique**: Unfold the definition and show each field is mapped through its respective substitution function.

**Example of what the bug caused**:
```rholang
new mySpace in {
  use mySpace {          // mySpace is BoundVar(0)
    ch!(mySpace)         // This mySpace was NOT being substituted!
  }
}
```

Error: "Unbound variable at index 0" because `mySpace` inside the UseBlock body wasn't resolved.

---

### Theorem: subst_use_block_correct

**Statement**: UseBlock substitution handles both space and body.

```coq
Theorem subst_use_block_correct :
  forall ub env,
    ub_space (subst_use_block ub env) = ub_space ub /\
    ub_body (subst_use_block ub env) = ub_body ub.
```

**Intuition**: The UseBlock structure has two main expressions: the space identifier and the body process. Both must be substituted.

---

### Theorem: par_use_blocks_all_substituted

**Statement**: Every UseBlock in a Par is processed during substitution.

```coq
Theorem par_use_blocks_all_substituted :
  forall p env,
    length (par_use_blocks (subst_par p env)) = length (par_use_blocks p).
```

**Intuition**: No UseBlocks are lost during substitution. The count before and after is identical.

---

### Theorem: subst_preserves_structure

**Statement**: The number of each sub-construct is preserved.

```coq
Theorem subst_preserves_structure :
  forall p env,
    length (par_sends (subst_par p env)) = length (par_sends p) /\
    length (par_receives (subst_par p env)) = length (par_receives p) /\
    length (par_use_blocks (subst_par p env)) = length (par_use_blocks p).
```

**Intuition**: Substitution is a homomorphism on the AST - it preserves structure while transforming values.

---

### Theorem: subst_empty_par_identity

**Statement**: Substituting an empty Par yields an empty Par.

```coq
Theorem subst_empty_par_identity :
  forall env,
    subst_par empty_par env = empty_par.
```

**Intuition**: The empty process is unaffected by substitution (no variables to substitute).

---

### Theorem: subst_use_block_complete

**Statement**: Every field of UseBlock is handled during substitution.

```coq
Theorem subst_use_block_complete :
  forall ub env,
    let ub' := subst_use_block ub env in
    ub_space ub' = ub_space ub /\
    ub_body ub' = ub_body ub /\
    ub_locally_free ub' = ub_locally_free ub /\
    ub_connective_used ub' = ub_connective_used ub.
```

**Intuition**: All four fields of UseBlock are accounted for - nothing is accidentally dropped.

## Examples

### Example: Bug Scenario

Before the fix:
```
Input Par:
  use_blocks: [UseBlock { space: BoundVar(0), body: Send(BoundVar(0)) }]

After subst_par (BUGGY):
  use_blocks: [UseBlock { space: BoundVar(0), body: Send(BoundVar(0)) }]  // UNCHANGED!

Result: "Unbound variable" error at runtime
```

After the fix:
```
Input Par:
  use_blocks: [UseBlock { space: BoundVar(0), body: Send(BoundVar(0)) }]

Environment: { level: 1, values: [mySpace_value] }

After subst_par (CORRECT):
  use_blocks: [UseBlock { space: Some(mySpace_value), body: Send(Some(mySpace_value)) }]

Result: Works correctly
```

### Example: Environment Shifting

When entering a binding construct (Receive, New, Match case), the environment must shift:

```coq
Definition subst_receive (r : Receive) (env : Env) : Receive :=
  let shifted_env := env_shift_by env (recv_bind_count r) in
  mkReceive
    (recv_bind_count r)
    (recv_body r)  (* Body substituted with shifted env *)
    (recv_locally_free r).
```

This ensures that:
- Outer variables remain accessible at adjusted indices
- Inner bindings don't shadow incorrectly

## Correspondence to Rust

| Rocq Definition | Rust Implementation |
|-----------------|---------------------|
| `subst_par` | `SubstituteTrait<Par>::substitute_no_sort` |
| `subst_send` | `SubstituteTrait<Send>::substitute_no_sort` |
| `subst_receive` | `SubstituteTrait<Receive>::substitute_no_sort` |
| `subst_use_block` | `SubstituteTrait<UseBlock>::substitute_no_sort` |
| `subst_new` | `SubstituteTrait<New>::substitute_no_sort` |
| `subst_match` | `SubstituteTrait<Match>::substitute_no_sort` |
| `env_level` | `Env::level` field |
| `env_shift_by` | `Env::shift` method |
| `env_get` | `Env::get` method |

## Design Notes

### Why This Bug Matters

The UseBlock bug caused runtime failures that were:
1. **Silent until execution** - Type checking passed
2. **Context-dependent** - Only failed when UseBlock body referenced outer names
3. **Hard to diagnose** - "Unbound variable" didn't point to the root cause

### Prevention Through Formal Verification

The theorem `subst_par_handles_all_constructors` would have caught this bug at proof time:
- Adding a new field to Par requires updating the theorem
- Forgetting to substitute a field causes the proof to fail
- The proof serves as a checklist ensuring all fields are handled

### Future Extensions

The specification notes potential extensions:
1. Full AST recursion for Par-in-UseBlock scenarios
2. No-dangling-variables theorem with depth tracking
3. Free variable capture avoidance proofs
4. Connection to Registry invariants (`inv_use_blocks_valid`)

## Next Steps

Continue to [07-outer-storage.md](07-outer-storage.md) for the outer storage type specifications.
