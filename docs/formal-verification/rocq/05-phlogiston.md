# Gas Accounting: Phlogiston.v

This document covers `Phlogiston.v`, which specifies the cost accounting (phlogiston/gas) system for RSpace operations.

## Overview

"Phlogiston" is the historical name for the gas/cost accounting system in RSpace. The module proves properties about:
1. Charging costs for operations
2. Resource exhaustion behavior
3. Cost bounds and accumulation
4. Refund operations

The key safety property is **non-negative preservation**: successful charges never drive the balance negative.

## Key Concepts

### Cost Record

A cost represents a charge for an operation:

```coq
Record Cost := mkCost {
  cost_value : Z;          (* Amount to charge *)
  cost_operation : nat;    (* Operation identifier *)
}.
```

### Operation Identifiers

Predefined operation types:

```coq
Definition OP_SEND := 1%nat.
Definition OP_RECEIVE := 2%nat.
Definition OP_NEW := 3%nat.
Definition OP_LOOKUP := 4%nat.
Definition OP_REMOVE := 5%nat.
Definition OP_ADD := 6%nat.
Definition OP_MATCH := 7%nat.
Definition OP_PRODUCE := 8%nat.
Definition OP_CONSUME := 9%nat.
Definition OP_INSTALL := 10%nat.
```

### Standard Cost Values

From the Rust implementation (`costs.rs`):

```coq
Definition COST_SEND := 11%Z.
Definition COST_RECEIVE := 11%Z.
Definition COST_NEW_BINDING := 2%Z.
Definition COST_NEW_EVAL := 10%Z.
Definition COST_LOOKUP := 3%Z.
Definition COST_REMOVE := 3%Z.
Definition COST_ADD := 3%Z.
Definition COST_MATCH := 12%Z.
```

### Cost Manager State

The CostManager tracks available phlogiston and operation history:

```coq
Record CostManager := mkCostManager {
  cm_available : Z;      (* Available phlogiston *)
  cm_log : list Cost;    (* Log of charged costs *)
  cm_initial : Z;        (* Initial phlogiston amount *)
}.
```

### Charge Result

Charging can succeed or fail:

```coq
Inductive ChargeResult :=
  | ChargeOk : CostManager -> ChargeResult
  | OutOfPhlogiston : ChargeResult.
```

### The Charge Operation

```coq
Definition charge (cm : CostManager) (cost : Cost) : ChargeResult :=
  let available := cm_available cm in
  let amount := cost_value cost in
  if (available <? 0)%Z then
    OutOfPhlogiston
  else if (available <? amount)%Z then
    OutOfPhlogiston
  else
    ChargeOk (mkCostManager
      (available - amount)
      (cost :: cm_log cm)
      (cm_initial cm)).
```

## Theorem Hierarchy

```
CHARGE PROPERTIES (TOP-LEVEL)
├── charge_succeeds_with_sufficient_balance    ★ Main theorem
├── charge_fails_when_exhausted
├── charge_fails_when_insufficient
├── charge_decreases_available
├── charge_strictly_decreases
└── charge_preserves_non_negative              ★ Safety theorem

SEQUENTIAL CHARGING
├── sequential_charges_accumulate
├── sequential_charges_fail
└── charges_preserve_non_negative              ★ Inductive safety

LOG TRACKING
├── charge_logs_cost
├── charge_preserves_log
└── charge_increases_log

COST BOUNDS
├── produce_cost_bounded
├── consume_cost_bounded
└── install_cost_bounded

INITIAL PRESERVATION
├── new_manager_full
└── charge_preserves_initial

REFUND OPERATIONS
├── refund_increases_available
├── refund_zero_identity
├── refunds_additive
└── refund_preserves_initial

COST COMPARISON
├── cost_le_refl
├── cost_le_trans
└── cost_lt_irrefl

HELPER LEMMAS
└── fold_out_of_phlogiston_fails
```

## Detailed Theorems

### Theorem: charge_succeeds_with_sufficient_balance (MAIN)

**Statement**: Charging a non-negative cost from sufficient balance succeeds.

```coq
Theorem charge_succeeds_with_sufficient_balance :
  forall cm cost,
    (0 <= cost_value cost)%Z ->
    (cost_value cost <= cm_available cm)%Z ->
    (0 <= cm_available cm)%Z ->
    exists cm', charge cm cost = ChargeOk cm'.
```

**Intuition**: If you have enough phlogiston and the cost is valid (non-negative), the charge will succeed. This is the "happy path" theorem.

**Proof Technique**: Case analysis on the comparison conditions. The three premises rule out both failure cases in `charge`.

**Example**:
```
CostManager: { available: 100, log: [], initial: 100 }
Cost: { value: 25, operation: OP_PRODUCE }

Preconditions:
  - 0 <= 25         ✓ (cost is non-negative)
  - 25 <= 100       ✓ (cost <= available)
  - 0 <= 100        ✓ (available is non-negative)

Result: ChargeOk { available: 75, log: [cost], initial: 100 }
```

---

### Theorem: charge_fails_when_exhausted

**Statement**: Charging fails when phlogiston is already negative.

```coq
Theorem charge_fails_when_exhausted :
  forall cm cost,
    (cm_available cm < 0)%Z ->
    charge cm cost = OutOfPhlogiston.
```

**Intuition**: If the balance is already negative (exhausted), no further charging is possible. This is the first check in `charge`.

**Example**:
```
CostManager: { available: -5, ... }
Any cost

Result: OutOfPhlogiston (immediate, regardless of cost amount)
```

---

### Theorem: charge_fails_when_insufficient

**Statement**: Charging fails when cost exceeds available phlogiston.

```coq
Theorem charge_fails_when_insufficient :
  forall cm cost,
    (0 <= cm_available cm)%Z ->
    (cm_available cm < cost_value cost)%Z ->
    charge cm cost = OutOfPhlogiston.
```

**Intuition**: Even with positive balance, if the cost exceeds what's available, the charge fails.

**Example**:
```
CostManager: { available: 10, ... }
Cost: { value: 25, ... }

0 <= 10       ✓ (available is non-negative)
10 < 25       ✓ (cost exceeds available)

Result: OutOfPhlogiston
```

---

### Theorem: charge_preserves_non_negative (SAFETY)

**Statement**: If a charge succeeds with non-negative cost, available phlogiston remains non-negative.

```coq
Theorem charge_preserves_non_negative :
  forall cm cost cm',
    (0 <= cost_value cost)%Z ->
    charge cm cost = ChargeOk cm' ->
    (0 <= cm_available cm')%Z.
```

**Intuition**: This is the key safety property. If `charge` returns `ChargeOk`, the new balance is guaranteed non-negative. The charge operation only succeeds when `available >= cost`, so `available - cost >= 0`.

**Proof Technique**: Unfold `charge`, analyze the conditions under which it succeeds.

---

### Theorem: charge_decreases_available

**Statement**: Successful charging decreases available phlogiston by exactly the cost amount.

```coq
Theorem charge_decreases_available :
  forall cm cost cm',
    charge cm cost = ChargeOk cm' ->
    (cm_available cm' = cm_available cm - cost_value cost)%Z.
```

**Intuition**: Charging is exact - you pay exactly what the cost specifies.

---

### Theorem: charge_strictly_decreases

**Statement**: Charging a positive cost strictly decreases available phlogiston.

```coq
Theorem charge_strictly_decreases :
  forall cm cost cm',
    (0 < cost_value cost)%Z ->
    charge cm cost = ChargeOk cm' ->
    (cm_available cm' < cm_available cm)%Z.
```

**Intuition**: If you charge something positive, you have strictly less than before. This guarantees progress toward termination (you can't charge forever).

**Proof Technique**: Uses `charge_decreases_available` and arithmetic.

---

### Theorem: sequential_charges_accumulate

**Statement**: Sequential charges accumulate correctly.

```coq
Theorem sequential_charges_accumulate :
  forall cm cost1 cost2 cm1 cm2,
    charge cm cost1 = ChargeOk cm1 ->
    charge cm1 cost2 = ChargeOk cm2 ->
    (cm_available cm2 = cm_available cm - cost_value cost1 - cost_value cost2)%Z.
```

**Intuition**: Two sequential charges subtract both costs from the original balance. No hidden fees.

**Example**:
```
Start: available = 100
Charge 1: 25 -> available = 75
Charge 2: 30 -> available = 45

45 = 100 - 25 - 30 ✓
```

---

### Theorem: charges_preserve_non_negative (INDUCTIVE SAFETY)

**Statement**: A sequence of non-negative charges preserves non-negative balance.

```coq
Theorem charges_preserve_non_negative :
  forall costs cm cm',
    (forall c, In c costs -> (0 <= cost_value c)%Z) ->
    (0 <= cm_available cm)%Z ->
    fold_left (fun acc c =>
      match acc with
      | ChargeOk m => charge m c
      | OutOfPhlogiston => OutOfPhlogiston
      end) costs (ChargeOk cm) = ChargeOk cm' ->
    (0 <= cm_available cm')%Z.
```

**Intuition**: This extends single-charge safety to arbitrary sequences. If all costs are non-negative, start with non-negative balance, and all charges succeed, the final balance is non-negative.

**Proof Technique**: Induction on the list of costs, using `charge_preserves_non_negative` at each step.

**Key Lemma**: `fold_out_of_phlogiston_fails` - once you hit `OutOfPhlogiston`, you can never recover.

---

### Theorem: charge_logs_cost

**Statement**: Charging adds the cost to the log.

```coq
Theorem charge_logs_cost :
  forall cm cost cm',
    charge cm cost = ChargeOk cm' ->
    In cost (cm_log cm').
```

**Intuition**: The log records all successful charges for auditing/debugging.

---

### Theorem: charge_preserves_log

**Statement**: Charging preserves existing log entries.

```coq
Theorem charge_preserves_log :
  forall cm cost cm' c,
    charge cm cost = ChargeOk cm' ->
    In c (cm_log cm) ->
    In c (cm_log cm').
```

**Intuition**: Charging only prepends to the log, never removes existing entries.

---

### Theorem: charge_increases_log

**Statement**: Log length increases by one on successful charge.

```coq
Theorem charge_increases_log :
  forall cm cost cm',
    charge cm cost = ChargeOk cm' ->
    length (cm_log cm') = S (length (cm_log cm)).
```

---

### Theorem: produce_cost_bounded

**Statement**: Produce cost is bounded by input sizes.

```coq
Theorem produce_cost_bounded :
  forall channel_size data_size,
    (0 <= channel_size)%Z ->
    (0 <= data_size)%Z ->
    (0 <= cost_value (produce_cost channel_size data_size))%Z.
```

**Intuition**: If inputs have non-negative size, the produce cost is non-negative. This ensures produce costs satisfy the precondition for `charge_succeeds_with_sufficient_balance`.

Similar theorems exist for:
- `consume_cost_bounded`
- `install_cost_bounded`

---

### Theorem: new_manager_full

**Statement**: For new managers, available equals initial.

```coq
Theorem new_manager_full :
  forall initial,
    cm_available (cost_manager_new initial) = initial.
```

**Intuition**: A fresh manager has full balance.

---

### Theorem: charge_preserves_initial

**Statement**: Charging preserves the initial phlogiston amount.

```coq
Theorem charge_preserves_initial :
  forall cm cost cm',
    charge cm cost = ChargeOk cm' ->
    cm_initial cm' = cm_initial cm.
```

**Intuition**: The `cm_initial` field is a constant - it records how much phlogiston was allocated, not how much remains.

---

### Theorem: refund_increases_available

**Statement**: Refund increases available phlogiston.

```coq
Theorem refund_increases_available :
  forall cm amount,
    cm_available (refund cm amount) = (cm_available cm + amount)%Z.
```

**Intuition**: Refunds add back to the balance. Used for operations that reserve phlogiston then return unused amounts.

---

### Theorem: refund_zero_identity

**Statement**: Refund of zero is identity.

```coq
Theorem refund_zero_identity :
  forall cm,
    cm_available (refund cm 0) = cm_available cm.
```

---

### Theorem: refunds_additive

**Statement**: Refunds are additive.

```coq
Theorem refunds_additive :
  forall cm a1 a2,
    cm_available (refund (refund cm a1) a2) =
    cm_available (refund cm (a1 + a2)).
```

**Intuition**: Order of refunds doesn't matter; they accumulate.

---

### Theorem: cost_le_refl

**Statement**: Cost comparison is reflexive.

```coq
Theorem cost_le_refl : forall c, cost_le c c = true.
```

---

### Theorem: cost_le_trans

**Statement**: Cost comparison is transitive.

```coq
Theorem cost_le_trans : forall c1 c2 c3,
  cost_le c1 c2 = true ->
  cost_le c2 c3 = true ->
  cost_le c1 c3 = true.
```

---

### Theorem: cost_lt_irrefl

**Statement**: Strict cost comparison is irreflexive.

```coq
Theorem cost_lt_irrefl : forall c, cost_lt c c = false.
```

## Examples

### Example: Basic Charge Sequence

```
1. Create manager with 100 phlogiston
   cm0 = { available: 100, log: [], initial: 100 }

2. Charge 25 for produce
   cm1 = { available: 75, log: [produce_cost], initial: 100 }

3. Charge 11 for send
   cm2 = { available: 64, log: [send_cost, produce_cost], initial: 100 }

4. Charge 80 for expensive operation
   Result: OutOfPhlogiston (64 < 80)

Invariants maintained throughout:
- available >= 0 (until exhaustion)
- initial unchanged at 100
- log grows monotonically
```

### Example: Refund After Speculative Execution

```
1. Start: available = 100
2. Reserve 50 for speculation: charge(50) -> available = 50
3. Speculation uses only 20
4. Refund unused: refund(30) -> available = 80

The refund recovers unused phlogiston from over-estimation.
```

### Example: Proving Operation Safety

To show an operation is safe:
1. Show cost is bounded: `produce_cost_bounded`
2. Show sufficient balance: `has_phlogiston` check
3. Apply `charge_succeeds_with_sufficient_balance`
4. Result preserves non-negativity: `charge_preserves_non_negative`

## RSpace Operation Costs

The module defines costs for RSpace operations:

```coq
(* Produce: cost = channel_size + data_size *)
Definition produce_cost (channel_size data_size : Z) : Cost :=
  mkCost (channel_size + data_size) OP_PRODUCE.

(* Consume: cost = channels_size + patterns_size + body_size *)
Definition consume_cost (channels_size patterns_size body_size : Z) : Cost :=
  mkCost (channels_size + patterns_size + body_size) OP_CONSUME.

(* Install: same as consume *)
Definition install_cost (channels_size patterns_size body_size : Z) : Cost :=
  mkCost (channels_size + patterns_size + body_size) OP_INSTALL.
```

## Correspondence to Rust

| Rocq Definition | Rust Implementation |
|-----------------|---------------------|
| `Cost` | `accounting::Cost` struct |
| `CostManager` | `accounting::CostManager` |
| `charge` | `CostManager::charge()` |
| `has_phlogiston` | `CostManager::has_phlogiston()` |
| `refund` | `CostManager::refund()` |
| `OP_*` constants | `costs::OPERATION_*` |
| `COST_*` constants | `costs::COST_*` |
| `produce_cost` | `costs::produce_cost()` |
| `consume_cost` | `costs::consume_cost()` |

## Design Notes

### Why Track Initial?

The `cm_initial` field enables computing:
- Total spent = `cm_initial - cm_available`
- Remaining percentage = `cm_available / cm_initial`

### Why Refunds Don't Log?

Refunds are internal operations (returning unused reserves), not billable operations. Logging them would inflate the operation count without representing actual work.

### Invariant: Log + Available = Initial

The specification notes that a well-formedness invariant would be:
```
total_cost(cm_log) + cm_available = cm_initial
```

This follows from `charge_logs_cost` and `charge_decreases_available` but would require tracking across the full operation sequence to prove formally.

## Next Steps

Continue to [06-substitution.md](06-substitution.md) for the variable substitution proofs.
