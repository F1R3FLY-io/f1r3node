(** * Phlogiston Gas Accounting

    This module specifies the cost accounting (phlogiston/gas) system
    for RSpace operations. It proves properties about charging, resource
    exhaustion, and cost bounds.

    Reference: Rust implementation in
      rholang/src/rust/interpreter/accounting/mod.rs
      rholang/src/rust/interpreter/accounting/costs.rs
*)

From Stdlib Require Import List Bool ZArith Lia.
From ReifiedRSpaces Require Import Prelude.
Import ListNotations.

(** ** Cost Types *)

(** A Cost represents a charge for an operation *)
Record Cost := mkCost {
  cost_value : Z;
  cost_operation : nat;  (* Operation identifier *)
}.

(** Predefined operation identifiers *)
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

(** Standard cost values (from costs.rs) *)
Definition COST_SEND := 11%Z.
Definition COST_RECEIVE := 11%Z.
Definition COST_NEW_BINDING := 2%Z.
Definition COST_NEW_EVAL := 10%Z.
Definition COST_LOOKUP := 3%Z.
Definition COST_REMOVE := 3%Z.
Definition COST_ADD := 3%Z.
Definition COST_MATCH := 12%Z.

(** Cost arithmetic *)
Definition cost_add (c1 c2 : Cost) : Cost :=
  mkCost (cost_value c1 + cost_value c2) 0%nat.

Definition cost_sub (c1 c2 : Cost) : Cost :=
  mkCost (cost_value c1 - cost_value c2) 0%nat.

Definition cost_mul (c1 c2 : Cost) : Cost :=
  mkCost (cost_value c1 * cost_value c2) 0%nat.

(** ** Cost Manager State *)

(** CostManager tracks available phlogiston and operation log *)
Record CostManager := mkCostManager {
  cm_available : Z;          (* Available phlogiston *)
  cm_log : list Cost;        (* Log of charged costs *)
  cm_initial : Z;            (* Initial phlogiston amount *)
}.

(** Create a new cost manager with initial phlogiston *)
Definition cost_manager_new (initial : Z) : CostManager :=
  mkCostManager initial [] initial.

(** Get current available phlogiston *)
Definition cost_manager_get (cm : CostManager) : Z :=
  cm_available cm.

(** ** Charge Operation *)

(** Result of a charge operation *)
Inductive ChargeResult :=
  | ChargeOk : CostManager -> ChargeResult
  | OutOfPhlogiston : ChargeResult.

(** Charge a cost from the manager *)
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

(** Check if manager has sufficient phlogiston *)
Definition has_phlogiston (cm : CostManager) (amount : Z) : bool :=
  (amount <=? cm_available cm)%Z.

(** ** Charge Properties *)

(** Charging a non-negative cost from sufficient balance succeeds *)
Theorem charge_succeeds_with_sufficient_balance :
  forall cm cost,
    (0 <= cost_value cost)%Z ->
    (cost_value cost <= cm_available cm)%Z ->
    (0 <= cm_available cm)%Z ->
    exists cm', charge cm cost = ChargeOk cm'.
Proof.
  intros cm cost Hcost_pos Hcost_le Havail_pos.
  unfold charge.
  destruct (cm_available cm <? 0)%Z eqn:Hneg.
  - (* cm_available cm < 0 - contradiction with Havail_pos *)
    apply Z.ltb_lt in Hneg. lia.
  - (* cm_available cm >= 0 *)
    destruct (cm_available cm <? cost_value cost)%Z eqn:Hlt.
    + (* cm_available cm < cost_value cost - contradiction with Hcost_le *)
      apply Z.ltb_lt in Hlt. lia.
    + (* cm_available cm >= cost_value cost *)
      eexists. reflexivity.
Qed.

(** Charging fails when phlogiston is exhausted *)
Theorem charge_fails_when_exhausted :
  forall cm cost,
    (cm_available cm < 0)%Z ->
    charge cm cost = OutOfPhlogiston.
Proof.
  intros cm cost Hneg.
  unfold charge.
  destruct (cm_available cm <? 0)%Z eqn:Hcmp.
  - reflexivity.
  - apply Z.ltb_ge in Hcmp. lia.
Qed.

(** Charging fails when cost exceeds available *)
Theorem charge_fails_when_insufficient :
  forall cm cost,
    (0 <= cm_available cm)%Z ->
    (cm_available cm < cost_value cost)%Z ->
    charge cm cost = OutOfPhlogiston.
Proof.
  intros cm cost Havail_pos Hinsuff.
  unfold charge.
  destruct (cm_available cm <? 0)%Z eqn:Hneg.
  - apply Z.ltb_lt in Hneg. lia.
  - destruct (cm_available cm <? cost_value cost)%Z eqn:Hlt.
    + reflexivity.
    + apply Z.ltb_ge in Hlt. lia.
Qed.

(** ** Monotonicity Properties *)

(** Charging decreases available phlogiston *)
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

(** Note: available_bounded_by_initial would require a well-formedness
    predicate on CostManager that tracks the invariant:
      cm_available <= cm_initial
    This invariant is preserved by charge (which decreases available)
    and refund (with proper bounds). For simplicity, we prove weaker
    properties about individual operations instead. *)

(** For new managers, available equals initial *)
Theorem new_manager_full :
  forall initial,
    cm_available (cost_manager_new initial) = initial.
Proof.
  intros initial.
  unfold cost_manager_new. simpl. reflexivity.
Qed.

(** Charging positive cost strictly decreases available phlogiston *)
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

(** ** Log Properties *)

(** Charging adds cost to log *)
Theorem charge_logs_cost :
  forall cm cost cm',
    charge cm cost = ChargeOk cm' ->
    In cost (cm_log cm').
Proof.
  intros cm cost cm' Hcharge.
  unfold charge in Hcharge.
  destruct (cm_available cm <? 0)%Z eqn:Hneg; [discriminate |].
  destruct (cm_available cm <? cost_value cost)%Z eqn:Hlt; [discriminate |].
  injection Hcharge as Hcm'.
  subst cm'. simpl. left. reflexivity.
Qed.

(** Charging preserves existing log entries *)
Theorem charge_preserves_log :
  forall cm cost cm' c,
    charge cm cost = ChargeOk cm' ->
    In c (cm_log cm) ->
    In c (cm_log cm').
Proof.
  intros cm cost cm' c Hcharge Hin.
  unfold charge in Hcharge.
  destruct (cm_available cm <? 0)%Z eqn:Hneg; [discriminate |].
  destruct (cm_available cm <? cost_value cost)%Z eqn:Hlt; [discriminate |].
  injection Hcharge as Hcm'.
  subst cm'. simpl. right. exact Hin.
Qed.

(** Log length increases by one on successful charge *)
Theorem charge_increases_log :
  forall cm cost cm',
    charge cm cost = ChargeOk cm' ->
    length (cm_log cm') = S (length (cm_log cm)).
Proof.
  intros cm cost cm' Hcharge.
  unfold charge in Hcharge.
  destruct (cm_available cm <? 0)%Z eqn:Hneg; [discriminate |].
  destruct (cm_available cm <? cost_value cost)%Z eqn:Hlt; [discriminate |].
  injection Hcharge as Hcm'.
  subst cm'. simpl. reflexivity.
Qed.

(** ** Sequential Charging *)

(** Sequential charges accumulate correctly *)
Theorem sequential_charges_accumulate :
  forall cm cost1 cost2 cm1 cm2,
    charge cm cost1 = ChargeOk cm1 ->
    charge cm1 cost2 = ChargeOk cm2 ->
    (cm_available cm2 = cm_available cm - cost_value cost1 - cost_value cost2)%Z.
Proof.
  intros cm cost1 cost2 cm1 cm2 Hc1 Hc2.
  apply charge_decreases_available in Hc1.
  apply charge_decreases_available in Hc2.
  lia.
Qed.

(** If either charge fails, combined charge fails *)
Theorem sequential_charges_fail :
  forall cm cost1 cost2 cm1,
    charge cm cost1 = ChargeOk cm1 ->
    charge cm1 cost2 = OutOfPhlogiston ->
    (cm_available cm < cost_value cost1 + cost_value cost2)%Z \/
    (cm_available cm < 0)%Z.
Proof.
  intros cm cost1 cost2 cm1 Hc1 Hc2.
  unfold charge in Hc1, Hc2.
  destruct (cm_available cm <? 0)%Z eqn:Hneg1; [discriminate |].
  destruct (cm_available cm <? cost_value cost1)%Z eqn:Hlt1; [discriminate |].
  injection Hc1 as Hcm1. subst cm1. simpl in Hc2.
  destruct (cm_available cm - cost_value cost1 <? 0)%Z eqn:Hneg2.
  - right. apply Z.ltb_lt in Hneg2. apply Z.ltb_ge in Hneg1. lia.
  - destruct (cm_available cm - cost_value cost1 <? cost_value cost2)%Z eqn:Hlt2.
    + left. apply Z.ltb_lt in Hlt2. lia.
    + discriminate.
Qed.

(** ** RSpace Operation Costs *)

(** Cost of a produce operation (channel + data) *)
Definition produce_cost (channel_size data_size : Z) : Cost :=
  mkCost (channel_size + data_size) OP_PRODUCE.

(** Cost of a consume operation (channels + patterns + body) *)
Definition consume_cost (channels_size patterns_size body_size : Z) : Cost :=
  mkCost (channels_size + patterns_size + body_size) OP_CONSUME.

(** Cost of an install operation (same as consume) *)
Definition install_cost (channels_size patterns_size body_size : Z) : Cost :=
  mkCost (channels_size + patterns_size + body_size) OP_INSTALL.

(** ** Total Cost Properties *)

(** Total cost of a sequence of operations *)
Fixpoint total_cost (costs : list Cost) : Z :=
  match costs with
  | [] => 0%Z
  | c :: rest => (cost_value c + total_cost rest)%Z
  end.

(** Note: log_equals_spent would require a well-formedness predicate
    that tracks the invariant:
      total_cost (cm_log cm) + cm_available cm = cm_initial cm
    This would need to be proven inductively over the sequence of
    charge operations that produced the cost manager. The individual
    charge_decreases_available and charge_logs_cost theorems establish
    the per-operation correctness that underlies this invariant. *)

(** ** Bounded Operation Costs *)

(** Produce cost is bounded by input sizes *)
Theorem produce_cost_bounded :
  forall channel_size data_size,
    (0 <= channel_size)%Z ->
    (0 <= data_size)%Z ->
    (0 <= cost_value (produce_cost channel_size data_size))%Z.
Proof.
  intros channel_size data_size Hch Hdata.
  unfold produce_cost. simpl. lia.
Qed.

(** Consume cost is bounded by input sizes *)
Theorem consume_cost_bounded :
  forall channels_size patterns_size body_size,
    (0 <= channels_size)%Z ->
    (0 <= patterns_size)%Z ->
    (0 <= body_size)%Z ->
    (0 <= cost_value (consume_cost channels_size patterns_size body_size))%Z.
Proof.
  intros ch pat body Hch Hpat Hbody.
  unfold consume_cost. simpl. lia.
Qed.

(** Install cost is bounded by input sizes *)
Theorem install_cost_bounded :
  forall channels_size patterns_size body_size,
    (0 <= channels_size)%Z ->
    (0 <= patterns_size)%Z ->
    (0 <= body_size)%Z ->
    (0 <= cost_value (install_cost channels_size patterns_size body_size))%Z.
Proof.
  intros ch pat body Hch Hpat Hbody.
  unfold install_cost. simpl. lia.
Qed.

(** ** Resource Exhaustion Safety *)

(** If a charge succeeds, available phlogiston remains non-negative *)
Theorem charge_preserves_non_negative :
  forall cm cost cm',
    (0 <= cost_value cost)%Z ->
    charge cm cost = ChargeOk cm' ->
    (0 <= cm_available cm')%Z.
Proof.
  intros cm cost cm' Hcost_pos Hcharge.
  unfold charge in Hcharge.
  destruct (cm_available cm <? 0)%Z eqn:Hneg; [discriminate |].
  destruct (cm_available cm <? cost_value cost)%Z eqn:Hlt; [discriminate |].
  injection Hcharge as Hcm'.
  subst cm'. simpl.
  apply Z.ltb_ge in Hneg.
  apply Z.ltb_ge in Hlt.
  lia.
Qed.

(** Helper: fold from OutOfPhlogiston never succeeds *)
Lemma fold_out_of_phlogiston_fails :
  forall costs cm',
    fold_left (fun acc c =>
      match acc with
      | ChargeOk m => charge m c
      | OutOfPhlogiston => OutOfPhlogiston
      end) costs OutOfPhlogiston = ChargeOk cm' -> False.
Proof.
  induction costs as [| c rest IH]; intros cm' Hfold.
  - simpl in Hfold. discriminate.
  - simpl in Hfold. apply IH in Hfold. exact Hfold.
Qed.

(** A sequence of non-negative charges preserves non-negative balance *)
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
Proof.
  intros costs.
  induction costs as [| c rest IH]; intros cm cm' Hcosts Hcm Hfold.
  - simpl in Hfold. injection Hfold as Heq. subst cm'. exact Hcm.
  - simpl in Hfold.
    destruct (charge cm c) eqn:Hcharge.
    + (* charge succeeded *)
      assert (Hc_pos : (0 <= cost_value c)%Z).
      { apply Hcosts. left. reflexivity. }
      assert (Hc0_pos : (0 <= cm_available c0)%Z).
      { eapply charge_preserves_non_negative; eassumption. }
      apply IH with (cm := c0).
      * intros c' Hin. apply Hcosts. right. exact Hin.
      * exact Hc0_pos.
      * exact Hfold.
    + (* charge failed - fold returns OutOfPhlogiston, contradiction *)
      exfalso. apply (fold_out_of_phlogiston_fails rest cm'). exact Hfold.
Qed.

(** ** Initial Phlogiston Invariant *)

(** Charging preserves initial phlogiston *)
Theorem charge_preserves_initial :
  forall cm cost cm',
    charge cm cost = ChargeOk cm' ->
    cm_initial cm' = cm_initial cm.
Proof.
  intros cm cost cm' Hcharge.
  unfold charge in Hcharge.
  destruct (cm_available cm <? 0)%Z eqn:Hneg; [discriminate |].
  destruct (cm_available cm <? cost_value cost)%Z eqn:Hlt; [discriminate |].
  injection Hcharge as Hcm'.
  subst cm'. simpl. reflexivity.
Qed.

(** ** Refund Operation *)

(** Refund phlogiston to the manager *)
Definition refund (cm : CostManager) (amount : Z) : CostManager :=
  mkCostManager
    (cm_available cm + amount)
    (cm_log cm)  (* Log unchanged - refunds not logged *)
    (cm_initial cm).

(** Refund increases available phlogiston *)
Theorem refund_increases_available :
  forall cm amount,
    cm_available (refund cm amount) = (cm_available cm + amount)%Z.
Proof.
  intros cm amount.
  unfold refund. simpl. reflexivity.
Qed.

(** Refund of zero is identity *)
Theorem refund_zero_identity :
  forall cm,
    cm_available (refund cm 0) = cm_available cm.
Proof.
  intros cm.
  unfold refund. simpl. lia.
Qed.

(** Refunds are additive *)
Theorem refunds_additive :
  forall cm a1 a2,
    cm_available (refund (refund cm a1) a2) =
    cm_available (refund cm (a1 + a2)).
Proof.
  intros cm a1 a2.
  unfold refund. simpl. lia.
Qed.

(** Refund preserves initial *)
Theorem refund_preserves_initial :
  forall cm amount,
    cm_initial (refund cm amount) = cm_initial cm.
Proof.
  intros cm amount.
  unfold refund. simpl. reflexivity.
Qed.

(** ** Cost Comparison *)

(** Decidable cost comparison *)
Definition cost_le (c1 c2 : Cost) : bool :=
  (cost_value c1 <=? cost_value c2)%Z.

Definition cost_lt (c1 c2 : Cost) : bool :=
  (cost_value c1 <? cost_value c2)%Z.

(** cost_le is reflexive *)
Theorem cost_le_refl : forall c, cost_le c c = true.
Proof.
  intros c. unfold cost_le. apply Z.leb_refl.
Qed.

(** cost_le is transitive *)
Theorem cost_le_trans : forall c1 c2 c3,
  cost_le c1 c2 = true ->
  cost_le c2 c3 = true ->
  cost_le c1 c3 = true.
Proof.
  intros c1 c2 c3 H12 H23.
  unfold cost_le in *.
  apply Z.leb_le in H12.
  apply Z.leb_le in H23.
  apply Z.leb_le. lia.
Qed.

(** cost_lt is irreflexive *)
Theorem cost_lt_irrefl : forall c, cost_lt c c = false.
Proof.
  intros c. unfold cost_lt. apply Z.ltb_irrefl.
Qed.

